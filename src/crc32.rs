//! CRC32 - SIMD (PCLMULQDQ) + slice-by-8 hybrid
//!
//! Based on Intel whitepaper: "Fast CRC computation for generic polynomials
//! using PCLMULQDQ instruction"

const POLY: u32 = 0xEDB88320;

const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ POLY } else { crc >> 1 };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

const S8: [[u32; 256]; 8] = {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 { t[0][i] = TABLE[i]; i += 1; }
    i = 0;
    while i < 256 {
        t[1][i] = (t[0][i] >> 8) ^ TABLE[(t[0][i] & 0xFF) as usize];
        t[2][i] = (t[1][i] >> 8) ^ TABLE[(t[1][i] & 0xFF) as usize];
        t[3][i] = (t[2][i] >> 8) ^ TABLE[(t[2][i] & 0xFF) as usize];
        t[4][i] = (t[3][i] >> 8) ^ TABLE[(t[3][i] & 0xFF) as usize];
        t[5][i] = (t[4][i] >> 8) ^ TABLE[(t[4][i] & 0xFF) as usize];
        t[6][i] = (t[5][i] >> 8) ^ TABLE[(t[5][i] & 0xFF) as usize];
        t[7][i] = (t[6][i] >> 8) ^ TABLE[(t[6][i] & 0xFF) as usize];
        i += 1;
    }
    t
};

/// Fallback: slice-by-8 (pure software, no SIMD)
#[inline]
fn crc32_slice8(crc: u32, data: &[u8]) -> u32 {
    let mut crc = !crc;
    let mut ptr = data.as_ptr();
    let mut len = data.len();

    while len >= 8 {
        unsafe {
            let chunk = u64::from_le_bytes(std::ptr::read_unaligned(ptr as *const u64).to_le_bytes());
            let b = chunk.to_le_bytes();
            let lo = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let hi = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let x = crc ^ lo;
            crc = S8[3][(hi & 0xFF) as usize] ^ S8[2][((hi >> 8) & 0xFF) as usize]
                ^ S8[1][((hi >> 16) & 0xFF) as usize] ^ S8[0][((hi >> 24) & 0xFF) as usize]
                ^ S8[7][(x & 0xFF) as usize] ^ S8[6][((x >> 8) & 0xFF) as usize]
                ^ S8[5][((x >> 16) & 0xFF) as usize] ^ S8[4][((x >> 24) & 0xFF) as usize];
            ptr = ptr.add(8);
        }
        len -= 8;
    }

    while len >= 4 {
        unsafe {
            let lo = u32::from_le_bytes([*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]);
            let x = crc ^ lo;
            crc = S8[3][(x & 0xFF) as usize] ^ S8[2][((x >> 8) & 0xFF) as usize]
                ^ S8[1][((x >> 16) & 0xFF) as usize] ^ S8[0][((x >> 24) & 0xFF) as usize];
            ptr = ptr.add(4);
        }
        len -= 4;
    }

    for i in 0..len {
        unsafe { crc = TABLE[((crc ^ *ptr.add(i) as u32) & 0xFF) as usize] ^ (crc >> 8); }
    }
    !crc
}

// SIMD implementation using PCLMULQDQ
#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;
    use std::arch::x86_64::*;

    // Folding constants from Intel paper / crc32fast
    const K1: i64 = 0x154442bd4;
    const K2: i64 = 0x1c6e41596;
    const K3: i64 = 0x1751997d0;
    const K4: i64 = 0x0ccaa009e;
    const K5: i64 = 0x163cd6124;

    // Barrett reduction constants
    const P_X: i64 = 0x1DB710641;
    const U_PRIME: i64 = 0x1F7011641;

    #[inline(always)]
    unsafe fn reduce128(a: __m128i, b: __m128i, keys: __m128i) -> __m128i {
        let t1 = _mm_clmulepi64_si128::<0x00>(a, keys);
        let t2 = _mm_clmulepi64_si128::<0x11>(a, keys);
        _mm_xor_si128(_mm_xor_si128(b, t1), t2)
    }

    #[inline(always)]
    unsafe fn get(data: &mut &[u8]) -> __m128i {
        debug_assert!(data.len() >= 16);
        let r = _mm_loadu_si128(data.as_ptr() as *const __m128i);
        *data = &data[16..];
        r
    }

    #[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
    pub unsafe fn calculate(mut crc: u32, mut data: &[u8]) -> u32 {
        // For small chunks, use slice-by-8
        if data.len() < 128 {
            return crc32_slice8(crc, data);
        }

        // Step 1: fold by 4 loop (process 64 bytes per iteration)
        let mut x3 = get(&mut data);
        let mut x2 = get(&mut data);
        let mut x1 = get(&mut data);
        let mut x0 = get(&mut data);

        // Fold in initial CRC
        x3 = _mm_xor_si128(x3, _mm_cvtsi32_si128(!crc as i32));

        let k1k2 = _mm_set_epi64x(K2, K1);
        while data.len() >= 64 {
            x3 = reduce128(x3, get(&mut data), k1k2);
            x2 = reduce128(x2, get(&mut data), k1k2);
            x1 = reduce128(x1, get(&mut data), k1k2);
            x0 = reduce128(x0, get(&mut data), k1k2);
        }

        // Fold 4 -> 1
        let k3k4 = _mm_set_epi64x(K4, K3);
        let mut x = reduce128(x3, x2, k3k4);
        x = reduce128(x, x1, k3k4);
        x = reduce128(x, x0, k3k4);

        // Step 2: fold remaining 16-byte chunks
        while data.len() >= 16 {
            x = reduce128(x, get(&mut data), k3k4);
        }

        // Step 3: reduce 128 -> 64 bits
        let x = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x10>(x, k3k4),
            _mm_srli_si128(x, 8),
        );
        let x = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x00>(
                _mm_and_si128(x, _mm_set_epi32(0, 0, 0, !0)),
                _mm_set_epi64x(0, K5),
            ),
            _mm_srli_si128(x, 4),
        );

        // Barrett reduction: 64 -> 32 bits
        let pu = _mm_set_epi64x(U_PRIME, P_X);

        let t1 = _mm_clmulepi64_si128::<0x10>(
            _mm_and_si128(x, _mm_set_epi32(0, 0, 0, !0)),
            pu,
        );
        let t2 = _mm_clmulepi64_si128::<0x00>(
            _mm_and_si128(t1, _mm_set_epi32(0, 0, 0, !0)),
            pu,
        );
        let c = _mm_extract_epi32::<1>(_mm_xor_si128(x, t2)) as u32;

        // Process remaining bytes
        if !data.is_empty() {
            crc32_slice8(!c, data)
        } else {
            !c
        }
    }

    #[inline]
    pub fn crc32_simd(crc: u32, data: &[u8]) -> u32 {
        unsafe { calculate(crc, data) }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    if is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("sse2")
        && is_x86_feature_detected!("sse4.1")
    {
        simd::crc32_simd(0, data)
    } else {
        crc32_slice8(0, data)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_slice8(0, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_empty() { assert_eq!(crc32(b""), 0); }
    #[test] fn test_hello_world() { assert_eq!(crc32(b"Hello, World!"), 0xEC4AC3D0); }
    #[test] fn test_123456789() { assert_eq!(crc32(b"123456789"), 0xCBF43926); }
    #[test]
    fn test_matches_reference() {
        let test_data: Vec<u8> = (0..=255).cycle().take(1000).collect();
        assert_eq!(crc32(&test_data), crc32fast::hash(&test_data));
    }
    #[test]
    fn test_large_matches() {
        let test_data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        assert_eq!(crc32(&test_data), crc32fast::hash(&test_data));
    }
    #[test]
    fn test_very_large() {
        let test_data: Vec<u8> = (0..=255u8).cycle().take(1_000_000).collect();
        assert_eq!(crc32(&test_data), crc32fast::hash(&test_data));
    }
}