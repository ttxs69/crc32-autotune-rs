//! CRC32 - SIMD (PCLMULQDQ) + slice-by-8 hybrid
//!
//! Supports: AVX-512 (VPCLMULQDQ), SSE (PCLMULQDQ), software fallback
//! Based on Intel whitepaper: "Fast CRC computation for generic polynomials"

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

// ============================================================================
// SSE implementation (PCLMULQDQ) - 128-bit vectors
// ============================================================================
#[cfg(target_arch = "x86_64")]
mod sse {
    use super::*;
    use std::arch::x86_64::*;

    // Folding constants from Intel paper
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
    unsafe fn get128(data: &mut &[u8]) -> __m128i {
        debug_assert!(data.len() >= 16);
        let r = _mm_loadu_si128(data.as_ptr() as *const __m128i);
        *data = &data[16..];
        r
    }

    #[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
    pub unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
        if data.len() < 128 {
            return crc32_slice8(crc, data);
        }

        // Fold by 4 loop
        let mut x3 = get128(&mut data);
        let mut x2 = get128(&mut data);
        let mut x1 = get128(&mut data);
        let mut x0 = get128(&mut data);

        x3 = _mm_xor_si128(x3, _mm_cvtsi32_si128(!crc as i32));

        let k1k2 = _mm_set_epi64x(K2, K1);
        while data.len() >= 64 {
            x3 = reduce128(x3, get128(&mut data), k1k2);
            x2 = reduce128(x2, get128(&mut data), k1k2);
            x1 = reduce128(x1, get128(&mut data), k1k2);
            x0 = reduce128(x0, get128(&mut data), k1k2);
        }

        let k3k4 = _mm_set_epi64x(K4, K3);
        let mut x = reduce128(x3, x2, k3k4);
        x = reduce128(x, x1, k3k4);
        x = reduce128(x, x0, k3k4);

        while data.len() >= 16 {
            x = reduce128(x, get128(&mut data), k3k4);
        }

        // Reduce 128 -> 64
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

        // Barrett reduction: 64 -> 32
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

        if !data.is_empty() { crc32_slice8(!c, data) } else { !c }
    }
}

// ============================================================================
// AVX-512 implementation (VPCLMULQDQ) - 512-bit vectors
// ============================================================================
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    // Folding constants for 512-bit (extended from 128-bit)
    // These are for folding 512 bits down
    const K1_512: i64 = 0x154442bd4;  // x^(128+32) mod poly
    const K2_512: i64 = 0x1c6e41596;  // x^(128+64) mod poly
    const K3_512: i64 = 0x1751997d0;  // x^(256+32) mod poly
    const K4_512: i64 = 0x0ccaa009e;  // x^(256+64) mod poly
    const K5_512: i64 = 0x163cd6124;

    // Barrett reduction
    const P_X: i64 = 0x1DB710641;
    const U_PRIME: i64 = 0x1F7011641;

    #[inline(always)]
    unsafe fn get512(data: &mut &[u8]) -> __m512i {
        debug_assert!(data.len() >= 64);
        let r = _mm512_loadu_si512(data.as_ptr() as *const __m512i);
        *data = &data[64..];
        r
    }

    // Fold 512 bits using VPCLMULQDQ (4 x 128-bit chunks in parallel)
    #[inline(always)]
    unsafe fn fold512(a: __m512i, b: __m512i, keys: __m512i) -> __m512i {
        // VPCLMULQDQ with imm8=0x00: multiply low 64-bits of each 128-bit lane
        // VPCLMULQDQ with imm8=0x11: multiply high 64-bits of each 128-bit lane
        let t1 = _mm512_clmulepi64_epi128::<0x00>(a, keys);
        let t2 = _mm512_clmulepi64_epi128::<0x11>(a, keys);
        _mm512_ternarylogic_epi32::<0x96>(t1, t2, b)  // t1 ^ t2 ^ b
    }

    // Reduce 512 to 128 using folding
    unsafe fn reduce512_to_128(x: __m512i, k3k4: __m512i) -> __m128i {
        let k3k4_128 = _mm512_castsi512_si128(k3k4);

        // Fold the 4 x 128-bit chunks
        let chunk0 = _mm512_extracti32x4_epi32::<0>(x);
        let chunk1 = _mm512_extracti32x4_epi32::<1>(x);
        let chunk2 = _mm512_extracti32x4_epi32::<2>(x);
        let chunk3 = _mm512_extracti32x4_epi32::<3>(x);

        // Fold chunk0 with chunk1, chunk2 with chunk3
        let t1 = _mm_clmulepi64_si128::<0x00>(chunk0, k3k4_128);
        let t2 = _mm_clmulepi64_si128::<0x11>(chunk0, k3k4_128);
        let fold01 = _mm_xor_si128(_mm_xor_si128(t1, t2), chunk1);

        let t1 = _mm_clmulepi64_si128::<0x00>(chunk2, k3k4_128);
        let t2 = _mm_clmulepi64_si128::<0x11>(chunk2, k3k4_128);
        let fold23 = _mm_xor_si128(_mm_xor_si128(t1, t2), chunk3);

        // Final fold
        let t1 = _mm_clmulepi64_si128::<0x00>(fold01, k3k4_128);
        let t2 = _mm_clmulepi64_si128::<0x11>(fold01, k3k4_128);
        _mm_xor_si128(_mm_xor_si128(t1, t2), fold23)
    }

    #[target_feature(enable = "avx512f", enable = "avx512vl", enable = "vpclmulqdq", enable = "sse4.1")]
    pub unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
        if data.len() < 256 {
            // Fallback to SSE for medium chunks
            return sse::calculate(crc, data);
        }

        // Load first 4 x 512 bits (256 bytes)
        let mut x3 = get512(&mut data);
        let mut x2 = get512(&mut data);
        let mut x1 = get512(&mut data);
        let mut x0 = get512(&mut data);

        // Fold in initial CRC to first 128-bit chunk of x3
        let crc_128 = _mm_cvtsi32_si128(!crc as i32);
        let crc_512 = _mm512_castsi128_si512(crc_128);
        x3 = _mm512_xor_si512(x3, crc_512);

        // Folding constants
        let k1k2 = _mm512_set_epi64(K2_512, K1_512, K2_512, K1_512, K2_512, K1_512, K2_512, K1_512);

        // Main loop: fold 256 bytes per iteration
        while data.len() >= 256 {
            x3 = fold512(x3, get512(&mut data), k1k2);
            x2 = fold512(x2, get512(&mut data), k1k2);
            x1 = fold512(x1, get512(&mut data), k1k2);
            x0 = fold512(x0, get512(&mut data), k1k2);
        }

        // Fold 4 -> 1
        let k3k4 = _mm512_set_epi64(K4_512, K3_512, K4_512, K3_512, K4_512, K3_512, K4_512, K3_512);
        let mut x = fold512(x3, x2, k3k4);
        x = fold512(x, x1, k3k4);
        x = fold512(x, x0, k3k4);

        // Fold remaining 64-byte chunks
        while data.len() >= 64 {
            x = fold512(x, get512(&mut data), k3k4);
        }

        // Reduce 512 -> 128
        let mut x128 = reduce512_to_128(x, k3k4);

        // Process remaining 16-byte chunks
        let k3k4_128 = _mm_set_epi64x(K4_512, K3_512);
        while data.len() >= 16 {
            let chunk = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let t1 = _mm_clmulepi64_si128::<0x00>(x128, k3k4_128);
            let t2 = _mm_clmulepi64_si128::<0x11>(x128, k3k4_128);
            x128 = _mm_xor_si128(_mm_xor_si128(t1, t2), chunk);
            data = &data[16..];
        }

        // Reduce 128 -> 64
        let x128 = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x10>(x128, k3k4_128),
            _mm_srli_si128(x128, 8),
        );
        let x128 = _mm_xor_si128(
            _mm_clmulepi64_si128::<0x00>(
                _mm_and_si128(x128, _mm_set_epi32(0, 0, 0, !0)),
                _mm_set_epi64x(0, K5_512),
            ),
            _mm_srli_si128(x128, 4),
        );

        // Barrett reduction: 64 -> 32
        let pu = _mm_set_epi64x(U_PRIME, P_X);
        let t1 = _mm_clmulepi64_si128::<0x10>(
            _mm_and_si128(x128, _mm_set_epi32(0, 0, 0, !0)),
            pu,
        );
        let t2 = _mm_clmulepi64_si128::<0x00>(
            _mm_and_si128(t1, _mm_set_epi32(0, 0, 0, !0)),
            pu,
        );
        let c = _mm_extract_epi32::<1>(_mm_xor_si128(x128, t2)) as u32;

        if !data.is_empty() { crc32_slice8(!c, data) } else { !c }
    }
}

// ============================================================================
// Public API with runtime dispatch
// ============================================================================
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    // Check for AVX-512 support
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("vpclmulqdq")
    {
        unsafe { avx512::calculate(0, data) }
    } else if is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("sse2")
        && is_x86_feature_detected!("sse4.1")
    {
        unsafe { sse::calculate(0, data) }
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
    #[test]
    fn test_avx512_path() {
        // Large enough to trigger AVX-512 path
        let test_data: Vec<u8> = (0..=255u8).cycle().take(1_000_000).collect();
        assert_eq!(crc32(&test_data), crc32fast::hash(&test_data));
    }
}