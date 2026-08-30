//! CRC32 - SIMD (PCLMULQDQ) + slice-by-8 hybrid with parallel support
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

// Pre-computed x^(2^n) mod poly for combine operation.
//
// This is a doubling chain: entry[i+1] == gf2_multiply(entry[i], entry[i]),
// and it closes with period 32: gf2_multiply(entry[31], entry[31]) == entry[0]
// (verified numerically; follows from the CRC32 polynomial being primitive).
// That closure is what makes crc32_combine's `i & 0x1F` index wrap exact
// for len2 >= 2^32 (e.g. parallel hashing of inputs larger than 4 GiB).
// test_x2n_table_closure pins these properties.
static X2N_TABLE: [u32; 32] = [
    0x00800000, 0x00008000, 0xedb88320, 0xb1e6b092, 0xa06a2517, 0xed627dae, 0x88d14467, 0xd7bbfe6a,
    0xec447f11, 0x8e7ea170, 0x6427800e, 0x4d47bae0, 0x09fe548f, 0x83852d0f, 0x30362f1a, 0x7b5a9cc3,
    0x31fec169, 0x9fec022a, 0x6c8dedc4, 0x15d6874d, 0x5fde7a4e, 0xbad90e37, 0x2e4e5eef, 0x4eaba214,
    0xa8a472c0, 0x429a969e, 0x148d302a, 0xc40ba6d0, 0xc4e22c3c, 0x40000000, 0x20000000, 0x08000000,
];

/// GF(2) polynomial multiplication for combine
fn gf2_multiply(a: u32, mut b: u32) -> u32 {
    let mut p = 0u32;
    for i in 0..32 {
        p ^= b & ((a >> (31 - i)) & 1).wrapping_neg();
        b = (b >> 1) ^ ((b & 1).wrapping_neg() & POLY);
    }
    p
}

/// Combine two CRC32 values: crc(a||b) = combine(crc(a), crc(b), len(b))
#[inline]
pub fn crc32_combine(crc1: u32, crc2: u32, len2: u64) -> u32 {
    if len2 == 0 {
        return crc1;
    }
    let mut p = crc1;
    let n = 64 - len2.leading_zeros();
    for i in 0..n {
        if (len2 >> i & 1) != 0 {
            p = gf2_multiply(X2N_TABLE[(i & 0x1F) as usize], p);
        }
    }
    p ^ crc2
}

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

    // Folding constants for 512-bit
    const K1_512: i64 = 0x154442bd4;
    const K2_512: i64 = 0x1c6e41596;
    const K3_512: i64 = 0x1751997d0;
    const K4_512: i64 = 0x0ccaa009e;
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

    #[inline(always)]
    unsafe fn fold512(a: __m512i, b: __m512i, keys: __m512i) -> __m512i {
        let t1 = _mm512_clmulepi64_epi128::<0x00>(a, keys);
        let t2 = _mm512_clmulepi64_epi128::<0x11>(a, keys);
        _mm512_ternarylogic_epi32::<0x96>(t1, t2, b)
    }

    unsafe fn reduce512_to_128(x: __m512i, k3k4: __m512i) -> __m128i {
        let k3k4_128 = _mm512_castsi512_si128(k3k4);

        let chunk0 = _mm512_extracti32x4_epi32::<0>(x);
        let chunk1 = _mm512_extracti32x4_epi32::<1>(x);
        let chunk2 = _mm512_extracti32x4_epi32::<2>(x);
        let chunk3 = _mm512_extracti32x4_epi32::<3>(x);

        let t1 = _mm_clmulepi64_si128::<0x00>(chunk0, k3k4_128);
        let t2 = _mm_clmulepi64_si128::<0x11>(chunk0, k3k4_128);
        let fold01 = _mm_xor_si128(_mm_xor_si128(t1, t2), chunk1);

        let t1 = _mm_clmulepi64_si128::<0x00>(chunk2, k3k4_128);
        let t2 = _mm_clmulepi64_si128::<0x11>(chunk2, k3k4_128);
        let fold23 = _mm_xor_si128(_mm_xor_si128(t1, t2), chunk3);

        let t1 = _mm_clmulepi64_si128::<0x00>(fold01, k3k4_128);
        let t2 = _mm_clmulepi64_si128::<0x11>(fold01, k3k4_128);
        _mm_xor_si128(_mm_xor_si128(t1, t2), fold23)
    }

    #[target_feature(enable = "avx512f", enable = "avx512vl", enable = "vpclmulqdq", enable = "sse4.1")]
    pub unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
        if data.len() < 256 {
            return sse::calculate(crc, data);
        }

        let mut x3 = get512(&mut data);
        let mut x2 = get512(&mut data);
        let mut x1 = get512(&mut data);
        let mut x0 = get512(&mut data);

        let crc_128 = _mm_cvtsi32_si128(!crc as i32);
        let crc_512 = _mm512_castsi128_si512(crc_128);
        x3 = _mm512_xor_si512(x3, crc_512);

        let k1k2 = _mm512_set_epi64(K2_512, K1_512, K2_512, K1_512, K2_512, K1_512, K2_512, K1_512);

        while data.len() >= 256 {
            x3 = fold512(x3, get512(&mut data), k1k2);
            x2 = fold512(x2, get512(&mut data), k1k2);
            x1 = fold512(x1, get512(&mut data), k1k2);
            x0 = fold512(x0, get512(&mut data), k1k2);
        }

        let k3k4 = _mm512_set_epi64(K4_512, K3_512, K4_512, K3_512, K4_512, K3_512, K4_512, K3_512);
        let mut x = fold512(x3, x2, k3k4);
        x = fold512(x, x1, k3k4);
        x = fold512(x, x0, k3k4);

        while data.len() >= 64 {
            x = fold512(x, get512(&mut data), k3k4);
        }

        let mut x128 = reduce512_to_128(x, k3k4);

        let k3k4_128 = _mm_set_epi64x(K4_512, K3_512);
        while data.len() >= 16 {
            let chunk = _mm_loadu_si128(data.as_ptr() as *const __m128i);
            let t1 = _mm_clmulepi64_si128::<0x00>(x128, k3k4_128);
            let t2 = _mm_clmulepi64_si128::<0x11>(x128, k3k4_128);
            x128 = _mm_xor_si128(_mm_xor_si128(t1, t2), chunk);
            data = &data[16..];
        }

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
// aarch64 NEON implementation (PMULL folding)
// ============================================================================
// Port of the Intel "fast CRC computation for generic polynomials using
// PCLMULQDQ" technique to NEON: 128-bit vectors folded with carry-less
// multiplies (vmull_p64), then Barrett reduction. Same folding constants
// as the x86 SSE path (polynomial constants are architecture-independent).
//
// Unlike the x86 modules, this path is exercised and verified by the test
// suite on aarch64 hardware.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    // Folding constants (CRC32-IEEE, reflected)
    const K1: u64 = 0x154442bd4;
    const K2: u64 = 0x1c6e41596;
    const K3: u64 = 0x1751997d0;
    const K4: u64 = 0x0ccaa009e;
    const K5: u64 = 0x163cd6124;

    // Barrett reduction constants
    const P_X: u64 = 0x1DB710641;
    const U_PRIME: u64 = 0x1F7011641;

    /// Minimum length for the folding path; below it the hardware CRC32
    /// chain (or slice-by-8) wins on start-up cost. Tuned by measurement.
    pub const MIN_LEN: usize = 192;

    type V = uint64x2_t; // 128-bit vector

    #[inline(always)]
    unsafe fn load(data: &mut &[u8]) -> V {
        debug_assert!(data.len() >= 16);
        let r = vld1q_u64(data.as_ptr() as *const u64);
        *data = &data[16..];
        r
    }

    /// Carry-less multiply of the low 64-bit lanes (pclmulqdq imm 0x00).
    #[inline(always)]
    unsafe fn clmul_lo(a: V, b: V) -> V {
        vreinterpretq_u64_p128(vmull_p64(
            vgetq_lane_p64::<0>(vreinterpretq_p64_u64(a)),
            vgetq_lane_p64::<0>(vreinterpretq_p64_u64(b)),
        ))
    }

    /// Carry-less multiply of the high 64-bit lanes (pclmulqdq imm 0x11).
    #[inline(always)]
    unsafe fn clmul_hi(a: V, b: V) -> V {
        vreinterpretq_u64_p128(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }

    /// Carry-less multiply of the low lane of `a` with scalar constant `k`.
    #[inline(always)]
    unsafe fn clmul_k(a: V, k: u64) -> V {
        let kv = vreinterpretq_p64_u64(vdupq_n_u64(k));
        vreinterpretq_u64_p128(vmull_p64(
            vgetq_lane_p64::<0>(vreinterpretq_p64_u64(a)),
            vgetq_lane_p64::<0>(kv),
        ))
    }

    /// x = (x_lo * keys_lo) ^ (x_hi * keys_hi) ^ b  (carry-less products)
    #[inline(always)]
    unsafe fn fold(a: V, b: V, keys: V) -> V {
        veorq_u64(veorq_u64(b, clmul_lo(a, keys)), clmul_hi(a, keys))
    }

    #[target_feature(enable = "aes")] // vmull_p64 is in the AES feature group
    pub unsafe fn calculate(crc: u32, mut data: &[u8]) -> u32 {
        if data.len() < 128 {
            return crc32_slice8(crc, data);
        }

        // Stream order: x3 holds the oldest 16 bytes (gets the initial crc).
        let mut x3 = load(&mut data);
        let mut x2 = load(&mut data);
        let mut x1 = load(&mut data);
        let mut x0 = load(&mut data);

        // XOR the (inverted) initial crc into the low 32 bits of the first block
        // (zero-extended, like _mm_cvtsi32_si128 — NOT broadcast).
        x3 = veorq_u64(
            x3,
            vcombine_u64(vdup_n_u64(!crc as u64), vdup_n_u64(0)),
        );

        // Fold by 4: consume 64 bytes per iteration.
        let k1k2 = vcombine_u64(vdup_n_u64(K1), vdup_n_u64(K2));
        while data.len() >= 64 {
            x3 = fold(x3, load(&mut data), k1k2);
            x2 = fold(x2, load(&mut data), k1k2);
            x1 = fold(x1, load(&mut data), k1k2);
            x0 = fold(x0, load(&mut data), k1k2);
        }

        // Fold the four accumulators into one.
        let k3k4 = vcombine_u64(vdup_n_u64(K3), vdup_n_u64(K4));
        let mut x = fold(x3, x2, k3k4);
        x = fold(x, x1, k3k4);
        x = fold(x, x0, k3k4);

        // Fold remaining whole 16-byte blocks.
        while data.len() >= 16 {
            x = fold(x, load(&mut data), k3k4);
        }

        // Reduce 128 -> 64: x = (x_lo * K4) ^ (x_hi << 64)
        let t = clmul_k(x, K4);
        let hi = vcombine_u64(vdup_n_u64(vgetq_lane_u64(x, 1)), vdup_n_u64(0));
        let x = veorq_u64(t, hi);

        // Reduce 64 -> 32: x = (x_lo32 * K5) ^ (x shifted right 4 bytes)
        let x_lo32 = vandq_u64(x, vdupq_n_u64(0xFFFF_FFFF));
        let t = clmul_k(x_lo32, K5);
        // byte-shift right by 4: (x_u32[1], x_u32[2], x_u32[3], 0)
        let shifted = vreinterpretq_u64_u32(vextq_u32(
            vreinterpretq_u32_u64(x),
            vdupq_n_u32(0),
            1,
        ));
        let x = veorq_u64(t, shifted);

        // Barrett reduction 32: x_lo32 * U', then * P, keep bits [63:32]
        let x_lo = vandq_u64(x, vdupq_n_u64(0xFFFF_FFFF));
        let t1 = clmul_k(x_lo, U_PRIME);
        let t1_lo = vandq_u64(t1, vdupq_n_u64(0xFFFF_FFFF));
        let t2 = clmul_k(t1_lo, P_X);
        let c = vgetq_lane_u32(vreinterpretq_u32_u64(veorq_u64(x, t2)), 1);

        if !data.is_empty() {
            crc32_slice8(!c, data)
        } else {
            !c
        }
    }
}

// ============================================================================
// aarch64 implementation (hardware CRC32 instructions)
// ============================================================================
// ARMv8 provides CRC32B/H/W/X instructions for the IEEE polynomial (the same
// one used by crc32fast/zlib). A serial dependency chain on CRC32X runs at
// ~8 bytes per 3 cycles; splitting the input into 4 contiguous segments with
// independent chains hides that latency, then the partials are folded with
// `crc32_combine`.
#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;
    use std::arch::aarch64::*;

    /// Minimum length for the 4-way interleaved path (below it the folding
    /// overhead outweighs the ILP win; the serial chain is already fast).
    /// Also used by the top-level dispatcher as the upper bound of the NEON
    /// PMULL path: on M1 the hardware CRC32 4-way interleave (≈23 GiB/s)
    /// beats PMULL folding (≈17 GiB/s) above this size, while PMULL beats
    /// the serial chain (≈5 GiB/s) everywhere above 256 B.
    pub const INTERLEAVE_THRESHOLD: usize = 24 * 1024;

    #[inline(always)]
    unsafe fn read_u64(p: *const u8) -> u64 {
        u64::from_le(std::ptr::read_unaligned(p as *const u64))
    }

    /// Serial hardware-CRC chain over `data`, starting from the internal
    /// (pre-inverted) state. Returns the internal state.
    #[inline(always)]
    unsafe fn chain(mut state: u32, data: &[u8]) -> u32 {
        let mut ptr = data.as_ptr();
        let mut len = data.len();
        while len >= 8 {
            state = __crc32d(state, read_u64(ptr));
            ptr = ptr.add(8);
            len -= 8;
        }
        if len >= 4 {
            state = __crc32w(
                state,
                u32::from_le(std::ptr::read_unaligned(ptr as *const u32)),
            );
            ptr = ptr.add(4);
            len -= 4;
        }
        if len >= 2 {
            state = __crc32h(
                state,
                u16::from_le(std::ptr::read_unaligned(ptr as *const u16)),
            );
            ptr = ptr.add(2);
            len -= 2;
        }
        if len == 1 {
            state = __crc32b(state, *ptr);
        }
        state
    }

    /// 4-way interleaved hardware CRC for large buffers.
    ///
    /// Splits `data` into 4 contiguous segments of `n` bytes (plus a tail),
    /// runs one independent CRC chain per segment, then folds the partial
    /// results: crc(s0||s1||s2||s3) = combine(combine(combine(f0,f1),f2),f3).
    #[target_feature(enable = "crc")]
    unsafe fn interleaved4(crc: u32, data: &[u8]) -> u32 {
        let len = data.len();
        let n = (len / 4) & !7; // segment length, a multiple of 8
        debug_assert!(n > 0);
        let base = data.as_ptr();
        let seg1 = base.add(n);
        let seg2 = base.add(2 * n);
        let seg3 = base.add(3 * n);

        let mut c0 = !crc; // segment 0 carries the incoming state
        let mut c1 = !0u32; // the others start from the standard init
        let mut c2 = !0u32;
        let mut c3 = !0u32;

        let mut i = 0usize;
        while i < n {
            c0 = __crc32d(c0, read_u64(base.add(i)));
            c1 = __crc32d(c1, read_u64(seg1.add(i)));
            c2 = __crc32d(c2, read_u64(seg2.add(i)));
            c3 = __crc32d(c3, read_u64(seg3.add(i)));
            i += 8;
        }

        let f0 = !c0;
        let f1 = !c1;
        let f2 = !c2;
        let f3 = !c3;

        let mut acc = crc32_combine(f0, f1, n as u64);
        acc = crc32_combine(acc, f2, n as u64);
        acc = crc32_combine(acc, f3, n as u64);

        // Tail bytes [4n, len): fewer than 8*4, chain handles 8/4/2/1 words.
        !chain(!acc, &data[4 * n..])
    }

    #[target_feature(enable = "crc")]
    pub unsafe fn calculate(crc: u32, data: &[u8]) -> u32 {
        if data.len() >= INTERLEAVE_THRESHOLD {
            interleaved4(crc, data)
        } else {
            !chain(!crc, data)
        }
    }
}

// ============================================================================
// Single-threaded implementation
// ============================================================================
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn crc32_single(data: &[u8]) -> u32 {
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

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn crc32_single(data: &[u8]) -> u32 {
    let len = data.len();
    // PMULL folding for small/medium inputs (needs FEAT_PMULL, detected as
    // "aes"); 4-way interleaved hardware CRC above the crossover point.
    if len >= neon::MIN_LEN
        && len < arm::INTERLEAVE_THRESHOLD
        && std::arch::is_aarch64_feature_detected!("aes")
    {
        unsafe { neon::calculate(0, data) }
    } else if std::arch::is_aarch64_feature_detected!("crc") {
        unsafe { arm::calculate(0, data) }
    } else {
        crc32_slice8(0, data)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub fn crc32_single(data: &[u8]) -> u32 {
    crc32_slice8(0, data)
}

// ============================================================================
// Parallel implementation (rayon)
// ============================================================================
#[cfg(feature = "parallel")]
pub fn crc32_parallel(data: &[u8]) -> u32 {
    use rayon::prelude::*;

    // Threshold: only use parallel for data > 1MB
    const PARALLEL_THRESHOLD: usize = 1 << 20;

    if data.len() < PARALLEL_THRESHOLD {
        return crc32_single(data);
    }

    // Split into chunks of at least 64KB each.
    let num_threads = rayon::current_num_threads().max(1);
    // For DRAM-bandwidth-bound sizes (> 32 MiB, beyond LLC), a few large
    // streams beat many small ones: on big.LITTLE-style chips the slow cores'
    // extra streams increase DRAM contention more than they add throughput
    // (measured on M1: 4 x 25MB chunks ~41-46 GiB/s vs 8 x 12.5MB ~35).
    let num_chunks = if data.len() >= 33_554_432 {
        num_threads.min(4)
    } else {
        num_threads
    };
    let chunk_size = (data.len() / num_chunks).max(64 * 1024);

    // Compute CRCs in parallel
    let chunks: Vec<(u32, usize)> = data
        .par_chunks(chunk_size)
        .map(|chunk| (crc32_single(chunk), chunk.len()))
        .collect();

    // Combine results: crc(a||b) = combine(crc(a), crc(b), len(b))
    // We combine left-to-right, passing the length of the right chunk
    if chunks.is_empty() {
        return 0;
    }

    let mut result = chunks[0].0;
    for i in 1..chunks.len() {
        result = crc32_combine(result, chunks[i].0, chunks[i].1 as u64);
    }

    result
}

// ============================================================================
// Public API
// ============================================================================
#[cfg(feature = "parallel")]
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_parallel(data)
}

#[cfg(not(feature = "parallel"))]
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_single(data)
}

// ============================================================================
// Hasher implementation for incremental hashing
// ============================================================================

use std::hash::Hasher;

/// A CRC32 hasher that implements `std::hash::Hasher`.
///
/// # Example
///
/// ```
/// use std::hash::Hasher;
/// use crc32_autotune::Crc32Hasher;
///
/// let mut hasher = Crc32Hasher::new();
/// hasher.write(b"hello ");
/// hasher.write(b"world");
/// // CRC32 of "hello world" = 222957957 (0x0D4A1185)
/// assert_eq!(hasher.finish() as u32, 222957957);
/// ```
pub struct Crc32Hasher {
    state: u32,
    // For incremental hashing, we buffer small writes
    buffer: Vec<u8>,
}

impl Default for Crc32Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32Hasher {
    /// Create a new CRC32 hasher with initial state 0.
    pub fn new() -> Self {
        Self {
            state: 0,
            buffer: Vec::with_capacity(256),
        }
    }

    /// Create a hasher with a custom initial state.
    pub fn with_initial(initial: u32) -> Self {
        Self {
            state: initial,
            buffer: Vec::with_capacity(256),
        }
    }

    /// Finalize and return the CRC32 checksum.
    pub fn finalize(&self) -> u32 {
        self.state
    }

    /// Reset the hasher to initial state.
    pub fn reset(&mut self) {
        self.state = 0;
        self.buffer.clear();
    }

    fn flush_buffer(&mut self) {
        if !self.buffer.is_empty() {
            self.state = crc32_combine(self.state, crc32(&self.buffer), self.buffer.len() as u64);
            self.buffer.clear();
        }
    }
}

impl Hasher for Crc32Hasher {
    fn finish(&self) -> u64 {
        // Note: caller should ensure buffer is flushed before calling finish
        // For safety, we compute the final CRC including any buffered data
        if self.buffer.is_empty() {
            self.state as u64
        } else {
            crc32_combine(self.state, crc32(&self.buffer), self.buffer.len() as u64) as u64
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() < 256 {
            // Buffer small writes
            self.buffer.extend_from_slice(bytes);
            if self.buffer.len() >= 1024 {
                self.flush_buffer();
            }
        } else {
            // Large writes go directly through
            self.flush_buffer();
            self.state = crc32_combine(self.state, crc32(bytes), bytes.len() as u64);
        }
    }
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
    fn test_combine() {
        let data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let mid = data.len() / 2;

        let crc_full = crc32(&data);
        let crc1 = crc32(&data[..mid]);
        let crc2 = crc32(&data[mid..]);
        let combined = crc32_combine(crc1, crc2, (data.len() - mid) as u64);

        assert_eq!(crc_full, combined);
    }
    #[test]
    fn test_parallel_path() {
        // Large enough to trigger parallel path
        let test_data: Vec<u8> = (0..=255u8).cycle().take(2_000_000).collect();
        assert_eq!(crc32(&test_data), crc32fast::hash(&test_data));
    }
    #[test]
    fn test_hasher() {
        let data = b"hello world";
        
        // Direct computation
        let direct = crc32(data);
        
        // Incremental via Hasher
        let mut hasher = Crc32Hasher::new();
        hasher.write(data);
        let incremental = hasher.finish() as u32;
        
        assert_eq!(direct, incremental);
    }
    #[test]
    fn test_hasher_chunked() {
        let data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        
        // Direct
        let direct = crc32(&data);
        
        // Chunked incremental
        let mut hasher = Crc32Hasher::new();
        for chunk in data.chunks(1024) {
            hasher.write(chunk);
        }
        let incremental = hasher.finish() as u32;
        
        assert_eq!(direct, incremental);
    }
    #[test]
    fn test_x2n_table_closure() {
        // Doubling chain consistency (spot checks) ...
        for i in 0..31 {
            assert_eq!(gf2_multiply(X2N_TABLE[i], X2N_TABLE[i]), X2N_TABLE[i + 1], "chain broken at {i}");
        }
        // ... and period-32 closure, which makes the `i & 0x1F` wrap in
        // crc32_combine exact for len2 >= 2^32.
        assert_eq!(gf2_multiply(X2N_TABLE[31], X2N_TABLE[31]), X2N_TABLE[0]);
    }
    #[test]
    fn test_single_thread_hw_paths() {
        // Sizes chosen to cross the aarch64 interleave threshold (16 KiB)
        // and exercise various tail/segment lengths. All below the parallel
        // threshold so crc32_single is what gets tested.
        for &size in &[
            10_000usize, 127, 191, 192, 193, 255, 256, 257, 16_383, 16_384, 16_391, 24_575, 24_577,
            65_536, 131_072, 131_080, 200_003, 262_144, 262_157, 500_009, 1_048_575,
        ] {
            let data: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(31) % 251) as u8).collect();
            assert_eq!(
                crc32_single(&data),
                crc32fast::hash(&data),
                "mismatch at size {size}"
            );
        }
    }
}