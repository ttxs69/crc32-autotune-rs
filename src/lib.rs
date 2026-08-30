//! # CRC32 Auto-Tune
//!
//! High-performance CRC32 implementation with SIMD and parallel processing.
//!
//! ## Features
//!
//! - **SIMD acceleration**: AVX-512, SSE PCLMULQDQ on x86_64; hardware CRC32
//!   instructions with 4-way interleaving on aarch64 (ARMv8)
//! - **Parallel processing**: Multi-threaded computation for large data
//! - **Zero dependencies**: No runtime dependencies (parallel feature is optional)
//! - **Drop-in compatible**: Implements `std::hash::Hasher` trait
//!
//! ## Usage
//!
//! ```rust
//! use crc32_autotune::{crc32, Crc32Hasher};
//!
//! // Simple usage
//! let checksum = crc32(b"hello world");
//!
//! // Incremental hashing
//! use std::hash::Hasher;
//! let mut hasher = Crc32Hasher::new();
//! hasher.write(b"hello ");
//! hasher.write(b"world");
//! let checksum = hasher.finish() as u32;
//! ```
//!
//! ## Performance
//!
//! Measured on Apple M1 (aarch64). Single-threaded dispatch:
//! - < 192 B: hardware CRC32 serial chain, ~5 GiB/s
//! - >= 192 B: NEON PMULL folding with two interleaved fold-by-4 groups
//!   (8 independent chains, ~2.25 of the PMULL unit's ~2.45 ops/cycle):
//!   ~16 GiB/s at 1 KiB rising to ~53 GiB/s at 64 KiB — within ~10% of
//!   libdeflate, the fastest known single-threaded implementation
//!
//! Multi-threaded (rayon, > 1 MiB): ~36 GiB/s (1 MiB), ~78 GiB/s (10 MiB),
//! ~46 GiB/s (100 MiB, DRAM-bound).
//! Hardware-CRC interleave and slice-by-8 remain as fallbacks for cores
//! without PMULL.
//! Note: the criterion bench inflates small-size numbers for pure callees
//! (compiler hoisting); the figures above are from controlled measurement.
//!
//! ## Feature flags
//!
//! - `parallel` (default): Enable multi-threaded processing via rayon

pub mod crc32;

pub use crc32::{crc32, crc32_combine, crc32_single, Crc32Hasher};