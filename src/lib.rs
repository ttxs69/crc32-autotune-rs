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
//! - 256 B - 24 KiB: NEON PMULL folding, ~9-17 GiB/s (2-3.4x crc32fast)
//! - > 24 KiB: 4-way interleaved hardware CRC32, ~22 GiB/s (~3x crc32fast)
//!
//! Multi-threaded (rayon, > 1 MiB): ~40-54 GiB/s (5-6x crc32fast).
//! Note: the criterion bench inflates small-size numbers for pure callees
//! (compiler hoisting); the figures above are from controlled measurement.
//!
//! ## Feature flags
//!
//! - `parallel` (default): Enable multi-threaded processing via rayon

pub mod crc32;

pub use crc32::{crc32, crc32_combine, crc32_single, Crc32Hasher};