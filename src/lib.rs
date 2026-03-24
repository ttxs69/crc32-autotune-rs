//! # CRC32 Auto-Tune
//!
//! High-performance CRC32 implementation with SIMD and parallel processing.
//!
//! ## Features
//!
//! - **SIMD acceleration**: AVX-512, SSE PCLMULQDQ on x86_64
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
//! On a 2-core machine with AVX-512:
//! - Single-threaded: ~8-9 GiB/s
//! - Multi-threaded: ~17 GiB/s (2.8x faster than crc32fast)
//!
//! ## Feature flags
//!
//! - `parallel` (default): Enable multi-threaded processing via rayon

pub mod crc32;

pub use crc32::{crc32, crc32_combine, crc32_single, Crc32Hasher};