//! CRC32 Auto-Optimization Benchmark
//!
//! This file is READ-ONLY. The agent should NOT modify it.
//! It provides:
//! - Test data generation
//! - Reference CRC32 implementation
//! - Performance measurement
//! - Correctness verification

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Import the candidate implementation
use crc32_autotune::crc32;

// Reference implementation for correctness verification
fn reference_crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

// Generate test data with fixed seed for reproducibility
fn generate_test_data(size: usize, seed: u64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen()).collect()
}

// Test cases with different sizes
fn test_cases() -> Vec<(Vec<u8>, u32)> {
    let sizes = [
        1024,              // 1KB
        64 * 1024,         // 64KB
        1024 * 1024,       // 1MB
        10 * 1024 * 1024,  // 10MB
        100 * 1024 * 1024, // 100MB
    ];
    
    sizes.iter().enumerate().map(|(i, &size)| {
        let data = generate_test_data(size, 42 + i as u64);
        let expected = reference_crc32(&data);
        (data, expected)
    }).collect()
}

// Verify correctness against all test cases
fn verify_correctness() -> bool {
    for (i, (data, expected)) in test_cases().iter().enumerate() {
        let result = crc32(data);
        if result != *expected {
            eprintln!("Test case {} failed: expected {:08x}, got {:08x}", i, expected, result);
            return false;
        }
    }
    true
}

fn correctness_benchmark(c: &mut Criterion) {
    // First, verify correctness
    if !verify_correctness() {
        eprintln!("ERROR: Correctness verification failed!");
        std::process::exit(1);
    }
    println!("✓ All correctness tests passed!");
    
    // Benchmark different sizes
    let mut group = c.benchmark_group("crc32");
    
    for (data, expected) in test_cases() {
        let size = data.len();
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = crc32(black_box(data));
                    assert_eq!(result, expected, "Correctness check failed during benchmark");
                    result
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, correctness_benchmark);
criterion_main!(benches);