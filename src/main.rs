//! CRC32 Auto-Tune CLI

use std::env;
use std::fs;
use std::time::Instant;

use crc32_autotune::crc32;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <file>", args[0]);
        eprintln!("       {} --bench <size_mb>", args[0]);
        std::process::exit(1);
    }
    
    if args[1] == "--bench" {
        // Benchmark mode
        let size_mb: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
        let size = size_mb * 1024 * 1024;
        
        // Generate random data
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        // Warmup
        for _ in 0..3 {
            crc32(&data);
        }
        
        // Measure
        let iterations = 5;
        let start = Instant::now();
        for _ in 0..iterations {
            crc32(&data);
        }
        let elapsed = start.elapsed();
        
        let total_bytes = size * iterations;
        let throughput_mb_s = total_bytes as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0;
        
        println!("Size: {} MB", size_mb);
        println!("Iterations: {}", iterations);
        println!("Time: {:?}", elapsed);
        println!("Throughput: {:.2} MB/s", throughput_mb_s);
        
        return;
    }
    
    // File mode
    let filename = &args[1];
    let data = fs::read(filename).expect("Failed to read file");
    
    let start = Instant::now();
    let checksum = crc32(&data);
    let elapsed = start.elapsed();
    
    println!("File: {}", filename);
    println!("Size: {} bytes", data.len());
    println!("CRC32: {:08X}", checksum);
    println!("Time: {:?}", elapsed);
    
    if data.len() > 0 {
        let throughput = data.len() as f64 / elapsed.as_secs_f64() / 1024.0 / 1024.0;
        println!("Throughput: {:.2} MB/s", throughput);
    }
}