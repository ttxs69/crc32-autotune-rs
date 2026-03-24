fn main() {
    // Test with different sizes
    let sizes = [
        (100_000_000, "100 MB"),
        (500_000_000, "500 MB"),
        (1_000_000_000, "1 GB"),
    ];

    for (size, label) in sizes {
        let data: Vec<u8> = (0..=255u8).cycle().take(size).collect();

        // Test crc32fast
        let start = std::time::Instant::now();
        let result1 = crc32fast::hash(&data);
        let t1 = start.elapsed().as_secs_f64();

        // Test our implementation
        let start = std::time::Instant::now();
        let result2 = crc32_autotune::crc32(&data);
        let t2 = start.elapsed().as_secs_f64();

        let mb = data.len() as f64 / 1024.0 / 1024.0;
        let gib = mb / 1024.0;

        println!("\n=== {} ===", label);
        println!("crc32fast:  {:.2} GiB/s ({:.3}s)", gib/t1, t1);
        println!("ours:       {:.2} GiB/s ({:.3}s)", gib/t2, t2);
        println!("speedup:    {:.2}x", t1/t2);
        println!("Match: {} vs {} = {}", result1, result2, result1 == result2);
    }
}