fn main() {
    let data: Vec<u8> = (0..=255u8).cycle().take(100_000_000).collect();
    
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
    
    println!("crc32fast:  {:.2} GiB/s ({} MB in {:.3}s)", gib/t1, mb as u64, t1);
    println!("ours:       {:.2} GiB/s ({} MB in {:.3}s)", gib/t2, mb as u64, t2);
    println!("Match: {} vs {} = {}", result1, result2, result1 == result2);
}
