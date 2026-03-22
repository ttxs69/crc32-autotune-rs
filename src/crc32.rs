//! CRC32 Implementation - Agent can modify this file
//!
//! Goal: Maximize throughput (MB/s) while maintaining correctness.
//!
//! The `crc32()` function must:
//! - Take a `&[u8]` as input
//! - Return a u32 CRC32 checksum
//! - Produce identical results to the standard CRC32 (IEEE 802.3)

/// CRC32 lookup table (256 entries, generated at compile time)
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let poly: u32 = 0xEDB88320;
    
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Calculate CRC32 checksum using standard table lookup method.
///
/// This is the baseline implementation - Agent should optimize this.
///
/// # Arguments
/// * `data` - Input bytes
///
/// # Returns
/// * CRC32 checksum as u32
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    
    crc ^ 0xFFFFFFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn test_hello_world() {
        // Known CRC32 of "Hello, World!"
        assert_eq!(crc32(b"Hello, World!"), 0xEC4AC3D0);
    }

    #[test]
    fn test_123456789() {
        // Standard test vector
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_matches_reference() {
        // Compare with crc32fast crate behavior
        let test_data: Vec<u8> = (0..=255).collect();
        let expected = crc32fast::hash(&test_data);
        assert_eq!(crc32(&test_data), expected);
    }
}