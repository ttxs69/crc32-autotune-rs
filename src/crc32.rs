//! CRC32 - Optimized slice-by-8

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

#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = !0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_empty() { assert_eq!(crc32(b""), 0); }
    #[test] fn test_hello_world() { assert_eq!(crc32(b"Hello, World!"), 0xEC4AC3D0); }
    #[test] fn test_123456789() { assert_eq!(crc32(b"123456789"), 0xCBF43926); }
    #[test] fn test_matches_reference() {
        let test_data: Vec<u8> = (0..=255).collect();
        assert_eq!(crc32(&test_data), crc32fast::hash(&test_data));
    }
}