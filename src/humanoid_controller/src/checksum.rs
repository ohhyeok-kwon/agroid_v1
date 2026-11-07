//! CRC shared by MCU frames, BMS packets and the RF remote (`remote_controller` includes this file).

const POLY: u32 = 0x04c1_1db7;
/// MSB-first table: `TABLE[b]` is eight bit steps of a register whose top byte is `b`.
const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            c = if c & 0x8000_0000 != 0 {
                (c << 1) ^ POLY
            } else {
                c << 1
            };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// Initial 0xffffffff, polynomial 0x04c11db7, little-endian u32 words,
/// most-significant bit first, no final xor. Same result as the 32 bit steps per word,
/// with a fixed four table steps (the 1 kHz USB path computes this every cycle).
pub fn crc32_words(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for word in bytes.chunks_exact(4) {
        crc ^= u32::from_le_bytes(word.try_into().unwrap());
        for _ in 0..4 {
            crc = (crc << 8) ^ TABLE[(crc >> 24) as usize];
        }
    }
    crc
}
