//! Controller-local L2+B handling. Only complete, checksum-valid RF packets affect it.
use crate::checksum::crc32_words;
pub struct Receiver {
    bytes: [u8; 40],
    packet: [u8; 40],
    valid: bool,
    next: u8,
    pressed: bool,
}
impl Default for Receiver {
    fn default() -> Self {
        Self {
            bytes: [0; 40],
            packet: [0; 40],
            valid: false,
            next: 0,
            pressed: false,
        }
    }
}
impl Receiver {
    pub fn active(&self) -> bool {
        self.pressed
    }
    pub fn packet(&self) -> Option<[u8; 40]> {
        self.valid.then_some(self.packet)
    }
    pub fn receive(&mut self, frame: &[u8]) {
        let Some(data) = frame.get(0x138..0x1d8) else {
            return;
        };
        if data[..2] != [0xfe, 0xfd]
            || crc32_words(&data[..156]) != u32::from_le_bytes(data[156..160].try_into().unwrap())
        {
            return;
        }
        let index = data[0x72];
        if index < 4
            && index + 1 == self.next
            && self.bytes[index as usize * 10..index as usize * 10 + 10] == data[0x68..0x72]
        {
            return;
        }
        if index == 0 {
            self.next = 0;
        }
        if index > 3 || index != self.next {
            self.next = 0;
            return;
        }
        self.bytes[index as usize * 10..index as usize * 10 + 10]
            .copy_from_slice(&data[0x68..0x72]);
        self.next += 1;
        if self.next != 4 {
            return;
        }
        self.next = 0;
        let b = &self.bytes;
        if b[..2] != [0x55, 0x51]
            || b[2..38].iter().map(|v| u16::from(*v)).sum::<u16>()
                != u16::from_be_bytes([b[38], b[39]])
        {
            return;
        }
        let buttons = u16::from_le_bytes([b[2], b[3]]);
        self.packet = self.bytes;
        self.valid = true;
        self.pressed = buttons & 0x220 == 0x220;
    }
}
