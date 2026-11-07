//! Manufacturer LowCmd fields and CRC layout, independent of ROS allocation.
use crate::checksum::crc32_words;

#[derive(Clone, Copy, Debug, Default)]
pub struct Motor {
    pub mode: u8,
    pub q: f32,
    pub dq: f32,
    pub tau: f32,
    pub kp: f32,
    pub kd: f32,
    pub reserve: u32,
}
#[derive(Clone, Copy)]
pub struct LowCommand {
    pub mode_pr: u8,
    pub mode_machine: u8,
    pub motors: [Motor; 35],
    pub reserve: [u32; 4],
    pub crc: u32,
}
impl Default for LowCommand {
    fn default() -> Self {
        Self {
            mode_pr: 0,
            mode_machine: 1,
            motors: [Motor::default(); 35],
            reserve: [0; 4],
            crc: 0,
        }
    }
}
impl LowCommand {
    pub fn checksum(&self) -> u32 {
        // C/C++ LowCmd: two mode bytes, two zero alignment bytes, 35*28-byte motors,
        // four reserve words, CRC. Padding must never contain uninitialized bytes.
        let mut bytes = [0u8; 1000];
        bytes[0] = self.mode_pr;
        bytes[1] = self.mode_machine;
        for (i, m) in self.motors.iter().enumerate() {
            let at = 4 + i * 28;
            bytes[at] = m.mode;
            for (j, v) in [m.q, m.dq, m.tau, m.kp, m.kd].iter().enumerate() {
                bytes[at + 4 + j * 4..at + 8 + j * 4].copy_from_slice(&v.to_le_bytes());
            }
            bytes[at + 24..at + 28].copy_from_slice(&m.reserve.to_le_bytes());
        }
        for (i, r) in self.reserve.iter().enumerate() {
            bytes[984 + i * 4..988 + i * 4].copy_from_slice(&r.to_le_bytes());
        }
        crc32_words(&bytes)
    }
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.crc != self.checksum() {
            return Err("LowCmd CRC mismatch");
        }
        if self.mode_pr != 0 || self.mode_machine != 1 {
            return Err("unsupported LowCmd robot or joint mode");
        }
        for m in self.motors {
            if m.mode > 1
                || [m.q, m.dq, m.tau, m.kp, m.kd]
                    .iter()
                    .any(|v| !v.is_finite())
                || !(0.0..=2000.0).contains(&m.kp)
                || !(0.0..=25.0).contains(&m.kd)
            {
                return Err("invalid LowCmd motor values");
            }
        }
        Ok(())
    }
}
