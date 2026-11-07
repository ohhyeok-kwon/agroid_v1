//! Small wire encoder and the factory head conversions used only by startup.
use crate::{
    checksum::crc32_words,
    r1_profile::{slot_offset, MotorAddress, TX_SLOT_OFFSET},
};

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct MotorCommand {
    pub mode: u8,
    pub tau: i16,
    pub dq: i16,
    pub q: i32,
    pub kp: i16,
    pub kd: i16,
}

impl MotorCommand {
    pub fn write(self, frame: &mut [u8], address: MotorAddress, exchange: u8, watchdog: bool) {
        let offset = slot_offset(TX_SLOT_OFFSET, address);
        let p = &mut frame[offset + 2..offset + 22];
        p.fill(0);
        p[..2].copy_from_slice(&[0xfe, 0xee]);
        p[2] = address.id | self.mode << 4 | if watchdog { 0x80 } else { 0 };
        p[3] = exchange;
        p[4..6].copy_from_slice(&self.tau.to_le_bytes());
        p[6..8].copy_from_slice(&self.dq.to_le_bytes());
        p[8..12].copy_from_slice(&self.q.to_le_bytes());
        p[12..14].copy_from_slice(&self.kp.to_le_bytes());
        p[14..16].copy_from_slice(&self.kd.to_le_bytes());
        let crc = crc32_words(&p[..16]);
        p[16..].copy_from_slice(&crc.to_le_bytes());
        frame[offset + 1] = 20;
    }
}

// f32 constants and truncation are intentional: they reproduce native FCVTZS.
pub const POSITION_SCALE: f32 = 5215.189;
pub const PITCH_RATIO: f32 = 12.666667;
pub const YAW_RATIO: f32 = 214.34805;
pub const YAW_REFERENCE: f32 = 115.0 * std::f32::consts::PI / 180.0;

pub fn pitch_position(raw: i32) -> f32 {
    (raw as f32 / PITCH_RATIO) / POSITION_SCALE
}
pub fn yaw_position(raw: i32) -> f32 {
    (-(raw as f32) / YAW_RATIO) / POSITION_SCALE
}
pub fn yaw_torque(raw: i16) -> f32 {
    (-f32::from(raw) * YAW_RATIO) / 256000.0
}

pub fn pitch_command(q: f32) -> MotorCommand {
    MotorCommand {
        mode: 1,
        q: (q * PITCH_RATIO * POSITION_SCALE) as i32,
        kp: (30.0 / (PITCH_RATIO * PITCH_RATIO) * 12800.0) as i16,
        kd: (2.0 / (PITCH_RATIO * PITCH_RATIO) * 51200.0) as i16,
        ..Default::default()
    }
}

pub fn yaw_search_command() -> MotorCommand {
    MotorCommand {
        mode: 1,
        tau: (-0.45 / YAW_RATIO * 256000.0) as i16,
        ..Default::default()
    }
}
