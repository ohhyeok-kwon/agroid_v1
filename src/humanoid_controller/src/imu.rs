//! R1 body IMU (UART 6/id 0), native basic_service 0x101847e8.
//! On sect=10 this sensor feeds LowState; MCU mainboard IMU is secondary.
use crate::r1_profile::{slot_offset, AUXILIARY_IMU, RX_SLOT_OFFSET};

#[derive(Clone, Copy, Debug)]
pub struct Imu {
    pub quaternion: [f32; 4],
    pub gyroscope: [f32; 3],
    pub accelerometer: [f32; 3],
    pub rpy: [f32; 3],
    pub temperature: i16,
}
impl Imu {
    pub fn decode(frame: &[u8]) -> Option<Self> {
        let at = slot_offset(RX_SLOT_OFFSET, AUXILIARY_IMU) + 2;
        let data = frame.get(at..at + 34)?;
        if data[..2] != [0xfd, 0xee]
            || crc16_imu(&data[..32]) != u16::from_le_bytes(data[32..].try_into().ok()?)
        {
            return None;
        }
        let float = |offset| f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let short = |offset| {
            f32::from(i16::from_le_bytes(
                data[offset..offset + 2].try_into().unwrap(),
            ))
        };
        let quaternion = [16, 20, 24, 28].map(float);
        let (accelerometer, gyroscope) = match data[3] {
            0 => (
                [4, 6, 8].map(|o| short(o) / 100.),
                [10, 12, 14].map(|o| short(o) * 0.001745329),
            ),
            0x15 => (
                [8, 6, 4].map(|o| -short(o) * 3.0517578e-5 * 3. * 9.80665),
                [14, 12, 10].map(|o| -short(o) * 3.0517578e-5 * 2000. * 0.01745329),
            ),
            _ => return None,
        };
        let temperature = i16::from(data[2] as i8);
        if quaternion
            .iter()
            .chain(&accelerometer)
            .chain(&gyroscope)
            .any(|x| !x.is_finite())
        {
            return None;
        }
        if quaternion.iter().map(|v| v * v).sum::<f32>() < 1e-6 {
            return None;
        }
        let [w, x, y, z] = quaternion;
        let rpy = [
            (2. * (w * x + y * z)).atan2(1. - 2. * (x * x + y * y)),
            (2. * (w * y - z * x)).clamp(-1., 1.).asin(),
            (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z)),
        ];
        Some(Self {
            quaternion,
            gyroscope,
            accelerometer,
            rpy,
            temperature,
        })
    }
}

/// Native auxiliary IMU CRC: reflected 0x1021, seed 0xab12, no final xor.
pub fn crc16_imu(bytes: &[u8]) -> u16 {
    let mut crc = 0xab12u16;
    for &byte in bytes {
        crc ^= u16::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0x8408 } else { 0 };
        }
    }
    crc
}
