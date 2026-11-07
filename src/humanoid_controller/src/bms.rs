//! BMS-only fragment assembly, field decoding and health decisions.
pub use crate::checksum::crc32_words;
use serde::Serialize;
use std::{
    io,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const MAX_AGE: Duration = Duration::from_secs(3);
const PACKET_SIZE: usize = 168;
const CHUNK_SIZE: usize = 10;
const CHUNKS: usize = PACKET_SIZE.div_ceil(CHUNK_SIZE);

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn supported_header(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0x55, 0x51, 0x0f, 168]) || bytes.starts_with(&[0xfe, 0xef, 0x78, 168])
}

#[derive(Clone, Debug, Serialize)]
pub struct BmsData {
    pub version_high: u8,
    pub version_low: u8,
    #[serde(rename = "fn")]
    pub fn_code: u8,
    pub soc: u8,
    pub soh: u8,
    pub cell_vol: Vec<u16>,
    pub bmsvoltage: [u32; 2],
    pub current: i32,
    pub temperature: [i16; 12],
    pub cycle: u16,
    pub manufacturer_date: Option<u16>,
    pub bmsstate: [Option<u32>; 5],
    pub reserve: Option<[u32; 3]>,
    /// Preserves all bytes, including fields whose meanings are not yet verified.
    pub raw_frame: Vec<u8>,
}

impl BmsData {
    pub fn decode(packet: &[u8]) -> io::Result<Self> {
        if packet.len() != PACKET_SIZE || !supported_header(packet) {
            return Err(invalid("unexpected BMS header, model or length"));
        }
        let u16_at = |i| u16::from_le_bytes([packet[i], packet[i + 1]]);
        let u32_at = |i| u32::from_le_bytes(packet[i..i + 4].try_into().unwrap());
        if crc32_words(&packet[..164]) != u32_at(164) {
            return Err(invalid("BMS CRC mismatch"));
        }
        if u16_at(10) > 1000 || u16_at(12) > 1000 {
            return Err(invalid("BMS SOC/SOH outside 0..100%"));
        }
        let soc = u16_at(10) / 10;
        let soh = u16_at(12) / 10;
        Ok(Self {
            version_high: packet[7],
            version_low: packet[6],
            fn_code: packet[2],
            soc: soc as u8,
            soh: soh as u8,
            cell_vol: (0..28).map(|i| u16_at(20 + i * 2)).collect(),
            bmsvoltage: [u32::from(u16_at(16)) * 10, u32::from(u16_at(14)) * 10],
            current: u32_at(132) as i32,
            temperature: std::array::from_fn(|i| (u16_at(76 + i * 2) as i16) / 10),
            cycle: u16_at(18),
            manufacturer_date: None,
            bmsstate: [
                Some(u32::from(u16_at(4))),
                None,
                Some(u32_at(152)),
                None,
                Some(u32::from(u16_at(8))),
            ],
            reserve: None,
            raw_frame: packet.to_vec(),
        })
    }
}

#[derive(Clone)]
pub struct BmsReceiver {
    packet: [u8; CHUNKS * CHUNK_SIZE],
    seen: [bool; CHUNKS],
    active: bool,
}

impl Default for BmsReceiver {
    fn default() -> Self {
        Self {
            packet: [0; CHUNKS * CHUNK_SIZE],
            seen: [false; CHUNKS],
            active: false,
        }
    }
}

impl BmsReceiver {
    pub fn reset(&mut self) {
        self.active = false;
        self.seen.fill(false);
    }

    pub fn receive(&mut self, fragment: &[u8; 11]) -> io::Result<Option<BmsData>> {
        let index = usize::from(fragment[10]);
        if index == 0 {
            self.reset();
            if !supported_header(fragment) {
                return Ok(None);
            }
            self.active = true;
        }
        if !self.active {
            return Ok(None);
        }
        if index >= CHUNKS {
            self.reset();
            return Err(invalid("invalid BMS fragment index"));
        }
        self.packet[index * CHUNK_SIZE..(index + 1) * CHUNK_SIZE].copy_from_slice(&fragment[..10]);
        self.seen[index] = true;
        if self.seen.iter().all(|seen| *seen) {
            self.active = false;
            return BmsData::decode(&self.packet[..PACKET_SIZE]).map(Some);
        }
        Ok(None)
    }
}

#[derive(Serialize)]
pub struct BmsStatus<'a> {
    pub status_code: u16,
    pub message: &'a str,
    pub usb_connected: bool,
    pub received_unix_ms: Option<u64>,
    /// Same receipt on CLOCK_MONOTONIC: use this for ages on this host.
    pub received_mono_ms: Option<u64>,
    pub age_ms: Option<u64>,
    pub data: Option<&'a BmsData>,
}

/// Cloned under the USB-thread lock so JSON is built after the lock is released.
#[derive(Clone, Default)]
pub struct BmsMonitor {
    receiver: BmsReceiver,
    /// (receipt instant, wall ms, CLOCK_MONOTONIC ms, data)
    latest: Option<(Instant, u64, u64, BmsData)>,
    connected: bool,
    problem: Option<String>,
}

impl BmsMonitor {
    pub fn connection(&mut self, connected: bool, error: Option<String>) {
        if !connected {
            self.receiver.reset();
        }
        self.connected = connected;
        self.problem = error;
    }

    pub fn receive(&mut self, fragment: &[u8; 11]) {
        match self.receiver.receive(fragment) {
            Ok(Some(data)) => {
                let unix_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                self.latest = Some((Instant::now(), unix_ms, crate::clock::mono_ms(), data));
                self.problem = None;
            }
            Ok(None) => {}
            Err(error) => self.problem = Some(error.to_string()),
        }
    }

    pub fn status(&self) -> BmsStatus<'_> {
        let age = self.latest.as_ref().map(|(time, _, _, _)| time.elapsed());
        let data = self.latest.as_ref().map(|(_, _, _, data)| data);
        let (code, message) = if !self.connected {
            (202, self.problem.as_deref().unwrap_or("USB disconnected"))
        } else if data.is_none() {
            (
                202,
                self.problem.as_deref().unwrap_or("waiting for BMS data"),
            )
        } else if age.unwrap() > MAX_AGE {
            (202, "BMS data is older than 3 seconds")
        } else if data.unwrap().soc < 5 {
            (202, "battery below 5%")
        } else if let Some(problem) = &self.problem {
            (202, problem.as_str())
        } else {
            (200, "normal")
        };
        BmsStatus {
            status_code: code,
            message,
            usb_connected: self.connected,
            received_unix_ms: self.latest.as_ref().map(|(_, time, _, _)| *time),
            received_mono_ms: self.latest.as_ref().map(|(_, _, mono, _)| *mono),
            age_ms: age.map(|age| age.as_millis() as u64),
            data,
        }
    }
}
