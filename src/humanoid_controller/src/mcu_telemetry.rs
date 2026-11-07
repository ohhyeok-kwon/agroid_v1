//! Read-only MCU telemetry handoff. Never waits for ROS on the USB thread.
use crate::{checksum::crc32_words, mcu::McuFrame};
use serde::Serialize;
use std::{
    sync::mpsc::SyncSender,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Serialize)]
pub struct McuSample {
    /// Display stamp. Consumers order and age samples by `received_mono_ms`.
    pub received_unix_ms: u64,
    pub received_mono_ms: u64,
    pub data: Vec<u8>,
}

pub struct McuTelemetry {
    sender: SyncSender<McuSample>,
    previous: Option<[u8; 11]>,
    emitted: Option<Instant>,
}

impl McuTelemetry {
    pub fn new(sender: SyncSender<McuSample>) -> Self {
        Self {
            sender,
            previous: None,
            emitted: None,
        }
    }

    pub fn reset(&mut self) {
        self.previous = None;
        self.emitted = None;
    }

    pub fn receive(&mut self, frame: &McuFrame) {
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.receive_at(frame, unix_ms, crate::clock::mono_ms());
    }

    pub fn receive_at(&mut self, frame: &McuFrame, unix_ms: u64, mono_ms: u64) {
        let data = &frame.bytes[0x138..0x1d8];
        if data[..2] != [0xfe, 0xfd]
            || crc32_words(&data[..156]) != u32::from_le_bytes(data[156..].try_into().unwrap())
        {
            return;
        }
        // Every RF fragment transition matters. Decimating to a fixed rate can
        // skip one of the four pieces. Also send a 10 Hz transport heartbeat.
        let fragment: [u8; 11] = data[0x68..0x73].try_into().unwrap();
        if self.previous == Some(fragment)
            && self
                .emitted
                .is_some_and(|t| t.elapsed() < Duration::from_millis(100))
        {
            return;
        }
        let sample = McuSample {
            received_unix_ms: unix_ms,
            received_mono_ms: mono_ms,
            data: data.to_vec(),
        };
        if self.sender.try_send(sample).is_ok() {
            self.previous = Some(fragment);
            self.emitted = Some(Instant::now());
        }
    }
}
