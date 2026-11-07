//! Native R1 RF framing; no USB ownership or actuator commands.
// The MCU frame CRC is owned by the controller; one implementation for every MCU reader.
#[path = "../../humanoid_controller/src/checksum.rs"]
mod checksum;
use checksum::crc32_words;
use serde::{Deserialize, Serialize};

pub const BUTTON_NAMES: [&str; 16] = [
    "R1", "L1", "start", "select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "up", "right",
    "down", "left",
];
pub const WARN_AGE_MS: u64 = 250;
pub const ERROR_AGE_MS: u64 = 750;
/// A sample stamped at least this much before the newest one means the wall clock was
/// adjusted; accept it instead of dropping input until the clock catches up.
pub const CLOCK_STEP_MS: u64 = 1000;

#[derive(Deserialize)]
pub struct McuSample {
    /// Display stamp; ages and order use `received_mono_ms` when the controller sends it.
    pub received_unix_ms: u64,
    #[serde(default)]
    pub received_mono_ms: Option<u64>,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RemoteData {
    pub keys: u16,
    pub lx: f32,
    pub ly: f32,
    pub rx: f32,
    pub ry: f32,
    /// Analog L2 trigger: `float L2` at byte 16 of the native `unitree_joystick.hpp` packet.
    pub l2: f32,
    pub buttons: [i32; 16],
    pub pressed: Vec<&'static str>,
    pub raw_frame: Vec<u8>,
}

impl RemoteData {
    pub fn decode(packet: &[u8]) -> Result<Self, String> {
        if packet.len() != 40 || packet[..2] != [0x55, 0x51] {
            return Err("invalid RF header or length".into());
        }
        let sum = packet[2..38].iter().map(|&b| u16::from(b)).sum::<u16>();
        if sum != u16::from_be_bytes([packet[38], packet[39]]) {
            return Err("RF checksum mismatch".into());
        }
        let float = |i| f32::from_le_bytes(packet[i..i + 4].try_into().unwrap());
        let [lx, rx, ry, l2, ly] = [4, 8, 12, 16, 20].map(float);
        // Like the native receiver and the SDK there is no range check (a stick may read
        // slightly past 1.0); only values no working remote can produce are rejected.
        if [lx, ly, rx, ry, l2].iter().any(|v| !v.is_finite()) {
            return Err("non-finite RF axis".into());
        }
        let keys = u16::from_le_bytes([packet[2], packet[3]]);
        Ok(Self {
            keys,
            lx,
            ly,
            rx,
            ry,
            l2,
            buttons: std::array::from_fn(|i| i32::from(keys & (1 << i) != 0)),
            pressed: BUTTON_NAMES
                .iter()
                .enumerate()
                .filter_map(|(i, &name)| (keys & (1 << i) != 0).then_some(name))
                .collect(),
            raw_frame: packet.to_vec(),
        })
    }

    pub fn axes(&self) -> Vec<f32> {
        vec![self.lx, self.ly, self.rx, self.ry]
    }
}

pub struct Receiver {
    packet: [u8; 40],
    next: usize,
    started_ms: u64,
}

impl Default for Receiver {
    fn default() -> Self {
        Self {
            packet: [0; 40],
            next: 0,
            started_ms: 0,
        }
    }
}

impl Receiver {
    pub fn reset(&mut self) {
        self.next = 0;
    }

    pub fn receive(&mut self, fragment: &[u8], now: u64) -> Result<Option<RemoteData>, String> {
        if fragment.len() != 11 {
            self.reset();
            return Err("invalid RF fragment length".into());
        }
        if now.saturating_sub(self.started_ms) > 100 {
            self.reset();
        }
        let index = usize::from(fragment[10]);
        if index == 0 {
            if fragment[..2] != [0x55, 0x51] {
                self.reset();
                return if fragment.iter().all(|&b| b == 0) {
                    Ok(None)
                } else {
                    Err("invalid RF fragment header".into())
                };
            }
            // USB can repeat a fragment until the next radio fragment arrives.
            if self.next == 1 && self.packet[..10] == fragment[..10] {
                return Ok(None);
            }
            self.reset();
            self.started_ms = now;
        } else if self.next == 0 {
            return Ok(None);
        }
        if index >= 4 {
            self.reset();
            return Err("invalid RF fragment index".into());
        }
        if index + 1 == self.next && self.packet[index * 10..(index + 1) * 10] == fragment[..10] {
            return Ok(None);
        }
        if index != self.next {
            self.reset();
            return Err("missing or reordered RF fragment".into());
        }
        self.packet[index * 10..(index + 1) * 10].copy_from_slice(&fragment[..10]);
        self.next += 1;
        if self.next != 4 {
            return Ok(None);
        }
        self.reset();
        RemoteData::decode(&self.packet).map(Some)
    }
}

#[derive(Serialize)]
pub struct RemoteStatus<'a> {
    pub status_code: u16,
    pub message: &'a str,
    pub connected: bool,
    pub received_unix_ms: Option<u64>,
    pub age_ms: Option<u64>,
    pub packets_received: u64,
    /// Discarded samples, fragments and packets. Diagnostic only: like the native receiver,
    /// a lost or corrupt piece does not change the status while valid packets keep arriving.
    pub receive_errors: u64,
    pub last_error: Option<&'a str>,
    pub data: Option<&'a RemoteData>,
}

#[derive(Default)]
pub struct Monitor {
    receiver: Receiver,
    /// (wall-clock USB stamp, monotonic ms when accepted, packet)
    latest: Option<(u64, u64, RemoteData)>,
    source_ms: Option<u64>,
    source_mono: Option<u64>,
    last_error: Option<String>,
    packets: u64,
    errors: u64,
}

impl Monitor {
    pub fn error(&mut self, message: String) {
        self.errors += 1;
        self.last_error = Some(message);
        self.receiver.reset();
    }

    /// `now`: wall clock in ms (legacy samples). `mono`: CLOCK_MONOTONIC ms, the clock of
    /// the controller's `received_mono_ms` on this host.
    pub fn receive(&mut self, sample: McuSample, now: u64, mono: u64) -> Option<(u64, RemoteData)> {
        let result = (|| {
            if let Some(stamp) = sample.received_mono_ms {
                // CLOCK_MONOTONIC never steps: age and order need no clock heuristics.
                if stamp > mono.saturating_add(100) || mono.saturating_sub(stamp) > WARN_AGE_MS {
                    return Err("stale MCU sample".into());
                }
                if self.source_mono.is_some_and(|last| stamp < last) {
                    return Ok(None);
                }
            } else {
                if sample.received_unix_ms > now.saturating_add(100)
                    || now.saturating_sub(sample.received_unix_ms) > WARN_AGE_MS
                {
                    return Err("stale MCU sample or clock mismatch".into());
                }
                // Equal or slightly older stamps are repeats or reordering; a step back of
                // CLOCK_STEP_MS or more is a wall-clock adjustment and starts a new sequence.
                if self.source_ms.is_some_and(|last| {
                    sample.received_unix_ms < last && last - sample.received_unix_ms < CLOCK_STEP_MS
                }) {
                    return Ok(None);
                }
            }
            let data = sample.data;
            if data.len() != 160 || data[..2] != [0xfe, 0xfd] {
                return Err("invalid MCU status header or length".into());
            }
            if crc32_words(&data[..156]) != u32::from_le_bytes(data[156..].try_into().unwrap()) {
                return Err("MCU status CRC mismatch".into());
            }
            self.source_ms = Some(sample.received_unix_ms);
            self.source_mono = sample.received_mono_ms;
            self.receiver.receive(
                &data[0x68..0x73],
                sample.received_mono_ms.unwrap_or(sample.received_unix_ms),
            )
        })();
        match result {
            Ok(Some(data)) => {
                self.packets += 1;
                self.latest = Some((sample.received_unix_ms, mono, data.clone()));
                Some((sample.received_unix_ms, data))
            }
            Ok(None) => None,
            Err(error) => {
                self.error(error);
                None
            }
        }
    }

    /// The code depends only on the monotonic age of the last valid packet (native rule:
    /// the remote is lost only when no valid packet arrives for a while).
    pub fn status(&self, mono: u64) -> RemoteStatus<'_> {
        let age = self
            .latest
            .as_ref()
            .map(|(_, at, _)| mono.saturating_sub(*at));
        let (code, message) = match age {
            None => (202, "waiting for wireless remote"),
            Some(age) if age >= ERROR_AGE_MS => (202, "wireless remote timed out"),
            Some(age) if age >= WARN_AGE_MS => (201, "wireless remote data delayed"),
            Some(_) => (200, "normal"),
        };
        RemoteStatus {
            status_code: code,
            message,
            connected: code == 200,
            received_unix_ms: self.latest.as_ref().map(|(time, _, _)| *time),
            age_ms: age,
            packets_received: self.packets,
            receive_errors: self.errors,
            last_error: self.last_error.as_deref(),
            data: if code == 200 {
                self.latest.as_ref().map(|(_, _, data)| data)
            } else {
                None
            },
        }
    }
}
