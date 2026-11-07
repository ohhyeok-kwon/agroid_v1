//! Motor receive slots only: no joint calibration, motion calculation or commands.
use crate::{
    checksum::crc32_words,
    clock::mono_ms,
    mcu::FRAME_SIZE,
    motor_init::InitializationStatus,
    r1_profile::{
        MotorAddress, AUXILIARY_IMU, BUS_COUNT, MOTORS, RX_SLOT_OFFSET, SLOTS_PER_BUS, SLOT_SIZE,
    },
    rt::CycleTiming,
};
use serde::Serialize;
use std::{
    io,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

// basic_service 0x101854f8: object+0xb3d2 + bus*0x1f8 + id*0x38.
// Two uninterpreted MCU metadata bytes precede each motor packet.
const SLOT_OFFSET: usize = RX_SLOT_OFFSET;
pub const MAX_AGE: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MotorSample {
    pub mode: u8,
    pub timeout: bool,
    /// Driver (signed) and winding (unsigned) temperatures, in Celsius.
    pub temperature: [i16; 2],
    pub voltage_raw: u8,
    pub voltage_v: f32,
    pub tau_raw: i16,
    pub dq_raw: i16,
    pub q_raw: i32,
    pub error_code: u32,
    pub warning_code: u8,
    pub reserve: [u8; 4],
    pub exchange_byte: u8,
}

impl MotorSample {
    pub(crate) fn decode(packet: &[u8], id: u8) -> Result<Self, &'static str> {
        if packet.len() < 26 || packet[..2] != [0xfc, 0xee] {
            return Err("unsupported motor packet header");
        }
        if packet[2] & 0x0f != id {
            return Err("motor ID does not match receive slot");
        }
        let u32_at = |i| u32::from_le_bytes(packet[i..i + 4].try_into().unwrap());
        if crc32_words(&packet[2..22]) != u32_at(22) {
            return Err("motor CRC mismatch");
        }
        Ok(Self {
            mode: (packet[2] >> 4) & 7,
            timeout: packet[2] & 0x80 != 0,
            temperature: [i16::from(packet[3] as i8), i16::from(packet[4])],
            voltage_raw: packet[5],
            voltage_v: f32::from(packet[5]) / 2.0,
            tau_raw: i16::from_le_bytes(packet[6..8].try_into().unwrap()),
            dq_raw: i16::from_le_bytes(packet[8..10].try_into().unwrap()),
            q_raw: u32_at(10) as i32,
            error_code: u32_at(14),
            warning_code: (u16::from_le_bytes(packet[18..20].try_into().unwrap()) >> 13) as u8,
            reserve: packet[18..22].try_into().unwrap(),
            exchange_byte: packet[21],
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MotorSpecification {
    /// Native Spec phase stores bytes 3..14 as identification/configuration,
    /// not temperature, voltage or position. Keep unverified meanings raw.
    pub raw_fields: [u8; 11],
    pub exchange_byte: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct MotorSlot {
    pub bus: u8,
    pub id: u8,
    pub status_code: u16,
    pub message: &'static str,
    pub data: Option<MotorSample>,
    pub specification: Option<MotorSpecification>,
    pub raw_slot: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MotorData {
    pub slot_count: usize,
    pub valid_count: usize,
    pub expected_count: usize,
    pub missing_motors: Vec<MotorAddress>,
    /// The factory's bus 6/id 0 is an auxiliary IMU, not a motor.
    pub auxiliary_imu: Option<RawSlot>,
    /// Nonzero storage outside the configured motor/IMU addresses. It can carry
    /// firmware metadata; it is not evidence of an extra motor or a motor fault.
    pub unconfigured_slots: Vec<RawSlot>,
    /// Only nonzero slots are included; empty slots are not zero-valued measurements.
    pub motors: Vec<MotorSlot>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RawSlot {
    pub bus: u8,
    pub id: u8,
    pub raw_slot: Vec<u8>,
}

impl MotorData {
    pub fn decode(frame: &[u8]) -> io::Result<Self> {
        Self::decode_with_context(frame, false)
    }

    pub fn decode_with_context(frame: &[u8], specification: bool) -> io::Result<Self> {
        if frame.len() != FRAME_SIZE || frame[..2] != [0xfc, 0xfd] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid MCU frame",
            ));
        }
        let mut motors = Vec::new();
        let mut auxiliary_imu = None;
        let mut unconfigured_slots = Vec::new();
        for bus in 0..BUS_COUNT {
            for id in 0..SLOTS_PER_BUS {
                let start = SLOT_OFFSET + (bus * SLOTS_PER_BUS + id) * SLOT_SIZE;
                let slot = &frame[start..start + SLOT_SIZE];
                if slot.iter().all(|byte| *byte == 0) {
                    continue;
                }
                if (bus as u8, id as u8) == (AUXILIARY_IMU.bus, AUXILIARY_IMU.id) {
                    auxiliary_imu = Some(RawSlot {
                        bus: bus as u8,
                        id: id as u8,
                        raw_slot: slot.to_vec(),
                    });
                    continue;
                }
                if !MOTORS.contains(&MotorAddress {
                    bus: bus as u8,
                    id: id as u8,
                }) {
                    unconfigured_slots.push(RawSlot {
                        bus: bus as u8,
                        id: id as u8,
                        raw_slot: slot.to_vec(),
                    });
                    continue;
                }
                let mut spec = None;
                let (data, status_code, message) = match MotorSample::decode(&slot[2..], id as u8) {
                    Ok(data) if specification => {
                        spec = Some(MotorSpecification {
                            raw_fields: slot[5..16].try_into().unwrap(),
                            exchange_byte: data.exchange_byte,
                        });
                        (
                            None,
                            201,
                            "motor specification exchange; telemetry not interpreted",
                        )
                    }
                    Ok(data) => {
                        let (status_code, message) = if data.timeout || data.error_code != 0 {
                            (202, "motor reported fault or communication timeout")
                        } else if data.warning_code != 0 {
                            (201, "motor reported warning")
                        } else {
                            (200, "normal")
                        };
                        (Some(data), status_code, message)
                    }
                    Err(message) => (None, 202, message),
                };
                motors.push(MotorSlot {
                    bus: bus as u8,
                    id: id as u8,
                    status_code,
                    message,
                    data,
                    specification: spec,
                    raw_slot: slot.to_vec(),
                });
            }
        }
        let missing_motors = MOTORS
            .iter()
            .copied()
            .filter(|expected| {
                !motors.iter().any(|motor| {
                    motor.bus == expected.bus
                        && motor.id == expected.id
                        && (motor.data.is_some() || motor.specification.is_some())
                })
            })
            .collect();
        Ok(Self {
            slot_count: BUS_COUNT * SLOTS_PER_BUS,
            valid_count: motors
                .iter()
                .filter(|motor| motor.data.is_some() || motor.specification.is_some())
                .count(),
            expected_count: MOTORS.len(),
            missing_motors,
            auxiliary_imu,
            unconfigured_slots,
            motors,
        })
    }
}

#[derive(Serialize)]
pub struct MotorStatus<'a> {
    pub status_code: u16,
    pub message: &'a str,
    pub usb_connected: bool,
    /// Receipt of the MCU snapshot, not a timestamp produced by the motor. Display only.
    pub received_unix_ms: Option<u64>,
    /// Same receipt on CLOCK_MONOTONIC: use this for ages and ordering on this host.
    pub received_mono_ms: Option<u64>,
    pub age_ms: Option<u64>,
    /// New for every controller process (motor-initialization session).
    pub epoch: u64,
    /// USB I/O loop timing over the last window.
    pub timing: Option<CycleTiming>,
    pub execution: Option<crate::motion_tx::ExecutionStatus>,
    pub telemetry_dropped: u64,
    pub data: Option<&'a MotorData>,
    pub initialization: Option<&'a InitializationStatus>,
}

/// Cloned under the USB-thread lock so JSON is built after the lock is released.
#[derive(Clone, Default)]
pub struct MotorMonitor {
    /// (receipt instant, wall ms, CLOCK_MONOTONIC ms, data)
    latest: Option<(Instant, u64, u64, MotorData)>,
    connected: bool,
    problem: Option<String>,
    initialization: Option<InitializationStatus>,
    epoch: u64,
    timing: Option<CycleTiming>,
    execution: Option<crate::motion_tx::ExecutionStatus>,
    telemetry_dropped: u64,
}

impl MotorMonitor {
    pub fn set_execution(&mut self, value: crate::motion_tx::ExecutionStatus, dropped: u64) {
        self.execution = Some(value);
        self.telemetry_dropped = dropped;
    }
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    pub fn set_timing(&mut self, timing: CycleTiming) {
        self.timing = Some(timing);
    }

    pub fn set_initialization(&mut self, status: &InitializationStatus) {
        self.initialization = Some(status.clone());
    }

    pub fn latest_data(&self) -> Option<&MotorData> {
        self.latest.as_ref().map(|(_, _, _, data)| data)
    }

    pub fn connection(&mut self, connected: bool, error: Option<String>) {
        if connected && !self.connected {
            self.latest = None;
        }
        self.connected = connected;
        self.problem = error;
    }

    pub fn receive(&mut self, frame: &[u8]) {
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.receive_at(frame, unix_ms, mono_ms());
    }

    pub fn receive_at(&mut self, frame: &[u8], unix_ms: u64, measured_mono: u64) {
        let spec = self
            .initialization
            .as_ref()
            .is_some_and(|init| init.phase.is_specification());
        match MotorData::decode_with_context(frame, spec) {
            Ok(data) => {
                let age = Duration::from_millis(mono_ms().saturating_sub(measured_mono));
                self.latest = Some((
                    Instant::now().checked_sub(age).unwrap_or_else(Instant::now),
                    unix_ms,
                    measured_mono,
                    data,
                ));
                self.problem = None;
            }
            Err(error) => self.problem = Some(error.to_string()),
        }
    }

    pub fn status(&self) -> MotorStatus<'_> {
        let age = self.latest.as_ref().map(|(time, _, _, _)| time.elapsed());
        let data = self.latest.as_ref().map(|(_, _, _, data)| data);
        let (status_code, message) = if !self.connected {
            (202, self.problem.as_deref().unwrap_or("USB disconnected"))
        } else if let Some(problem) = &self.problem {
            (202, problem.as_str())
        } else if let Some(init) = self
            .initialization
            .as_ref()
            .filter(|init| init.status_code == 202)
        {
            (202, init.message.as_str())
        } else if data.is_none() {
            (202, "waiting for MCU data")
        } else if age.unwrap() > MAX_AGE {
            (202, "MCU motor snapshot is older than 3 seconds")
        } else if data.unwrap().motors.is_empty() {
            (202, "waiting for motor packets; MCU slots are empty")
        } else if data
            .unwrap()
            .motors
            .iter()
            .any(|motor| motor.status_code == 202)
        {
            (
                202,
                "motor packet error or motor fault; check individual slots",
            )
        } else if !data.unwrap().missing_motors.is_empty() {
            (
                202,
                "missing valid feedback from configured R1 motors; check missing_motors",
            )
        } else if data
            .unwrap()
            .motors
            .iter()
            .any(|motor| motor.status_code == 201)
        {
            (201, "motor warning; check individual slots")
        } else if let Some(init) = self
            .initialization
            .as_ref()
            .filter(|init| init.status_code != 200)
        {
            (init.status_code, init.message.as_str())
        } else if self
            .execution
            .is_some_and(|e| e.enabled && e.fault != crate::motion_tx::Fault::None)
        {
            (
                202,
                "motion watchdog active; waiting for a new command",
            )
        } else {
            (200, "all 26 configured R1 motor packets are valid")
        };
        MotorStatus {
            status_code,
            message,
            usb_connected: self.connected,
            received_unix_ms: self.latest.as_ref().map(|(_, time, _, _)| *time),
            received_mono_ms: self.latest.as_ref().map(|(_, _, mono, _)| *mono),
            age_ms: age.map(|age| age.as_millis() as u64),
            epoch: self.epoch,
            timing: self.timing,
            execution: self.execution,
            telemetry_dropped: self.telemetry_dropped,
            data,
            initialization: self.initialization.as_ref(),
        }
    }
}
