//! R1 startup preparation. This is not the external motion planner.
//! Native stages: Spec -> Spec zero -> JLimit -> JSpecial -> release.
use crate::{
    motor::{MotorData, MotorSample},
    motor_init::Phase,
    motor_wire::{self, MotorCommand, YAW_REFERENCE},
    r1_profile::{MotorAddress, HEAD_PITCH, HEAD_YAW},
};
use serde::Serialize;
use std::{
    io,
    time::{Duration, Instant},
};

pub const ACTIVE_FEEDBACK_TIMEOUT: Duration = Duration::from_millis(100);
const UNCHANGED_TIMEOUT: Duration = Duration::from_millis(250);
const STABLE_DELTA: f32 = 0.0008726646; // native 0.05 degrees

#[derive(Clone, Copy, Default, Debug, Serialize)]
pub struct HeadReference {
    pub phase_cycles: u32,
    pub stable_windows: u8,
    pub pitch_rad: Option<f32>,
    pub yaw_uncalibrated_rad: Option<f32>,
    pub yaw_torque_nm: Option<f32>,
    pub yaw_travel_rad: f32,
    pub movement_observed: bool,
    pub load_observed: bool,
    pub yaw_limit_raw: Option<i32>,
    pub yaw_offset_rad: Option<f32>,
    pub yaw_reference_rad: Option<f32>,
}

pub struct HeadPreparation {
    phase: Phase,
    entered: Instant,
    last_valid: Instant,
    last_change: [Instant; 2],
    last_packets: [Vec<u8>; 2],
    pitch_start: f32,
    yaw_start: f32,
    window_position: Option<f32>,
    window_count: u32,
    window_loaded: u32,
    pub reference: HeadReference,
}

impl HeadPreparation {
    pub fn new(now: Instant) -> Self {
        Self {
            phase: Phase::ReadingSpecs,
            entered: now,
            last_valid: now,
            last_change: [now; 2],
            last_packets: Default::default(),
            pitch_start: 0.0,
            yaw_start: 0.0,
            window_position: None,
            window_count: 0,
            window_loaded: 0,
            reference: HeadReference::default(),
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn active(&self) -> bool {
        matches!(self.phase, Phase::CenteringPitch | Phase::SeekingYaw)
    }

    pub fn command(&self, address: MotorAddress) -> MotorCommand {
        if address == HEAD_PITCH {
            if self.phase == Phase::CenteringPitch && self.reference.phase_cycles > 0 {
                let fraction = (self.reference.phase_cycles as f32 / 800.0).min(1.0);
                return motor_wire::pitch_command(self.pitch_start * (1.0 - fraction));
            }
            if self.phase == Phase::SeekingYaw {
                return motor_wire::pitch_command(0.0);
            }
        }
        if address == HEAD_YAW && self.phase == Phase::SeekingYaw {
            return motor_wire::yaw_search_command();
        }
        MotorCommand::default()
    }

    fn enter(&mut self, phase: Phase, now: Instant) {
        self.phase = phase;
        self.entered = now;
        self.last_valid = now;
        self.last_change = [now; 2];
        self.reference.phase_cycles = 0;
    }

    pub fn tick(&self, now: Instant) -> io::Result<()> {
        if self.active() && now.saturating_duration_since(self.last_valid) > ACTIVE_FEEDBACK_TIMEOUT
        {
            return Err(io::Error::other(
                "head feedback missing for 100ms; releasing motors",
            ));
        }
        let limit = match self.phase {
            Phase::ReadingSpecs => 15,
            Phase::SpecZero | Phase::WaitingFeedback => 3,
            Phase::CenteringPitch => 4,
            Phase::SeekingYaw => 9,
            Phase::Releasing => 5,
            _ => return Ok(()),
        };
        if now.saturating_duration_since(self.entered) > Duration::from_secs(limit) {
            return Err(io::Error::other(format!(
                "{:?} timed out; releasing motors",
                self.phase
            )));
        }
        Ok(())
    }

    pub fn observe(&mut self, data: &MotorData, now: Instant) -> io::Result<()> {
        self.tick(now)?;
        self.reference.phase_cycles += 1;
        let cycle = self.reference.phase_cycles;
        let complete = data.missing_motors.is_empty();
        match self.phase {
            Phase::ReadingSpecs => {
                if cycle >= 2000 {
                    if !complete {
                        return Err(io::Error::other("incomplete motor specification exchange"));
                    }
                    self.enter(Phase::SpecZero, now);
                }
                return Ok(());
            }
            Phase::SpecZero => {
                if cycle >= 250 {
                    self.enter(Phase::WaitingFeedback, now);
                }
                return Ok(());
            }
            _ => {}
        }
        if !complete || data.motors.iter().any(|m| m.status_code == 202) {
            return Err(io::Error::other(
                "missing, corrupt or faulted motor feedback; releasing motors",
            ));
        }
        let pitch = sample(data, HEAD_PITCH)?;
        let yaw = sample(data, HEAD_YAW)?;
        let pitch_q = motor_wire::pitch_position(pitch.q_raw);
        let yaw_q = motor_wire::yaw_position(yaw.q_raw);
        let yaw_tau = motor_wire::yaw_torque(yaw.tau_raw);
        self.reference.pitch_rad = Some(pitch_q);
        self.reference.yaw_uncalibrated_rad = Some(yaw_q);
        self.reference.yaw_torque_nm = Some(yaw_tau);
        self.last_valid = now;
        if pitch_q.abs() > 37.0_f32.to_radians() {
            return Err(io::Error::other(
                "head pitch outside factory range; releasing motors",
            ));
        }
        if self.active() {
            // Constant MCU snapshots do not prove that a motor is still replying.
            // Include RX[21]: the live command/response channel changes at rest.
            for (i, address) in [HEAD_PITCH, HEAD_YAW].iter().enumerate() {
                let slot = data
                    .motors
                    .iter()
                    .find(|m| (m.bus, m.id) == (address.bus, address.id))
                    .unwrap();
                let packet = &slot.raw_slot[2..28];
                if packet != self.last_packets[i] {
                    self.last_packets[i] = packet.to_vec();
                    self.last_change[i] = now;
                } else if now.saturating_duration_since(self.last_change[i]) > UNCHANGED_TIMEOUT {
                    return Err(io::Error::other(
                        "unchanged head motor response for 250ms; releasing motors",
                    ));
                }
            }
        }
        match self.phase {
            Phase::WaitingFeedback if cycle >= 250 => {
                self.pitch_start = pitch_q;
                self.enter(Phase::CenteringPitch, now);
            }
            Phase::CenteringPitch => {
                if cycle > 250 && pitch.mode != 1 {
                    return Err(io::Error::other(
                        "pitch did not enter FOC mode; releasing motors",
                    ));
                }
                if cycle >= 1000 {
                    if pitch_q.abs() > 3.0_f32.to_radians() {
                        return Err(io::Error::other(
                            "pitch did not reach center; releasing motors",
                        ));
                    }
                    self.yaw_start = yaw_q;
                    self.enter(Phase::SeekingYaw, now);
                }
            }
            Phase::SeekingYaw => {
                if cycle > 250 && (pitch.mode != 1 || yaw.mode != 1) {
                    return Err(io::Error::other(
                        "head did not enter FOC mode; releasing motors",
                    ));
                }
                if pitch_q.abs() > 5.0_f32.to_radians() {
                    return Err(io::Error::other(
                        "pitch moved away from center during yaw search",
                    ));
                }
                let travel = yaw_q - self.yaw_start;
                self.reference.yaw_travel_rad = travel;
                self.reference.movement_observed |= travel > 0.5_f32.to_radians();
                if !(-2.0_f32.to_radians()..=235.0_f32.to_radians()).contains(&travel) {
                    return Err(io::Error::other(
                        "yaw search moved in wrong direction or exceeded travel limit",
                    ));
                }
                self.window_count += 1;
                self.window_loaded += u32::from(yaw.mode == 1 && (0.15..=0.9).contains(&yaw_tau));
                if self.window_count >= 251 {
                    let stable = self
                        .window_position
                        .is_some_and(|last| (yaw_q - last).abs() < STABLE_DELTA);
                    // Native stability alone accepts a stationary idle motor.
                    // Require live FOC and measured load through each stable window,
                    // including when booting with the head already at the endpoint.
                    let loaded = self.window_loaded >= 200;
                    self.reference.load_observed = loaded;
                    self.reference.stable_windows = if stable && loaded {
                        self.reference.stable_windows + 1
                    } else {
                        0
                    };
                    self.window_position = Some(yaw_q);
                    self.window_count = 0;
                    self.window_loaded = 0;
                    if self.reference.stable_windows >= 5 {
                        self.reference.yaw_limit_raw = Some(yaw.q_raw);
                        self.reference.yaw_offset_rad = Some(YAW_REFERENCE - yaw_q);
                        self.reference.yaw_reference_rad = Some(YAW_REFERENCE);
                        // Intentional improvement: release as soon as verified,
                        // instead of pushing until the native 9000-cycle timeout.
                        self.enter(Phase::Releasing, now);
                    }
                }
                if self.phase == Phase::SeekingYaw && cycle >= 9000 {
                    return Err(io::Error::other(
                        "yaw reference not found in 9000 cycles; releasing motors",
                    ));
                }
            }
            Phase::Releasing => {
                if cycle > 250 && (pitch.mode != 0 || yaw.mode != 0) {
                    return Err(io::Error::other("head release not acknowledged"));
                }
                if cycle >= 3000 {
                    self.enter(Phase::Ready, now);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn sample(data: &MotorData, address: MotorAddress) -> io::Result<&MotorSample> {
    data.motors
        .iter()
        .find(|m| (m.bus, m.id) == (address.bus, address.id))
        .and_then(|m| m.data.as_ref())
        .ok_or_else(|| io::Error::other("missing head telemetry"))
}
