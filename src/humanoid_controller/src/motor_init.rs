//! Hardware startup and preparation, using one USB owner and one attempt.
use crate::{
    checksum::crc32_words,
    head_init::{HeadPreparation, HeadReference},
    mcu::FRAME_SIZE,
    motor::{MotorData, MAX_AGE},
    motor_exchange::MotorExchange,
    motor_wire::MotorCommand,
    r1_profile::{
        slot_offset, R1Profile, AUXILIARY_IMU, CONFIG_END, CONFIG_OFFSET, HEAD_YAW, MOTORS,
        TX_SLOT_OFFSET,
    },
    tx::{control_frame, seal_control_frame, REQUEST_SIZE},
    usb::UsbExchange,
};
use serde::Serialize;
use std::{
    io,
    time::{Duration, Instant},
};

pub const INIT_TIMEOUT: Duration = Duration::from_secs(1);
pub const INIT_PAUSE: Duration = Duration::from_millis(500);
pub const STARTUP_SETTLE: Duration = Duration::from_secs(5);
pub const FEEDBACK_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    WaitingUsb,
    Configuring,
    StartingMotors,
    Settling,
    ReadingSpecs,
    SpecZero,
    WaitingFeedback,
    CenteringPitch,
    SeekingYaw,
    Releasing,
    Ready,
    Failed,
}

impl Phase {
    pub fn is_specification(self) -> bool {
        matches!(self, Self::ReadingSpecs | Self::SpecZero)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct InitializationStatus {
    pub phase: Phase,
    pub status_code: u16,
    pub message: String,
    pub uart_configured: bool,
    /// May have reached the MCU even if a transfer or its response failed.
    pub head_init_requested: bool,
    pub feedback_verified: bool,
    head_reference_verified: bool,
    motion_ready: bool,
    pub head: Option<HeadReference>,
}

impl Default for InitializationStatus {
    fn default() -> Self {
        Self {
            phase: Phase::WaitingUsb,
            status_code: 202,
            message: "waiting for USB before motor initialization".into(),
            uart_configured: false,
            head_init_requested: false,
            feedback_verified: false,
            head_reference_verified: false,
            motion_ready: false,
            head: None,
        }
    }
}

impl InitializationStatus {
    pub fn head_reference_verified(&self) -> bool {
        self.head_reference_verified
    }
    #[allow(clippy::too_many_arguments)]
    pub fn from_snapshot(
        phase: Phase,
        status_code: u16,
        message: String,
        uart_configured: bool,
        head_init_requested: bool,
        feedback_verified: bool,
        head_reference_verified: bool,
        motion_ready: bool,
        head: Option<HeadReference>,
    ) -> Self {
        Self {
            phase,
            status_code,
            message,
            uart_configured,
            head_init_requested,
            feedback_verified,
            head_reference_verified,
            motion_ready,
            head,
        }
    }
    pub fn motion_ready(&self) -> bool {
        self.motion_ready && self.head_reference_verified && self.feedback_verified
    }
}

#[derive(Default)]
pub struct MotorInitializer {
    status: InitializationStatus,
    attempted: bool,
    poll_motors: bool,
    feedback_deadline: Option<Instant>,
    last_feedback: Option<Instant>,
    preparation: Option<HeadPreparation>,
    exchanges: [MotorExchange; 26],
    exchange_rx: [u8; 26],
    random: u32,
}

impl MotorInitializer {
    pub fn status(&self) -> &InitializationStatus {
        &self.status
    }
    pub fn attempted(&self) -> bool {
        self.attempted
    }
    pub fn poll_motors(&self) -> bool {
        self.poll_motors
    }

    pub fn reject_profile(&mut self, error: &io::Error) {
        self.attempted = true;
        self.fail(format!("motor initialization blocked: {error}"));
    }

    /// One attempt per process. Never automatically replay the yaw command after
    /// a timeout, partial write, disconnect or failed acknowledgment.
    /// `update` also delivers the two received frames to the normal decoders.
    pub fn initialize(
        &mut self,
        profile: &R1Profile,
        usb: &mut impl UsbExchange,
        mut wait: impl FnMut(Duration) -> io::Result<()>,
        mut update: impl FnMut(&InitializationStatus, Option<&[u8]>),
    ) -> io::Result<()> {
        if self.attempted {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "motor initialization was already attempted; restart required",
            ));
        }
        self.attempted = true;
        let result = (|| {
            wait(Duration::ZERO)?;
            MotorExchange::self_check()?;
            let mut reply = [0; 6000];
            let mut request = startup_frame(profile);
            // Latch before OUT: failure to receive an ACK does not prove that
            // the MCU failed to execute the embedded initialization request.
            self.status.head_init_requested = true;
            self.stage(
                Phase::Configuring,
                "configuring UARTs and requesting head yaw initialization",
            );
            update(&self.status, None);
            let count = usb.exchange(&request, &mut reply, INIT_TIMEOUT)?;
            let reply_data = reply.get(..count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid USB reply length")
            })?;
            validate_reply(reply_data, Some(&request))?;
            self.status.uart_configured = true;
            self.stage(
                Phase::Settling,
                "UART configuration acknowledged; waiting 500ms",
            );
            update(&self.status, Some(&reply[..count]));
            wait(INIT_PAUSE)?;

            // Native code retains the exact motor/config payload in this F1.
            request[8] = 0xf1;
            seal_control_frame(&mut request);
            self.stage(
                Phase::StartingMotors,
                "confirming startup with the retained F1 payload",
            );
            update(&self.status, None);
            let count = usb.exchange(&request, &mut reply, INIT_TIMEOUT)?;
            let reply_data = reply.get(..count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid USB reply length")
            })?;
            validate_reply(reply_data, None)?;
            self.stage(
                Phase::Settling,
                "startup acknowledged; waiting 500ms then 5s for settling",
            );
            update(&self.status, Some(&reply[..count]));
            wait(INIT_PAUSE)?;
            // The caller of 0x101964e8 waits 5s before starting periodic exchange.
            wait(STARTUP_SETTLE)?;
            self.poll_motors = true;
            self.feedback_deadline = Some(Instant::now() + FEEDBACK_TIMEOUT);
            self.preparation = Some(HeadPreparation::new(Instant::now()));
            self.stage(
                Phase::ReadingSpecs,
                "exchanging motor specifications before head preparation",
            );
            update(&self.status, None);
            Ok(())
        })();
        if let Err(error) = &result {
            self.poll_motors = false;
            self.fail(format!(
                "motor initialization failed: {error}; restart required"
            ));
            update(&self.status, None);
        }
        result
    }

    pub fn observe(&mut self, data: &MotorData) {
        if !self.poll_motors || self.status.phase == Phase::Failed {
            return;
        }
        self.last_feedback = Some(Instant::now());
        for (i, address) in MOTORS.iter().enumerate() {
            if let Some(slot) = data
                .motors
                .iter()
                .find(|m| (m.bus, m.id) == (address.bus, address.id))
            {
                if let Some(sample) = &slot.data {
                    self.exchange_rx[i] = sample.exchange_byte;
                } else if let Some(spec) = &slot.specification {
                    self.exchange_rx[i] = spec.exchange_byte;
                }
            }
        }
        self.status.feedback_verified = !self.status.phase.is_specification()
            && data.missing_motors.is_empty()
            && data.motors.iter().all(|motor| motor.status_code != 202);
        if self.status.feedback_verified {
            self.feedback_deadline = None;
        }
        if let Some(preparation) = &mut self.preparation {
            let result = preparation.observe(data, Instant::now());
            self.status.head = Some(preparation.reference.clone());
            if let Err(error) = result {
                self.fail(error.to_string());
                return;
            }
            let phase = preparation.phase();
            self.stage(
                phase,
                match phase {
                    Phase::ReadingSpecs => "exchanging motor specifications",
                    Phase::SpecZero => "resetting specification requests",
                    Phase::WaitingFeedback => "checking all 26 motors before head movement",
                    Phase::CenteringPitch => {
                        "centering head pitch using the factory startup trajectory"
                    }
                    Phase::SeekingYaw => "seeking yaw reference with one-direction 0.45Nm torque",
                    Phase::Releasing => {
                        "yaw reference found; releasing motors and checking feedback"
                    }
                    Phase::Ready => "motor startup complete; head reference verified",
                    _ => "motor preparation",
                },
            );
            if phase == Phase::Ready {
                self.status.head_reference_verified = true;
                self.status.motion_ready = self.status.feedback_verified;
                self.status.status_code = 200;
            }
        }
    }

    /// Ready-state feedback without constructing heap-backed diagnostic MotorData.
    pub fn observe_ready(&mut self, frame: &[u8]) {
        if self.status.phase != Phase::Ready {
            return;
        }
        self.last_feedback = Some(Instant::now());
        let mut valid = true;
        for (i, address) in MOTORS.iter().enumerate() {
            let at = slot_offset(crate::r1_profile::RX_SLOT_OFFSET, *address) + 2;
            match crate::motor::MotorSample::decode(&frame[at..at + 26], address.id) {
                Ok(sample) => {
                    self.exchange_rx[i] = sample.exchange_byte;
                    valid &= !sample.timeout && sample.error_code == 0;
                }
                Err(_) => valid = false,
            }
        }
        self.status.feedback_verified = valid;
    }

    pub fn tick(&mut self, now: Instant) {
        if self.status.phase == Phase::Failed {
            return;
        }
        if let Some(preparation) = &self.preparation {
            if let Err(error) = preparation.tick(now) {
                self.fail(error.to_string());
                return;
            }
        }
        if self.status.feedback_verified
            && self
                .last_feedback
                .is_some_and(|last| now.saturating_duration_since(last) > MAX_AGE)
        {
            self.fail("MCU motor feedback is stale; motion blocked".into());
        }
        if self
            .feedback_deadline
            .is_some_and(|deadline| now >= deadline)
        {
            self.fail("motor feedback validation timed out after 15s; restart required".into());
        }
    }

    /// Fill only motor slots; the dispatcher owns the envelope/sequence, and
    /// external motion continues to have its own opaque forwarding module.
    pub fn prepare_frame(&mut self, frame: &mut [u8; REQUEST_SIZE]) -> io::Result<()> {
        if !self.poll_motors {
            return Ok(());
        }
        let phase = self.status.phase;
        for (i, address) in MOTORS.iter().copied().enumerate() {
            let exchange = if matches!(phase, Phase::SpecZero | Phase::WaitingFeedback) {
                0
            } else {
                // Private PRNG: the protocol uses only bit 4; no global libc lock.
                if self.random == 0 {
                    self.random = 0x7a31_209d;
                }
                self.random ^= self.random << 13;
                self.random ^= self.random >> 17;
                self.random ^= self.random << 5;
                self.exchanges[i].next(self.exchange_rx[i], self.random as u8)?
            };
            let command = if phase == Phase::Failed {
                MotorCommand::default()
            } else {
                self.preparation
                    .as_ref()
                    .map(|p| p.command(address))
                    .unwrap_or_default()
            };
            command.write(
                frame,
                address,
                exchange,
                !phase.is_specification() && phase != Phase::WaitingFeedback,
            );
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        if self.poll_motors {
            self.fail("controller stopping; motor commands released".into());
        }
    }

    pub fn disconnected(&mut self, error: &io::Error) {
        if !self.attempted {
            return;
        }
        self.poll_motors = false;
        self.last_feedback = None;
        self.status.feedback_verified = false;
        if self.status.phase != Phase::Failed {
            self.fail(format!(
                "USB connection lost: {error}; restart required for initialization"
            ));
        }
    }

    fn stage(&mut self, phase: Phase, message: &str) {
        self.status.phase = phase;
        self.status.status_code = 201;
        self.status.message = message.into();
    }

    fn fail(&mut self, message: String) {
        self.status.phase = Phase::Failed;
        self.status.status_code = 202;
        self.status.message = message;
        self.status.motion_ready = false;
        self.status.head_reference_verified = false;
        self.status.feedback_verified = false;
        self.feedback_deadline = None;
    }
}

pub(crate) fn startup_frame(profile: &R1Profile) -> [u8; REQUEST_SIZE] {
    let mut frame = control_frame(0xe2, 0);
    profile.write_config(&mut frame);
    write_motor_requests(&mut frame, true);
    seal_control_frame(&mut frame);
    frame
}

/// Native 0x10194a50(mode=2) idle requests, without replaying special startup.
pub(crate) fn polling_frame(profile: &R1Profile, sequence: u32) -> [u8; REQUEST_SIZE] {
    let mut frame = control_frame(0xf1, sequence);
    profile.write_config(&mut frame);
    write_motor_requests(&mut frame, false);
    seal_control_frame(&mut frame);
    frame
}

fn write_motor_requests(frame: &mut [u8], startup: bool) {
    for address in MOTORS {
        let mut packet = [0; 20];
        packet[..2].copy_from_slice(&[0xfe, 0xee]);
        packet[2] = address.id;
        if startup {
            let yaw = address == HEAD_YAW;
            packet[2] |= if yaw { 6 << 4 } else { 0 };
            packet[4..6].copy_from_slice(&(if yaw { -1280_i16 } else { -128_i16 }).to_le_bytes());
            if yaw {
                packet[6..8].copy_from_slice(&955_i16.to_le_bytes());
            }
        }
        let crc = crc32_words(&packet[..16]);
        packet[16..20].copy_from_slice(&crc.to_le_bytes());
        let offset = slot_offset(TX_SLOT_OFFSET, address);
        frame[offset + 1] = packet.len() as u8;
        frame[offset + 2..offset + 22].copy_from_slice(&packet);
    }
    // 0x10184620: exact native packets including its separate CRC16 variant.
    // Configuration (source=3, command=0x15, value=6) vs normal IMU query.
    let imu = if startup {
        [0xfb, 0xee, 3, 0x15, 0, 0, 0xc0, 0x40, 0x82, 0x95]
    } else {
        [0xfb, 0xee, 1, 2, 0x1e, 0x69, 0x48, 0x40, 0xf4, 0xaf]
    };
    let offset = slot_offset(TX_SLOT_OFFSET, AUXILIARY_IMU);
    frame[offset + 1] = imu.len() as u8;
    frame[offset + 2..offset + 12].copy_from_slice(&imu);
}

fn validate_reply(reply: &[u8], configuration: Option<&[u8]>) -> io::Result<()> {
    if reply.len() != FRAME_SIZE || reply[..2] != [0xfc, 0xfd] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "expected FC FD and {FRAME_SIZE} bytes, received {} bytes",
                reply.len()
            ),
        ));
    }
    if let Some(request) = configuration {
        // Check all 9 baud values and all 80 enable/timeout entries, including
        // indices 8/9 that the manufacturer's comparison loop misses.
        for offset in CONFIG_OFFSET..CONFIG_END {
            if request[offset] != reply[offset] {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("MCU UART configuration echo mismatch at 0x{offset:x}: expected {}, received {}",
                        request[offset], reply[offset])));
            }
        }
    }
    Ok(())
}
