//! Shared send dispatch and status-frame encoding; no motion interpretation.
use crate::{
    checksum::crc32_words,
    command::ControllerCommand,
    motion_tx,
    motor_init::{polling_frame, InitializationStatus, MotorInitializer},
    motor_wire::MotorCommand,
    r1_profile::{R1Profile, MOTORS},
    usb::UsbWrite,
};
use std::{io, time::Duration};

pub const REQUEST_PERIOD: Duration = Duration::from_micros(950);
pub const REQUEST_SIZE: usize = 0x1474;

#[derive(Default)]
pub struct Transmitter {
    sequence: u32,
    profile: Option<R1Profile>,
    motion_ready: bool,
}

impl Transmitter {
    pub fn with_motor_polling(profile: R1Profile) -> Self {
        Self {
            sequence: 1,
            profile: Some(profile),
            motion_ready: false,
        }
    }

    pub fn update_initialization(&mut self, status: &InitializationStatus) {
        self.motion_ready = status.motion_ready();
    }

    pub fn send_status(
        &mut self,
        usb: &mut impl UsbWrite,
        initializer: &mut MotorInitializer,
    ) -> io::Result<()> {
        self.send_control(usb, initializer, None)
    }

    pub fn send_control(
        &mut self,
        usb: &mut impl UsbWrite,
        initializer: &mut MotorInitializer,
        motion: Option<&[MotorCommand; 26]>,
    ) -> io::Result<()> {
        let mut frame = self.status_frame();
        initializer.prepare_frame(&mut frame)?;
        if let Some(commands) =
            motion.filter(|_| initializer.status().phase == crate::motor_init::Phase::Ready)
        {
            motion_tx::write_slots(&mut frame, commands);
        }
        usb.write_packet(&frame)
    }

    /// Best-effort shutdown/error release, independent of the exchange cipher.
    pub fn release(&mut self, usb: &mut impl UsbWrite) -> io::Result<()> {
        if self.profile.is_some() {
            let mut frame = self.status_frame();
            for address in MOTORS {
                MotorCommand::default().write(&mut frame, address, 0, true);
            }
            usb.write_packet(&frame)?;
        }
        Ok(())
    }

    /// All request types use the same USB owner. Motion bytes are already prepared
    /// by the external producer; the ingress format/topic is not connected yet.
    pub fn send(
        &mut self,
        usb: &mut impl UsbWrite,
        command: &ControllerCommand<'_>,
    ) -> io::Result<()> {
        match command {
            ControllerCommand::ReadStatus => usb.write_packet(&self.status_frame()),
            ControllerCommand::Motion(data) if self.motion_ready => motion_tx::send(usb, data),
            ControllerCommand::Motion(_) => Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "motion blocked: motor initialization and head reference must be verified",
            )),
            ControllerCommand::Peripheral(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "peripheral command format is not implemented",
            )),
        }
    }

    fn status_frame(&mut self) -> [u8; REQUEST_SIZE] {
        let frame = match &self.profile {
            Some(profile) => polling_frame(profile, self.sequence),
            None => control_frame(0xf1, self.sequence),
        };
        self.sequence = self.sequence.wrapping_add(1);
        frame
    }
}

pub(crate) fn control_frame(command: u8, sequence: u32) -> [u8; REQUEST_SIZE] {
    let mut frame = [0; REQUEST_SIZE];
    frame[..2].copy_from_slice(&[0xfb, 0xfd]);
    frame[4..6].copy_from_slice(&[0xfa, 0xfd]);
    frame[8] = command;
    frame[11..15].copy_from_slice(&sequence.to_le_bytes());
    seal_control_frame(&mut frame);
    frame
}

pub(crate) fn seal_control_frame(frame: &mut [u8; REQUEST_SIZE]) {
    let crc = crc32_words(&frame[4..0x134]);
    frame[0x134..0x138].copy_from_slice(&crc.to_le_bytes());
}
