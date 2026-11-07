//! Fixed-size USB-to-ROS snapshot. Decoding and strings are constructed on the ROS side.
use crate::{
    head_init::HeadReference,
    mcu::FRAME_SIZE,
    motion_tx::ExecutionStatus,
    motor_init::{InitializationStatus, Phase},
    rt::CycleTiming,
};

#[derive(Clone, Copy)]
pub struct Initialization {
    phase: Phase,
    code: u16,
    uart: bool,
    requested: bool,
    feedback: bool,
    reference: bool,
    ready: bool,
    head: Option<HeadReference>,
    message: [u8; 192],
    length: usize,
}
impl Initialization {
    pub fn capture(s: &InitializationStatus) -> Self {
        let mut message = [0; 192];
        let mut length = s.message.len().min(message.len());
        while !s.message.is_char_boundary(length) {
            length -= 1;
        }
        message[..length].copy_from_slice(&s.message.as_bytes()[..length]);
        Self {
            phase: s.phase,
            code: s.status_code,
            uart: s.uart_configured,
            requested: s.head_init_requested,
            feedback: s.feedback_verified,
            reference: s.head_reference_verified(),
            ready: s.motion_ready(),
            head: s.head,
            message,
            length,
        }
    }
    pub fn restore(self) -> InitializationStatus {
        InitializationStatus::from_snapshot(
            self.phase,
            self.code,
            String::from_utf8_lossy(&self.message[..self.length]).into_owned(),
            self.uart,
            self.requested,
            self.feedback,
            self.reference,
            self.ready,
            self.head,
        )
    }
}

#[derive(Clone, Copy)]
pub struct Snapshot {
    pub connected: bool,
    pub frame: Option<[u8; FRAME_SIZE]>,
    pub mono_ms: u64,
    pub unix_ms: u64,
    pub initialization: Initialization,
    pub timing: Option<CycleTiming>,
    pub execution: ExecutionStatus,
    pub dropped: u64,
}
