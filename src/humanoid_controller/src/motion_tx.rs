//! USB-cycle command execution. No ROS, JSON, allocation, waits or locks in the executor.
use crate::{
    motion_command::{Action, Command, COUNT},
    motor::MotorSample,
    motor_wire::MotorCommand,
    r1_profile::{slot_offset, JOINT_MOTORS, RX_SLOT_OFFSET, TX_SLOT_OFFSET},
    usb::UsbWrite,
};
use serde::Serialize;
use std::io;

#[derive(Clone, Copy, Default)]
pub struct Feedback {
    pub q: [i32; COUNT],
    pub received_us: u64,
    pub valid: bool,
}
impl Feedback {
    pub fn receive(&mut self, frame: &[u8], now_us: u64) {
        if frame.len() != crate::mcu::FRAME_SIZE || frame[..2] != [0xfc, 0xfd] {
            return;
        }
        self.valid = true;
        for (i, (address, _, _)) in JOINT_MOTORS.iter().enumerate() {
            let at = slot_offset(RX_SLOT_OFFSET, *address) + 2;
            match MotorSample::decode(&frame[at..at + 26], address.id) {
                Ok(sample) if !sample.timeout && sample.error_code == 0 => self.q[i] = sample.q_raw,
                _ => self.valid = false,
            }
        }
        self.received_us = now_us;
    }
    pub fn available(&self) -> bool {
        self.valid
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fault {
    None,
    CommandExpired,
    Emergency,
    InitializationLost,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ExecutionStatus {
    pub enabled: bool,
    pub active: bool,
    pub action: Option<Action>,
    pub goal: u64,
    pub fault: Fault,
    pub accepted: u64,
    pub rejected: u64,
    pub command_age_ms: Option<u64>,
    pub sent_zero_frames: u64,
    pub sent_damping_frames: u64,
    pub sent_position_frames: u64,
}

pub struct Executor {
    epoch: u64,
    command: Option<Command>,
    fault: Fault,
    accepted: u64,
    rejected: u64,
    last_goal: u64,
    last_issue: u64,
    sent: [u64; 3],
}
impl Executor {
    pub fn new(epoch: u64) -> Self {
        Self {
            epoch,
            command: None,
            fault: Fault::None,
            accepted: 0,
            rejected: 0,
            last_goal: 0,
            last_issue: 0,
            sent: [0; 3],
        }
    }
    fn stop(&mut self, reason: Fault) {
        self.fault = reason;
        self.command = None;
    }
    pub fn tick(
        &mut self,
        now_us: u64,
        ready: bool,
        feedback: &Feedback,
        incoming: Option<Command>,
        emergency: bool,
    ) -> [MotorCommand; COUNT] {
        if self.command.is_some() {
            if !ready {
                self.stop(Fault::InitializationLost);
            } else if now_us / 1000 >= self.command.unwrap().expires_ms {
                self.stop(Fault::CommandExpired);
            }
        }
        if emergency {
            self.stop(Fault::Emergency);
        }
        if let Some(cmd) = incoming {
            let valid = ready
                && feedback.available()
                && !emergency
                && cmd.epoch == self.epoch
                && cmd.issued_ms <= now_us / 1000
                && now_us / 1000 < cmd.expires_ms
                && cmd.issued_ms >= self.last_issue
                && (self.accepted == 0 || cmd.issued_ms > self.last_issue);
            if valid {
                self.last_goal = cmd.goal;
                self.last_issue = cmd.issued_ms;
                self.command = Some(cmd);
                self.fault = Fault::None;
                self.accepted += 1;
            } else {
                self.rejected += 1;
            }
        }
        if !ready {
            return [MotorCommand::default(); COUNT];
        }
        if self.fault != Fault::None {
            return damping();
        }
        let Some(cmd) = self.command else {
            return [MotorCommand::default(); COUNT];
        };
        if cmd.action == Action::Zero {
            return [MotorCommand::default(); COUNT];
        }
        if cmd.action == Action::Damping {
            return std::array::from_fn(|i| MotorCommand {
                mode: 1,
                kd: cmd.targets[i].kd,
                ..Default::default()
            });
        }
        std::array::from_fn(|i| {
            let target = cmd.targets[i];
            // Action packages own trajectories. Native policy targets are sent directly;
            // imposing a second 1 rad/s ramp here would distort the walking controller.
            MotorCommand {
                mode: 1,
                q: target.q,
                dq: target.dq,
                tau: target.tau,
                kp: target.kp,
                kd: target.kd,
            }
        })
    }
    pub fn applies(&self) -> bool {
        self.command.is_some() || self.fault != Fault::None
    }
    /// Count only a successfully completed USB write, never just a ROS receipt.
    pub fn record_applied(&mut self) {
        let action = if self.fault != Fault::None {
            Action::Damping
        } else if let Some(c) = self.command {
            c.action
        } else {
            return;
        };
        self.sent[match action {
            Action::Zero => 0,
            Action::Damping => 1,
            Action::Position => 2,
        }] += 1;
    }
    pub fn status(&self, now_us: u64) -> ExecutionStatus {
        ExecutionStatus {
            enabled: true,
            active: self.command.is_some(),
            action: self.command.map(|c| c.action),
            goal: self.last_goal,
            fault: self.fault,
            accepted: self.accepted,
            rejected: self.rejected,
            command_age_ms: (self.accepted != 0)
                .then_some((now_us / 1000).saturating_sub(self.last_issue)),
            sent_zero_frames: self.sent[0],
            sent_damping_frames: self.sent[1],
            sent_position_frames: self.sent[2],
        }
    }
}

pub fn damping() -> [MotorCommand; COUNT] {
    std::array::from_fn(|i| {
        let (_, _, r) = JOINT_MOTORS[i];
        let (kd, scale) = if i == 25 {
            (0.1, 128000000.)
        } else if i == 24 {
            (0.1, 51200.)
        } else {
            (4., 51200.)
        };
        MotorCommand {
            mode: 1,
            kd: (kd / (r * r) * scale) as i16,
            ..Default::default()
        }
    })
}

/// Reuse the normal F1 transfer and its exchange byte; never send an extra USB OUT.
pub fn write_slots(frame: &mut [u8], commands: &[MotorCommand; COUNT]) {
    for (i, (address, _, _)) in JOINT_MOTORS.iter().enumerate() {
        let exchange = frame[slot_offset(TX_SLOT_OFFSET, *address) + 5];
        commands[i].write(frame, *address, exchange, true);
    }
}

/// Legacy prepared-byte API; production motion uses write_slots in the periodic transfer.
pub(crate) fn send(usb: &mut impl UsbWrite, data: &[u8]) -> io::Result<()> {
    usb.write_packet(data)
}
