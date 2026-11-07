//! Fixed-size, versioned motor targets. The ROS adapter allocates; the USB consumer does not.
use serde::Serialize;

pub const TOPIC: &str = "/safty_manager/lowcmd";
pub const COUNT: usize = 26;
pub const LEASE_MS: u64 = 380;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Target {
    pub q: i32,
    pub dq: i16,
    pub tau: i16,
    pub kp: i16,
    pub kd: i16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Zero,
    Damping,
    Position,
}

#[derive(Clone, Copy, Debug)]
pub struct Command {
    pub epoch: u64,
    pub goal: u64,
    pub sequence: u64,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub action: Action,
    pub targets: [Target; COUNT],
}
