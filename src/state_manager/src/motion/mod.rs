//! Joint-space generation contract between state_manager and the action managers.
//! One copy: the action managers include this directory by path. Not USB motor packets.
#[path = "../../../humanoid_controller/src/clock.rs"]
mod clock;
pub mod profile;
/// Action-side node loop; compiled only in the action managers (`ros` feature).
#[cfg(feature = "ros")]
pub mod runtime;

pub use clock::mono_ms;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

pub const GOAL_TOPIC: &str = "/motion_manager/goal";
pub const RESULT_TOPIC: &str = "/motion_manager/result";

/// Wall clock: identifiers, display and ROS header stamps only. Ages and leases use `mono_ms`.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Both values on the same clock (CLOCK_MONOTONIC for every lease and receipt on this host).
pub fn fresh(stamp: u64, now: u64, max_age: u64) -> bool {
    stamp <= now.saturating_add(100) && now.saturating_sub(stamp) <= max_age
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    ZeroMoment,
    Damping,
    LockedStanding,
    Walking,
}

impl Action {
    pub fn fsm_id(self) -> i32 {
        match self {
            Self::ZeroMoment => 0,
            Self::Damping => 1,
            Self::LockedStanding => 4,
            Self::Walking => 811,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Goal {
    pub id: String,
    pub action: Action,
    pub started_mono_ms: u64,
    pub issued_mono_ms: u64,
    /// URDF joint order in profile::NAMES, rad. Required for a new stance.
    pub start_positions: Option<[f64; 26]>,
}

impl Goal {
    pub fn same_activation(&self, other: &Self) -> bool {
        self.id == other.id
            && self.action == other.action
            && self.started_mono_ms == other.started_mono_ms
            && self.start_positions == other.start_positions
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct JointCommand {
    pub name: String,
    pub position_rad: f64,
    pub velocity_rad_s: f64,
    pub torque_nm: f64,
    pub kp: f64,
    pub kd: f64,
}

pub fn commands(q: [f64; 26], kp: [f64; 26], kd: [f64; 26]) -> Vec<JointCommand> {
    (0..26)
        .map(|i| JointCommand {
            name: profile::NAMES[i].into(),
            position_rad: q[i],
            velocity_rad_s: 0.0,
            torque_nm: 0.0,
            kp: kp[i],
            kd: kd[i],
        })
        .collect()
}

pub fn validate_commands(joints: &[JointCommand]) -> Result<(), String> {
    if joints.len() != 26 {
        return Err("expected all 26 joint commands".into());
    }
    for (i, joint) in joints.iter().enumerate() {
        if joint.name != profile::NAMES[i]
            || !joint.position_rad.is_finite()
            || !joint.velocity_rad_s.is_finite()
            || !joint.torque_nm.is_finite()
            || !joint.kp.is_finite()
            || !joint.kd.is_finite()
            || joint.kp < 0.0
            || joint.kd < 0.0
        {
            return Err(format!("invalid joint command: {}", profile::NAMES[i]));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ActionPhase {
    Transitioning,
    Holding,
    Unavailable,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActionResult {
    pub goal_id: String,
    pub action: Action,
    pub sequence: u64,
    pub generated_mono_ms: u64,
    pub status_code: u16,
    pub phase: ActionPhase,
    pub message: String,
    pub joints: Vec<JointCommand>,
}

pub struct Output {
    pub phase: ActionPhase,
    pub joints: Vec<JointCommand>,
}

/// Pure generators own action calculation; the shared runner only owns ROS I/O.
pub trait Generator: Send + 'static {
    const ACTION: Action;
    const NODE: &'static str;
    fn start(&mut self, goal: &Goal) -> Result<(), String>;
    fn sample(&self, elapsed_s: f64) -> Result<Output, String>;
}
