//! Safety decisions in joint space. Motion generation remains in the action packages.
#[path = "../../humanoid_controller/src/checksum.rs"]
mod checksum;
#[path = "../../humanoid_controller/src/clock.rs"]
pub mod clock;
pub mod config;
#[path = "../../humanoid_controller/src/r1_profile.rs"]
#[allow(dead_code)]
pub mod hardware;
pub mod layers;
#[path = "../../humanoid_controller/src/lowcmd.rs"]
pub mod lowcmd;
#[path = "../../state_manager/src/motion/profile.rs"]
pub mod profile;

use config::Config;
use lowcmd::LowCommand;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Warning,
    Fault,
}

#[derive(Clone, Debug, Serialize)]
pub struct Issue {
    pub layer: &'static str,
    pub level: Level,
    pub source: &'static str,
    pub joint: Option<&'static str>,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct Feedback {
    pub received_ms: u64,
    pub position: [f64; 26],
    pub velocity: [f64; 26],
}
pub struct BootSample {
    pub received_ms: u64,
    pub names: Vec<String>,
    pub codes: Vec<i32>,
}
pub struct Accepted {
    pub received_ms: u64,
    pub command: LowCommand,
}
struct Fault {
    id: String,
    message: String,
    clear_since: Option<u64>,
    corrective_forwarded: bool,
}
struct SafeGoal {
    fault_id: String,
    goal_hash: u32,
    epoch: u64,
}
#[derive(Deserialize)]
struct State {
    state: String,
    goal_id: Option<String>,
    controller_epoch: Option<u64>,
    safety: Option<StateSafety>,
}
#[derive(Deserialize)]
struct StateSafety {
    active: bool,
    fault_id: Option<String>,
}

#[derive(Serialize)]
pub struct Status {
    pub status_code: u16,
    pub active: bool,
    pub fault_id: Option<String>,
    pub message: String,
    pub requested_action: &'static str,
    pub goal_acknowledged: bool,
    pub corrective_command_forwarded: bool,
    pub warning_layers: Vec<Issue>,
    pub fault_layers: Vec<Issue>,
    pub received_commands: u64,
    pub forwarded_commands: u64,
    pub blocked_commands: u64,
    pub updated_mono_ms: u64,
}

pub struct Guard {
    config: Config,
    session: String,
    started: u64,
    generation: u64,
    feedback: Option<Feedback>,
    previous_feedback: Option<Feedback>,
    boot: Option<BootSample>,
    accepted: Option<Accepted>,
    safe_goal: Option<SafeGoal>,
    fault: Option<Fault>,
    sensor_issues: Vec<Issue>,
    command_issues: Vec<Issue>,
    received: u64,
    forwarded: u64,
}

impl Guard {
    pub fn new(config: Config, session: String, now: u64) -> Self {
        Self {
            config,
            session,
            started: now,
            generation: 0,
            feedback: None,
            previous_feedback: None,
            boot: None,
            accepted: None,
            safe_goal: None,
            fault: None,
            sensor_issues: Vec::new(),
            command_issues: Vec::new(),
            received: 0,
            forwarded: 0,
        }
    }
    pub fn receive_feedback(&mut self, feedback: Feedback, now: u64) {
        self.previous_feedback = self.feedback.replace(feedback);
        self.tick(now);
    }
    pub fn receive_boot(&mut self, sample: BootSample, now: u64) {
        self.boot = Some(sample);
        self.tick(now);
    }
    pub fn receive_state(&mut self, text: &str) {
        let Ok(state) = serde_json::from_str::<State>(text) else {
            return;
        };
        self.safe_goal = match (state.safety, state.goal_id, state.controller_epoch) {
            (Some(safety), Some(goal_id), Some(epoch))
                if safety.active && state.state == "locked_standing" =>
            {
                safety.fault_id.map(|fault_id| SafeGoal {
                    fault_id,
                    goal_hash: goal_hash(&goal_id),
                    epoch,
                })
            }
            _ => None,
        };
    }
    fn inspect(&self, command: Option<&LowCommand>, now: u64, fallback_start: bool) -> Vec<Issue> {
        let context = layers::Context {
            now,
            started: self.started,
            command,
            previous_command: self.accepted.as_ref(),
            feedback: self.feedback.as_ref(),
            previous_feedback: self.previous_feedback.as_ref(),
            boot: self.boot.as_ref(),
            fallback_start,
        };
        let mut issues = Vec::new();
        layers::command_format::inspect(&self.config.command_format, &context, &mut issues);
        layers::joint_range::inspect(&self.config.joint_range, &context, &mut issues);
        layers::sudden_motion::inspect(&self.config.sudden_motion, &context, &mut issues);
        layers::speed::inspect(&self.config.speed, &context, &mut issues);
        layers::boot_status::inspect(&self.config.boot_status, &context, &mut issues);
        layers::feedback::inspect(&self.config.feedback, &context, &mut issues);
        issues
    }
    pub fn tick(&mut self, now: u64) {
        self.sensor_issues = self.inspect(None, now, false);
        self.reconcile(now);
    }
    fn reconcile(&mut self, now: u64) {
        let reason = self
            .sensor_issues
            .iter()
            .chain(&self.command_issues)
            .find(|issue| issue.level == Level::Fault)
            .map(|issue| format!("{}: {}", issue.layer, issue.message));
        if let Some(message) = reason {
            if let Some(fault) = &mut self.fault {
                fault.message = message;
                fault.clear_since = None;
            } else {
                self.generation += 1;
                self.fault = Some(Fault {
                    id: format!("{}:{}", self.session, self.generation),
                    message,
                    clear_since: None,
                    corrective_forwarded: false,
                });
                self.safe_goal = None;
            }
        } else if let Some(fault) = &mut self.fault {
            // Recovery does not resume the action interrupted by this incident.
            if fault.corrective_forwarded {
                let since = *fault.clear_since.get_or_insert(now);
                if now.saturating_sub(since) >= self.config.recovery_hold_ms {
                    self.fault = None;
                    self.safe_goal = None;
                }
            }
        }
    }
    fn is_safe_goal(&self, command: &LowCommand) -> bool {
        match (&self.fault, &self.safe_goal) {
            (Some(fault), Some(goal)) => {
                fault.id == goal.fault_id
                    && command.reserve[3] == goal.goal_hash
                    && (u64::from(command.reserve[0]) | u64::from(command.reserve[1]) << 32)
                        == goal.epoch
                    && hardware::NATIVE_INDICES.iter().all(|&i| {
                        let motor = command.motors[i];
                        motor.mode == 1 && motor.kp.is_finite() && motor.kp > 0.0
                    })
            }
            _ => false,
        }
    }
    pub fn command(&mut self, command: LowCommand, now: u64) -> bool {
        self.received += 1;
        let safe_goal = self.is_safe_goal(&command);
        let first_corrective = safe_goal
            && self
                .accepted
                .as_ref()
                .is_none_or(|previous| previous.command.reserve[3] != command.reserve[3]);
        let issues = self.inspect(Some(&command), now, first_corrective);
        (self.command_issues, self.sensor_issues) = issues
            .into_iter()
            .partition(|issue| issue.source == "command");
        self.reconcile(now);
        let command_fault = self
            .command_issues
            .iter()
            .any(|issue| issue.level == Level::Fault);
        // Keep explicit remote emergency damping available; do not substitute it
        // automatically for the requested locked-standing fallback.
        let damping = is_damping(&command)
            && !self
                .command_issues
                .iter()
                .any(|issue| issue.layer == "command_format");
        let forward = damping || (!command_fault && (self.fault.is_none() || safe_goal));
        if forward {
            self.forwarded += 1;
            self.accepted = Some(Accepted {
                received_ms: now,
                command,
            });
            if let Some(fault) = &mut self.fault {
                if safe_goal || damping {
                    fault.corrective_forwarded = true;
                }
            }
        }
        forward
    }
    pub fn marker(&self) -> (u16, Option<String>) {
        (
            self.code(),
            self.fault.as_ref().map(|fault| fault.id.clone()),
        )
    }
    fn code(&self) -> u16 {
        if self.fault.is_some() {
            202
        } else if self
            .sensor_issues
            .iter()
            .chain(&self.command_issues)
            .any(|issue| issue.level == Level::Warning)
        {
            201
        } else {
            200
        }
    }
    pub fn status(&self, now: u64) -> Status {
        let issues = || self.sensor_issues.iter().chain(&self.command_issues);
        Status {
            status_code: self.code(),
            active: self.fault.is_some(),
            fault_id: self.fault.as_ref().map(|fault| fault.id.clone()),
            message: self
                .fault
                .as_ref()
                .map(|fault| fault.message.clone())
                .or_else(|| issues().next().map(|issue| issue.message.clone()))
                .unwrap_or_else(|| "normal".into()),
            requested_action: "locked_standing",
            goal_acknowledged: self
                .fault
                .as_ref()
                .zip(self.safe_goal.as_ref())
                .is_some_and(|(fault, goal)| fault.id == goal.fault_id),
            corrective_command_forwarded: self
                .fault
                .as_ref()
                .is_some_and(|f| f.corrective_forwarded),
            warning_layers: issues()
                .filter(|issue| issue.level == Level::Warning)
                .cloned()
                .collect(),
            fault_layers: issues()
                .filter(|issue| issue.level == Level::Fault)
                .cloned()
                .collect(),
            received_commands: self.received,
            forwarded_commands: self.forwarded,
            blocked_commands: self.received - self.forwarded,
            updated_mono_ms: now,
        }
    }
}

pub fn goal_hash(goal: &str) -> u32 {
    goal.bytes()
        .fold(2166136261u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(16777619)
        })
        .max(1)
}
fn is_damping(command: &LowCommand) -> bool {
    let motors = hardware::NATIVE_INDICES.map(|index| command.motors[index]);
    motors.iter().all(|m| {
        m.mode == 1
            && m.kp == 0.0
            && m.dq == 0.0
            && m.tau == 0.0
            && m.kd.is_finite()
            && m.kd >= 0.0
            && m.kd <= 25.0
    }) && motors.iter().any(|m| m.kd > 0.0)
}
