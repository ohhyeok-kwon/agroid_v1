//! Overall humanoid state: joint feedback and URDF transforms (one /motor_status conversion),
//! health inputs and the sole action selector. No motor encoding or USB I/O.
//! Every time here is CLOCK_MONOTONIC milliseconds (`motion::mono_ms`).
pub mod ankle;
#[path = "../../humanoid_controller/src/checksum.rs"]
mod checksum;
#[path = "../../humanoid_controller/src/r1_profile.rs"]
#[allow(dead_code)] // The state manager uses the layout; device setup belongs to the controller.
pub mod hardware;
pub mod joints;
#[path = "../../humanoid_controller/src/lowcmd.rs"]
pub mod low_level;
pub mod model;
pub mod motion;
pub mod remote_keys;
mod safety;
use crate::motion::{fresh, validate_commands, Action, ActionPhase, ActionResult, Goal};
use serde::Deserialize;
use serde_json::{json, Value};

const BOOT_AGE_MS: u64 = 3000;
const BOOT_CHECKS: usize = 8;
const FEEDBACK_AGE_MS: u64 = 500;
#[derive(Deserialize)]
struct MotorHealth {
    status_code: u16,
    usb_connected: bool,
    #[serde(default)]
    epoch: Option<u64>,
    initialization: Option<Initialization>,
    #[serde(default)]
    execution: Option<Execution>,
}
#[derive(Deserialize)]
struct Execution {
    enabled: bool,
    #[serde(default)]
    fault: String,
}
#[derive(Deserialize)]
struct Initialization {
    phase: String,
}
#[derive(Deserialize)]
struct RemoteHealth {
    status_code: u16,
}
#[derive(Deserialize)]
struct WalkingHealth {
    ready: bool,
    moving: bool,
    message: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    action: Action,
}

pub struct Manager {
    session: String,
    generation: u64,
    boot: Option<(u64, Vec<i32>)>,
    motor: Option<(u64, MotorHealth)>,
    remote: Option<(u64, RemoteHealth)>,
    walking: Option<WalkingHealth>,
    careful_walk: Option<WalkingHealth>,
    safe_crouch: Option<WalkingHealth>,
    joints: Option<(u64, [f64; 26])>,
    goal: Option<Goal>,
    result: Option<(u64, u64, ActionPhase)>,
    fault: Option<String>,
    default_decided: bool,
    /// The current goal belongs to one controller run.
    goal_epoch: Option<u64>,
    last_request: Option<Value>,
    control_source: &'static str,
    safety: Option<safety::Override>,
    safety_code: u16,
}

impl Manager {
    pub fn new(session: String) -> Self {
        Self {
            session,
            generation: 0,
            boot: None,
            motor: None,
            remote: None,
            walking: None,
            careful_walk: None,
            safe_crouch: None,
            joints: None,
            goal: None,
            result: None,
            fault: None,
            default_decided: false,
            goal_epoch: None,
            last_request: None,
            control_source: "startup",
            safety: None,
            safety_code: 200,
        }
    }

    pub fn receive_boot(&mut self, codes: Vec<i32>, now: u64) {
        self.boot = Some((now, codes));
    }
    pub fn receive_motor(&mut self, text: &str, now: u64) {
        self.motor = serde_json::from_str(text).ok().map(|health| (now, health));
    }
    pub fn receive_remote(&mut self, text: &str, now: u64) {
        self.remote = serde_json::from_str(text).ok().map(|health| (now, health));
    }
    pub fn receive_walking(&mut self, text: &str) {
        self.walking = serde_json::from_str(text).ok();
    }
    pub fn receive_policy(&mut self, action: Action, text: &str) {
        match action {
            Action::CarefulWalk => self.careful_walk = serde_json::from_str(text).ok(),
            Action::SafeCrouch => self.safe_crouch = serde_json::from_str(text).ok(),
            _ => {}
        }
    }
    /// The native planner starts from the latest complete measured joint positions.
    pub fn receive_joints(&mut self, positions: &[Option<f64>; 26], measured: u64) {
        let mut q = [0.0; 26];
        let complete = positions.iter().enumerate().all(|(i, value)| {
            value.is_some_and(|value| {
                if !value.is_finite() {
                    return false;
                }
                q[i] = value;
                true
            })
        });
        self.joints = complete.then_some((measured, q));
    }

    /// What damping itself needs: a live USB link with fresh motor feedback. Damping is the
    /// emergency state (native `L2_hold+B_pressed` from any static state, hard e-stop).
    fn motor_link(&self, _now: u64) -> Result<(), String> {
        let (_, motor) = self.motor.as_ref().ok_or("waiting for motor feedback")?;
        if !motor.usb_connected {
            return Err("USB is disconnected".into());
        }
        Ok(())
    }

    fn motors_ready(&self, now: u64) -> Result<(), String> {
        self.motor_link(now)?;
        let (_, motor) = self.motor.as_ref().unwrap();
        if motor
            .initialization
            .as_ref()
            .is_none_or(|init| init.phase != "ready")
        {
            return Err("waiting for motor initialization".into());
        }
        Ok(())
    }

    fn epoch(&self) -> Option<u64> {
        self.motor.as_ref().and_then(|(_, motor)| motor.epoch)
    }

    pub fn controller_epoch(&self) -> Option<u64> {
        self.epoch()
    }

    fn complete_joints(&self) -> Result<[f64; 26], String> {
        self.joints
            .as_ref()
            .map(|(_, q)| *q)
            .ok_or_else(|| "complete joint feedback required".into())
    }

    fn activate(&mut self, action: Action, now: u64) -> Result<(), String> {
        let start_positions = if action == Action::LockedStanding {
            Some(self.complete_joints()?)
        } else {
            None
        };
        self.generation += 1;
        self.goal = Some(Goal {
            id: format!("{}:{}", self.session, self.generation),
            action,
            started_mono_ms: now,
            issued_mono_ms: now,
            start_positions,
        });
        self.result = None;
        self.fault = None;
        // Bind this goal to the current controller session.
        self.goal_epoch = self.epoch();
        Ok(())
    }

    pub fn request(&mut self, text: &str, now: u64) {
        let result = (|| -> Result<(), String> {
            let request: Request = serde_json::from_str(text).map_err(|error| error.to_string())?;
            self.safety_allows(request.action)?;
            if request.action == Action::Damping {
                if let Some(safety) = &mut self.safety {
                    safety.manual_damping = true;
                }
            }
            let current = self.selected_action();
            if current == Some(request.action) && self.fault.is_none() {
                return Ok(());
            }
            match request.action {
                Action::Walking => {
                    self.motors_ready(now)?;
                    if current != Some(Action::CarefulWalk)
                        && (current != Some(Action::LockedStanding)
                            || self
                                .result
                                .as_ref()
                                .is_none_or(|(_, _, phase)| *phase != ActionPhase::Holding))
                    {
                        return Err("walking starts from completed locked standing".into());
                    }
                    let walking = self
                        .walking
                        .as_ref()
                        .ok_or("waiting for walking_action_manager")?;
                    if !walking.ready {
                        return Err(walking.message.clone());
                    }
                }
                Action::CarefulWalk | Action::SafeCrouch => {
                    self.motors_ready(now)?;
                    if request.action == Action::CarefulWalk
                        && current != Some(Action::Walking)
                        && (current != Some(Action::LockedStanding)
                            || self
                                .result
                                .as_ref()
                                .is_none_or(|(_, _, phase)| *phase != ActionPhase::Holding))
                    {
                        return Err(
                            "careful walking starts from walking or completed standing".into()
                        );
                    }
                    let health = if request.action == Action::CarefulWalk {
                        &self.careful_walk
                    } else {
                        &self.safe_crouch
                    };
                    let health = health
                        .as_ref()
                        .ok_or("waiting for learned action manager")?;
                    if !health.ready {
                        return Err(health.message.clone());
                    }
                }
                Action::Damping => self.motor_link(now)?,
                Action::ZeroMoment => {
                    self.motor_link(now)?;
                }
                Action::LockedStanding => {
                    self.motors_ready(now)?;
                    if (current == Some(Action::Walking)
                        && self.walking.as_ref().is_some_and(|w| w.moving))
                        || (current == Some(Action::CarefulWalk)
                            && self.careful_walk.as_ref().is_some_and(|w| w.moving))
                    {
                        return Err(
                            "native walking exit: release the joystick before standing".into()
                        );
                    }
                }
            }
            self.activate(request.action, now)?;
            if request.action == Action::Damping {
                if let Some(safety) = &mut self.safety {
                    safety.manual_damping = true;
                }
            }
            Ok(())
        })();
        self.last_request = Some(match result {
            Ok(()) => {
                self.control_source = "request_topic";
                json!({"accepted": true, "message": "action selected"})
            }
            Err(error) => json!({"accepted": false, "message": error}),
        });
    }

    pub fn request_remote(&mut self, action: Action, now: u64) {
        self.request(&json!({"action": action}).to_string(), now);
        if let Some(request) = self.last_request.as_mut() {
            request["source"] = json!("remote");
        }
        if self
            .last_request
            .as_ref()
            .is_some_and(|value| value["accepted"] == true)
        {
            self.control_source = "remote";
        }
    }

    pub fn selected_action(&self) -> Option<Action> {
        self.goal.as_ref().map(|goal| goal.action)
    }

    /// Each native FSM instance starts at ZeroTorque after initialization.
    pub fn tick(&mut self, now: u64) {
        if self.goal.is_some() && self.epoch() != self.goal_epoch {
            self.goal = None;
            self.result = None;
            self.default_decided = false;
            if let Some(safety) = &mut self.safety {
                safety.goal_id = None;
            }
        }
        if !self.default_decided && self.motors_ready(now).is_ok() {
            self.default_decided = true;
            let _ = self.activate(Action::ZeroMoment, now);
            self.control_source = "startup";
        }
        // Match the native hard emergency transition without a separate rearm state.
        if self
            .motor
            .as_ref()
            .and_then(|(_, m)| m.execution.as_ref())
            .is_some_and(|e| e.fault == "emergency")
        {
            if self.selected_action() != Some(Action::Damping) {
                let _ = self.activate(Action::Damping, now);
            }
            self.control_source = "remote";
            if let Some(safety) = &mut self.safety {
                safety.manual_damping = true;
            }
        }
        self.apply_safety(now);
        if self.safety.is_none()
            && matches!(
                self.selected_action(),
                Some(Action::Walking | Action::CarefulWalk | Action::SafeCrouch)
            )
            && self
                .motor
                .as_ref()
                .and_then(|(_, m)| m.execution.as_ref())
                .is_some_and(|e| e.fault == "command_expired")
        {
            self.action_failed("walking command timeout: damping".into(), now);
        }
    }

    fn action_failed(&mut self, reason: String, now: u64) {
        if matches!(
            self.selected_action(),
            Some(Action::Walking | Action::CarefulWalk | Action::SafeCrouch)
        ) {
            let _ = self.activate(Action::Damping, now);
            self.control_source = "walking_failure";
            self.last_request =
                Some(json!({"accepted":true,"source":"walking_failure","message":reason}));
        } else {
            self.fault = Some(reason);
        }
    }

    pub fn goal(&self, now: u64) -> Option<Goal> {
        self.goal.clone().map(|mut goal| {
            goal.issued_mono_ms = now;
            goal
        })
    }

    pub fn receive_result(&mut self, text: &str, now: u64) -> Option<ActionResult> {
        self.tick(now);
        let result: ActionResult = serde_json::from_str(text).ok()?;
        let goal = self.goal.as_ref()?;
        // Results from old states, old processes or another action never reach the output.
        if result.goal_id != goal.id
            || result.action != goal.action
            || result.generated_mono_ms < goal.started_mono_ms
            || self
                .result
                .as_ref()
                .is_some_and(|(_, sequence, _)| result.sequence <= *sequence)
        {
            return None;
        }
        if result.status_code != 200 || result.phase == ActionPhase::Unavailable {
            self.action_failed(format!("action failed: {}", result.message), now);
            return None;
        }
        if let Err(error) = validate_commands(&result.joints) {
            self.action_failed(error, now);
            return None;
        }
        if result.action != Action::LockedStanding && result.phase != ActionPhase::Holding {
            self.action_failed("invalid action phase".into(), now);
            return None;
        }
        self.fault = None;
        self.result = Some((
            result.generated_mono_ms,
            result.sequence,
            result.phase.clone(),
        ));
        Some(result)
    }

    pub fn status(&self, now: u64) -> Value {
        let boot_code = self
            .boot
            .as_ref()
            .filter(|(t, codes)| fresh(*t, now, BOOT_AGE_MS) && codes.len() == BOOT_CHECKS)
            .and_then(|(_, codes)| codes.iter().max().copied())
            .filter(|c| matches!(c, 200..=202))
            .unwrap_or(202);
        let motor_code = self
            .motor
            .as_ref()
            .filter(|(t, _)| fresh(*t, now, FEEDBACK_AGE_MS))
            .map_or(202, |(_, m)| m.status_code);
        let remote_code = self
            .remote
            .as_ref()
            .filter(|(t, _)| fresh(*t, now, 1000))
            .map_or(202, |(_, r)| r.status_code);
        let phase = self.result.as_ref().map(|(_, _, phase)| phase);
        let state = self
            .goal
            .as_ref()
            .map_or(json!("init"), |goal| json!(goal.action));
        let code = if self.fault.is_some() || self.safety.is_some() {
            202
        } else if self.goal.is_none() || phase != Some(&ActionPhase::Holding) {
            201
        } else {
            boot_code
                .max(i32::from(motor_code))
                .max(i32::from(self.safety_code))
                .max(if remote_code == 200 { 200 } else { 201 })
        };
        json!({
            "status_code": code, "state": state, "phase": phase,
            "fsm_id": self.goal.as_ref().map(|g| if phase == Some(&ActionPhase::Transitioning) { 6 } else { g.action.fsm_id() }),
            "goal_id": self.goal.as_ref().map(|g| &g.id), "updated_mono_ms": now,
            "fault": self.fault,
            "message": self.fault.clone()
                .or_else(|| self.safety.as_ref().map(|safety| safety.reason.clone()))
                .or_else(|| self.motors_ready(now).err())
                .unwrap_or_else(|| if self.motor.as_ref().and_then(|(_,m)|m.execution.as_ref()).is_some_and(|e|e.enabled) {
                    "USB motion enabled".into()
                } else {
                    "joint-space command generation only".into()
                }),
            "execution": if self.motor.as_ref().and_then(|(_,m)|m.execution.as_ref()).is_some_and(|e|e.enabled) {"usb"} else {"preview"},
            "usb_motion_connected": self.motor.as_ref().and_then(|(_,m)|m.execution.as_ref()).is_some_and(|e|e.enabled),
            "default_action": Action::ZeroMoment, "control_source": self.control_source,
            "controller_epoch": self.motor.as_ref().and_then(|(_, m)| m.epoch),
            "sources": {"boot": boot_code, "motor": motor_code, "remote": remote_code, "safety": self.safety_code},
            "safety": self.safety.as_ref().map(|safety| json!({
                "active": true, "fault_id": safety.id,
                "requested_action": "locked_standing", "goal_id": safety.goal_id,
                "manual_damping": safety.manual_damping,
            })),
            "last_request": self.last_request,
            "actions": {"zero_moment": true, "damping": true, "locked_standing": true,
                "walking": self.walking.as_ref().is_some_and(|w|w.ready),
                "careful_walk":self.careful_walk.as_ref().is_some_and(|w|w.ready),
                "safe_crouch":self.safe_crouch.as_ref().is_some_and(|w|w.ready)}
        })
    }
}
