//! Safety selects a state; the action manager still owns the standing trajectory.
use crate::{
    motion::{fresh, Action},
    Manager, FEEDBACK_AGE_MS,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Signal {
    status_code: u16,
    active: bool,
    fault_id: Option<String>,
    requested_action: String,
    message: String,
}

pub(super) struct Override {
    pub id: String,
    pub reason: String,
    pub goal_id: Option<String>,
    pub manual_damping: bool,
}

impl Manager {
    pub fn receive_safety(&mut self, text: &str, now: u64) {
        let Ok(signal) = serde_json::from_str::<Signal>(text) else {
            return;
        };
        if !matches!(signal.status_code, 200..=202) || signal.requested_action != "locked_standing"
        {
            return;
        }
        if !signal.active && signal.status_code != 202 {
            self.safety = None;
            self.safety_code = signal.status_code;
            return;
        }
        if signal.status_code != 202 || !signal.active {
            return;
        }
        let Some(id) = signal
            .fault_id
            .filter(|id| !id.is_empty() && id.len() <= 256)
        else {
            return;
        };
        self.safety_code = 202;
        if self.safety.as_ref().is_none_or(|safety| safety.id != id) {
            self.safety = Some(Override {
                id,
                reason: signal.message,
                goal_id: None,
                manual_damping: self.selected_action() == Some(Action::Damping)
                    && matches!(self.control_source, "remote" | "request_topic"),
            });
        } else if let Some(safety) = &mut self.safety {
            safety.reason = signal.message;
        }
        self.apply_safety(now);
    }

    pub(super) fn apply_safety(&mut self, now: u64) {
        if self
            .motor
            .as_ref()
            .and_then(|(_, motor)| motor.execution.as_ref())
            .is_some_and(|execution| execution.fault == "emergency")
        {
            return;
        }
        let Some(safety) = &self.safety else {
            return;
        };
        if safety.manual_damping {
            return;
        }
        if self.selected_action() == Some(Action::LockedStanding)
            && safety.goal_id.as_deref() == self.goal.as_ref().map(|goal| goal.id.as_str())
        {
            return;
        }
        // Never build a new stiff posture from missing or old joint measurements.
        if self.motors_ready(now).is_err()
            || self
                .motor
                .as_ref()
                .is_none_or(|(at, _)| !fresh(*at, now, FEEDBACK_AGE_MS))
            || self
                .joints
                .as_ref()
                .is_none_or(|(at, _)| !fresh(*at, now, FEEDBACK_AGE_MS))
        {
            return;
        }
        // Safety does not wait for the normal joystick-neutral walking exit.
        if self.activate(Action::LockedStanding, now).is_ok() {
            self.default_decided = true;
            self.control_source = "safty_manager";
            if let Some(safety) = &mut self.safety {
                safety.goal_id = self.goal.as_ref().map(|goal| goal.id.clone());
            }
            self.last_request = Some(json!({
                "accepted": true, "source": "safty_manager", "message": "locked standing requested",
            }));
        }
    }

    pub(super) fn safety_allows(&self, action: Action) -> Result<(), String> {
        if self.safety.is_some() && !matches!(action, Action::Damping | Action::LockedStanding) {
            Err(
                "safety fault active: locked standing is selected; damping remains available"
                    .into(),
            )
        } else {
            Ok(())
        }
    }
}
