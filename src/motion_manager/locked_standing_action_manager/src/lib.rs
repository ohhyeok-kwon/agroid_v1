//! Native Stance (FSM 4) entered through MoveTo (FSM 6), never by a position jump.
// One contract copy, owned by state_manager.
#[path = "../../../state_manager/src/motion/mod.rs"]
pub mod motion;
use crate::motion::{commands, Action, ActionPhase, Generator, Goal, Output};
use serde::Deserialize;

#[derive(Deserialize)]
struct Settings {
    position_rad: [f64; 26],
    kp: [f64; 26],
    kd: [f64; 26],
    joint_velocity_limit_rad_s: f64,
}

pub struct LockedStanding {
    settings: Settings,
    start: Option<[f64; 26]>,
    duration_s: f64,
}

impl LockedStanding {
    pub fn load() -> Result<Self, String> {
        let settings: Settings = serde_json::from_str(include_str!("../config/stance.json"))
            .map_err(|error| error.to_string())?;
        crate::motion::validate_commands(&commands(
            settings.position_rad,
            settings.kp,
            settings.kd,
        ))?;
        if !settings.joint_velocity_limit_rad_s.is_finite()
            || settings.joint_velocity_limit_rad_s <= 0.0
        {
            return Err("stance velocity limit must be positive and finite".into());
        }
        Ok(Self {
            settings,
            start: None,
            duration_s: 0.0,
        })
    }
}

impl Generator for LockedStanding {
    const ACTION: Action = Action::LockedStanding;
    const NODE: &'static str = "locked_standing_action_manager";

    fn start(&mut self, goal: &Goal) -> Result<(), String> {
        // A cancelled MoveTo cannot leave a flag that skips interpolation next time.
        self.start = None;
        self.duration_s = 0.0;
        let start = goal
            .start_positions
            .ok_or("stance requires complete joint feedback")?;
        if start.iter().any(|q| !q.is_finite()) {
            return Err("invalid stance start position".into());
        }
        self.duration_s = start
            .iter()
            .zip(self.settings.position_rad)
            .map(|(a, b)| (b - a).abs() / self.settings.joint_velocity_limit_rad_s)
            .fold(0.0, f64::max);
        self.start = Some(start);
        Ok(())
    }

    fn sample(&self, elapsed_s: f64) -> Result<Output, String> {
        let start = self.start.ok_or("stance has no active start position")?;
        if !elapsed_s.is_finite() || elapsed_s < 0.0 {
            return Err("invalid action elapsed time".into());
        }
        let alpha = if self.duration_s == 0.0 {
            1.0
        } else {
            (elapsed_s / self.duration_s).min(1.0)
        };
        let q = std::array::from_fn(|i| {
            start[i] * (1.0 - alpha) + self.settings.position_rad[i] * alpha
        });
        Ok(Output {
            phase: if alpha < 1.0 {
                ActionPhase::Transitioning
            } else {
                ActionPhase::Holding
            },
            joints: commands(q, self.settings.kp, self.settings.kd),
        })
    }
}
