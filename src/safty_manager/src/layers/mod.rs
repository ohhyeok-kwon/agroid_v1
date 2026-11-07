pub mod boot_status;
pub mod command_format;
pub mod feedback;
pub mod joint_range;
pub mod speed;
pub mod sudden_motion;

use crate::{lowcmd::LowCommand, Accepted, BootSample, Feedback, Issue, Level};

pub struct Context<'a> {
    pub now: u64,
    pub started: u64,
    pub command: Option<&'a LowCommand>,
    pub previous_command: Option<&'a Accepted>,
    pub feedback: Option<&'a Feedback>,
    pub previous_feedback: Option<&'a Feedback>,
    pub boot: Option<&'a BootSample>,
    pub fallback_start: bool,
}

impl Context<'_> {
    pub fn baseline(&self, index: usize) -> Option<(f64, u64)> {
        if !self.fallback_start {
            if let Some(previous) = self.previous_command {
                let motor = previous.command.motors[crate::hardware::NATIVE_INDICES[index]];
                if motor.mode == 1 && motor.kp > 0.0 {
                    return Some((f64::from(motor.q), previous.received_ms));
                }
            }
        }
        self.feedback
            .map(|feedback| (feedback.position[index], feedback.received_ms))
    }
}

pub fn issue(
    layer: &'static str,
    level: Level,
    source: &'static str,
    index: Option<usize>,
    message: String,
    out: &mut Vec<Issue>,
) {
    out.push(Issue {
        layer,
        level,
        source,
        joint: index.map(|i| crate::profile::NAMES[i]),
        message,
    });
}
