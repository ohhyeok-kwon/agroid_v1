use super::{issue, Context};
use crate::{config::JointRange, hardware::NATIVE_INDICES, profile::LIMITS, Issue, Level};

pub fn inspect(config: &JointRange, context: &Context<'_>, out: &mut Vec<Issue>) {
    if !config.enable {
        return;
    }
    for (i, (low, high)) in LIMITS.into_iter().enumerate() {
        let command = context.command.and_then(|command| {
            let motor = command.motors[NATIVE_INDICES[i]];
            (motor.mode == 1 && motor.kp > 0.0).then_some(f64::from(motor.q))
        });
        let feedback = context.feedback.map(|feedback| feedback.position[i]);
        for (source, position) in [("command", command), ("feedback", feedback)] {
            let Some(q) = position else {
                continue;
            };
            let level = if config.fault.enable
                && (!q.is_finite()
                    || q < low - config.fault.tolerance_rad
                    || q > high + config.fault.tolerance_rad)
            {
                Some(Level::Fault)
            } else if config.warning.enable
                && (!q.is_finite()
                    || q < low + config.warning.margin_rad
                    || q > high - config.warning.margin_rad)
            {
                Some(Level::Warning)
            } else {
                None
            };
            if let Some(level) = level {
                issue(
                    "joint_range",
                    level,
                    source,
                    Some(i),
                    format!("position {q:.4} rad, range [{low:.4}, {high:.4}]"),
                    out,
                );
            }
        }
    }
}
