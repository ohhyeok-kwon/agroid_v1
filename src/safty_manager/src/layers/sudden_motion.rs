use super::{issue, Context};
use crate::{config::SuddenMotion, hardware::NATIVE_INDICES, Issue, Level};

pub fn inspect(config: &SuddenMotion, context: &Context<'_>, out: &mut Vec<Issue>) {
    if !config.enable {
        return;
    }
    for (i, slot) in NATIVE_INDICES.into_iter().enumerate() {
        if let Some(command) = context.command {
            let motor = command.motors[slot];
            if motor.mode == 1 && motor.kp > 0.0 {
                let q = f64::from(motor.q);
                if let Some((previous, _)) = context.baseline(i) {
                    check(config, (q - previous).abs(), false, "command", i, out);
                }
                if let Some(feedback) = context.feedback {
                    check(
                        config,
                        (q - feedback.position[i]).abs(),
                        true,
                        "command",
                        i,
                        out,
                    );
                }
            }
        }
        if let (Some(current), Some(previous)) = (context.feedback, context.previous_feedback) {
            check(
                config,
                (current.position[i] - previous.position[i]).abs(),
                false,
                "feedback",
                i,
                out,
            );
        }
    }
}

fn check(
    config: &SuddenMotion,
    delta: f64,
    tracking: bool,
    source: &'static str,
    index: usize,
    out: &mut Vec<Issue>,
) {
    let limit = |fault: bool| {
        let c = if fault {
            &config.fault
        } else {
            &config.warning
        };
        if tracking {
            c.max_tracking_error_rad
        } else {
            c.max_step_rad
        }
    };
    let level = if config.fault.enable && (!delta.is_finite() || delta > limit(true)) {
        Some(Level::Fault)
    } else if config.warning.enable && (!delta.is_finite() || delta > limit(false)) {
        Some(Level::Warning)
    } else {
        None
    };
    if let Some(level) = level {
        let quantity = if tracking {
            "target error"
        } else {
            "position step"
        };
        issue(
            "sudden_motion",
            level,
            source,
            Some(index),
            format!("{quantity} {delta:.4} rad"),
            out,
        );
    }
}
