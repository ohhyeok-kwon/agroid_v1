use super::{issue, Context};
use crate::{config::Speed, hardware::NATIVE_INDICES, Issue, Level};

pub fn inspect(config: &Speed, context: &Context<'_>, out: &mut Vec<Issue>) {
    if !config.enable {
        return;
    }
    for (i, slot) in NATIVE_INDICES.into_iter().enumerate() {
        if let Some(command) = context.command {
            let motor = command.motors[slot];
            if motor.mode == 1 {
                check(
                    config,
                    f64::from(motor.dq).abs(),
                    "command",
                    "target velocity",
                    i,
                    out,
                );
                if motor.kp > 0.0 {
                    if let Some((previous, at)) = context.baseline(i) {
                        // DDS can batch delivery. Do not turn a sub-period callback gap into
                        // an artificial infinite target velocity.
                        let elapsed = context.now.saturating_sub(at).max(config.sample_period_ms);
                        let rate = (f64::from(motor.q) - previous).abs() * 1000.0 / elapsed as f64;
                        check(config, rate, "command", "target position rate", i, out);
                    }
                }
            }
        }
        if let Some(feedback) = context.feedback {
            check(
                config,
                feedback.velocity[i].abs(),
                "feedback",
                "measured velocity",
                i,
                out,
            );
        }
    }
}

fn check(
    config: &Speed,
    speed: f64,
    source: &'static str,
    quantity: &str,
    index: usize,
    out: &mut Vec<Issue>,
) {
    let level = if config.fault.enable && (!speed.is_finite() || speed > config.fault.max_rad_s) {
        Some(Level::Fault)
    } else if config.warning.enable && (!speed.is_finite() || speed > config.warning.max_rad_s) {
        Some(Level::Warning)
    } else {
        None
    };
    if let Some(level) = level {
        issue(
            "speed",
            level,
            source,
            Some(index),
            format!("{quantity} {speed:.3} rad/s"),
            out,
        );
    }
}
