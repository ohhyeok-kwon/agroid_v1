use super::{issue, Context};
use crate::{config::Switch, hardware::NATIVE_INDICES, Issue, Level};

pub fn inspect(config: &Switch, context: &Context<'_>, out: &mut Vec<Issue>) {
    if !config.enable {
        return;
    }
    let Some(command) = context.command else {
        return;
    };
    let result = command.validate().and_then(|()| {
        if command.motors.iter().enumerate().any(|(i, m)| {
            !NATIVE_INDICES.contains(&i)
                && (m.mode != 0
                    || m.q != 0.0
                    || m.dq != 0.0
                    || m.tau != 0.0
                    || m.kp != 0.0
                    || m.kd != 0.0)
        }) {
            Err("command targets an unconfigured motor slot")
        } else {
            Ok(())
        }
    });
    if let Err(error) = result {
        issue(
            "command_format",
            Level::Fault,
            "command",
            None,
            error.into(),
            out,
        );
    }
}
