use super::{issue, Context};
use crate::{config::Timeout, Issue, Level};

pub fn inspect(config: &Timeout, context: &Context<'_>, out: &mut Vec<Issue>) {
    if !config.enable || context.now.saturating_sub(context.started) < config.startup_grace_ms {
        return;
    }
    let age = context
        .feedback
        .map(|feedback| context.now.saturating_sub(feedback.received_ms));
    let level = if config.fault.enable && age.is_none_or(|age| age > config.fault.timeout_ms) {
        Some(Level::Fault)
    } else if config.warning.enable && age.is_none_or(|age| age > config.warning.timeout_ms) {
        Some(Level::Warning)
    } else {
        None
    };
    if let Some(level) = level {
        issue(
            "feedback",
            level,
            "feedback",
            None,
            age.map_or_else(
                || "joint feedback not received".into(),
                |age| format!("joint feedback age: {age} ms"),
            ),
            out,
        );
    }
}
