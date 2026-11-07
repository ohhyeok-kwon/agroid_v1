use super::{issue, Context};
use crate::{config::Timeout, Issue, Level};

pub fn inspect(config: &Timeout, context: &Context<'_>, out: &mut Vec<Issue>) {
    if !config.enable || context.now.saturating_sub(context.started) < config.startup_grace_ms {
        return;
    }
    let Some(boot) = context.boot else {
        missing(config, "boot status not received".into(), out);
        return;
    };
    let age = context.now.saturating_sub(boot.received_ms);
    if age > config.fault.timeout_ms && config.fault.enable {
        issue(
            "boot_status",
            Level::Fault,
            "boot",
            None,
            format!("boot status stale: {age} ms"),
            out,
        );
        return;
    }
    if age > config.warning.timeout_ms && config.warning.enable {
        issue(
            "boot_status",
            Level::Warning,
            "boot",
            None,
            format!("boot status aging: {age} ms"),
            out,
        );
    }
    if boot.codes.is_empty()
        || boot.codes.len() != boot.names.len()
        || boot.names.iter().any(|name| name.is_empty())
        || boot.codes.iter().any(|code| !matches!(code, 200..=202))
    {
        missing(config, "invalid boot status codes".into(), out);
        return;
    }
    for (index, code) in boot.codes.iter().enumerate() {
        let name = boot
            .names
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("manager[{index}]"));
        let level = if *code == 202 && config.fault.enable {
            Some(Level::Fault)
        } else if *code >= 201 && config.warning.enable {
            Some(Level::Warning)
        } else {
            None
        };
        if let Some(level) = level {
            issue(
                "boot_status",
                level,
                "boot",
                None,
                format!("{name}: {code}"),
                out,
            );
        }
    }
}

fn missing(config: &Timeout, message: String, out: &mut Vec<Issue>) {
    if config.fault.enable {
        issue("boot_status", Level::Fault, "boot", None, message, out);
    } else if config.warning.enable {
        issue("boot_status", Level::Warning, "boot", None, message, out);
    }
}
