use std::{fs, io};

const MAX_USAGE_PERCENT: f64 = 95.0;

pub fn check() -> io::Result<bool> {
    healthy(&fs::read_to_string("/proc/meminfo")?)
}

fn healthy(text: &str) -> io::Result<bool> {
    let field = |name: &str| -> io::Result<u64> {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| super::invalid_data("missing or invalid memory counters"))
    };
    let total = field("MemTotal:")?;
    let available = field("MemAvailable:")?;
    if total == 0 || available > total {
        return Err(super::invalid_data("invalid memory counters"));
    }
    Ok(100.0 * (total - available) as f64 / (total as f64) < MAX_USAGE_PERCENT)
}
