use std::{fs, io};

const MAX_USAGE_PERCENT: f64 = 90.0;

#[derive(Clone, Copy)]
struct Sample {
    total: u64,
    idle: u64,
}

impl Sample {
    fn read() -> io::Result<Self> {
        Self::parse(&fs::read_to_string("/proc/stat")?)
    }

    fn parse(text: &str) -> io::Result<Self> {
        let line = text.lines().next().unwrap_or_default();
        let mut fields = line.split_whitespace();
        if fields.next() != Some("cpu") {
            return Err(super::invalid_data("missing aggregate CPU counters"));
        }
        // guest/guest_nice are already included in user/nice: do not count twice.
        let values = fields
            .take(8)
            .map(str::parse::<u64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| super::invalid_data("invalid CPU counters"))?;
        if values.len() != 8 {
            return Err(super::invalid_data("incomplete CPU counters"));
        }
        Ok(Self {
            total: values.iter().sum(),
            idle: values[3] + values[4],
        })
    }

    fn healthy_since(self, before: Self) -> bool {
        match (
            self.total.checked_sub(before.total),
            self.idle.checked_sub(before.idle),
        ) {
            (Some(total), Some(idle)) if total > 0 && idle <= total => {
                100.0 * (total - idle) as f64 / (total as f64) < MAX_USAGE_PERCENT
            }
            _ => false,
        }
    }
}

pub struct CpuManager {
    previous: Option<Sample>,
}

impl CpuManager {
    pub fn new() -> Self {
        Self {
            previous: Sample::read().ok(),
        }
    }

    pub fn check(&mut self) -> io::Result<bool> {
        let current = match Sample::read() {
            Ok(sample) => sample,
            Err(error) => {
                self.previous = None;
                return Err(error);
            }
        };
        Ok(self
            .previous
            .replace(current)
            .is_some_and(|before| current.healthy_since(before)))
    }
}
