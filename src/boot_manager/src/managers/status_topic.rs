//! Shared JSON status parsing and receipt timeout; package policy lives in its own manager.
use super::{ERROR, NORMAL, WARNING};
use std::{
    io,
    time::{Duration, Instant},
};

pub const MAX_AGE: Duration = Duration::from_secs(3);

pub(super) struct StatusTopic {
    topic: &'static str,
    latest: Option<(Instant, i32)>,
    problem: Option<String>,
}

impl StatusTopic {
    pub fn new(topic: &'static str) -> Self {
        Self {
            topic,
            latest: None,
            problem: None,
        }
    }

    pub fn receive(&mut self, payload: &str, policy: impl FnOnce(i32, &serde_json::Value) -> i32) {
        let code = serde_json::from_str::<serde_json::Value>(payload)
            .ok()
            .and_then(|value| {
                let code = i32::try_from(value["status_code"].as_i64()?).ok()?;
                if !matches!(code, NORMAL | WARNING | ERROR) {
                    return None;
                }
                Some(policy(code, &value))
            });
        self.latest = code.map(|code| (Instant::now(), code));
        self.problem = code
            .is_none()
            .then(|| format!("{}: invalid status JSON or code", self.topic));
    }

    pub fn check(&self) -> io::Result<i32> {
        self.check_at(Instant::now())
    }

    fn check_at(&self, now: Instant) -> io::Result<i32> {
        let (received, code) = self.latest.ok_or_else(|| {
            super::invalid_data(
                &self
                    .problem
                    .clone()
                    .unwrap_or_else(|| format!("{}: no status received", self.topic)),
            )
        })?;
        if now.saturating_duration_since(received) >= MAX_AGE {
            return Err(super::invalid_data(&format!(
                "{}: no status for 3 seconds",
                self.topic
            )));
        }
        Ok(code)
    }
}
