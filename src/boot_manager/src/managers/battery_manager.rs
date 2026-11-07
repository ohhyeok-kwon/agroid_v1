//! Watches humanoid_controller's bms_status; owns no hardware connection.
use super::{ERROR, NORMAL, WARNING};
use serde_json::Value;
use std::{
    io,
    time::{Duration, Instant},
};

pub const MAX_AGE: Duration = Duration::from_secs(3);

#[derive(Default)]
pub struct BatteryManager {
    latest: Option<(Instant, i32)>,
    problem: Option<String>,
}

fn parse_status(payload: &str) -> io::Result<(i32, Duration)> {
    let value: Value = serde_json::from_str(payload)
        .map_err(|_| super::invalid_data("invalid bms_status JSON"))?;
    let code = match value["status_code"].as_i64() {
        Some(200) => NORMAL,
        Some(201) => WARNING,
        Some(202) => ERROR,
        _ => return Err(super::invalid_data("invalid BMS status code")),
    };
    let mut sample_age = Duration::ZERO;
    if code != ERROR {
        let valid_sample = value["usb_connected"].as_bool() == Some(true)
            && value["age_ms"]
                .as_u64()
                .is_some_and(|age| age <= MAX_AGE.as_millis() as u64)
            && value["data"]["soc"].as_u64().is_some_and(|soc| soc <= 100);
        if !valid_sample {
            return Err(super::invalid_data("BMS status has no fresh valid sample"));
        }
        sample_age = Duration::from_millis(value["age_ms"].as_u64().unwrap());
    }
    Ok((code, sample_age))
}

impl BatteryManager {
    pub fn receive(&mut self, payload: &str) {
        match parse_status(payload) {
            Ok((code, age)) => {
                self.latest = Some((Instant::now() - age, code));
                self.problem = None;
            }
            Err(error) => {
                self.latest = None;
                self.problem = Some(error.to_string());
            }
        }
    }

    pub fn check(&self) -> io::Result<i32> {
        let (received, code) = self.latest.ok_or_else(|| {
            super::invalid_data(
                self.problem
                    .as_deref()
                    .unwrap_or("no bms_status message received"),
            )
        })?;
        if received.elapsed() > MAX_AGE {
            return Err(super::invalid_data("bms_status is older than 3 seconds"));
        }
        Ok(code)
    }
}
