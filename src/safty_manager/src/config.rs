use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub recovery_hold_ms: u64,
    pub command_format: Switch,
    pub joint_range: JointRange,
    pub sudden_motion: SuddenMotion,
    pub speed: Speed,
    pub boot_status: Timeout,
    pub feedback: Timeout,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Switch {
    pub enable: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointRange {
    pub enable: bool,
    pub warning: RangeWarning,
    pub fault: RangeFault,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeWarning {
    pub enable: bool,
    pub margin_rad: f64,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeFault {
    pub enable: bool,
    pub tolerance_rad: f64,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuddenMotion {
    pub enable: bool,
    pub warning: JumpLimit,
    pub fault: JumpLimit,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JumpLimit {
    pub enable: bool,
    pub max_step_rad: f64,
    pub max_tracking_error_rad: f64,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Speed {
    pub enable: bool,
    pub sample_period_ms: u64,
    pub warning: SpeedLimit,
    pub fault: SpeedLimit,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedLimit {
    pub enable: bool,
    pub max_rad_s: f64,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timeout {
    pub enable: bool,
    pub startup_grace_ms: u64,
    pub warning: TimeoutLimit,
    pub fault: TimeoutLimit,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeoutLimit {
    pub enable: bool,
    pub timeout_ms: u64,
}

impl Config {
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Document {
            safty_manager: Node,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Node {
            #[serde(rename = "ros__parameters")]
            ros_parameters: Config,
        }
        let config = serde_yaml::from_str::<Document>(text)
            .map_err(|error| format!("invalid safety config: {error}"))?
            .safty_manager
            .ros_parameters;
        config.validate()?;
        Ok(config)
    }

    pub fn load() -> Result<(Self, PathBuf), String> {
        let path = if let Some(path) = std::env::var_os("AGROID_SAFTY_CONFIG") {
            PathBuf::from(path)
        } else {
            std::env::var_os("AMENT_PREFIX_PATH")
                .into_iter()
                .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
                .map(|prefix| prefix.join("share/safty_manager/config/default.yaml"))
                .find(|path| path.is_file())
                .unwrap_or_else(|| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/default.yaml")
                })
        };
        let text =
            fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok((Self::from_yaml(&text)?, path))
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = |name: &str, values: &[f64]| {
            if values.iter().all(|value| value.is_finite() && *value > 0.0) {
                Ok(())
            } else {
                Err(format!("{name}: limits must be positive and finite"))
            }
        };
        positive("joint_range", &[self.joint_range.warning.margin_rad])?;
        if !self.joint_range.fault.tolerance_rad.is_finite()
            || self.joint_range.fault.tolerance_rad < 0.0
        {
            return Err("joint_range: tolerance must be nonnegative and finite".into());
        }
        for limit in [&self.sudden_motion.warning, &self.sudden_motion.fault] {
            positive(
                "sudden_motion",
                &[limit.max_step_rad, limit.max_tracking_error_rad],
            )?;
        }
        positive(
            "speed",
            &[self.speed.warning.max_rad_s, self.speed.fault.max_rad_s],
        )?;
        if self.sudden_motion.warning.max_step_rad > self.sudden_motion.fault.max_step_rad
            || self.sudden_motion.warning.max_tracking_error_rad
                > self.sudden_motion.fault.max_tracking_error_rad
            || self.speed.warning.max_rad_s > self.speed.fault.max_rad_s
        {
            return Err("warning limits must not exceed fault limits".into());
        }
        if self.speed.sample_period_ms == 0 || self.recovery_hold_ms == 0 {
            return Err("sample_period_ms and recovery_hold_ms must be positive".into());
        }
        for timeout in [&self.boot_status, &self.feedback] {
            if timeout.warning.timeout_ms == 0
                || timeout.warning.timeout_ms > timeout.fault.timeout_ms
            {
                return Err("timeouts must be positive; warning <= fault".into());
            }
        }
        Ok(())
    }
}
