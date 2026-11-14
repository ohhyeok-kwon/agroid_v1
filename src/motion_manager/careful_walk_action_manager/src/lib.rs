use walking_action_manager::{
    motion::{Action, Output},
    observation::{Input, Observation},
    onnx::Session,
    policy_output, Parameters,
};

/// Legacy stick mapping (no "command" block): per-tick slew of every axis.
const LEGACY_SLEW: f32 = 0.03;

/// Command processing of a contract-v2 student ("careful-walk-command-student-v2-lateral",
/// r1lab STUDENT_CONTRACT_LATERAL), read from the optional "command" block of careful_walk.json:
/// stick limits, per-tick slew and the heading hold the student was trained and evaluated with.
/// Without the block the bundled v1 policy keeps the legacy mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct CommandContract {
    pub forward: f32,
    pub reverse: f32,
    pub lateral: f32,
    pub yaw: f32,
    pub slew: [f32; 3],
    pub heading_hold: bool,
    pub heading_gain: f32,
    pub heading_rate_limit: f32,
    pub heading_deadband: f32,
    pub hold_settle_s: f32,
    pub stick_deadzone: f32,
}
impl CommandContract {
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let number = |key: &str| -> Result<f32, String> {
            let v = value
                .get(key)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("careful_walk.json command.{key} missing"))?;
            if !v.is_finite() || v < 0. {
                return Err(format!("careful_walk.json command.{key} must be a non-negative number"));
            }
            Ok(v as f32)
        };
        let slew: Vec<f32> = value
            .get("slew")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect())
            .unwrap_or_default();
        if slew.len() != 3 || slew.iter().any(|v| !(v.is_finite() && *v > 0.)) {
            return Err("careful_walk.json command.slew must hold three positive numbers".into());
        }
        Ok(Self {
            forward: number("forward")?,
            reverse: number("reverse")?,
            lateral: number("lateral")?,
            yaw: number("yaw")?,
            slew: [slew[0], slew[1], slew[2]],
            heading_hold: value
                .get("heading_hold")
                .and_then(|v| v.as_bool())
                .ok_or("careful_walk.json command.heading_hold missing")?,
            heading_gain: number("heading_gain")?,
            heading_rate_limit: number("heading_rate_limit")?,
            heading_deadband: number("heading_deadband")?,
            hold_settle_s: number("hold_settle_s")?,
            stick_deadzone: number("stick_deadzone")?,
        })
    }
}

/// Wrap an angle to [-pi, pi).
fn wrap(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(2. * std::f32::consts::PI) - std::f32::consts::PI
}

/// Pelvis yaw from the IMU quaternion (w, x, y, z).
pub fn yaw(quaternion: [f32; 4]) -> f32 {
    let [w, x, y, z] = quaternion;
    (2. * (w * z + x * y)).atan2(1. - 2. * (y * y + z * z))
}

/// Heading hold of r1lab StudentMatchingCommand: armed while the user does not steer; the target
/// follows the yaw for hold_settle_s (residual rotation), then stays fixed. Steering disarms it.
#[derive(Clone, Debug)]
pub struct HeadingHold {
    target: f32,
    age: f32,
}
impl Default for HeadingHold {
    fn default() -> Self {
        Self { target: 0., age: -1. }
    }
}
impl HeadingHold {
    pub fn reset(&mut self) {
        self.age = -1.;
    }
    /// One policy tick. None while the user steers, otherwise the heading-error yaw rate.
    pub fn rate(&mut self, c: &CommandContract, yaw: f32, steering: bool, dt: f32) -> Option<f32> {
        if steering {
            self.age = -1.;
            return None;
        }
        self.age = if self.age < 0. { 0. } else { self.age + dt };
        if self.age < c.hold_settle_s {
            self.target = yaw;
        }
        let error = wrap(self.target - yaw);
        Some(if error.abs() < c.heading_deadband {
            0.
        } else {
            (c.heading_gain * error).clamp(-c.heading_rate_limit, c.heading_rate_limit)
        })
    }
}

/// Stick axes -> target velocity and per-tick slew (legacy mapping or contract v2 with heading hold).
#[derive(Clone, Debug, Default)]
pub struct Commander {
    pub contract: Option<CommandContract>,
    hold: HeadingHold,
}
impl Commander {
    pub fn new(contract: Option<CommandContract>) -> Self {
        Self { contract, hold: HeadingHold::default() }
    }
    pub fn reset(&mut self) {
        self.hold.reset();
    }
    pub fn target(&mut self, p: &Parameters, input: &Input, axes: [f32; 4]) -> ([f32; 3], [f32; 3]) {
        let Some(c) = &self.contract else {
            let target = [
                axes[1]
                    * if axes[1] >= 0. {
                        p.low_speed_forward_vx_limit
                    } else {
                        p.reverse_vx_limit
                    },
                -axes[0] * p.vy_limit,
                -axes[2] * p.pure_wz_limit,
            ];
            return (target, [LEGACY_SLEW; 3]);
        };
        let steering = axes[2].abs() > c.stick_deadzone;
        let held = if c.heading_hold {
            self.hold.rate(c, yaw(input.quaternion), steering, p.dt)
        } else {
            None
        };
        let wz = held.unwrap_or(if steering { -axes[2] * c.yaw } else { 0. });
        let vx = axes[1] * if axes[1] >= 0. { c.forward } else { c.reverse };
        ([vx, -axes[0] * c.lateral, wz], c.slew)
    }
    /// One tick of the slewed command (the policy input).
    pub fn step(&mut self, p: &Parameters, input: &Input, axes: [f32; 4], velocity: &mut [f32; 3]) {
        let (target, slew) = self.target(p, input, axes);
        for i in 0..3 {
            velocity[i] += (target[i] - velocity[i]).clamp(-slew[i], slew[i]);
        }
    }
}

/// The optional contract-v2 "command" block of a careful_walk.json text.
pub fn command_contract(text: &str) -> Result<Option<CommandContract>, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    value.get("command").map(CommandContract::from_json).transpose()
}

pub struct CarefulWalk {
    session: Session,
    parameters: Parameters,
    commander: Commander,
    observations: Observation,
    velocity: [f32; 3],
}
impl CarefulWalk {
    pub fn load() -> Result<Self, String> {
        let text = include_str!("../config/careful_walk.json");
        Ok(Self {
            session: Session::load(include_bytes!("../config/careful_walk.onnx"))?,
            parameters: serde_json::from_str(text).map_err(|e| e.to_string())?,
            commander: Commander::new(command_contract(text)?),
            observations: Observation::default(),
            velocity: [0.; 3],
        })
    }
    pub fn start(&mut self, input: &Input) -> Result<(), String> {
        input.validate()?;
        self.velocity = [0.; 3];
        self.commander.reset();
        self.observations
            .reset(input, self.velocity, &self.parameters);
        Ok(())
    }
    pub fn sample(&mut self, input: &Input, axes: [f32; 4]) -> Result<Output, String> {
        input.validate()?;
        let p = &self.parameters;
        self.commander.step(p, input, axes, &mut self.velocity);
        self.observations.update(input, self.velocity, p);
        let actions = self.session.infer(&mut self.observations.history)?;
        self.observations.previous_action = actions;
        Ok(policy_output(p, [0.; 2], actions))
    }
}
#[cfg(feature = "ros")]
impl walking_action_manager::policy_runtime::Policy for CarefulWalk {
    const ACTION: Action = Action::CarefulWalk;
    const NODE: &'static str = "careful_walk_action_manager";
    const STATUS: &'static str = "/careful_walk_status";
    fn start(&mut self, input: &Input) -> Result<(), String> {
        CarefulWalk::start(self, input)
    }
    fn sample(&mut self, input: &Input, axes: [f32; 4], _: f64) -> Result<Output, String> {
        CarefulWalk::sample(self, input, axes)
    }
    fn moving(&self) -> bool {
        self.velocity.iter().any(|v| v.abs() > 0.02)
    }
}
