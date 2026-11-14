use serde::Deserialize;
use walking_action_manager::{
    motion::{Action, Output},
    observation::{Input, Observation},
    onnx::Session,
    policy_output, Parameters,
};

#[derive(Deserialize)]
struct Settings {
    action_idx: [usize; 24],
    default_joint_pos: [f32; 24],
    action_scale: [f32; 24],
    stiffness: [f32; 24],
    damping: [f32; 24],
    command: Command,
}
#[derive(Deserialize)]
struct Command {
    #[serde(rename = "H0")]
    h0: f32,
    recommended: Target,
}
#[derive(Deserialize)]
struct Target {
    h_target: f32,
    rate: f32,
    delay: f32,
}
pub struct SafeCrouch {
    session: Session,
    parameters: Parameters,
    observations: Observation,
    command: Command,
}
impl SafeCrouch {
    pub fn load() -> Result<Self, String> {
        let s: Settings = serde_json::from_str(include_str!("../config/safe_crouch.json"))
            .map_err(|e| e.to_string())?;
        let mut p = Parameters::default();
        p.action_idx = s.action_idx;
        p.default_joint_pos = s.default_joint_pos;
        p.action_scale = s.action_scale;
        p.stiffness = s.stiffness;
        p.damping = s.damping;
        Ok(Self {
            session: Session::load(include_bytes!("../config/safe_crouch.onnx"))?,
            parameters: p,
            observations: Observation::default(),
            command: s.command,
        })
    }
    pub fn command(&self, elapsed: f64) -> [f32; 3] {
        let c = &self.command;
        let t = &c.recommended;
        let h = (c.h0 - t.rate * ((elapsed as f32 - t.delay).max(0.))).max(t.h_target);
        [h - c.h0, t.h_target - c.h0, t.rate]
    }
    pub fn start(&mut self, input: &Input) -> Result<(), String> {
        input.validate()?;
        let command = self.command(0.);
        self.observations.reset(input, command, &self.parameters);
        Ok(())
    }
    pub fn sample(&mut self, input: &Input, elapsed: f64) -> Result<Output, String> {
        input.validate()?;
        if !elapsed.is_finite() || elapsed < 0. {
            return Err("invalid crouch elapsed time".into());
        }
        let command = self.command(elapsed);
        self.observations.update(input, command, &self.parameters);
        let actions = self.session.infer(&mut self.observations.history)?;
        self.observations.previous_action = actions;
        Ok(policy_output(&self.parameters, [0.; 2], actions))
    }
}
#[cfg(feature = "ros")]
impl walking_action_manager::policy_runtime::Policy for SafeCrouch {
    const ACTION: Action = Action::SafeCrouch;
    const NODE: &'static str = "safe_crouch_action_manager";
    const STATUS: &'static str = "/safe_crouch_status";
    fn start(&mut self, input: &Input) -> Result<(), String> {
        SafeCrouch::start(self, input)
    }
    fn sample(&mut self, input: &Input, _: [f32; 4], elapsed: f64) -> Result<Output, String> {
        SafeCrouch::sample(self, input, elapsed)
    }
}
