//! Manufacturer AmpLocomotion (FSM 811): observations, inference and joint targets.
pub mod command;
#[path = "../../../state_manager/src/motion/mod.rs"]
pub mod motion;
pub mod observation;
mod onnx;
#[cfg(feature = "ros")]
pub mod runtime;
use motion::{commands, ActionPhase, Output};
use observation::{Input, Observation};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub struct Parameters {
    pub dt: f32,
    pub action_idx: [usize; 24],
    pub client_idx: [usize; 24],
    pub default_joint_pos: [f32; 24],
    pub action_scale: [f32; 24],
    pub stiffness: [f32; 24],
    pub damping: [f32; 24],
    pub low_speed_forward_vx_limit: f32,
    pub high_speed_forward_vx_limit: f32,
    pub reverse_vx_limit: f32,
    pub vy_limit: f32,
    pub pure_wz_limit: f32,
    pub loco_wz_limit: f32,
    pub lateral_vy_zero_vx_ratio: f32,
    pub longitudinal_vx_zero_vy_ratio: f32,
    pub forward_vx_acc_up_delta: f32,
    pub forward_vx_acc_down_delta: f32,
    pub reverse_vx_acc_up_delta: f32,
    pub reverse_vx_acc_down_delta: f32,
    pub vy_acc_up_delta: f32,
    pub vy_acc_down_delta: f32,
    pub yaw_acc_up_delta: f32,
    pub yaw_acc_down_delta: f32,
    pub safety_fail_frame_threshold: u32,
}
impl Default for Parameters {
    fn default() -> Self {
        serde_json::from_str(include_str!("../config/amp_locomotion.json"))
            .expect("bundled manufacturer AmpLocomotion parameters")
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Walk,
    Run,
    Turn,
}
impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Self::Walk => "walk",
            Self::Run => "run",
            Self::Turn => "turn",
        }
    }
    /// Native conditional models: turn has priority over run (0x103a75d4).
    pub fn for_velocity([vx, vy, wz]: [f32; 3]) -> Self {
        if (-0.3..=0.3).contains(&vx) && (-0.3..=0.3).contains(&vy) && wz.abs() >= 1.0 {
            Self::Turn
        } else if vx >= 1.5 {
            Self::Run
        } else {
            Self::Walk
        }
    }
}
pub struct Walking {
    sessions: [onnx::Session; 3],
    pub parameters: Parameters,
    pub observations: Observation,
    pub command: command::Command,
    head: [f32; 2],
    pub model: Model,
}
impl Walking {
    pub fn load() -> Result<Self, String> {
        Ok(Self {
            sessions: [
                onnx::Session::load(include_bytes!("../config/walk.onnx"))?,
                onnx::Session::load(include_bytes!("../config/run.onnx"))?,
                onnx::Session::load(include_bytes!("../config/turn.onnx"))?,
            ],
            parameters: Parameters::default(),
            observations: Observation::default(),
            command: command::Command::default(),
            head: [0.; 2],
            model: Model::Walk,
        })
    }
    pub fn start(&mut self, input: &Input, axes: [f32; 4]) -> Result<(), String> {
        input.validate()?;
        if !input.upright() {
            return Err("native walking entry: roll/pitch exceeds 1 rad".into());
        }
        self.command = command::Command::default();
        let velocity = self.command.step(axes, &self.parameters);
        self.observations.reset(input, velocity, &self.parameters);
        self.head = [0.; 2]; // Stance's existing neck target in the native LowCmd buffer.
        self.model = Model::Walk;
        Ok(())
    }
    pub fn sample(&mut self, input: &Input, axes: [f32; 4]) -> Result<Output, String> {
        input.validate()?;
        let velocity = self.command.step(axes, &self.parameters);
        self.observations.update(input, velocity, &self.parameters);
        self.model = Model::for_velocity(velocity);
        let index = match self.model {
            Model::Walk => 0,
            Model::Run => 1,
            Model::Turn => 2,
        };
        let actions = self.sessions[index].infer(&mut self.observations.history)?;
        self.observations.previous_action = actions;
        Ok(self.output(actions))
    }
    pub fn output(&self, actions: [f32; 24]) -> Output {
        let p = &self.parameters;
        let mut q = [0.; 26];
        let mut kp = [0.; 26];
        let mut kd = [0.; 26];
        // Policy indices differ from the SDK order of our first 24 body joints.
        for (joint, &policy) in p.action_idx.iter().enumerate() {
            q[joint] =
                f64::from(p.default_joint_pos[policy] + actions[policy] * p.action_scale[policy]);
            kp[joint] = f64::from(p.stiffness[policy]);
            kd[joint] = f64::from(p.damping[policy]);
        }
        // Native policy leaves the two neck commands from Stance in the buffer.
        q[24] = f64::from(self.head[0]);
        q[25] = f64::from(self.head[1]);
        kp[24] = 50.;
        kp[25] = 10.;
        kd[24] = 2.;
        kd[25] = 0.1;
        Output {
            phase: ActionPhase::Holding,
            joints: commands(q, kp, kd),
        }
    }
}
