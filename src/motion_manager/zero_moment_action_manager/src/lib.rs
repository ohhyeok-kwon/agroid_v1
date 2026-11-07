//! Native ZeroTorque (FSM 0, 0x103538a0): every joint field is zero.
// One contract copy, owned by state_manager.
#[path = "../../../state_manager/src/motion/mod.rs"]
pub mod motion;
use crate::motion::{commands, Action, ActionPhase, Generator, Goal, Output};

#[derive(Default)]
pub struct ZeroMoment;

impl Generator for ZeroMoment {
    const ACTION: Action = Action::ZeroMoment;
    const NODE: &'static str = "zero_moment_action_manager";

    fn start(&mut self, _: &Goal) -> Result<(), String> {
        Ok(())
    }

    fn sample(&self, _: f64) -> Result<Output, String> {
        Ok(Output {
            phase: ActionPhase::Holding,
            joints: commands([0.0; 26], [0.0; 26], [0.0; 26]),
        })
    }
}
