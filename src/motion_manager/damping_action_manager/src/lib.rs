//! Native Damping (FSM 1, 0x1034e440): kd=4, head kd=0.1.
// One contract copy, owned by state_manager.
#[path = "../../../state_manager/src/motion/mod.rs"]
pub mod motion;
use crate::motion::{commands, Action, ActionPhase, Generator, Goal, Output};

#[derive(Default)]
pub struct Damping;

impl Generator for Damping {
    const ACTION: Action = Action::Damping;
    const NODE: &'static str = "damping_action_manager";

    fn start(&mut self, _: &Goal) -> Result<(), String> {
        Ok(())
    }

    fn sample(&self, _: f64) -> Result<Output, String> {
        let mut kd = [4.0; 26];
        kd[24] = 0.1;
        kd[25] = 0.1;
        Ok(Output {
            phase: ActionPhase::Holding,
            joints: commands([0.0; 26], [0.0; 26], kd),
        })
    }
}
