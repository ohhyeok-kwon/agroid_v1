//! The controller owns USB/motor/initialization checks; forward its status and detect silence.
use super::status_topic::StatusTopic;
use std::io;

pub struct HumanoidControllerManager {
    status: StatusTopic,
}

impl Default for HumanoidControllerManager {
    fn default() -> Self {
        Self {
            status: StatusTopic::new("/motor_status"),
        }
    }
}

impl HumanoidControllerManager {
    pub fn receive(&mut self, payload: &str) {
        self.status.receive(payload, |code, _| code);
    }

    pub fn check(&self) -> io::Result<i32> {
        self.status.check()
    }
}
