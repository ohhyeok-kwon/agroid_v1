//! Boot requires a connected remote, including when remote_controller reports a delay warning.
use super::{status_topic::StatusTopic, ERROR};
use std::io;

pub struct RemoteControllerManager {
    status: StatusTopic,
}

impl Default for RemoteControllerManager {
    fn default() -> Self {
        Self {
            status: StatusTopic::new("/remote_status"),
        }
    }
}

impl RemoteControllerManager {
    pub fn receive(&mut self, payload: &str) {
        self.status.receive(payload, |code, value| {
            if value["connected"].as_bool() == Some(true) {
                code
            } else {
                ERROR
            }
        });
    }

    pub fn check(&self) -> io::Result<i32> {
        self.status.check()
    }
}
