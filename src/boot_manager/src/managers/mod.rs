pub mod battery_manager;
pub mod cpu_manager;
pub mod disk_manager;
pub mod humanoid_controller_manager;
pub mod memory_manager;
pub mod network_manager;
pub mod remote_controller_manager;
mod status_topic;
pub mod temp_manager;

pub const NORMAL: i32 = 200;
pub const WARNING: i32 = 201;
pub const ERROR: i32 = 202;
pub const NAMES: [&str; 8] = [
    "cpu_manager",
    "temp_manager",
    "memory_manager",
    "disk_manager",
    "network_manager",
    "battery_manager",
    "humanoid_controller",
    "remote_controller",
];

fn invalid_data(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}
