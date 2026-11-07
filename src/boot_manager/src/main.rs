//! Sequential system checks, then continuous status publication (1 Hz).
mod managers;

use managers::{
    battery_manager::BatteryManager, cpu_manager::CpuManager,
    humanoid_controller_manager::HumanoidControllerManager,
    remote_controller_manager::RemoteControllerManager, ERROR, NAMES, NORMAL, WARNING,
};
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::std_msgs::msg::{
    Int32MultiArray, MultiArrayDimension, MultiArrayLayout, String as RosString,
};
use std::{
    io,
    process::ExitCode,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const USAGE: &str = "usage: boot_manager_node [-h] [--check]
  --check  collect battery and node status for 3 seconds, then check without publishing";

struct BootManager {
    cpu: CpuManager,
    battery: Arc<Mutex<BatteryManager>>,
    controller: Arc<Mutex<HumanoidControllerManager>>,
    remote: Arc<Mutex<RemoteControllerManager>>,
    next: usize,
    status: [Option<i32>; NAMES.len()],
}

impl BootManager {
    fn new() -> Self {
        Self {
            cpu: CpuManager::new(),
            battery: Arc::new(Mutex::new(BatteryManager::default())),
            controller: Arc::new(Mutex::new(HumanoidControllerManager::default())),
            remote: Arc::new(Mutex::new(RemoteControllerManager::default())),
            next: 0,
            status: [None; NAMES.len()],
        }
    }

    fn check_through(&mut self, last: usize) {
        for (index, name) in NAMES.iter().enumerate().take(last + 1) {
            let result = match index {
                0 => self.cpu.check().map(resource_status),
                1 => managers::temp_manager::check().map(resource_status),
                2 => managers::memory_manager::check().map(resource_status),
                3 => managers::disk_manager::check().map(resource_status),
                4 => managers::network_manager::check().map(resource_status),
                5 => self
                    .battery
                    .lock()
                    .map_err(|_| io::Error::other("battery monitor unavailable"))
                    .and_then(|battery| battery.check()),
                6 => self
                    .controller
                    .lock()
                    .map_err(|_| io::Error::other("controller monitor unavailable"))
                    .and_then(|controller| controller.check()),
                _ => self
                    .remote
                    .lock()
                    .map_err(|_| io::Error::other("remote monitor unavailable"))
                    .and_then(|remote| remote.check()),
            };
            let code = result.as_ref().copied().unwrap_or(ERROR);
            if self.status[index] != Some(code) {
                eprintln!("boot_manager: {name} status={code}");
                if let Err(error) = result {
                    eprintln!("boot_manager: {name}: {error}");
                }
            }
            self.status[index] = Some(code);
        }
    }

    fn tick(&mut self) {
        self.check_through(self.next.min(NAMES.len() - 1));
        if advance(&mut self.next, &self.status) {
            if self.next == NAMES.len() {
                eprintln!("boot_manager: phase=complete");
            } else {
                eprintln!("boot_manager: phase={}", NAMES[self.next]);
            }
        }
    }

    fn message(&self) -> Int32MultiArray {
        // One-dimensional array; labels and values use the same fixed manager order.
        Int32MultiArray {
            layout: MultiArrayLayout {
                dim: vec![MultiArrayDimension {
                    label: NAMES.join(","),
                    size: NAMES.len() as u32,
                    stride: NAMES.len() as u32,
                }],
                data_offset: 0,
            },
            data: self
                .status
                .iter()
                .map(|code| code.unwrap_or(ERROR))
                .collect(),
        }
    }
}

fn resource_status(healthy: bool) -> i32 {
    if healthy {
        NORMAL
    } else {
        WARNING
    }
}

fn advance(next: &mut usize, status: &[Option<i32>; NAMES.len()]) -> bool {
    if *next < NAMES.len()
        && status[..=*next]
            .iter()
            .all(|code| matches!(code, Some(NORMAL) | Some(WARNING)))
    {
        *next += 1;
        true
    } else {
        false
    }
}

struct ShutdownSignals(libc::sigset_t);

impl ShutdownSignals {
    fn block() -> io::Result<Self> {
        // SAFETY: initialize a signal set and block it before ROS creates threads.
        unsafe {
            let mut set = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, libc::SIGTERM);
            libc::sigaddset(&mut set, libc::SIGINT);
            let rc = libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
            if rc != 0 {
                return Err(io::Error::from_raw_os_error(rc));
            }
            Ok(Self(set))
        }
    }

    fn requested(&self) -> io::Result<bool> {
        let timeout = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: the signal set and zero timeout are initialized; siginfo is optional.
        if unsafe { libc::sigtimedwait(&self.0, std::ptr::null_mut(), &timeout) } >= 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EAGAIN | libc::EINTR) => Ok(false),
            _ => Err(error),
        }
    }
}

fn run(check: bool) -> Result<(), Box<dyn std::error::Error>> {
    let signals = ShutdownSignals::block()?;
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("boot_manager")?;
    let mut manager = BootManager::new();
    let battery = Arc::clone(&manager.battery);
    let _subscription = node.create_subscription::<RosString, _>(
        "bms_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| {
            if let Ok(mut battery) = battery.lock() {
                battery.receive(&message.data);
            }
        },
    )?;
    let controller = Arc::clone(&manager.controller);
    let _controller = node.create_subscription::<RosString, _>(
        "/motor_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| {
            if let Ok(mut controller) = controller.lock() {
                controller.receive(&message.data);
            }
        },
    )?;
    let remote = Arc::clone(&manager.remote);
    let _remote = node.create_subscription::<RosString, _>(
        "/remote_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| {
            if let Ok(mut remote) = remote.lock() {
                remote.receive(&message.data);
            }
        },
    )?;
    if check {
        let until = Instant::now() + managers::battery_manager::MAX_AGE;
        while Instant::now() < until && !signals.requested()? {
            executor
                .spin(SpinOptions::default().timeout(Duration::from_millis(100)))
                .timeout_ok()
                .first_error()?;
        }
        manager.check_through(NAMES.len() - 1);
        return if manager.status.iter().all(|code| *code == Some(NORMAL)) {
            Ok(())
        } else {
            Err(io::Error::other("one or more managers reported a warning or error").into())
        };
    }
    let publisher = node.create_publisher::<Int32MultiArray>(
        "humanoid_status".keep_last(1).reliable().transient_local(),
    )?;
    eprintln!("boot_manager: phase=start");
    eprintln!("boot_manager: phase={}", NAMES[0]);
    publisher.publish(manager.message())?;
    let _timer = node.create_timer_repeating(Duration::from_secs(1), move || {
        manager.tick();
        if let Err(error) = publisher.publish(manager.message()) {
            eprintln!("boot_manager: humanoid_status publish failed: {error}");
        }
    })?;
    while !signals.requested()? {
        executor
            .spin(SpinOptions::default().timeout(Duration::from_millis(100)))
            .timeout_ok()
            .first_error()?;
    }
    eprintln!("boot_manager: phase=stopped");
    Ok(())
}

/// Keep ROS remapping/parameter arguments available to Context, but out of CLI parsing.
fn strip_ros_args(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut kept = Vec::new();
    let mut in_ros_args = false;
    for arg in args {
        if arg == "--ros-args" {
            in_ros_args = true;
        } else if in_ros_args && arg == "--" {
            in_ros_args = false;
        } else if !in_ros_args {
            kept.push(arg);
        }
    }
    kept
}

fn main() -> ExitCode {
    let mut check = false;
    for arg in strip_ros_args(std::env::args().skip(1)) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--check" => check = true,
            _ => {
                eprintln!("{USAGE}\nunknown argument: {arg}");
                return ExitCode::from(2);
            }
        }
    }
    match run(check) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("boot_manager: error: {error}");
            ExitCode::FAILURE
        }
    }
}
