#[path = "../../humanoid_controller/src/clock.rs"]
mod clock;
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use remote_controller::{McuSample, Monitor};
use ros_env::{
    builtin_interfaces::msg::Time,
    sensor_msgs::msg::Joy,
    std_msgs::msg::{Header, String as RosString},
};
use std::{
    process::ExitCode,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("remote_controller")?;
    let joy = node.create_publisher::<Joy>("/joy".keep_last(10).best_effort())?;
    let status = node.create_publisher::<RosString>(
        "/remote_status".keep_last(1).reliable().transient_local(),
    )?;
    let monitor = Arc::new(Mutex::new(Monitor::default()));
    let input = Arc::clone(&monitor);
    // Packet age, order and status use CLOCK_MONOTONIC (shared with the controller's stamps).
    let _subscription = node.create_subscription::<RosString, _>(
        "/mcu_status".keep_last(256).reliable(),
        move |message: RosString| {
            let mut input = input.lock().unwrap();
            match serde_json::from_str::<McuSample>(&message.data) {
                Ok(sample) => {
                    if let Some((time, data)) = input.receive(sample, now_ms(), clock::mono_ms()) {
                        if let Err(error) = joy.publish(Joy {
                            header: Header {
                                stamp: Time {
                                    sec: (time / 1000) as i32,
                                    nanosec: (time % 1000) as u32 * 1_000_000,
                                },
                                frame_id: "remote_controller".into(),
                            },
                            axes: data.axes(),
                            buttons: data.buttons.to_vec(),
                        }) {
                            eprintln!("remote_controller: joy publish failed: {error}");
                        }
                    }
                }
                Err(error) => input.error(error.to_string()),
            }
        },
    )?;
    let mut last_code = None;
    let (mut reported_errors, mut reported_at) = (0, None::<Instant>);
    let _timer = node.create_timer_repeating(Duration::from_millis(100), move || {
        let monitor = monitor.lock().unwrap();
        let value = monitor.status(clock::mono_ms());
        if last_code != Some(value.status_code) {
            eprintln!("remote_controller: {} {}", value.status_code, value.message);
            last_code = Some(value.status_code);
        }
        // Discarded input does not change the code; keep it visible, at most every 10 s.
        if value.receive_errors > reported_errors
            && reported_at.is_none_or(|t| t.elapsed() >= Duration::from_secs(10))
        {
            eprintln!(
                "remote_controller: {} input errors discarded (total {}), last: {}",
                value.receive_errors - reported_errors,
                value.receive_errors,
                value.last_error.unwrap_or("-")
            );
            reported_errors = value.receive_errors;
            reported_at = Some(Instant::now());
        }
        if let Ok(data) = serde_json::to_string(&value) {
            if let Err(error) = status.publish(RosString { data }) {
                eprintln!("remote_controller: status publish failed: {error}");
            }
        }
    })?;
    eprintln!("remote_controller: /mcu_status -> /joy, /remote_status (receive only)");
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("remote_controller: {error}");
            ExitCode::FAILURE
        }
    }
}
