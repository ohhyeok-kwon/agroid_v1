//! ROS adapter. Safety decisions live in the package's independent modules.
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::{
    humanoid_controller_msg::msg::{LowCmd, LowState},
    std_msgs::msg::{Int32MultiArray, String as RosString},
};
use safty_manager::{
    clock::mono_ms,
    config::Config,
    hardware::NATIVE_INDICES,
    lowcmd::{LowCommand, Motor},
    BootSample, Feedback, Guard, Status,
};
use std::{
    process::ExitCode,
    sync::{Arc, Mutex},
    time::Duration,
};

const INPUT_TOPIC: &str = "/lowcmd";
const OUTPUT_TOPIC: &str = "/safty_manager/lowcmd";
const STATUS_TOPIC: &str = "/safty_status";

fn status_message(value: &Status) -> RosString {
    RosString {
        data: serde_json::to_string(value).expect("safety status serializes"),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (config, path) = Config::load()?;
    let now = mono_ms();
    let guard = Arc::new(Mutex::new(Guard::new(
        config,
        format!("{now}-{}", std::process::id()),
        now,
    )));
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("safty_manager")?;
    let output = node.create_publisher::<LowCmd>(OUTPUT_TOPIC.keep_last(1).best_effort())?;
    let status =
        node.create_publisher::<RosString>(STATUS_TOPIC.keep_last(1).reliable().transient_local())?;
    let input = Arc::clone(&guard);
    let events = Arc::clone(&status);
    let _commands = node.create_subscription::<LowCmd, _>(
        INPUT_TOPIC.keep_last(1).best_effort(),
        move |message: LowCmd| {
            let command = LowCommand {
                mode_pr: message.mode_pr,
                mode_machine: message.mode_machine,
                motors: std::array::from_fn(|i| {
                    let m = &message.motor_cmd[i];
                    Motor {
                        mode: m.mode,
                        q: m.q,
                        dq: m.dq,
                        tau: m.tau,
                        kp: m.kp,
                        kd: m.kd,
                        reserve: m.reserve,
                    }
                }),
                reserve: message.reserve,
                crc: message.crc,
            };
            let now = mono_ms();
            let (forward, changed) = {
                let mut guard = input.lock().unwrap();
                let before = guard.marker();
                let forward = guard.command(command, now);
                (
                    forward,
                    (before != guard.marker()).then(|| guard.status(now)),
                )
            };
            if let Some(value) = changed {
                if let Err(error) = events.publish(status_message(&value)) {
                    eprintln!("safty_manager: status publish failed: {error}");
                }
            }
            if forward {
                if let Err(error) = output.publish(message) {
                    eprintln!("safty_manager: command publish failed: {error}");
                }
            }
        },
    )?;

    let input = Arc::clone(&guard);
    let events = Arc::clone(&status);
    let _feedback = node.create_subscription::<LowState, _>(
        "/lowstate".keep_last(1).best_effort(),
        move |message: LowState| {
            let now = mono_ms();
            let changed = {
                let mut guard = input.lock().unwrap();
                let before = guard.marker();
                guard.receive_feedback(
                    Feedback {
                        received_ms: now,
                        position: NATIVE_INDICES.map(|i| f64::from(message.motor_state[i].q)),
                        velocity: NATIVE_INDICES.map(|i| f64::from(message.motor_state[i].dq)),
                    },
                    now,
                );
                (before != guard.marker()).then(|| guard.status(now))
            };
            if let Some(value) = changed {
                if let Err(error) = events.publish(status_message(&value)) {
                    eprintln!("safty_manager: status publish failed: {error}");
                }
            }
        },
    )?;
    let input = Arc::clone(&guard);
    let events = Arc::clone(&status);
    let _boot = node.create_subscription::<Int32MultiArray, _>(
        "/humanoid_status".keep_last(1).reliable().transient_local(),
        move |message: Int32MultiArray| {
            let now = mono_ms();
            let changed = {
                let mut guard = input.lock().unwrap();
                let before = guard.marker();
                guard.receive_boot(
                    BootSample {
                        received_ms: now,
                        codes: message.data,
                        names: message.layout.dim.first().map_or_else(Vec::new, |dim| {
                            dim.label.split(',').map(str::to_string).collect()
                        }),
                    },
                    now,
                );
                (before != guard.marker()).then(|| guard.status(now))
            };
            if let Some(value) = changed {
                if let Err(error) = events.publish(status_message(&value)) {
                    eprintln!("safty_manager: status publish failed: {error}");
                }
            }
        },
    )?;
    let input = Arc::clone(&guard);
    let _state = node.create_subscription::<RosString, _>(
        "/humanoid_state".keep_last(1).reliable().transient_local(),
        move |message: RosString| input.lock().unwrap().receive_state(&message.data),
    )?;
    let mut last_marker = None;
    let _timer = node.create_timer_repeating(Duration::from_millis(100), move || {
        let now = mono_ms();
        let value = {
            let mut guard = guard.lock().unwrap();
            guard.tick(now);
            guard.status(now)
        };
        let marker = (value.status_code, value.fault_id.clone());
        if last_marker.as_ref() != Some(&marker) {
            eprintln!("safty_manager: {} {}", value.status_code, value.message);
            last_marker = Some(marker);
        }
        if let Err(error) = status.publish(status_message(&value)) {
            eprintln!("safty_manager: status publish failed: {error}");
        }
    })?;
    eprintln!(
        "safty_manager: {INPUT_TOPIC} -> {OUTPUT_TOPIC}; config={}",
        path.display()
    );
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("safty_manager: {error}");
            ExitCode::FAILURE
        }
    }
}
