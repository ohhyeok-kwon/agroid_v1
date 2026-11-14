//! state_manager node. /motor_status -> joint feedback (/joint_states, /tf) and motor
//! health; boot checks, remote keys, requests and action results -> one selected action
//! (/humanoid_state, goals, /motion_command). No USB I/O. Ages use CLOCK_MONOTONIC (`mono_ms`).
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::humanoid_controller_msg::msg::LowCmd;
use ros_env::{
    builtin_interfaces::msg::Time,
    geometry_msgs::msg::{Quaternion, Transform as RosTransform, TransformStamped, Vector3},
    sensor_msgs::msg::{JointState, Joy},
    std_msgs::msg::{Header, Int32MultiArray, String as RosString},
    tf2_msgs::msg::TFMessage,
};
use state_manager::{
    ankle::AnkleMap,
    joints::{is_repeat, MotorStatus, NAMES},
    model::{Model, Transform, URDF},
    motion::{mono_ms, now_ms, Action, GOAL_TOPIC, RESULT_TOPIC},
    remote_keys::{action_for, RemoteKeys},
    Manager,
};
use std::{
    process::ExitCode,
    sync::{Arc, Mutex},
    time::Duration,
};

/// ROS header stamp from wall-clock milliseconds (display only).
fn stamp(ms: u64) -> Time {
    Time {
        sec: (ms / 1000) as i32,
        nanosec: (ms % 1000) as u32 * 1_000_000,
    }
}

fn transform_message(transforms: Vec<Transform>, time: Time) -> TFMessage {
    TFMessage {
        transforms: transforms
            .into_iter()
            .map(|t| TransformStamped {
                header: Header {
                    stamp: time.clone(),
                    frame_id: t.parent,
                },
                child_frame_id: t.child,
                transform: RosTransform {
                    translation: Vector3 {
                        x: t.translation[0],
                        y: t.translation[1],
                        z: t.translation[2],
                    },
                    rotation: Quaternion {
                        x: t.rotation[0],
                        y: t.rotation[1],
                        z: t.rotation[2],
                        w: t.rotation[3],
                    },
                },
            })
            .collect(),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("state_manager")?;
    let manager = Arc::new(Mutex::new(Manager::new(format!(
        "{}-{}",
        now_ms(),
        std::process::id()
    ))));
    let remote_keys = Arc::new(Mutex::new(RemoteKeys::default()));
    let status = node.create_publisher::<RosString>(
        "/humanoid_state".keep_last(1).reliable().transient_local(),
    )?;
    let goals = node.create_publisher::<RosString>(GOAL_TOPIC.keep_last(1).reliable())?;
    // Volatile: restarting a downstream consumer must never replay an old motion.
    let command = node.create_publisher::<LowCmd>("/lowcmd".keep_last(1).best_effort())?;
    let joint_states =
        node.create_publisher::<JointState>("/joint_states".keep_last(10).reliable())?;
    let tf = node.create_publisher::<TFMessage>("/tf".keep_last(100).reliable())?;
    let tf_static =
        node.create_publisher::<TFMessage>("/tf_static".keep_last(1).reliable().transient_local())?;
    let description = node.create_publisher::<RosString>(
        "/robot_description"
            .keep_last(1)
            .reliable()
            .transient_local(),
    )?;
    let model = Model::load()?;
    tf_static.publish(transform_message(model.fixed(), stamp(0)))?;
    description.publish(RosString { data: URDF.into() })?;
    let ankle = AnkleMap::default();

    let input = Arc::clone(&manager);
    let _boot = node.create_subscription::<Int32MultiArray, _>(
        "/humanoid_status".keep_last(1).reliable().transient_local(),
        move |message: Int32MultiArray| input.lock().unwrap().receive_boot(message.data, mono_ms()),
    )?;

    // One /motor_status sample feeds health, the stance input and the published joint feedback.
    let input = Arc::clone(&manager);
    let feedback: Mutex<(Option<u64>, Option<String>)> = Mutex::new((None, None));
    let _motor = node.create_subscription::<RosString, _>(
        "/motor_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| {
            let now = mono_ms();
            let frame = serde_json::from_str::<MotorStatus>(&message.data)
                .map_err(|e| e.to_string())
                .and_then(|raw| raw.convert(&ankle, now_ms(), now));
            let mut state = feedback.lock().unwrap();
            let (last_order, last_report) = &mut *state;
            let repeat = frame
                .as_ref()
                .is_ok_and(|f| last_order.is_some_and(|last| is_repeat(f.order_ms, last)));
            {
                // The stance input: the newest complete frame; any measurement fault clears it
                // at once. Publishing below never affects it.
                let mut manager = input.lock().unwrap();
                manager.receive_motor(&message.data, now);
                match &frame {
                    Ok(_) if repeat => {}
                    Ok(frame) => manager.receive_joints(&frame.positions, frame.order_ms),
                    Err(_) => manager.receive_joints(&[None; 26], now),
                }
            }
            if repeat {
                return;
            }
            // Ok(report): published; empty report = all valid.
            let report = frame.and_then(|mut frame| {
                model.check_limits(&mut frame);
                let valid = frame.valid();
                if valid.is_empty() {
                    return Err(format!("no valid joint: {}", frame.problems.join("; ")));
                }
                // One /tf message carries every valid joint of one measurement; invalid joints
                // are omitted from /tf and /joint_states instead of stopping the whole robot.
                let time = stamp(frame.stamp_ms);
                tf.publish(transform_message(model.moving(&frame), time.clone()))
                    .map_err(|e| e.to_string())?;
                joint_states
                    .publish(JointState {
                        header: Header {
                            stamp: time,
                            frame_id: "pelvis_link".into(),
                        },
                        name: valid.iter().map(|&(i, _)| NAMES[i].into()).collect(),
                        position: valid.iter().map(|&(_, q)| q).collect(),
                        velocity: Vec::new(),
                        effort: Vec::new(),
                    })
                    .map_err(|e| e.to_string())?;
                *last_order = Some(frame.order_ms);
                Ok(if frame.problems.is_empty() {
                    String::new()
                } else {
                    format!("{}/26 joints; {}", valid.len(), frame.problems.join("; "))
                })
            });
            let report = report.unwrap_or_else(|report| report);
            if last_report.as_deref() != Some(report.as_str()) {
                if report.is_empty() {
                    eprintln!("state_manager: all 26 joints valid");
                } else {
                    eprintln!("state_manager: joints: {report}");
                }
                *last_report = Some(report);
            }
        },
    )?;

    let input = Arc::clone(&manager);
    let keys = Arc::clone(&remote_keys);
    let _remote = node.create_subscription::<RosString, _>(
        "/remote_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| {
            let now = mono_ms();
            keys.lock().unwrap().health(&message.data, now);
            input.lock().unwrap().receive_remote(&message.data, now)
        },
    )?;
    let keys = Arc::clone(&remote_keys);
    let _joy = node.create_subscription::<Joy, _>(
        "/joy".keep_last(10).best_effort(),
        move |message: Joy| {
            let header = message.header.stamp;
            let now = mono_ms();
            let mut keys = keys.lock().unwrap();
            if header.sec < 0
                || header.nanosec >= 1_000_000_000
                || message.axes.len() != 4
                || message.axes.iter().any(|value| !value.is_finite())
            {
                keys.receive(&[], now);
            } else {
                keys.receive(&message.buttons, now);
            }
        },
    )?;
    // Remote loss keeps the current action, as in the native FSM.
    let input = Arc::clone(&manager);
    let _remote_tick = node.create_timer_repeating(Duration::from_millis(5), move || {
        let event = remote_keys.lock().unwrap().poll(mono_ms());
        if let Some(event) = event {
            let mut manager = input.lock().unwrap();
            if let Some(action) = action_for(manager.selected_action(), &event) {
                manager.request_remote(action, mono_ms());
            }
        }
    })?;
    let input = Arc::clone(&manager);
    let _walking = node.create_subscription::<RosString, _>(
        "/walking_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| input.lock().unwrap().receive_walking(&message.data),
    )?;
    let mut _learned_policies = Vec::new();
    for (topic, action) in [
        ("/careful_walk_status", Action::CarefulWalk),
        ("/safe_crouch_status", Action::SafeCrouch),
    ] {
        let input = Arc::clone(&manager);
        _learned_policies.push(node.create_subscription::<RosString, _>(
            topic.keep_last(1).reliable().transient_local(),
            move |message: RosString| input.lock().unwrap().receive_policy(action, &message.data),
        )?);
    }
    let input = Arc::clone(&manager);
    let _requests = node.create_subscription::<RosString, _>(
        "/humanoid_state/request".keep_last(1).reliable(),
        move |message: RosString| input.lock().unwrap().request(&message.data, mono_ms()),
    )?;
    let input = Arc::clone(&manager);
    let safety_goals = Arc::clone(&goals);
    let safety_status = Arc::clone(&status);
    let _safety = node.create_subscription::<RosString, _>(
        "/safty_status".keep_last(1).reliable().transient_local(),
        move |message: RosString| {
            let now = mono_ms();
            let (goal, state) = {
                let mut manager = input.lock().unwrap();
                let before = manager.status(now);
                manager.receive_safety(&message.data, now);
                let state = manager.status(now);
                let goal_changed = before["goal_id"] != state["goal_id"];
                let changed = goal_changed
                    || before["safety"] != state["safety"]
                    || before["sources"]["safety"] != state["sources"]["safety"];
                (
                    goal_changed.then(|| manager.goal(now)).flatten(),
                    changed.then_some(state),
                )
            };
            // Publish the corrective goal immediately rather than wait for the 10 Hz timer.
            if let Some(goal) = goal {
                if let Ok(data) = serde_json::to_string(&goal) {
                    if let Err(error) = safety_goals.publish(RosString { data }) {
                        eprintln!("state_manager: safety goal publish failed: {error}");
                    }
                }
            }
            if let Some(state) = state {
                if let Err(error) = safety_status.publish(RosString {
                    data: state.to_string(),
                }) {
                    eprintln!("state_manager: safety state publish failed: {error}");
                }
            }
        },
    )?;
    let input = Arc::clone(&manager);
    let _results = node.create_subscription::<RosString, _>(
        RESULT_TOPIC.keep_last(10).reliable(),
        move |message: RosString| {
            let mut manager = input.lock().unwrap();
            let epoch = manager.controller_epoch();
            if let (Some(result), Some(epoch)) =
                (manager.receive_result(&message.data, mono_ms()), epoch)
            {
                drop(manager);
                let mut raw = state_manager::low_level::LowCommand::default();
                for (i, joint) in result.joints.iter().enumerate() {
                    raw.motors[state_manager::hardware::NATIVE_INDICES[i]] =
                        state_manager::low_level::Motor {
                            mode: if result.action == state_manager::motion::Action::ZeroMoment {
                                0
                            } else {
                                1
                            },
                            q: joint.position_rad as f32,
                            dq: joint.velocity_rad_s as f32,
                            tau: joint.torque_nm as f32,
                            kp: joint.kp as f32,
                            kd: joint.kd as f32,
                            reserve: 0,
                        };
                }
                let goal = result
                    .goal_id
                    .bytes()
                    .fold(2166136261u32, |h, b| {
                        (h ^ u32::from(b)).wrapping_mul(16777619)
                    })
                    .max(1);
                raw.reserve = [epoch as u32, (epoch >> 32) as u32, mono_ms() as u32, goal];
                let msg = LowCmd {
                    mode_pr: raw.mode_pr,
                    mode_machine: raw.mode_machine,
                    motor_cmd: std::array::from_fn(|i| {
                        let m = raw.motors[i];
                        ros_env::humanoid_controller_msg::msg::MotorCmd {
                            mode: m.mode,
                            q: m.q,
                            dq: m.dq,
                            tau: m.tau,
                            kp: m.kp,
                            kd: m.kd,
                            reserve: m.reserve,
                        }
                    }),
                    reserve: raw.reserve,
                    crc: raw.checksum(),
                };
                if let Err(error) = command.publish(msg) {
                    eprintln!("state_manager: command publish failed: {error}");
                }
            }
        },
    )?;
    let input = Arc::clone(&manager);
    let _tick = node.create_timer_repeating(Duration::from_millis(20), move || {
        input.lock().unwrap().tick(mono_ms())
    })?;
    let mut last_state = None;
    let _status = node.create_timer_repeating(Duration::from_millis(100), move || {
        let manager = manager.lock().unwrap();
        let now = mono_ms();
        if let Some(goal) = manager.goal(now) {
            if let Ok(data) = serde_json::to_string(&goal) {
                if let Err(error) = goals.publish(RosString { data }) {
                    eprintln!("state_manager: {error}");
                }
            }
        }
        let value = manager.status(now);
        let current = (
            value["state"].clone(),
            value["phase"].clone(),
            value["status_code"].clone(),
        );
        if last_state.as_ref() != Some(&current) {
            eprintln!("state_manager: {}", value);
            last_state = Some(current);
        }
        if let Err(error) = status.publish(RosString {
            data: value.to_string(),
        }) {
            eprintln!("state_manager: {error}");
        }
    })?;
    eprintln!(
        "state_manager: /motor_status -> /joint_states, /tf; one selected action, no USB I/O"
    );
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("state_manager: {error}");
            ExitCode::FAILURE
        }
    }
}
