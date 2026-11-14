//! ROS I/O owned by the walking package; no USB access.
use crate::{
    motion::{mono_ms, Action, ActionPhase, ActionResult, Goal, GOAL_TOPIC, RESULT_TOPIC},
    observation::Input,
    Walking,
};
#[path = "../../../humanoid_controller/src/r1_profile.rs"]
#[allow(dead_code)]
mod hardware;
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::{
    humanoid_controller_msg::msg::LowState, sensor_msgs::msg::Joy,
    std_msgs::msg::String as RosString,
};
use serde_json::json;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub const STATUS_TOPIC: &str = "/walking_status";
struct State {
    walking: Result<Walking, String>,
    input: Option<Input>,
    axes: [f32; 4],
    keys: u16,
    goal: Option<Goal>,
    error: Option<String>,
    sequence: u64,
    tilt_count: u32,
    inference_us: u64,
}
impl State {
    fn readiness(&self) -> Result<(), String> {
        self.walking.as_ref().map_err(Clone::clone)?;
        let input = self.input.as_ref().ok_or("waiting for /lowstate")?;
        input.validate()?;
        if !input.upright() {
            return Err("native walking entry: roll/pitch exceeds 1 rad".into());
        }
        Ok(())
    }
}
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("walking_action_manager")?;
    let state = Arc::new(Mutex::new(State {
        walking: Walking::load(),
        input: None,
        axes: [0.; 4],
        keys: 0,
        goal: None,
        error: None,
        sequence: 0,
        tilt_count: 0,
        inference_us: 0,
    }));
    let results = node.create_publisher::<RosString>(RESULT_TOPIC.keep_last(10).reliable())?;
    let status =
        node.create_publisher::<RosString>(STATUS_TOPIC.keep_last(1).reliable().transient_local())?;
    let requests =
        node.create_publisher::<RosString>("/humanoid_state/request".keep_last(1).reliable())?;

    let input = Arc::clone(&state);
    let _feedback = node.create_subscription::<LowState, _>(
        "/lowstate".keep_last(1).best_effort(),
        move |m: LowState| {
            let sample = Input {
                q: hardware::NATIVE_INDICES.map(|i| m.motor_state[i].q),
                dq: hardware::NATIVE_INDICES.map(|i| m.motor_state[i].dq),
                quaternion: m.imu_state.quaternion,
                gyroscope: m.imu_state.gyroscope,
            };
            input.lock().unwrap().input = Some(sample);
        },
    )?;
    let input = Arc::clone(&state);
    let _joy =
        node.create_subscription::<Joy, _>("/joy".keep_last(1).best_effort(), move |m: Joy| {
            if m.axes.len() != 4
                || m.axes.iter().any(|v| !v.is_finite())
                || m.buttons.len() != 16
                || m.buttons.iter().any(|&v| v != 0 && v != 1)
            {
                return;
            }
            let mut s = input.lock().unwrap();
            s.axes = m.axes.try_into().unwrap();
            s.keys = m
                .buttons
                .iter()
                .enumerate()
                .fold(0, |bits, (i, b)| bits | ((*b as u16) << i));
            let keys = s.keys;
            if let Ok(walking) = &mut s.walking {
                walking.command.buttons(keys);
            }
        })?;
    let input = Arc::clone(&state);
    let _goals = node.create_subscription::<RosString, _>(
        GOAL_TOPIC.keep_last(1).reliable(),
        move |m: RosString| {
            let mut s = input.lock().unwrap();
            let Ok(goal) = serde_json::from_str::<Goal>(&m.data) else {
                return;
            };
            if goal.action != Action::Walking {
                s.goal = None;
                s.error = None;
                s.tilt_count = 0;
                return;
            }
            if let Some(previous) = s.goal.as_ref().filter(|g| g.id == goal.id) {
                if !previous.same_activation(&goal) {
                    s.error = Some("activation changed without a new goal id".into());
                }
            } else {
                let outcome = s.readiness().and_then(|_| {
                    let sample = s.input.clone().unwrap();
                    let axes = s.axes;
                    let keys = s.keys;
                    let walking = s.walking.as_mut().map_err(|e| e.clone())?;
                    walking.start(&sample, axes)?;
                    walking.command.buttons(keys);
                    Ok(())
                });
                s.error = outcome.err();
                s.sequence = 0;
                s.tilt_count = 0;
            }
            s.goal = Some(goal);
        },
    )?;

    // Native AmpLocomotion checks tilt in its 1 ms FSM update, five bad updates.
    let input = Arc::clone(&state);
    let _guard = node.create_timer_repeating(Duration::from_millis(1), move || {
        let mut s = input.lock().unwrap();
        if s.goal.is_none() {
            return;
        }
        if s.input.as_ref().is_some_and(|i| i.upright()) {
            s.tilt_count = 0;
            return;
        }
        s.tilt_count += 1;
        let limit = s
            .walking
            .as_ref()
            .map_or(5, |w| w.parameters.safety_fail_frame_threshold);
        if s.tilt_count >= limit {
            s.goal = None;
            s.error = Some("native tilt transition to damping".into());
            drop(s);
            if let Err(e) = requests.publish(RosString {
                data: "{\"action\":\"damping\"}".into(),
            }) {
                eprintln!("walking_action_manager: {e}");
            }
        }
    })?;
    let input = Arc::clone(&state);
    let _policy = node.create_timer_repeating(Duration::from_millis(20), move || {
        let mut s = input.lock().unwrap();
        let Some(goal) = s.goal.clone() else {
            return;
        };
        s.sequence += 1;
        let started = Instant::now();
        let output = if let Some(error) = &s.error {
            Err(error.clone())
        } else {
            let axes = s.axes;
            let sample = s.input.clone();
            sample
                .ok_or_else(|| "waiting for /lowstate".to_string())
                .and_then(|sample| {
                    s.walking
                        .as_mut()
                        .map_err(|e| e.clone())?
                        .sample(&sample, axes)
                })
        };
        s.inference_us = started.elapsed().as_micros() as u64;
        let (status_code, phase, message, joints) = match output {
            Ok(o) => (200, o.phase, "normal".to_string(), o.joints),
            Err(e) => {
                s.error = Some(e.clone());
                (202, ActionPhase::Unavailable, e, Vec::new())
            }
        };
        let result = ActionResult {
            goal_id: goal.id,
            action: Action::Walking,
            sequence: s.sequence,
            generated_mono_ms: mono_ms(),
            status_code,
            phase,
            message,
            joints,
        };
        drop(s);
        if let Ok(data) = serde_json::to_string(&result) {
            if let Err(e) = results.publish(RosString { data }) {
                eprintln!("walking_action_manager: {e}");
            }
        }
    })?;
    let _status=node.create_timer_repeating(Duration::from_millis(100),move || {
        let s=state.lock().unwrap();
        let ready=s.readiness();
        let error=s.error.clone().or_else(||ready.as_ref().err().cloned());
        let data=json!({"status_code":if error.is_none(){200}else{202},"ready":ready.is_ok() && s.error.is_none(),
            "active":s.goal.is_some(),"message":error.unwrap_or_else(||"ready".into()),
            "fsm_id":811,"model":s.walking.as_ref().ok().map(|w|w.model.name()),
            "velocity":s.walking.as_ref().ok().map(|w|w.command.velocity),
            "high_speed":s.walking.as_ref().ok().is_some_and(|w|w.command.high_speed),
            "moving":s.walking.as_ref().ok().is_some_and(|w|w.command.moving(s.axes,&w.parameters)),
            "sequence":s.sequence,"inference_us":s.inference_us});
        drop(s);
        if let Err(e)=status.publish(RosString{data:data.to_string()}) {eprintln!("walking_action_manager: {e}");}
    })?;
    eprintln!("walking_action_manager: manufacturer AmpLocomotion, 50 Hz; /lowstate + /joy -> joint targets");
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
