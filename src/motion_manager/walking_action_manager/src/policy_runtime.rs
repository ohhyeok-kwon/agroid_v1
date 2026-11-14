//! ROS transport for learned actions; policy and command generation stay in each action package.
use crate::{
    motion::{
        fresh, mono_ms, Action, ActionPhase, ActionResult, Goal, Output, GOAL_TOPIC, RESULT_TOPIC,
    },
    observation::Input,
};
#[path = "../../../humanoid_controller/src/r1_profile.rs"]
#[allow(dead_code)]
mod hardware;
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::{
    humanoid_controller_msg::msg::LowState, sensor_msgs::msg::Joy,
    std_msgs::msg::String as RosString,
};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

pub trait Policy: Send + 'static {
    const ACTION: Action;
    const NODE: &'static str;
    const STATUS: &'static str;
    fn start(&mut self, input: &Input) -> Result<(), String>;
    fn sample(&mut self, input: &Input, axes: [f32; 4], elapsed: f64) -> Result<Output, String>;
    fn moving(&self) -> bool {
        false
    }
}

struct State<P> {
    policy: Result<P, String>,
    input: Option<(u64, Input)>,
    axes: [f32; 4],
    joy_at: u64,
    goal: Option<Goal>,
    error: Option<String>,
    sequence: u64,
}
impl<P: Policy> State<P> {
    fn input(&self) -> Result<Input, String> {
        self.policy.as_ref().map_err(Clone::clone)?;
        let (at, input) = self.input.as_ref().ok_or("waiting for /lowstate")?;
        if !fresh(*at, mono_ms(), 500) {
            return Err("stale /lowstate".into());
        }
        input.validate()?;
        if !input.upright() {
            return Err("pelvis tilt exceeds 1 rad".into());
        }
        Ok(input.clone())
    }
}

pub fn run<P: Policy>(policy: Result<P, String>) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node(P::NODE)?;
    let state = Arc::new(Mutex::new(State {
        policy,
        input: None,
        axes: [0.; 4],
        joy_at: 0,
        goal: None,
        error: None,
        sequence: 0,
    }));
    let publisher = node.create_publisher::<RosString>(RESULT_TOPIC.keep_last(10).reliable())?;
    let status =
        node.create_publisher::<RosString>(P::STATUS.keep_last(1).reliable().transient_local())?;
    let input = Arc::clone(&state);
    let _low = node.create_subscription::<LowState, _>(
        "/lowstate".keep_last(1).best_effort(),
        move |m: LowState| {
            input.lock().unwrap().input = Some((
                mono_ms(),
                Input {
                    q: hardware::NATIVE_INDICES.map(|i| m.motor_state[i].q),
                    dq: hardware::NATIVE_INDICES.map(|i| m.motor_state[i].dq),
                    quaternion: m.imu_state.quaternion,
                    gyroscope: m.imu_state.gyroscope,
                },
            ));
        },
    )?;
    let input = Arc::clone(&state);
    let _joy =
        node.create_subscription::<Joy, _>("/joy".keep_last(1).best_effort(), move |m: Joy| {
            if m.axes.len() != 4 || m.axes.iter().any(|v| !v.is_finite() || v.abs() > 1.) {
                return;
            }
            let mut s = input.lock().unwrap();
            s.axes = m.axes.try_into().unwrap();
            s.joy_at = mono_ms();
        })?;
    let input = Arc::clone(&state);
    let _goal = node.create_subscription::<RosString, _>(
        GOAL_TOPIC.keep_last(1).reliable(),
        move |m: RosString| {
            let Ok(goal) = serde_json::from_str::<Goal>(&m.data) else {
                return;
            };
            let mut s = input.lock().unwrap();
            if goal.action != P::ACTION {
                s.goal = None;
                s.error = None;
                return;
            }
            if let Some(previous) = s.goal.as_ref().filter(|g| g.id == goal.id) {
                if !previous.same_activation(&goal) {
                    s.error = Some("activation changed without a new goal id".into());
                }
            } else {
                s.error = s
                    .input()
                    .and_then(|sample| s.policy.as_mut().map_err(|e| e.clone())?.start(&sample))
                    .err();
                s.sequence = 0;
            }
            s.goal = Some(goal);
        },
    )?;
    let input = Arc::clone(&state);
    let _timer = node.create_timer_repeating(Duration::from_millis(20), move || {
        let mut s = input.lock().unwrap();
        let Some(goal) = s.goal.clone() else { return };
        s.sequence += 1;
        let now = mono_ms();
        let output = if let Some(error) = &s.error {
            Err(error.clone())
        } else if !fresh(goal.issued_mono_ms, now, 500) {
            Err("stale action goal".into())
        } else {
            let axes = if fresh(s.joy_at, now, 500) {
                s.axes
            } else {
                [0.; 4]
            };
            s.input().and_then(|sample| {
                s.policy.as_mut().map_err(|e| e.clone())?.sample(
                    &sample,
                    axes,
                    now.saturating_sub(goal.started_mono_ms) as f64 / 1000.,
                )
            })
        };
        let (status_code, phase, message, joints) = match output {
            Ok(o) => (200, o.phase, "normal".to_string(), o.joints),
            Err(e) => {
                s.error = Some(e.clone());
                (202, ActionPhase::Unavailable, e, Vec::new())
            }
        };
        let result = ActionResult {
            goal_id: goal.id,
            action: P::ACTION,
            sequence: s.sequence,
            generated_mono_ms: now,
            status_code,
            phase,
            message,
            joints,
        };
        if let Ok(data) = serde_json::to_string(&result) {
            let _ = publisher.publish(RosString { data });
        }
    })?;
    let _status=node.create_timer_repeating(Duration::from_millis(100),move || {
        let s=state.lock().unwrap(); let ready=s.input();
        let error=s.error.clone().or_else(||ready.err());
        let data=serde_json::json!({"ready":error.is_none(),"active":s.goal.is_some(),"moving":s.policy.as_ref().is_ok_and(|p|p.moving()),"message":error.unwrap_or_else(||"ready".into()),"fsm_id":P::ACTION.fsm_id()});
        let _=status.publish(RosString{data:data.to_string()});
    })?;
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
