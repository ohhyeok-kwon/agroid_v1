use crate::motion::{
    mono_ms, ActionPhase, ActionResult, Generator, Goal, GOAL_TOPIC, RESULT_TOPIC,
};
use rclrs::{Context, CreateBasicExecutor, IntoPrimitiveOptions, RclrsErrorFilter, SpinOptions};
use ros_env::std_msgs::msg::String as RosString;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

struct Active<G> {
    generator: G,
    goal: Option<Goal>,
    started: Instant,
    error: Option<String>,
    sequence: u64,
}

pub fn run<G: Generator>(generator: G) -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node(G::NODE)?;
    let publisher = node.create_publisher::<RosString>(RESULT_TOPIC.keep_last(10).reliable())?;
    let active = Arc::new(Mutex::new(Active {
        generator,
        goal: None,
        started: Instant::now(),
        error: None,
        sequence: 0,
    }));
    let input = Arc::clone(&active);
    let _goal = node.create_subscription::<RosString, _>(
        GOAL_TOPIC.keep_last(1).reliable(),
        move |message: RosString| {
            let mut state = input.lock().unwrap();
            let Ok(goal) = serde_json::from_str::<Goal>(&message.data) else {
                state.goal = None;
                return;
            };
            if goal.action != G::ACTION {
                state.goal = None;
                return;
            }
            if let Some(previous) = state.goal.as_ref().filter(|g| g.id == goal.id) {
                if !previous.same_activation(&goal) {
                    state.error = Some("activation changed without a new goal id".into());
                }
            } else {
                state.started = Instant::now();
                state.sequence = 0;
                state.error = state.generator.start(&goal).err();
            }
            state.goal = Some(goal);
        },
    )?;
    let _timer = node.create_timer_repeating(Duration::from_millis(20), move || {
        let mut state = active.lock().unwrap();
        let Some(goal) = state.goal.clone() else {
            return;
        };
        state.sequence += 1;
        let output = match &state.error {
            Some(error) => Err(error.clone()),
            None => state
                .generator
                .sample(state.started.elapsed().as_secs_f64()),
        };
        let (status_code, phase, message, joints) = match output {
            Ok(output) => (200, output.phase, "normal".into(), output.joints),
            Err(error) => (202, ActionPhase::Unavailable, error, Vec::new()),
        };
        let result = ActionResult {
            goal_id: goal.id,
            action: G::ACTION,
            sequence: state.sequence,
            generated_mono_ms: mono_ms(),
            status_code,
            phase,
            message,
            joints,
        };
        if let Ok(data) = serde_json::to_string(&result) {
            if let Err(error) = publisher.publish(RosString { data }) {
                eprintln!("{}: {error}", G::NODE);
            }
        }
    })?;
    eprintln!("{}: joint-space generation only; no USB interface", G::NODE);
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
