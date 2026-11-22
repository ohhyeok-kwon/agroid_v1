//! CarefulWalk (FSM 812): the SONIC-R1 whole-body tracker (R14@100) following the calm v4.3s walking reference.
//!
//! 50 Hz tick: stick -> one bank command -> reference frames (bank) -> tracker observation -> ONNX -> joint
//! targets. Nothing here is tuned on the robot: joint order, default pose, gains and action scale are the
//! training contract (`careful_walk.json`), the reference is the generator's own baked output.
pub mod bank;
pub mod tracker;

use bank::{Bank, Player, JOINTS};
use serde::Deserialize;
use tracker::{Reference, Sample, Tracker, OBS};
use walking_action_manager::{
    motion::{commands, profile, ActionPhase, Output},
    observation::Input,
    onnx::Session,
};

static MODEL: &[u8] = include_bytes!("../config/r14s100_tracker.onnx");
static BANK: &[u8] = include_bytes!("../config/calm43s_bank.bin");
static GOLDEN: &[u8] = include_bytes!("../config/tracker_golden.bin");

/// Training contract of the tracker, policy joint order unless noted.
#[derive(Deserialize)]
pub struct Settings {
    pub joint_names: Vec<String>,
    /// SDK joint i (the first 24 of `profile::NAMES`) -> policy index.
    pub action_idx: [usize; JOINTS],
    pub default_joint_pos: [f64; JOINTS],
    pub action_scale: [f64; JOINTS],
    pub stiffness: [f64; JOINTS],
    pub damping: [f64; JOINTS],
}
impl Settings {
    pub fn load() -> Result<Self, String> {
        let s: Self = serde_json::from_str(include_str!("../config/careful_walk.json")).map_err(|e| e.to_string())?;
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        if s.joint_names.len() != JOINTS
            || (0..JOINTS).any(|i| s.action_idx[i] >= JOINTS || s.joint_names[s.action_idx[i]] != profile::NAMES[i])
            || !(finite(&s.default_joint_pos) && finite(&s.action_scale))
            || s.stiffness.iter().chain(&s.damping).any(|v| !(v.is_finite() && *v > 0.))
        {
            return Err("careful_walk.json does not match the R1 joint table".into());
        }
        Ok(s)
    }
}

const ENTER: f32 = 0.3;
const RELEASE: f32 = 0.15;
const SETTLE_TICKS: u32 = 10;
const RELEASE_TICKS: u32 = 3;
const CENTRE_TICKS: u32 = 10;

/// Stick -> one bank command. Six directions (forward, back, left, right, turn left, turn right). A direction is
/// taken once it has been the largest deflection (at least 0.3) for 0.2 s; its speed level is read from the
/// deflection at that moment and kept. The command ends when the deflection along it stays under 0.15 for 60 ms
/// or when another direction is taken.
///
/// A stick that is already pushed never starts a walk: after entry and after a lost remote no direction is taken
/// until the three command axes have been centred (under 0.15) for 0.2 s. A lost remote ends the command at once.
pub struct Selector {
    /// Per direction: bank command indices, slowest first.
    levels: [Vec<usize>; 6],
    active: Option<(usize, usize)>,
    candidate: Option<(usize, u32)>,
    released: u32,
    /// Ticks the command axes have been centred since entry or since the remote was lost, up to CENTRE_TICKS.
    centred: u32,
}
impl Selector {
    pub fn new(bank: &Bank) -> Self {
        let mut levels: [Vec<usize>; 6] = Default::default();
        for (index, command) in bank.commands.iter().enumerate() {
            let axis = command.twist.iter().position(|v| *v != 0.).unwrap_or(0);
            levels[2 * axis + usize::from(command.twist[axis] < 0.)].push(index);
        }
        let speed = |i: &usize| bank.commands[*i].twist.iter().map(|v| v.abs()).sum::<f64>();
        for direction in &mut levels {
            direction.sort_by(|a, b| speed(a).total_cmp(&speed(b)));
        }
        Self { levels, active: None, candidate: None, released: 0, centred: 0 }
    }
    /// Entry: standing, and the sticks have to be seen centred before a direction is taken.
    pub fn reset(&mut self) {
        (self.active, self.candidate, self.released, self.centred) = (None, None, 0, 0);
    }
    pub fn active(&self) -> Option<usize> {
        self.active.map(|(_, command)| command)
    }
    /// A tick without fresh remote data: stand, and wait for centred sticks again.
    pub fn lost(&mut self) -> Option<usize> {
        self.reset();
        None
    }
    /// One tick of /joy axes [lx, ly, rx, ry]: forward = ly, left = -lx, turn left = -rx.
    pub fn step(&mut self, axes: [f32; 4]) -> Option<usize> {
        let values = [axes[1], -axes[0], -axes[2]];
        if self.centred < CENTRE_TICKS {
            self.centred = if values.iter().all(|v| v.abs() < RELEASE) { self.centred + 1 } else { 0 };
            return None;
        }
        let along = |direction: usize| if direction % 2 == 0 { values[direction / 2] } else { -values[direction / 2] };
        let direction = (1..6).fold(0, |best, d| if along(d) > along(best) { d } else { best });
        if self.active.is_some_and(|(current, _)| along(current) < RELEASE) {
            self.released += 1;
            if self.released >= RELEASE_TICKS {
                self.active = None;
            }
        } else {
            self.released = 0;
        }
        let levels = &self.levels[direction];
        let size = along(direction);
        if size >= ENTER && !levels.is_empty() && self.active.map(|(current, _)| current) != Some(direction) {
            let ticks = self.candidate.filter(|(d, _)| *d == direction).map_or(1, |(_, t)| t + 1);
            self.candidate = Some((direction, ticks));
            if ticks >= SETTLE_TICKS {
                let level = ((size - ENTER) / (1. - ENTER) * levels.len() as f32) as usize;
                self.active = Some((direction, levels[level.min(levels.len() - 1)]));
                (self.candidate, self.released) = (None, 0);
            }
        } else {
            self.candidate = None;
        }
        self.active()
    }
}

pub struct CarefulWalk {
    session: Session,
    pub settings: Settings,
    pub player: Player,
    pub selector: Selector,
    pub tracker: Tracker,
    obs: Box<[f32; OBS]>,
}
impl CarefulWalk {
    pub fn load() -> Result<Self, String> {
        let settings = Settings::load()?;
        let bank = Bank::load(BANK, &settings.joint_names)?;
        Ok(Self {
            session: Self::model()?,
            selector: Selector::new(&bank),
            player: Player::new(bank),
            settings,
            tracker: Tracker::default(),
            obs: Box::new([0.; OBS]),
        })
    }
    /// Entry from completed standing: empty history, reference standing, heading taken at the first tick.
    pub fn start(&mut self, input: &Input) -> Result<(), String> {
        input.validate()?;
        self.player.reset();
        self.selector.reset();
        self.tracker.reset();
        Ok(())
    }
    /// `axes`: the remote's sticks of this tick, or None when no fresh remote data exists.
    pub fn sample(&mut self, input: &Input, axes: Option<[f32; 4]>) -> Result<Output, String> {
        input.validate()?;
        let command = match axes {
            Some(axes) => self.selector.step(axes),
            None => self.selector.lost(),
        };
        self.step(input, command)
    }
    /// One tick with the bank command already chosen (None = stand).
    pub fn step(&mut self, input: &Input, command: Option<usize>) -> Result<Output, String> {
        self.player.tell(command);
        let s = &self.settings;
        let mut sample = Sample { q: [0.; JOINTS], dq: [0.; JOINTS], gyro: input.gyroscope, quaternion: input.quaternion };
        for (joint, &policy) in s.action_idx.iter().enumerate() {
            sample.q[policy] = (f64::from(input.q[joint]) - s.default_joint_pos[policy]) as f32;
            sample.dq[policy] = input.dq[joint];
        }
        self.tracker.observe(&sample, &self.player, &mut self.obs)?;
        let actions = Self::infer(&mut self.session, &mut self.obs)?;
        self.tracker.last_action = actions;
        self.player.advance();
        Ok(self.output(&actions))
    }
    /// The bundled tracker network: observation -> raw actions in policy order.
    pub fn infer(session: &mut Session, obs: &mut [f32; OBS]) -> Result<[f32; JOINTS], String> {
        let mut actions = [0.; JOINTS];
        session.run(c"obs", obs, c"actions", &mut actions)?;
        Ok(actions)
    }
    /// A session of the bundled tracker network.
    pub fn model() -> Result<Session, String> {
        Session::load(MODEL)
    }
    /// The observation of the last tick (encoder input, then proprioception).
    pub fn observation(&self) -> &[f32; OBS] {
        &self.obs
    }
    pub fn output(&self, actions: &[f32; JOINTS]) -> Output {
        let s = &self.settings;
        let (mut q, mut kp, mut kd) = ([0.; 26], [0.; 26], [0.; 26]);
        for (joint, &policy) in s.action_idx.iter().enumerate() {
            q[joint] = s.default_joint_pos[policy] + f64::from(actions[policy]) * s.action_scale[policy];
            kp[joint] = s.stiffness[policy];
            kd[joint] = s.damping[policy];
        }
        // The tracker has no head joints: keep the two neck targets and gains of standing.
        (kp[24], kp[25], kd[24], kd[25]) = (50., 10., 2., 0.1);
        Output { phase: ActionPhase::Holding, joints: commands(q, kp, kd) }
    }
    pub fn moving(&self) -> bool {
        self.selector.active().is_some() || !self.player.standing()
    }
}

/// CPU of the whole node on the robot. The R1 board's fast cores are CPUs 4-7; CPU 7 is reserved for the motor USB
/// loop and CPU 6 shares its frequency floor (`scripts/daemon/04_realtime_tuning.sh`). One tracker inference per
/// 20 ms tick needs a fast core that never idles down to a low frequency, so the node is bound to CPU 6 before any
/// other thread exists. `AGROID_CAREFUL_WALK_CPU` (a CPU number, or `none`) overrides it for measurements.
/// Err: a binding was due and did not happen. The node then stays up but reports not ready, because the tick
/// computation is too slow on the robot's other cores (measured on the board).
pub fn bind_cpu() -> Result<String, String> {
    let wanted = match std::env::var("AGROID_CAREFUL_WALK_CPU") {
        Ok(text) if text == "none" => None,
        Ok(text) => match text.parse::<usize>() {
            Ok(cpu) if cpu < libc::CPU_SETSIZE as usize => Some(cpu),
            _ => return Err(format!("cpu: bad AGROID_CAREFUL_WALK_CPU={text}, not bound")),
        },
        Err(_) => cfg!(target_arch = "aarch64").then_some(6),
    };
    let Some(cpu) = wanted else {
        return Ok("cpu: not bound".into());
    };
    // SAFETY: the set is initialised and the CPU index is bounded; pid 0 is the calling thread, and the threads
    // created afterwards inherit its mask.
    let result = unsafe {
        let mut cpus = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_ZERO(&mut cpus);
        libc::CPU_SET(cpu, &mut cpus);
        libc::sched_setaffinity(0, std::mem::size_of_val(&cpus), &cpus)
    };
    if result != 0 {
        return Err(format!("cpu: binding to {cpu} failed ({}), not bound", std::io::Error::last_os_error()));
    }
    Ok(format!("cpu: bound to {cpu}"))
}

fn percentile(sorted: &[f64], percent: usize) -> f64 {
    sorted[(sorted.len() * percent / 100).min(sorted.len() - 1)]
}

/// Longest allowed 95th-percentile computation of one tick (ms). The tracker was trained with 15 ms between a
/// measurement and its action taking effect; the computation has to leave room for the transport inside that.
const TICK_BUDGET_MS: f64 = 10.;

/// On-target check without ROS or motion: the bundled model against stored training-side outputs, a walk through
/// every bank command, and the time the whole tick computation takes at the real 20 ms cadence (`paced_ticks`
/// ticks: standing, then walking the first bank command on a robot that stays in the default pose).
/// Run as `careful_walk_action_manager_node --self-test`.
pub fn self_test(paced_ticks: usize) -> Result<String, String> {
    const ROW: usize = OBS + JOINTS;
    let mut walk = CarefulWalk::load()?;
    if GOLDEN.len() < 12 || &GOLDEN[..8] != b"R1GOLD01" {
        return Err("golden file: bad magic".into());
    }
    let count = u32::from_le_bytes(GOLDEN[8..12].try_into().unwrap()) as usize;
    if count == 0 || GOLDEN.len() != 12 + count * ROW * 4 {
        return Err("golden file: bad size".into());
    }
    let value = |row: usize, i: usize| {
        let at = 12 + (row * ROW + i) * 4;
        f32::from_le_bytes(GOLDEN[at..at + 4].try_into().unwrap())
    };
    let (mut worst, mut burst) = (0f32, Vec::with_capacity(200));
    let mut obs = [0f32; OBS];
    for pass in 0..200 {
        let row = pass % count;
        obs.iter_mut().enumerate().for_each(|(i, v)| *v = value(row, i));
        let started = std::time::Instant::now();
        let actions = CarefulWalk::infer(&mut walk.session, &mut obs)?;
        burst.push(started.elapsed().as_secs_f64() * 1e3);
        for (i, a) in actions.iter().enumerate() {
            worst = worst.max((a - value(row, OBS + i)).abs());
        }
    }
    burst.sort_by(f64::total_cmp);
    let mut jump = 0f32;
    for command in 0..walk.player.bank().commands.len() {
        walk.player.reset();
        for tick in 0..400 {
            walk.player.tell(Some(command).filter(|_| (20..220).contains(&tick)));
            let (a, b) = (walk.player.joints(0), walk.player.joints(1));
            jump = (0..JOINTS).fold(jump, |m, j| m.max((b[j] - a[j]).abs()));
            walk.player.advance();
        }
        if !walk.player.standing() {
            return Err(format!("{} does not return to standing", walk.player.bank().commands[command].name));
        }
    }
    let mut q = [0f32; 26];
    for (joint, &policy) in walk.settings.action_idx.iter().enumerate() {
        q[joint] = walk.settings.default_joint_pos[policy] as f32;
    }
    let input = Input { q, dq: [0.; 26], quaternion: [1., 0., 0., 0.], gyroscope: [0.; 3] };
    walk.start(&input)?;
    let (period, begin) = (std::time::Duration::from_millis(20), std::time::Instant::now());
    let mut paced = Vec::with_capacity(paced_ticks.max(1));
    for tick in 0..paced_ticks.max(1) {
        if let Some(wait) = (begin + period * tick as u32).checked_duration_since(std::time::Instant::now()) {
            std::thread::sleep(wait);
        }
        let started = std::time::Instant::now();
        walk.step(&input, Some(0).filter(|_| tick >= 25))?;
        paced.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let late = paced.iter().filter(|ms| **ms > TICK_BUDGET_MS).count();
    paced.sort_by(f64::total_cmp);
    walk.player.reset();
    let report = format!(
        "self-test: golden rows {count}, max |action difference| {worst:.2e}; bank commands {}, largest joint step \
         between frames {jump:.4} rad; inference back to back ms p50 {:.2} p95 {:.2} max {:.2}; whole tick at 50 Hz \
         ({} ticks) ms p50 {:.2} p95 {:.2} max {:.2}, over {TICK_BUDGET_MS} ms: {late}; ran on CPU {}",
        walk.player.bank().commands.len(),
        percentile(&burst, 50),
        percentile(&burst, 95),
        burst[burst.len() - 1],
        paced.len(),
        percentile(&paced, 50),
        percentile(&paced, 95),
        paced[paced.len() - 1],
        // SAFETY: no arguments, no memory access.
        unsafe { libc::sched_getcpu() },
    );
    if worst > 1e-3 || jump > 0.35 || percentile(&paced, 95) > TICK_BUDGET_MS {
        return Err(format!("FAILED {report}"));
    }
    Ok(report)
}

#[cfg(feature = "ros")]
impl walking_action_manager::policy_runtime::Policy for CarefulWalk {
    const ACTION: walking_action_manager::motion::Action = walking_action_manager::motion::Action::CarefulWalk;
    const NODE: &'static str = "careful_walk_action_manager";
    const STATUS: &'static str = "/careful_walk_status";
    fn start(&mut self, input: &Input) -> Result<(), String> {
        CarefulWalk::start(self, input)
    }
    fn sample(&mut self, input: &Input, axes: Option<[f32; 4]>, _: f64) -> Result<Output, String> {
        CarefulWalk::sample(self, input, axes)
    }
    fn moving(&self) -> bool {
        CarefulWalk::moving(self)
    }
}
