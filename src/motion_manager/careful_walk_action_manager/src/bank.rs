//! Reference bank and its player.
//!
//! The reference generator (calm v4.3s) needs MuJoCo and Python, which the R1 does not carry. Its output is baked
//! offline into `calm43s_bank.bin`; the player splices the baked frames so that the played reference is the
//! generator's own output for the same commands (checked frame by frame against the generator off the robot).
//!
//! File (little endian): `R1BANK01`, u32 header length, header JSON, then float32 frames of 28 values =
//! 24 joints in policy order + root quaternion w,x,y,z relative to the heading of the segment the frame belongs to.
//! A command has a `start` (from standing to the steady gait) and a `cycle` (two steps, repeated); a `branch` at
//! frame `at` of either holds the `stop` that replaces the parent from that frame on and ends in standing.
//!
//! Rule (50 Hz ticks, `delay_ticks` = 5 as on the Operation Center; reference implementation `bank_play.py`):
//! a command given at tick n never changes a frame before tick n + 5. Standing + command: its `start` begins at
//! tick n + 5. Walking + anything else: the first branch at or after tick n + 5 is taken and stays taken; a stop's
//! last frame is the first standing frame, and a command given meanwhile starts there (or at n + 5 if later).
use serde::Deserialize;

pub const JOINTS: usize = 24;
const FRAME: usize = JOINTS + 4;
/// Frames of one control tick n: ticks n-1 ..= n+46 (ten frames 5 ticks apart and their velocity neighbours).
pub const WINDOW: usize = 48;
const MAGIC: &[u8; 8] = b"R1BANK01";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct Span {
    offset: usize,
    frames: usize,
    #[serde(default)]
    dpsi: f64,
}
#[derive(Deserialize)]
struct BranchHeader {
    at: usize,
    stop: Span,
}
#[derive(Deserialize)]
struct SegmentHeader {
    offset: usize,
    frames: usize,
    dpsi: f64,
    branches: Vec<BranchHeader>,
}
#[derive(Deserialize)]
struct CommandHeader {
    name: String,
    twist: [f64; 3],
    start: SegmentHeader,
    cycle: SegmentHeader,
}
#[derive(Deserialize)]
struct Header {
    version: u32,
    rate_hz: u32,
    delay_ticks: usize,
    joint_names: Vec<String>,
    total_frames: usize,
    stand: Span,
    commands: Vec<CommandHeader>,
}

struct Segment {
    span: Span,
    stops: Vec<Option<Span>>,
}
pub struct Command {
    pub name: String,
    /// Forward m/s, left m/s, counter-clockwise rad/s.
    pub twist: [f64; 3],
    start: Segment,
    cycle: Segment,
}
pub struct Bank {
    frames: &'static [u8],
    pub delay_ticks: usize,
    stand: usize,
    pub commands: Vec<Command>,
}

impl Bank {
    /// `joint_names`: the tracker's policy order; the bank must have been baked in exactly that order.
    pub fn load(bytes: &'static [u8], joint_names: &[String]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[..8] != MAGIC {
            return Err("reference bank: bad magic".into());
        }
        let length = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let body = bytes.get(12..12 + length).ok_or("reference bank: short header")?;
        let header: Header = serde_json::from_slice(body).map_err(|e| format!("reference bank: {e}"))?;
        let frames = &bytes[12 + length..];
        if header.version != 1 || header.rate_hz != 50 {
            return Err("reference bank: unsupported version or rate".into());
        }
        if header.joint_names != joint_names {
            return Err("reference bank: joint order differs from the tracker".into());
        }
        if header.total_frames == 0 || frames.len() != header.total_frames * FRAME * 4 {
            return Err("reference bank: frame count does not match the file".into());
        }
        if header.delay_ticks + 2 > WINDOW || header.commands.is_empty() {
            return Err("reference bank: bad delay or no command".into());
        }
        let total = header.total_frames;
        let inside = |s: &Span| s.frames > 0 && s.offset + s.frames <= total && s.dpsi.is_finite();
        if !inside(&header.stand) {
            return Err("reference bank: stand frame out of range".into());
        }
        let segment = |h: SegmentHeader, name: &str| -> Result<Segment, String> {
            let span = Span { offset: h.offset, frames: h.frames, dpsi: h.dpsi };
            if !inside(&span) {
                return Err(format!("reference bank: {name} out of range"));
            }
            let mut stops = vec![None; span.frames];
            for branch in h.branches {
                let slot = stops
                    .get_mut(branch.at)
                    .filter(|slot| slot.is_none() && inside(&branch.stop))
                    .ok_or_else(|| format!("reference bank: bad branch in {name}"))?;
                *slot = Some(branch.stop);
            }
            Ok(Segment { span, stops })
        };
        let mut commands = Vec::with_capacity(header.commands.len());
        for c in header.commands {
            if c.twist.iter().any(|v| !v.is_finite()) || c.twist.iter().filter(|v| **v != 0.).count() != 1 {
                return Err(format!("reference bank: {} is not a single-axis command", c.name));
            }
            commands.push(Command {
                start: segment(c.start, &c.name)?,
                cycle: segment(c.cycle, &c.name)?,
                name: c.name,
                twist: c.twist,
            });
        }
        let bank = Self { frames, delay_ticks: header.delay_ticks, stand: header.stand.offset, commands };
        for frame in 0..total {
            let joints = bank.joints(frame);
            let norm = bank.quaternion(frame).iter().map(|v| v * v).sum::<f64>();
            if joints.iter().any(|v| !v.is_finite()) || !(0.98..=1.02).contains(&norm) {
                return Err(format!("reference bank: bad frame {frame}"));
            }
        }
        Ok(bank)
    }
    fn value(&self, frame: usize, index: usize) -> f32 {
        let at = (frame * FRAME + index) * 4;
        f32::from_le_bytes(self.frames[at..at + 4].try_into().unwrap())
    }
    fn joints(&self, frame: usize) -> [f32; JOINTS] {
        std::array::from_fn(|i| self.value(frame, i))
    }
    fn quaternion(&self, frame: usize) -> [f64; 4] {
        std::array::from_fn(|i| f64::from(self.value(frame, JOINTS + i)))
    }
    fn segment(&self, command: usize, cycle: bool) -> &Segment {
        let c = &self.commands[command];
        if cycle {
            &c.cycle
        } else {
            &c.start
        }
    }
}

/// Where the next planned frame comes from. `psi`: heading of the segment's reference direction (rad).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cursor {
    Stand { psi: f64 },
    Walk { command: usize, cycle: bool, index: usize, psi: f64 },
    Stop { span: Span, index: usize, psi: f64 },
}
#[derive(Clone, Copy)]
struct Planned {
    /// Cursor before this frame was chosen: re-planning from this slot restarts here.
    before: Cursor,
    frame: usize,
    psi: f64,
    standing: bool,
}

pub struct Player {
    bank: Bank,
    /// Slot i is the frame of tick n - 1 + i.
    plan: [Planned; WINDOW],
    tail: Cursor,
    tick: u64,
    told: Option<usize>,
    stop_pending: bool,
    /// Command waiting for standing, and the first tick it may start at.
    queued: Option<(usize, u64)>,
}

impl Player {
    pub fn new(bank: Bank) -> Self {
        let stand = Planned { before: Cursor::Stand { psi: 0. }, frame: bank.stand, psi: 0., standing: true };
        Self {
            bank,
            plan: [stand; WINDOW],
            tail: Cursor::Stand { psi: 0. },
            tick: 0,
            told: None,
            stop_pending: false,
            queued: None,
        }
    }
    pub fn bank(&self) -> &Bank {
        &self.bank
    }
    /// Standing at tick 0 with heading 0.
    pub fn reset(&mut self) {
        let stand = Planned { before: Cursor::Stand { psi: 0. }, frame: self.bank.stand, psi: 0., standing: true };
        self.plan = [stand; WINDOW];
        self.tail = stand.before;
        self.tick = 0;
        self.told = None;
        self.stop_pending = false;
        self.queued = None;
    }
    /// The command of the current tick: a bank command index, or None for standing.
    pub fn tell(&mut self, want: Option<usize>) {
        if want == self.told || want.is_some_and(|c| c >= self.bank.commands.len()) {
            return;
        }
        self.told = want;
        let slot = self.bank.delay_ticks + 1;
        let effective = self.tick + self.bank.delay_ticks as u64;
        let cursor = self.plan[slot].before;
        // Still walking at the effective tick: stop first (a stop told earlier is never taken back).
        self.stop_pending = matches!(cursor, Cursor::Walk { .. });
        self.queued = want.map(|command| (command, effective));
        self.tail = cursor;
        for i in slot..WINDOW {
            self.plan[i] = self.next(self.tick + i as u64 - 1);
        }
    }
    /// Tick n -> n + 1.
    pub fn advance(&mut self) {
        self.tick += 1;
        self.plan.copy_within(1.., 0);
        self.plan[WINDOW - 1] = self.next(self.tick + WINDOW as u64 - 2);
    }
    /// True once every planned frame is standing and no command is given or waiting.
    pub fn standing(&self) -> bool {
        self.told.is_none() && self.queued.is_none() && self.plan.iter().all(|p| p.standing)
    }
    fn next(&mut self, tick: u64) -> Planned {
        let before = self.tail;
        loop {
            match self.tail {
                Cursor::Stand { psi } => {
                    if let Some((command, _)) = self.queued.filter(|(_, earliest)| tick >= *earliest) {
                        self.queued = None;
                        self.tail = Cursor::Walk { command, cycle: false, index: 0, psi };
                        continue;
                    }
                    return Planned { before, frame: self.bank.stand, psi, standing: true };
                }
                Cursor::Walk { command, cycle, index, psi } => {
                    let segment = self.bank.segment(command, cycle);
                    if let Some(span) = segment.stops[index].filter(|_| self.stop_pending) {
                        self.stop_pending = false;
                        self.tail = Cursor::Stop { span, index: 0, psi };
                        continue;
                    }
                    let frame = segment.span.offset + index;
                    self.tail = if index + 1 == segment.span.frames {
                        Cursor::Walk { command, cycle: true, index: 0, psi: psi + segment.span.dpsi }
                    } else {
                        Cursor::Walk { command, cycle, index: index + 1, psi }
                    };
                    return Planned { before, frame, psi, standing: false };
                }
                Cursor::Stop { span, index, psi } => {
                    if index + 1 >= span.frames {
                        // The stop's last frame is the first standing frame.
                        self.tail = Cursor::Stand { psi: psi + span.dpsi };
                        continue;
                    }
                    self.tail = Cursor::Stop { span, index: index + 1, psi };
                    return Planned { before, frame: span.offset + index, psi, standing: false };
                }
            }
        }
    }
}

impl crate::tracker::Reference for Player {
    fn joints(&self, ahead: isize) -> [f32; JOINTS] {
        self.bank.joints(self.plan[(ahead + 1) as usize].frame)
    }
    /// In the frame the walk started in: the segment's heading turned onto the stored frame.
    fn quaternion(&self, ahead: isize) -> [f64; 4] {
        let planned = &self.plan[(ahead + 1) as usize];
        let [w, x, y, z] = self.bank.quaternion(planned.frame);
        let (s, c) = (0.5 * planned.psi).sin_cos();
        [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
    }
}
