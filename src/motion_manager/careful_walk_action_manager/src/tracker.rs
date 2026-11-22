//! SONIC-R1 tracker input, line for line with the training-side deploy harness
//! (`sonic_g1.DeployR1Native.control_tick`, reference in R1 joint space).
//!
//! `obs` [1320] = encoder input [540] followed by the decoder's proprioception [780].
//! Encoder: ten reference frames 0, 5, .. 45 ticks ahead. The harness stacks the ten joint positions and then the
//! ten joint velocities, cuts that block into ten rows of 48 and appends to row i the first two columns of the
//! rotation "reference root seen from the robot's pelvis" of frame i (6 values).
//! Proprioception: ten ticks oldest first, term by term: gyro, q - default, dq, previous action, gravity in the
//! pelvis frame. Ticks before the start are zero.
use crate::bank::JOINTS;

pub const ENCODER: usize = 540;
pub const OBS: usize = 1320;
const HISTORY: usize = 10;
const DT: f64 = 0.02;

/// The reference the tracker follows, frame by frame relative to the current tick (-1 ..= 46).
pub trait Reference {
    /// Joint positions in policy order (rad).
    fn joints(&self, ahead: isize) -> [f32; JOINTS];
    /// Root quaternion w,x,y,z.
    fn quaternion(&self, ahead: isize) -> [f64; 4];
}

type Quat = [f64; 4];
fn mul(a: Quat, b: Quat) -> Quat {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn conj(q: Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}
/// Twist about the vertical axis (not the Euler yaw), as the trained deploy code takes it.
fn heading(q: Quat) -> Quat {
    let n = (q[0] * q[0] + q[3] * q[3]).sqrt();
    if n > 1e-9 {
        [q[0] / n, 0., 0., q[3] / n]
    } else {
        [1., 0., 0., 0.]
    }
}

/// One tick of measured state in policy joint order.
pub struct Sample {
    /// Joint position minus the default pose.
    pub q: [f32; JOINTS],
    pub dq: [f32; JOINTS],
    pub gyro: [f32; 3],
    /// Pelvis IMU quaternion w,x,y,z.
    pub quaternion: [f32; 4],
}

pub struct Tracker {
    gyro: [[f32; 3]; HISTORY],
    q: [[f32; JOINTS]; HISTORY],
    dq: [[f32; JOINTS]; HISTORY],
    action: [[f32; JOINTS]; HISTORY],
    gravity: [[f32; 3]; HISTORY],
    /// Raw network output of the previous tick.
    pub last_action: [f32; JOINTS],
    /// Fixed turn that lines the reference's start heading up with the robot's.
    turn: Option<Quat>,
}
impl Default for Tracker {
    fn default() -> Self {
        Self {
            gyro: [[0.; 3]; HISTORY],
            q: [[0.; JOINTS]; HISTORY],
            dq: [[0.; JOINTS]; HISTORY],
            action: [[0.; JOINTS]; HISTORY],
            gravity: [[0.; 3]; HISTORY],
            last_action: [0.; JOINTS],
            turn: None,
        }
    }
}
fn push<const N: usize>(history: &mut [[f32; N]; HISTORY], newest: [f32; N]) {
    history.copy_within(1.., 0);
    history[HISTORY - 1] = newest;
}
impl Tracker {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    /// Log this tick's state and fill `obs` for the frames the reference holds for this tick.
    pub fn observe(&mut self, sample: &Sample, player: &impl Reference, obs: &mut [f32; OBS]) -> Result<(), String> {
        let norm = sample.quaternion.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
        if !(norm > 1e-3) {
            return Err("waiting for IMU quaternion".into());
        }
        let base: Quat = sample.quaternion.map(|v| f64::from(v) / norm);
        let [w, x, y, z] = base;
        push(&mut self.gyro, sample.gyro);
        push(&mut self.q, sample.q);
        push(&mut self.dq, sample.dq);
        push(&mut self.action, self.last_action);
        push(
            &mut self.gravity,
            [
                (2. * (w * y - x * z)) as f32,
                (-2. * (y * z + w * x)) as f32,
                (2. * (x * x + y * y) - 1.) as f32,
            ],
        );
        let turn = *self
            .turn
            .get_or_insert_with(|| mul(heading(base), conj(heading(player.quaternion(0)))));
        let mut block = [0f32; 2 * HISTORY * JOINTS];
        let mut rotation = [[0f32; 6]; HISTORY];
        for k in 0..HISTORY {
            let ahead = 5 * k as isize;
            let (now, before, after) = (player.joints(ahead), player.joints(ahead - 1), player.joints(ahead + 1));
            for j in 0..JOINTS {
                block[k * JOINTS + j] = now[j];
                block[(HISTORY + k) * JOINTS + j] =
                    ((f64::from(after[j]) - f64::from(before[j])) / (2. * DT)) as f32;
            }
            let [w, x, y, z] = mul(conj(base), mul(turn, player.quaternion(ahead)));
            rotation[k] = [
                1. - 2. * (y * y + z * z),
                2. * (x * y - w * z),
                2. * (x * y + w * z),
                1. - 2. * (x * x + z * z),
                2. * (x * z - w * y),
                2. * (y * z + w * x),
            ]
            .map(|v| v as f32);
        }
        for k in 0..HISTORY {
            obs[54 * k..54 * k + 48].copy_from_slice(&block[48 * k..48 * k + 48]);
            obs[54 * k + 48..54 * k + 54].copy_from_slice(&rotation[k]);
        }
        let mut at = ENCODER;
        let mut put = |values: &[f32]| {
            obs[at..at + values.len()].copy_from_slice(values);
            at += values.len();
        };
        self.gyro.iter().for_each(|v| put(v));
        self.q.iter().for_each(|v| put(v));
        self.dq.iter().for_each(|v| put(v));
        self.action.iter().for_each(|v| put(v));
        self.gravity.iter().for_each(|v| put(v));
        debug_assert_eq!(at, OBS);
        Ok(())
    }
}
