use crate::ankle::AnkleMap;
use serde::Deserialize;

use crate::hardware::JOINT_MOTORS;
/// The same URDF order as the motion contract.
pub use crate::motion::profile::NAMES;
/// Head pitch and yaw are only shown once the head reference is verified.
const HEAD: [usize; 2] = [24, 25];
/// Parallel ankles: (left side, motor A slot -> pitch, motor B slot -> roll).
const ANKLES: [(bool, usize, usize); 2] = [(true, 4, 5), (false, 10, 11)];

#[derive(Deserialize)]
pub struct MotorStatus {
    pub usb_connected: bool,
    /// Display stamp (TF header). Ages and order use `received_mono_ms` when present.
    pub received_unix_ms: Option<u64>,
    #[serde(default)]
    pub received_mono_ms: Option<u64>,
    pub age_ms: Option<u64>,
    data: Option<MotorData>,
    initialization: Option<Initialization>,
}
#[derive(Deserialize)]
struct MotorData {
    motors: Vec<MotorSlot>,
}
#[derive(Deserialize)]
struct MotorSlot {
    bus: u8,
    id: u8,
    data: Option<Sample>,
}
#[derive(Deserialize)]
struct Sample {
    q_raw: i32,
    timeout: bool,
    error_code: u32,
}
#[derive(Deserialize)]
struct Initialization {
    phase: String,
    head_reference_verified: bool,
    head: Option<Head>,
}
#[derive(Deserialize)]
struct Head {
    yaw_offset_rad: Option<f64>,
}

pub struct JointFrame {
    /// Wall-clock receipt, for the ROS header only.
    pub stamp_ms: u64,
    /// Ordering key: CLOCK_MONOTONIC receipt when the controller sends it, else `stamp_ms`.
    pub order_ms: u64,
    /// `None`: no valid measurement for this joint in this frame; it is not published.
    pub positions: [Option<f64>; 26],
    /// Why joints are missing. Stable text (no measured values) so it can be logged on change.
    pub problems: Vec<String>,
}

impl JointFrame {
    pub fn valid(&self) -> Vec<(usize, f64)> {
        (0..NAMES.len())
            .filter_map(|i| self.positions[i].map(|q| (i, q)))
            .collect()
    }
    pub fn remove(&mut self, joints: &[usize], problem: String) {
        for &i in joints {
            self.positions[i] = None;
        }
        self.problems.push(problem);
    }
}

/// Same sample again (Transient Local re-delivery) or slightly out of order. A jump back
/// of 1 s or more is a wall-clock adjustment: accept it instead of stalling until it passes.
pub fn is_repeat(stamp_ms: u64, last_ms: u64) -> bool {
    stamp_ms == last_ms || (stamp_ms < last_ms && last_ms - stamp_ms < 1000)
}

impl MotorStatus {
    /// Errors only when there is no fresh USB measurement at all. A faulty motor, an
    /// unverified head reference or an unmappable ankle removes just the affected joints.
    /// `now_ms`: wall clock (legacy status without `received_mono_ms`); `now_mono`: CLOCK_MONOTONIC.
    pub fn convert(
        &self,
        ankle: &AnkleMap,
        now_ms: u64,
        now_mono: u64,
    ) -> Result<JointFrame, String> {
        let stamp = self
            .received_unix_ms
            .ok_or("no motor measurement timestamp")?;
        let (order, now) = match self.received_mono_ms {
            Some(mono) => (mono, now_mono),
            None => (stamp, now_ms),
        };
        if !self.usb_connected
            || self.age_ms.is_none_or(|age| age > 3000)
            || now.saturating_sub(order) > 3000
            || order.saturating_sub(now) > 1000
        {
            return Err("motor measurement is disconnected or stale".into());
        }
        let data = self.data.as_ref().ok_or("no motor data")?;
        let mut frame = JointFrame {
            stamp_ms: stamp,
            order_ms: order,
            positions: [None; 26],
            problems: Vec::new(),
        };
        for (i, &(address, direction, ratio)) in JOINT_MOTORS.iter().enumerate() {
            let (bus, id) = (address.bus, address.id);
            let mut slots = data.motors.iter().filter(|m| m.bus == bus && m.id == id);
            let problem = match (slots.next().and_then(|m| m.data.as_ref()), slots.next()) {
                (_, Some(_)) => "duplicate".to_string(),
                (None, None) => "missing".to_string(),
                (Some(s), None) if s.timeout => "timeout".to_string(),
                (Some(s), None) if s.error_code != 0 => format!("error {}", s.error_code),
                (Some(s), None) => {
                    // Native position conversion is single precision, after direction.
                    frame.positions[i] =
                        Some(((s.q_raw as f32 * direction / ratio) / 5215.189) as f64);
                    continue;
                }
            };
            frame
                .problems
                .push(format!("{}: motor {bus}/{id} {problem}", NAMES[i]));
        }
        match self.head_yaw_offset() {
            Ok(offset) => {
                if let Some(q) = frame.positions[25].as_mut() {
                    *q += offset;
                }
            }
            Err(problem) => frame.remove(&HEAD, format!("head joints: {problem}")),
        }
        for (left, a, b) in ANKLES {
            let side = if left { "left" } else { "right" };
            match (frame.positions[a], frame.positions[b]) {
                (Some(motor_a), Some(motor_b)) => match ankle.position(motor_a, motor_b, left) {
                    Some([pitch, roll]) => {
                        frame.positions[a] = Some(pitch);
                        frame.positions[b] = Some(roll);
                    }
                    None => frame.remove(&[a, b], format!("{side} ankle: outside the factory map")),
                },
                // Pitch and roll both need motors A and B; the motor problem is already listed.
                _ => {
                    frame.positions[a] = None;
                    frame.positions[b] = None;
                }
            }
        }
        Ok(frame)
    }

    fn head_yaw_offset(&self) -> Result<f64, &'static str> {
        let init = self
            .initialization
            .as_ref()
            .ok_or("no motor initialization status")?;
        if init.phase != "ready" || !init.head_reference_verified {
            return Err("waiting for verified head initialization");
        }
        init.head
            .as_ref()
            .and_then(|h| h.yaw_offset_rad)
            .filter(|v| v.is_finite())
            .ok_or("no verified head yaw offset")
    }
}
