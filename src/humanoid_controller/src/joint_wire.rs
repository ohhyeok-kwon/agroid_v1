//! Non-real-time adaptation of factory LowCmd joint units to motor wire units.
//! No trajectory planning here. Parallel-ankle impedance uses the factory local Jacobian.
#[path = "../../state_manager/src/ankle.rs"]
mod ankle;
use crate::{
    lowcmd::{LowCommand, Motor},
    motion_command::{Action, Command, Target, LEASE_MS},
    r1_profile::{JOINT_MOTORS, NATIVE_INDICES},
};

pub struct Encoder {
    ankle: ankle::AnkleMap,
}
impl Default for Encoder {
    fn default() -> Self {
        Self {
            ankle: ankle::AnkleMap::default(),
        }
    }
}
impl Encoder {
    /// Convert parallel-ankle motor speeds using the same Jacobian as positions.
    pub fn velocities(&self, raw: [i32; 26], mut dq: [f32; 26]) -> Option<[f32; 26]> {
        for (left, a, b) in [(true, 4, 5), (false, 10, 11)] {
            let qa = f64::from(raw[a]) * f64::from(JOINT_MOTORS[a].1)
                / f64::from(JOINT_MOTORS[a].2)
                / 5215.189;
            let qb = f64::from(raw[b]) * f64::from(JOINT_MOTORS[b].1)
                / f64::from(JOINT_MOTORS[b].2)
                / 5215.189;
            let (_, j) = self.ankle.position_jacobian(qa, qb, left)?;
            let velocity = [f64::from(dq[a]), f64::from(dq[b])];
            dq[a] = (j[0][0] * velocity[0] + j[0][1] * velocity[1]) as f32;
            dq[b] = (j[1][0] * velocity[0] + j[1][1] * velocity[1]) as f32;
        }
        Some(dq)
    }
    pub fn positions(&self, raw: [i32; 26], yaw_offset: Option<f32>) -> Option<[f32; 26]> {
        let mut q = std::array::from_fn(|i| {
            raw[i] as f32 * JOINT_MOTORS[i].1 / JOINT_MOTORS[i].2 / 5215.189
        });
        q[25] += yaw_offset?;
        for (left, a, b) in [(true, 4, 5), (false, 10, 11)] {
            let p = self
                .ankle
                .position(f64::from(q[a]), f64::from(q[b]), left)?;
            q[a] = p[0] as f32;
            q[b] = p[1] as f32;
        }
        Some(q)
    }
    pub fn encode(
        &self,
        low: &LowCommand,
        raw: [i32; 26],
        yaw_offset: Option<f32>,
        now: u64,
    ) -> Result<Command, &'static str> {
        low.validate()?;
        for (i, m) in low.motors.iter().enumerate() {
            if !NATIVE_INDICES.contains(&i)
                && (m.mode != 0
                    || m.q != 0.
                    || m.dq != 0.
                    || m.tau != 0.
                    || m.kp != 0.
                    || m.kd != 0.)
            {
                return Err("command targets a slot not configured on this R1");
            }
        }
        let mut values = NATIVE_INDICES.map(|i| low.motors[i]);
        let action = if values
            .iter()
            .all(|m| m.kp == 0. && m.kd == 0. && m.tau == 0. && m.dq == 0.)
        {
            Action::Zero
        } else if values
            .iter()
            .all(|m| m.kp == 0. && m.tau == 0. && m.dq == 0.)
        {
            Action::Damping
        } else {
            Action::Position
        };
        if action == Action::Damping && values.iter().any(|m| m.mode != 1) {
            return Err("damping requires all configured motors in FOC mode");
        }
        if action == Action::Position {
            for m in &values {
                if m.mode != 1 {
                    return Err("position control requires FOC mode");
                }
            }
            values[25].q -= yaw_offset
                .filter(|v| v.is_finite())
                .ok_or("head reference unavailable")?;
        }
        for (left, a, b) in [(true, 4, 5), (false, 10, 11)] {
            if action == Action::Zero {
                continue;
            }
            let qa = raw[a] as f64 / f64::from(JOINT_MOTORS[a].2) / 5215.189;
            let qb = raw[b] as f64 / f64::from(JOINT_MOTORS[b].2) / 5215.189;
            let (q, j) = match self.ankle.position_jacobian(qa, qb, left) {
                Some(mapped) => mapped,
                // Keep damping available even when a fall leaves the ankle outside
                // the position map. This uses the motor damping of the native watchdog.
                None if action == Action::Damping => continue,
                None => return Err("ankle measurement outside factory map"),
            };
            let desired = [values[a], values[b]];
            // J^T K J and J^T tau, like native 0x101a20e0. The off-diagonal
            // stiffness is represented by shifting the motor position reference.
            let error = [
                f64::from(desired[0].q) - q[0],
                f64::from(desired[1].q) - q[1],
            ];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det.abs() < 1e-9 {
                return Err("singular ankle Jacobian");
            }
            for k in 0..2 {
                let kp = (j[0][k].powi(2) * f64::from(desired[0].kp)
                    + j[1][k].powi(2) * f64::from(desired[1].kp))
                .clamp(0., 2000.);
                let kd = (j[0][k].powi(2) * f64::from(desired[0].kd)
                    + j[1][k].powi(2) * f64::from(desired[1].kd))
                .clamp(0., 25.);
                let shift = j[0][k] * f64::from(desired[0].kp) * error[0]
                    + j[1][k] * f64::from(desired[1].kp) * error[1];
                let dq = if k == 0 {
                    (j[1][1] * f64::from(desired[0].dq) - j[0][1] * f64::from(desired[1].dq)) / det
                } else {
                    (-j[1][0] * f64::from(desired[0].dq) + j[0][0] * f64::from(desired[1].dq)) / det
                };
                values[if k == 0 { a } else { b }] = Motor {
                    mode: 1,
                    q: ((if k == 0 { qa } else { qb }) + if kp > 1e-4 { shift / kp } else { 0. })
                        as f32,
                    dq: dq as f32,
                    tau: (j[0][k] * f64::from(desired[0].tau) + j[1][k] * f64::from(desired[1].tau))
                        as f32,
                    kp: kp as f32,
                    kd: kd as f32,
                    reserve: 0,
                };
            }
        }
        let mut targets = [Target::default(); 26];
        for (i, (m, (_, direction, ratio))) in values.iter().zip(JOINT_MOTORS).enumerate() {
            let (qs, ds, ts, ks, kds) = if i == 25 {
                (5215.189, 0.40743667, 256000., 1280000., 128000000.)
            } else {
                (5215.189, 10.185916, 2560., 12800., 51200.)
            };
            let q = (m.q * direction * ratio * qs).trunc();
            let dq = (m.dq * direction * ratio * ds).trunc();
            let tau = (m.tau * direction / ratio * ts).trunc();
            let kp = (m.kp / (ratio * ratio) * ks).trunc();
            let kd = (m.kd / (ratio * ratio) * kds).trunc();
            if !q.is_finite()
                || q.abs() > 20_000_000.
                || [dq, tau, kp, kd]
                    .iter()
                    .any(|x| !x.is_finite() || *x < i16::MIN as f32 || *x > i16::MAX as f32)
            {
                return Err("motor command exceeds wire range");
            }
            targets[i] = Target {
                q: q as i32,
                dq: dq as i16,
                tau: tau as i16,
                kp: kp as i16,
                kd: kd as i16,
            };
        }
        Ok(Command {
            epoch: u64::from(low.reserve[0]) | u64::from(low.reserve[1]) << 32,
            goal: u64::from(low.reserve[3]),
            sequence: now,
            issued_ms: now,
            expires_ms: now + LEASE_MS,
            action,
            targets,
        })
    }
}
