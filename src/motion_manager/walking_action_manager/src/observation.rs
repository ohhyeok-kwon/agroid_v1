//! Native 81-value observation, ten frames oldest-first (0x103a7920/0x103a3e08).
use crate::Parameters;
#[derive(Clone, Debug)]
pub struct Input {
    pub q: [f32; 26],
    pub dq: [f32; 26],
    /// Manufacturer LowState quaternion: w,x,y,z; gyro is rad/s.
    pub quaternion: [f32; 4],
    pub gyroscope: [f32; 3],
}
impl Input {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .q
            .iter()
            .chain(&self.dq)
            .chain(&self.quaternion)
            .chain(&self.gyroscope)
            .any(|v| !v.is_finite())
        {
            return Err("non-finite walking input".into());
        }
        let norm = self.quaternion.iter().map(|v| v * v).sum::<f32>();
        if norm < 1e-6 {
            return Err("waiting for IMU quaternion".into());
        }
        Ok(())
    }
    pub fn upright(&self) -> bool {
        let [w, x, y, z] = self.quaternion;
        let roll = (2. * (w * x + y * z)).atan2(1. - 2. * (x * x + y * y));
        let pitch = (2. * (w * y - z * x)).clamp(-1., 1.).asin();
        roll.abs() <= 1. && pitch.abs() <= 1.
    }
    pub fn gravity(&self) -> [f32; 3] {
        let [w, x, y, z] = self.quaternion;
        [
            2. * (w * y - x * z),
            -2. * (w * x + y * z),
            1. - 2. * (w * w + z * z),
        ]
    }
}
pub struct Observation {
    pub history: [f32; 810],
    pub previous_action: [f32; 24],
}
impl Default for Observation {
    fn default() -> Self {
        Self {
            history: [0.; 810],
            previous_action: [0.; 24],
        }
    }
}
impl Observation {
    fn frame(&self, input: &Input, velocity: [f32; 3], p: &Parameters) -> [f32; 81] {
        let mut frame = [0.; 81];
        frame[..3].copy_from_slice(&input.gyroscope);
        frame[3..6].copy_from_slice(&input.gravity());
        frame[6..9].copy_from_slice(&velocity);
        for (joint, &policy) in p.action_idx.iter().enumerate() {
            frame[9 + policy] = input.q[joint] - p.default_joint_pos[policy];
            frame[33 + policy] = input.dq[joint];
        }
        frame[57..].copy_from_slice(&self.previous_action);
        frame
    }
    pub fn reset(&mut self, input: &Input, velocity: [f32; 3], p: &Parameters) {
        self.previous_action = [0.; 24];
        let frame = self.frame(input, velocity, p);
        for chunk in self.history.chunks_exact_mut(81) {
            chunk.copy_from_slice(&frame);
        }
    }
    pub fn update(&mut self, input: &Input, velocity: [f32; 3], p: &Parameters) {
        let frame = self.frame(input, velocity, p);
        self.history.copy_within(81.., 0);
        self.history[729..].copy_from_slice(&frame);
    }
}
