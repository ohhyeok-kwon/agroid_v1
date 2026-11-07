//! Native gamepad scaling (0x103a4418) and 20 ms slew (0x103a4784).
use crate::Parameters;
#[derive(Default)]
pub struct Command {
    pub high_speed: bool,
    pub velocity: [f32; 3],
    buttons: u16,
}
impl Command {
    pub fn buttons(&mut self, keys: u16) {
        const R2: u16 = 1 << 4;
        let pressed = keys & !self.buttons;
        if keys & self.buttons & R2 != 0 {
            if pressed & (1 << 12) != 0 {
                self.high_speed = true;
            }
            if pressed & (1 << 14) != 0 {
                self.high_speed = false;
            }
        }
        self.buttons = keys;
    }
    pub fn target(&self, [lx, ly, rx, _]: [f32; 4], p: &Parameters) -> [f32; 3] {
        let forward = if self.high_speed {
            p.high_speed_forward_vx_limit
        } else {
            p.low_speed_forward_vx_limit
        };
        let mut vx = ly * forward;
        let mut vy = -lx * p.vy_limit;
        if vx.abs() < 0.05 {
            vx = 0.;
        }
        if vy.abs() < 0.05 {
            vy = 0.;
        }
        if vx < 0. {
            vx = if vx <= -0.3 {
                ly * p.reverse_vx_limit
            } else {
                0.
            };
        }
        let limit = if vx < 0. { p.reverse_vx_limit } else { forward };
        if vy.abs() > p.lateral_vy_zero_vx_ratio * p.vy_limit {
            vx = 0.;
        } else if limit > 1e-6 && vx.abs() > p.longitudinal_vx_zero_vy_ratio * limit {
            vy = 0.;
        }
        let mut wz = -rx * yaw_limit(vx, vy, p);
        if wz.abs() < 0.05 {
            wz = 0.;
        }
        [vx, vy, wz]
    }
    pub fn moving(&self, axes: [f32; 4], p: &Parameters) -> bool {
        self.target(axes, p).iter().map(|x| x * x).sum::<f32>() > 0.0001
    }
    pub fn step(&mut self, axes: [f32; 4], p: &Parameters) -> [f32; 3] {
        let mut target = self.target(axes, p);
        for i in 0..3 {
            let previous = self.velocity[i];
            let (up, down, reverse_up, reverse_down) = match i {
                0 => (
                    p.forward_vx_acc_up_delta,
                    p.forward_vx_acc_down_delta,
                    p.reverse_vx_acc_up_delta,
                    p.reverse_vx_acc_down_delta,
                ),
                1 => (
                    p.vy_acc_up_delta,
                    p.vy_acc_down_delta,
                    p.vy_acc_up_delta,
                    p.vy_acc_down_delta,
                ),
                _ => (
                    p.yaw_acc_up_delta,
                    p.yaw_acc_down_delta,
                    p.yaw_acc_up_delta,
                    p.yaw_acc_down_delta,
                ),
            };
            if previous >= 0. {
                target[i] = target[i].clamp(previous - down, previous + up);
            }
            if previous <= 0. {
                target[i] = target[i].clamp(previous - reverse_up, previous + reverse_down);
            }
        }
        let yaw = yaw_limit(target[0], target[1], p);
        target[2] = target[2].clamp(-yaw, yaw);
        self.velocity = target;
        target
    }
}
fn yaw_limit(vx: f32, vy: f32, p: &Parameters) -> f32 {
    if vx.abs() > 0.0001 || vy.abs() > 0.0001 {
        p.loco_wz_limit
    } else {
        p.pure_wz_limit
    }
}
