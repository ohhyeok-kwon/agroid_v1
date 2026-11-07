//! R1's 26 URDF joints in /joint_states order (also used by `joints.rs`).
//! Limits: state_manager/config/r1.urdf (Unitree R1).
pub const NAMES: [&str; 26] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
];
pub const LIMITS: [(f64, f64); 26] = [
    (-2.932153143, 2.548180708),
    (-1.047197551, 1.745329252),
    (-2.7402, 2.7402),
    (-0.174532925, 2.42600766),
    (-0.87266, 0.57596),
    (-0.2618, 0.2618),
    (-2.932153143, 2.548180708),
    (-1.745329252, 1.047197551),
    (-2.7402, 2.7402),
    (-0.17453, 2.42600766),
    (-0.87266, 0.57596),
    (-0.261799388, 0.261799388),
    (-0.5236, 0.5236),
    (-2.618, 2.618),
    (-3.1416, 2.0944),
    (-0.22689, 2.4784),
    (-1.9199, 1.9199),
    (-0.97564, 2.1852),
    (-1.9199, 1.9199),
    (-3.1416, 2.0944),
    (-2.47849, 0.2268),
    (-1.9199, 1.9199),
    (-0.97564, 2.1852),
    (-1.9199, 1.9199),
    (-0.62832, 0.62832),
    (-2.0071, 2.0071),
];

pub fn in_range(index: usize, q: f64) -> bool {
    q.is_finite() && q >= LIMITS[index].0 && q <= LIMITS[index].1
}

/// Reject partial, duplicate, unknown or out-of-range joint feedback; reorder by name.
pub fn positions(names: &[String], values: &[f64]) -> Result<[f64; 26], String> {
    if names.len() != 26 || values.len() != 26 {
        return Err("complete 26-joint feedback required".into());
    }
    let mut result = [0.0; 26];
    let mut seen = [false; 26];
    for (name, &q) in names.iter().zip(values) {
        let i = NAMES
            .iter()
            .position(|n| n == name)
            .ok_or("unknown joint")?;
        if seen[i] || !in_range(i, q) {
            return Err(format!("duplicate or invalid joint: {name}"));
        }
        seen[i] = true;
        result[i] = q;
    }
    Ok(result)
}
