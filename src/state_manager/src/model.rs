use crate::joints::{JointFrame, NAMES};
use roxmltree::{Document, Node};
use std::collections::HashSet;

pub const URDF: &str = include_str!("../config/r1.urdf");
/// The manufacturer's URDF rounds the calibrated mechanical limits.
pub const MEASUREMENT_TOLERANCE_RAD: f64 = 0.00175;

#[derive(Clone, Debug)]
pub struct Transform {
    pub parent: String,
    pub child: String,
    pub translation: [f64; 3],
    /// ROS order x,y,z,w.
    pub rotation: [f64; 4],
}
pub struct Joint {
    pub transform: Transform,
    pub axis: [f64; 3],
    pub index: Option<usize>,
    pub limit: Option<[f64; 2]>,
}
pub struct Model {
    pub joints: Vec<Joint>,
}

pub fn multiply(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [x, y, z, w] = a;
    let [u, v, s, t] = b;
    [
        w * u + x * t + y * s - z * v,
        w * v - x * s + y * t + z * u,
        w * s + x * v - y * u + z * t,
        w * t - x * u - y * v - z * s,
    ]
}
fn vector(n: Option<Node<'_, '_>>, attr: &str, default: [f64; 3]) -> Result<[f64; 3], String> {
    let Some(text) = n.and_then(|n| n.attribute(attr)) else {
        return Ok(default);
    };
    let v: Vec<f64> = text
        .split_whitespace()
        .map(|v| v.parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    if v.len() != 3 || v.iter().any(|v| !v.is_finite()) {
        return Err("invalid URDF vector".into());
    }
    Ok([v[0], v[1], v[2]])
}
impl Model {
    pub fn load() -> Result<Self, String> {
        let doc = Document::parse(URDF).map_err(|e| e.to_string())?;
        let robot = doc.root_element();
        let mut joints = Vec::new();
        let mut children = HashSet::new();
        let mut indices = HashSet::new();
        let links: HashSet<_> = robot
            .children()
            .filter(|n| n.has_tag_name("link"))
            .filter_map(|n| n.attribute("name"))
            .collect();
        for n in robot.children().filter(|n| n.has_tag_name("joint")) {
            let child_node = |tag| n.children().find(|c| c.has_tag_name(tag));
            let link = |tag| {
                child_node(tag)
                    .and_then(|n| n.attribute("link"))
                    .map(str::to_string)
                    .ok_or("missing URDF link")
            };
            let parent = link("parent")?;
            let child = link("child")?;
            if !links.contains(parent.as_str())
                || !links.contains(child.as_str())
                || !children.insert(child.clone())
            {
                return Err("invalid URDF tree".into());
            }
            let translation = vector(child_node("origin"), "xyz", [0.; 3])?;
            let [r, p, y] = vector(child_node("origin"), "rpy", [0.; 3])?;
            let rotation = multiply(
                multiply(
                    [0., 0., (y / 2.).sin(), (y / 2.).cos()],
                    [0., (p / 2.).sin(), 0., (p / 2.).cos()],
                ),
                [(r / 2.).sin(), 0., 0., (r / 2.).cos()],
            );
            let index = match n.attribute("type") {
                Some("fixed") => None,
                Some("revolute") => Some(
                    NAMES
                        .iter()
                        .position(|name| Some(*name) == n.attribute("name"))
                        .ok_or("unknown R1 joint")?,
                ),
                _ => return Err("unsupported URDF joint".into()),
            };
            if let Some(i) = index {
                if !indices.insert(i) {
                    return Err("duplicate R1 joint".into());
                }
            }
            let mut axis = vector(child_node("axis"), "xyz", [1., 0., 0.])?;
            let norm = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
            if index.is_some() && norm < 1e-12 {
                return Err("zero URDF axis".into());
            }
            if norm > 0. {
                for value in &mut axis {
                    *value /= norm;
                }
            }
            let limit = if index.is_some() {
                let l = child_node("limit").ok_or("no joint limits")?;
                Some([
                    l.attribute("lower")
                        .ok_or("no lower limit")?
                        .parse()
                        .map_err(|_| "bad limit")?,
                    l.attribute("upper")
                        .ok_or("no upper limit")?
                        .parse()
                        .map_err(|_| "bad limit")?,
                ])
            } else {
                None
            };
            joints.push(Joint {
                transform: Transform {
                    parent,
                    child,
                    translation,
                    rotation,
                },
                axis,
                index,
                limit,
            });
        }
        if indices.len() != 26
            || links.len() != joints.len() + 1
            || children.contains("pelvis_link")
        {
            return Err("incomplete R1 URDF".into());
        }
        // Also reject disconnected components or cycles.
        let mut reachable = HashSet::from(["pelvis_link".to_string()]);
        loop {
            let before = reachable.len();
            for j in &joints {
                if reachable.contains(&j.transform.parent) {
                    reachable.insert(j.transform.child.clone());
                }
            }
            if before == reachable.len() {
                break;
            }
        }
        if reachable.len() != links.len() {
            return Err("disconnected R1 URDF".into());
        }
        Ok(Self { joints })
    }
    pub fn fixed(&self) -> Vec<Transform> {
        self.joints
            .iter()
            .filter(|j| j.index.is_none())
            .map(|j| j.transform.clone())
            .collect()
    }
    /// Removes joints outside the URDF limits from the frame; a measurement is never clamped.
    pub fn check_limits(&self, frame: &mut JointFrame) {
        for j in &self.joints {
            let (Some(i), Some([low, high])) = (j.index, j.limit) else {
                continue;
            };
            // URDF limits are rounded; allow 0.1 degree.
            if frame.positions[i].is_some_and(|q| {
                !q.is_finite()
                    || q < low - MEASUREMENT_TOLERANCE_RAD
                    || q > high + MEASUREMENT_TOLERANCE_RAD
            }) {
                frame.remove(&[i], format!("{}: outside URDF limits", NAMES[i]));
            }
        }
    }
    /// Parent -> child transforms of the joints that have a valid position in this frame.
    pub fn moving(&self, frame: &JointFrame) -> Vec<Transform> {
        self.joints
            .iter()
            .filter_map(|j| {
                let q = frame.positions[j.index?]?;
                let mut t = j.transform.clone();
                let s = (q / 2.).sin();
                t.rotation = multiply(
                    t.rotation,
                    [j.axis[0] * s, j.axis[1] * s, j.axis[2] * s, (q / 2.).cos()],
                );
                Some(t)
            })
            .collect()
    }
}
