"""E1b reference feasibility (W1 identity principle: price and end only deviations from a *feasible* reference).

(a) `project_clip`: per frame, the lowest point of every robot collision geom (r1.xml: spheres and capsules, so the
    lowest point is exact) below the floor z = 0 beyond `tol` lifts the root by that depth; the lift is enveloped over
    +-`env_s` and zero-phase low-passed at `cut_hz`, then never below the raw need, so the correction adds no high
    frequency. A clip is projected only if it goes deeper than `select_tol` anywhere (3 cm sits in the gap of the
    training part: every ordinary clip's deepest geom is >= -2.3 cm, retargeted feet, median -0.5 cm; the infeasible
    ones are <= -7.4 cm, lying idles and a fall's landing: 5 of 177 training clips, 2 of 31 validation clips); a
    projected clip is lifted to `tol` (0.5 cm, the ordinary feet's median) below the floor at most.
(b) `trunk_clearance_library`: per library frame, the lowest point of the trunk_ground bodies' collision geoms (pelvis,
    torso_link, waist_roll_link) above the floor, from the library's own root pose and joint angles -> <library>.trunk_clear.npy.
(c) `unexpected_trunk_contact`: the trunk_ground ending only where the reference trunk is more than `margin` above the
    floor at the current reference frame. Contact the reference itself has (lying, sitting, a fall's landing) is part of
    the motion; the tracking rewards price any deviation from it.
Nothing here is imported by the E1 tasks (E1b candidate, coordinator 17:00: after 6000 only)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).resolve().parents[2] / "r1lab/assets/r1/r1.xml"
TRUNK_BODIES = ("pelvis", "torso_link", "waist_roll_link")


@lru_cache(maxsize=1)
def _model():
  m = mujoco.MjModel.from_xml_path(str(XML))
  col = np.array([g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]])
  trunk_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TRUNK_BODIES}
  trunk = np.array([m.geom_bodyid[g] in trunk_ids for g in col])
  names = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
  qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
  assert set(m.geom_type[col]) <= {mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE}
  return m, col, trunk, names, qadr


def lowest_points(root_pos, root_quat, joint_pos):
  """(T, n_collision_geoms) lowest z of every collision geom; joint_pos in r1.xml joint order (24)."""
  m, col, _, _, qadr = _model()
  d = mujoco.MjData(m)
  cap = m.geom_type[col] == mujoco.mjtGeom.mjGEOM_CAPSULE
  rad, half = m.geom_size[col, 0], np.where(cap, m.geom_size[col, 1], 0.0)
  out = np.zeros((len(root_pos), len(col)))
  for t in range(len(root_pos)):
    d.qpos[:3] = root_pos[t]; d.qpos[3:7] = root_quat[t]; d.qpos[qadr] = joint_pos[t]
    mujoco.mj_kinematics(m, d)
    zaxis_z = d.geom_xmat[col].reshape(-1, 3, 3)[:, 2, 2]          # world z of each geom's local z axis
    out[t] = d.geom_xpos[col, 2] - np.abs(zaxis_z) * half - rad
  return out


def project_clip(root_pos, root_quat, joint_pos, fps=50.0, select_tol=0.03, tol=0.005, env_s=0.2, cut_hz=2.0):
  """returns (root_pos lifted, lift per frame, deepest penetration before) for one clip (joint_pos: 24, r1.xml order)"""
  from scipy.ndimage import maximum_filter1d
  from scipy.signal import butter, filtfilt
  low = lowest_points(root_pos, root_quat, joint_pos).min(axis=1)
  if -low.min() <= select_tol:                       # an ordinary clip (retargeted feet a few mm in the floor)
    return root_pos.copy(), np.zeros(len(root_pos)), float(low.min())
  need = np.maximum(0.0, -low - tol)
  env = maximum_filter1d(need, size=max(1, int(2 * env_s * fps) + 1), mode="nearest")
  b, a = butter(2, cut_hz / (fps / 2))
  lift = filtfilt(b, a, env, padlen=min(len(env) - 1, 3 * max(len(a), len(b)))) if len(env) > 9 else env
  lift = np.maximum(lift, need)
  out = root_pos.copy(); out[:, 2] += lift
  return out, lift, float(low.min())


def trunk_clearance_library(lib_path):
  """per-frame trunk clearance of an r1wb-library-v2 file (pelvis pose = body 0) -> saved beside it; returns the array"""
  import json
  _, _, trunk, names, _ = _model()
  with np.load(lib_path, allow_pickle=False) as z:
    man = json.loads(str(z["manifest"]))
    assert man["joint_names"] == names, "library joint order differs from r1.xml"
    low = lowest_points(z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"])
  clear = low[:, trunk].min(axis=1).astype(np.float32)
  np.save(str(lib_path) + ".trunk_clear.npy", clear)
  return clear


_CLEAR: dict = {}


def unexpected_trunk_contact(env, sensor_name: str, command_name: str, clear_file: str, margin: float = 0.05,
                             force_threshold: float = 1.0):
  import torch
  if clear_file not in _CLEAR:
    _CLEAR[clear_file] = torch.as_tensor(np.load(clear_file), device=env.device)
  term = env.command_manager.get_term(command_name)
  force = env.scene[sensor_name].data.force
  contact = torch.norm(force.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=1) > force_threshold
  return contact & (_CLEAR[clear_file][term.time_steps] > margin)
