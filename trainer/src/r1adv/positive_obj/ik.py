"""mink IK for the contact ladder's two-hand squeeze path (positive 18, R11).

The ladder holds one object between the two O6 boxes at a fixed grasp point straight ahead of the torso; per world only the
hand half-separation s changes (object width, squeeze depth, clearance). The arm solution is computed in the torso frame
(waist roll = waist yaw = 0): squeeze/release at height Z_G, lift/lower +LIFT. The carry is a waist-yaw turn with the arm
joints held (the object centre travels CARRY m along an arc about the waist-yaw axis), because the left/right arm cannot
cross the body midline (shoulder roll adduction limit -0.227 rad): a lateral carry at fixed torso yaw is infeasible for
narrow objects (IK check 2026-11-19 12:3x KST, 4-12 cm residuals).
IK is solved on a grid s x dz (dz = 0 .. LIFT) and joint targets are interpolated bilinearly (ladder.py).

Hand targets (torso yaw 0, pelvis at robot_fixed.PELVIS_POS): left O6 box centre (X_G, +s, Z_G + dz), right (X_G, -s, ...);
box x axis (the forearm) along +x, box z axis (thickness = palm normal) toward the object (left -y, right +y). Rotation
about the palm normal has a low cost: the forearm may tilt in the palm plane.

  python -m r1adv.positive_obj.ik <out.npz> [x_g z_g]
"""
from __future__ import annotations

import sys
import time

import mink
import mujoco
import numpy as np
from mink.lie import SE3, SO3

from r1lab import contract as C
from r1adv.positive_obj.robot_fixed import PELVIS_POS, get_spec_o6_fixed

X_G, Z_G = 0.40, 0.76                                       # natural elbow height: forearm ~horizontal
LIFT, CARRY = 0.20, 0.30
YAW_AXIS_X = 0.0325                                         # waist-yaw axis x in the pelvis frame (r1.xml torso_link pos)
S_GRID = np.round(np.arange(0.03, 0.3001, 0.01), 4)          # hand-centre half-separation (m)
DU = 0.01
# Orientation cost on the box-frame rotation error (x = forearm roll, y = forearm yaw, z = forearm pitch about the palm
# normal). The pitch is nearly free: a higher pitch cost trades centimetres of position for degrees of forearm pitch.
ORI_COST = (0.5, 0.5, 0.02)
# Box axes in world: columns are the box x, y, z axes.
R_LEFT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])    # x=(1,0,0) y=(0,0,1) z=(0,-1,0)
R_RIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])   # x=(1,0,0) y=(0,0,-1) z=(0,1,0)
assert abs(np.linalg.det(R_LEFT) - 1) < 1e-9 and abs(np.linalg.det(R_RIGHT) - 1) < 1e-9
ARM_GEOMS = {s: (f"{s}_o6_collision", f"{s}_hand_collision", f"{s}_elbow_collision") for s in ("left", "right")}
BODY_GEOMS = ("torso_collision", "pelvis_collision", "head_collision", "left_thigh_collision", "right_thigh_collision",
              "left_hip_collision", "right_hip_collision")


def carry_yaw(x_g: float = X_G) -> float:
  """Total waist-yaw turn that moves the grasp point CARRY m along its arc."""
  return CARRY / (x_g - YAW_AXIS_X)


def u_grid(du: float = DU) -> np.ndarray:
  n = int(round(LIFT / du))
  return np.linspace(0.0, LIFT, n + 1)


def path_yz(u: np.ndarray | float, z_g: float = Z_G) -> tuple[np.ndarray, np.ndarray]:
  u = np.asarray(u, dtype=float)
  return np.zeros_like(u), z_g + u


def build_model() -> mujoco.MjModel:
  spec = get_spec_o6_fixed()
  spec.body("pelvis").pos = list(PELVIS_POS)
  return spec.compile()


def default_qpos(m: mujoco.MjModel) -> np.ndarray:
  q = np.zeros(m.nq)
  for name, val in C.DEFAULT_POS_BY_NAME.items():
    q[m.jnt_qposadr[m.joint(name).id]] = val
  return q


def _angle(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def solve_grid(x_g: float = X_G, z_g: float = Z_G, s_grid: np.ndarray = S_GRID, du: float = DU,
               iters: int = 60, tol: float = 1e-4, collision: bool = True) -> dict:
  m = build_model()
  d = mujoco.MjData(m)
  cfg = mink.Configuration(m)
  q0 = default_qpos(m)
  cfg.update(q0)
  hands = {s: mink.FrameTask(f"{s}_o6_hand", "body", position_cost=1.0, orientation_cost=list(ORI_COST))
           for s in ("left", "right")}
  cost = np.full(m.nv, 1e-2)
  for j in range(m.njnt):
    name = m.joint(j).name
    if "hip" in name or "knee" in name or "ankle" in name:
      cost[m.jnt_dofadr[j]] = 1.0
    elif "waist" in name:
      cost[m.jnt_dofadr[j]] = 10.0                          # torso frame: waist held at 0
  posture = mink.PostureTask(m, cost=cost)
  posture.set_target(q0)
  limits = [mink.ConfigurationLimit(m, min_distance_from_limits=0.03)]
  if collision:
    pairs = [(list(ARM_GEOMS[s]), list(BODY_GEOMS)) for s in ("left", "right")]
    limits.append(mink.CollisionAvoidanceLimit(m, geom_pairs=pairs, minimum_distance_from_collisions=0.01,
                                               collision_detection_distance=0.05))
  tasks = [hands["left"], hands["right"], posture]
  us = u_grid(du)
  ys, zs = path_yz(us, z_g)
  Q = np.zeros((len(s_grid), len(us), m.nq))
  pos_err = np.zeros((len(s_grid), len(us), 2))
  nrm_err = np.zeros((len(s_grid), len(us), 2))
  tilt = np.zeros((len(s_grid), len(us), 2))
  min_dist = np.full((len(s_grid), len(us)), np.inf)
  elbow_low = np.zeros((len(s_grid), len(us)))              # lowest point of the elbow/forearm capsules minus Z_G + dz
  low_geoms = [m.geom(f"{s}_{g}_collision").id for s in ("left", "right") for g in ("elbow", "hand")]
  hand_low = np.full((len(s_grid), len(us)), np.inf)          # lowest O6 box corner minus Z_G + dz
  hid = {s: m.body(f"{s}_o6_hand").id for s in ("left", "right")}
  gpairs = [(m.geom(a).id, m.geom(b).id) for s in ("left", "right") for a in ARM_GEOMS[s] for b in BODY_GEOMS]
  q_start = q0.copy()
  for i, s in enumerate(s_grid):
    cfg.update(q_start)
    for j in range(len(us)):
      pl = np.array([x_g, ys[j] + s, zs[j]])
      pr = np.array([x_g, ys[j] - s, zs[j]])
      hands["left"].set_target(SE3.from_rotation_and_translation(SO3.from_matrix(R_LEFT), pl))
      hands["right"].set_target(SE3.from_rotation_and_translation(SO3.from_matrix(R_RIGHT), pr))
      n_it = iters * (6 if j == 0 else 1)
      for _ in range(n_it):
        v = mink.solve_ik(cfg, tasks, 0.02, "daqp", damping=1e-3, limits=limits)
        cfg.integrate_inplace(v, 0.02)
        el = np.linalg.norm(hands["left"].compute_error(cfg)[:3])
        er = np.linalg.norm(hands["right"].compute_error(cfg)[:3])
        if max(el, er) < tol:
          break
      Q[i, j] = cfg.q
      d.qpos[:] = cfg.q
      mujoco.mj_kinematics(m, d)
      for k, (side, tgt, Rt) in enumerate((("left", pl, R_LEFT), ("right", pr, R_RIGHT))):
        b = hid[side]
        pos_err[i, j, k] = np.linalg.norm(d.xpos[b] - tgt)
        Rb = d.xmat[b].reshape(3, 3)
        nrm_err[i, j, k] = _angle(Rb[:, 2], Rt[:, 2])
        tilt[i, j, k] = float(np.degrees(np.arcsin(np.clip(Rb[2, 0], -1, 1))))   # forearm pitch (+ = up)
      lows = []
      for g in low_geoms:
        c, Rg, (r, hl) = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g][:2]
        lows += [(c + sg * hl * Rg[:, 2])[2] - r for sg in (-1, 1)]
      elbow_low[i, j] = min(lows) - zs[j]
      for side in ("left", "right"):
        g = m.geom(f"{side}_o6_collision").id
        c, Rg, hs = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        corners = [c + Rg @ (hs * np.array([a, b, e])) for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]
        hand_low[i, j] = min(hand_low[i, j], min(cc[2] for cc in corners) - zs[j])
      for ga, gb in gpairs:
        dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.2, None)
        min_dist[i, j] = min(min_dist[i, j], dist)
      if j == 0:
        q_start = cfg.q.copy()
  lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
  margin = np.full(Q.shape[:2], np.inf)
  for jn in range(m.njnt):
    if m.jnt_limited[jn]:
      a = m.jnt_qposadr[jn]
      margin = np.minimum(margin, np.minimum(Q[..., a] - lo[jn], hi[jn] - Q[..., a]))
  return dict(Q=Q, s_grid=np.asarray(s_grid), u_grid=us, x_g=x_g, z_g=z_g, lift=LIFT, carry=CARRY, carry_yaw=carry_yaw(x_g),
              pos_err=pos_err, nrm_err=nrm_err, tilt=tilt, min_dist=min_dist, limit_margin=margin, elbow_low=elbow_low, hand_low=hand_low,
              joint_names=np.array([m.joint(j).name for j in range(m.njnt)]), pelvis=np.array(PELVIS_POS))


PSI_GRID_N = 9                                              # waist-yaw samples between the pick and place yaw


def psi_grid(x_g: float = X_G) -> np.ndarray:
  return np.linspace(-carry_yaw(x_g) / 2, carry_yaw(x_g) / 2, PSI_GRID_N)


def gravity_offsets(Q: np.ndarray, psis: np.ndarray) -> np.ndarray:
  """Static gravity feed-forward for PD targets (rules_amend A1): for the IK pose Q[s, dz] with the waist yaw set to psi,
  offset = qfrc_bias(q, qdot=0) / kp for every actuated joint, so the deploy PD holds the pose against the robot's own
  weight (no object load). Waist yaw moves the forward arms' centre of mass sideways, so the waist-roll offset depends
  on psi. Returns G[s, dz, psi, nq]."""
  m = build_model()
  d = mujoco.MjData(m)
  kp = np.zeros(m.nv)
  for j in range(m.njnt):
    name = m.joint(j).name
    if name in C.POLICY_JOINTS:
      kp[m.jnt_dofadr[j]] = C.KP[C.POLICY_JOINTS.index(name)]
  yaw_adr = m.jnt_qposadr[m.joint("waist_yaw_joint").id]
  G = np.zeros(Q.shape[:2] + (len(psis), m.nq))
  for i in range(Q.shape[0]):
    for j in range(Q.shape[1]):
      for k, psi in enumerate(psis):
        d.qpos[:] = Q[i, j]
        d.qpos[yaw_adr] = psi
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for jn in range(m.njnt):
          dof = m.jnt_dofadr[jn]
          if kp[dof] > 0:
            G[i, j, k, m.jnt_qposadr[jn]] = d.qfrc_bias[dof] / kp[dof]
  return G


def summary(r: dict) -> str:
  pe, ne, md = r["pos_err"].max(-1), r["nrm_err"].max(-1), r["min_dist"]
  lines = [f"x_g {r['x_g']:.2f} z_g {r['z_g']:.2f}: grid {r['Q'].shape[:2]}"]
  for i, s in enumerate(r["s_grid"]):
    if i % 3 and i != len(r["s_grid"]) - 1:
      continue
    lines.append(f"  s {s:.2f}: pos err max {1000 * pe[i].max():6.1f} mm, palm normal err max {ne[i].max():5.1f} deg, "
                 f"min body dist {100 * md[i].min():5.1f} cm, limit margin min {r['limit_margin'][i].min():.3f} rad, "
                 f"forearm tilt {r['tilt'][i].min():.0f}..{r['tilt'][i].max():.0f} deg, "
                 f"arm low {100 * r['elbow_low'][i, 0]:.1f} cm, box low {100 * r['hand_low'][i, 0]:.1f} cm (dz=0)")
  return "\n".join(lines)


if __name__ == "__main__":
  out = sys.argv[1]
  xg, zg = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (X_G, Z_G)
  t0 = time.time()
  res = solve_grid(xg, zg)
  res["psi_grid"] = psi_grid(xg)
  res["G"] = gravity_offsets(res["Q"], res["psi_grid"])
  np.savez_compressed(out, **res)
  print(summary(res))
  print(f"{time.time() - t0:.1f} s")
