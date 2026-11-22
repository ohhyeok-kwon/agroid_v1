"""Positive 24 (E1 c09) chest-hold poses by IK (rounds/R04/positive/e1_c09/rules.md 4).

Box: depth D_BOX (x), width w (y), height H_BOX (z), centre (X_C, 0, Z_C): back face 1 mm in front of the torso capsule
(r1.xml torso_collision), centre height = torso capsule centre (the back face meets the capsule's straight part).

  mode 'edge' (grip C, main): each fist hooks the box's front vertical edge (x_front, +-w/2) with its contact side; the patch
         normal is the diagonal (-1, -+1, 0)/sqrt2 (backward and inward), so the fists' normal forces push the box into the
         chest (load path through normal force, not friction). Forearm direction (site x) forward-inward (1, -+1, 0)/sqrt2,
         low cost on the rotation about the normal.
  mode 'side' (grip Cs, side arm): fists squeeze the side-face centres (X_C, +-w/2, Z_C), orientation as positive 18 ik.py
         R_LEFT / R_RIGHT; the chest carries no designed load.
A front-face press (capsule side flat on the front face) was tried first in dev (16:1x): the IK did not converge
(diverged, |v| 89-629 rad/s); not pursued.

Whole-robot mink IK on the fixed body (waist held at 0 by a high posture cost, legs high cost), site task on the arm's
<side>_patch (p20_fist), collision avoidance: upper arm / forearm capsules vs the box and the body, fist geoms vs the body.
Staged path: 10 cm out along the patch normal -> start (1 mm gap) -> contact.

  python -m r1adv.positive_obj.p24_hug_ik <out.npz> <mode> <arm> <w> [<w> ...]
Feasible (rules 4): for both hands at the start and contact poses: gap along the patch normal <= 5 mm, offset along the
contact feature within the fist contact half-length (edge: along the forearm <= 35 mm; side: in the face plane, face
half-size less 35 mm), height offset <= H/2 - 35 mm, patch normal error <= 5 deg, no penetration between a non-fist arm
geom and the box or between arm and body geoms, joint limits respected. Changed in dev (16:2x, before any hold value)
from 'patch position residual <= 5 mm', which the pad body missed at w 0.20 / 0.30 (10-18 mm in the face plane at the
elbow limit) while its patch stayed on the face.
"""
from __future__ import annotations

import sys
import time

import mink
import mujoco
import numpy as np
from mink.lie import SE3, SO3

from r1lab import contract as C
from r1adv.positive_obj import p20_fist as F

D_BOX, H_BOX = 0.125, 0.25
TORSO_FRONT_X, Z_C = 0.0925, 0.9153          # world, pelvis at PELVIS_POS (torso capsule centre x 0.0125 + r 0.08; centre z)
GAP_BACK = 0.001
GAP_START = 0.001
POS_TOL, NRM_TOL = 0.005, 5.0
HALF_CONTACT = 0.035                         # fist contact half-length: capsule straight part 0.035, pad grid 0.022 + 0.005
ORI_COST = (0.5, 0.5, 0.02)
A2 = 1.0 / np.sqrt(2.0)


def frames(mode: str) -> dict:
  """Site rotation (columns x, y, z) per side; z = patch normal toward the box."""
  if mode == "side":
    return {"left": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            "right": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])}
  xl, zl = np.array([A2, -A2, 0.0]), np.array([-A2, -A2, 0.0])
  xr, zr = np.array([A2, A2, 0.0]), np.array([-A2, A2, 0.0])
  return {"left": np.stack([xl, np.cross(zl, xl), zl], axis=1), "right": np.stack([xr, np.cross(zr, xr), zr], axis=1)}


for _m in ("side", "edge"):
  for _R in frames(_m).values():
    assert abs(np.linalg.det(_R) - 1) < 1e-9 and np.allclose(_R.T @ _R, np.eye(3))
ARM_NONFIST = {s: (f"{s}_shoulder_yaw_collision", f"{s}_elbow_collision") for s in ("left", "right")}
BODY = ("torso_collision", "pelvis_collision", "head_collision", "left_thigh_collision", "right_thigh_collision",
        "left_hip_collision", "right_hip_collision")


def box_centre(w: float) -> np.ndarray:
  return np.array([TORSO_FRONT_X + GAP_BACK + D_BOX / 2, 0.0, Z_C])


def contact_point(mode: str, w: float, side: str) -> np.ndarray:
  c = box_centre(w)
  sy = 1.0 if side == "left" else -1.0
  if mode == "side":
    return np.array([c[0], sy * w / 2, Z_C])
  return np.array([c[0] + D_BOX / 2, sy * w / 2, Z_C])


def n_out(mode: str, side: str) -> np.ndarray:
  """Outward normal of the box feature the fist touches (= minus the patch normal)."""
  return -frames(mode)[side][:, 2]


def fist_geoms(arm: str, side: str) -> list[str]:
  g = [f"{side}_hand_collision"]
  if arm != "fist":
    g.append(f"{side}_o6_collision")
  if arm == "fistpads":
    g += [f"{side}_pad{i}_collision" for i in range(len(F.PADS))]
  return g


def build(arm: str, w: float) -> mujoco.MjModel:
  spec = F.fixed_spec(arm)
  c = box_centre(w)
  spec.worldbody.add_geom(name="ik_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[D_BOX / 2, w / 2, H_BOX / 2], pos=list(c),
                          contype=0, conaffinity=0, group=3)
  return spec.compile()


def default_q(m) -> np.ndarray:
  q = np.zeros(m.nq)
  for name, val in C.DEFAULT_POS_BY_NAME.items():
    q[m.jnt_qposadr[m.joint(name).id]] = val
  return q


def targets(mode: str, w: float, stage: str) -> dict:
  gap = {"far": 0.10, "start": GAP_START, "contact": 0.0}[stage]
  return {s: contact_point(mode, w, s) + gap * n_out(mode, s) for s in ("left", "right")}


def offsets_at(mode: str, w: float, p: np.ndarray, side: str, gap0: float) -> list[float]:
  """(along-feature offset, gap along n_out minus the stage gap, height offset) of a patch position."""
  e = contact_point(mode, w, side)
  n = n_out(mode, side)
  dvec = p - e
  gap = float(dvec @ n) - gap0
  if mode == "side":
    along = float(dvec[0])
  else:
    along = float(dvec @ frames(mode)[side][:, 0])
  return [along, gap, float(dvec[2])]


def solve(mode: str, arm: str, w: float, iters: int = 200) -> dict:
  m = build(arm, w)
  d = mujoco.MjData(m)
  cfg = mink.Configuration(m)
  q0 = default_q(m)
  cfg.update(q0)
  Rh = frames(mode)
  tasks_h = {s: mink.FrameTask(f"{s}_patch", "site", position_cost=1.0, orientation_cost=list(ORI_COST)) for s in ("left", "right")}
  cost = np.full(m.nv, 1e-2)
  for j in range(m.njnt):
    name = m.joint(j).name
    if "hip" in name or "knee" in name or "ankle" in name:
      cost[m.jnt_dofadr[j]] = 1.0
    elif "waist" in name:
      cost[m.jnt_dofadr[j]] = 10.0
  posture = mink.PostureTask(m, cost=cost)
  posture.set_target(q0)
  tasks = [tasks_h["left"], tasks_h["right"], posture]
  pairs = [(list(ARM_NONFIST[s]) + fist_geoms(arm, s), list(BODY)) for s in ("left", "right")]
  pairs += [(list(ARM_NONFIST[s]), ["ik_box"]) for s in ("left", "right")]
  lim = [mink.ConfigurationLimit(m, min_distance_from_limits=0.02),
         mink.CollisionAvoidanceLimit(m, geom_pairs=pairs, minimum_distance_from_collisions=0.005, collision_detection_distance=0.05)]
  out = {}
  for stage, mult in (("far", 6), ("start", 3), ("contact", 3)):
    tg = targets(mode, w, stage)
    for s in ("left", "right"):
      tasks_h[s].set_target(SE3.from_rotation_and_translation(SO3.from_matrix(Rh[s]), tg[s]))
    for _ in range(iters * mult):
      v = mink.solve_ik(cfg, tasks, 0.02, "daqp", damping=1e-3, limits=lim)
      cfg.integrate_inplace(v, 0.02)
      if max(np.linalg.norm(tasks_h[s].compute_error(cfg)[:3]) for s in ("left", "right")) < 1e-4:
        break
    d.qpos[:] = cfg.q
    mujoco.mj_kinematics(m, d)
    pe, ne, offs = [], [], []
    for s in ("left", "right"):
      sid = m.site(f"{s}_patch").id
      pe.append(float(np.linalg.norm(d.site_xpos[sid] - tg[s])))
      ne.append(float(np.degrees(np.arccos(np.clip(d.site_xmat[sid].reshape(3, 3)[:, 2] @ Rh[s][:, 2], -1, 1)))))
      offs.append(offsets_at(mode, w, d.site_xpos[sid].copy(), s, {"far": 0.10, "start": GAP_START, "contact": 0.0}[stage]))
    out[stage] = dict(q=cfg.q.copy(), pos_err=pe, nrm_err=ne, patch_offset=offs)
  ib = m.geom("ik_box").id
  for stage in ("start", "contact"):
    d.qpos[:] = out[stage]["q"]
    mujoco.mj_kinematics(m, d)
    clear_box = min(mujoco.mj_geomDistance(m, d, m.geom(g).id, ib, 0.2, None) for s in ("left", "right") for g in ARM_NONFIST[s])
    clear_body = min(mujoco.mj_geomDistance(m, d, m.geom(a).id, m.geom(b).id, 0.2, None)
                     for s in ("left", "right") for a in ARM_NONFIST[s] + tuple(fist_geoms(arm, s)) for b in BODY)
    fist_box = min(mujoco.mj_geomDistance(m, d, m.geom(g).id, ib, 0.2, None) for s in ("left", "right") for g in fist_geoms(arm, s))
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    margin = min(min(d.qpos[m.jnt_qposadr[j]] - lo[j], hi[j] - d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.jnt_limited[j])
    out[stage].update(clear_box=float(clear_box), clear_body=float(clear_body), fist_box=float(fist_box), limit_margin=float(margin))
  along_max = HALF_CONTACT if mode == "edge" else D_BOX / 2 - HALF_CONTACT

  def placed(st):
    return all(abs(o[1]) <= POS_TOL and abs(o[0]) <= along_max and abs(o[2]) <= H_BOX / 2 - HALF_CONTACT for o in out[st]["patch_offset"])
  ok = all(placed(s) and max(out[s]["nrm_err"]) <= NRM_TOL and out[s]["clear_box"] >= 0.0 and out[s]["clear_body"] >= 0.0
           and out[s]["limit_margin"] >= -1e-6 for s in ("start", "contact"))
  return dict(mode=mode, arm=arm, w=w, feasible=bool(ok), stages=out, joint_names=[m.joint(j).name for j in range(m.njnt)],
              box_centre=box_centre(w))


def main(argv):
  out_path, mode, arm = argv[0], argv[1], argv[2]
  ws = [float(x) for x in argv[3:]]
  res = {}
  for w in ws:
    t0 = time.time()
    r = solve(mode, arm, w)
    st = r["stages"]
    print(f"{mode} {arm} w {w:.2f}: feasible {r['feasible']} | " + " | ".join(
      f"{s}: pos {1000 * max(st[s]['pos_err']):.1f} mm nrm {max(st[s]['nrm_err']):.1f} deg off(mm) {np.round(1000 * np.array(st[s]['patch_offset']), 1).tolist()}"
      + (f" box {100 * st[s]['clear_box']:.1f} cm body {100 * st[s]['clear_body']:.1f} cm fist-box {1000 * st[s]['fist_box']:.1f} mm lim {st[s]['limit_margin']:.3f}"
         if "clear_box" in st[s] else "") for s in ("far", "start", "contact")) + f" ({time.time() - t0:.0f} s)", flush=True)
    res[w] = r
  S = ("start", "contact")
  np.savez(out_path, mode=mode, arm=arm, widths=np.array(ws), feasible=np.array([res[w]["feasible"] for w in ws]),
           q_start=np.array([res[w]["stages"]["start"]["q"] for w in ws]), q_contact=np.array([res[w]["stages"]["contact"]["q"] for w in ws]),
           pos_err=np.array([[res[w]["stages"][s]["pos_err"] for s in S] for w in ws]),
           nrm_err=np.array([[res[w]["stages"][s]["nrm_err"] for s in S] for w in ws]),
           patch_offset=np.array([[res[w]["stages"][s]["patch_offset"] for s in S] for w in ws]),
           clear_box=np.array([[res[w]["stages"][s]["clear_box"] for s in S] for w in ws]),
           clear_body=np.array([[res[w]["stages"][s]["clear_body"] for s in S] for w in ws]),
           fist_box=np.array([[res[w]["stages"][s]["fist_box"] for s in S] for w in ws]),
           limit_margin=np.array([[res[w]["stages"][s]["limit_margin"] for s in S] for w in ws]),
           joint_names=np.array(res[ws[0]]["joint_names"]), box_centre=np.array([res[w]["box_centre"] for w in ws]),
           d_box=D_BOX, h_box=H_BOX, z_c=Z_C)


if __name__ == "__main__":
  main(sys.argv[1:])
