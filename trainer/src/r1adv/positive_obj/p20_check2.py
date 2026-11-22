"""Positive 20 check 2 (rounds/R04/positive/r11_contact2/checks_rules.md section 2): grasp reach map with the wrist
roll as the only wrist joint. Whole-body mink IK on the free-base training body (feet fixed, COM over the support
polygon), then per load: joint soft limits, self / table / object penetration, feet, COM margin, static joint torques.

  python -m r1adv.positive_obj.p20_check2 <out.json> [--workers N] [--limit N]
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import mink
import mujoco
import numpy as np
from mink.lie import SE3, SO3
from scipy.spatial import ConvexHull

from r1lab import contract as C
from r1lab import robot as RB

WRIST_CAP = ((0.02, 0.0, 0.0, 0.05, 0.0, 0.0), 0.03)   # bare-wrist capsule up to the flange x 0.08 (checks_rules 3)
LITE_DX = 0.1315 - 0.15                                 # O6-lite palm centre relative to the O6 box centre
HEIGHTS = (0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.85)
WIDTHS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
XO = (0.20, 0.25, 0.30, 0.35, 0.40)
YP = (0.10, 0.15, 0.20)
LOADS = (0.2, 0.5, 1.0, 2.0, 3.0)
ONE_HAND_WMAX = 0.125
G = 9.81
ORI = (0.5, 0.5, 0.02)
R_LEFT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # palm normal (z) -y
R_RIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # palm normal +y
R_TOP = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # palm normal -z
ARM = {s: (f"{s}_o6_collision", f"{s}_hand_collision", f"{s}_elbow_collision", f"{s}_shoulder_yaw_collision")
       for s in ("left", "right")}
BODY = ("torso_collision", "pelvis_collision", "head_collision", "left_thigh_collision", "right_thigh_collision",
        "left_hip_collision", "right_hip_collision", "left_shin_collision", "right_shin_collision")
LEGS_TABLE = ("left_thigh_collision", "right_thigh_collision", "left_shin_collision", "right_shin_collision",
              "pelvis_collision", "torso_collision")


def build_model() -> mujoco.MjModel:
  spec = RB.get_spec_o6()
  for side in ("left", "right"):
    g = spec.geom(f"{side}_hand_collision")
    g.fromto = list(WRIST_CAP[0])
    g.size = [WRIST_CAP[1], 0.0, 0.0]
    hb = spec.body(f"{side}_o6_hand")
    hb.add_site(name=f"{side}_palm_box", pos=[0.0, 0.0, 0.0])
    hb.add_site(name=f"{side}_palm_lite", pos=[LITE_DX, 0.0, 0.0])
  spec.worldbody.add_geom(name="tab", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.6, 0.02], pos=[9, 0, -2])
  spec.worldbody.add_geom(name="objc", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], pos=[9, 2, -2])
  m = spec.compile()
  for j in range(m.njnt):                       # IK works inside the soft limits (95 % of each range)
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
      lo, hi = m.jnt_range[j]
      c, r = (lo + hi) / 2, (hi - lo) / 2 * 0.95
      m.jnt_range[j] = (c - r, c + r)
  return m


def set_box(m, name, pos, half):
  g = m.geom(name).id
  m.geom_pos[g] = pos
  m.geom_size[g] = half
  m.geom_rbound[g] = float(np.linalg.norm(half))
  m.geom_aabb[g] = np.r_[np.zeros(3), np.asarray(half)]


def default_q(m) -> np.ndarray:
  q = m.qpos0.copy()
  q[:3] = (0.0, 0.0, C.STAND_HEIGHT)
  q[3:7] = (1.0, 0.0, 0.0, 0.0)
  for name, val in C.DEFAULT_POS_BY_NAME.items():
    q[m.jnt_qposadr[m.joint(name).id]] = val
  return q


def foot_points(m, d) -> np.ndarray:
  pts = []
  for s in ("left", "right"):
    for i in range(1, 8):
      g = m.geom(f"{s}_foot{i}_collision").id
      c, R, hl = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g][1]
      pts += [c + sg * hl * R[:, 2] for sg in (-1, 1)]
  return np.array(pts)


def hull_margin(pts2: np.ndarray, p: np.ndarray) -> float:
  h = ConvexHull(pts2)
  # equations: a x + b y + c <= 0 inside, (a, b) unit
  return float(-(h.equations[:, :2] @ p + h.equations[:, 2]).max())


def problems(limit=None) -> list[dict]:
  out = []
  for t, w, xo in itertools.product(HEIGHTS, WIDTHS, XO):
    for typ in ("G1box", "G1lite"):
      out.append(dict(type=typ, t=t, w=w, xo=xo, yp=None))
    if w <= ONE_HAND_WMAX + 1e-9:
      for typ, yp in itertools.product(("G2", "G3"), YP):
        out.append(dict(type=typ, t=t, w=w, xo=xo, yp=yp))
  return out if limit is None else out[:limit]


class Solver:
  def __init__(self):
    self.m = build_model()
    self.d = mujoco.MjData(self.m)
    m, d = self.m, self.d
    self.q0 = default_q(m)
    d.qpos[:] = self.q0
    mujoco.mj_forward(m, d)
    self.foot_T = {s: (d.xpos[m.body(f"{s}_ankle_roll_link").id].copy(), d.xmat[m.body(f"{s}_ankle_roll_link").id].reshape(3, 3).copy())
                   for s in ("left", "right")}
    fp = foot_points(m, d)
    self.ankle_mid = 0.5 * (self.foot_T["left"][0] + self.foot_T["right"][0])
    hull = ConvexHull(fp[:, :2])
    self.support_c = fp[hull.vertices, :2].mean(0)
    self.mass = float(m.body_subtreemass[1])
    self.act_dof = {}
    self.effort = {}
    for name in C.POLICY_JOINTS:
      j = m.joint(name).id
      self.act_dof[name] = int(m.jnt_dofadr[j])
      gi = next(k for k, v in C.GROUPS.items() if any(__import__("re").fullmatch(p, name) for p in v["joints"]))
      self.effort[name] = C.GROUPS[gi]["effort"]
    self.ids = {n: m.geom(n).id for n in [*ARM["left"], *ARM["right"], *BODY, "tab", "objc"]}

  def pairs(self, typ: str) -> list[tuple[str, str]]:
    pr = [(a, b) for s in ("left", "right") for a in ARM[s] for b in BODY]
    pr += [(a, b) for a in ARM["left"] for b in ARM["right"]]
    pr += [(a, "tab") for s in ("left", "right") for a in ARM[s]]
    pr += [(a, "tab") for a in LEGS_TABLE]
    grasping = {"left_o6_collision", "right_o6_collision"} if typ.startswith("G1") else {"left_o6_collision"}
    pr += [(a, "objc") for a in [*ARM["left"], *ARM["right"], *BODY] if a not in grasping]
    return pr

  SEEDS = (None, (-0.8, 0.2, 0.0, 0.8, 0.0), (-0.8, 0.2, 0.0, 0.8, 1.5))   # default pose, arms forward, + wrist roll

  def seed_q(self, sd):
    q = self.q0.copy()
    if sd is None:
      return q
    m = self.m
    for s, sg in (("left", 1), ("right", -1)):
      for jn, v in zip(("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"), sd):
        q[m.jnt_qposadr[m.joint(f"{s}_{jn}_joint").id]] = v * (sg if jn in ("shoulder_roll", "shoulder_yaw", "wrist_roll") else 1)
    return q

  def solve(self, pb: dict) -> dict:
    """Multi-start: the IK from each seed pose; the best result by (criteria passed at the lightest load, residual)."""
    best = None
    for k, sd in enumerate(self.SEEDS):
      r = self.solve_one(pb, self.seed_q(sd))
      if "loads" not in r:
        continue
      r["seed"] = k
      key = (r["loads"]["0.2"]["ok"], r["c1"] and r["c3"] and r["c2"] and r["c4"], -r["pos_err"])
      if best is None or key > best[0]:
        best = (key, r)
    return best[1] if best else dict(pb, ok_ik=False)

  def solve_one(self, pb: dict, q_init: np.ndarray, iters: int = 400) -> dict:
    m, d = self.m, self.d
    typ, t, w, xo, yp = pb["type"], pb["t"], pb["w"], pb["xo"], pb["yp"]
    x_o = self.ankle_mid[0] + xo
    site = "palm_lite" if typ != "G1box" else "palm_box"
    if typ.startswith("G1"):
      zg = t + 0.10
      targets = {"left": (np.array([x_o, w / 2, zg]), R_LEFT), "right": (np.array([x_o, -w / 2, zg]), R_RIGHT)}
      obj_c, obj_half = np.array([x_o, 0.0, zg]), np.array([w / 2, w / 2 - 0.02, 0.10])
      load_sites = ["left", "right"]
    elif typ == "G2":
      zg = t + 0.10
      targets = {"left": (np.array([x_o, yp, zg]), R_LEFT)}
      obj_c, obj_half = np.array([x_o, yp - 0.02 - w / 2, zg]), np.array([w / 2, w / 2, 0.10])
      load_sites = ["left"]
    else:
      h = 0.10
      targets = {"left": (np.array([x_o, yp, t + h + 0.02]), R_TOP)}
      obj_c, obj_half = np.array([x_o, yp, t + h / 2]), np.array([w / 2, w / 2, h / 2])
      load_sites = ["left"]
    # scene geoms: object slightly shrunk so the grasping palm overlap (by design) is not counted; table under it
    set_box(m, "objc", obj_c, np.maximum(obj_half - 0.002, 0.005))
    if t > 0:
      front = x_o - w / 2 - 0.02
      set_box(m, "tab", np.array([front + 0.5, obj_c[1], t - 0.02]), np.array([0.5, 0.6, 0.02]))
    else:
      set_box(m, "tab", np.array([9.0, 0.0, -2.0]), np.array([0.5, 0.6, 0.02]))
    cfg = mink.Configuration(m)
    cfg.update(q_init.copy())
    tasks = []
    for s in ("left", "right"):
      ft = mink.FrameTask(f"{s}_ankle_roll_link", "body", position_cost=100.0, orientation_cost=100.0)
      p, R = self.foot_T[s]
      ft.set_target(SE3.from_rotation_and_translation(SO3.from_matrix(R), p))
      tasks.append(ft)
    hand_tasks = {}
    for s, (p, R) in targets.items():
      ht = mink.FrameTask(f"{s}_{site}", "site", position_cost=10.0, orientation_cost=[10 * c for c in ORI])
      ht.set_target(SE3.from_rotation_and_translation(SO3.from_matrix(R), p))
      tasks.append(ht)
      hand_tasks[s] = ht
    com = mink.ComTask(cost=[10.0, 10.0, 0.0])
    com.set_target(np.r_[self.support_c, 0.0])
    tasks.append(com)
    cost = np.full(m.nv, 1e-3)
    cost[:6] = 0.0
    post = mink.PostureTask(m, cost=cost)
    post.set_target(self.q0)
    tasks.append(post)
    limits = [mink.ConfigurationLimit(m, min_distance_from_limits=0.005),
              mink.CollisionAvoidanceLimit(m, geom_pairs=[([a], [b]) for a, b in self.pairs(typ)],
                                           minimum_distance_from_collisions=0.005, collision_detection_distance=0.05)]
    for it in range(iters):
      try:
        v = mink.solve_ik(cfg, tasks, 0.02, "daqp", damping=1e-3, limits=limits)
      except Exception as e:                                       # infeasible QP
        return dict(pb, ok_ik=False, err=str(e)[:80])
      cfg.integrate_inplace(v, 0.02)
      if it % 10 == 9:
        e = max(np.linalg.norm(ht.compute_error(cfg)[:3]) for ht in hand_tasks.values())
        if e < 5e-4:
          break
    q = cfg.q.copy()
    return self.evaluate(pb, q, targets, site, obj_c, load_sites, it + 1)

  def evaluate(self, pb, q, targets, site, obj_c, load_sites, n_it) -> dict:
    m, d = self.m, self.d
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    res = dict(pb, iters=n_it)
    # 1 palm residuals
    pos_e, nrm_e = [], []
    for s, (p, R) in targets.items():
      sid = m.site(f"{s}_{site}").id
      pos_e.append(float(np.linalg.norm(d.site_xpos[sid] - p)))
      nrm_e.append(float(np.degrees(np.arccos(np.clip(d.site_xmat[sid].reshape(3, 3)[:, 2] @ R[:, 2], -1, 1)))))
    res.update(pos_err=max(pos_e), nrm_err=max(nrm_e))
    c1 = max(pos_e) <= 0.005 and max(nrm_e) <= 5.0
    # 2 soft joint limits (95 % of range; the model ranges were already shrunk to the soft range)
    worst = np.inf
    for j in range(m.njnt):
      if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
        continue
      lo, hi = m.jnt_range[j]
      worst = min(worst, min(q[m.jnt_qposadr[j]] - lo, hi - q[m.jnt_qposadr[j]]))
    res["soft_margin"] = float(worst)
    c2 = worst >= 0.0
    # 3 penetration (self, table, object)
    pen, obj_pen = {}, {}
    for a, b in self.pairs(pb["type"]):
      dist = mujoco.mj_geomDistance(m, d, self.ids[a], self.ids[b], 0.05, None)
      if dist < 0:
        (obj_pen if b == "objc" else pen)[f"{a}|{b}"] = round(float(dist), 4)
    res["pen"] = pen
    res["obj_pen"] = obj_pen          # robot geom inside the object box: reported, not a frozen criterion (rules 2.3)
    c3 = not pen
    # 4 feet
    fe = 0.0
    for s in ("left", "right"):
      b = m.body(f"{s}_ankle_roll_link").id
      p, R = self.foot_T[s]
      fe = max(fe, float(np.linalg.norm(d.xpos[b] - p)) / 0.002,
               float(np.degrees(np.arccos(np.clip((np.trace(d.xmat[b].reshape(3, 3).T @ R) - 1) / 2, -1, 1)))) / 1.0)
    res["feet_score"] = fe
    c4 = fe <= 1.0
    # 5, 6 per load
    fp = foot_points(m, d)[:, :2]
    com_r = d.subtree_com[1].copy()
    res["com_margin_noload"] = hull_margin(fp, com_r[:2])
    nv = m.nv
    jac_feet = []
    for s in ("left", "right"):
      jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
      mujoco.mj_jacBody(m, d, jp, jr, m.body(f"{s}_ankle_roll_link").id)
      jac_feet.append(np.r_[jp, jr])
    Jc = np.concatenate(jac_feet, 0)                          # (12, nv): wrenches at the ankle origins
    jl = []
    for s in load_sites:
      jp = np.zeros((3, nv))
      mujoco.mj_jacSite(m, d, jp, None, m.site(f"{s}_{site}").id)
      jl.append(jp)
    per = {}
    for mass in LOADS:
      com = (self.mass * com_r + mass * obj_c) / (self.mass + mass)
      marg = hull_margin(fp, com[:2])
      f = np.array([0.0, 0.0, -mass * G / len(load_sites)])
      gen = d.qfrc_bias.copy() - sum(J.T @ f for J in jl)        # tau + Jc^T lam = gen  (static)
      # base rows: Jc[:, :6]^T lam = gen[:6]  -> least-norm lam
      lam = np.linalg.lstsq(Jc[:, :6].T, gen[:6], rcond=None)[0]
      tau = gen - Jc.T @ lam
      ratio = max(abs(tau[self.act_dof[n]]) / self.effort[n] for n in C.POLICY_JOINTS)
      worst_j = max(C.POLICY_JOINTS, key=lambda n: abs(tau[self.act_dof[n]]) / self.effort[n])
      ok = c1 and c2 and c3 and c4 and marg >= 0.02 and ratio <= 1.0
      per[str(mass)] = dict(com_margin=float(marg), tau_ratio=float(ratio), tau_worst=worst_j, ok=bool(ok))
    res.update(c1=bool(c1), c2=bool(c2), c3=bool(c3), c4=bool(c4), loads=per, q=[round(float(v), 5) for v in q])
    return res


_S = None


def _work(pb):
  global _S
  if _S is None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _S = Solver()
  try:
    return _S.solve(pb)
  except Exception as e:
    return dict(pb, ok_ik=False, err=repr(e)[:120])


def main(argv):
  out = argv[0]
  workers, limit = 12, None
  if "--workers" in argv:
    workers = int(argv[argv.index("--workers") + 1])
  if "--limit" in argv:
    limit = int(argv[argv.index("--limit") + 1])
  pbs = problems(limit)
  t0 = time.time()
  from multiprocessing import Pool
  with Pool(workers) as pool:
    rows = []
    for i, r in enumerate(pool.imap_unordered(_work, pbs, chunksize=2)):
      rows.append(r)
      if i % 100 == 0:
        print(f"{i}/{len(pbs)} {time.time() - t0:.0f} s", flush=True)
  json.dump(dict(rules="rounds/R04/positive/r11_contact2/checks_rules.md", n=len(rows), wall_s=time.time() - t0,
                 rows=rows), open(out, "w"))
  print("wrote", out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
  main(sys.argv[1:])
