"""Positive 20 fist check B (rounds/R04/positive/r11_hold/fist_checks_rules.md section B): check 2 (p20_check2) redone
with the fist hand frame. Two-hand squeeze only (a fist cannot hold one-handed). Whole-body mink IK on the free-base
training body (feet fixed, COM over the support polygon); per load 0.2-3 kg: soft joint limits, self / table / non-grasp
penetration, feet, COM margin >= 2 cm, static joint torques <= effort (criteria and code of p20_check2.evaluate).

Targets: the arm's contact patch site (p20_fist.PATCH / R_FACE) at the object surface, (x_o, +-w/2, z_g), frame z toward
the object; object box (w/2, w/2, 0.10) on a table of height t (z_g = t + 0.10), as check 2. 'fist' = stock capsule patch,
'fit' = fitted fist box face; both on the fistfit geometry (fist box + wrist capsule) for the collision criteria.
Extra 'hold' problems: the hold test's grasp point x_o = X_G (world, pelvis at x 0), z_g = Z_G - 0.01 (free-base pelvis
1 cm lower than the hold test's fixed pelvis).

  python -m r1adv.positive_obj.p20_fist_checkB <out.json> [--workers N] [--limit N]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import mink
import mujoco
import numpy as np
from mink.lie import SE3, SO3

from r1adv.positive_obj import p20_check2 as C2
from r1adv.positive_obj import p20_fist as F

W_OBJ = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
X_G_HOLD, Z_G_HOLD = 0.36, 0.76
SITE = {"G1fist": "patch_fist", "G1fit": "patch"}


def build_model() -> mujoco.MjModel:
  spec = F.spec_fist("fistfit")
  dx, dy = F.PATCH["fist"]
  for side in F.SIDES:
    spec.body(f"{side}_o6_hand").add_site(name=f"{side}_patch_fist", pos=[dx, 0.0, F.R_FACE["fist"]])
  spec.worldbody.add_geom(name="tab", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.6, 0.02], pos=[9, 0, -2])
  spec.worldbody.add_geom(name="objc", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], pos=[9, 2, -2])
  m = spec.compile()
  for j in range(m.njnt):
    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.jnt_limited[j]:
      lo, hi = m.jnt_range[j]
      c, r = (lo + hi) / 2, (hi - lo) / 2 * 0.95
      m.jnt_range[j] = (c - r, c + r)
  return m


def problems(limit=None) -> list[dict]:
  out = []
  for typ in SITE:
    for t, w, xo in itertools.product(C2.HEIGHTS, W_OBJ, C2.XO):
      out.append(dict(type=typ, t=t, w=w, xo=xo, yp=None))
    for w in W_OBJ:
      out.append(dict(type=typ, t=round(Z_G_HOLD - 0.01 - 0.10, 4), w=w, xo=None, yp=None, hold=True))
  return out if limit is None else out[:limit]


class FistSolver(C2.Solver):
  def __init__(self):
    C2.build_model = build_model                          # Solver.__init__ looks build_model up in p20_check2
    super().__init__()

  def pairs(self, typ: str) -> list[tuple[str, str]]:
    pr = [(a, b) for s in ("left", "right") for a in C2.ARM[s] for b in C2.BODY]
    pr += [(a, b) for a in C2.ARM["left"] for b in C2.ARM["right"]]
    pr += [(a, "tab") for s in ("left", "right") for a in C2.ARM[s]]
    pr += [(a, "tab") for a in C2.LEGS_TABLE]
    grasping = {f"{s}_{g}" for s in ("left", "right") for g in ("o6_collision", "hand_collision")}
    pr += [(a, "objc") for a in [*C2.ARM["left"], *C2.ARM["right"], *C2.BODY] if a not in grasping]
    return pr

  def solve_one(self, pb: dict, q_init: np.ndarray, iters: int = 400) -> dict:
    m = self.m
    t, w = pb["t"], pb["w"]
    x_o = X_G_HOLD if pb["xo"] is None else self.ankle_mid[0] + pb["xo"]
    site = SITE[pb["type"]]
    zg = t + 0.10
    targets = {"left": (np.array([x_o, w / 2, zg]), C2.R_LEFT), "right": (np.array([x_o, -w / 2, zg]), C2.R_RIGHT)}
    obj_c, obj_half = np.array([x_o, 0.0, zg]), np.array([w / 2, w / 2, 0.10])
    C2.set_box(m, "objc", obj_c, np.maximum(obj_half - 0.002, 0.005))
    if t > 0:
      front = x_o - w / 2 - 0.02
      C2.set_box(m, "tab", np.array([front + 0.5, 0.0, t - 0.02]), np.array([0.5, 0.6, 0.02]))
    else:
      C2.set_box(m, "tab", np.array([9.0, 0.0, -2.0]), np.array([0.5, 0.6, 0.02]))
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
      ht = mink.FrameTask(f"{s}_{site}", "site", position_cost=10.0, orientation_cost=[10 * c for c in C2.ORI])
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
              mink.CollisionAvoidanceLimit(m, geom_pairs=[([a], [b]) for a, b in self.pairs(pb["type"])],
                                           minimum_distance_from_collisions=0.005, collision_detection_distance=0.05)]
    for it in range(iters):
      try:
        v = mink.solve_ik(cfg, tasks, 0.02, "daqp", damping=1e-3, limits=limits)
      except Exception as e:
        return dict(pb, ok_ik=False, err=str(e)[:80])
      cfg.integrate_inplace(v, 0.02)
      if it % 10 == 9:
        e = max(np.linalg.norm(ht.compute_error(cfg)[:3]) for ht in hand_tasks.values())
        if e < 5e-4:
          break
    return self.evaluate(pb, cfg.q.copy(), targets, site, obj_c, ["left", "right"], it + 1)


_S = None


def _work(pb):
  global _S
  if _S is None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _S = FistSolver()
  try:
    return _S.solve(pb)
  except Exception as e:
    return dict(pb, ok_ik=False, err=repr(e)[:120])


def summary(rows: list[dict]) -> dict:
  """Frozen rules B: hold-point pass per (arm, w) = c1-c4 and the 3 kg load ok; reach map per (arm, t): object widths with
  at least one xo passing at 3 kg (and at 0.2 kg)."""
  out = {}
  for typ in SITE:
    rs = [r for r in rows if r["type"] == typ]
    hold = {str(r["w"]): bool(r.get("loads", {}).get("3.0", {}).get("ok", False)) for r in rs if r.get("hold")}
    reach = {}
    for t in C2.HEIGHTS:
      reach[str(t)] = {L: sorted({r["w"] for r in rs if not r.get("hold") and r["t"] == t
                                  and r.get("loads", {}).get(L, {}).get("ok", False)}) for L in ("0.2", "3.0")}
    out[typ] = dict(hold_point_3kg=hold, reach_widths=reach)
  return out


def main(argv):
  out = argv[0]
  workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 24
  limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
  pbs = problems(limit)
  t0 = time.time()
  from multiprocessing import Pool
  with Pool(workers) as pool:
    rows = list(pool.imap_unordered(_work, pbs, chunksize=2))
  res = dict(rules="rounds/R04/positive/r11_hold/fist_checks_rules.md B", n=len(rows), wall_s=time.time() - t0,
             summary=summary(rows), rows=rows)
  json.dump(res, open(out, "w"))
  print(json.dumps(res["summary"], indent=0)[:3000])
  print("wrote", out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
  main(sys.argv[1:])
