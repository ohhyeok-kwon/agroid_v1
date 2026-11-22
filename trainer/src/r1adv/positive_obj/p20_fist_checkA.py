"""Positive 20 fist check A (rounds/R04/positive/r11_hold/fist_checks_rules.md section A): static contact counts of the
fist arms' hand geoms (fist capsule, fitted fist box + wrist capsule, + pads) against the hold-test object shapes (box,
mesh cylinder, mesh capsule) at hand tilts 0, 1, 5 deg about the two in-face axes; C MuJoCo (mj_forward) vs MuJoCo Warp
(forward on the warp CPU device). Hands are mocap bodies; the hand patch point penetrates the object by DELTA.

  python -m r1adv.positive_obj.p20_fist_checkA <out.json>
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import mujoco
import numpy as np

from r1adv.positive_obj import p20_check1 as C1
from r1adv.positive_obj import p20_fist as F
from r1adv.positive_obj import p20_scene as PS

DELTA = 0.001
OBJS = {"box": (0.25, 0.30, 0.25), "cylinder": (0.30, 0.30, 0.25), "capsule": (0.35, 0.25, 0.25),   # hold-test mid sizes
        "capsule_up": (0.35, 0.25, 0.25)}                       # addendum A': the H1V capsule, upright (rules.md 4)
Q_UP = [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]
TILTS = [("none", 0.0), ("a", 1.0), ("b", 1.0), ("a", 5.0), ("b", 5.0)]
R_SIDE = {1: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),      # left hand at +y: frame z -> -y
          -1: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])}
# frozen expectation (rules A): cells whose true penetrating contact set is a line or an area (>= 1 cm long) at every tilt;
# only the stock capsule against the upright cylinder (crossed axes) is a point contact
EXPECT_MULTI = {(arm, obj): not (arm == "fist" and obj in ("cylinder", "capsule_up")) for arm in F.ARMS for obj in OBJS}


def hand_geoms(arm: str, side: int) -> list[dict]:
  """Hand collision geoms of one side in the <side>_o6_hand frame (side +1 left, -1 right)."""
  x0 = F.FIST_FRAME[0]
  out = []
  cap = F.STOCK_CAP if arm == "fist" else F.WRIST_CAP
  a, b = np.array(cap[0][:3]) - [x0, 0, 0], np.array(cap[0][3:]) - [x0, 0, 0]
  out.append(dict(type="capsule", fromto=np.r_[a, b], size=cap[1]))
  if arm != "fist":
    out.append(dict(type="box", pos=np.array([0.0, side * F.FIST_C[1], F.FIST_C[2]]), size=np.array(F.FIST_HALF)))
  if arm == "fistpads":
    zc = F.FIST_C[2] + F.FIST_HALF[2] + F.PAD_PROUD - F.PAD_R
    for x, y in F.PADS:
      out.append(dict(type="sphere", pos=np.array([x, side * (F.FIST_C[1] + y), zc]), size=F.PAD_R))
  return out


def patch_local(arm: str, side: int) -> np.ndarray:
  dx, dy = F.PATCH[arm]
  return np.array([dx, side * dy, F.R_FACE[arm]])


def build(arm: str, obj: str) -> mujoco.MjModel:
  spec = C1.base_spec()
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
  dims = OBJS[obj]
  if obj == "cylinder":
    spec.add_mesh(name="om", uservert=PS.cyl_verts(dims[0], dims[2]).reshape(-1).tolist())
  elif obj in ("capsule", "capsule_up"):
    spec.add_mesh(name="om", uservert=PS.cap_verts(dims[1], dims[0]).reshape(-1).tolist())
  for side, nm in ((1, "hL"), (-1, "hR")):
    b = spec.worldbody.add_body(name=nm, mocap=True, pos=[0, 0.3 * side, 0])
    for k, g in enumerate(hand_geoms(arm, side)):
      kw = dict(name=f"{nm}_{k}", contype=1, conaffinity=1, condim=1)
      if g["type"] == "capsule":
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=g["fromto"].tolist(), size=[g["size"], 0, 0], **kw)
      elif g["type"] == "box":
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=g["pos"].tolist(), size=g["size"].tolist(), **kw)
      else:
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=g["pos"].tolist(), size=[g["size"], 0, 0], **kw)
  ob = spec.worldbody.add_body(name="objb")
  ob.add_freejoint(name="objfree")
  kw = dict(name="obj", mass=1.0, priority=2, condim=3, friction=[0.7, 0.005, 0.0001], solref=[0.02, 1.0],
            solimp=list(C1.SOLIMP), contype=1, conaffinity=1)
  if obj == "box":
    ob.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[d / 2 for d in dims], **kw)
  elif obj == "capsule_up":
    ob.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="om", quat=Q_UP, **kw)
  else:
    ob.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="om", **kw)
  return spec.compile()


def support_y(m: mujoco.MjModel, obj: str) -> float:
  if obj == "box":
    return OBJS[obj][1] / 2
  g = m.geom("obj").id
  mid = m.geom_dataid[g]
  V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, m.geom_quat[g])
  Vb = V @ R.reshape(3, 3).T + m.geom_pos[g]
  return float(Vb[:, 1].max())


def pose(arm: str, side: int, sup: float, tilt) -> tuple[np.ndarray, np.ndarray]:
  R = R_SIDE[side].copy()
  ax, deg = tilt
  if deg:
    a = R[:, 0] if ax == "a" else R[:, 1]
    ang = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = (np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K) @ R
  p_world = np.array([0.0, side * (sup - DELTA), 0.0])
  centre = p_world - R @ patch_local(arm, side)
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, R.reshape(-1))
  return centre, q


def contacts_of(geom, pos, wid, w, gs, gO):
  sel = (wid == w) & ((np.isin(geom[:, 0], gs) & (geom[:, 1] == gO)) | (np.isin(geom[:, 1], gs) & (geom[:, 0] == gO)))
  pts = pos[sel]
  sep = float(max((np.linalg.norm(a - b) for a in pts for b in pts), default=0.0))
  return int(sel.sum()), sep


def run(objs=tuple(OBJS)) -> list[dict]:
  import mujoco_warp as mjw
  import warp as wp
  wp.config.quiet = True
  rows = []
  for arm, obj in itertools.product(F.ARMS, objs):
    m = build(arm, obj)
    d = mujoco.MjData(m)
    gs = {k: [g for g in range(m.ngeom) if m.geom(g).name.startswith(nm + "_")] for k, nm in (("L", "hL"), ("R", "hR"))}
    gO = m.geom("obj").id
    sup = support_y(m, obj)
    n = len(TILTS)
    mpos, mquat, res_c = np.zeros((n, 2, 3)), np.zeros((n, 2, 4)), []
    for w, tilt in enumerate(TILTS):
      for j, side in enumerate((1, -1)):
        mpos[w, j], mquat[w, j] = pose(arm, side, sup, tilt)
      mujoco.mj_resetData(m, d)
      d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
      d.mocap_pos[:], d.mocap_quat[:] = mpos[w], mquat[w]
      mujoco.mj_forward(m, d)
      geom = np.array([[d.contact[i].geom1, d.contact[i].geom2] for i in range(d.ncon)]).reshape(-1, 2)
      pos = np.array([d.contact[i].pos for i in range(d.ncon)]).reshape(-1, 3)
      wid = np.zeros(len(geom), int)
      res_c.append({k: contacts_of(geom, pos, wid, 0, gs[k], gO) for k in ("L", "R")})
    qpos = np.tile(d.qpos * 0, (n, 1))
    qpos[:, 3] = 1.0
    with wp.ScopedDevice("cpu"):
      mw = mjw.put_model(m)
      dw = mjw.put_data(m, d, nworld=n, nconmax=256, njmax=1024)
      dw.qpos.assign(qpos.astype(np.float32))
      dw.mocap_pos.assign(mpos.astype(np.float32))
      dw.mocap_quat.assign(mquat.astype(np.float32))
      mjw.forward(mw, dw)
      na = int(dw.nacon.numpy()[0])
      geom, wid, pos = dw.contact.geom.numpy()[:na], dw.contact.worldid.numpy()[:na], dw.contact.pos.numpy()[:na]
    for w, tilt in enumerate(TILTS):
      rw = {k: contacts_of(geom, pos, wid, w, gs[k], gO) for k in ("L", "R")}
      rows.append(dict(arm=arm, obj=obj, tilt_axis=tilt[0], tilt_deg=tilt[1], expect_multi=EXPECT_MULTI[(arm, obj)],
                       c_L=res_c[w]["L"][0], c_sep_L=round(res_c[w]["L"][1], 4), c_R=res_c[w]["R"][0], c_sep_R=round(res_c[w]["R"][1], 4),
                       w_L=rw["L"][0], w_sep_L=round(rw["L"][1], 4), w_R=rw["R"][0], w_sep_R=round(rw["R"][1], 4)))
    print(arm, obj, "done", flush=True)
  return rows


def judge(rows: list[dict]) -> dict:
  """Frozen rules A: A1 agreement share (MJWarp count == C count, per hand side); A2 per arm x object: MJWarp >= 2
  contacts with spread >= 1 cm on both hands at every tilt where EXPECT_MULTI."""
  out = {}
  for arm in F.ARMS:
    rs = [r for r in rows if r["arm"] == arm]
    agree = np.mean([r["w_L"] == r["c_L"] for r in rs] + [r["w_R"] == r["c_R"] for r in rs])
    per = {}
    for obj in sorted({r["obj"] for r in rs}):
      ro = [r for r in rs if r["obj"] == obj]
      if EXPECT_MULTI[(arm, obj)]:
        ok = all(r["w_L"] >= 2 and r["w_R"] >= 2 and r["w_sep_L"] >= 0.01 and r["w_sep_R"] >= 0.01 for r in ro)
        ok_c = all(r["c_L"] >= 2 and r["c_R"] >= 2 and r["c_sep_L"] >= 0.01 and r["c_sep_R"] >= 0.01 for r in ro)
        per[obj] = dict(A2_mjwarp=bool(ok), A2_c=bool(ok_c))
      else:
        per[obj] = dict(A2_mjwarp=None, A2_c=None, note="point contact expected")
    out[arm] = dict(A1_agree=float(agree), A2=per)
  return out


if __name__ == "__main__":
  rows = run(tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else tuple(OBJS))
  res = dict(rules="rounds/R04/positive/r11_hold/fist_checks_rules.md A", rows=rows, judge=judge(rows))
  json.dump(res, open(sys.argv[1], "w"), indent=1)
  print(json.dumps(res["judge"], indent=1))
