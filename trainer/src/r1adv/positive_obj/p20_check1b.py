"""Positive 20 check 1 amendment A1b (rounds/R04/positive/r11_contact2/checks_amend.md): a lying convex-mesh capsule
(16 sides around its axis, hemispherical ends approximated by rings) measured with the check 1 squeeze and table tests.

  python -m r1adv.positive_obj.p20_check1b <out.json>
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import mujoco
import numpy as np

from r1adv.positive_obj import p20_check1 as K

N_AROUND = 16
CAP_RINGS = (22.5, 45.0, 67.5)                   # hemisphere rings (deg from the cylinder end); tips added


def mcap_verts(r: float, L: float, phase: float = 0.0) -> np.ndarray:
  """Convex lying capsule along x, total length L, radius r; ring vertex angles phase + 2 pi k / N about x."""
  hl = max(L / 2 - r, 1e-4)
  a = phase + 2 * np.pi * np.arange(N_AROUND) / N_AROUND
  pts = []
  for sgn in (-1, 1):
    for th in (0.0,) + CAP_RINGS:
      t = math.radians(th)
      x = sgn * (hl + r * math.sin(t))
      rr = r * math.cos(t)
      pts += [(x, rr * math.cos(b), rr * math.sin(b)) for b in a]
    pts.append((sgn * (hl + r), 0.0, 0.0))
  return np.array(pts)


def squeeze_model_mcap(hand: str, phase: float) -> mujoco.MjModel:
  spec = K.base_spec()
  spec.add_mesh(name="objmesh_mcap16", uservert=mcap_verts(K.OBJ_R, 0.30, phase).reshape(-1).tolist())
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
  for s, nm in ((1, "hL"), (-1, "hR")):
    b = spec.worldbody.add_body(name=nm, mocap=True, pos=[0, 0.3 * s, 0])
    K.add_hand_geom(spec, b, hand, f"geom_{nm}")
  ob = spec.worldbody.add_body(name="objb")
  ob.add_freejoint(name="objfree")
  K.add_object_geom(spec, ob, "mcap16")
  return spec.compile()


def run(out: str) -> None:
  import warp as wp
  import mujoco_warp as mjw
  wp.config.quiet = True
  res = {"squeeze": [], "table": []}
  # face-aligned: a face normal on +-y -> vertex angles offset by pi/16 from y; edge-aligned: a vertex on +y
  for hand, ph in itertools.product(("palm_box", "fin_box", "fin_mesh", "fin_cap"), ("face", "edge")):
    phase = (math.pi / N_AROUND) if ph == "face" else 0.0     # angle 0 in the (y, z) ring plane is +y
    m = squeeze_model_mcap(hand, phase)
    gL, gR, gO = m.geom("geom_hL").id, m.geom("geom_hR").id, m.geom("obj").id
    sup = K.OBJ_R * (math.cos(math.pi / N_AROUND) if ph == "face" else 1.0)
    orients = ["along_x"] if hand == "palm_box" else ["axis_x", "axis_vertical"]
    confs = [dict(orient=o, tilt_axis=a, tilt_deg=t) for o in orients for a, t in (("none", 0), ("a", 1), ("b", 1))]
    d = mujoco.MjData(m)
    n = len(confs)
    mpos = np.zeros((n, 2, 3)); mquat = np.zeros((n, 2, 4)); qpos = np.zeros((n, m.nq)); qpos[:, 3] = 1.0
    rc = []
    for w, c in enumerate(confs):
      for j, side in enumerate((1, -1)):
        cc = dict(c, phase="face")
        R, half = K.hand_frame(hand, c["orient"], side)
        nvec = np.array([0.0, -side, 0.0])
        face_c = np.array([0.0, side * (sup - K.DELTA), 0.0])
        if c["tilt_deg"]:
          axw = (R[:, 0] if c["tilt_axis"] == "a" else R[:, 1]) if K.HANDS[hand][0] != "capsule" else (R[:, 2] if c["tilt_axis"] == "a" else R[:, 1])
          axw = axw - (axw @ nvec) * nvec; axw /= np.linalg.norm(axw)
          ang = math.radians(c["tilt_deg"])
          Kx = np.array([[0, -axw[2], axw[1]], [axw[2], 0, -axw[0]], [-axw[1], axw[0], 0]])
          R = (np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx) @ R
        nrm = R[:, 2] if K.HANDS[hand][0] != "capsule" else R[:, 0]
        mpos[w, j] = face_c - half * nrm
        mquat[w, j] = K.quat_from_mat(R)
      mujoco.mj_resetData(m, d)
      d.qpos[:] = qpos[w]; d.mocap_pos[:] = mpos[w]; d.mocap_quat[:] = mquat[w]
      mujoco.mj_forward(m, d)
      rc.append(K.count_c(m, d, gL, gR, gO))
    with wp.ScopedDevice("cpu"):
      mw = mjw.put_model(m)
      dw = mjw.put_data(m, d, nworld=n, nconmax=64, njmax=256)
      dw.qpos.assign(qpos.astype(np.float32)); dw.mocap_pos.assign(mpos.astype(np.float32))
      dw.mocap_quat.assign(mquat.astype(np.float32))
      mjw.forward(mw, dw)
      na = int(dw.nacon.numpy()[0]); geom = dw.contact.geom.numpy()[:na]; wid = dw.contact.worldid.numpy()[:na]
      pos = dw.contact.pos.numpy()[:na]
    for w, c in enumerate(confs):
      cnt = {}
      for k, g in (("L", gL), ("R", gR)):
        sel = (wid == w) & (((geom[:, 0] == g) & (geom[:, 1] == gO)) | ((geom[:, 1] == g) & (geom[:, 0] == gO)))
        pts = pos[sel]
        cnt[k] = (int(sel.sum()), float(max((np.linalg.norm(a - b) for a in pts for b in pts), default=0.0)))
      res["squeeze"].append(dict(hand=hand, phase=ph, **c, c_L=rc[w]["L"][0], w_L=cnt["L"][0], w_sep_L=cnt["L"][1],
                                 w_R=cnt["R"][0]))
      print(hand, ph, c, "C", rc[w]["L"][0], "W", cnt["L"], cnt["R"][0], flush=True)
  # table
  for dd, mass, st in itertools.product((0.05, 0.10, 0.40), (1.0, 3.0), (("pyramidal", 3, 1.0), ("elliptic", 4, 10.0))):
    L = max(dd, 0.30)
    spec = K.base_spec(st[0]); spec.option.impratio = st[2]
    spec.add_mesh(name="objmesh_mcap16", uservert=mcap_verts(dd / 2, L, math.pi / N_AROUND).reshape(-1).tolist())
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    tb = spec.worldbody.add_body(name="table", mocap=True, pos=[0, 0, 0.70])
    tb.add_geom(name="top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.30, 0.60, 0.02], pos=[0, 0, -0.02], condim=3,
                friction=[1.0, 0.005, 0.0001])
    zr = 0.70 + dd / 2 * math.cos(math.pi / N_AROUND)       # a flat face down (phase pi/16 puts a face normal on -z)
    ob = spec.worldbody.add_body(name="objb", pos=[0, 0, zr + 0.001])
    ob.add_freejoint(name="objfree")
    K.add_object_geom(spec, ob, "mcap16", mass=mass, condim=st[1])
    m = spec.compile()
    row = dict(d=dd, mass=mass, cone=st[0], condim=st[1])
    for eng in ("c", "w"):
      d = mujoco.MjData(m); mujoco.mj_forward(m, d); p0 = d.qpos[:3].copy(); tmax = 0.0
      if eng == "c":
        for _ in range(200):
          mujoco.mj_step(m, d); tmax = max(tmax, K.tilt_deg(d.qpos[3:7], "cap"))
        q = d.qpos.copy(); nc = d.ncon
      else:
        with wp.ScopedDevice("cpu"):
          mw = mjw.put_model(m); dw = mjw.put_data(m, d, nworld=1, nconmax=64, njmax=256)
          for _ in range(200):
            mjw.step(mw, dw); q = dw.qpos.numpy()[0].astype(np.float64); tmax = max(tmax, K.tilt_deg(q[3:7], "cap"))
          nc = int(dw.nacon.numpy()[0])
      row.update({f"{eng}_tilt": tmax, f"{eng}_dz": 1000 * (q[2] - zr), f"{eng}_drift": 1000 * float(np.linalg.norm(q[:2] - p0[:2])),
                  f"{eng}_ncon": int(nc)})
    res["table"].append(row)
    print("table", row, flush=True)
  json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
  run(sys.argv[1])
