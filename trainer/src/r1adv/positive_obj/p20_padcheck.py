"""Contact-pad check (coordinator 14:3x, rules_amend A1; check 1 style): hand = palm box (+ 3 x 3 sphere pads 1.5 mm proud,
p20_bodies.BOX_PADS / PALM_PADS) on mocap bodies squeezing an object from both sides, face tilted 0 / 1 / 5 deg about each
in-face axis, pad surfaces 1 mm (pads only) or 3 mm (pads + palm face) inside the object. Contacts per hand in C MuJoCo and
MuJoCo Warp (warp CPU device).

  python -m r1adv.positive_obj.p20_padcheck <out.json>
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import mujoco
import numpy as np

from r1adv.positive_obj import p20_bodies as B
from r1adv.positive_obj import p20_check1 as K
from r1adv.positive_obj import p20_scene as PS

HANDS = {"box": ((0.07, 0.045, 0.02), B.BOX_PADS), "palm": (B.PALM_HALF, B.PALM_PADS)}
OBJ = {"box": (0.2, 0.2, 0.2), "cylinder": (0.1, 0.1, 0.2), "capsule": (0.3, 0.1, 0.1)}


def model(hand: str, pads: bool, obj: str) -> mujoco.MjModel:
  spec = K.base_spec()
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
  half, padxy = HANDS[hand]
  for s, nm in ((1, "hL"), (-1, "hR")):
    b = spec.worldbody.add_body(name=nm, mocap=True, pos=[0, 0.3 * s, 0])
    b.add_geom(name=f"{nm}_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), condim=1)
    if pads:
      zc = half[2] + B.PAD_PROUD - B.PAD_R
      for i, (x, y) in enumerate(padxy):
        b.add_geom(name=f"{nm}_pad{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[B.PAD_R, 0, 0], pos=[x, y, zc], condim=1)
  osp = PS.object_spec(obj, 3, OBJ[obj], all_meshes=False)
  ob = spec.worldbody.add_body(name="objb")
  ob.add_freejoint(name="objfree")
  # copy the object geom (mesh asset first)
  for ms in osp.meshes:
    spec.add_mesh(name=ms.name, uservert=list(ms.uservert))
  g = osp.geoms[0]
  kw = dict(name="obj", type=g.type, size=list(g.size), mass=1.0, priority=2, condim=3, friction=[0.7, 0.005, 0.0001])
  if g.type == mujoco.mjtGeom.mjGEOM_MESH:
    kw["meshname"] = g.meshname
  ob.add_geom(**kw)
  return spec.compile()


def surface_y(obj: str) -> float:
  if obj == "box":
    return 0.1
  # mesh with a face normal on +-y (phase pi/16 -> faces at multiples of pi/8: +y is a face)
  r = OBJ[obj][1] / 2
  return r * math.cos(math.pi / 16)


def run(out: str) -> None:
  import warp as wp
  import mujoco_warp as mjw
  wp.config.quiet = True
  rows = []
  for hand, pads, obj in itertools.product(HANDS, (False, True), OBJ):
    m = model(hand, pads, obj)
    half = HANDS[hand][0]
    hand_geoms = {k: [g for g in range(m.ngeom) if m.geom(g).name.startswith(f"{nm}_")] for k, nm in ((0, "hL"), (1, "hR"))}
    go = m.geom("obj").id
    confs = [(depth, ax, deg) for depth in (0.001, 0.003) for ax, deg in (("none", 0), ("a", 1), ("b", 1), ("a", 5), ("b", 5))]
    n = len(confs)
    mpos, mquat = np.zeros((n, 2, 3)), np.zeros((n, 2, 4))
    qpos = np.zeros((n, m.nq))
    qpos[:, 3] = 1.0
    face_out = half[2] + (B.PAD_PROUD if pads else 0.0)       # outermost contact surface along the normal
    for w, (depth, ax, deg) in enumerate(confs):
      for j, side in enumerate((1, -1)):
        nvec = np.array([0.0, -side, 0.0])
        R = np.c_[[1, 0, 0], np.cross(nvec, [1, 0, 0]), nvec]     # box x forward, z = face normal toward the object
        if deg:
          axw = R[:, 0] if ax == "a" else R[:, 1]
          a = math.radians(deg)
          Kx = np.array([[0, -axw[2], axw[1]], [axw[2], 0, -axw[0]], [-axw[1], axw[0], 0]])
          R = (np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx) @ R
        face_c = np.array([0.0, side * (surface_y(obj) - depth), 0.0])
        mpos[w, j] = face_c - face_out * R[:, 2]
        mquat[w, j] = K.quat_from_mat(R)
    d = mujoco.MjData(m)
    cres = []
    for w in range(n):
      mujoco.mj_resetData(m, d)
      d.qpos[:] = qpos[w]
      d.mocap_pos[:] = mpos[w]
      d.mocap_quat[:] = mquat[w]
      mujoco.mj_forward(m, d)
      cnt = [sum(1 for i in range(d.ncon) if go in (d.contact[i].geom1, d.contact[i].geom2)
                 and ({int(d.contact[i].geom1), int(d.contact[i].geom2)} - {go}) <= set(hand_geoms[k])) for k in (0, 1)]
      cres.append(cnt)
    with wp.ScopedDevice("cpu"):
      mw = mjw.put_model(m)
      dw = mjw.put_data(m, d, nworld=n, nconmax=96, njmax=512)
      dw.qpos.assign(qpos.astype(np.float32))
      dw.mocap_pos.assign(mpos.astype(np.float32))
      dw.mocap_quat.assign(mquat.astype(np.float32))
      mjw.forward(mw, dw)
      na = int(dw.nacon.numpy()[0])
      geom = dw.contact.geom.numpy()[:na]
      wid = dw.contact.worldid.numpy()[:na]
      pos = dw.contact.pos.numpy()[:na]
    for w, (depth, ax, deg) in enumerate(confs):
      res = {}
      for k in (0, 1):
        hg = set(hand_geoms[k])
        sel = (wid == w) & (((geom[:, 0] == go) & np.isin(geom[:, 1], list(hg))) | ((geom[:, 1] == go) & np.isin(geom[:, 0], list(hg))))
        pts = pos[sel]
        res[k] = (int(sel.sum()), float(max((np.linalg.norm(a - b) for a in pts for b in pts), default=0.0)))
      rows.append(dict(hand=hand, pads=pads, obj=obj, depth=depth, axis=ax, deg=deg, c_L=cres[w][0], w_L=res[0][0], w_sep_L=res[0][1],
                       w_R=res[1][0]))
    print(hand, "pads" if pads else "nopad", obj, [(r["depth"], r["axis"], r["deg"], r["c_L"], r["w_L"]) for r in rows[-n:]], flush=True)
  json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
  run(sys.argv[1])
