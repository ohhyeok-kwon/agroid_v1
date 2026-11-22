"""Positive 20 check 1 (rounds/R04/positive/r11_contact2/checks_rules.md section 1): contact counts of hand / finger
shapes x object shapes in a static squeeze (C MuJoCo vs MuJoCo Warp), and 1 s standing stability of objects on a table
box. MuJoCo Warp runs on the warp CPU device (same kernels as CUDA).

  python -m r1adv.positive_obj.p20_check1 <out.json> [squeeze|table|all]
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time

import mujoco
import numpy as np

OPT = dict(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50, tolerance=1e-8, ls_tolerance=0.01)
DELTA = 0.001                                   # squeeze penetration at the face centre
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
OBJ_R, OBJ_HH = 0.05, 0.10                     # object radius / half height for the squeeze cells
HANDS = {                                       # name: (geom type, size, contact half-thickness along the face normal)
  "palm_box": ("box", (0.07, 0.045, 0.02)),
  "wrist_cap": ("capsule", (0.03, 0.035)),
  "fin_cap": ("capsule", (0.006, 0.0185)),
  "fin_box": ("box", (0.0185, 0.006, 0.007)),
  "fin_mesh": ("mesh", None),
}
MESH_N = (16, 24, 32)


def prism_verts(n: int, r: float, hh: float, phase: float = 0.0) -> np.ndarray:
  """Regular n-gon prism along z, circumradius r, half height hh."""
  a = phase + 2 * np.pi * np.arange(n) / n
  ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
  return np.concatenate([np.c_[ring, -hh * np.ones(n)], np.c_[ring, hh * np.ones(n)]])


def finger_mesh_verts() -> np.ndarray:
  """Octagonal prism finger: axis x, half length 0.0185, apothem 0.006 (12 mm across flats), a face normal on +-z."""
  n, ap, hl = 8, 0.006, 0.0185
  rc = ap / math.cos(math.pi / n)
  a = math.pi / n + 2 * np.pi * np.arange(n) / n           # faces at 0, 45, ... deg: a face normal along +z (90 deg)
  ring = np.stack([rc * np.cos(a), rc * np.sin(a)], 1)
  return np.concatenate([np.c_[-hl * np.ones(n), ring], np.c_[hl * np.ones(n), ring]])


def quat_from_mat(R: np.ndarray) -> list:
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
  return q.tolist()


def rot(axis: str, ang: float) -> np.ndarray:
  c, s = math.cos(ang), math.sin(ang)
  return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
          "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def add_hand_geom(spec, body, hand: str, name: str):
  gtype, size = HANDS[hand]
  kw = dict(name=name, contype=1, conaffinity=1, condim=3)
  if gtype == "box":
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), **kw)
  elif gtype == "capsule":
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[size[0], size[1], 0], **kw)
  else:
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="finmesh", **kw)


def add_object_geom(spec, body, obj: str, dims=None, mass=1.0, condim=3, mu=0.7, tc=0.02):
  """obj in box, cyl, cap, mcyl16/24/32. dims: (r or half x, ..)."""
  kw = dict(name="obj", mass=mass, priority=2, condim=condim, friction=[mu, 0.005, 0.0001], solref=[tc, 1.0],
            solimp=list(SOLIMP), contype=1, conaffinity=1)
  if obj == "box":
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(dims), **kw)
  elif obj == "cyl":
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[dims[0], dims[1], 0], **kw)
  elif obj == "cap":                                    # lying along x: geom z -> body x
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[dims[0], dims[1], 0],
                  quat=[math.sqrt(0.5), 0, math.sqrt(0.5), 0], **kw)
  else:
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"objmesh_{obj}", **kw)


def base_spec(cone: str = "pyramidal") -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.option.timestep = OPT["timestep"]
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.iterations = OPT["iterations"]
  spec.option.ls_iterations = OPT["ls_iterations"]
  spec.option.ccd_iterations = OPT["ccd_iterations"]
  spec.option.tolerance = OPT["tolerance"]
  spec.option.ls_tolerance = OPT["ls_tolerance"]
  spec.add_mesh(name="finmesh", uservert=finger_mesh_verts().reshape(-1).tolist())
  return spec


# ----------------------------------------------------------------------------------------------- squeeze contact counts
def squeeze_cells() -> list[dict]:
  """Per (hand, object) model: a list of world configurations (hand pose about its face centre, object yaw)."""
  objs = ["box", "cyl", "cap"] + [f"mcyl{n}" for n in MESH_N]
  cells = []
  for hand, obj in itertools.product(HANDS, objs):
    confs = []
    orients = ["along_x"] if hand in ("palm_box", "wrist_cap") else ["axis_vertical", "axis_x"]
    phases = ["face", "edge"] if obj.startswith("mcyl") else ["face"]
    for ori, ph, (tax, tdeg) in itertools.product(orients, phases, [("none", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2)]):
      confs.append(dict(orient=ori, phase=ph, tilt_axis=tax, tilt_deg=tdeg))
    cells.append(dict(hand=hand, obj=obj, confs=confs))
  return cells


def expected_multi(hand: str, obj: str, orient: str) -> bool:
  """checks_rules 1: cells whose true contact set is a line or an area."""
  if hand == "wrist_cap":
    return False                                            # reference only
  if hand == "palm_box":
    return True
  if obj == "box":
    return True
  if obj == "cap":
    return orient == "axis_x"                                # finger along the lying capsule's axis
  return orient == "axis_vertical"                          # cylinders: finger along the generator


def hand_frame(hand: str, orient: str, side: int) -> tuple[np.ndarray, float]:
  """Rotation of the hand geom (columns = geom axes in world) with its contact face normal pointing to the object
  (side +1: hand at +y, normal -y), and the geom's half-thickness along that normal."""
  gtype, size = HANDS[hand]
  n = np.array([0.0, -side, 0.0])
  if hand == "palm_box":                                    # box x along +x (forearm), box z = face normal
    R = np.c_[[1, 0, 0], np.cross(n, [1, 0, 0]), n]
    return R, size[2]
  if hand == "wrist_cap":                                   # capsule axis (geom z) along +x, geom x = face normal
    ax = np.array([1.0, 0, 0])
    R = np.c_[n, np.cross(ax, n), ax]
    return R, size[0]
  long_ax = np.array([0, 0, 1.0]) if orient == "axis_vertical" else np.array([1.0, 0, 0])
  if gtype == "capsule":                                    # geom z = long axis
    x = n
    y = np.cross(long_ax, x)
    R = np.c_[x, y, long_ax]
    return R, size[0]
  # box / mesh finger: geom x = long axis, geom z = face normal
  z = n
  y = np.cross(z, long_ax)
  R = np.c_[long_ax, y, z]
  return R, (size[2] if gtype == "box" else 0.006)


def object_support(obj: str, phase: str) -> float:
  """Distance from the object axis to its surface along +-y for the squeeze cells."""
  if obj == "box":
    return OBJ_R
  if obj in ("cyl", "cap"):
    return OBJ_R
  n = int(obj[4:])
  return OBJ_R if phase == "edge" else OBJ_R * math.cos(math.pi / n)


def squeeze_model(hand: str, obj: str) -> mujoco.MjModel:
  spec = base_spec()
  for n in MESH_N:
    spec.add_mesh(name=f"objmesh_mcyl{n}", uservert=prism_verts(n, OBJ_R, OBJ_HH).reshape(-1).tolist())
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY
  for s, nm in ((1, "hL"), (-1, "hR")):
    b = spec.worldbody.add_body(name=nm, mocap=True, pos=[0, 0.3 * s, 0])
    add_hand_geom(spec, b, hand, f"geom_{nm}")
  ob = spec.worldbody.add_body(name="objb")
  ob.add_freejoint(name="objfree")
  dims = {"box": (OBJ_R, OBJ_R, OBJ_HH), "cyl": (OBJ_R, OBJ_HH), "cap": (OBJ_R, 0.10)}.get(obj)
  add_object_geom(spec, ob, obj, dims)
  return spec.compile()


def conf_pose(hand: str, obj: str, c: dict, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Mocap pos, quat of one hand and the object yaw quat for configuration c."""
  R, half = hand_frame(hand, c["orient"], side)
  n = np.array([0.0, -side, 0.0])
  sup = object_support(obj, c["phase"])
  face_c = np.array([0.0, side * (sup - DELTA), 0.0])     # contact face centre (penetrating DELTA)
  if c["tilt_deg"]:
    ax_world = R[:, 0] if c["tilt_axis"] == "a" else R[:, 1]   # in-face axes (geom x / geom y; for capsules x = normal)
    if HANDS[hand][0] == "capsule":
      ax_world = R[:, 2] if c["tilt_axis"] == "a" else R[:, 1]
    ax_world = ax_world - (ax_world @ n) * n
    ax_world /= np.linalg.norm(ax_world)
    ang = math.radians(c["tilt_deg"])
    K = np.array([[0, -ax_world[2], ax_world[1]], [ax_world[2], 0, -ax_world[0]], [-ax_world[1], ax_world[0], 0]])
    Rt = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    R = Rt @ R
  # the geom centre sits 'half' behind the face centre along the (possibly tilted) normal axis
  nrm = R[:, 2] if HANDS[hand][0] != "capsule" else R[:, 0]
  centre = face_c - half * nrm
  yaw = 0.0
  if obj.startswith("mcyl"):
    n_sides = int(obj[4:])
    # vertices at angles 2 pi k / n: a vertex points to +y when yaw = pi/2; a face normal points to +y when yaw = pi/2 + pi/n
    yaw = math.pi / 2 + (math.pi / n_sides if c["phase"] == "face" else 0.0)
  oq = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
  return centre, np.array(quat_from_mat(R)), oq


def count_c(m, d, gL, gR, gO):
  out = {}
  for k, g in (("L", gL), ("R", gR)):
    pts = [d.contact[i].pos.copy() for i in range(d.ncon) if {int(d.contact[i].geom1), int(d.contact[i].geom2)} == {g, gO}]
    sep = max((np.linalg.norm(a - b) for a in pts for b in pts), default=0.0)
    out[k] = (len(pts), float(sep))
  return out


def run_squeeze() -> list[dict]:
  import warp as wp
  import mujoco_warp as mjw
  wp.config.quiet = True
  rows = []
  for cell in squeeze_cells():
    hand, obj = cell["hand"], cell["obj"]
    m = squeeze_model(hand, obj)
    gL, gR, gO = m.geom("geom_hL").id, m.geom("geom_hR").id, m.geom("obj").id
    d = mujoco.MjData(m)
    n = len(cell["confs"])
    mpos = np.zeros((n, 2, 3)); mquat = np.zeros((n, 2, 4)); qpos = np.zeros((n, m.nq))
    res_c = []
    for w, c in enumerate(cell["confs"]):
      for j, side in enumerate((1, -1)):
        p, q, oq = conf_pose(hand, obj, c, side)
        mpos[w, j], mquat[w, j] = p, q
      qpos[w, :3] = 0.0
      qpos[w, 3:7] = oq
      mujoco.mj_resetData(m, d)
      d.qpos[:] = qpos[w]
      d.mocap_pos[:] = mpos[w]
      d.mocap_quat[:] = mquat[w]
      mujoco.mj_forward(m, d)
      res_c.append(count_c(m, d, gL, gR, gO))
    with wp.ScopedDevice("cpu"):
      mw = mjw.put_model(m)
      dw = mjw.put_data(m, d, nworld=n, nconmax=64, njmax=256)
      dw.qpos.assign(qpos.astype(np.float32))
      dw.mocap_pos.assign(mpos.astype(np.float32))
      dw.mocap_quat.assign(mquat.astype(np.float32))
      mjw.forward(mw, dw)
      na = int(dw.nacon.numpy()[0])
      geom = dw.contact.geom.numpy()[:na]
      wid = dw.contact.worldid.numpy()[:na]
      pos = dw.contact.pos.numpy()[:na]
    for w, c in enumerate(cell["confs"]):
      rw = {}
      for k, g in (("L", gL), ("R", gR)):
        sel = (wid == w) & (((geom[:, 0] == g) & (geom[:, 1] == gO)) | ((geom[:, 1] == g) & (geom[:, 0] == gO)))
        pts = pos[sel]
        sep = float(max((np.linalg.norm(a - b) for a in pts for b in pts), default=0.0))
        rw[k] = (int(sel.sum()), sep)
      rows.append(dict(hand=hand, obj=obj, **c, expected_multi=expected_multi(hand, obj, c["orient"]),
                       c_L=res_c[w]["L"][0], c_sep_L=res_c[w]["L"][1], c_R=res_c[w]["R"][0],
                       w_L=rw["L"][0], w_sep_L=rw["L"][1], w_R=rw["R"][0], w_sep_R=rw["R"][1]))
    print(f"{hand:9s} {obj:7s} done ({n} confs)", flush=True)
  return rows


# ------------------------------------------------------------------------------------------------ table standing test
def table_cases() -> list[dict]:
  cases = []
  cyl_sizes = [(0.05, 0.20), (0.10, 0.10), (0.10, 0.40), (0.20, 0.20), (0.40, 0.10), (0.40, 0.40)]
  for obj in ["cyl"] + [f"mcyl{n}" for n in MESH_N]:
    for dd, h in cyl_sizes:
      cases.append(dict(obj=obj, dims=(dd / 2, h / 2), label=f"d{dd:.2f}h{h:.2f}"))
  for bx in [(0.05, 0.05, 0.40), (0.10, 0.10, 0.10), (0.10, 0.30, 0.20), (0.40, 0.40, 0.40)]:
    cases.append(dict(obj="box", dims=tuple(v / 2 for v in bx), label="b" + "x".join(f"{v:.2f}" for v in bx)))
  for dd in (0.05, 0.10, 0.40):
    L = max(dd, 0.30)
    cases.append(dict(obj="cap", dims=(dd / 2, max(L / 2 - dd / 2, 1e-4)), label=f"cap_d{dd:.2f}"))
  out = []
  for c, mass, st in itertools.product(cases, (1.0, 3.0), (("pyramidal", 3, 1.0), ("elliptic", 4, 10.0))):
    out.append(dict(c, mass=mass, cone=st[0], condim=st[1], impratio=st[2]))
  return out


def rest_half_height(obj: str, dims) -> float:
  return dims[0] if obj == "cap" else (dims[2] if obj == "box" else dims[1])


def table_model(case: dict) -> mujoco.MjModel:
  spec = base_spec(case["cone"])
  spec.option.impratio = case["impratio"]
  obj = case["obj"]
  if obj.startswith("mcyl"):
    n = int(obj[4:])
    spec.add_mesh(name=f"objmesh_{obj}", uservert=prism_verts(n, case["dims"][0], case["dims"][1]).reshape(-1).tolist())
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
  tb = spec.worldbody.add_body(name="table", mocap=True, pos=[0, 0, 0.70])
  tb.add_geom(name="top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.30, 0.60, 0.02], pos=[0, 0, -0.02], condim=3,
              friction=[1.0, 0.005, 0.0001])
  ob = spec.worldbody.add_body(name="objb", pos=[0, 0, 0.70 + rest_half_height(obj, case["dims"]) + 0.001])
  ob.add_freejoint(name="objfree")
  dims = case["dims"]
  add_object_geom(spec, ob, obj, dims if obj != "box" else dims, mass=case["mass"], condim=case["condim"])
  return spec.compile()


def tilt_deg(q: np.ndarray, obj: str) -> float:
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, q)
  R = R.reshape(3, 3)
  ax = R[:, 0] if obj == "cap" else R[:, 2]                # lying capsule: its axis is body x (horizontal at rest)
  if obj == "cap":
    return float(np.degrees(np.arcsin(np.clip(abs(ax[2]), 0, 1))))
  return float(np.degrees(np.arccos(np.clip(abs(ax[2]), 0, 1))))


def run_table() -> list[dict]:
  import warp as wp
  import mujoco_warp as mjw
  wp.config.quiet = True
  rows = []
  for case in table_cases():
    m = table_model(case)
    hh = rest_half_height(case["obj"], case["dims"])
    z_rest = 0.70 + hh
    res = {}
    # C
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p0 = d.qpos[:3].copy()
    tmax = 0.0
    for _ in range(200):
      mujoco.mj_step(m, d)
      tmax = max(tmax, tilt_deg(d.qpos[3:7], case["obj"]))
    ncon_c = sum(1 for i in range(d.ncon) if m.geom(int(d.contact[i].geom1)).name in ("top", "obj")
                 and m.geom(int(d.contact[i].geom2)).name in ("top", "obj"))
    res["c"] = dict(tilt=tmax, dz=1000 * (d.qpos[2] - z_rest), drift=1000 * float(np.linalg.norm(d.qpos[:2] - p0[:2])),
                    ncon=ncon_c)
    # MJWarp (CPU device)
    d2 = mujoco.MjData(m)
    mujoco.mj_forward(m, d2)
    with wp.ScopedDevice("cpu"):
      mw = mjw.put_model(m)
      dw = mjw.put_data(m, d2, nworld=1, nconmax=64, njmax=256)
      tmax = 0.0
      for _ in range(200):
        mjw.step(mw, dw)
        q = dw.qpos.numpy()[0]
        tmax = max(tmax, tilt_deg(q[3:7].astype(np.float64), case["obj"]))
      na = int(dw.nacon.numpy()[0])
    res["w"] = dict(tilt=tmax, dz=1000 * (float(q[2]) - z_rest), drift=1000 * float(np.linalg.norm(q[:2] - p0[:2])),
                    ncon=na)
    rows.append(dict(obj=case["obj"], label=case["label"], mass=case["mass"], cone=case["cone"], condim=case["condim"],
                     impratio=case["impratio"], **{f"{e}_{k}": v for e in ("c", "w") for k, v in res[e].items()}))
    r = rows[-1]
    print(f"{case['obj']:7s} {case['label']:18s} m{case['mass']:.0f} {case['cone'][:2]}{case['condim']}: "
          f"C tilt {r['c_tilt']:.2f} dz {r['c_dz']:.2f} drift {r['c_drift']:.2f} n{r['c_ncon']} | "
          f"W tilt {r['w_tilt']:.2f} dz {r['w_dz']:.2f} drift {r['w_drift']:.2f} n{r['w_ncon']}", flush=True)
  return rows


def main(argv):
  out, what = argv[0], (argv[1] if len(argv) > 1 else "all")
  t0 = time.time()
  res = {"rules": "rounds/R04/positive/r11_contact2/checks_rules.md", "mujoco": mujoco.__version__}
  if what in ("squeeze", "all"):
    res["squeeze"] = run_squeeze()
  if what in ("table", "all"):
    res["table"] = run_table()
  res["wall_s"] = time.time() - t0
  with open(out, "w") as f:
    json.dump(res, f, indent=1)
  print("wrote", out, f"{res['wall_s']:.0f} s")


if __name__ == "__main__":
  main(sys.argv[1:])
