"""Negative 40 (rough walk, H0; coordinator 11-22 08:46 / 08:54 / 09:03): seeded terrain panel on the flat ruler's conditions.
Same runner as the flat ruler (neg30_v4: b1 body via NEG30_BODY, v4.3s reference via NEG30_GEN, real-time path, tracker action
delay 15 ms, manufacturer 0 ms); this file only adds the ground. Terrain geoms are static world geoms (group 5) put into the b1
spec before its single compile (MjSpec.from_file wrapped while SimR1.__init__ runs), so the robot bodies, masses and body_hash are
untouched (neg30_body check still runs). Course frame: x along the course, obstacle starts at x = 0, robot starts at (-d0, y0)
with heading psi relative to the course; the course is placed so the robot reset (origin, yaw 0, ground z = 0) is unchanged.
Kinds (size = near / far end): thr (box 10 cm deep, h, front or +-30 deg oblique), stup / stdn (five risers h, tread 0.28 m),
rpup / rpdn (2.5 m slope at theta), side (height field z = s * y * tan(phi) * smoothstep(x / 1 m)), grav (height field on a 5 cm
grid over x 0-4 m, |y| <= 1.5 m: heights uniform in [-a, a] (rank transform of Gaussian-smoothed (sigma 1 cell) white noise),
0.3 m taper; friction cells 0.5 m with mu ~ U(0.4, 1.0) applied to each foot capsule over the cell every physics step),
flat (plane), controls hflat (zero height field) / bflat (box ground).
Per tick: ground under base, sole corners and foot centres by mj_ray (group 5 + plane); fall height = ground + 0.45 m
(ground = lowest of base / foot grounds; flat: 0.45 as before); vertical-face contacts of a foot with a riser box (|n_z| < 0.5,
below the top by 5 mm, normal force > 5 N) every physics step.
  python -I neg40_terrain.py gen OUTDIR --set eval|dev|diag --seeds S0,S1,S2 [--end far|near|both|house|flight|mix] [--ctl]
         [--stand] [--push] [--starts rand|k12]     (diag: seeds salted, k12 starts off the eval grid; never meets eval / dev)
  mix (11-22 10:3x, eval only): threshold, stairs (5 up, 1 m landing, 5 down), ramp (up, 1 m top, down), gravel 4 m and flat 2 m
  in a seeded order, 0.8 m flat between, box ground +-8 m. push: stand on a far-end ramp / side slope / gravel, push J in the Hold.
  python -I neg40_terrain.py geom SPECS.jsonl OUTDIR     (world-frame scene geometry per spec; eval specs -> eval-only)
  python -I neg40_terrain.py run --specs S.jsonl --policy NAME=v4:CKPT | mf=mf:ONNX --out DIR --procs N
  python -I neg40_terrain.py sum OUT.md TRK_DIR MF_DIR"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_v4 as V4  # noqa: E402  (b1 check, P.episode = V4.episode, P.measure; imports sonic_g1 with the body env)

P, S, SR, NB, RG = V4.P, V4.S, V4.SR, V4.NB, V4.RG
import mujoco  # noqa: E402

DT = P.DT
TREAD = 0.28
YB = 4.0                                                                  # box / height-field half width (m), 11-22 09:3x (was 2.0 / 2.5)
RAMP_L = 2.5
GOAL_PAST = 0.6
GROUP = 5
FALL_H = 0.45
SIZES = dict(thr=(0.02, 0.05), st=(0.08, 0.15), rp=(10.0, 15.0), side=(5.0, 10.0), grav=(0.015, 0.03))
# 11-22 10:3x mixed path (doc row 'the four above and flat in random order', eval only): segment lengths (m), half width
MIX_SEGS = ("thr", "st", "rp", "grav", "flat")
MIX_LAND, MIX_GRAV, MIX_FLAT, MIX_GAP, YM = 1.0, 4.0, 2.0, 0.8, 20.0   # 11-22 10:4x: YM 8 -> 20 (mf yaw drift walked off the +-8 m box ground, 4/4 mf mix falls)
_T: dict = dict(build=None, m=None, sim=None, riser={}, feet=None, mu=None, toe=np.zeros(2), rec=None, ray=None)


# ---------------------------------------------------------------- course frame and geometry
def qmul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def course_tf(ter):
  psi_c = -math.radians(float(ter["psi_deg"]))
  c, s = math.cos(psi_c), math.sin(psi_c)
  R = np.array([[c, -s], [s, c]])
  t = -R @ np.array([-float(ter["d0"]), float(ter["y0"])])
  return psi_c, R, t


def to_course(R, t, xy):
  return (np.asarray(xy, float) - t) @ R


def smoothstep(u):
  u = np.clip(u, 0.0, 1.0)
  return u * u * (3 - 2 * u)


def gauss_smooth(a, sig):
  r = int(math.ceil(3 * sig))
  k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
  k /= k.sum()
  a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="reflect"), k, "valid"), 0, a)
  return np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="reflect"), k, "valid"), 1, a)


def build(ter):
  """Course-frame geometry: boxes (centre, half sizes, yaw/pitch), height fields, plane height (None = off), risers, goal."""
  kind, sz = ter["kind"], float(ter.get("size", 0.0))
  boxes, hfs, riser, plane_z, x_end = [], [], {}, 0.0, 0.0
  mu = None

  def box(name, c, half, yaw=0.0, pitch=0.0, top=None):
    boxes.append(dict(name=name, c=list(map(float, c)), half=list(map(float, half)), yaw=float(yaw), pitch=float(pitch)))
    if top is not None:
      riser[name] = float(top)

  if kind == "thr":
    al = math.radians(float(ter["alpha_deg"]))            # the cell fixes the direction; mirror = start only
    box("t40_thr", (0.05 * math.cos(al), 0.05 * math.sin(al), sz / 2), (0.05, YB, sz / 2), yaw=al, top=sz)
    x_end = (0.10 + 0.5 * abs(math.sin(al))) / math.cos(al)
  elif kind == "stup":
    tr = float(ter.get("tread", TREAD))                                 # 11-22 09:5x: house-size cells (tread 26 cm); default 28 cm
    ns = int(ter.get("steps", 5))                                       # 11-22 10:1x: one flight of 11 risers (coordinator 10:03); default 5
    for i in range(1, ns):
      box(f"t40_st{i}", ((i - 0.5) * tr, 0, i * sz / 2), (tr / 2, YB, i * sz / 2), top=i * sz)
    box(f"t40_st{ns}", ((ns - 1) * tr + 4.0, 0, ns * sz / 2), (4.0, YB, ns * sz / 2), top=ns * sz)
    x_end = (ns - 1) * tr
  elif kind == "stdn":
    tr = float(ter.get("tread", TREAD))
    ns = int(ter.get("steps", 5))
    plane_z = None                                                      # 11-22 09:3x fix: a moved world plane keeps xpos 0
    box("t40_st0", (-3.0, 0, -ns * sz / 2), (3.0, YB, ns * sz / 2), top=0.0)
    for i in range(1, ns):
      box(f"t40_st{i}", ((i - 0.5) * tr, 0, (-i * sz - ns * sz) / 2), (tr / 2, YB, (ns * sz - i * sz) / 2), top=-i * sz)
    box("t40_low", ((ns - 1) * tr + 4.0, 0, -ns * sz - 0.1), (4.0, YB, 0.1))
    x_end = (ns - 1) * tr
  elif kind in ("rpup", "rpdn"):
    th = math.radians(sz)
    H = RAMP_L * math.tan(th)
    half = (RAMP_L / math.cos(th) / 2, YB, 0.1)
    if kind == "rpup":
      box("t40_slope", (RAMP_L / 2 + 0.1 * math.sin(th), 0, H / 2 - 0.1 * math.cos(th)), half, pitch=-th)
      box("t40_top", (RAMP_L + 4.0, 0, H / 2), (4.0, YB, H / 2))
    else:
      plane_z = None                                                    # 11-22 09:3x fix (see stdn)
      box("t40_start", (-3.0, 0, -(H + 0.1) / 2), (3.0, YB, (H + 0.1) / 2))
      box("t40_slope", (RAMP_L / 2 - 0.1 * math.sin(th), 0, -H / 2 - 0.1 * math.cos(th)), half, pitch=th)
      box("t40_low", (RAMP_L + 4.0, 0, -H - 0.1), (4.0, YB, 0.1))
    x_end = RAMP_L
  elif kind in ("side", "grav", "hflat"):
    plane_z = None
    cell = 0.05
    x0, x1, yh = -4.0, 10.0, (12.0 if kind == "side" else YB)              # 11-22 09:3x: R14 drifts 6 m downhill at v0.3, so +-12 m
    nc, nr = int(round((x1 - x0) / cell)) + 1, int(round(2 * yh / cell)) + 1
    X, Y = np.meshgrid(np.linspace(x0, x1, nc), np.linspace(-yh, yh, nr))
    if kind == "side":
      sg = float(ter["side"])
      zoff = yh * math.tan(math.radians(sz))
      Z = sg * Y * math.tan(math.radians(sz)) * smoothstep(X / 1.0)
      x_end = 4.0
    elif kind == "grav":
      rng = np.random.default_rng(int(ter["gravel_seed"]))
      Zs = gauss_smooth(rng.uniform(-1, 1, size=(nr, nc)), 1.0)
      u = (np.argsort(np.argsort(Zs.ravel())) + 0.5) / Zs.size           # heights uniform in [-a, a], Gaussian-smoothed correlation
      w = smoothstep(X / 0.3) * smoothstep((4.0 - X) / 0.3) * smoothstep((1.8 - np.abs(Y)) / 0.3)
      Z = sz * (2 * u.reshape(nr, nc) - 1) * w
      mu_cells = rng.uniform(0.4, 1.0, size=(6, 8))                      # rows y in [-1.5, 1.5], cols x in [0, 4]
      if ter.get("mirror"):
        Z, mu_cells = Z[::-1], mu_cells[::-1]
      mu = dict(x0=0.0, y0=-1.5, cell=0.5, grid=mu_cells.tolist())
      zoff = sz
      x_end = 4.0
    else:
      Z = np.zeros((nr, nc))
      zoff = 0.0005
      x_end = 4.0
    data = (Z + zoff) / (2 * zoff)
    data[0, 0], data[-1, -1] = 0.0, 1.0                                  # pins the [0, 1] range at the two far corners
    hfs.append(dict(name="t40_hf", nrow=nr, ncol=nc, data=data, size=[(x1 - x0) / 2, yh, 2 * zoff, 0.1],
                    c=[(x0 + x1) / 2, 0.0, -zoff]))
  elif kind == "mix":
    # segments in ter['order'] from x = 0, MIX_GAP flat between; every segment ends at ground 0; box ground (plane off) with a
    # hole under the gravel height field; stairs = 5 risers up, 1 m landing, 5 risers down; ramp = up 2.5 m, 1 m top, down 2.5 m
    plane_z = None
    szs, tr, W = ter["sizes"], float(ter.get("tread", TREAD)), YM
    segs, holes, x = [], [], 0.0
    for q, nm in enumerate(ter["order"]):
      a0 = x
      if nm == "thr":
        h = float(szs["thr"])
        box(f"t40_m{q}_thr", (a0 + 0.05, 0, h / 2), (0.05, W, h / 2), top=h)
        x = a0 + 0.10
      elif nm == "st":
        h, ns = float(szs["st"]), 5
        for i in range(1, ns):
          box(f"t40_m{q}_u{i}", (a0 + (i - 0.5) * tr, 0, i * h / 2), (tr / 2, W, i * h / 2), top=i * h)
        xl = a0 + (ns - 1) * tr
        box(f"t40_m{q}_land", (xl + MIX_LAND / 2, 0, ns * h / 2), (MIX_LAND / 2, W, ns * h / 2), top=ns * h)
        xd = xl + MIX_LAND
        for i in range(1, ns):
          top = (ns - i) * h
          box(f"t40_m{q}_d{i}", (xd + (i - 0.5) * tr, 0, top / 2), (tr / 2, W, top / 2), top=top)
        x = xd + (ns - 1) * tr
      elif nm == "rp":
        th = math.radians(float(szs["rp"]))
        H = RAMP_L * math.tan(th)
        half = (RAMP_L / math.cos(th) / 2, W, 0.1)
        box(f"t40_m{q}_up", (a0 + RAMP_L / 2 + 0.1 * math.sin(th), 0, H / 2 - 0.1 * math.cos(th)), half, pitch=-th)
        box(f"t40_m{q}_top", (a0 + RAMP_L + MIX_LAND / 2, 0, H / 2), (MIX_LAND / 2, W, H / 2))
        xb = a0 + RAMP_L + MIX_LAND
        box(f"t40_m{q}_dn", (xb + RAMP_L / 2 - 0.1 * math.sin(th), 0, H / 2 - 0.1 * math.cos(th)), half, pitch=th)
        x = xb + RAMP_L
      elif nm == "grav":
        a = float(szs["grav"])
        cell = 0.05
        nc, nr = int(round(MIX_GRAV / cell)) + 1, int(round(2 * W / cell)) + 1
        X, Y = np.meshgrid(np.linspace(0.0, MIX_GRAV, nc), np.linspace(-W, W, nr))
        rng = np.random.default_rng(int(ter["gravel_seed"]))
        Zs = gauss_smooth(rng.uniform(-1, 1, size=(nr, nc)), 1.0)
        u = (np.argsort(np.argsort(Zs.ravel())) + 0.5) / Zs.size
        w = smoothstep(X / 0.3) * smoothstep((MIX_GRAV - X) / 0.3) * smoothstep((1.8 - np.abs(Y)) / 0.3)
        Z = a * (2 * u.reshape(nr, nc) - 1) * w
        mu_cells = rng.uniform(0.4, 1.0, size=(6, 8))
        if ter.get("mirror"):
          Z, mu_cells = Z[::-1], mu_cells[::-1]
        mu = dict(x0=a0, y0=-1.5, cell=0.5, grid=mu_cells.tolist())
        data = (Z + a) / (2 * a)
        data[0, 0], data[-1, -1] = 0.0, 1.0
        hfs.append(dict(name=f"t40_m{q}_hf", nrow=nr, ncol=nc, data=data, size=[MIX_GRAV / 2, W, 2 * a, 0.1],
                        c=[a0 + MIX_GRAV / 2, 0.0, -a]))
        holes.append((a0, a0 + MIX_GRAV))
        x = a0 + MIX_GRAV
      elif nm == "flat":
        x = a0 + MIX_FLAT
      else:
        raise ValueError(nm)
      segs.append(dict(name=nm, x0=round(a0, 4), x1=round(x, 4), h=float(szs[nm]) if nm in ("st", "thr") else 0.0))
      x += MIX_GAP
    x_end = x - MIX_GAP
    lo = -6.0
    for h0, h1 in sorted(holes) + [(x_end + 6.0, None)]:
      box(f"t40_m_base{len(boxes)}", ((lo + h0) / 2, 0, -0.1), ((h0 - lo) / 2, W, 0.1))
      lo = h1
  elif kind == "bflat":
    plane_z = None
    box("t40_ground", (3.0, 0, -0.1), (7.0, YB, 0.1))
    x_end = 4.0
  elif kind == "flat":
    x_end = 4.0
  else:
    raise ValueError(kind)
  out = dict(kind=kind, boxes=boxes, hfs=hfs, riser=riser, plane_z=plane_z, x_goal=x_end + GOAL_PAST, x_end=x_end, mu=mu,
             tf=course_tf(ter))
  if kind == "mix":
    out["segs"] = segs
  return out


def add_terrain(sp, b):
  psi_c, R, t = b["tf"]
  for g in b["boxes"]:
    xy = R @ np.array(g["c"][:2]) + t
    yaw = psi_c + g["yaw"]
    qz = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    qy = [math.cos(g["pitch"] / 2), 0.0, math.sin(g["pitch"] / 2), 0.0]
    sp.worldbody.add_geom(name=g["name"], type=mujoco.mjtGeom.mjGEOM_BOX, size=g["half"], pos=[xy[0], xy[1], g["c"][2]],
                          quat=qmul(qz, qy), group=GROUP)
  for h in b["hfs"]:
    sp.add_hfield(name=h["name"], size=h["size"], nrow=h["nrow"], ncol=h["ncol"], userdata=list(map(float, h["data"].ravel())))
    xy = R @ np.array(h["c"][:2]) + t
    sp.worldbody.add_geom(name=h["name"], type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=h["name"], pos=[xy[0], xy[1], h["c"][2]],
                          quat=[math.cos(psi_c / 2), 0.0, 0.0, math.sin(psi_c / 2)], group=GROUP)


# ---------------------------------------------------------------- patches (process-local; spawn workers import this module)
_ff = mujoco.MjSpec.from_file
_init_prev = S.SimR1.__init__                       # already wrapped by neg30_body (b1 check)


def _init_terrain(self, *a, **k):
  b = _T["build"]
  if b is None:
    return _init_prev(self, *a, **k)
  xml = os.path.realpath(S.R1_XML)

  def from_file(path, *aa, **kk):
    sp = _ff(path, *aa, **kk)
    if os.path.realpath(str(path)) == xml:
      add_terrain(sp, b)
    return sp
  mujoco.MjSpec.from_file = staticmethod(from_file)
  try:
    _init_prev(self, *a, **k)
  finally:
    mujoco.MjSpec.from_file = _ff


S.SimR1.__init__ = _init_terrain
_orig_apply = SR.apply_body


def _apply(sim, body, floor):
  _orig_apply(sim, body, floor)
  b = _T["build"]
  if b is None:
    return
  m = sim.m
  fid = m.geom("floor").id
  ids = {m.geom(g["name"]).id for g in b["boxes"]} | {m.geom(h["name"]).id for h in b["hfs"]}
  if b["plane_z"] is None:
    m.geom_contype[fid], m.geom_conaffinity[fid] = 0, 0
  else:
    m.geom_pos[fid, 2] = float(b["plane_z"])
    ids.add(fid)
  sim.floor_ids = ids
  feet = np.array(sorted(sim.foot_geoms))
  _T.update(m=m, sim=sim, riser={m.geom(n).id: top for n, top in b["riser"].items()}, feet=feet,
            side=np.array([sim.foot_side[g] for g in feet]), toe=np.zeros(2), rec=[], xs=0, xb=set(), xbt=[],
            mu=(dict(b["mu"], grid=np.array(b["mu"]["grid"])) if b["mu"] else None),
            gid=np.zeros(1, np.int32), gg=np.array([1 if g == GROUP else 0 for g in range(6)], np.uint8))


SR.apply_body = _apply
_orig_step = mujoco.mj_step


def _set_mu(m, d):
  T = _T
  psi_c, R, t = T["build"]["tf"]
  mu = T["mu"]
  xy = to_course(R, t, d.geom_xpos[T["feet"], :2])
  ix = np.floor((xy[:, 0] - mu["x0"]) / mu["cell"]).astype(int)
  iy = np.floor((xy[:, 1] - mu["y0"]) / mu["cell"]).astype(int)
  G = mu["grid"]
  ok = (ix >= 0) & (ix < G.shape[1]) & (iy >= 0) & (iy < G.shape[0])
  val = np.ones(len(ix))
  val[ok] = G[iy[ok], ix[ok]]
  m.geom_friction[T["feet"], 0] = val


def _toe(m, d):
  T = _T
  rz = T["riser"]
  feet = T["sim"].foot_geoms
  f6 = np.zeros(6)
  for i in range(d.ncon):
    c = d.contact[i]
    g1, g2 = int(c.geom1), int(c.geom2)
    if g1 in rz and g2 in feet:
      gt, gf = g1, g2
    elif g2 in rz and g1 in feet:
      gt, gf = g2, g1
    else:
      continue
    if abs(float(c.frame[2])) < 0.5 and float(c.pos[2]) < rz[gt] - 0.005:
      mujoco.mj_contactForce(m, d, i, f6)
      s = T["sim"].foot_side[gf]
      if f6[0] > 5.0 and f6[0] > T["toe"][s]:
        T["toe"][s] = float(f6[0])


def _xsub(m, d):
  """11-22 10:4x (coordinator 10:40; user 11-18 scene rule: nothing but the feet touches the ground): after every physics substep,
  any non-foot geom touching a terrain / floor geom (sim.floor_ids) -> T['xs'] += 1 (per control tick), its body -> T['xb']."""
  T = _T
  sim = T["sim"]
  fl, feet = sim.floor_ids, sim.foot_geoms
  hit = False
  for i in range(d.ncon):
    c = d.contact[i]
    g1, g2 = int(c.geom1), int(c.geom2)
    f1, f2 = g1 in fl, g2 in fl
    if f1 == f2:
      continue
    g = g2 if f1 else g1
    if g not in feet:
      hit = True
      T["xb"].add(int(m.geom_bodyid[g]))
  if hit:
    T["xs"] += 1


def _step(m, d, *a):
  T = _T
  if T["m"] is m:
    if T["mu"] is not None:
      _set_mu(m, d)
    _orig_step(m, d, *a)
    if T["riser"]:
      _toe(m, d)
    _xsub(m, d)
  else:
    _orig_step(m, d, *a)


mujoco.mj_step = _step
_orig_contacts = SR.contacts


def ground(m, d, pts):
  T = _T
  out = np.full(len(pts), np.nan)
  pz = T["build"]["plane_z"]
  for i, p in enumerate(pts):
    dist = mujoco.mj_ray(m, d, np.array([p[0], p[1], 3.0]), np.array([0.0, 0.0, -1.0]), T["gg"], 1, -1, T["gid"])
    if dist >= 0:
      out[i] = 3.0 - dist
    elif pz is not None:
      out[i] = pz
  if pz is not None:
    out = np.maximum(out, pz)
  return out


def _contacts(sim, f6):
  F, other = _orig_contacts(sim, f6)
  T = _T
  if T["build"] is not None and T["sim"] is sim and T["rec"] is not None:
    m, d = sim.m, sim.d
    co = SR.corners(sim)
    pts = np.concatenate([d.qpos[None, 0:3], co.reshape(8, 3), co.mean(1)], 0)
    g = ground(m, d, pts)
    T["rec"].append(np.concatenate([g, T["toe"], [float(T["xs"])]]))
    T["toe"] = np.zeros(2)
    if T["xb"]:
      T["xbt"].append((len(T["rec"]) - 1, sorted(sim.m.body(b_).name for b_ in T["xb"])))
    T["xs"], T["xb"] = 0, set()
    ref = np.nanmin(g[[0, 9, 10]]) if np.isfinite(g[[0, 9, 10]]).any() else 0.0
    SR.FALL_Z = float(ref) + FALL_H
  return F, other


SR.contacts = _contacts
_inner_episode = P.episode                          # = V4.episode


def episode(job):
  ter = (job["spec"].get("env") or {}).get("terrain")
  _T.update(build=build(ter) if ter else None, m=None, sim=None, riser={}, mu=None, rec=None, toe=np.zeros(2))
  try:
    arr, info = _inner_episode(job)
    if ter:
      rec = np.array(_T["rec"], np.float32)
      arr["tg"] = rec[:, :11]
      arr["toe"] = rec[:, 11:13]
      arr["xsub"] = rec[:, 13]                     # substeps with a non-foot terrain contact, per control tick
      info["x_body_ticks"] = list(_T["xbt"])
      info["terrain_ngeom"] = int(_T["m"].ngeom)
    return arr, info
  finally:
    SR.FALL_Z = FALL_H
    if _T["m"] is not None and _T["mu"] is not None:
      _T["m"].geom_friction[_T["feet"], 0] = 1.0
    _T.update(build=None, m=None, sim=None, riser={}, mu=None, rec=None)


P.episode = episode


# ---------------------------------------------------------------- measures (one row per episode)
def swings(fz, k_lo, k_hi):
  out = []
  for f in range(2):
    off = fz[:, f] < 5.0
    k = k_lo
    while k < k_hi:
      if off[k]:
        j = k
        while j < k_hi and off[j]:
          j += 1
        if j - k >= 2 and j < k_hi and k > 0:
          out.append((f, k, j))
        k = j
      else:
        k += 1
  return out


def tilt_of(q):
  w, x, y, z = q
  return math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y))))


def measure(spec, arr, info):
  ter = (spec.get("env") or {}).get("terrain")
  if not ter:
    return V4._measure_body(spec, arr, info)
  b = build(ter)
  psi_c, R, t = b["tf"]
  kind, sz, v = b["kind"], float(ter.get("size", 0.0)), float(ter["v"])
  n = len(arr["pos"])
  pos, sole, fz = arr["pos"].astype(float), arr["sole"].astype(float), arr["force"][:, :, 2].astype(float)
  tg, toe = arr["tg"].astype(float), arr["toe"].astype(float)
  xc = to_course(R, t, pos[:, :2])
  settle = float(spec["mission"]["settle_s"])
  T_go = sum(float(s["duration_s"]) for s in spec["mission"]["steps"] if s["type"] == "Go")
  k0, k1 = int(round(settle / DT)), min(n, int(round((settle + T_go) / DT)))
  fell_k = int(info["fell_k"])
  fell = fell_k >= 0
  k_end = fell_k if fell else n
  reach = [k for k in range(k0, min(k1, k_end)) if xc[k, 0] >= b["x_goal"]]
  k_r = reach[0] if reach else None
  nominal = (float(ter["d0"]) + b["x_goal"]) / v
  k_stop = min(k_r if k_r is not None else k1, k_end)
  stall = any(xc[k + 150, 0] - xc[k, 0] < 0.15 for k in range(k0, k_stop - 150))
  hops = P.hops(arr["force"].astype(float), sole, k0)
  # 11-22 10:4x non-foot contact X (coordinator 10:40): window settle .. fall - 0.1 s; substep flag when recorded, else the runner's
  # end-of-tick 'other' (older npz). 11-22 11:1x order F -> X -> H -> S -> L -> C (coordinator 11:09 item 2: a Go-phase 3 s stall
  # is S even if the goal is reached later; L = not reached inside the Go window, which already holds the 1.4x time budget)
  k_x = max(k0, fell_k - int(round(0.1 / DT))) if fell else n
  x_src = "substep" if "xsub" in arr else "tick_end"
  xw = (arr["xsub"][k0:k_x] > 0) if "xsub" in arr else (arr["other"][k0:k_x] > 0)
  xcon = bool(xw.any())
  disp = "F" if fell else ("X" if xcon else ("H" if hops > 0 else ("S" if stall else ("C" if k_r is not None else "L"))))
  row = dict(spec_sha256=spec["spec_sha256"], family="terrain", cell=spec["cell"], i=spec["i"], policy=info["policy"],
             kind=kind, size=sz, v=v, seed_i=int(spec["i"]) // 4, j=int(spec["i"]) % 4, mirror=bool(ter.get("mirror")),
             disp=disp, completed=disp == "C", reached=k_r is not None, fell=fell, hops=int(hops), stall=bool(stall),
             t_reach=round((k_r - k0 + 1) * DT, 2) if k_r is not None else None,
             t_ratio=round((k_r - k0 + 1) * DT / nominal, 3) if k_r is not None else None, wall_s=info["wall_s"],
             x_contact=xcon, x_src=x_src, x_ticks=int(xw.sum()),
             x_sub=int(arr["xsub"][k0:k_x].sum()) if "xsub" in arr else None,
             x_t_first=round((int(np.argmax(xw)) + 1) * DT, 2) if xcon else None,
             x_bodies=sorted({b_ for k_, bs in (info.get("x_body_ticks") or []) if k0 <= k_ < k_x for b_ in bs}) if xcon and "xsub" in arr else None)
  if fell:
    tl = tilt_of(arr["quat"][fell_k].astype(float))
    row.update(t_fall=round((fell_k - k0 + 1) * DT, 2), x_fall=round(float(xc[fell_k, 0]), 3), fall_tilt=round(tl, 3),
               fall_by="tilt" if tl > SR.FALL_TILT else "height",
               fall_zb=round(float(pos[fell_k, 2]), 3), fall_gref=round(float(np.nanmin(tg[fell_k, [0, 9, 10]])), 3),
               fall_footz=[round(float(sole[fell_k, f, :, 2].min()), 3) for f in range(2)])
  # ground-relative sole gaps (2, 4) per tick; foot centre course x
  gap = sole[..., 2] - tg[:, 1:9].reshape(n, 2, 4)
  cxy = to_course(R, t, sole[..., :2].reshape(-1, 2)).reshape(n, 2, 4, 2)
  fcx = cxy[..., 0].mean(2)
  sw = swings(fz, k0, k_end)
  lift_pre, lift_obs, clear_obs, tread, miss = [], [], [], [], 0
  box_kind = kind in ("thr", "stup", "stdn", "mix")
  bsegs = [g for g in b.get("segs", []) if g["name"] in ("thr", "st")]       # mix: box segments (missteps, toe)
  TR = float(ter.get("tread", TREAD))
  NS = int(ter.get("steps", 5))
  for f, a, e in sw:
    z0 = sole[a - 1, f, :, 2].min()
    lift = float(sole[a:e, f, :, 2].min(1).max() - z0)
    clr = float(gap[a:e, f].min(1).max())
    xa, xe = fcx[a - 1, f], fcx[min(e, n - 1), f]
    if xe < -0.3:
      lift_pre.append(lift)
    if kind == "mix":
      hit = [g for g in bsegs if min(xa, xe) < g["x1"] + 0.15 and max(xa, xe) > g["x0"] - 0.15]
      obstacle = bool(hit)
    else:
      obstacle = (min(xa, xe) < b["x_end"] + 0.15 and max(xa, xe) > -0.15) if box_kind else (0.0 <= xe <= b["x_end"])
    if obstacle:
      lift_obs.append(lift)
      clear_obs.append(clr)
    kt = e + 5
    if kt < k_end and box_kind and obstacle:
      thr_gap = min(0.03, 0.5 * (hit[0]["h"] if kind == "mix" else sz))
      if int((gap[kt, f] > thr_gap).sum()) >= 2:
        miss += 1
    if kt < k_end and kind in ("stup", "stdn") and -0.3 <= fcx[kt, f] <= (NS - 1) * TR + 0.3:
      zs = float(tg[kt, 9 + f])
      i_t = int(round(zs / sz)) if kind == "stup" else int(round(-zs / sz))
      cx = cxy[kt, f, :, 0]
      if kind == "stup" and 1 <= i_t <= NS:
        tread.append(float(cx.min() - (i_t - 1) * TR))
      elif kind == "stdn" and 0 <= i_t <= NS - 1:
        tread.append(float(i_t * TR - cx.max()))
  # stance slip, ground-relative (neg28 definition with the ground under the lowest corner instead of z = 0)
  lo = gap.argmin(2)
  vel = np.linalg.norm(np.gradient(sole[..., :2], DT, axis=0), axis=3)
  spl = np.take_along_axis(vel, lo[..., None], 2)[..., 0]
  mk = np.take_along_axis(gap, lo[..., None], 2)[..., 0] < 0.002
  sl = spl[k0:k_end][mk[k0:k_end]]
  toe_k = (toe[k0:k_end] > 5.0).any(1)
  med = (lambda x: round(float(np.median(x)), 4) if len(x) else None)
  row.update(toe_ep=bool(toe_k.any()) if box_kind else None,
             toe_events=int(np.sum(toe_k[1:] & ~toe_k[:-1]) + int(toe_k[:1].sum())) if box_kind else None,
             toe_maxN=round(float(toe[k0:k_end].max()), 1) if box_kind and k_end > k0 else None,
             lift_pre_med=med(lift_pre), lift_obs_med=med(lift_obs),
             lift_obs_max=round(max(lift_obs), 4) if lift_obs else None, clear_obs_min=round(min(clear_obs), 4) if clear_obs else None,
             n_swing=len(sw), n_obs=len(lift_obs), misstep=miss if box_kind else None,
             tread_min=round(min(tread), 4) if tread else None, tread_neg=int(sum(x < 0 for x in tread)) if tread else None,
             slip_p95=round(float(np.percentile(sl, 95)), 4) if sl.size else None,
             y_drift=round(float(xc[(k_r if k_r is not None else k_end - 1), 1] - float(ter["y0"])), 3),
             other_ticks=int(arr["other"][k0:k_end].sum()), x_final=round(float(xc[k_end - 1, 0]), 3))
  if kind == "mix":
    # per segment: entered = base course x reached x0, passed = reached x1 (before the fall / end); end segment = where it stopped
    xe_ = xc[k0:k_stop, 0]
    xmax = float(xe_.max()) if len(xe_) else float(xc[k0, 0])
    xlast = float(xc[k_stop - 1, 0])
    segs = b["segs"]
    at = next((g["name"] for g in segs if g["x0"] - 0.3 <= xlast <= g["x1"] + 0.3), "gap")
    row.update(seg_order=[g["name"] for g in segs], seg_entered=[bool(xmax >= g["x0"]) for g in segs],
               seg_passed=[bool(xmax >= g["x1"]) for g in segs], seg_end=None if k_r is not None else at,
               y_abs_max=round(float(np.abs(xc[k0:k_end, 1]).max()), 3) if k_end > k0 else None, path_len=round(b["x_goal"], 3))
  if ter.get("push"):
    kp = int(round(float(ter["push"]["t_s"]) / DT))
    row.update(push_J=float(ter["push"]["J"]), push_dir_deg=float(ter["push"]["dir_deg"]), fell_after_push=bool(fell and fell_k >= kp),
               hops_after_push=int(P.hops(arr["force"].astype(float), sole, min(kp, n - 1))))
  if ter.get("stand"):
    # standing on the slope (table row 'ramp: ... standing on it'): walk on for T_on, then Hold; the window starts 1 s into the Hold
    k_h0 = k1
    w0 = k_h0 + 50
    x_h0 = float(xc[min(k_h0, n - 1), 0])
    on_slope = (0.0 <= x_h0 <= RAMP_L) if kind in ("rpup", "rpdn") else ((0.3 <= x_h0 <= 3.7) if kind == "grav" else (1.0 <= x_h0 <= 10.0))
    ok_w = (not fell or fell_k > w0) and w0 < k_end
    st_sw = [(f, a, e) for f, a, e in sw if a >= w0]
    slide = None
    if ok_w:
      cen = sole[..., :2].mean(2)
      d_f = [float(np.linalg.norm(cen[k_end - 1, f] - cen[w0, f])) for f in range(2) if not any(g == f for g, _, _ in st_sw)]
      slide = round(max(d_f), 4) if d_f else None
    sdisp = "F" if fell else ("X" if xcon else ("H" if hops > 0 else ("C" if on_slope else "N")))   # 11-22 11:1x X added (same window)
    row.update(stand=True, disp=sdisp, completed=sdisp == "C", x_h0=round(x_h0, 3), on_slope=bool(on_slope),
               fell_in_hold=bool(fell and fell_k >= k_h0),
               hold_drift=round(float(np.linalg.norm(xc[k_end - 1] - xc[w0])), 4) if ok_w else None,
               hold_steps=len(st_sw) if ok_w else None, hold_slide=slide)
  if NB.BODY != "r1wb":
    row.update(NB.row_tag(S))
  return row


P.measure = measure


# ---------------------------------------------------------------- specs
def cells(end):
  out = []
  for e in (("near", "far") if end == "both" else (end,)):
    ix = 0 if e == "near" else 1
    out += [("thr", dict(size=SIZES["thr"][ix], alpha_deg=a), f"thr{a:+d}".replace("+0", "0"), e) for a in (0, 30, -30)]
    out += [(k, dict(size=SIZES["st"][ix]), k, e) for k in ("stup", "stdn")]
    out += [(k, dict(size=SIZES["rp"][ix]), k, e) for k in ("rpup", "rpdn")]
    out += [("side", dict(size=SIZES["side"][ix], side=s), f"side{'+' if s > 0 else '-'}", e) for s in (1, -1)]
    out += [("grav", dict(size=SIZES["grav"][ix]), "grav", e)]
  return out


def house_cells():
  """11-22 09:48 coordinator (supplier house sheet 09:42): stairs 18 cm / tread 26 cm, ramps 1:8 and 1:12; stairs first."""
  r8, r12 = round(math.degrees(math.atan(1 / 8)), 3), round(math.degrees(math.atan(1 / 12)), 3)
  return ([(k, dict(size=0.18, tread=0.26), k, "h18") for k in ("stup", "stdn")] +
          [(k, dict(size=r8), k, "h71") for k in ("rpup", "rpdn")] + [(k, dict(size=r12), k, "h48") for k in ("rpup", "rpdn")])


X_ON = dict(rpup=0.5 * RAMP_L, rpdn=0.5 * RAMP_L, side=2.5)                 # stand cells: nominal walk-on point (course x, m)
STRIDE = {0.3: 0.198, 0.5: 0.330}     # 11-22 10:0x: reference v4.3s foot advance per swing (= one stride, two steps), measured
BASE_D0 = 1.0                          # k12 starts: d0 = BASE_D0 + STRIDE[v] * k / 12, k = 3 j + seed index (coordinator 09:33)
HOLD_STAND = 8.0


def flight_cells():
  """11-22 10:03 coordinator: mission condition, one flight of 11 risers (landing 2 m / 18 cm), 8 cm / 28 cm and 18 cm / 26 cm."""
  return [(k, dict(size=0.08, steps=11), f"{k}11", "n08") for k in ("stup", "stdn")] + \
         [(k, dict(size=0.18, tread=0.26, steps=11), f"{k}11", "h18") for k in ("stup", "stdn")]


def mix_cells():
  """11-22 10:3x: mixed path at the near and the provisional far sizes (far = H0 provisional; regenerate when H0 fixes the ends)."""
  return [("mix", dict(sizes=dict(thr=SIZES["thr"][ix], st=SIZES["st"][ix], rp=SIZES["rp"][ix], grav=SIZES["grav"][ix])), "mix", e)
          for ix, e in ((0, "near"), (1, "far"))]


PUSH_X_ON = dict(X_ON, grav=2.0)       # push-on-terrain cells: stand on the far-end terrain, push in the Hold
PUSH_J = (12.0, 18.0, 25.0)            # the flat push panel's three levels (neg30_v4 v4push)
PUSH_AT = 3.0                          # s into the Hold; direction (3 j + seed index) x 30 deg relative to the robot heading


def gen(outdir, setname, seeds, end, ctl, stand=False, starts="rand", push=False, k12shift=0.0):
  src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  assert k12shift == 0.0 or (setname == "eval" and starts == "k12" and 0.0 < k12shift < 1.0), "k12shift: eval k12 only"
  loop = dict(planner="calm4_causal", path="realtime", bridge="none", ctx_robot=False, bridge_lat_s=0.0)
  base_cells = house_cells() if end == "house" else flight_cells() if end == "flight" else mix_cells() if end == "mix" else cells(end)
  if push:
    cl = [(k, dict(p, stand=True, push_J=J), f"{t}_push{int(J)}", e) for k, p, t, e in cells("far") if k in PUSH_X_ON for J in PUSH_J]
  elif stand:
    cl = [(k, dict(p, stand=True), f"{t}_stand", e) for k, p, t, e in base_cells if k in X_ON]
  else:
    cl = base_cells + ([("flat", dict(size=0.0), "flat", "ctl")] if end != "house" else [])
  if ctl and not stand:
    cl += [("hflat", dict(size=0.0), "hflat", "ctl"), ("bflat", dict(size=0.0), "bflat", "ctl")]
  files = {d: [] for d in ("d15", "d00")}
  snap = (lambda x: math.ceil(round(x * 10, 6)) / 10) if setname == "diag" else (lambda x: x)   # 11-22 10:4x (coordinator 10:38): diag
  for kind, par, tag, e in cl:                                       # command times on the 0.1 s grid (the planner replans at 10 Hz)
    for v in (0.3, 0.5):
      for si, seed in enumerate(seeds):
        sk = SR.child_seed(seed, "t40_diag") if setname == "diag" else seed   # 11-22 10:1x: diag namespace never meets eval / dev
        rng = np.random.default_rng(SR.child_seed(sk, "t40_start"))
        st = [(float(rng.uniform(0.8, 1.6)), float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-3.0, 3.0))) for _ in range(2)]
        for j in range(4):
          d0, y0, psi = st[j // 2]
          if starts == "k12":
            off = 0.0
            if setname in ("diag", "dev"):   # 11-22 10:3x diag, 10:4x dev (coordinator 10:26 item 3): continuous salted offset inside
              off = float(np.random.default_rng(SR.child_seed(sk, f"t40_k12off|{j}")).uniform(0.0, 1.0))   # each 1/12 slot, off the eval grid
            else:
              off = float(k12shift)          # eval: 0 = official k/12 grid; 0.5 = the grid shifted by 1/24 (boundary rule, coordinator 10:26)
            d0 = BASE_D0 + STRIDE[v] * (3 * j + si + off) / 12
          mir = bool(j % 2)
          ter = dict(kind=kind, **par, v=v, d0=round(d0, 4), y0=round(-y0 if mir else y0, 4), psi_deg=round(-psi if mir else psi, 3),
                     mirror=mir, end=e, **({"starts": "k12"} if starts == "k12" else {}),
                     **({"k12shift": float(k12shift)} if k12shift else {}))
          if kind in ("grav", "mix"):
            ter["gravel_seed"] = SR.child_seed(sk, f"t40_gravel|{j // 2}")
          if kind == "mix":       # a mirror pair shares the course (order + gravel), mirrored
            ter["order"] = [str(x) for x in np.random.default_rng(SR.child_seed(sk, f"t40_mix|{j // 2}")).permutation(MIX_SEGS)]
          hit = None
          if push:
            J = ter.pop("push_J")
            T_go, T_hold = snap(round((d0 + PUSH_X_ON[kind]) / v, 2)), HOLD_STAND
            t_p = round(2.0 + T_go + PUSH_AT, 2)
            dd = 30.0 * (3 * j + si)
            ter["push"] = dict(J=J, dir_deg=dd, t_s=t_p)
            hit = [dict(t_s=t_p, J_Ns=J, dir_rad=round(math.radians(dd), 6), z_off_m=0.1)]
          b = build(ter)
          if stand and not push:
            T_go, T_hold = snap(round((d0 + X_ON[kind]) / v, 2)), HOLD_STAND
          elif not push:
            T_go, T_hold = snap(round(1.4 * (d0 + b["x_goal"] + 0.3) / v, 2)), 3.0
          base = f"{tag}_{e}_v{int(round(v * 10)):02d}"
          for dl, ms in (("d15", 15.0), ("d00", 0.0)):
            sp = dict(schema="r1adv-scenario-v1", generator=dict(name="neg40-terrain", version="v1", source_sha256=src), seed=int(seed),
                      family="terrain", cell=f"{base}_{dl}", evalset=dict(eval="E40", dev="D40", diag="X40")[setname], i=si * 4 + j,
                      mission=dict(settle_s=2.0, steps=[dict(type="Go", vx=v, duration_s=T_go, label=f"fwd{v}"),
                                                        dict(type="Hold", duration_s=T_hold)]),
                      env=dict(loop=loop, body=dict(action_delay_ms=ms), onset_s=1.0, sense={},
                               noise_seed=SR.child_seed(sk, f"t40_noise|{j}"), terrain=ter, **({"hit": hit} if hit else {})),
                      solvability_evidence=None)
            sp["spec_sha256"] = hashlib.sha256(json.dumps(sp, sort_keys=True).encode()).hexdigest()
            files[dl].append(sp)
  out = Path(outdir)
  out.mkdir(parents=True, exist_ok=True)
  for dl, L in files.items():
    (out / f"terrain_{'push_' if push else 'stand_' if stand else ''}{end}{'_k12' if starts == 'k12' else ''}{f'_s{int(round(k12shift * 100)):02d}' if k12shift else ''}_{dl}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in L))
  print({dl: len(L) for dl, L in files.items()}, "cells", len(files["d15"]) // 12)


def geom_dump(specs, outdir):
  """11-22 10:3x (coordinator 10:03 item 5): per spec, the scene geometry in the WORLD frame (the frame of pos / sole in the npz).
  Rebuilt from the spec (build() is deterministic; old vs new build compared equal over every eval / dev spec at each code bump).
  The course transform encodes the start, so a dump of eval specs is eval-only."""
  out = Path(outdir)
  out.mkdir(parents=True, exist_ok=True)
  src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  n = 0
  for l in open(specs):
    sp = json.loads(l)
    ter = sp["env"]["terrain"]
    b = build(ter)
    psi_c, R, t = b["tf"]
    bp, bq, bh, top, names = [], [], [], [], []
    for g in b["boxes"]:
      xy = R @ np.array(g["c"][:2]) + t
      yaw = psi_c + g["yaw"]
      bq.append(qmul([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], [math.cos(g["pitch"] / 2), 0.0, math.sin(g["pitch"] / 2), 0.0]))
      bp.append([xy[0], xy[1], g["c"][2]]); bh.append(g["half"]); top.append(b["riser"].get(g["name"], np.nan)); names.append(g["name"])
    hf = {}
    for k_, h in enumerate(b["hfs"]):
      xy = R @ np.array(h["c"][:2]) + t
      hf.update({f"hf{k_}_pos": np.array([xy[0], xy[1], h["c"][2]]), f"hf{k_}_quat": np.array([math.cos(psi_c / 2), 0, 0, math.sin(psi_c / 2)]),
                 f"hf{k_}_size": np.array(h["size"]), f"hf{k_}_data": np.asarray(h["data"], np.float32)})
    meta = dict(spec_sha256=sp["spec_sha256"], cell=sp["cell"], i=sp["i"], evalset=sp["evalset"], generator_sha256=src,
                frame="world (mujoco), boxes = MuJoCo half sizes, hfield data in [0, 1] -> z = pos_z + data * size[2]",
                plane_z=b["plane_z"], x_goal=b["x_goal"], x_end=b["x_end"], tf=dict(psi_c=psi_c, R=np.asarray(R).tolist(), t=list(map(float, t))),
                mu=b["mu"], segs=b.get("segs"), eval_only=sp["evalset"] == "E40")
    np.savez_compressed(out / f"{sp['spec_sha256'][:16]}.npz", meta=json.dumps(meta), box_name=np.array(names), box_pos=np.array(bp).reshape(-1, 3),
                        box_quat=np.array(bq).reshape(-1, 4), box_half=np.array(bh).reshape(-1, 3), box_riser_top=np.array(top), **hf)
    n += 1
  print(n, "geometry files ->", out)


# ---------------------------------------------------------------- summary table
def summarize(md, trk_dir, mf_dir):
  def load(d):
    rows = {}
    for l in (Path(d) / "rows.jsonl").read_text().splitlines():
      r = json.loads(l)
      if r.get("family") == "terrain" and "error" not in r:
        rows.setdefault(r["cell"].rsplit("_", 1)[0], {})[r["i"]] = r
    return rows
  T, M = load(trk_dir), load(mf_dir)
  f = (lambda x, k: round(float(np.median(x)) * k, 1) if len(x) else "-")

  def stat(rs):
    rs = list(rs.values())
    c = sum(r["disp"] == "C" for r in rs)
    per = [sum(r["disp"] == "C" for r in rs if r["seed_i"] == s) for s in range(3)]
    dsp = "/".join(str(sum(r["disp"] == x for r in rs)) for x in "FHSL")
    toe = sum(bool(r.get("toe_ep")) for r in rs) if rs and rs[0].get("toe_ep") is not None else "-"
    lp = [r["lift_pre_med"] for r in rs if r.get("lift_pre_med") is not None]
    lo = [r["lift_obs_max"] for r in rs if r.get("lift_obs_max") is not None]
    tr = [r["tread_min"] for r in rs if r.get("tread_min") is not None]
    ms = sum(r.get("misstep") or 0 for r in rs) if rs and rs[0].get("misstep") is not None else "-"
    slp = [r["slip_p95"] for r in rs if r.get("slip_p95") is not None]
    tt = [r["t_ratio"] for r in rs if r.get("t_ratio") is not None]
    return dict(n=len(rs), c=c, per="-".join(map(str, per)), dsp=dsp, toe=toe, lp=f(lp, 100), lo=f(lo, 100),
                tr=(round(min(tr) * 100, 1) if tr else "-"), ms=ms, slp=f(slp, 100), tt=f(tt, 1) if not tt else round(float(np.median(tt)), 2))
  lines = ["| 칸 | 정책 | 완주 /n | 씨앗별 완주(/4) | 넘어짐/뜀/서버림/늦음 | 발끝 걸림 판 | 들기 평지 중앙 cm | 들기 장애 최대 중앙 cm | 디딤 여유 최소 cm | 헛디딤 | 미끄럼 p95 cm/s | 걸린 시간 ÷ 명목 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
  for cell in sorted(set(T) | set(M)):
    for name, src in (("mf", M), ("R14", T)):
      if cell in src:
        s = stat(src[cell])
        lines.append(f"| {cell} | {name} | {s['c']}/{s['n']} | {s['per']} | {s['dsp']} | {s['toe']} | {s['lp']} | {s['lo']} | {s['tr']} | {s['ms']} | {s['slp']} | {s['tt']} |")
  Path(md).write_text("\n".join(lines) + "\n")
  print(len(lines) - 2, "rows ->", md)


if __name__ == "__main__":
  a = sys.argv
  if len(a) > 1 and a[1] == "gen":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("outdir")
    ap.add_argument("--set", required=True, choices=["eval", "dev", "diag"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--end", default="both", choices=["far", "near", "both", "house", "flight", "mix"])
    ap.add_argument("--ctl", action="store_true")
    ap.add_argument("--stand", action="store_true")
    ap.add_argument("--starts", default="rand", choices=["rand", "k12"])
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--k12shift", type=float, default=0.0)
    g = ap.parse_args()
    gen(g.outdir, g.set, [int(x) for x in g.seeds.split(",")], g.end, g.ctl, g.stand, g.starts, g.push, g.k12shift)
  elif len(a) > 1 and a[1] == "geom":
    geom_dump(a[2], a[3])
  elif len(a) > 1 and a[1] == "sum":
    summarize(a[2], a[3], a[4])
  else:
    if len(a) > 1 and a[1] == "run":
      del a[1]
    sys.argv[0] = __file__
    if any(x.split("=", 1)[-1].startswith("v4:") for i, x in enumerate(a) if i and a[i - 1] == "--policy"):
      RG.claim(a[a.index("--out") + 1])
    NB.claim(a[a.index("--out") + 1])
    if "--specs" in a:   # 11-22 10:4x (coordinator 10:38 item 1): scene geometry of every spec next to the run (<out>/geom/<sha16>.npz)
      sp_, gd = a[a.index("--specs") + 1], Path(a[a.index("--out") + 1]) / "geom"
      geom_dump(sp_, gd)
      if any(json.loads(l).get("evalset") == "E40" for l in open(sp_)):
        (gd / "EVAL_ONLY.txt").write_text("eval specs (E40): the geometry encodes the eval starts and draws; not for positive / tracker / synthesizer / planner\n")
    P.main()
