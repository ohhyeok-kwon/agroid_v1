"""Positive 20 hold test with the stock R1 fist (rounds/R04/positive/r11_hold/rules.md; coordinator 16:0x).

The object starts between the two fists (no approach, no table). [0, T_SQ): the fists move from just touching (s_start)
to the set squeeze (s_sq, PD targets d_in inside the object). [0, T_A): a soft support holds the object (its weight plus
a critically damped 4 Hz spring to the start pose, as an external force / torque at its centre of mass); [T_A, T_REL):
weight only (the object settles into the fists' own equilibrium); from T_REL on the object is free and the fists hold it
for 10 s. Upper bound of the hold any policy could reach with this fist body, this engine and these contact settings.

  python -m r1adv.positive_obj.p20_hold cpu|mjw <ik_file> <out_dir> <seed> <arm> [panel ...] [--workers N] [--limit N]
                                                                    [--strata a,b] [--first s:n,...] [--tend T]
Arms: fist (stock capsule), fistfit (box fitted to the visual fist), fistpads (fistfit + 3 x 3 pads; side arm).
"""
from __future__ import annotations

import functools
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

from r1adv.positive_obj import p20_fist as F
from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_scene as PS

RULES = "rounds/R04/positive/r11_hold/rules.md"
DT, DECIM = 0.005, 4
T_SQ, T_A, T_REL, T_END = 0.5, 0.8, 1.2, 11.2
NSTEP, NCTRL = int(round(T_END / DT)), int(round(T_END / DT / DECIM))      # 2240, 560
K_A, K_REL = int(round(T_A / DT)), int(round(T_REL / DT))                    # 160, 240
SUP_W = 2 * math.pi * 4.0
X_G, Z_G = 0.36, 0.76
K_PD = 240.0                                  # positive 18 measured squeeze stiffness per hand (N/m); world design only
S_MIN, S_MAX = 0.09, 0.25                     # fist IK grid at (X_G, Z_G): position residual <= 5 mm at dz = 0
D_IN_CAP = 0.10
R_MIN = min(F.R_FACE.values())                # 0.030: d_max is arm independent (same worlds for every arm)
ARMS = {"fist": dict(code=10), "fistfit": dict(code=11), "fistpads": dict(code=12)}
STRATA = ((20, "H1V", 1024), (21, "H1A", 512), (22, "HC1", 512))
PARAM_NAMES = ("stratum", "shape", "dx", "dy", "dz", "mass", "mu", "tc", "impratio", "d_in", "s_start", "s_sq", "z_top",
               "table_x", "twin_of", "ik_ok", "grasp", "arm", "mesh_id", "orient", "budget", "forearm_clear")
# forearm clearance (rules 3): the forearm capsule (r1.xml <side>_elbow_collision, r 0.03, coaxial with the wrist roll axis)
# ends at wrist-frame x 0.0028; an object side that is straight along the forearm reaches it when its rear extent behind
# the arm's patch centre exceeds patch x - 0.0028 - 5 mm.
FOREARM_FRONT_X = 0.085 + 0.03 - 0.11218
X_CLEAR = {a: F.FIST_FRAME[0] + F.PATCH[a][0] - FOREARM_FRONT_X - 0.005 for a in ARMS}       # fist 0.077, fit/pads 0.104
Q_VERTICAL = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)       # capsule axis (body x) turned to world -z
P = {n: i for i, n in enumerate(PARAM_NAMES)}


# ------------------------------------------------------------------------------------------------ model
def robot_cfg(arm: str):
  from r1adv.positive_obj.robot_fixed import PELVIS_POS
  return F.robot_cfg(arm, fixed=True, pelvis_pos=PELVIS_POS)


def object_spec_bits(shape, condim=3, dims=None, mass=1.0, mu=0.7, tc=0.02):
  spec = PS.object_spec(shape, condim, dims, mass, mu, tc)
  for g in spec.geoms:
    g.contype, g.conaffinity = 3, 3
  return spec


def object_entity_cfg(shape: str, condim: int = 3):
  from mjlab.entity import EntityCfg
  px, py = PS.PARK_POS[shape]
  return EntityCfg(spec_fn=functools.partial(object_spec_bits, shape, condim),
                   init_state=EntityCfg.InitialStateCfg(pos=(px, py, PS.NOMINAL[shape][2] / 2 + 0.002)))


def build_model(arm: str, cone: str, condim: int):
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim import MujocoCfg
  from mjlab.terrains import TerrainEntityCfg
  ents = {"robot": robot_cfg(arm)}
  for shape in PS.SHAPES:
    ents[PS.ENTITY_OF[shape]] = object_entity_cfg(shape, condim)
  m = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"), entities=ents), device="cpu").compile()
  MujocoCfg(cone=cone, **L.SIM_OPT).apply(m)
  return m


class FistGrid(L.Grid):
  """IK grid of the fist frame; residual read at dz = 0 only (no lift in the hold test)."""

  def __init__(self, path: str):
    super().__init__(path)
    self.res = np.load(path)["pos_err"][:, 0, :].max(-1)


# ------------------------------------------------------------------------------------------------ worlds
def d_max(w: float) -> float:
  return float(np.clip(w / 2 + R_MIN - S_MIN, 0.0, D_IN_CAP))


def sample_base(seed: int) -> list[dict]:
  rng = np.random.Generator(np.random.PCG64(seed))
  U = rng.uniform
  out = []
  for code, _, n in STRATA:
    for _ in range(n):
      u = U(0.10, 0.40, size=3)
      w = U(0.20, 0.40)
      shape = int(rng.integers(3))
      L_ = U(w, 0.40)
      if code == 21:                                   # task distribution: deep boxes, capsules lying along the forearm
        dims, orient = {0: (u[0], w, u[2]), 1: (w, w, u[2]), 2: (L_, w, w)}[shape], 0
      else:                                            # verdict / creep: forearm clear by construction (rules 3)
        dx = 0.10 + 0.05 * (u[0] - 0.10) / 0.30        # box depth 0.10-0.15
        dims, orient = {0: (dx, w, u[2]), 1: (w, w, u[2]), 2: (L_, w, w)}[shape], (1 if shape == 2 else 0)
      dm = d_max(w)
      if code == 20:                                   # verdict: mu 0.5, friction budget >= 2 by parameters
        mu = 0.5
        mass = U(0.2, min(3.0, max(0.2, dm * mu * K_PD / 9.81)))
      elif code == 21:                                 # task distribution
        mu = 0.5 if rng.random() < 0.25 else U(0.4, 1.0)
        mass = U(0.2, 3.0)
      else:                                            # creep: light, high friction
        mu = U(0.7, 1.0)
        mass = U(0.2, 0.5)
      d_req = mass * 9.81 / (mu * K_PD)                # mu * 2 * K * d_in = 2 * m g (budget 2)
      d_in = float(np.clip(max(d_req, 0.02 if code == 22 else 0.01), 0.01, max(dm, 0.01)))
      tc = U(0.010, 0.040)
      imp = 10.0 if rng.random() < 0.5 else 1.0
      out.append(dict(stratum=code, shape=shape, dims=PS.snap_dims(PS.SHAPES[shape], dims), mass=mass, mu=mu, tc=tc, imp=imp,
                      d_in=d_in, orient=orient))
  return out


def arm_params(arm: str, base: list[dict], grid: FistGrid) -> np.ndarray:
  r = F.R_FACE[arm]
  rows = []
  for w in base:
    x, y, z = w["dims"]
    s_sq = y / 2 + r - w["d_in"]
    s_start = y / 2 + r + 0.001
    ik_ok = float(max(grid.residual(s_sq), grid.residual(s_start)) <= 0.005 and S_MIN - 1e-9 <= s_sq and s_start <= S_MAX + 1e-9)
    budget = w["mu"] * 2 * K_PD * w["d_in"] / (w["mass"] * 9.81)
    sname = PS.SHAPES[w["shape"]]
    mesh_id = -1 if sname == "box" else PS.mesh_index(sname, w["dims"])
    rear = {"box": x / 2, "cylinder": 0.0, "capsule": 0.0 if w["orient"] == 1 else x / 2}[sname]
    clear = float(rear <= X_CLEAR[arm] + 1e-9)
    rows.append([w["stratum"], w["shape"], x, y, z, w["mass"], w["mu"], w["tc"], w["imp"], w["d_in"], s_start, s_sq,
                 float("nan"), float("nan"), -1, ik_ok, 0, ARMS[arm]["code"], mesh_id, float(w["orient"]), budget, clear])
  return np.array(rows, dtype=np.float64)


# ------------------------------------------------------------------------------------------------ script
def world_targets(row, grid: FistGrid, lay, which="T"):
  t = np.arange(NCTRL) * DT * DECIM
  s = row[P["s_start"]] + (row[P["s_sq"]] - row[P["s_start"]]) * L.ramp(t, 0.0, T_SQ)
  zero = np.zeros_like(t)
  q = grid.joints(s, zero, zero, which)
  tgt = np.zeros((NCTRL, lay.nu))
  tgt[:, lay.ik_act] = q[:, lay.ik_from]
  return tgt


def initial_q(row, grid, lay):
  return world_targets(row, grid, lay, "Q")[0].copy(), world_targets(row, grid, lay, "T")


def set_fields_c(m, idx, row, condim):
  active = PS.SHAPES[int(row[P["shape"]])]
  for shape in PS.SHAPES:
    if shape == active:
      PS.write_world_fields_c(m, idx, shape, row[[P["dx"], P["dy"], P["dz"]]], row[P["mass"]], condim, row[P["mu"]], row[P["tc"]])
    else:
      PS.write_world_fields_c(m, idx, shape, PS.NOMINAL[shape], 1.0, condim, 0.7, 0.02)
  m.opt.impratio = row[P["impratio"]]
  return active


def init_state_c(m, d, idx, lay, tgt0, q0, active, orient=0):
  """Robot at the t = 0 pose; the object centred between the two contact-patch sites (kinematics only); orient 1 turns a
  capsule upright."""
  mujoco.mj_resetData(m, d)
  d.qpos[lay.act_qadr] = q0
  for shape, qa in idx.obj_qadr.items():
    px, py = PS.PARK_POS[shape]
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = (px, py, PS.NOMINAL[shape][2] / 2 + 0.002), (1.0, 0, 0, 0)
  mujoco.mj_kinematics(m, d)
  pl, pr = d.site_xpos[m.site("robot/left_patch").id], d.site_xpos[m.site("robot/right_patch").id]
  qa = idx.obj_qadr[active]
  d.qpos[qa:qa + 3] = (pl + pr) / 2
  if orient == 1:
    d.qpos[qa + 3:qa + 7] = Q_VERTICAL
  d.ctrl[:] = tgt0
  mujoco.mj_forward(m, d)


def init_gate(m, d, idx, active) -> tuple[float, float, float]:
  """t = 0 geometry (rules 'gate'): deepest hand-object, non-hand robot-object and hand-hand penetration (m)."""
  ag = idx.obj_geom[active]
  hand = {g: k for k, gs in idx.hand_geoms.items() for g in gs}
  robot = set(idx.robot_geoms)
  pen_h = pen_r = pen_hh = 0.0
  for i in range(d.ncon):
    c = d.contact[i]
    g1, g2 = int(c.geom1), int(c.geom2)
    dep = max(-float(c.dist), 0.0)
    if ag in (g1, g2):
      o = g2 if g1 == ag else g1
      if o in hand:
        pen_h = max(pen_h, dep)
      elif o in robot:
        pen_r = max(pen_r, dep)
    elif g1 in hand and g2 in hand and hand[g1] != hand[g2]:
      pen_hh = max(pen_hh, dep)
  return pen_h, pen_r, pen_hh


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def support_c(m, d, ob, oq, od, qfix, spring: bool) -> np.ndarray:
  """Soft support (rules 'start'): weight (+ critically damped spring to the start pose when spring), world-frame force
  at the centre of mass and torque (MuJoCo xfrc_applied). Scalar inertia = largest principal moment."""
  mass, inert = float(m.body_mass[ob]), float(m.body_inertia[ob].max())
  out = np.zeros(6)
  out[2] = mass * 9.81
  if not spring:
    return out
  kp, kd = mass * SUP_W ** 2, 2 * mass * SUP_W
  kr, kdr = inert * SUP_W ** 2, 2 * inert * SUP_W
  p, q = d.qpos[oq:oq + 3], d.qpos[oq + 3:oq + 7]
  v, w_loc = d.qvel[od:od + 3], d.qvel[od + 3:od + 6]
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, q)
  w = R.reshape(3, 3) @ w_loc
  qe = quat_mul(q, qfix[3:7] * np.array([1.0, -1.0, -1.0, -1.0]))
  e = 2.0 * (1.0 if qe[0] >= 0 else -1.0) * qe[1:]
  out[:3] += -kp * (p - qfix[:3]) - kd * v
  out[3:] = -kr * e - kdr * w
  return out


def alloc(n, nstep=NSTEP):
  T, T50 = nstep, nstep // DECIM
  return dict(obj_v=np.zeros((n, T, 3), np.float32), hand_v=np.zeros((n, T, 2, 3), np.float32),
              f_hand=np.zeros((n, T, 2, 3), np.float32), fn_hand=np.zeros((n, T, 2), np.float32),
              dist_min=np.ones((n, T, 2), np.float32), ncon=np.zeros((n, T, 8), np.int16), ncap=np.zeros((n, T, 2), np.int16),
              obj_p=np.zeros((n, T50, 3), np.float32), obj_q=np.zeros((n, T50, 4), np.float32),
              hand_p=np.zeros((n, T50, 2, 3), np.float32), hand_q=np.zeros((n, T50, 2, 4), np.float32),
              tau_ratio=np.zeros((n, T50, 10), np.float32), waist_yaw=np.zeros((n, T50), np.float32),
              nefc_max=np.zeros(n, np.int32), init_gate=np.zeros((n, 3), np.float32))


def run_worlds_c(arm, panel, rows, ik_file, tend=None):
  cone, condim = L.PANELS[panel]
  grid = FistGrid(ik_file)
  m = build_model(arm, cone, condim)
  idx = PS.Index(m)
  lay = L.Layout(m, grid.names)
  d = mujoco.MjData(m)
  nstep = NSTEP if tend is None else int(round(tend / DT))
  out = alloc(len(rows), nstep)
  res6 = np.zeros(6)
  for w, row in enumerate(rows):
    active = set_fields_c(m, idx, row, condim)
    q0, tgt = initial_q(row, grid, lay)
    init_state_c(m, d, idx, lay, tgt[0], q0, active, int(row[P["orient"]]))
    out["init_gate"][w] = init_gate(m, d, idx, active)
    oq, od, ag, ob = idx.obj_qadr[active], idx.obj_dadr[active], idx.obj_geom[active], idx.obj_body[active]
    qfix = d.qpos[oq:oq + 7].copy()
    tmax = np.zeros(10)
    for k in range(nstep):
      if k % DECIM == 0:
        d.ctrl[:] = tgt[k // DECIM]
        tmax[:] = 0.0
      d.xfrc_applied[ob] = support_c(m, d, ob, oq, od, qfix, k < K_A) if k < K_REL else 0.0
      mujoco.mj_step(m, d)
      out["obj_v"][w, k] = d.qvel[od:od + 3]
      for h, hb in enumerate(idx.hand_bodies):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, hb, res6, 0)
        out["hand_v"][w, k, h] = res6[3:]
      counts, fh, fn, dmin, ncap = PS.classify_contacts_c(m, d, idx, ag)
      out["ncon"][w, k], out["f_hand"][w, k], out["fn_hand"][w, k] = counts, fh, fn
      out["dist_min"][w, k], out["ncap"][w, k] = dmin, ncap
      out["nefc_max"][w] = max(out["nefc_max"][w], d.nefc)
      tmax = np.maximum(tmax, np.abs(d.actuator_force[lay.arm_act]) / lay.effort[lay.arm_act])
      if k % DECIM == DECIM - 1:
        c = k // DECIM
        out["obj_p"][w, c], out["obj_q"][w, c] = d.qpos[oq:oq + 3], d.qpos[oq + 3:oq + 7]
        for h, hb in enumerate(idx.hand_bodies):
          out["hand_p"][w, c, h], out["hand_q"][w, c, h] = d.xpos[hb], d.xquat[hb]
        out["tau_ratio"][w, c] = tmax
        out["waist_yaw"][w, c] = d.qpos[lay.waist_yaw_qadr]
  return out


def _worker(args):
  arm, panel, rows, ik_file, tend = args
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  return run_worlds_c(arm, panel, rows, ik_file, tend)


def run_panel_cpu(arm, panel, params, ik_file, workers, chunk=8, tend=None):
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  jobs = [(arm, panel, params[i:i + chunk], ik_file, tend) for i in range(0, len(params), chunk)]
  with ProcessPoolExecutor(max_workers=workers) as ex:
    parts = list(ex.map(_worker, jobs))
  return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, time.time() - t0


def select(params, strata=None, first=None, limit=None):
  if strata is not None:
    keep = np.zeros(len(params), bool)
    for s in strata:
      ids = np.nonzero(params[:, P["stratum"]] == s)[0]
      if first is not None:
        ids = ids[:first.get(s, len(ids))]
      keep[ids] = True
    params = params[keep]
  return params if limit is None else params[:limit]


def main(argv):
  engine, ik_file, out_dir, seed, arm = argv[0], argv[1], argv[2], int(argv[3]), argv[4]
  rest = argv[5:]
  workers, limit, strata, first, tend = 32, None, None, None, None
  for flag in ("--workers", "--limit", "--strata", "--first", "--tend"):
    if flag in rest:
      i = rest.index(flag)
      v = rest[i + 1]
      del rest[i:i + 2]
      if flag == "--workers":
        workers = int(v)
      elif flag == "--limit":
        limit = int(v)
      elif flag == "--strata":
        strata = [int(x) for x in v.split(",")]
      elif flag == "--first":
        first = {int(a): int(b) for a, b in (x.split(":") for x in v.split(","))}
      else:
        tend = float(v)
  panels = rest or list(L.PANELS)
  grid = FistGrid(ik_file)
  params = select(arm_params(arm, sample_base(seed), grid), strata, first, limit)
  seed_dir = os.path.join(out_dir, arm, engine, f"s{seed}")
  os.makedirs(seed_dir, exist_ok=True)
  for panel in panels:
    cone, condim = L.PANELS[panel]
    if engine == "cpu":
      out, wall = run_panel_cpu(arm, panel, params, ik_file, workers, tend=tend)
      extra = {}
    else:
      from r1adv.positive_obj import p20_hold_mjw as HW
      out, wall, extra = HW.run_panel(arm, panel, params, ik_file, tend=tend)
    meta = dict(engine=engine, arm=arm, body=arm, panel=panel, cone=cone, condim=condim, seed=seed, n=len(params), wall_s=wall,
                mujoco=mujoco.__version__, opt=L.SIM_OPT, rules=RULES, t_end=T_END if tend is None else tend, t_sq=T_SQ,
                t_a=T_A, t_release=T_REL, x_g=X_G, z_g=Z_G,
                note="hold test: soft support (weight + 4 Hz spring) t < t_a, weight only t_a <= t < t_release; no table")
    meta.update(extra)
    o = dict(out)
    o["ncon_wrist"] = o["ncap"]
    np.savez(os.path.join(seed_dir, f"{panel}.npz"), params=params, param_names=np.array(PARAM_NAMES), meta=json.dumps(meta), **o)
    print(f"{engine} {arm} seed {seed} {panel}: {len(params)} worlds, {wall:.1f} s, nefc max {int(out['nefc_max'].max())}, "
          f"ncon max {int(out['ncon'][..., 0].max())}, overflow {extra.get('overflow')}", flush=True)


if __name__ == "__main__":
  main(sys.argv[1:])
