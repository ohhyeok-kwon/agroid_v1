"""Positive 20 two-arm contact ladder (rounds/R04/positive/r11_contact2/rules.md c394a1ac + rules_amend.md).
World sampling, per-arm scripts (G1 two-hand squeeze, G2 O6-lite one-hand side grasp), scene build and the C runner.
The MJWarp runner is p20_ladder_mjw.py (same worlds, targets, initial state and trace format).

  python -m r1adv.positive_obj.p20_ladder cpu <ik_dir> <out_dir> <seed> <arm> [panel ...] [--workers N] [--limit N] [--strata 0,1,..] [--first K]
  python -m r1adv.positive_obj.p20_ladder mjw <ik_dir> <out_dir> <seed> <arm> [panel ...] [--limit N]
Arms: box, o6, o6straight, o6full (+ pad arms of rules_amend). Panels: py3 py4 el3 el4.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import mujoco
import numpy as np

from r1adv.positive_obj import ik as IK
from r1adv.positive_obj import p20_bodies as B
from r1adv.positive_obj import p20_scene as PS
from r1adv.positive_obj import scene as S

RULES_SHA = "c394a1ac44fda9c05e2b953f7fbda7f73b8884966e77f68483eb5ac596f0bf02"
DT, DECIM = 0.005, 4
NSTEP, NCTRL = 4800, 1200
PHASES = dict(P0=(0.0, 1.0), P1=(1.0, 2.5), P2=(2.5, 3.5), P3=(3.5, 5.5), P4=(5.5, 15.5), P5=(15.5, 18.5),
              P6=(18.5, 20.5), P7=(20.5, 22.0), P8=(22.0, 24.0))
X_G, Z_G, LIFT = IK.X_G, IK.Z_G, IK.LIFT
YAW_AXIS_X = IK.YAW_AXIS_X
PSI_PICK, PSI_PLACE = -IK.carry_yaw(X_G) / 2, IK.carry_yaw(X_G) / 2
PANELS = {"py3": ("pyramidal", 3), "py4": ("pyramidal", 4), "el3": ("elliptic", 3), "el4": ("elliptic", 4)}
SIM_OPT = dict(timestep=DT, iterations=10, ls_iterations=20, ccd_iterations=50, tolerance=1e-8, ls_tolerance=0.01)
# arm: body, force scale, G1 finger mode, pads, arm code, strata, panels
ARMS = {
  "box": dict(body="box_wristfix", scale=1.0, g1="none", pads=False, code=0, strata=(0, 1, 4, 5, 6), panels=tuple(PANELS)),
  "o6": dict(body="o6lite", scale=B.GRIP_FORCE_SCALE, g1="close", pads=False, code=1, strata=(0, 1, 4, 5, 6), panels=tuple(PANELS)),
  "o6straight": dict(body="o6lite", scale=B.GRIP_FORCE_SCALE, g1="straight", pads=False, code=2, strata=(0, 1), panels=("py3", "el4")),
  "o6full": dict(body="o6lite", scale=1.0, g1="close", pads=False, code=3, strata=(5, 6), panels=tuple(PANELS)),
  # rules_amend A1 (coordinator 14:3x): contact-pad arms, secondary
  "boxpads": dict(body="box_wristfix", scale=1.0, g1="none", pads=True, code=4, strata=(0, 1, 4), panels=tuple(PANELS)),
  "o6pads": dict(body="o6lite", scale=B.GRIP_FORCE_SCALE, g1="close", pads=True, code=5, strata=(0, 1, 4, 5, 6), panels=tuple(PANELS)),
}
STRATA = ((0, "A1", 1024), (1, "C1", 512), (4, "L", 512), (5, "A2", 512), (6, "C2", 512))
G2_STRATA = (5, 6)
O6_G1_PATCH = 0.037                                    # O6-lite G1: object centre = palm centre + 0.037 (palm + straight fingers)
G2_S_SQ, G2_S_PRE, G2_S_R = 0.155, 0.21, 0.22          # rules_amend A1: s_sq 0.155 (palm face touches, no push: tall thin objects tipped at 5 mm), s_R 0.22 (IK residual at 0.25 is 9.8 mm)
G2_OBJ_Y = 0.135
# rules_amend A1 (dev seed 0): the G2 forearm is pitched -13.5 deg at the grasp pose (a level forearm is out of reach,
# residual 5-9 cm), so the thumb sweep plane leans forward. The thumb is opposed first, far from the object (P0, s_pre),
# and the object sits with its front face on the MCP line, its rear face at least 8 mm ahead of the palm centre.
G2_REAR_MIN, G2_MCP_X = 0.008, B.PALM_HALF[0]
PARAM_NAMES = ("stratum", "shape", "dx", "dy", "dz", "mass", "mu", "tc", "impratio", "d_in", "s_pre", "s_sq", "z_top",
               "table_x", "twin_of", "ik_ok", "grasp", "arm", "mesh_id")
P = {n: i for i, n in enumerate(PARAM_NAMES)}
ARM_JOINTS = tuple(f"{s}_{j}_joint" for s in ("left", "right")
                   for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"))
RIGHT_ARM = tuple(f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"))
IK_FILES = {("box_wristfix", 0): "box_g1.npz", ("o6lite", 0): "o6_g1.npz", ("o6lite", 1): "o6_g2.npz"}


def minjerk(tau):
  tau = np.clip(tau, 0.0, 1.0)
  return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def ramp(t, a, b):
  return minjerk((t - a) / (b - a))


class Grid:
  """IK grid of one (body, grasp type): joint targets bilinear in (s, dz) with waist yaw psi, waist roll 0, plus the
  trilinear gravity feed-forward (positive 18 rules_amend A1)."""

  def __init__(self, path: str):
    z = np.load(path)
    self.Q, self.s, self.dz, self.G, self.psi = z["Q"], z["s_grid"], z["u_grid"], z["G"], z["psi_grid"]
    self.x_g = float(z["x_g"])
    self.names = [str(n) for n in z["joint_names"]]
    self.yaw_i, self.roll_i = self.names.index("waist_yaw_joint"), self.names.index("waist_roll_joint")
    self.res = z["pos_err"].max(-1).max(-1)
    self.low = -z["hand_low"][:, 0]

  def residual(self, s):
    return np.interp(s, self.s, self.res)

  def box_low(self, s):
    return np.interp(s, self.s, self.low)

  def joints(self, s, dz, psi, which: str = "T") -> np.ndarray:
    s = np.clip(np.asarray(s, float), self.s[0], self.s[-1])
    dz = np.clip(np.asarray(dz, float), self.dz[0], self.dz[-1])
    psi = np.clip(np.asarray(psi, float), self.psi[0], self.psi[-1])
    fi = (s - self.s[0]) / (self.s[1] - self.s[0])
    fj = (dz - self.dz[0]) / (self.dz[1] - self.dz[0])
    fk = (psi - self.psi[0]) / (self.psi[1] - self.psi[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, len(self.s) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(self.dz) - 2)
    k0 = np.clip(np.floor(fk).astype(int), 0, len(self.psi) - 2)
    a, b, c = (fi - i0)[:, None], (fj - j0)[:, None], (fk - k0)[:, None]
    Q = self.Q
    q = (1 - a) * (1 - b) * Q[i0, j0] + a * (1 - b) * Q[i0 + 1, j0] + (1 - a) * b * Q[i0, j0 + 1] + a * b * Q[i0 + 1, j0 + 1]
    q[:, self.yaw_i] = psi
    q[:, self.roll_i] = 0.0
    if which == "T":
      g = 0.0
      for di, wi in ((0, 1 - a), (1, a)):
        for dj, wj in ((0, 1 - b), (1, b)):
          for dk, wk in ((0, 1 - c), (1, c)):
            g = g + wi * wj * wk * self.G[i0 + di, j0 + dj, k0 + dk]
      q = q + g
    return q


def load_grids(ik_dir: str) -> dict:
  return {k: Grid(os.path.join(ik_dir, f)) for k, f in IK_FILES.items()}


# ------------------------------------------------------------------------------------------------ worlds (rules 3)
def sample_base(seed: int) -> list[dict]:
  """Arm-independent world parameters, strata in rules order (A1, C1, L, A2, C2)."""
  rng = np.random.Generator(np.random.PCG64(seed))
  U = rng.uniform
  out = []
  for code, _, n in STRATA:
    for _ in range(n):
      if code not in G2_STRATA:
        u = U(0.10, 0.40, size=3)
        shape = int(rng.integers(3))
        L = U(u[1], 0.40)
        dims = {0: (u[0], u[1], u[2]), 1: (u[1], u[1], u[2]), 2: (L, u[1], u[1])}[shape]
        mass = U(0.2, 0.5) if code in (1, 4) else U(0.2, 3.0)
        half = rng.random() < 0.25
        mu_u = U(0.7, 1.0) if code == 1 else U(0.4, 1.0)
        mu = 0.5 if (code == 4 or (code == 0 and half)) else mu_u
        tc = U(0.010, 0.040)
        imp = 10.0 if rng.random() < 0.5 else 1.0
        d_in = U(0.02, 0.03) if code == 1 else U(0.01, 0.03)
        grasp = 0
      else:
        shape = 1 if rng.random() < 0.5 else 0          # 1 cylinder, 0 box
        if shape == 1:
          d, h = U(0.05, 0.10), U(0.10, 0.30)
          dims = (d, d, h)
        else:
          w, dx, h = U(0.04, 0.08), U(0.03, 0.05), U(0.10, 0.30)
          dims = (dx, w, h)
        mass = U(0.2, 0.5) if code == 6 else U(0.2, 3.0)
        half = rng.random() < 0.25
        mu_u = U(0.7, 1.0) if code == 6 else U(0.4, 1.0)
        mu = 0.5 if (code == 5 and half) else mu_u
        tc = U(0.010, 0.040)
        imp = 10.0 if rng.random() < 0.5 else 1.0
        d_in = 0.005
        grasp = 1
      sname = PS.SHAPES[shape]
      dims = PS.snap_dims(sname, dims)
      out.append(dict(stratum=code, shape=shape, dims=dims, mass=mass, mu=mu, tc=tc, imp=imp, d_in=d_in, grasp=grasp))
  return out


def obj_xy_torso(arm: str, w: dict) -> tuple[float, float]:
  """Object centre in the torso (pick) frame: G1 at the grasp point (x_g 0.40, 0); G2 behind-face-at-palm-centre."""
  if w["grasp"] == 1 and ARMS[arm]["body"] == "o6lite":
    dx, wy = w["dims"][0], w["dims"][1]
    rear = max(G2_MCP_X - dx, G2_REAR_MIN)
    return X_G + rear + dx / 2, G2_OBJ_Y - wy / 2
  return X_G, 0.0


def to_world_xy(xt: float, yt: float, psi: float) -> np.ndarray:
  c, s = math.cos(psi), math.sin(psi)
  return np.array([YAW_AXIS_X + c * (xt - YAW_AXIS_X) - s * yt, s * (xt - YAW_AXIS_X) + c * yt])


def arm_params(arm: str, base: list[dict], grids: dict) -> np.ndarray:
  body = ARMS[arm]["body"]
  rows = []
  for w in base:
    if w["stratum"] not in ARMS[arm]["strata"]:
      continue
    g2 = (w["grasp"] == 1 and body == "o6lite")
    x, y, z = w["dims"]
    sname = PS.SHAPES[w["shape"]]
    grid = grids[(body, 1 if g2 else 0)]
    if g2:
      s_sq, s_pre = G2_S_SQ, G2_S_PRE
    else:
      s_sq = y / 2 - w["d_in"] + 0.02
      s_pre = min(y / 2 + 0.08, 0.27)
    ik_ok = float(grid.residual(s_sq) <= 0.005)
    b = max(grid.box_low(s_pre), grid.box_low(s_sq))
    z_top = Z_G - max(z / 2, b + 0.005)
    xt, yt = obj_xy_torso(arm, w)
    xmin = min(S.footprint_xmin(sname, w["dims"], to_world_xy(xt, yt, p)[0], p) for p in (PSI_PICK, PSI_PLACE))
    mesh_id = -1 if sname == "box" else PS.mesh_index(sname, w["dims"])
    rows.append([w["stratum"], w["shape"], x, y, z, w["mass"], w["mu"], w["tc"], w["imp"], w["d_in"], s_pre, s_sq, z_top,
                 xmin - 0.02, -1, ik_ok, 1 if g2 else 0, ARMS[arm]["code"], mesh_id])
  return np.array(rows, dtype=np.float64)


def object_pose(arm: str, row: np.ndarray, psi: float = PSI_PICK) -> tuple[np.ndarray, np.ndarray]:
  w = dict(grasp=int(row[P["grasp"]]), dims=row[[P["dx"], P["dy"], P["dz"]]])
  xt, yt = obj_xy_torso(arm, w) if w["grasp"] == 1 else (X_G, 0.0)
  xy = to_world_xy(xt, yt, psi)
  return np.array([xy[0], xy[1], row[P["z_top"]] + row[P["dz"]] / 2 + 0.001]), S.yaw_quat(psi)


# ------------------------------------------------------------------------------------------------ scripts (rules 2.5)
def schedule(arm: str, row: np.ndarray, t: np.ndarray) -> dict:
  """s_L, s_R, dz_L, dz_R, psi, and per hand (grip, thumb yaw) at times t."""
  g2 = int(row[P["grasp"]]) == 1
  s_pre, s_sq = row[P["s_pre"]], row[P["s_sq"]]
  dz = LIFT * ramp(t, *PHASES["P3"]) - LIFT * ramp(t, *PHASES["P6"])
  psi = PSI_PICK + (PSI_PLACE - PSI_PICK) * ramp(t, *PHASES["P5"])
  zero = np.zeros_like(t)
  if not g2:
    s = s_pre + (s_sq - s_pre) * ramp(t, *PHASES["P1"]) + (s_pre - s_sq) * ramp(t, *PHASES["P7"])
    mode = ARMS[arm]["g1"]
    grip = (ramp(t, 2.5, 3.5) - ramp(t, 20.5, 21.25)) if mode == "close" else zero
    return dict(sL=s, sR=s, dzL=dz, dzR=dz, psi=psi, gripL=grip, gripR=grip, yawL=zero, yawR=zero)
  # rules_amend A1 (dev seed 0): thumb opposed in P0 at s_pre (away from the object), approach in P1, fingers in P2;
  # release: fingers open, hand back to s_pre, then the thumb returns
  s = s_pre + (s_sq - s_pre) * ramp(t, *PHASES["P1"]) + (s_pre - s_sq) * ramp(t, 21.0, 21.6)
  grip = ramp(t, 2.75, 3.5) - ramp(t, 20.5, 21.0)
  yaw = ramp(t, 0.1, 0.9) - ramp(t, 21.6, 22.0)
  return dict(sL=s, sR=np.full_like(t, G2_S_R), dzL=dz, dzR=zero, psi=psi, gripL=grip, gripR=zero, yawL=yaw, yawR=zero)


class Layout:
  def __init__(self, m: mujoco.MjModel, grid_names: list[str]):
    self.nu = m.nu
    self.act_joint = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])).removeprefix("robot/")
                      for a in range(m.nu)]
    self.ik_act = np.array([a for a, n in enumerate(self.act_joint) if n in grid_names])
    self.ik_from = np.array([grid_names.index(self.act_joint[a]) for a in self.ik_act])
    self.right_arm_act = np.array([a for a, n in enumerate(self.act_joint) if n in RIGHT_ARM])
    self.right_arm_from = np.array([grid_names.index(self.act_joint[a]) for a in self.right_arm_act])
    self.hand_act = {s: np.array([self.act_joint.index(j) for j in B.HAND_ACT_JOINTS[s]]) if B.HAND_ACT_JOINTS[s][0] in self.act_joint
                     else np.zeros(0, int) for s in ("left", "right")}
    self.act_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    self.arm_act = np.array([self.act_joint.index(n) for n in ARM_JOINTS])
    self.effort = np.abs(m.actuator_forcerange[:, 1]).copy()
    self.waist_yaw_qadr = int(m.jnt_qposadr[m.joint("robot/waist_yaw_joint").id])


def world_targets(arm: str, row: np.ndarray, grids: dict, lay: Layout, which: str = "T") -> tuple[np.ndarray, np.ndarray]:
  """(NCTRL, nu) PD targets at the control instants and (NCTRL, 2, 2) hand actions (grip, thumb yaw)."""
  body = ARMS[arm]["body"]
  g2 = int(row[P["grasp"]]) == 1
  grid = grids[(body, 1 if g2 else 0)]
  t = np.arange(NCTRL) * DT * DECIM
  sc = schedule(arm, row, t)
  qL = grid.joints(sc["sL"], sc["dzL"], sc["psi"], which)
  tgt = np.zeros((NCTRL, lay.nu))
  tgt[:, lay.ik_act] = qL[:, lay.ik_from]
  if g2:
    qR = grid.joints(sc["sR"], sc["dzR"], sc["psi"], which)
    tgt[:, lay.right_arm_act] = qR[:, lay.right_arm_from]
  hq = np.stack([np.stack([sc["gripL"], sc["yawL"]], -1), np.stack([sc["gripR"], sc["yawR"]], -1)], 1)
  for k, s in enumerate(("left", "right")):
    if len(lay.hand_act[s]):
      tgt[:, lay.hand_act[s]] = B.grip_targets(s, hq[:, k, 0], hq[:, k, 1])
  return tgt, hq


# ------------------------------------------------------------------------------------------------ scene
def robot_cfg(arm: str):
  from r1adv.positive_obj.robot_fixed import PELVIS_POS
  a = ARMS[arm]
  return B.robot_cfg(a["body"], fixed=True, pelvis_pos=PELVIS_POS, force_scale=a["scale"], pads=bool(a.get("pads")))


def build_model(arm: str, cone: str, condim: int):
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim import MujocoCfg
  from mjlab.terrains import TerrainEntityCfg
  ents = {"robot": robot_cfg(arm), "table": S.table_entity_cfg()}
  for shape in PS.SHAPES:
    ents[PS.ENTITY_OF[shape]] = PS.object_entity_cfg(shape, condim)
  scene = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"), entities=ents), device="cpu")
  m = scene.compile()
  MujocoCfg(cone=cone, **SIM_OPT).apply(m)
  return m


def set_fields_c(m, idx: PS.Index, row, condim) -> str:
  active = PS.SHAPES[int(row[P["shape"]])]
  for shape in PS.SHAPES:
    if shape == active:
      PS.write_world_fields_c(m, idx, shape, row[[P["dx"], P["dy"], P["dz"]]], row[P["mass"]], condim, row[P["mu"]], row[P["tc"]])
    else:
      PS.write_world_fields_c(m, idx, shape, PS.NOMINAL[shape], 1.0, condim, 0.7, 0.02)
  m.opt.impratio = row[P["impratio"]]
  return active


def init_state_c(arm, m, d, idx, lay, row, tgt0, q0, active):
  mujoco.mj_resetData(m, d)
  d.qpos[lay.act_qadr] = q0
  for shape, qa in idx.obj_qadr.items():
    if shape == active:
      pos, quat = object_pose(arm, row)
    else:
      px, py = PS.PARK_POS[shape]
      pos, quat = np.array([px, py, PS.NOMINAL[shape][2] / 2 + 0.002]), np.array([1.0, 0, 0, 0])
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, quat
  if idx.table_mocap >= 0:
    d.mocap_pos[idx.table_mocap] = [row[P["table_x"]], 0.0, row[P["z_top"]]]
    d.mocap_quat[idx.table_mocap] = [1.0, 0.0, 0.0, 0.0]
  d.ctrl[:] = tgt0
  mujoco.mj_forward(m, d)


def alloc(n: int) -> dict:
  return dict(obj_v=np.zeros((n, NSTEP, 3), np.float32), hand_v=np.zeros((n, NSTEP, 2, 3), np.float32),
              f_hand=np.zeros((n, NSTEP, 2, 3), np.float32), fn_hand=np.zeros((n, NSTEP, 2), np.float32),
              dist_min=np.ones((n, NSTEP, 2), np.float32), ncon=np.zeros((n, NSTEP, 8), np.int16),
              ncap=np.zeros((n, NSTEP, 2), np.int16), obj_p=np.zeros((n, NCTRL, 3), np.float32),
              obj_q=np.zeros((n, NCTRL, 4), np.float32), hand_p=np.zeros((n, NCTRL, 2, 3), np.float32),
              hand_q=np.zeros((n, NCTRL, 2, 4), np.float32), tau_ratio=np.zeros((n, NCTRL, 10), np.float32),
              waist_yaw=np.zeros((n, NCTRL), np.float32), hand_q_ctrl=np.zeros((n, NCTRL, 2, 2), np.float32),
              nefc_max=np.zeros(n, np.int32))


def initial_q(arm, row, grids, lay):
  """Initial actuated joint positions: the IK pose of t = 0 (no feed-forward), hand joints at their t = 0 targets."""
  tq, _ = world_targets(arm, row, grids, lay, "Q")
  tt, _ = world_targets(arm, row, grids, lay, "T")
  q0 = tq[0].copy()
  for s in ("left", "right"):
    if len(lay.hand_act[s]):
      q0[lay.hand_act[s]] = tt[0, lay.hand_act[s]]
  return q0, tt


def run_worlds_c(arm: str, panel: str, rows: np.ndarray, ik_dir: str) -> dict:
  cone, condim = PANELS[panel]
  grids = load_grids(ik_dir)
  m = build_model(arm, cone, condim)
  idx = PS.Index(m)
  lay = Layout(m, grids[(ARMS[arm]["body"], 0)].names)
  d = mujoco.MjData(m)
  out = alloc(len(rows))
  res6 = np.zeros(6)
  for w, row in enumerate(rows):
    active = set_fields_c(m, idx, row, condim)
    q0, tgt = initial_q(arm, row, grids, lay)
    _, hq = world_targets(arm, row, grids, lay)
    out["hand_q_ctrl"][w] = hq
    init_state_c(arm, m, d, idx, lay, row, tgt[0], q0, active)
    oq, od, ag = idx.obj_qadr[active], idx.obj_dadr[active], idx.obj_geom[active]
    tmax = np.zeros(10)
    for k in range(NSTEP):
      if k % DECIM == 0:
        d.ctrl[:] = tgt[k // DECIM]
        tmax[:] = 0.0
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
  arm, panel, rows, ik_dir = args
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  return run_worlds_c(arm, panel, rows, ik_dir)


def run_panel_cpu(arm, panel, params, ik_dir, workers, chunk=8):
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  jobs = [(arm, panel, params[i:i + chunk], ik_dir) for i in range(0, len(params), chunk)]
  with ProcessPoolExecutor(max_workers=workers) as ex:
    parts = list(ex.map(_worker, jobs))
  return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, time.time() - t0


def save(path: str, out: dict, params: np.ndarray, meta: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  out = dict(out)
  out["ncon_wrist"] = out["ncap"]
  np.savez(path, params=params, param_names=np.array(PARAM_NAMES), meta=json.dumps(meta), **out)


def save_empty(seed_dir: str) -> None:
  os.makedirs(seed_dir, exist_ok=True)
  np.savez(os.path.join(seed_dir, "empty.npz"), params=np.zeros((0, len(PARAM_NAMES))), param_names=np.array(PARAM_NAMES),
           hand_v=np.zeros((0, NSTEP, 2, 3), np.float32), ncon=np.zeros((0, NSTEP, 8), np.int16),
           meta=json.dumps(dict(note="no empty twins in P20 (rules 3); zero-world file for reader compatibility")))


def select(params: np.ndarray, strata=None, first=None, limit=None) -> np.ndarray:
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
  engine, ik_dir, out_dir, seed, arm = argv[0], argv[1], argv[2], int(argv[3]), argv[4]
  rest = argv[5:]
  workers, limit, strata, first = 16, None, None, None
  for flag in ("--workers", "--limit", "--strata", "--first"):
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
      else:
        first = {int(a): int(b) for a, b in (x.split(":") for x in v.split(","))}
  panels = rest or list(ARMS[arm]["panels"])
  grids = load_grids(ik_dir)
  base = sample_base(seed)
  params_all = arm_params(arm, base, grids)
  params = select(params_all, strata, first, limit)
  seed_dir = os.path.join(out_dir, arm, engine, f"s{seed}")
  save_empty(seed_dir)
  sig = B.body_signature(ARMS[arm]["body"], force_scale=ARMS[arm]["scale"], **({"pads": True} if ARMS[arm].get("pads") else {}))["total"]
  for panel in panels:
    cone, condim = PANELS[panel]
    if engine == "cpu":
      out, wall = run_panel_cpu(arm, panel, params, ik_dir, workers)
      extra = {}
    else:
      from r1adv.positive_obj import p20_ladder_mjw as W
      out, wall, extra = W.run_panel(arm, panel, params, ik_dir)
    meta = dict(engine=engine, arm=arm, body=ARMS[arm]["body"], force_scale=ARMS[arm]["scale"], g1=ARMS[arm]["g1"],
                panel=panel, cone=cone, condim=condim, seed=seed, n=len(params), wall_s=wall, mujoco=mujoco.__version__,
                opt=SIM_OPT, rules_sha256=RULES_SHA, body_signature_v2=sig,
                note="obj_* = state after the physics step; hand_*, contacts, forces = kinematics/contacts of the step start")
    meta.update(extra)
    save(os.path.join(seed_dir, f"{panel}.npz"), out, params, meta)
    print(f"{engine} {arm} seed {seed} {panel}: {len(params)} worlds, {wall:.1f} s, nefc max {int(out['nefc_max'].max())}, "
          f"ncon max {int(out['ncon'][..., 0].max())}, overflow {extra.get('overflow')}", flush=True)


if __name__ == "__main__":
  main(sys.argv[1:])
