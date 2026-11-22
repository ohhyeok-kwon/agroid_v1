"""Contact ladder without learning (positive 18, R11). Rules: .agent/adversarial/rounds/R04/positive/r11_objects/rules.md
(sha256 9b788f10...). Fixed-base R1 (O6 box hands), mink IK joint targets tracked by the deploy PD at 50 Hz, one free
object squeezed between the hands: squeeze -> lift 0.2 m -> hold 10 s -> carry (waist-yaw arc 0.3 m) -> lower -> release.

  python -m r1adv.positive_obj.ladder cpu <ik.npz> <out_dir> <seed> [panel ...] [--workers N] [--limit N]
  python -m r1adv.positive_obj.ladder mjw <ik.npz> <out_dir> <seed> [panel ...] [--limit N]
Panels: py3 py4 el3 el4 (cone x condim, object worlds of strata A, C, B) and empty (twins of A and C, no object).
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
from r1adv.positive_obj import robot_fixed as RF
from r1adv.positive_obj import scene as S

DT, DECIM = 0.005, 4
NSTEP, NCTRL = 4800, 1200
PHASES = dict(P0=(0.0, 1.0), P1=(1.0, 2.5), P2=(2.5, 3.5), P3=(3.5, 5.5), P4=(5.5, 15.5), P5=(15.5, 18.5),
              P6=(18.5, 20.5), P7=(20.5, 22.0), P8=(22.0, 24.0))
X_G, Z_G, LIFT = IK.X_G, IK.Z_G, IK.LIFT
YAW_AXIS_X = IK.YAW_AXIS_X
RHO = X_G - YAW_AXIS_X
PSI_PICK, PSI_PLACE = -IK.carry_yaw(X_G) / 2, IK.carry_yaw(X_G) / 2
PANELS = {"py3": ("pyramidal", 3), "py4": ("pyramidal", 4), "el3": ("elliptic", 3), "el4": ("elliptic", 4),
          "empty": ("pyramidal", 3)}
# rules_amend A2 (secondary diagnostic, not used by D1-D3): the same panels with the stock wrist capsule
# (*_hand_collision, r 0.03, 1 cm proud of the O6 box face) excluded from contact with the objects.
PANELS.update({f"{k}nc": v for k, v in list(PANELS.items()) if k != "empty"})
N_A, N_C, N_B = 1024, 512, 512
PARAM_NAMES = ("stratum", "shape", "dx", "dy", "dz", "mass", "mu", "tc", "impratio", "d_in", "s_pre", "s_sq", "z_top",
               "table_x", "twin_of", "ik_ok")
P = {n: i for i, n in enumerate(PARAM_NAMES)}
ARM_JOINTS = tuple(f"{s}_{j}_joint" for s in ("left", "right")
                   for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"))
SIM_OPT = dict(timestep=DT, iterations=10, ls_iterations=20, ccd_iterations=50, tolerance=1e-8, ls_tolerance=0.01)


def minjerk(tau: np.ndarray) -> np.ndarray:
  tau = np.clip(tau, 0.0, 1.0)
  return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def script(t: np.ndarray, s_pre: float, s_sq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Hand half-separation s, lift dz and waist yaw psi at times t (rules 2)."""
  a = lambda name: (t - PHASES[name][0]) / (PHASES[name][1] - PHASES[name][0])
  s = s_pre + (s_sq - s_pre) * minjerk(a("P1")) + (s_pre - s_sq) * minjerk(a("P7"))
  dz = LIFT * minjerk(a("P3")) - LIFT * minjerk(a("P6"))
  psi = PSI_PICK + (PSI_PLACE - PSI_PICK) * minjerk(a("P5"))
  return s, dz, psi


class IKGrid:
  def __init__(self, path: str):
    z = np.load(path)
    self.Q, self.s, self.dz = z["Q"], z["s_grid"], z["u_grid"]
    # rules_amend A1: PD targets carry the static gravity feed-forward G[s, dz, psi] (robot weight) on top of the IK pose
    self.G, self.psi = z["G"], z["psi_grid"]
    self.names = [str(n) for n in z["joint_names"]]
    self.yaw_i, self.roll_i = self.names.index("waist_yaw_joint"), self.names.index("waist_roll_joint")
    self.names = [str(n) for n in z["joint_names"]]
    self.res = z["pos_err"].max(-1).max(-1)                 # per s: worst hand residual over the lift path
    self.low = -z["hand_low"][:, 0]                         # per s: depth of the lowest O6 box corner at dz = 0
    assert abs(float(z["x_g"]) - X_G) < 1e-9 and abs(float(z["z_g"]) - Z_G) < 1e-9

  def residual(self, s):
    return np.interp(s, self.s, self.res)

  def box_low(self, s):
    return np.interp(s, self.s, self.low)

  def joints(self, s: np.ndarray, dz: np.ndarray, psi: np.ndarray, which: str = "T") -> np.ndarray:
    """PD target (which="T": IK pose + gravity feed-forward) or IK pose ("Q") at (s, dz, waist yaw psi), (len(s), nq).
    The IK pose is bilinear in (s, dz) with waist yaw psi and waist roll 0; the feed-forward is trilinear in (s, dz, psi).
    s, dz, psi are clipped to the grid."""
    s = np.clip(s, self.s[0], self.s[-1])
    dz = np.clip(dz, self.dz[0], self.dz[-1])
    psi = np.clip(psi, self.psi[0], self.psi[-1])
    fi = (s - self.s[0]) / (self.s[1] - self.s[0])
    fj = (dz - self.dz[0]) / (self.dz[1] - self.dz[0])
    fk = (psi - self.psi[0]) / (self.psi[1] - self.psi[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, len(self.s) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(self.dz) - 2)
    k0 = np.clip(np.floor(fk).astype(int), 0, len(self.psi) - 2)
    a, b, c = (fi - i0)[:, None], (fj - j0)[:, None], (fk - k0)[:, None]
    Q = self.Q
    q = ((1 - a) * (1 - b) * Q[i0, j0] + a * (1 - b) * Q[i0 + 1, j0] + (1 - a) * b * Q[i0, j0 + 1]
         + a * b * Q[i0 + 1, j0 + 1])
    q[:, self.yaw_i] = psi
    q[:, self.roll_i] = 0.0
    if which == "T":
      G = self.G
      g = 0.0
      for di, wi in ((0, 1 - a), (1, a)):
        for dj, wj in ((0, 1 - b), (1, b)):
          for dk, wk in ((0, 1 - c), (1, c)):
            g = g + wi * wj * wk * G[i0 + di, j0 + dj, k0 + dk]
      q = q + g
    return q


def sample_worlds(seed: int, grid: IKGrid, n_a: int = N_A, n_c: int = N_C, n_b: int = N_B) -> np.ndarray:
  """World parameters (rules 3). Order: A, C, B; empty twins are built by twins()."""
  rng = np.random.Generator(np.random.PCG64(seed))
  rows = []
  for stratum, n in ((0, n_a), (1, n_c), (2, n_b)):
    for _ in range(n):
      u = rng.uniform(*S.SIZE_RANGE, size=3)
      shape = int(rng.integers(3))
      L = rng.uniform(u[1], S.SIZE_RANGE[1])
      dims = {0: (u[0], u[1], u[2]), 1: (u[1], u[1], u[2]), 2: (L, u[1], u[1])}[shape]
      mass = rng.uniform(0.2, 0.5) if stratum == 1 else rng.uniform(*S.MASS_RANGE)
      half = rng.random() < 0.25
      mu_u = rng.uniform(0.7, 1.0) if stratum == 1 else rng.uniform(*S.FRICTION_RANGE)
      mu = 0.5 if (stratum != 1 and half) else mu_u
      tc = rng.uniform(*S.SOLREF_TC_RANGE)
      imp = 10.0 if rng.random() < 0.5 else 1.0
      d_in = {0: rng.uniform(0.01, 0.03), 1: rng.uniform(0.02, 0.03), 2: rng.uniform(0.03, 0.10)}[stratum]
      w, h = dims[1], dims[2]
      s_pre = min(w / 2 + 0.08, 0.27)
      s_sq = w / 2 - d_in + 0.02
      ik_ok = float(grid.residual(s_sq) <= 0.005)
      b = max(grid.box_low(s_pre), grid.box_low(s_sq))
      z_top = Z_G - max(h / 2, b + 0.005)
      cx = [YAW_AXIS_X + RHO * math.cos(p) for p in (PSI_PICK, PSI_PLACE)]
      xmin = min(S.footprint_xmin(S.SHAPES[shape], dims, c, p) for c, p in zip(cx, (PSI_PICK, PSI_PLACE)))
      rows.append([stratum, shape, *dims, mass, mu, tc, imp, d_in, s_pre, s_sq, z_top, xmin - 0.02, -1, ik_ok])
  return np.array(rows, dtype=np.float64)


def twins(params: np.ndarray) -> np.ndarray:
  keep = np.nonzero(params[:, P["stratum"]] <= 1)[0]
  tw = params[keep].copy()
  tw[:, P["stratum"]] = 3
  tw[:, P["twin_of"]] = keep
  return tw


def panel_params(seed: int, grid: IKGrid, panel: str, limit: int | None = None) -> np.ndarray:
  params = sample_worlds(seed, grid)
  if panel == "empty":                                     # twins of strata A and C, no object
    params = twins(params)
  return params if limit is None else params[:limit]


def object_pose(row: np.ndarray, psi: float = PSI_PICK) -> tuple[np.ndarray, np.ndarray]:
  h = row[P["dz"]]
  pos = np.array([YAW_AXIS_X + RHO * math.cos(psi), RHO * math.sin(psi), row[P["z_top"]] + h / 2 + 0.001])
  return pos, S.yaw_quat(psi)


def build_model(cone: str, condim: int, free_robot: bool = False, shapes=S.SHAPES, table: bool = True,
                nocap: bool = False):
  """The mjlab scene compiled on CPU with the training options (rules 1). nocap: rules_amend A2 diagnostic."""
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim import MujocoCfg
  from mjlab.terrains import TerrainEntityCfg
  from r1lab import robot as RB
  robot = RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="o6") if free_robot else RF.fixed_robot_cfg()
  ents = {"robot": robot}
  if table:
    ents["table"] = S.table_entity_cfg()
  for shape in shapes:
    ents[S.ENTITY_OF[shape]] = S.object_entity_cfg(shape, condim)
  def _nocap(spec):
    for side in ("left", "right"):
      for shape in shapes:
        spec.add_exclude(bodyname1=f"robot/{side}_wrist_roll_link", bodyname2=f"{S.ENTITY_OF[shape]}/body")
  scene = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"), entities=ents,
                         spec_fn=_nocap if nocap else None), device="cpu")
  m = scene.compile()
  MujocoCfg(cone=cone, **SIM_OPT).apply(m)
  return m


class Layout:
  """Actuator / joint / qpos addresses of the ladder model."""

  def __init__(self, m: mujoco.MjModel, grid: IKGrid):
    self.nu = m.nu
    self.act_joint = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])) for a in range(m.nu)]
    names = [n.removeprefix("robot/") for n in self.act_joint]
    self.act_from_ik = np.array([grid.names.index(n) for n in names])
    self.act_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    self.waist_yaw_act = names.index("waist_yaw_joint")
    self.waist_roll_act = names.index("waist_roll_joint")
    self.arm_act = np.array([names.index(n) for n in ARM_JOINTS])
    self.effort = np.abs(m.actuator_forcerange[:, 1]).copy()
    self.waist_yaw_qadr = int(m.jnt_qposadr[m.joint("robot/waist_yaw_joint").id])


def world_targets(row: np.ndarray, grid: IKGrid, lay: Layout, which: str = "T") -> np.ndarray:
  """(NCTRL, nu) PD targets at the control instants t_k = k * 0.02 s (which="Q": the IK pose itself)."""
  t = np.arange(NCTRL) * DT * DECIM
  s, dz, psi = script(t, row[P["s_pre"]], row[P["s_sq"]])
  return grid.joints(s, dz, psi, which)[:, lay.act_from_ik]


def _set_object_fields_c(m, idx, row, condim):
  active = S.SHAPES[int(row[P["shape"]])] if row[P["stratum"]] != 3 else None
  for shape in idx.obj_body:
    if shape == active:
      S.write_world_fields_c(m, idx, shape, row[[P["dx"], P["dy"], P["dz"]]], row[P["mass"]], condim, row[P["mu"]],
                             row[P["tc"]])
    else:
      S.write_world_fields_c(m, idx, shape, S.NOMINAL_DIMS, 1.0, condim, 0.7, 0.02)
  m.opt.impratio = row[P["impratio"]]
  return active


def init_state_c(m, d, idx, lay, row, tgt0, active, q0=None):
  mujoco.mj_resetData(m, d)
  d.qpos[lay.act_qadr] = tgt0 if q0 is None else q0
  for shape, qa in idx.obj_qadr.items():
    if shape == active:
      pos, quat = object_pose(row)
    else:
      px, py = S.PARK_POS[shape]
      pos, quat = np.array([px, py, S.NOMINAL_DIMS[2] / 2 + 0.002]), np.array([1.0, 0, 0, 0])
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, quat
  if idx.table_mocap >= 0:
    d.mocap_pos[idx.table_mocap] = [row[P["table_x"]], 0.0, row[P["z_top"]]]
    d.mocap_quat[idx.table_mocap] = [1.0, 0.0, 0.0, 0.0]
  d.ctrl[:] = tgt0
  mujoco.mj_forward(m, d)


def alloc(n: int) -> dict:
  return dict(obj_v=np.zeros((n, NSTEP, 3), np.float32), hand_v=np.zeros((n, NSTEP, 2, 3), np.float32),
              f_hand=np.zeros((n, NSTEP, 2, 3), np.float32), dist_min=np.ones((n, NSTEP, 2), np.float32),
              ncon=np.zeros((n, NSTEP, 8), np.int16), ncap=np.zeros((n, NSTEP, 2), np.int16),
              obj_p=np.zeros((n, NCTRL, 3), np.float32), obj_q=np.zeros((n, NCTRL, 4), np.float32),
              hand_p=np.zeros((n, NCTRL, 2, 3), np.float32), hand_q=np.zeros((n, NCTRL, 2, 4), np.float32),
              tau_ratio=np.zeros((n, NCTRL, 10), np.float32), waist_yaw=np.zeros((n, NCTRL), np.float32),
              nefc_max=np.zeros(n, np.int32))


def run_worlds_c(panel: str, rows: np.ndarray, ik_path: str) -> dict:
  cone, condim = PANELS[panel]
  grid = IKGrid(ik_path)
  m = build_model(cone, condim, nocap=panel.endswith("nc"))
  idx = S.SceneIndex(m)
  lay = Layout(m, grid)
  d = mujoco.MjData(m)
  out = alloc(len(rows))
  res6 = np.zeros(6)
  for w, row in enumerate(rows):
    active = _set_object_fields_c(m, idx, row, condim)
    tgt = world_targets(row, grid, lay)
    init_state_c(m, d, idx, lay, row, tgt[0], active, q0=world_targets(row, grid, lay, "Q")[0])
    ob = idx.obj_body.get(active) if active else None
    oq = idx.obj_qadr.get(active) if active else None
    od = idx.obj_dadr.get(active) if active else None
    ag = idx.obj_geom.get(active, -1) if active else -1
    tmax = np.zeros(10)
    for k in range(NSTEP):
      if k % DECIM == 0:
        d.ctrl[:] = tgt[k // DECIM]
        tmax[:] = 0.0
      mujoco.mj_step(m, d)
      if ob is not None:
        out["obj_v"][w, k] = d.qvel[od:od + 3]
      for h, hb in enumerate(idx.hand_bodies):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, hb, res6, 0)
        out["hand_v"][w, k, h] = res6[3:]
      counts, fh, dmin, ncap = S.classify_contacts_c(m, d, idx, ag)
      out["ncon"][w, k], out["f_hand"][w, k], out["dist_min"][w, k], out["ncap"][w, k] = counts, fh, dmin, ncap
      out["nefc_max"][w] = max(out["nefc_max"][w], d.nefc)
      tmax = np.maximum(tmax, np.abs(d.actuator_force[lay.arm_act]) / lay.effort[lay.arm_act])
      if k % DECIM == DECIM - 1:
        c = k // DECIM
        if ob is not None:
          out["obj_p"][w, c], out["obj_q"][w, c] = d.qpos[oq:oq + 3], d.qpos[oq + 3:oq + 7]
        for h, hb in enumerate(idx.hand_bodies):
          out["hand_p"][w, c, h], out["hand_q"][w, c, h] = d.xpos[hb], d.xquat[hb]
        out["tau_ratio"][w, c] = tmax
        out["waist_yaw"][w, c] = d.qpos[lay.waist_yaw_qadr]
  return out


def _worker(args):
  panel, rows, ik_path = args
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  return run_worlds_c(panel, rows, ik_path)


def run_panel_cpu(panel: str, params: np.ndarray, ik_path: str, workers: int, chunk: int = 16) -> tuple[dict, float]:
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  jobs = [(panel, params[i:i + chunk], ik_path) for i in range(0, len(params), chunk)]
  parts = []
  with ProcessPoolExecutor(max_workers=workers) as ex:
    for part in ex.map(_worker, jobs):
      parts.append(part)
  out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
  return out, time.time() - t0


def save(path: str, out: dict, params: np.ndarray, meta: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez(path, params=params, param_names=np.array(PARAM_NAMES), meta=json.dumps(meta), **out)


def main(argv: list[str]) -> None:
  engine, ik_path, out_dir, seed = argv[0], argv[1], argv[2], int(argv[3])
  rest = argv[4:]
  workers, limit = 32, None
  if "--workers" in rest:
    i = rest.index("--workers"); workers = int(rest[i + 1]); del rest[i:i + 2]
  if "--limit" in rest:
    i = rest.index("--limit"); limit = int(rest[i + 1]); del rest[i:i + 2]
  panels = rest or list(PANELS)
  grid = IKGrid(ik_path)
  for panel in panels:
    params = panel_params(seed, grid, panel, limit)
    cone, condim = PANELS[panel]
    if engine == "cpu":
      out, wall = run_panel_cpu(panel, params, ik_path, workers)
    else:
      from r1adv.positive_obj import ladder_mjw
      out, wall, extra = ladder_mjw.run_panel(panel, params, ik_path)
    meta = dict(engine=engine, panel=panel, cone=cone, condim=condim, seed=seed, n=len(params), wall_s=wall,
                mujoco=mujoco.__version__, opt=SIM_OPT, rules_sha256="9b788f103b556de1b49ab6d5adb15648c814e15978fb74aaa219d71be3213c98",
                note="obj_* = state after the physics step; hand_*, contacts, forces = kinematics/contacts of the step start "
                     "(5 ms earlier), identical in both engines")
    if engine != "cpu":
      meta.update(extra)
    save(os.path.join(out_dir, engine, f"s{seed}", f"{panel}.npz"), out, params, meta)
    print(f"{engine} seed {seed} {panel}: {len(params)} worlds, {wall:.1f} s, nefc max {int(out['nefc_max'].max())}, "
          f"ncon max {int(out['ncon'][..., 0].max())}", flush=True)


if __name__ == "__main__":
  main(sys.argv[1:])
