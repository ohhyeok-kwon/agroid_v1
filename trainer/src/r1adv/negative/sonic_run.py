"""Negative 28: MuJoCo CPU runner for r1adv-scenario-v1 specs on the SONIC -> R1 line (R04).

Harness: T1 `sonic_g1.py` (SimR1 r1wb body, DeployR1Native, SONIC G1 planner) imported unmodified through P27
`qp_sonic.py` (planner-only patches from NEG28_PLANNER, default in06 + stride 0.45). Nothing of theirs is edited; the
d_env axes are applied to this process' own SimR1 instance after construction:
  body:  foot sliding mu + contact cone (pyramidal / elliptic + impratio), mass x group, torso COM shift, per-joint kp/kd
         scale (realised gain error), action delay (5 ms resolution)
  hit:   impulses J N s over one control tick at the torso, any time / direction / height offset
  sense: IMU gyro / orientation / joint q / dq noise (Gaussian, seeded; env.noise_seed if given) and sensing delay
         (5 ms resolution); env.onset_s starts action delay, sensing delay and noise at that time (spawn excluded)
  floor: 1-5 mm random height field (seeded), plane collision off
Policies: sonic:<ckpt> (planner loop; mission.ref = "static" uses T1's static R1 default-pose reference instead) | sonicsym:<ckpt> (same, test-time mirror-symmetrised actor, P29) |
          mf (manufacturer walk.onnx, evaluation only; twist command, deploy contract).
Physics: 200 Hz explicit PD (same loop as SimR1.step_ctrl), 50 Hz policy.

  python -I sonic_run.py --specs SPECS.jsonl --policy NAME=sonic:CKPT --policy mf=mf:ONNX --out DIR [--procs 32]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

T1_WORK = "/rl-agent/.agent/tmp/t1_sonic/work"
QP_DIR = "/rl-agent/.agent/tmp/p27_t2/qp"
DEFAULT_PLANNER = '{"inward":0.06,"min_sep":0.09,"skip_side":true,"stride_vref":0.45}'
os.environ.setdefault("NEG28_PLANNER", DEFAULT_PLANNER)
os.environ["P27_QP"] = os.environ["NEG28_PLANNER"]
os.environ.setdefault("R1_BODY", "r1wb")
os.environ.setdefault("SONIC_QUIET_PLANNER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
for p in ("/rl-agent/src", T1_WORK, QP_DIR):
  if p not in sys.path:
    sys.path.insert(0, p)

import qp_sonic as QP  # noqa: E402  (imports and patches sonic_g1 at import; planner keys only)

S = QP.S
DT, SUB = S.CTRL_DT, S.DECIM
FOOT_RE = re.compile(r"^(left|right)_foot[1-7]_collision$")
SOLE = np.array(S.R1_SOLE)
FALL_Z, FALL_TILT = 0.45, 1.0
_ROUGH = {"spec": None}

# ---------------------------------------------------------------- caching of stateless network objects per worker
_NP, _POL = {}, {}
_OrigNP, _OrigPol = S.NativeR1Policy, S.Policy


def _cached_np(ckpt):
  if ckpt not in _NP:
    _NP[ckpt] = _OrigNP(ckpt)
  return _NP[ckpt]


def _cached_pol(threads):
  if "p" not in _POL:
    _POL["p"] = _OrigPol(threads)
  return _POL["p"]


S.NativeR1Policy = _cached_np
S.Policy = _cached_pol

# ---------------------------------------------------------------- rough floor: r1wb get_spec wrapped (process-local)
import r1wb.robot as _R  # noqa: E402

_orig_get_spec = _R.get_spec


def _get_spec_rough(*a, **k):
  spec = _orig_get_spec(*a, **k)
  fl = _ROUGH["spec"]
  if fl and fl.get("rough_h_m", 0.0) > 0:
    import mujoco
    h, cell, half = float(fl["rough_h_m"]), float(fl.get("cell_m", 0.05)), float(fl.get("half_m", 12.0))
    n = int(round(2 * half / cell)) + 1
    rng = np.random.default_rng(int(fl["seed"]))
    steps = max(1, int(round(h / 0.001)))                 # 1 mm quantised levels in [0, h]
    data = rng.integers(0, steps + 1, size=n * n) / steps
    spec.add_hfield(name="neg28_rough", size=[half, half, h, 0.05], nrow=n, ncol=n, userdata=list(map(float, data)))
    spec.worldbody.add_geom(name="neg28_rough", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="neg28_rough")
  return spec


_R.get_spec = _get_spec_rough


def child_seed(parent: int, name: str) -> int:
  return int.from_bytes(hashlib.sha256(f"{int(parent)}|{name}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


def wrap(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- mission timeline
def timeline(mission):
  """[(t_begin, t_end, step)], total seconds; settle_s of standing first (as a Hold step)."""
  out, t = [], 0.0
  if mission.get("settle_s", 0) > 0:
    out.append((0.0, mission["settle_s"], dict(type="Hold", duration_s=mission["settle_s"], settle=True)))
    t = mission["settle_s"]
  for st in mission["steps"]:
    out.append((t, t + st["duration_s"], st))
    t += st["duration_s"]
  return out, t


def step_at(tl, t):
  for k, (b, e, st) in enumerate(tl):
    if b - 1e-9 <= t < e - 1e-9:
      return k, b, st
  return len(tl) - 1, tl[-1][0], tl[-1][2]


def twist(st):
  if st["type"] != "Go":
    return 0.0, 0.0, 0.0
  return float(st.get("vx", 0.0)), float(st.get("vy", 0.0)), float(st.get("wz", 0.0))


def heading_cmd(tl, t):
  """Commanded heading at t (turn steps advance it; planner facing is quantised to 10 Hz ticks like T1)."""
  psi = 0.0
  for b, e, st in tl:
    wz = twist(st)[2]
    if t >= e:
      psi += wz * (e - b)
    elif t >= b:
      psi += wz * (math.floor((t - b) / 0.1 + 1e-9) * 0.1)
      break
  return psi


def sonic_cmd_fn(tl):
  def cmd(t):
    k, b, st = step_at(tl, t)
    psi = heading_cmd(tl, t)
    face = (math.cos(psi), math.sin(psi), 0.0)
    vx, vy, wz = twist(st)
    sp = math.hypot(vx, vy)
    if st["type"] == "Go" and sp > 1e-6:
      mx, my = (math.cos(psi) * vx - math.sin(psi) * vy) / sp, (math.sin(psi) * vx + math.cos(psi) * vy) / sp
      return (S.SLOW_WALK, sp, (mx, my, 0.0), face, -1.0)
    return (S.IDLE, -1.0, (0.0, 0.0, 0.0), face, -1.0)
  return cmd


# ---------------------------------------------------------------- sensing (noise + delay) and actuation (delay)
class Sensor:
  def __init__(self, sim, cfg, seed):
    self.sim, self.cfg = sim, cfg or {}
    self.rng = np.random.default_rng(seed)
    self.active = True                                     # False before env.onset_s: newest state, no noise
    self.delay = int(round(float(self.cfg.get("delay_ms", 0.0)) / 5.0))
    self.buf = []
    self.record()

  def record(self):
    s, d = self.sim, self.sim.d
    self.buf.append((d.qpos[3:7].copy(), d.qvel[3:6].copy(), d.qpos[s.jq].copy(), d.qvel[s.jv].copy()))
    if len(self.buf) > self.delay + 1:
      self.buf.pop(0)

  def read(self):
    if not self.active:
      return tuple(x.copy() for x in self.buf[-1])
    quat, w, q, dq = (x.copy() for x in self.buf[0])
    c, r = self.cfg, self.rng
    if c.get("gyro_std", 0) > 0:
      w = w + r.normal(0.0, c["gyro_std"], 3)
    if c.get("ori_std", 0) > 0:
      v = r.normal(0.0, c["ori_std"], 3)
      ang = float(np.linalg.norm(v))
      if ang > 0:
        dqt = np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * v / ang])
        quat = S.qmul(quat, dqt)
    if c.get("q_std", 0) > 0:
      q = q + r.normal(0.0, c["q_std"], 24)
    if c.get("dq_std", 0) > 0:
      dq = dq + r.normal(0.0, c["dq_std"], 24)
    return quat, w, q, dq


class Actuator:
  def __init__(self, q0, delay_ms):
    self.d = int(round(float(delay_ms or 0.0) / 5.0))
    self.hist = [(-10 ** 9, q0.copy())]

  def issue(self, s, qt):
    self.hist.append((s, qt.copy()))
    self.hist = self.hist[-6:]

  def at(self, s):
    tgt = self.hist[0][1]
    for si, q in self.hist:
      if si + self.d <= s:
        tgt = q
    return tgt


# ---------------------------------------------------------------- policies
class NegDeploy(S.DeployR1Native):
  """DeployR1Native whose proprioception comes from Sensor (noise / delay); the planner side is unchanged."""
  obs = None

  def log_state(self):
    quat, w, q24, dq24 = self.obs
    self.hist.append(dict(q=q24[self.il_to_ent] - S.R1_DEF_IL, dq=dq24[self.il_to_ent].copy(), act=self.last_a24.copy(),
                          quat=quat, w=w))
    if len(self.hist) > 10:
      self.hist.pop(0)


class SymPolicy:
  """Test-time mirror-symmetrised R1 policy a = 1/2 [pi(s) + M_a pi(M_s s)] (positive 29, p29_qp.py native_call_sym;
  same algorithm re-implemented here, their file untouched). M_s: reference stream (10 frames of R1 joint positions /
  velocities, left-right swap and roll / yaw negated; anchor 6D as S R S, S = diag(1, -1, 1)) and proprioception (gyro
  (-wx, wy, -wz), joints / velocities / last actions like the joints, gravity (gx, -gy, gz)); M_a mirrors the 24 actions."""

  def __init__(self, base):
    self.base = base
    names = list(S.R1_IL)
    self.perm = np.array([names.index(n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for n in names])
    self.sgn = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in names])

  def __call__(self, enc_in, dec_rest):
    perm, sgn = self.perm, self.sgn
    e = enc_in.reshape(10, 54).copy()
    em = np.empty_like(e)
    em[:, 0:24] = sgn * e[:, 0:24][:, perm]
    em[:, 24:48] = sgn * e[:, 24:48][:, perm]
    r00, r01, r10, r11, r20, r21 = (e[:, 48 + i] for i in range(6))
    em[:, 48:54] = np.stack([r00, -r01, -r10, r11, r20, -r21], 1)
    r = dec_rest.copy()
    blocks = [(r[0:30].reshape(10, 3) * np.array([-1.0, 1.0, -1.0])).reshape(-1)]
    for off in (30, 270, 510):
      blocks.append((sgn * r[off:off + 240].reshape(10, 24)[:, perm]).reshape(-1))
    blocks.append((r[750:780].reshape(10, 3) * np.array([1.0, -1.0, 1.0])).reshape(-1))
    a, tok = self.base(enc_in, dec_rest)
    am, _ = self.base(em.reshape(-1), np.concatenate(blocks))
    return 0.5 * (a + sgn * am[perm]), tok


class Manufacturer:
  """walk.onnx (evaluation only): 10 frames oldest first of [gyro, gravity, twist, q - q0, dq, last action] (policy order)."""

  def __init__(self, onnx_path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
    self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    self.inp = self.sess.get_inputs()[0].name
    C = S.r1_contract()
    pol, ent = list(C.POLICY_JOINTS), list(C.ENTITY_JOINTS)
    self.p_from_e = np.array([ent.index(n) for n in pol])      # q_pol = q_ent[p_from_e]
    self.e_from_p = np.array([pol.index(n) for n in ent])      # x_ent = x_pol[e_from_p]
    self.default = np.array(C.DEFAULT_POS)
    self.scale = np.array(C.ACTION_SCALE)
    self.hist, self.prev = None, np.zeros(24)

  def __call__(self, obs, tw):
    quat, w, q, dq = obs
    grav = S.qrot(S.qconj(quat), (0.0, 0.0, -1.0))
    fr = np.concatenate([w, grav, tw, q[self.p_from_e] - self.default, dq[self.p_from_e], self.prev])
    self.hist = np.repeat(fr[None], 10, 0) if self.hist is None else np.concatenate([self.hist[1:], fr[None]], 0)
    a = self.sess.run(None, {self.inp: self.hist.reshape(1, 810).astype(np.float32)})[0][0].astype(np.float64)
    self.prev = a
    return (self.default + self.scale * a)[self.e_from_p]


# ---------------------------------------------------------------- body
def apply_body(sim, body, floor):
  import mujoco
  m = sim.m
  feet = [i for i in range(m.ngeom) if FOOT_RE.match(m.geom(i).name or "")]
  sim.foot_geoms = set(feet)
  sim.foot_side = {i: (0 if m.geom(i).name.startswith("left") else 1) for i in feet}
  if body.get("foot_mu") is not None and body.get("mu_switch_s") is None:
    m.geom_friction[feet, 0] = float(body["foot_mu"])
  if body.get("cone", "pyramidal") == "elliptic":
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  else:
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  m.opt.impratio = float(body.get("impratio", 1.0))
  changed = False
  for grp, sc in (body.get("mass_scale") or {}).items():
    for b in range(1, m.nbody):
      n = m.body(b).name
      g = "torso" if n in ("pelvis", "waist_roll_link", "torso_link") else (
        "legs" if any(x in n for x in ("hip", "knee", "ankle")) else "arms")
      if g == grp:
        m.body_mass[b] *= float(sc)
        m.body_inertia[b] *= float(sc)
        changed = True
  if body.get("torso_com"):
    m.body_ipos[m.body("torso_link").id] += np.array(body["torso_com"], float)
    changed = True
  if changed:
    mujoco.mj_setConst(m, sim.d)
    sim.mass = float(m.body_subtreemass[1])
  if body.get("kp_scale") is not None:
    sim.kp = sim.kp * np.array(body["kp_scale"], float)
  if body.get("kd_scale") is not None:
    sim.kd = sim.kd * np.array(body["kd_scale"], float)
  sim.floor_ids = {m.geom("floor").id}
  sim.rough_h = 0.0
  if floor and floor.get("rough_h_m", 0.0) > 0:
    g = m.geom("floor").id
    m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    sim.floor_ids = {m.geom("neg28_rough").id}
    sim.rough_h = float(floor["rough_h_m"])
    orig_reset = sim.reset

    def reset_rough():
      orig_reset()
      sim.d.qpos[2] += sim.rough_h + 0.001
      mujoco.mj_forward(sim.m, sim.d)
    sim.reset = reset_rough


def contacts(sim, f6):
  """Per-foot world contact force (2, 3) against the floor, and whether any non-foot geom touches the floor."""
  import mujoco
  m, d = sim.m, sim.d
  F = np.zeros((2, 3))
  other = False
  for i in range(d.ncon):
    c = d.contact[i]
    g1, g2 = c.geom1, c.geom2
    if g1 in sim.floor_ids:
      g = g2
    elif g2 in sim.floor_ids:
      g = g1
    else:
      continue
    if g in sim.foot_geoms:
      mujoco.mj_contactForce(m, d, i, f6)
      fr = c.frame.reshape(3, 3)
      fw = fr.T @ f6[:3]               # contact frame -> world; normal points geom1 -> geom2 (force on geom2)
      if g1 not in sim.floor_ids:
        fw = -fw
      F[sim.foot_side[g]] += fw
    else:
      other = True
  return F, other


def corners(sim):
  out = np.zeros((2, 4, 3))
  for s, b in enumerate(sim.feet):
    R = sim.d.xmat[b].reshape(3, 3)
    out[s] = sim.d.xpos[b][None] + SOLE @ R.T
  return out


# ---------------------------------------------------------------- one episode
def episode(job):
  import mujoco
  spec, pol = job["spec"], job["policy"]
  env = spec.get("env", {})
  body, floor, sense, hits = env.get("body", {}) or {}, env.get("floor") or {}, env.get("sense") or {}, env.get("hit") or []
  seed = int(spec["seed"])
  tl, T = timeline(spec["mission"])
  _ROUGH["spec"] = dict(floor, seed=child_seed(seed, "floor")) if floor.get("rough_h_m", 0) > 0 else None
  wall = time.time()
  if pol["kind"] in ("sonic", "sonicsym"):
    if spec["mission"].get("ref", "planner") == "static":          # T1 static reference (R1 default pose), no planner
      dep = NegDeploy("static", S.static_motion(), pol["path"], cmd=None, t_play=0.0)
    else:
      dep = NegDeploy("planner", None, pol["path"], cmd=sonic_cmd_fn(tl), t_play=0.0)
    dep.play_ok = True
    if pol["kind"] == "sonicsym":
      dep.npol = SymPolicy(dep.npol)
    sim = dep.sim
  else:
    sim = S.SimR1("r1")
    ctl = Manufacturer(pol["path"])
  apply_body(sim, body, floor)
  if pol["kind"] in ("sonic", "sonicsym"):
    dep.pre_roll()
  else:
    sim.reset()
  m, d = sim.m, sim.d
  sensor = Sensor(sim, sense, int(env["noise_seed"]) if env.get("noise_seed") is not None else child_seed(seed, "sense"))
  act = Actuator(sim.rec_default.copy(), body.get("action_delay_ms", 0.0))
  onset_k = int(round(float(env.get("onset_s", 0.0)) / DT))   # delay / sensing perturbations start here (spawn transient excluded)
  d_full = act.d
  if onset_k > 0:
    sensor.active, act.d = False, 0
  steps = int(round(T / DT))
  push_at = {}
  for h in hits:
    k = int(round(h["t_s"] / DT))
    push_at.setdefault(k, []).append(h)
  f6 = np.zeros(6)
  rec = {k: [] for k in ("pos", "quat", "torso_quat", "q", "sole", "force", "other", "cmd")}
  fell_k, end_k, s_idx = -1, steps, 0
  applied = []
  k_switch = int(round(float(body["mu_switch_s"]) / DT)) if body.get("mu_switch_s") is not None else -1
  for k in range(steps):
    t = k * DT
    if k == k_switch:
      m.geom_friction[sorted(sim.foot_geoms), 0] = float(body["foot_mu"])
    if k == onset_k and onset_k > 0:
      sensor.active, act.d = True, d_full
    obs = sensor.read()
    kk, b, st = step_at(tl, t)
    tw = np.array(twist(st))
    if pol["kind"] in ("sonic", "sonicsym"):
      dep.obs = obs
      sim.imu = (lambda o=obs: (o[0].copy(), o[1].copy()))
      dep.control_tick(k, t)
      qt = sim.target24()
    else:
      qt = ctl(obs, tw)
    act.issue(s_idx, qt)
    push = None
    if k in push_at:
      F_tot, Tq = np.zeros(3), np.zeros(3)
      yaw = S.yaw_of(d.qpos[3:7])
      for h in push_at[k]:
        ang = yaw + float(h["dir_rad"])
        F = np.array([math.cos(ang), math.sin(ang), 0.0]) * (float(h["J_Ns"]) / DT)
        F_tot += F
        Tq += np.cross([0.0, 0.0, float(h.get("z_off_m", 0.1))], F)
        applied.append(dict(t_s=t, J_Ns=h["J_Ns"], dir_rad=h["dir_rad"], z_off_m=h.get("z_off_m", 0.1)))
      push = np.concatenate([F_tot, Tq])
    for j in range(SUB):
      qtj = act.at(s_idx)
      q, dq = d.qpos[sim.jq], d.qvel[sim.jv]
      tau = np.clip(sim.kp * (qtj - q) - sim.kd * dq, -sim.eff, sim.eff)
      d.qfrc_applied[:] = 0.0
      d.qfrc_applied[sim.jv] = tau
      d.xfrc_applied[:] = 0.0
      if push is not None:
        d.xfrc_applied[sim.torso] = push
      mujoco.mj_step(m, d)
      s_idx += 1
      sensor.record()
    d.xfrc_applied[:] = 0.0
    F, other = contacts(sim, f6)
    rec["pos"].append(d.qpos[0:3].copy())
    rec["quat"].append(d.qpos[3:7].copy())
    rec["torso_quat"].append(d.xquat[sim.torso].copy())
    rec["q"].append(d.qpos[sim.jq].copy())
    rec["sole"].append(corners(sim))
    rec["force"].append(F)
    rec["other"].append(other)
    rec["cmd"].append(tw)
    g_b = S.qrot(S.qconj(d.qpos[3:7]), (0.0, 0.0, -1.0))
    tilt = math.acos(max(-1.0, min(1.0, -g_b[2])))
    if fell_k < 0 and (d.qpos[2] < FALL_Z or tilt > FALL_TILT or not np.isfinite(d.qpos[2])):
      fell_k = k
    if fell_k >= 0 and k >= fell_k + 25:
      end_k = k + 1
      break
  arr = {k: np.array(v, np.float32) for k, v in rec.items()}
  arr["other"] = np.array(rec["other"], bool)
  info = dict(spec_sha256=spec["spec_sha256"], policy=pol["name"], fell_k=fell_k, end_k=end_k, steps=steps,
              wall_s=round(time.time() - wall, 1), mass=float(sim.mass), pushes=applied, rough_h=sim.rough_h,
              planner_calls=(dep.planner.calls if pol["kind"] in ("sonic", "sonicsym") and dep.planner is not None else 0))
  return arr, info


# ---------------------------------------------------------------- measures (charter G / F / J + calm readouts)
JREF = {"hold": 20.0, "turn": 170.673, "go_0.05-0.25": 168.073, "go_0.25-0.45": 364.233, "go_0.45-0.80": 643.749,
        "go_0.80-3.00": 1062.681, "stop_transition": 379.685}          # R01_j_locked_v1_1.json J_ref (LOCKED v1.1)


def jtype(st):
  if st["type"] != "Go":
    return "hold"
  v = math.hypot(st.get("vx", 0.0), st.get("vy", 0.0))
  if v < 0.05:
    return "turn"
  for lo, hi in ((0.05, 0.25), (0.25, 0.45), (0.45, 0.8), (0.8, 3.0)):
    if lo <= v < hi:
      return f"go_{lo:.2f}-{hi:.2f}"
  return "go_0.80-3.00"


def yaw_series(quat):
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def roll_pitch(quat):
  out = np.zeros((len(quat), 2))
  for i, q in enumerate(quat):
    g = S.qrot(S.qconj(q.astype(np.float64)), (0.0, 0.0, -1.0))
    out[i] = math.atan2(-g[1], -g[2]), math.atan2(g[0], -g[2])
  return out


def measure(spec, arr, info):
  tl, T = timeline(spec["mission"])
  n = len(arr["pos"])
  fell_k = info["fell_k"]
  stop = fell_k if fell_k >= 0 else n
  pos = arr["pos"].astype(np.float64)
  yaw = yaw_series(arr["quat"].astype(np.float64))
  cmd = arr["cmd"].astype(np.float64)
  # commanded reference (global, from the start pose)
  # commanded reference, anchored at the robot pose at the settle end (rec index k = state after tick k)
  k0 = max(0, min(n - 1, int(round(tl[0][1] / DT)) - 1)) if tl[0][2].get("settle") else 0
  pr, ps, dist = np.zeros((n, 2)), np.zeros(n), np.zeros(n)
  pr[:k0 + 1], ps[:k0 + 1] = pos[k0, :2], yaw[k0]
  for k in range(k0 + 1, n):
    vx, vy, wz = cmd[k]
    c, s_ = math.cos(ps[k - 1]), math.sin(ps[k - 1])
    pr[k] = pr[k - 1] + DT * np.array([c * vx - s_ * vy, s_ * vx + c * vy])
    ps[k] = ps[k - 1] + DT * wz
    dist[k] = dist[k - 1] + DT * math.hypot(vx, vy)
  e_pos = np.linalg.norm(pos[:, :2] - pr, axis=1)
  e_yaw = np.abs(wrap(yaw - ps))
  went = np.cumsum([np.linalg.norm(c[:2]) > 1e-6 or abs(c[2]) > 1e-6 for c in cmd]) > 0
  tol_p = 0.10 + 0.10 * dist
  tol_y = np.where(went, 0.15, 0.10)
  g_t = np.maximum(e_pos / tol_p, e_yaw / tol_y)
  # jerk (24 joints, 50 Hz third difference, RMS over joints)
  q = arr["q"].astype(np.float64)
  jerk = np.full(n, np.nan)
  if n > 3:
    jerk[3:] = np.sqrt((((q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]) / DT ** 3) ** 2).mean(1))
  sole_z = arr["sole"][..., 2].min(2).astype(np.float64)        # (n, 2)
  thr = 0.010 + info.get("rough_h", 0.0)
  lifted = sole_z > thr
  rp = roll_pitch(arr["torso_quat"].astype(np.float64))
  steps_out, g_ok, j_ok = [], True, True
  prev_type = None
  for i, (b, e, st) in enumerate(tl):
    kb, ke = int(round(b / DT)), int(round(e / DT))
    if kb >= stop:
      steps_out.append(dict(i=i, type=st["type"], scored=False))
      g_ok = j_ok = False
      continue
    ke_eff = min(ke, stop)
    r = dict(i=i, type=st["type"], settle=bool(st.get("settle", False)), t_b=b, t_e=e, cmd=list(twist(st)),
             complete_window=ke_eff == ke)
    kend = ke_eff - 1
    r["e_pos_end"], r["e_yaw_end"] = float(e_pos[kend]), float(e_yaw[kend])
    r["G_ratio_end"] = float(g_t[kend])
    r["G_int"] = float(np.mean(g_t[kb:ke_eff]))
    jt = jtype(st)
    after_go = st["type"] == "Hold" and prev_type == "Go"
    if after_go:
      k1 = min(kb + 50, ke_eff)
      jj1 = jerk[kb + 3:k1]
      jj2 = jerk[k1 + 3:ke_eff] if ke_eff - k1 > 6 else np.array([])
      r["J_stop"] = float(np.nanmedian(jj1)) if len(jj1) else None
      r["J"] = float(np.nanmedian(jj2)) if len(jj2) else None
      okj = (r["J_stop"] is None or r["J_stop"] <= JREF["stop_transition"]) and (r["J"] is None or r["J"] <= JREF["hold"])
    else:
      jj = jerk[kb + 3:ke_eff]
      r["J"] = float(np.nanmedian(jj)) if len(jj) else None
      okj = r["J"] is None or r["J"] <= JREF[jt]
    r["J_type"], r["J_ref"], r["J_ok"] = jt, JREF[jt], bool(okj)
    # lift-off events (rising edges held >= 3 ticks)
    ev = 0
    for f in range(2):
      L = lifted[kb:ke_eff, f]
      run = 0
      for x in L:
        run = run + 1 if x else 0
        if run == 3:
          ev += 1
    r["liftoffs"] = ev
    r["roll_rms_deg"] = float(np.degrees(np.sqrt(np.mean(rp[kb:ke_eff, 0] ** 2))))
    r["pitch_rms_deg"] = float(np.degrees(np.sqrt(np.mean(rp[kb:ke_eff, 1] ** 2))))
    vx, vy, wz = twist(st)
    sp = math.hypot(vx, vy)
    if st["type"] == "Go" and sp > 1e-6 and ke_eff - kb > 75:
      ka = kb + 50
      u = np.array([math.cos(ps[kb]) * vx - math.sin(ps[kb]) * vy, math.sin(ps[kb]) * vx + math.cos(ps[kb]) * vy]) / sp
      perp = np.array([-u[1], u[0]])
      tt = np.arange(ka, ke_eff) * DT
      along = pos[ka:ke_eff, :2] @ u
      lat = pos[ka:ke_eff, :2] @ perp
      r["speed_ratio"] = float(np.polyfit(tt, along, 1)[0] / sp)
      res = lat - np.polyval(np.polyfit(tt, lat, 1), tt)
      r["lat_pp_m"] = float(res.max() - res.min())
    if not st.get("settle"):
      g_ok &= r["G_ratio_end"] <= 1.0 and r["complete_window"]
      j_ok &= r["J_ok"]
    steps_out.append(r)
    prev_type = st["type"]
  # stance slip and utilised friction (feet with sole within 2 mm of the floor top)
  F = arr["force"].astype(np.float64)
  W = info["mass"] * 9.81
  fn, ft = F[..., 2], np.linalg.norm(F[..., :2], axis=2)
  stance = fn > 0.2 * W
  stance[:k0 + 1] = False
  stance[stop:] = False
  util = ft[stance] / np.maximum(fn[stance], 1e-6)
  sole = arr["sole"].astype(np.float64)
  vel = np.gradient(sole[..., :2], DT, axis=0) if n > 2 else np.zeros_like(sole[..., :2])
  sp_c = np.linalg.norm(vel, axis=3)
  low = sole[..., 2].argmin(2)
  spl = np.take_along_axis(sp_c, low[..., None], 2)[..., 0]
  st_mask = sole_z < (0.002 + info.get("rough_h", 0.0))
  slips = spl[k0 + 1:stop][st_mask[k0 + 1:stop]]
  # path ratio (smoothed pelvis path / commanded distance)
  kk = max(1, int(round(0.5 / DT)))
  if stop > kk + 2:
    sm = np.stack([np.convolve(pos[:stop, i], np.ones(kk) / kk, mode="valid") for i in range(2)], 1)
    plen = float(np.linalg.norm(np.diff(sm, axis=0), axis=1).sum())
  else:
    plen = 0.0
  dcmd = float(dist[min(stop, n) - 1]) if stop > 0 else 0.0
  scored = [r for r in steps_out if "t_e" in r and not r.get("settle")]
  hold_only = all(st["type"] == "Hold" for _, _, st in tl)
  out = dict(
    spec_sha256=spec["spec_sha256"], family=spec.get("family"), policy=info["policy"], seed=spec["seed"],
    F=bool(fell_k >= 0), t_fall=(fell_k * DT if fell_k >= 0 else None), nonfoot_contact_s=float(arr["other"][:stop].sum() * DT),
    G_ok=bool(g_ok and fell_k < 0), J_ok=bool(j_ok and fell_k < 0),
    complete=bool(fell_k < 0 and g_ok and j_ok),
    G_int=float(np.mean(g_t[k0 + 1:stop])) if stop > k0 + 1 else None,
    G_ratio_max=max([r["G_ratio_end"] for r in scored if "G_ratio_end" in r], default=None),
    J_ratio_max=max([(r["J"] or 0.0) / r["J_ref"] for r in scored if "J_ref" in r], default=None),
    path_len_m=plen, cmd_dist_m=dcmd, path_ratio=(plen / dcmd if dcmd > 0.2 else None),
    util_p50=float(np.percentile(util, 50)) if util.size else None, util_p95=float(np.percentile(util, 95)) if util.size else None,
    util_max=float(util.max()) if util.size else None,
    slip_p50=float(np.percentile(slips, 50)) if slips.size else None, slip_p95=float(np.percentile(slips, 95)) if slips.size else None,
    swing_p90_cm=float(np.percentile(sole_z[:stop].max(1), 90) * 100) if stop > 0 else None,
    hold_only=hold_only,
    hold_liftoffs_per_min=(sum(r.get("liftoffs", 0) for r in scored if r["type"] == "Hold") /
                           max(1e-9, sum(r["t_e"] - r["t_b"] for r in scored if r["type"] == "Hold")) * 60.0)
    if any(r["type"] == "Hold" and "t_e" in r for r in scored) else None,
    hold_drift_m=(float(np.linalg.norm(pos[stop - 1, :2] - pos[k0, :2])) if hold_only else None),
    lat_pp_m=(float(np.mean([r["lat_pp_m"] for r in scored if "lat_pp_m" in r])) if any("lat_pp_m" in r for r in scored) else None),
    speed_ratio=(float(np.mean([r["speed_ratio"] for r in scored if "speed_ratio" in r])) if any("speed_ratio" in r for r in scored) else None),
    roll_rms_go_deg=(float(np.sqrt(np.mean([r["roll_rms_deg"] ** 2 for r in scored if r["type"] == "Go"]))) if any(r["type"] == "Go" and "roll_rms_deg" in r for r in scored) else None),
    roll_rms_hold_deg=(float(np.sqrt(np.mean([r["roll_rms_deg"] ** 2 for r in scored if r["type"] == "Hold" and "roll_rms_deg" in r]))) if any(r["type"] == "Hold" and "roll_rms_deg" in r for r in scored) else None),
    pushes=info["pushes"], wall_s=info["wall_s"], steps=steps_out)
  return out


def work(job):
  import traceback
  try:
    arr, info = episode(job)
    row = measure(job["spec"], arr, info)
    if job.get("save"):
      p = Path(job["save"])
      p.parent.mkdir(parents=True, exist_ok=True)
      np.savez_compressed(p, meta=json.dumps(info), **arr)
    return row
  except Exception as exc:                                    # one failed episode must not stop the batch
    return dict(spec_sha256=job["spec"].get("spec_sha256"), policy=job["policy"]["name"], error=repr(exc),
                trace=traceback.format_exc()[-2000:])


def sha16(p):
  return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--specs", required=True)
  ap.add_argument("--policy", action="append", required=True, help="NAME=sonic:CKPT | NAME=mf:ONNX")
  ap.add_argument("--out", required=True)
  ap.add_argument("--procs", type=int, default=32)
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--no-save", action="store_true")
  ap.add_argument("--order", choices=("spec", "instance"), default="spec",
                  help="instance: run level sets instance-major (all levels' i = 0 first) so a cut run still covers every level")
  a = ap.parse_args()
  specs = [json.loads(l) for l in Path(a.specs).read_text().splitlines() if l.strip()]
  if a.limit:
    specs = specs[:a.limit]
  pols = []
  for p in a.policy:
    name, rest = p.split("=", 1)
    kind, path = rest.split(":", 1)
    pols.append(dict(name=name, kind=kind, path=path, sha16=sha16(path)))
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  done = set()
  rows_path = out / "rows.jsonl"
  if rows_path.exists():
    for l in rows_path.read_text().splitlines():
      try:
        r = json.loads(l)
        if "error" not in r:
          done.add((r["spec_sha256"], r["policy"]))
      except Exception:
        pass
  jobs = []
  # interleave policies so partial results cover all policies
  for sp in specs:
    for pol in pols:
      if (sp["spec_sha256"], pol["name"]) in done:
        continue
      jobs.append(dict(spec=sp, policy=pol, save=None if a.no_save else str(out / "npz" / pol["name"] / f"{sp['spec_sha256'][:16]}.npz")))
  if a.order == "instance":
    jobs.sort(key=lambda j: (j["spec"].get("level", {}).get("i", 0), j["spec"].get("family", ""), j["spec"].get("level", {}).get("x", 0)))
  meta = dict(argv=sys.argv, planner=os.environ["NEG28_PLANNER"], policies=pols, n_specs=len(specs), n_jobs=len(jobs),
              runner_sha16=sha16(__file__), sonic_g1_sha16=sha16(Path(T1_WORK) / "sonic_g1.py"),
              qp_sonic_sha16=sha16(Path(QP_DIR) / "qp_sonic.py"), started=time.strftime("%Y-%m-%d %H:%M:%S"))
  (out / f"meta_{int(time.time())}.json").write_text(json.dumps(meta, indent=1))
  t0 = time.time()
  import multiprocessing as mp
  with open(rows_path, "a") as fo, mp.get_context("spawn").Pool(max(1, min(a.procs, len(jobs) or 1))) as pool:
    for i, r in enumerate(pool.imap_unordered(work, jobs)):
      fo.write(json.dumps(r) + "\n")
      fo.flush()
      if (i + 1) % 10 == 0 or i + 1 == len(jobs):
        print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
  print("done", flush=True)


if __name__ == "__main__":
  main()
