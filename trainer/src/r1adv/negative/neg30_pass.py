"""Negative 30: pass-table runner (R04 SONIC -> R1 line). Measures every pass criterion in absolute terms on the same
loop as deployment, from r1adv-scenario-v1 specs (neg30_specs.py). MuJoCo CPU only.

Loop (all imported unmodified; nothing of theirs is edited):
  planner   T1 sonic_g1 SONIC G1 planner + P27 qp_sonic 'cand' post-process (env.loop.planner -> P27_QP, read at import:
            one planner setting per process; the runner refuses specs whose planner differs from the process setting)
  bridge    P29 p29_qp_m1: B2 quiet_ik + M1 (each planner buffer -> P27 M1 core -> R1 joints; walk commands / 0.86)
            + LatBridge (env.loop.bridge_lat_s: a new buffer's conversion is used L seconds after it appears)
  replan    env.loop.ctx_robot: true = closed loop (P29 ctx_robot context read from the robot state; recorded here in our
            own Deploy subclass because neg28's sensor log_state hides P29's log_state patch), false = open loop (T1
            planner_tick: replans from its own reference = P29 row m1c)
  sensing   uniform noise with the D2 training bounds (gyro, orientation rotation vector, q, dq), from env.onset_s
  actuation neg28 Actuator: action delay env.body.action_delay_ms (5 ms resolution), from env.onset_s
  hits      neg28 push model (one control tick at the torso, z offset)
Policies: NAME=sonic:CKPT (R1-native SONIC actor) | NAME=mf:ONNX (manufacturer walk.onnx, evaluation only)

  python -I neg30_pass.py --specs S.jsonl --policy NAME=sonic:CKPT ... --out DIR --procs 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

CAND = {"inward": 0.06, "min_sep": 0.09, "skip_side": True, "stride_vref": 0.45, "start_frac": 0.8}
PLANNER = json.loads(os.environ.get("NEG30_PLANNER", json.dumps(CAND)))
os.environ["NEG28_PLANNER"] = json.dumps(PLANNER)
os.environ["P27_QP"] = os.environ["NEG28_PLANNER"]
os.environ["P29_QP"] = json.dumps({"quiet_ik": True, "m1": True, "m1_sleg": 0.86, "ctx_robot": True})
os.environ.setdefault("R1_BODY", "r1wb")
os.environ.setdefault("SONIC_QUIET_PLANNER", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
for _p in ("/rl-agent/src", "/rl-agent/.agent/tmp/t1_sonic/work", "/rl-agent/.agent/tmp/p27_t2/qp", "/rl-agent/.agent/tmp/p29"):
  if _p not in sys.path:
    sys.path.insert(0, _p)

import p29_qp_m1 as PQ  # noqa: E402  (imports qp_sonic -> sonic_g1; patches control_tick (M1), ctx_robot, B2)
from r1adv.negative import sonic_run as SR  # noqa: E402  (neg28: Actuator, Manufacturer, apply_body, contacts, ...)

S = PQ.S
DT, SUB = S.CTRL_DT, S.DECIM
NEG30 = Path("/rl-agent/.agent/tmp/neg30")


class M1Bridge30(PQ.M1Bridge):
  """P29 M1Bridge with its scratch folder in neg30 (P29's __init__ would create p29/m1work/<pid>)."""

  def __init__(self):
    import time as _t
    self.t = _t
    sys.path.insert(0, "/rl-agent/.agent/data/r1_caption_pilot/tools")
    import ad_robot as AR
    import core as C
    self.C, self.AR, self.RT = C, AR, C.RT
    self.g1m = AR.g1_model()
    self.work = NEG30 / "m1work" / str(os.getpid())
    self.work.mkdir(parents=True, exist_ok=True)
    self.cache_mo, self.Q = None, None
    self.n, self.stats = 0, []


PQ.M1Bridge = M1Bridge30


class Sensor30(SR.Sensor):
  """neg28 Sensor (delay buffer) with uniform noise at the D2 training bounds: cfg dist='uniform', gyro, ori, q, dq."""

  def read(self):
    if self.cfg.get("dist") != "uniform":
      return super().read()
    if not self.active:
      return tuple(x.copy() for x in self.buf[-1])
    quat, w, q, dq = (x.copy() for x in self.buf[0])
    c, r = self.cfg, self.rng
    if c.get("gyro", 0) > 0:
      w = w + r.uniform(-c["gyro"], c["gyro"], 3)
    if c.get("ori", 0) > 0:
      v = r.uniform(-c["ori"], c["ori"], 3)
      ang = float(np.linalg.norm(v))
      if ang > 0:
        quat = S.qmul(quat, np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * v / ang]))
    if c.get("q", 0) > 0:
      q = q + r.uniform(-c["q"], c["q"], 24)
    if c.get("dq", 0) > 0:
      dq = dq + r.uniform(-c["dq"], c["dq"], 24)
    return quat, w, q, dq


class Dep30(S.DeployR1Native):
  """DeployR1Native for the loop above: proprioception from Sensor30 (noise / delay); with loop_ctx the planner context
  history is recorded from the robot state exactly as P29 log_state_p29 does (copy of its body, without its call to
  T1's log_state, which would push a noise-free entry into the policy history), and every replan is counted."""
  obs = None
  loop_ctx = False

  hist_fill = "zero"                                         # 'first': start history = 10 copies of the first observation

  def log_state(self):
    quat, w, q24, dq24 = self.obs
    e = dict(q=q24[self.il_to_ent] - S.R1_DEF_IL, dq=dq24[self.il_to_ent].copy(), act=self.last_a24.copy(), quat=quat, w=w)
    if not self.hist and self.hist_fill == "first":           # training fills the history with the first observation (P33)
      self.hist.extend([dict(e) for _ in range(9)])
    self.hist.append(e)
    if len(self.hist) > 10:
      self.hist.pop(0)
    if self.loop_ctx and self.planner is not None:
      self._ctx_record()

  def _ctx_record(self):                                       # == p29_qp_m1.log_state_p29 after `_orig_log(self)`
    s = self.sim
    q24 = s.d.qpos[s.jq].copy()
    q_mj = S.DEFAULT.copy()
    q_mj[s.g1_of] = q24
    if not hasattr(self, "_p29_ctx_guess"):
      self._p29_ctx_guess = {sd: S.DEFAULT[PQ.HIP_MJ[sd]].copy() for sd in ("left", "right")}
      self._p29_hist = []
      self._p29_kz = PQ.R1Kin(s.m).kz if not hasattr(self, "_p29_kin") else self._p29_kin.kz
    for sd in ("left", "right"):
      r1h = q24[[s.rec_names.index(f"{sd}_{h}") for h in S.HipMap.J3]]
      g = self.hm.solve("r1", "g1", sd, r1h, self._p29_ctx_guess[sd])
      self._p29_ctx_guess[sd] = g
      q_mj[PQ.HIP_MJ[sd]] = g
    if not hasattr(self, "_p29_r1z0"):
      self._p29_r1z0 = float(s.d.qpos[2])
    pos = s.d.qpos[0:3].copy()
    pos[2] = S.DEFAULT_HEIGHT + (pos[2] - self._p29_r1z0) / self._p29_kz
    self._p29_hist.append((pos, S.heading_q(s.d.qpos[3:7].copy()), q_mj))
    if len(self._p29_hist) > 20:
      self._p29_hist.pop(0)

  def planner_tick(self, t):
    c0 = self.planner.calls
    robot_ctx = self.loop_ctx and hasattr(self, "_p29_hist") and len(self._p29_hist) >= 6
    if self.loop_ctx:
      PQ.planner_tick_p29(self, t)
    else:
      PQ._orig_pt(self, t)
    if self.planner.calls > c0:
      if robot_ctx:
        self.n_replan_robot = getattr(self, "n_replan_robot", 0) + 1
      else:
        self.n_replan_self = getattr(self, "n_replan_self", 0) + 1


# ---------------------------------------------------------------- one episode
def episode(job):
  import mujoco
  spec, pol = job["spec"], job["policy"]
  env = spec.get("env", {})
  loop = env.get("loop") or {}
  body, sense, hits = env.get("body", {}) or {}, env.get("sense") or {}, env.get("hit") or []
  seed = int(spec["seed"])
  tl, T = SR.timeline(spec["mission"])
  SR._ROUGH["spec"] = None
  wall = time.time()
  sonic = pol["kind"] == "sonic"
  if sonic:
    if loop.get("planner") is not None and loop["planner"] != PLANNER:
      raise RuntimeError(f"spec planner {loop['planner']} != process planner {PLANNER}")
    PQ.M1_LAT = float(loop.get("bridge_lat_s", 0.0))           # read by control_tick_p29 when the bridge is built
    if spec["mission"].get("ref", "planner") == "static":
      dep = Dep30("static", S.static_motion(), pol["path"], cmd=None, t_play=0.0)
    else:
      dep = Dep30("planner", None, pol["path"], cmd=SR.sonic_cmd_fn(tl), t_play=0.0)
      dep.loop_ctx = bool(loop.get("ctx_robot", False))
    dep.hist_fill = str(loop.get("hist_fill", "zero"))
    dep.play_ok = True
    import torch
    torch.set_num_threads(1)
    sim = dep.sim
  else:
    sim = S.SimR1("r1")
    ctl = SR.Manufacturer(pol["path"])
  SR.apply_body(sim, body, {})
  if sonic:
    dep.pre_roll()
  else:
    sim.reset()
  m, d = sim.m, sim.d
  sensor = Sensor30(sim, sense, int(env["noise_seed"]) if env.get("noise_seed") is not None else SR.child_seed(seed, "sense"))
  act = SR.Actuator(sim.rec_default.copy(), body.get("action_delay_ms", 0.0))
  onset_k = int(round(float(env.get("onset_s", 0.0)) / DT))
  d_full = act.d
  if onset_k > 0:
    sensor.active, act.d = False, 0
  steps = int(round(T / DT))
  push_at = {}
  for h in hits:
    push_at.setdefault(int(round(h["t_s"] / DT)), []).append(h)
  f6 = np.zeros(6)
  keys = ("pos", "quat", "v", "torso_quat", "q", "dq", "qt", "sole", "force", "other", "cmd", "ref_rpos")
  rec = {k: [] for k in keys}
  fell_k, end_k, s_idx = -1, steps, 0
  applied = []
  for k in range(steps):
    t = k * DT
    if k == onset_k and onset_k > 0:
      sensor.active, act.d = True, d_full
    obs = sensor.read()
    _kk, _b, st = SR.step_at(tl, t)
    tw = np.array(SR.twist(st))
    if sonic:
      mo_obs, cur_obs = dep.motion, dep.cur
      dep.obs = obs
      sim.imu = (lambda o=obs: (o[0].copy(), o[1].copy()))
      dep.control_tick(k, t)
      qt = sim.target24()
      rec["ref_rpos"].append(np.asarray(mo_obs.rpos[min(cur_obs, mo_obs.T - 1)], float).copy())
    else:
      qt = ctl(obs, tw)
      rec["ref_rpos"].append(np.zeros(3))
    act.issue(s_idx, qt)
    rec["qt"].append(np.asarray(qt, float).copy())            # issued PD target (entity order), before the action delay
    push = None
    if k in push_at:
      F_tot, Tq = np.zeros(3), np.zeros(3)
      yaw = S.yaw_of(d.qpos[3:7])
      for h in push_at[k]:
        ang = yaw + float(h["dir_rad"])
        F = np.array([math.cos(ang), math.sin(ang), 0.0]) * (float(h["J_Ns"]) / DT)
        F_tot += F
        Tq += np.cross([0.0, 0.0, float(h.get("z_off_m", 0.1))], F)
        applied.append(dict(t_s=t, k=k, J_Ns=h["J_Ns"], dir_rad=h["dir_rad"], z_off_m=h.get("z_off_m", 0.1)))
      push = np.concatenate([F_tot, Tq])
    for _j in range(SUB):
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
    F, other = SR.contacts(sim, f6)
    rec["pos"].append(d.qpos[0:3].copy())
    rec["quat"].append(d.qpos[3:7].copy())
    rec["v"].append(d.qvel[0:3].copy())
    rec["torso_quat"].append(d.xquat[sim.torso].copy())
    rec["q"].append(d.qpos[sim.jq].copy())
    rec["dq"].append(d.qvel[sim.jv].copy())
    rec["sole"].append(SR.corners(sim))
    rec["force"].append(F)
    rec["other"].append(other)
    rec["cmd"].append(tw)
    g_b = S.qrot(S.qconj(d.qpos[3:7]), (0.0, 0.0, -1.0))
    tilt = math.acos(max(-1.0, min(1.0, -g_b[2])))
    if fell_k < 0 and (d.qpos[2] < SR.FALL_Z or tilt > SR.FALL_TILT or not np.isfinite(d.qpos[2])):
      fell_k = k
    if fell_k >= 0 and k >= fell_k + 25:
      end_k = k + 1
      break
  arr = {k: np.array(v, np.float32) for k, v in rec.items()}
  arr["other"] = np.array(rec["other"], bool)
  info = dict(spec_sha256=spec["spec_sha256"], policy=pol["name"], fell_k=fell_k, end_k=end_k, steps=steps,
              wall_s=round(time.time() - wall, 1), mass=float(sim.mass), pushes=applied, rough_h=0.0,
              names=list(sim.rec_names))
  if sonic:
    br = getattr(dep, "_p29_m1", None)
    inner = getattr(br, "inner", br)
    info.update(planner_calls=dep.planner.calls if dep.planner is not None else 0,
                replan_robot=getattr(dep, "n_replan_robot", 0), replan_self=getattr(dep, "n_replan_self", 0),
                loop_ctx=bool(dep.loop_ctx), bridge_lat_ticks=getattr(br, "L", 0), bridge_late=getattr(br, "n_late", 0),
                m1_buffers=getattr(inner, "n", 0),
                m1_wall_med=float(np.median([s_[1] for s_ in inner.stats])) if inner is not None and getattr(inner, "stats", None) else None)
  return arr, info


# ---------------------------------------------------------------- measures (P24 readout definitions)
UP, DOWN, AIR = 0.015, 0.005, 0.010


def landing_idx(z):
  idx, up = [], z[0] > UP
  for k in range(1, len(z)):
    if up and z[k] < DOWN:
      idx.append(k)
      up = False
    elif not up and z[k] > UP:
      up = True
  return idx


def smooth(x, n=25):
  return np.convolve(x, np.ones(n) / n, mode="same")


def yaw_of(q):
  return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def roll_pitch_p24(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))


def fnan(x):
  return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else float(x)


def step_response(t, sig, cmd, t0):
  """P24: 0.5 s centred moving average; t50 / t90 after t0; overshoot over [t0, t0 + 8]; steady = raw mean [t0+3, t0+8]."""
  s = smooth(sig)
  after = t >= t0 - 1e-9
  win = after & (t <= t0 + 8.0 + 1e-9)

  def first(level):
    k = np.nonzero(after & (s >= level * cmd))[0]
    return float(t[k[0]] - t0) if len(k) else float("nan")
  steady = (t >= t0 + 3.0 - 1e-9) & (t <= t0 + 8.0 + 1e-9)
  return dict(t50_s=fnan(first(0.5)), t90_s=fnan(first(0.9)),
              overshoot_pct=fnan(max(0.0, float((s[win].max() / cmd - 1.0) * 100.0))) if win.any() else None,
              steady_ratio=fnan(float(sig[steady].mean() / cmd)) if steady.sum() > 10 else None)


def time_to_still(t, v, ts, thr=0.10):
  sp = np.linalg.norm(v[:, :2], axis=1)
  for k in np.nonzero(t >= ts - 1e-9)[0]:
    if k + 25 <= len(t) and (sp[k:k + 25] < thr).all():
      return float(t[k] - ts)
  return None


def walk_look(t, W, pos, quat, tquat, q, dq, sole_z, names):
  """P24 walking appearance inside window mask W."""
  sp_idx = [names.index("left_shoulder_pitch_joint"), names.index("right_shoulder_pitch_joint")]
  arm = [i for i, nm in enumerate(names) if any(k_ in nm for k_ in ("shoulder", "elbow", "wrist"))]
  qq, dd, p, tt = q[W], dq[W], pos[W], t[W]
  yw = yaw_of(quat[W])
  mh = math.atan2(np.sin(yw).mean(), np.cos(yw).mean())
  rel = p[:, :2] - p[0, :2]
  lateral = -rel[:, 0] * math.sin(mh) + rel[:, 1] * math.cos(mh)
  lat_d = lateral - np.polyval(np.polyfit(tt, lateral, 1), tt)
  z_d = p[:, 2] - np.polyval(np.polyfit(tt, p[:, 2], 1), tt)
  roll, pitch = roll_pitch_p24(tquat[W])
  so = sole_z[W]
  li = [landing_idx(so[:, f]) for f in range(2)]
  apex = [float(so[a:b, f].max()) for f in range(2) for a, b in zip(li[f][:-1], li[f][1:])]
  return dict(arm_swing_rad=float(np.mean([qq[:, j].max() - qq[:, j].min() for j in sp_idx])),
              arm_qd_rms=float(np.sqrt((dd[:, arm] ** 2).mean())),
              pelvis_lat_pp_m=float(lat_d.max() - lat_d.min()), pelvis_lat_rms_m=float(lat_d.std()),
              pelvis_vert_pp_m=float(z_d.max() - z_d.min()),
              roll_rms_deg=float(np.degrees((roll - roll.mean()).std())), pitch_rms_deg=float(np.degrees((pitch - pitch.mean()).std())),
              step_hz=float((len(li[0]) + len(li[1])) / max(tt[-1] - tt[0], 1e-9)),
              foot_lift_m=float(np.median(apex)) if apex else None,
              two_feet_off_share=float(((so[:, 0] > AIR) & (so[:, 1] > AIR)).mean()))


def jerk_series(q):
  j = np.full(len(q), np.nan)
  if len(q) > 3:
    j[3:] = np.sqrt((((q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]) / DT ** 3) ** 2).mean(1))
  return j


def stance_slip(sole, k_lo, k_hi):
  """neg28 definition (lowest sole corner speed while that foot's sole is within 2 mm of the floor), p95 in [k_lo, k_hi)."""
  if k_hi - k_lo < 5:
    return None
  sz = sole[..., 2].min(2)
  vel = np.gradient(sole[..., :2], DT, axis=0)
  sp_c = np.linalg.norm(vel, axis=3)
  low = sole[..., 2].argmin(2)
  spl = np.take_along_axis(sp_c, low[..., None], 2)[..., 0]
  m_ = sz < 0.002
  x = spl[k_lo:k_hi][m_[k_lo:k_hi]]
  return float(np.percentile(x, 95)) if x.size else None


def hops(force, sole, k_lo):
  """Two-foot hop count (P29 buridan_probe rule): both feet leave (Fz < 5 N for >= 2 ticks) and land within 0.04 s
  (2 ticks) of each other and both move > 3 cm; counted over every flight pair from k_lo on."""
  fz = force[:, :, 2]
  n = len(fz)
  foot_xy = sole[..., :2].mean(2)
  iv = []
  for f in range(2):
    off = fz[:, f] < 5.0
    spans, k = [], k_lo
    while k < n:
      if off[k]:
        j = k
        while j < n and off[j]:
          j += 1
        if j - k >= 2:
          spans.append((k, j))
        k = j
      else:
        k += 1
    iv.append(spans)
  cnt = 0
  for a, b in iv[0]:
    for c, e in iv[1]:
      if abs(a - c) <= 2 and abs(b - e) <= 2 and b < n and e < n:
        d0 = np.linalg.norm(foot_xy[min(b, n - 1), 0] - foot_xy[max(a - 1, 0), 0])
        d1 = np.linalg.norm(foot_xy[min(e, n - 1), 1] - foot_xy[max(c - 1, 0), 1])
        if d0 > 0.03 and d1 > 0.03:
          cnt += 1
  return cnt


def measure(spec, arr, info):
  fam = spec.get("family")
  mis = spec["mission"]
  names = info["names"]
  n = len(arr["pos"])
  t = (np.arange(n) + 1) * DT
  pos, quat, v = arr["pos"].astype(float), arr["quat"].astype(float), arr["v"].astype(float)
  tquat, q, dq = arr["torso_quat"].astype(float), arr["q"].astype(float), arr["dq"].astype(float)
  sole = arr["sole"].astype(float)
  sole_z = sole[..., 2].min(2)
  force = arr["force"].astype(float)
  fell_k = info["fell_k"]
  up = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
  tilt = np.degrees(np.arccos(np.clip(up, -1, 1)))
  fell24 = bool(((pos[:, 2] < 0.45) | (tilt > 60.0)).any())   # P24 fall rule over the recorded episode
  onset_k = int(round(float(spec["env"].get("onset_s", 0.0)) / DT))
  jer = jerk_series(q)
  out = dict(spec_sha256=spec["spec_sha256"], family=fam, cell=spec.get("cell"), evalset=spec.get("evalset"),
             i=spec.get("i"), policy=info["policy"], F=bool(fell_k >= 0 or fell24),
             t_fall=(fell_k * DT if fell_k >= 0 else None), nonfoot_contact_s=float(arr["other"].sum() * DT),
             hops=hops(force, sole, onset_k), wall_s=info["wall_s"],
             **{k_: info.get(k_) for k_ in ("planner_calls", "replan_robot", "replan_self", "loop_ctx", "bridge_lat_ticks",
                                            "bridge_late", "m1_buffers", "m1_wall_med")})
  if fam == "walk_stop":
    t0, ts = float(mis["settle_s"]), float(mis["settle_s"]) + float(mis["steps"][0]["duration_s"])
    vx = float(mis["steps"][0]["vx"])
    yaw = yaw_of(quat)
    fwd = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw)
    out.update(step_response(t, fwd, vx, t0))
    out["stop_s"] = time_to_still(t, v, ts) if n * DT >= ts + 0.6 else None
    out["stop2_s"] = time_to_still(t, v, ts, 0.03) if n * DT >= ts + 0.6 else None   # pelvis below 0.03 m/s (coordinator 17:40)
    rr = arr["ref_rpos"].astype(float)
    rv = np.gradient(rr[:, :2], DT, axis=0)
    rfwd = rv[:, 0] * np.cos(yaw) + rv[:, 1] * np.sin(yaw)
    stdy = (t >= t0 + 3 - 1e-9) & (t <= t0 + 8 + 1e-9)
    out["ref_g1_steady_ratio"] = float(rfwd[stdy].mean() / vx) if stdy.sum() > 10 and np.any(rr) else None
    W = (t >= t0 + 3.0 - 1e-9) & (t <= min(t0 + 13.0, ts) + 1e-9)    # walking look window, within the Go step
    if W.sum() > 100 and fell_k < 0:
      out.update(walk_look(t, W, pos, quat, tquat, q, dq, sole_z, names))
      out["realised_fwd_mps"] = float(fwd[W].mean())
      out["jerk_walk_med"] = float(np.nanmedian(jer[W]))
      kW = np.nonzero(W)[0]
      out["slip_p95_walk"] = stance_slip(sole, int(kW[0]), int(kW[-1]) + 1)
    out["G_walk"] = bool(fell_k < 0 and out.get("steady_ratio") is not None and out["steady_ratio"] >= 0.7)
  elif fam == "stand_push":
    hit = spec["env"]["hit"][0]
    tp = round(float(hit["t_s"]) / DT) * DT
    W = (t >= 2.0 - 1e-9) & (t <= tp + 1e-9)
    so = sole_z[W]
    p = pos[W]
    yw = yaw_of(quat[W])
    out.update(stand_steps=len(landing_idx(so[:, 0])) + len(landing_idx(so[:, 1])),
               stand_drift_m=float(np.linalg.norm(p[-1, :2] - p[0, :2])) if len(p) else None,
               stand_drift_max_m=float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max()) if len(p) else None,
               stand_heading_deg=float(abs(np.degrees(SR.wrap(yw[-1] - yw[0])))) if len(p) else None,
               stand_jerk_med=float(np.nanmedian(jer[W])) if W.any() else None,
               stand_two_feet_off=float(((so[:, 0] > AIR) & (so[:, 1] > AIR)).mean()) if len(so) else None)
    after = t > tp + 1e-9
    fell_p = bool(((pos[after, 2] < 0.45) | (tilt[after] > 60.0)).any()) or (fell_k >= 0 and fell_k * DT >= tp - 1e-9)
    rec_s = None
    calm = np.linalg.norm(v[:, :2], axis=1) < 0.10
    for k in np.nonzero(after)[0]:
      if k + 25 <= n and calm[k:k + 25].all():
        rec_s = float(t[k] - tp)
        break
    w6 = after & (t <= tp + 6.0 + 1e-9)
    w1 = after & (t <= tp + 1.0 + 1e-9)
    out.update(push_J=float(hit["J_Ns"]), push_dir=hit.get("dir_name"), push_fell=fell_p, push_recovery_s=rec_s,
               push_steps=(len(landing_idx(sole_z[w6, 0])) + len(landing_idx(sole_z[w6, 1]))) if w6.sum() > 2 else None,
               push_max_qd=float(np.abs(dq[w1]).max()) if w1.any() else None,
               push_hops=hops(force, sole, int(round(tp / DT))))
  elif fam in ("moves", "hunt"):
    tl, _T = SR.timeline(mis)
    if fam == "hunt":
      hits_ = spec["env"].get("hit") or []
      vv = np.linalg.norm(v[:, :2], axis=1)
      out["still_end"] = bool(fell_k < 0 and n >= 50 and (vv[-50:] < 0.10).all())
      if all(st_["type"] == "Hold" for _b, _e, st_ in tl):
        out["hold_drift_m"] = None if hits_ else (float(np.linalg.norm(pos[-1, :2] - pos[int(1.0 / DT), :2])) if fell_k < 0 else None)
        k2 = int(2.0 / DT)
        out["stand_steps"] = len(landing_idx(sole_z[k2:, 0])) + len(landing_idx(sole_z[k2:, 1]))
        out["stand_jerk_med"] = float(np.nanmedian(jer[k2:])) if n > k2 + 10 else None
    segs = []
    yaw = yaw_of(quat)
    for b, e, st in tl:
      if st["type"] != "Go":
        continue
      kb, ke = int(round(b / DT)), min(int(round(e / DT)), n) - 1
      vx_, vy_, wz_ = (float(st.get(x_, 0.0)) for x_ in ("vx", "vy", "wz"))
      r = dict(label=st.get("label"), cmd=[vx_, vy_, wz_], complete=bool(ke >= int(round(e / DT)) - 1 and (fell_k < 0 or fell_k > ke)))
      if ke - kb > 10:
        sp_ = math.hypot(vx_, vy_)
        dur = (ke - kb) * DT
        if sp_ > 1e-6:
          c, s_ = math.cos(yaw[kb]), math.sin(yaw[kb])
          u = np.array([c * vx_ - s_ * vy_, s_ * vx_ + c * vy_]) / sp_
          disp = pos[ke, :2] - pos[kb, :2]
          r.update(dist_m=float(disp @ u), realised=float(disp @ u / (sp_ * dur)))
        else:
          dy = float(np.unwrap(yaw[kb:ke + 1])[-1] - np.unwrap(yaw[kb:ke + 1])[0])
          r.update(turn_rad=dy, realised=float(dy / (wz_ * dur)) if abs(wz_) > 1e-9 else None)
        so = sole_z[kb:ke + 1]
        r.update(swing_share=float(((so[:, 0] > AIR) | (so[:, 1] > AIR)).mean()),
                 landings=len(landing_idx(so[:, 0])) + len(landing_idx(so[:, 1])))
      segs.append(r)
    out["segments"] = segs
  elif fam == "stand_static":
    k3 = int(round(3.0 / DT))
    if n > k3 + 10:
      def roll_g(qq):
        w_, x_, y_, z_ = qq[:, 0], qq[:, 1], qq[:, 2], qq[:, 3]
        gy, gz = -(2 * (y_ * z_ + w_ * x_)), -(w_ * w_ - x_ * x_ - y_ * y_ + z_ * z_)
        return np.degrees(np.arctan2(-gy, -gz))
      qm = np.degrees(q[k3:].mean(0))
      ix = {nm: i for i, nm in enumerate(names)}
      out.update(sym_pelvis_roll=float(roll_g(quat[k3:]).mean()), sym_torso_roll=float(roll_g(tquat[k3:]).mean()),
                 sym_waist_roll=float(qm[ix["waist_roll_joint"]]), sym_waist_yaw=float(qm[ix["waist_yaw_joint"]]),
                 sym_hip_yaw_LR=float(qm[ix["left_hip_yaw_joint"]] + qm[ix["right_hip_yaw_joint"]]),
                 sym_hip_roll_LR=float(qm[ix["left_hip_roll_joint"]] + qm[ix["right_hip_roll_joint"]]),
                 stand_static_drift_m=float(np.linalg.norm(pos[-1, :2] - pos[k3, :2])),
                 stand_static_steps=len(landing_idx(sole_z[k3:, 0])) + len(landing_idx(sole_z[k3:, 1])))
  elif fam == "cmd_t1":                                          # positive control: T1 cmd robot 0 timing
    t0 = float(mis["settle_s"])
    vx = float(mis["steps"][0]["vx"])
    yaw = yaw_of(quat)
    fwd = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw)
    out.update(step_response(t, fwd, vx, t0))
    rr = arr["ref_rpos"].astype(float)
    rv = np.gradient(rr[:, :2], DT, axis=0)
    rfwd = rv[:, 0] * np.cos(yaw) + rv[:, 1] * np.sin(yaw)
    stdy = (t >= t0 + 3 - 1e-9) & (t <= t0 + 8 + 1e-9)
    out["ref_g1_steady_ratio"] = float(rfwd[stdy].mean() / vx) if np.any(rr) else None
    W = (t >= 4.0 - 1e-9) & (t <= 14.0 + 1e-9)
    if fell_k < 0:
      out.update(walk_look(t, W, pos, quat, tquat, q, dq, sole_z, names))
    if len(mis["steps"]) > 1:
      out["stop_s"] = time_to_still(t, v, t0 + float(mis["steps"][0]["duration_s"]))
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
                trace=traceback.format_exc()[-2500:])


def sha16(p):
  return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--specs", required=True)
  ap.add_argument("--policy", action="append", required=True, help="NAME=sonic:CKPT | NAME=mf:ONNX")
  ap.add_argument("--out", required=True)
  ap.add_argument("--procs", type=int, default=16)
  ap.add_argument("--only", default="", help="comma list of families to run")
  ap.add_argument("--mf-cells", default="", help="comma list of cells on which mf runs (default: all)")
  ap.add_argument("--no-save", action="store_true")
  a = ap.parse_args()
  specs = [json.loads(l) for l in Path(a.specs).read_text().splitlines() if l.strip()]
  if a.only:
    fams = set(a.only.split(","))
    specs = [s for s in specs if s.get("family") in fams]
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
  mf_cells = set(a.mf_cells.split(",")) if a.mf_cells else None
  jobs = []
  for sp in specs:
    for pol in pols:
      if (sp["spec_sha256"], pol["name"]) in done:
        continue
      if pol["kind"] == "mf" and (sp["mission"].get("ref") == "static" or (mf_cells is not None and sp.get("cell") not in mf_cells)):
        continue
      jobs.append(dict(spec=sp, policy=pol, save=None if a.no_save else str(out / "npz" / pol["name"] / f"{sp['spec_sha256'][:16]}.npz")))
  jobs.sort(key=lambda j: (j["spec"].get("i", 0), j["spec"].get("family", ""), j["spec"].get("cell", "")))
  meta = dict(argv=sys.argv, planner=PLANNER, policies=pols, n_specs=len(specs), n_jobs=len(jobs), runner_sha16=sha16(__file__),
              sonic_run_sha16=sha16(SR.__file__), p29_qp_m1_sha16=sha16(PQ.__file__),
              sonic_g1_sha16=sha16(Path(S.__file__)), qp_sonic_sha16=sha16(Path(PQ.QP.__file__)),
              started=time.strftime("%Y-%m-%d %H:%M:%S"))
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
