"""Negative 30 (coordinator 11:2x): pass-table loop A with P29's R1 planner RP1 in the planner slot (posture board S1-S5,
W1-W5 and the walking / command / push items), same specs and measures as neg30_pass.py (imported unmodified).

Deploy of RP1 here (P29 README 25: 'replan at 10 Hz, gen = cur + 2 blend, the tracker reads R1 frames directly, context =
its own output (open loop)'); RP1 model code imported from P29 (rp1_train.py, read only), checkpoint copied by sha:
  - every 5 control ticks (10 Hz) RP1 replans: context = 4 frames at 30 Hz spacing ending at gen = cur + 2 of the current
    R1 reference (clamped at the buffer start), canonical frame of its last frame (rp1_train.canon / features);
  - command (rp1_kin_panel convention): speed = commanded |v| (no body-size scaling: RP1 is trained in R1 metres),
    direction = commanded world direction - reference heading, facing = commanded heading at the horizon end
    (heading + wz x 64 / 30 s) - reference heading, height = context root z, mode one-hot (idle < 0.08, walk < 1.0, run);
  - output 64 frames (30 Hz, after the last context frame) -> world -> 50 Hz with frame 0 = the reference at gen; blended
    into the current buffer from gen over 8 frames (T1 blend, written for 24 joints); no post-process (no quiet / no rule);
  - before the first replan the reference is the RP1 idle plan from the standing reset state (R1 default pose).
No bridge (RP1 emits R1 joints), so env.loop.bridge_lat_s is ignored; action delay and sensing as in neg30_pass.
Policy: NAME=rp1:TRACKER_CKPT@RP1_CKPT.

  python -I neg30_rp1.py --specs S.jsonl --policy NAME=rp1:TRACKER@RP1 --out DIR --procs 15
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_pass as P  # noqa: E402  (loop setup: env, p29_qp_m1, sonic_run, measures)

sys.path.insert(0, "/rl-agent/.agent/tmp/p29/rp1")
S, PQ, SR = P.S, P.PQ, P.SR
DT = S.CTRL_DT
DT30 = 1.0 / 30.0
_RT = {}
_CK = {}


def rt():
  if "m" not in _RT:
    import rp1_train as RT                                     # P29 code, imported unmodified
    _RT["m"] = RT
  return _RT["m"]


def yaw_q(q):
  return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def mat_to_quat(R):
  t = R[0, 0] + R[1, 1] + R[2, 2]
  if t > 0:
    s = math.sqrt(t + 1.0) * 2
    q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
  elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
    q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
  elif R[1, 1] > R[2, 2]:
    s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
    q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
  else:
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
  q = np.array(q)
  return q / np.linalg.norm(q)


class RP1Planner:
  """RP1 inference on CPU (one thread). Keeps call count / wall time like T1's Planner."""

  def __init__(self, ckpt):
    import torch
    RT = rt()
    if ckpt not in _CK:
      _CK[ckpt] = torch.load(ckpt, map_location="cpu", weights_only=False)   # cached per worker process
    ck = _CK[ckpt]
    a = ck["args"]
    self.model = RT.RP1(a["dim"], heads=max(4, a["dim"] // 32))
    self.model.load_state_dict(ck["model"])
    self.model.eval()
    self.jn = list(ck["joint_names"])
    self.to_il = np.array([self.jn.index(n) for n in S.R1_IL])  # RP1 order -> tracker IsaacLab order
    self.from_il = np.array([list(S.R1_IL).index(n) for n in self.jn])
    self.torch, self.RT = torch, RT
    self.calls, self.wall = 0, 0.0

  def plan(self, ctx_world, v, ang_world, face_world):
    """ctx_world (4, 31) R1 qpos (root pos, wxyz, joints in RP1 order) at 30 Hz, oldest first -> (64, 31) world frames."""
    torch, RT = self.torch, self.RT
    t0 = time.time()
    ctx = torch.tensor(ctx_world[None], dtype=torch.float32)
    pad = torch.cat([ctx, ctx[:, -1:].expand(-1, RT.HOR, -1)], 1)
    p, q, j = RT.canon(pad)
    feat = RT.features(p[:, :RT.CTX], q[:, :RT.CTX], j[:, :RT.CTX])
    ref = ctx_world[-1]
    yaw = yaw_q(ref[3:7])
    ang, face = ang_world - yaw, face_world - yaw
    mode = [float(v < 0.08), float(0.08 <= v < 1.0), float(v >= 1.0)]
    cmd = torch.tensor([[v, math.cos(ang), math.sin(ang), math.cos(face), math.sin(face), float(p[0, RT.CTX - 1, 2])] + mode],
                       dtype=torch.float32)
    with torch.no_grad():
      pred = self.model(feat, cmd)[0]
    R = RT.six_to_mat(pred[:, 3:9]).numpy()
    pr = pred.numpy()
    c, s_ = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    out = np.zeros((RT.HOR, 31))
    for i in range(RT.HOR):
      out[i, 0:3] = [ref[0] + c * pr[i, 0] - s_ * pr[i, 1], ref[1] + s_ * pr[i, 0] + c * pr[i, 1], pr[i, 2]]
      out[i, 3:7] = mat_to_quat(Rz @ R[i])
      out[i, 7:] = pr[i, 9:]
    self.calls += 1
    self.wall += time.time() - t0
    return out


def frames_to_motion(F30, f0):
  """F30 (64, 31) at times (i + 1) / 30 after frame f0 (31, time 0) -> 50 Hz S.Motion in R1 IsaacLab order (space r1)."""
  allf = np.vstack([f0[None], F30])
  t30 = np.arange(len(allf)) * DT30
  T = int(math.floor(t30[-1] / DT)) + 1
  rp, rq, jp = np.zeros((T, 3)), np.zeros((T, 4)), np.zeros((T, 24))
  for k in range(T):
    x = k * DT / DT30
    i0 = min(int(math.floor(x)), len(allf) - 2)
    w = x - i0
    rp[k] = (1 - w) * allf[i0, :3] + w * allf[i0 + 1, :3]
    rq[k] = S.slerp(allf[i0, 3:7], allf[i0 + 1, 3:7], w)
    jp[k] = (1 - w) * allf[i0, 7:] + w * allf[i0 + 1, 7:]
  return rp, rq, jp


def blend24(old, gen, cur, fgen):
  """T1 blend (8-frame ramp from fgen) for R1-space motions (24 joints)."""
  new_len = fgen - cur + gen.T
  if new_len <= 0:
    return None
  bs = max(0, fgen - cur)
  jp, rp, rq = np.zeros((new_len, 24)), np.zeros((new_len, 3)), np.zeros((new_len, 4))
  for f in range(new_len):
    fo = min(max(f + cur, 0), old.T - 1)
    fn = min(max(f + cur - fgen, 0), gen.T - 1)
    w = min(max((f - bs) / 8.0, 0.0), 1.0)
    jp[f] = (1 - w) * old.jp[fo] + w * gen.jp[fn]
    rp[f] = (1 - w) * old.rpos[fo] + w * gen.rpos[fn]
    rq[f] = S.slerp(old.rquat[fo], gen.rquat[fn], w)
  return r1_motion(rp, rq, jp)


def r1_motion(rp, rq, jp):
  jv = np.zeros_like(jp)
  if len(jp) > 1:
    jv[:-1] = (jp[1:] - jp[:-1]) / DT
    jv[-1] = jv[-2]
  mo = S.Motion(jp, jv, rp, rq, "rp1")
  mo.space = "r1"
  return mo


class Dep30RP1(P.Dep30):
  """Tracker with RP1 in the planner slot (open loop). Built as a static-reference deploy (no G1 planner ONNX is loaded),
  then switched to the RP1 reference; control_tick is T1's original DeployR1Native.control_tick (P29's M1 path unused)."""

  def __init__(self, tracker_ckpt, rp1_ckpt, tl):
    super().__init__("static", S.static_motion(), tracker_ckpt, cmd=None, t_play=0.0)
    self.static_r1 = False
    self.planner = RP1Planner(rp1_ckpt)
    self.tl = tl
    self.loop_ctx = False

  def command(self, t, yaw_ref):
    k, b, st = SR.step_at(self.tl, t)
    vx, vy, wz = SR.twist(st)
    psi = SR.heading_cmd(self.tl, t)
    v = math.hypot(vx, vy)
    ang = psi + (math.atan2(vy, vx) if v > 1e-6 else 0.0)
    face = psi + wz * self.planner.RT.HOR * DT30
    return v, ang, face

  def ctx_at(self, mo, gen):
    """4 frames (30 Hz spacing) ending at frame gen of the R1 reference, as RP1 qpos (RP1 joint order)."""
    out = np.zeros((4, 31))
    for n in range(4):
      x = gen - (3 - n) * DT30 / DT
      i0 = int(math.floor(max(x, 0.0)))
      i0 = min(i0, mo.T - 1)
      i1 = min(i0 + 1, mo.T - 1)
      w = min(max(x - i0, 0.0), 1.0) if x >= 0 else 0.0
      out[n, 0:3] = (1 - w) * mo.rpos[i0] + w * mo.rpos[i1]
      out[n, 3:7] = S.slerp(mo.rquat[i0], mo.rquat[i1], w)
      out[n, 7:] = ((1 - w) * mo.jp[i0] + w * mo.jp[i1])[self.planner.from_il]
    return out

  def pre_roll(self):
    self.sim.reset()
    d = self.sim.d
    f0 = np.concatenate([d.qpos[0:3], d.qpos[3:7], S.R1_DEF_IL[self.planner.from_il]])
    ctx = np.tile(f0, (4, 1))
    v, ang, face = self.command(0.0, yaw_q(f0[3:7]))
    F30 = self.planner.plan(ctx, v, ang, face)
    rp, rq, jp = frames_to_motion(F30, f0)
    self.motion = r1_motion(rp, rq, jp[:, self.planner.to_il])
    self.cur = 0
    self.reinit = True
    self.pending = None

  def planner_tick(self, t):
    mo = self.motion
    gen = self.cur + 2
    ctx = self.ctx_at(mo, gen)
    v, ang, face = self.command(t, yaw_q(ctx[-1, 3:7]))
    F30 = self.planner.plan(ctx, v, ang, face)
    rp, rq, jp = frames_to_motion(F30, ctx[-1])
    self.pending = (r1_motion(rp, rq, jp[:, self.planner.to_il]), gen)
    self.n_replan_self = getattr(self, "n_replan_self", 0) + 1

  def advance(self, t):
    mo = self.motion
    if self.pending is not None:
      gm, fgen = self.pending
      self.pending = None
      nb = blend24(mo, gm, self.cur, fgen)
      if nb is not None:
        self.motion = mo = nb
        self.cur = 0
    self.cur = min(self.cur + 1, mo.T - 1)

  def control_tick(self, k, t):
    return PQ._orig_ct(self, k, t)                              # T1 DeployR1Native.control_tick (sonic_g1 23b7e795)


def episode(job):
  import mujoco
  spec, pol = job["spec"], job["policy"]
  if pol["kind"] != "rp1":
    return _orig_episode(job)
  env = spec.get("env", {})
  body, sense, hits = env.get("body", {}) or {}, env.get("sense") or {}, env.get("hit") or []
  seed = int(spec["seed"])
  tl, T = SR.timeline(spec["mission"])
  SR._ROUGH["spec"] = None
  wall = time.time()
  tracker, rp1ck = pol["path"].split("@", 1)
  if spec["mission"].get("ref", "planner") == "static":
    raise RuntimeError("rp1 policy on a static-reference spec")
  dep = Dep30RP1(tracker, rp1ck, tl)
  import torch
  torch.set_num_threads(1)                                     # after the tracker constructor (it sets 2)
  dep.play_ok = True
  sim = dep.sim
  SR.apply_body(sim, body, {})
  dep.pre_roll()
  m, d = sim.m, sim.d
  sensor = P.Sensor30(sim, sense, int(env["noise_seed"]) if env.get("noise_seed") is not None else SR.child_seed(seed, "sense"))
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
  fell_k, s_idx = -1, 0
  applied = []
  for k in range(steps):
    t = k * DT
    if k == onset_k and onset_k > 0:
      sensor.active, act.d = True, d_full
    obs = sensor.read()
    _kk, _b, st = SR.step_at(tl, t)
    tw = np.array(SR.twist(st))
    mo_obs, cur_obs = dep.motion, dep.cur
    dep.obs = obs
    sim.imu = (lambda o=obs: (o[0].copy(), o[1].copy()))
    dep.control_tick(k, t)
    qt = sim.target24()
    rec["ref_rpos"].append(np.asarray(mo_obs.rpos[min(cur_obs, mo_obs.T - 1)], float).copy())
    act.issue(s_idx, qt)
    rec["qt"].append(np.asarray(qt, float).copy())
    push = None
    if k in push_at:
      F_tot, Tq = np.zeros(3), np.zeros(3)
      yaw = S.yaw_of(d.qpos[3:7])
      for h in push_at[k]:
        a_ = yaw + float(h["dir_rad"])
        F = np.array([math.cos(a_), math.sin(a_), 0.0]) * (float(h["J_Ns"]) / DT)
        F_tot += F
        Tq += np.cross([0.0, 0.0, float(h.get("z_off_m", 0.1))], F)
        applied.append(dict(t_s=t, k=k, J_Ns=h["J_Ns"], dir_rad=h["dir_rad"], z_off_m=h.get("z_off_m", 0.1)))
      push = np.concatenate([F_tot, Tq])
    for _j in range(S.DECIM):
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
      break
  arr = {k: np.array(v, np.float32) for k, v in rec.items()}
  arr["other"] = np.array(rec["other"], bool)
  info = dict(spec_sha256=spec["spec_sha256"], policy=pol["name"], fell_k=fell_k, end_k=len(rec["pos"]), steps=steps,
              wall_s=round(time.time() - wall, 1), mass=float(sim.mass), pushes=applied, rough_h=0.0, names=list(sim.rec_names),
              planner_calls=dep.planner.calls, planner_wall_s=round(dep.planner.wall, 2), replan_robot=0,
              replan_self=getattr(dep, "n_replan_self", 0), loop_ctx=False, bridge_lat_ticks=0, bridge_late=0, m1_buffers=0,
              m1_wall_med=None, planner="rp1")
  return arr, info


_orig_episode = P.episode
P.episode = episode                                            # neg30_pass.work looks `episode` up at call time
_orig_sha16 = P.sha16


def sha16(p):
  if "@" in str(p):
    a, b = str(p).split("@", 1)
    return hashlib.sha256((_orig_sha16(a) + _orig_sha16(b)).encode()).hexdigest()[:16]
  return _orig_sha16(p)


P.sha16 = sha16

if __name__ == "__main__":
  sys.argv[0] = __file__
  P.main()
  out = Path(sys.argv[sys.argv.index("--out") + 1])
  (out / f"meta_rp1_{int(time.time())}.json").write_text(json.dumps(dict(rp1_runner_sha16=_orig_sha16(__file__)), indent=1))
