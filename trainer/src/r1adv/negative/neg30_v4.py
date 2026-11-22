"""Negative 30 (coordinator 17:40): pass table and push panel on the v4 reference through the REAL-TIME path.
Reference = P34 causal synthesizer v4 (calm4_causal.Stream, cadence 3.0 steps/s, width 0.20 m, swing 5 cm; imported, not edited),
[23:2x: the generator is the ruler NEG30_GEN (neg30_gen.py): v4 by default, v43s = P38 calm43s_causal; OUT/ruler.json records it]
[11-22 04:3x: the body is NEG30_BODY (neg30_body.py): r1wb by default, b1 = panel body; OUT/body.json records it]
driven tick by tick (50 Hz) with the command of that tick only (body-frame twist from the mission timeline). The tracker reads
the current frame and 9 frames up to +0.9 s: those come from st.preview(command, 2 s), recomputed whenever the command changes
and when fewer than 46 frames are left (P34 v4 note: after a command change the old preview is no longer valid). Verified:
the preview equals the frames the stream then emits while the command is unchanged (max diff 0.0). No future command is
visible to the tracker (unlike fixed clips, P36). Sensing / action delay / pushes / records as neg30_pass (via neg30_rp1).
Families (neg30_specs missions): walk_stop, moves, stand_push (pass table) and v4push (push panel: standing Hold 12 s or walking
0.3 / 0.5 m/s, one push of J N s in one of 8 directions at a gait phase; measures below). Policy: NAME=v4:TRACKER_CKPT.
  python -I neg30_v4.py --specs S.jsonl --policy NAME=v4:CKPT --policy mf=mf:ONNX --out DIR --procs 28
Push measures (v4push): fell, two-foot hops after the push, recovery = first time after the push from which the base velocity
(0.5 s centred average, heading frame) stays within 0.10 m/s of the command (standing: |v| < 0.10) for 0.5 s; arm = max over [tp, tp + 2 s] of the largest arm joint
deviation from the R1 default (rad); torso = max torso tilt (deg) over the same window."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_body as NB  # noqa: E402  (NEG30_BODY r1wb | b1: sets R1_BODY / R1_XML before sonic_g1 is imported)
import neg30_rp1 as R  # noqa: E402

P, S, SR = R.P, R.S, R.SR
NB.patch(S)
sys.path.insert(0, "/rl-agent/.agent/tmp/p34")
sys.path.insert(0, "/rl-agent/.agent/tmp/t1_sonic/work")
import neg30_gen as RG  # noqa: E402  (ruler: NEG30_GEN, default v4)

CENTER = RG.CENTER
LOOK = 46


class _NoPlanner:
  calls, wall = 0, 0.0


class Dep30V4(P.Dep30):
  def __init__(self, tracker_ckpt, _unused, tl):
    super().__init__("static", S.static_motion(), tracker_ckpt, cmd=None, t_play=0.0)
    self.static_r1 = False
    self.planner = _NoPlanner()
    self.loop_ctx = False
    self.tl = tl
    import clip_eval as CE
    self.to_il = np.array([list(CE.R1_MJ).index(n) for n in S.R1_IL])

  def _cmd(self, t):
    _k, _b, st = SR.step_at(self.tl, t)
    return tuple(float(x) for x in SR.twist(st))

  def _motion(self, frames):
    F = np.asarray(frames, float)
    jp = F[:, 7:][:, self.to_il]
    jv = np.zeros_like(jp)
    jv[:-1] = (jp[1:] - jp[:-1]) / S.CTRL_DT
    jv[-1] = jv[-2]
    mo = S.Motion(jp, jv, F[:, :3].copy(), F[:, 3:7].copy(), "v4")
    mo.space = "r1"
    return mo

  def pre_roll(self):
    self.sim.reset()
    self.st = RG.stream()
    self.k, self.prev_c, self.pv, self.pv_k0 = 0, None, None, 0
    c = self._cmd(0.0)
    q0 = np.asarray(self.st.frame(c), float)
    self._q_now = q0
    self.pv, self.pv_k0, self.prev_c = np.asarray(self.st.preview(c, 2.0), float), 1, c
    self.motion = self._motion([q0, q0] + [self.pv[j] for j in range(LOOK)])
    self.cur, self.reinit, self.pending, self.play = 1, True, None, True
    self.n_preview = 1

  def planner_tick(self, t):
    return None

  def advance(self, t):
    return None

  def control_tick(self, k, t):
    c = self._cmd(t)
    if k > 0:
      self._q_now = np.asarray(self.st.frame(c), float)
      self.k += 1
      if c != self.prev_c or self.k + LOOK >= self.pv_k0 + len(self.pv):
        self.pv, self.pv_k0, self.prev_c = np.asarray(self.st.preview(c, 2.0), float), self.k + 1, c
        self.n_preview += 1
    fut = [self.pv[self.k + j - self.pv_k0] for j in range(1, LOOK + 1)]
    self.motion = self._motion([self._q_now, self._q_now] + fut)
    self.cur, self.play = 1, True
    return P.PQ._orig_ct(self, k, t)


_orig_episode = R.episode


def episode(job):
  pol = job["policy"]
  if pol["kind"] != "v4":
    return _orig_episode(job)
  saved = R.Dep30RP1
  R.Dep30RP1 = Dep30V4
  try:
    return _orig_episode(dict(job, policy=dict(pol, kind="rp1", path=pol["path"] + "@none")))
  finally:
    R.Dep30RP1 = saved


P.episode = episode
_orig_measure = P.measure
ARM = ("shoulder", "elbow", "wrist")


def flow_measure(spec, arr, info):
  """Indoor command flows (negative 35 scene_flow_*): charter G (neg28 sonic_run.measure: path / heading error against the
  integrated command, tolerance 0.10 m + 0.10 x distance), per Go segment realised + foot-lift share (moves rule), stuck =
  Go segments realised < 0.1, posture over the whole flow (time-mean torso roll / pitch, arm max |mean - default|, leg mirror
  asymmetry max)."""
  base = SR.measure(spec, arr, info)
  mv = _orig_measure(dict(spec, family="moves"), arr, info)
  segs = mv.get("segments", [])
  stuck = sum(1 for s_ in segs if s_.get("realised") is not None and s_["realised"] < 0.1)
  q, tq = arr["q"].astype(float), arr["torso_quat"].astype(float)
  n = len(q)
  fell = info["fell_k"] >= 0
  post = {}
  if not fell and n > 100:
    w_, x_, y_, z_ = tq[50:, 0], tq[50:, 1], tq[50:, 2], tq[50:, 3]
    roll = np.degrees(np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_ * x_ + y_ * y_)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1, 1)))
    names = info["names"]
    qm = q[50:].mean(0)
    default = dict(zip(S.R1_IL, S.R1_DEF_IL))
    arm = [abs(qm[i] - default[nm]) for i, nm in enumerate(names) if any(a in nm for a in ARM)]
    asym = []
    for j, sg in (("hip_pitch", 1), ("hip_roll", -1), ("hip_yaw", -1), ("knee", 1), ("ankle_pitch", 1), ("ankle_roll", -1)):
      asym.append(abs(math.degrees(qm[names.index(f"left_{j}_joint")] - sg * qm[names.index(f"right_{j}_joint")])))
    post = dict(torso_roll_mean=float(roll.mean()), torso_pitch_mean=float(pitch.mean()), arm_dev_max=float(max(arm)),
                leg_asym_max=float(max(asym)), roll_std=float(roll.std()))
  return dict(spec_sha256=spec["spec_sha256"], family=spec["family"], cell=spec.get("cell"), i=spec.get("i"), policy=info["policy"],
              F=bool(fell), t_fall=base.get("t_fall"), hops=mv.get("hops"), G_ok=base.get("G_ok"), G_int=base.get("G_int"),
              G_ratio_max=base.get("G_ratio_max"), path_ratio=base.get("path_ratio"), cmd_dist_m=base.get("cmd_dist_m"),
              stuck=stuck, n_go=len(segs), segments=segs, wall_s=info["wall_s"], **post)


def measure(spec, arr, info):
  if str(spec.get("family", "")).startswith("scene_flow"):
    return flow_measure(spec, arr, info)
  if spec.get("family") != "v4push":
    return _orig_measure(spec, arr, info)
  DT = P.DT
  names = info["names"]
  n = len(arr["pos"])
  t = (np.arange(n) + 1) * DT
  pos, quat, v = arr["pos"].astype(float), arr["quat"].astype(float), arr["v"].astype(float)
  q, tq = arr["q"].astype(float), arr["torso_quat"].astype(float)
  hit = spec["env"]["hit"][0]
  tp = round(float(hit["t_s"]) / DT) * DT
  kp = int(round(tp / DT))
  fell = info["fell_k"] >= 0
  tl, _T = SR.timeline(spec["mission"])
  yaw = P.yaw_of(quat)
  vb = np.stack([v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw), -v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw)], 1)
  vb = np.stack([P.smooth(vb[:, 0]), P.smooth(vb[:, 1])], 1)     # 0.5 s centred average (P24 step-response rule): no stride ripple
  cmd = np.array([SR.twist(SR.step_at(tl, tt - DT)[2])[:2] for tt in t])
  err = np.linalg.norm(vb - cmd, axis=1)
  rec = None
  if not fell:
    for k in range(kp + 1, n - 25):
      if (err[k:k + 25] < 0.10).all():
        rec = float((k - kp) * DT)
        break
  ix = [i for i, nm in enumerate(names) if any(a in nm for a in ARM)]
  default = dict(zip(S.R1_IL, S.R1_DEF_IL))
  dq0 = np.array([default[names[i]] for i in ix])
  w = slice(kp, min(n, kp + int(2.0 / DT)))
  arm = float(np.abs(q[w][:, ix] - dq0).max()) if not fell else None
  up = 1 - 2 * (tq[w, 1] ** 2 + tq[w, 2] ** 2)
  torso = float(np.degrees(np.arccos(np.clip(up, -1, 1))).max()) if not fell else None
  return dict(spec_sha256=spec["spec_sha256"], family="v4push", cell=spec.get("cell"), i=spec.get("i"), policy=info["policy"],
              mode=spec["push"]["mode"], J=float(hit["J_Ns"]), dir_deg=round(math.degrees(float(hit["dir_rad"])) % 360),
              phase=spec["push"]["phase"], F=bool(fell), hops=P.hops(arr["force"].astype(float), arr["sole"].astype(float), kp),
              recovery_s=rec, arm_max=arm, torso_max_deg=torso, wall_s=info["wall_s"])


def _measure_body(spec, arr, info):
  row = measure(spec, arr, info)
  if NB.BODY != "r1wb":                 # b1 rows carry the checked model (mass, root terms, body_hash)
    row.update(NB.row_tag(S))
  return row


P.measure = _measure_body


def gen(out):
  """Pass table (v4 real-time, noise-free; walk_stop also D2 x 0.1 n 8) and push panel specs; delay set at run time per cell
  name suffix: _d15 for trackers, _d00 for the manufacturer (both written; the runner skips by --mf-cells / policy filter)."""
  import json
  import neg30_specs as G

  def fin(s, cell, delay):
    s = json.loads(json.dumps(s))
    s.pop("spec_sha256", None)
    s["env"]["loop"] = dict(planner="calm4_causal", path="realtime", bridge="none", ctx_robot=False, bridge_lat_s=0.0)
    s["env"]["body"]["action_delay_ms"] = float(delay)
    s["cell"] = f"{cell}_d{delay:02d}"
    return G.finish(s)

  sp = []
  for d in (0, 15):
    sp.append(fin(G.make("walk_stop", "ol_L01_d00", "A", 0, noise=False), "v4_walk_nf", d))
    for i in range(8):
      s = G.make("walk_stop", "ol_L01_d00", "A", i, noise=True)
      s["env"]["sense"] = {k: (x * 0.1 if isinstance(x, float) else x) for k, x in s["env"]["sense"].items()}
      sp.append(fin(s, "v4_walk_lo", d))
    mv = G.make("moves", "ol_L01_d00", "A", 0, noise=False)
    for st_ in mv["mission"]["steps"]:                           # v4 synthesizer limit: side 0.25 m/s (P36 LIMITS); 0.3 fails to stop
      if st_.get("label") == "left0.3":
        st_["vy"], st_["label"] = 0.2, "left0.2"
    sp.append(fin(mv, "v4_moves_nf", d))
    sp += [fin(G.make("stand_push", "ol_L01_d00", "A", i, noise=False), "v4_stand_nf", d) for i in range(16)]
    for mode, v in (("stand", 0.0), ("walk03", 0.3), ("walk05", 0.5)):
      for J in (12.0, 18.0, 25.0):
        for k8 in range(8):
          for ph in ((0.0,) if mode == "stand" else (0.0, 0.167)):
            if mode == "stand":
              mis = dict(settle_s=0.0, steps=[dict(type="Hold", duration_s=12.0)])
              tp = 6.0
            else:
              mis = dict(settle_s=2.0, steps=[dict(type="Go", vx=v, duration_s=10.0, label=mode), dict(type="Hold", duration_s=3.0)])
              tp = 6.0 + ph
            s = dict(schema="r1adv-scenario-v1", generator=dict(name="neg30-v4push", version="v1"),
                     seed=G.H(G.ROUND, f"v4push|{mode}|{J}|{k8}|{ph}"), family="v4push", evalset="v4push", i=0, mission=mis,
                     push=dict(mode=mode, phase=ph),
                     env=dict(body={}, onset_s=1.0, sense={}, noise_seed=0,
                              hit=[dict(t_s=tp, J_Ns=J, dir_rad=round(k8 * math.pi / 4, 6), z_off_m=0.1)]))
            sp.append(fin(s, f"v4push_{mode}", d))
  d15 = [x for x in sp if x["cell"].endswith("_d15")]
  d00 = [x for x in sp if x["cell"].endswith("_d00")]
  open(out, "w").write("".join(json.dumps(x) + "\n" for x in d15))
  open(out.replace(".jsonl", "_d00.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in d00))
  print(len(d15), "specs (d15, trackers) +", len(d00), "(d00, manufacturer)")


def gen_dev(out):
  """Coordinator 17:40 item 3, one axis at a time from the base (v4 real-time, noise-free, delay 15 ms for trackers / 0 for the
  manufacturer unless the axis is the delay): walk_stop fwd 0.5 + Hold 12 s with a front push 18 N s at 6 s, per level."""
  import json
  import neg30_specs as G
  levels = [("delay", d, dict(action_delay_ms=float(d)), None) for d in (0, 10, 20, 30, 40)]
  levels += [("noise", x, {}, x) for x in (0.1, 0.3)]
  levels += [("friction", mu, dict(foot_mu=mu), None) for mu in (0.3, 0.5, 0.7, 1.0)]
  levels += [("torso_mass", sc, dict(mass_scale={"torso": sc}), None) for sc in (0.9, 1.1)]
  levels += [("torso_com", f"{dx:+.2f},{dy:+.2f}", dict(torso_com=[dx, dy, 0.0]), None) for dx, dy in ((0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03))]
  levels += [("gain", g, dict(kp_scale=g, kd_scale=g), None) for g in (0.85, 1.15)]
  sp = []
  for ax, x, body, noise in levels:
    for who, dflt in (("trk", 15.0), ("mf", 0.0)):
      reps = 8 if noise else 1
      for i in range(reps):
        for fam in ("walk_stop", "stand"):
          if fam == "walk_stop":
            s = G.make("walk_stop", "ol_L01_d00", "A", i, noise=bool(noise))
          else:
            s = G.make("stand_push", "ol_L01_d00", "A", i, noise=bool(noise))
            s["family"] = "v4push"
            s["push"] = dict(mode="stand", phase=0.0)
            s["mission"] = dict(settle_s=0.0, steps=[dict(type="Hold", duration_s=12.0)])
            s["env"]["hit"] = [dict(t_s=6.0, J_Ns=18.0, dir_rad=0.0, z_off_m=0.1)]
          s.pop("spec_sha256")
          if noise:
            s["env"]["sense"] = {k: (v * noise if isinstance(v, float) else v) for k, v in G.D2.items()}
            s["env"]["sense"]["scale_of_D2"] = noise
          b = dict(action_delay_ms=dflt)
          b.update(body)
          s["env"]["body"] = b
          s["env"]["loop"] = dict(planner="calm4_causal", path="realtime", bridge="none", ctx_robot=False, bridge_lat_s=0.0)
          s["dev"] = dict(axis=ax, level=x)
          s["cell"] = f"dev_{ax}_{x}_{fam}_{who}"
          sp.append(G.finish(s))
  for who in ("trk", "mf"):
    open(out.replace(".jsonl", f"_{who}.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in sp if x["cell"].endswith(who)))
  print(len(sp), "deviation specs")


def gen_flows(out):
  """Negative 35 evaluation flows (read only) on the v4 real-time path: delay 15 ms (trackers) / 0 (manufacturer), noise-free."""
  import glob
  import json
  import neg30_specs as G
  for who, d in (("trk", 15.0), ("mf", 0.0)):
    sp = []
    for fn in sorted(glob.glob("/rl-agent/.agent/tmp/neg35/flows/n35_v0_*_eval.jsonl")):
      for l in open(fn):
        if not l.strip():
          continue
        s = json.loads(l)
        src = s.pop("spec_sha256", None)
        s["env"] = dict(body=dict(action_delay_ms=d), onset_s=1.0, sense={}, noise_seed=0, hit=[],
                        loop=dict(planner="calm4_causal", path="realtime", bridge="none", ctx_robot=False, bridge_lat_s=0.0))
        s["n35_source_spec"] = src
        s["cell"] = f"n35_{s['family']}_{who}"
        sp.append(G.finish(s))
    open(out.replace(".jsonl", f"_{who}.jsonl"), "w").write("".join(json.dumps(x) + "\n" for x in sp))
    print(who, len(sp), "flow specs")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "gen":
    gen(sys.argv[2])
  elif len(sys.argv) > 1 and sys.argv[1] == "gen_dev":
    gen_dev(sys.argv[2])
  elif len(sys.argv) > 1 and sys.argv[1] == "gen_flows":
    gen_flows(sys.argv[2])
  else:
    sys.argv[0] = __file__
    _a = sys.argv
    if any(x.split("=", 1)[-1].startswith("v4:") for i, x in enumerate(_a) if i and _a[i - 1] == "--policy"):
      RG.claim(_a[_a.index("--out") + 1])      # reference-driven rows: one ruler per directory
    NB.claim(_a[_a.index("--out") + 1])        # one body per directory (body.json; none = r1wb)
    P.main()
