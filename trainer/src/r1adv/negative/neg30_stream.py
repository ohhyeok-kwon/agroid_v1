"""Negative 30 (coordinator 12:12 / 12:17): the same tracker on a precomputed R1 reference stream (e.g. P34 'calm v2', or the m1c
reference itself as the positive control), next to the m1c loop. No planner, no bridge: the stream is played frame by frame
(T1 DeployR1Native reads R1 frames directly, mo.space = 'r1'); sensing / action delay / pushes / measures as neg30_pass.
Streams file: npz, key -> (T, 31) R1 qpos [root pos 3, wxyz 4, 24 joints in rp1_seq30 / MuJoCo order] at STREAM_HZ (env
N30_STREAM_HZ, default 30; P29 kin panel / RP1 layout). The spec names its stream in env.loop.stream_key.
Specs (gen): per key, 12 s = idle 1 s + command 8 s + idle 3 s (the P29 kin panel timeline): fwd05 -> walk_stop (vx 0.5),
fwd03 / side03 / back03 / turn08 -> moves (one segment, labels fwd0.3 / left0.3 / back0.3 / turn0.8).
  python -I neg30_stream.py gen STREAMS.npz OUT_SPECS.jsonl CELL_TAG
  python -I neg30_stream.py --specs S.jsonl --policy NAME=stream:TRACKER@STREAMS.npz --out DIR --procs 10
  python -I neg30_stream.py table OUT.md RUN_DIR [RUN_DIR ...]
Note: moves rows count a command as walking only with realised >= 0.7 AND foot-lift share >= the manufacturer-based line
(pass table rule); the walk_stop walking-look window is [t0 + 3, end of the Go step].
Positive control (12:2x): B@0 on the m1c reference stream itself (P29 kin_baseline_g1m1.npz) realised fwd 0.5 0.81 (m1c loop
0.83), t50 0.56 (0.56), fwd 0.3 0.77, side 0.53, turn 0.93; back 0.3 fell (that stream's IK penetrates the floor 5.8 cm)."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_rp1 as R  # noqa: E402  (episode loop for R1-space references; neg30_pass measures)

P, S = R.P, R.S
HZ = float(os.environ.get("N30_STREAM_HZ", "30"))
JN = json.load(open("/rl-agent/dataset/r1_planner/rp1_seq30_v1/meta.json"))["joint_names"]
TO_IL = np.array([JN.index(n) for n in S.R1_IL])
CMDS = {"fwd05": ("walk_stop", 0.5, 0.0, 0.0, "fwd0.5"), "fwd03": ("moves", 0.3, 0.0, 0.0, "fwd0.3"),
        "side03": ("moves", 0.0, 0.3, 0.0, "left0.3"), "back03": ("moves", -0.3, 0.0, 0.0, "back0.3"),
        "turn08": ("moves", 0.0, 0.0, 0.8, "turn0.8")}
_CUR = {}


class _NoPlanner:
  calls, wall = 0, 0.0


class Dep30Stream(P.Dep30):
  def __init__(self, tracker_ckpt, streams_npz, tl):
    super().__init__("static", S.static_motion(), tracker_ckpt, cmd=None, t_play=0.0)
    self.static_r1 = False
    self.planner = _NoPlanner()
    self.loop_ctx = False
    self.npz, self.key = streams_npz, _CUR["spec"]["env"]["loop"]["stream_key"]

  def pre_roll(self):
    self.sim.reset()
    Q = np.load(self.npz)[self.key].astype(float)
    t_src = np.arange(len(Q)) / HZ
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, S.CTRL_DT)
    rp = np.stack([np.interp(t_dst, t_src, Q[:, i]) for i in range(3)], 1)
    jp = np.stack([np.interp(t_dst, t_src, Q[:, 7 + i]) for i in range(24)], 1)[:, TO_IL]
    rq = []
    for t in t_dst:
      i = min(int(np.searchsorted(t_src, t, side="right") - 1), len(t_src) - 2)
      a = (t - t_src[i]) * HZ
      rq.append(S.slerp(Q[i, 3:7], Q[i + 1, 3:7], min(max(a, 0.0), 1.0)))
    self.motion = R.r1_motion(rp, np.array(rq), jp)
    self.cur, self.reinit, self.pending, self.play = 0, True, None, True

  def planner_tick(self, t):
    return None

  def advance(self, t):
    self.cur = min(self.cur + 1, self.motion.T - 1)

  def control_tick(self, k, t):
    return P.PQ._orig_ct(self, k, t)


_orig_episode = R.episode


def episode(job):
  pol = job["policy"]
  if pol["kind"] != "stream":
    return _orig_episode(job)
  _CUR["spec"] = job["spec"]
  pol2 = dict(pol, kind="rp1")
  R.Dep30RP1, saved = Dep30Stream, R.Dep30RP1
  try:
    return _orig_episode(dict(job, policy=pol2))
  finally:
    R.Dep30RP1 = saved


P.episode = episode


def gen(npz, out, tag):
  sys.path.insert(0, "/rl-agent/src/r1adv/negative")
  import neg30_specs as G
  keys = [k for k in CMDS if k in np.load(npz).files]
  sp = []
  for k in keys:
    fam, vx, vy, wz, lab = CMDS[k]
    if fam == "walk_stop":
      mis = dict(settle_s=1.0, steps=[dict(type="Go", vx=vx, duration_s=8.0, label=lab), dict(type="Hold", duration_s=3.0)])
    else:
      mis = dict(settle_s=1.0, steps=[dict(type="Go", vx=vx, vy=vy, wz=wz, duration_s=8.0, label=lab), dict(type="Hold", duration_s=3.0)])
    env = dict(loop=dict(planner="stream", stream_key=k, stream_file=os.path.abspath(npz)), body=dict(action_delay_ms=0.0), onset_s=1.0,
               sense={}, noise_seed=0)
    s = dict(schema="r1adv-scenario-v1", generator=dict(name="neg30-stream", version="v1"), seed=G.H(G.ROUND, f"stream|{tag}|{k}"),
             family=fam, cell=f"stream_{tag}", evalset="stream", i=0, mission=mis, env=env, solvability_evidence=None)
    sp.append(G.finish(s))
  open(out, "w").write("".join(json.dumps(x) + "\n" for x in sp))
  print(len(sp), "specs", keys)


def table(out, dirs):
  rows = []
  for d in dirs:
    for l in open(os.path.join(d, "rows.jsonl")):
      r = json.loads(l)
      if "error" not in r:
        rows.append(r)
  f = lambda x, nd=2: "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))  # noqa: E731
  L = ["| 고리(칸) | 정책 | 명령 | 실현 | 발 뗌 | t50 | 정지 | roll ° | 골반 m | 걸음/s | 넘어짐 | 두 발 뛰기 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
  for r in sorted(rows, key=lambda r: (r["policy"], r["spec_sha256"], r["cell"])):
    seg = (r.get("segments") or [{}])[0]
    real = r.get("steady_ratio") if r["family"] == "walk_stop" else seg.get("realised")
    L.append(f"| {r['cell']} | {r['policy']} | {'fwd0.5' if r['family'] == 'walk_stop' else seg.get('label')} | {f(real)} | "
             f"{f(seg.get('swing_share'))} | {f(r.get('t50_s'))} | {f(r.get('stop_s'))} | {f(r.get('roll_rms_deg'))} | "
             f"{f(r.get('pelvis_lat_pp_m'), 3)} | {f(r.get('step_hz'))} | {r['F']} | {r.get('hops')} |")
  open(out, "w").write("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  if sys.argv[1] == "gen":
    gen(sys.argv[2], sys.argv[3], sys.argv[4])
  elif sys.argv[1] == "table":
    table(sys.argv[2], sys.argv[3:])
  else:
    sys.argv[0] = __file__
    P.main()
