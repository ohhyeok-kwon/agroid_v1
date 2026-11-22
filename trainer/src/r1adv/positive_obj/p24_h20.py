"""Positive 24 E3 h20: the E1 c09 hold frame with a long window (rounds/R04/positive/e3_h20/rules.md).

Same code path as E1 c09 (p24_hold.setup / run_world / target_at unchanged, frozen cfd60f65): squeeze, support, release at
1.2 s, then still 5 s [1.2, 6.2) and the replayed shake (positive 23 trace, env 7, t 5.51-8.505 s, a_k of shake_profiles
'trace', the 600-sample window repeated back to back) for 25 s [6.2, 31.2): 30 s after release. Delay 0 only.
Cells: C·K1, C·K3, H·K3, H·K2, C·K2 x width 0.20, 0.30 x mass 0.2, 0.6, 1.22 x 8 worlds (jitter seed 2411, as c09).
Readout (rules 3): held at 20 s and at 30 s after release (the c09 conditions c1-c6 on [1.2, 1.2 + T]), the pair-frame
slip every 5 s, the first time the slip from release exceeds 50 mm.

  python -m r1adv.positive_obj.p24_h20 <out_dir> <shake.npz> <hug_ik_dir> [--workers N]
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

from r1adv.positive_obj import p24_hold as P

RULES = "rounds/R04/positive/e3_h20/rules.md"
T_STILL, T_SHAKE = 5.0, 25.0
T_SH0 = P.T_REL + T_STILL                                       # 6.2
T_END = T_SH0 + T_SHAKE                                         # 31.2
K_SH0 = int(round(T_SH0 / P.DT))                                # 1240
NSTEP = int(round(T_END / P.DT))                                # 6240
CELLS = [("C", "K1"), ("C", "K3"), ("H", "K3"), ("H", "K2"), ("C", "K2")]
WIDTHS = (0.2, 0.3)
MASSES = (0.2, 0.6, 1.22)
MARKS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)                     # seconds after release
SHAKE_KEY = "trace_rep"


def jobs():
  return [(g, k, w, m, j, SHAKE_KEY, 0) for g, k in CELLS for w in WIDTHS for m in MASSES for j in range(P.N_WORLD)]


def c50(t_after: float) -> int:
  """50 Hz index of the state at t = T_REL + t_after (obj_p[c] is the state after step 4c + 3, time (c + 1) * 0.02)."""
  return int(round((P.T_REL + t_after) / (P.DT * P.DECIM))) - 1


def held_at(info, tr, T: float) -> dict:
  n = len(tr["fn"])
  S = (np.arange(n) + 1) * P.DT
  W = (S >= P.T_REL - 1e-9) & (S < P.T_REL + T + 1e-6)
  nc = tr["nc"]
  both = (nc[:, 0] > 0) & (nc[:, 1] > 0)
  op, hp = tr["obj_p"], tr["hand_p"]
  c0, c1_ = c50(0.0), c50(T)
  slip = P.rel(op, hp, c0, c1_)
  pen = float(max(-tr["dmin"][S < P.T_REL + T + 1e-6].min(), 0.0))
  c1 = bool((tr["flo"][W] == 0).all())
  c2 = bool(op[c1_, 2] >= op[c0, 2] - 0.05)
  c3 = slip <= 50.0
  c4 = bool(both[W].mean() >= 0.95)
  if info["grip"] == "C":
    c5 = bool((tr["ch_n"][W] > 0).mean() >= 0.95)
  else:
    c5 = bool(((tr["other"][W] + tr["ch_n"][W]) > 0).mean() <= 0.05)
  c6 = pen <= 0.005
  return dict(held=bool(c1 and c2 and c3 and c4 and c5 and c6), c=[c1, c2, c3, c4, c5, c6], slip=slip)


def metrics(info, tr) -> dict:
  n = len(tr["fn"])
  S = (np.arange(n) + 1) * P.DT
  Ss = (S >= P.T_REL - 0.1 - 1e-9) & (S < P.T_REL - 1e-9)
  nc = tr["nc"]
  both = (nc[:, 0] > 0) & (nc[:, 1] > 0)
  start_both = bool(both[Ss].all())
  start_chest = bool((tr["ch_n"][Ss] > 0).all() and tr["ch_f"][Ss].mean() >= P.CHEST_LOAD_MIN * info["chest_n_design"]) \
    if info["grip"] == "C" else True
  g = info["init_gate"]
  gate_ok = g[0] <= 0.002 and g[1] <= 0.001 and g[2] <= 0.001
  start_valid = bool(start_both and start_chest and info["v_rel"] < 0.05 and gate_ok and info["feasible"])
  op, hp = tr["obj_p"], tr["hand_p"]
  c0 = c50(0.0)
  curve = {f"{m:g}": P.rel(op, hp, c0, c50(m)) for m in MARKS}
  disp = np.array([P.rel(op, hp, c0, c) for c in range(c0, c50(30.0) + 1)])
  over = np.nonzero(disp > 50.0)[0]
  t50 = float(over[0] * P.DT * P.DECIM) if len(over) else float("nan")       # seconds after release
  h20, h30 = held_at(info, tr, 20.0), held_at(info, tr, 30.0)
  out = {k: info[k] for k in ("grip", "contact", "arm", "w", "mass", "world", "feasible", "f_n_design", "chest_n_design")}
  out.update(start_valid=start_valid, start_both=start_both, start_chest=start_chest, gate_ok=gate_ok, v_rel=info["v_rel"],
             init_gate=info["init_gate"], held20=h20["held"], held30=h30["held"], c20=h20["c"], c30=h30["c"],
             slip_curve_mm=curve, t50_after_release_s=t50,
             chest_frac=float((tr["ch_n"][S >= P.T_REL - 1e-9] > 0).mean()),
             drop30_mm=float((op[c0, 2] - op[c50(30.0), 2]) * 1000.0))
  return out


_PROF = {}


def _job(args):
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  job, shake_path, ik_dir = args
  if "p" not in _PROF:
    tr = np.load(shake_path)["trace"]
    reps = int(math.ceil(T_SHAKE / P.DT / len(tr)))
    _PROF["p"] = {SHAKE_KEY: np.tile(tr, reps)[:int(round(T_SHAKE / P.DT))]}
    P.K_SH0 = K_SH0                                         # the shake start used inside p24_hold.run_world
  info, tr = P.run_world(job, _PROF["p"], ik_dir, tend=T_END)
  return metrics(info, tr)


def main(argv):
  out_dir, shake_path, ik_dir = argv[0], argv[1], argv[2]
  workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 24
  os.makedirs(out_dir, exist_ok=True)
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  js = jobs()
  with ProcessPoolExecutor(max_workers=workers) as ex:
    rows = list(ex.map(_job, [(j, shake_path, ik_dir) for j in js], chunksize=2))
  here = os.path.dirname(__file__)
  meta = dict(rules=RULES, n=len(rows), wall_s=round(time.time() - t0, 1), t_still=T_STILL, t_shake=T_SHAKE, t_end=T_END,
              shake=shake_path, shake_sha256=hashlib.sha256(open(shake_path, "rb").read()).hexdigest(), ik_dir=ik_dir,
              code_sha256={f: hashlib.sha256(open(os.path.join(here, f), "rb").read()).hexdigest()
                           for f in ("p24_h20.py", "p24_hold.py", "p24_hug_ik.py", "p20_hold.py", "p20_gate.py", "p20_fist.py",
                                     "p20_scene.py", "p20_ladder.py", "p20_readout.py")})
  json.dump(dict(meta=meta, rows=rows), open(os.path.join(out_dir, "h20_rows.json"), "w"), default=float)
  print(f"done {len(rows)} worlds in {meta['wall_s']} s", flush=True)


if __name__ == "__main__":
  main(sys.argv[1:])
