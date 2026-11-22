"""Negative 30 (coordinator 11:49 item 5): RP1 self-feedback panel. Lines fixed BEFORE any v3 value from two witnesses:
(a) the G1 planner (cand) + B2 + M1 streams (P29 kin_baseline_g1m1.npz, 12 s per command, 30 Hz) and (b) human R1 clips
(rp1_seq30_v1, 30 Hz; random 12 s windows, seeded). Quantities per 12 s stream (R1 FK = rp1_fk.R1FK, the tracker's model):
  z_drift    max |root z(t) - root z(window start)| (m)
  z_range    p99 - p1 of root z (m)
  float      share of frames where the lower foot's lowest sole corner is > 2 cm above the floor while the other foot is
             also > 2 cm (both feet floating = flight) -> 'both_air'; and lowest-sole height p99 of the support foot (m)
  pen        lowest sole corner minimum (m, negative = through the floor)
  seam       largest per-frame joint step (rad / frame at 30 Hz) and largest root xy step (m / frame)
  slip       stance-foot (lowest corner < 1 cm) horizontal speed p95 (m/s)
Human witness = 12 s windows of locomotion at standing height (net root speed 0.05-0.8 m/s, median root z >= 0.70 m), 200
seeded windows. Stance for slip = lowest corner < 5 mm held 3 frames (touchdown / lift-off frames excluded).
Line = 1.5 x max(G1+M1 p95, human p95) ('deploy-replaceable': no worse than both witnesses); line_human = 1.5 x human p95
(stricter tier). A v3 stream passes when every quantity is inside its line over the whole 12 s self-feedback rollout.
  python -I neg30_rp1_selffeed.py lines OUT.json          (witness lines)
  python -I neg30_rp1_selffeed.py check LINES.json STREAMS.npz OUT.json   (STREAMS: name -> (T, 31) R1 qpos 30 Hz)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/rl-agent/.agent/tmp/p29/rp1")
from rp1_fk import R1FK  # noqa: E402  (P29, read only)

DATA = "/rl-agent/dataset/r1_planner/rp1_seq30_v1"
BASE = "/rl-agent/.agent/tmp/p29/rp1/kin_baseline_g1m1.npz"
JN = json.load(open(f"{DATA}/meta.json"))["joint_names"]
FK = R1FK(JN)


def quantities(Q):
  Q = np.asarray(Q, float)
  with torch.no_grad():
    o = FK(torch.tensor(Q[:, :3], dtype=torch.float32), torch.tensor(Q[:, 3:7], dtype=torch.float32),
           torch.tensor(Q[:, 7:], dtype=torch.float32))
  sole = o["sole"].numpy()                                   # (T, 2, 4, 3)
  hmin = sole[..., 2].min(-1)                                # (T, 2)
  low = hmin.min(1)
  cen = sole.mean(-2)
  st = hmin < 0.005
  st3 = st[1:-1] & st[:-2] & st[2:]                          # stance held over 3 frames (no touchdown / lift-off frames)
  fv = np.linalg.norm(cen[2:, :, :2] - cen[:-2, :, :2], axis=-1) * 15.0
  sl = np.concatenate([fv[:, f][st3[:, f]] for f in (0, 1)])
  return dict(z_drift=float(np.abs(Q[:, 2] - Q[0, 2]).max()), z_range=float(np.percentile(Q[:, 2], 99) - np.percentile(Q[:, 2], 1)),
              both_air=float(((hmin[:, 0] > 0.02) & (hmin[:, 1] > 0.02)).mean()), support_float_p99=float(np.percentile(low, 99)),
              pen=float(-sole[..., 2].min()), seam_joint=float(np.abs(np.diff(Q[:, 7:], axis=0)).max()),
              seam_root=float(np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1).max()),
              slip_p95=float(np.percentile(sl, 95)) if len(sl) else 0.0)


def lines(out):
  wit = {}
  z = np.load(BASE)
  wit["g1m1"] = {n: quantities(z[n]) for n in z.files}
  Qd = np.load(f"{DATA}/qpos30.npy", mmap_mode="r")
  off = np.load(f"{DATA}/offsets.npy")
  meta = json.load(open(f"{DATA}/meta.json"))
  rng = np.random.default_rng(20261121)
  hum = {}
  clips = [k for k, c in enumerate(meta["clips"]) if int(off[k + 1]) - int(off[k]) > 360]
  tried = 0
  for i in rng.permutation(len(clips)):
    if len(hum) >= 200 or tried > 3000:
      break
    tried += 1
    k = clips[int(i)]
    a = int(off[k]) + int(rng.integers(0, int(off[k + 1]) - int(off[k]) - 360))
    Q = np.asarray(Qd[a:a + 360])
    sp = float(np.linalg.norm(Q[-1, :2] - Q[0, :2]) / 12.0)
    if not (0.05 <= sp <= 0.8 and np.median(Q[:, 2]) >= 0.70):  # locomotion at standing height (deploy grammar)
      continue
    hum[f"clip{k}_{a}"] = quantities(Q)
  wit["human"] = hum
  keys = list(next(iter(hum.values())).keys())
  L = {}
  for k in keys:
    g = float(np.percentile([v[k] for v in wit["g1m1"].values()], 95))
    h = float(np.percentile([v[k] for v in hum.values()], 95))
    L[k] = dict(line=round(1.5 * max(g, h), 4), line_human=round(1.5 * h, 4), g1m1_p95=g, human_p95=h,
                human_med=float(np.median([v[k] for v in hum.values()])))
  json.dump(dict(lines=L, witnesses={"g1m1": wit["g1m1"], "human_n": len(hum)}), open(out, "w"), indent=1)
  for k, v in L.items():
    print(f"{k:18s} line {v['line']:.4f} (human-level {v['line_human']:.4f}) | G1+M1 p95 {v['g1m1_p95']:.4f} | human p95 {v['human_p95']:.4f} (med {v['human_med']:.4f})")


def check(lines_path, streams, out):
  L = json.load(open(lines_path))["lines"]
  z = np.load(streams)
  res = {}
  for n in z.files:
    q = quantities(z[n])
    res[n] = dict(values=q, fail=[k for k, v in q.items() if v > L[k]["line"]],
                  fail_human_level=[k for k, v in q.items() if v > L[k]["line_human"]])
  json.dump(res, open(out, "w"), indent=1)
  keys = list(L)
  md = [f"# RP1 자기 되먹임 판 — {streams}", "", "문턱: 배포 대체 가능 (사람 수준) = " +
        ", ".join(f"{k} {L[k]['line']:.4g} ({L[k]['line_human']:.4g})" for k in keys), "",
        "| 명령 | 판정 | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
  for n, r in res.items():
    md.append(f"| {n} | {'PASS' if not r['fail'] else 'FAIL'}{' (사람 수준 PASS)' if not r['fail_human_level'] else ''} | " +
              " | ".join(f"{r['values'][k]:.4g}{' ✗' if k in r['fail'] else ''}" for k in keys) + " |")
  Path(str(out).replace(".json", ".md")).write_text("\n".join(md) + "\n")
  for n, r in res.items():
    print(n, "PASS" if not r["fail"] else "FAIL " + ",".join(r["fail"]), {k: round(v, 4) for k, v in r["values"].items()})


if __name__ == "__main__":
  if sys.argv[1] == "lines":
    lines(sys.argv[2])
  else:
    check(sys.argv[2], sys.argv[3], sys.argv[4])
