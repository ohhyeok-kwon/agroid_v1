"""Negative 30 (coordinator 11:49 item 4, evaluation only): PD-target change statistics while walking 0.5 m/s, from
neg30_pass.py npz records (issued targets 'qt' and measured joints 'q', entity order, 50 Hz), window [t0 + 3, t0 + 13] s:
  per-step |delta qt| (rad per 20 ms control step): mean over joints and time, p95, per joint group (legs / waist / arms);
  leg joint angle 4-25 Hz RMS (rad): P24 band rule (linear detrend, Hann window, one-sided PSD, band power -> RMS), per leg
  joint, mean over the 12 leg joints; also 1-4 Hz (gait band) for contrast; and the same 4-25 Hz RMS on the targets qt.
  python -I neg30_pdstats.py OUT.md SPECS_GLOB RUN_DIR CELL [POLICY ...]"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

DT = 0.02


def bands(x, f, lo, hi):
  if x.shape[0] % 2:
    x = x[1:]
  n = x.shape[0]
  t = np.arange(n)
  A = np.stack([np.ones(n), t - t.mean()], axis=1)
  coef, *_ = np.linalg.lstsq(A, x, rcond=None)
  x = x - A @ coef
  w = np.hanning(n)
  X = np.fft.rfft(x * w, axis=0)
  fr = np.fft.rfftfreq(n, 1.0 / f)
  c = np.full(len(fr), 2.0)
  c[0] = 1.0
  c[-1] = 1.0
  P = c * np.abs(X) ** 2 / (n * (w ** 2).sum())
  m = (fr >= lo) & (fr < hi)
  return float(np.sqrt(P[m].sum()))


def main():
  out, spec_glob, run, cell = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
  pols = sys.argv[5:]
  specs = {}
  for fn in glob.glob(spec_glob):
    for l in open(fn):
      if l.strip():
        s = json.loads(l)
        specs[s["spec_sha256"]] = s
  rows = [json.loads(l) for l in (run / "rows.jsonl").read_text().splitlines() if l.strip()]
  rows = [r for r in rows if "error" not in r and r.get("family") == "walk_stop" and r.get("cell") == cell and not r.get("F")]
  res = {}
  for r in rows:
    if pols and r["policy"] not in pols:
      continue
    z = np.load(run / "npz" / r["policy"] / f"{r['spec_sha256'][:16]}.npz")
    if "qt" not in z.files:
      continue
    names = json.loads(str(z["meta"]))["names"]
    t0 = float(specs[r["spec_sha256"]]["mission"]["settle_s"])
    k0, k1 = int((t0 + 3) / DT), int((t0 + 13) / DT)
    qt, q = z["qt"][k0:k1].astype(float), z["q"][k0:k1].astype(float)
    dqt = np.abs(np.diff(qt, axis=0))
    leg = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
    waist = [i for i, n in enumerate(names) if "waist" in n]
    arm = [i for i, n in enumerate(names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    d = dict(dqt_mean=float(dqt.mean()), dqt_p95=float(np.percentile(dqt, 95)), dqt_leg=float(dqt[:, leg].mean()),
             dqt_waist=float(dqt[:, waist].mean()), dqt_arm=float(dqt[:, arm].mean()),
             dqt_leg_p95=float(np.percentile(dqt[:, leg], 95)),
             q_leg_4_25=float(np.mean([bands(q[:, j], 1 / DT, 4.0, 25.0) for j in leg])),
             q_leg_1_4=float(np.mean([bands(q[:, j], 1 / DT, 1.0, 4.0) for j in leg])),
             qt_leg_4_25=float(np.mean([bands(qt[:, j], 1 / DT, 4.0, 25.0) for j in leg])),
             qt_leg_1_4=float(np.mean([bands(qt[:, j], 1 / DT, 1.0, 4.0) for j in leg])),
             trk_err_leg=float(np.sqrt(((qt[:, leg] - q[:, leg]) ** 2).mean())))
    res.setdefault(r["policy"], []).append(d)
  keys = ["dqt_mean", "dqt_p95", "dqt_leg", "dqt_leg_p95", "dqt_waist", "dqt_arm", "q_leg_4_25", "q_leg_1_4", "qt_leg_4_25",
          "qt_leg_1_4", "trk_err_leg"]
  lab = {"dqt_mean": "|Δqt| 평균 rad/스텝", "dqt_p95": "|Δqt| p95", "dqt_leg": "다리 |Δqt| 평균", "dqt_leg_p95": "다리 |Δqt| p95",
         "dqt_waist": "허리 |Δqt| 평균", "dqt_arm": "팔 |Δqt| 평균", "q_leg_4_25": "다리 관절각 4–25 Hz RMS rad",
         "q_leg_1_4": "다리 관절각 1–4 Hz RMS", "qt_leg_4_25": "다리 표적 4–25 Hz RMS", "qt_leg_1_4": "다리 표적 1–4 Hz RMS",
         "trk_err_leg": "다리 표적−관절 RMS"}
  ps = sorted(res, key=lambda p: (p != "mf", p))
  L = [f"# PD 표적 변화 통계 (앞 0.5 m/s, 창 [t0+3, t0+13] s, 칸 {cell}; 평가 전용)", "",
       "| 값 | " + " | ".join(f"{p} (n {len(res[p])})" for p in ps) + " |", "|---|" + "---|" * len(ps)]
  for k in keys:
    cells = []
    for p in ps:
      v = [d[k] for d in res[p]]
      cells.append(f"{np.mean(v):.4f}" + (f" ± {np.std(v, ddof=1):.4f}" if len(v) > 1 else ""))
    L.append(f"| {lab[k]} | " + " | ".join(cells) + " |")
  if "mf" in res:
    L += ["", "제조사 대비 배수(평균):"]
    for p in ps:
      if p == "mf":
        continue
      L.append(f"- {p}: " + ", ".join(f"{lab[k]} ×{np.mean([d[k] for d in res[p]]) / max(np.mean([d[k] for d in res['mf']]), 1e-12):.2f}"
                                     for k in ("dqt_leg", "q_leg_4_25", "qt_leg_4_25", "q_leg_1_4")))
  Path(out).write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
