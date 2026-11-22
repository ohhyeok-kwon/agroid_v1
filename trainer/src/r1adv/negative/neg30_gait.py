"""Negative 30 (coordinator 12:12): foot placement while walking (W6 candidate) and reference quality.

Robot gait (neg30_pass.py walk_stop npz, window [t0 + 3, t0 + 13] s; heading frame = time-mean pelvis heading of the window):
  stance spans   per foot, vertical contact force Fz > 5 N for >= 3 ticks (50 Hz)
  footprint      foot centre (mean of the 4 sole corners) averaged over the middle third of each stance span
  step width     |lateral(footprint_k) - lateral(footprint_k-1)| for consecutive footprints of alternating feet (median, m)
  passing gap    at the middle of each stance span, the lateral distance between the stance foot and the other (swinging)
                 foot centre (median and min, m): legs drawn together or crossing make it small / negative
  midline cross  footprint lateral offset from the pelvis path (linear fit of pelvis lateral position) on the wrong side:
                 left foot must be left (+), right foot right (-); crossing = max(0, -side x offset) (max, m) and the share of
                 footprints on the wrong side
  steps / s      touchdowns (stance starts) per second, both feet
Reference quality (m1c reference = G1 planner cand + B2 + whole-stream M1 conversion, open loop; P29 kin_baseline_g1m1.npz,
30 Hz R1 qpos, R1 FK = rp1_fk.R1FK): per command (fwd 0.5 / fwd 0.3 / side 0.3 / back 0.3 / turn 0.8), window command start
+ 2 s .. end - 0.5 s: swing peak height per swing (median, cm), stance-foot slip ratio (stance foot centre horizontal speed /
root speed, stance = lowest corner < 5 mm held 3 frames; median), steps / s (touchdowns), step width (as above, from soles).
  python -I neg30_gait.py robot OUT.md SPECS_GLOB RUN_DIR CELL [POLICY ...]
  python -I neg30_gait.py ref OUT.md"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

DT = 0.02


def spans(mask, min_len=3):
  out, k = [], 0
  while k < len(mask):
    if mask[k]:
      j = k
      while j < len(mask) and mask[j]:
        j += 1
      if j - k >= min_len:
        out.append((k, j))
      k = j
    else:
      k += 1
  return out


def yaw_of(q):
  return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def gait(pos, quat, sole, fz, dt):
  """pos (T,3), quat (T,4), sole (T,2,4,3), fz (T,2) or None (then stance = lowest corner < 5 mm)."""
  T = len(pos)
  yw = np.unwrap(yaw_of(quat))
  mh = math.atan2(np.sin(yw).mean(), np.cos(yw).mean())
  lat_ax = np.array([-math.sin(mh), math.cos(mh)])
  fwd_ax = np.array([math.cos(mh), math.sin(mh)])
  cen = sole[..., :2].mean(2)                                 # (T, 2, 2)
  lat = cen @ lat_ax                                          # (T, 2)
  plat = pos[:, :2] @ lat_ax
  tt = np.arange(T) * dt
  fit = np.polyval(np.polyfit(tt, plat, 1), tt)
  if fz is not None:
    st = fz > 5.0
  else:
    hmin = sole[..., 2].min(-1)
    st = hmin < 0.005
  prints, gaps, cross, wrong, n_td = [], [], [], 0, 0
  for f in range(2):
    side = 1.0 if f == 0 else -1.0                            # left foot = +lateral
    for a, b in spans(st[:, f]):
      n_td += 1
      m0, m1 = a + (b - a) // 3, a + 2 * (b - a) // 3 + 1
      fl = float(lat[m0:m1, f].mean())
      mid = (a + b) // 2
      prints.append((a, f, fl, float((cen[m0:m1, f] @ fwd_ax).mean())))
      gaps.append(side * (lat[mid, f] - lat[mid, 1 - f]))
      off = fl - float(fit[m0:m1].mean())
      c = max(0.0, -side * off)
      cross.append(c)
      wrong += int(c > 0)
  prints.sort()
  widths = [abs(p[2] - q[2]) for q, p in zip(prints[:-1], prints[1:]) if p[1] != q[1]]
  dur = (T - 1) * dt
  return dict(step_width=float(np.median(widths)) if widths else None, passing_gap_med=float(np.median(gaps)) if gaps else None,
              passing_gap_min=float(np.min(gaps)) if gaps else None, midline_cross_max=float(max(cross)) if cross else None,
              midline_wrong_share=(wrong / len(cross)) if cross else None, steps_per_s=n_td / dur if dur > 0 else None)


def robot(out, spec_glob, run, cell, pols):
  specs = {}
  for fn in glob.glob(spec_glob):
    for l in open(fn):
      if l.strip():
        s = json.loads(l)
        specs[s["spec_sha256"]] = s
  rows = [json.loads(l) for l in (Path(run) / "rows.jsonl").read_text().splitlines() if l.strip()]
  res = {}
  for r in rows:
    if "error" in r or r.get("family") != "walk_stop" or r.get("cell") != cell or r.get("F"):
      continue
    if pols and r["policy"] not in pols:
      continue
    z = np.load(Path(run) / "npz" / r["policy"] / f"{r['spec_sha256'][:16]}.npz")
    t0 = float(specs[r["spec_sha256"]]["mission"]["settle_s"])
    k0, k1 = int((t0 + 3) / DT), int((t0 + 13) / DT)
    g = gait(z["pos"][k0:k1].astype(float), z["quat"][k0:k1].astype(float), z["sole"][k0:k1].astype(float),
             z["force"][k0:k1, :, 2].astype(float), DT)
    res.setdefault(r["policy"], []).append(g)
  keys = ["step_width", "passing_gap_med", "passing_gap_min", "midline_cross_max", "midline_wrong_share", "steps_per_s"]
  lab = {"step_width": "걸음 폭 (발자국 좌우 거리, m)", "passing_gap_med": "지나칠 때 두 발 좌우 간격 중앙 (m)",
         "passing_gap_min": "지나칠 때 두 발 좌우 간격 최소 (m)", "midline_cross_max": "가운데 선 넘음 최대 (m)",
         "midline_wrong_share": "가운데 선 넘은 발자국 몫", "steps_per_s": "걸음/s (착지)"}
  ps = sorted(res, key=lambda p: (p != "mf", p))
  L = [f"# 걸음 폭·가운데 선 (앞 0.5 m/s, 칸 {cell}, 창 [t0+3, t0+13] s)", "",
       "| 값 | " + " | ".join(f"{p} (n {len(res[p])})" for p in ps) + " |", "|---|" + "---|" * len(ps)]
  for k in keys:
    cells = []
    for p in ps:
      v = [d[k] for d in res[p] if d[k] is not None]
      cells.append("—" if not v else f"{np.median(v):.3f}" + (f" [{np.percentile(v, 10):.3f}–{np.percentile(v, 90):.3f}]" if len(v) > 1 else ""))
    L.append(f"| {lab[k]} | " + " | ".join(cells) + " |")
  Path(out).write_text("\n".join(L) + "\n")
  json.dump(res, open(str(out).replace(".md", ".json"), "w"), indent=1)
  print("\n".join(L))


def ref(out):
  import torch
  sys.path.insert(0, "/rl-agent/.agent/tmp/p29/rp1")
  from rp1_fk import R1FK
  jn = json.load(open("/rl-agent/dataset/r1_planner/rp1_seq30_v1/meta.json"))["joint_names"]
  fk = R1FK(jn)
  z = np.load("/rl-agent/.agent/tmp/p29/rp1/kin_baseline_g1m1.npz")
  dt = 1.0 / 30.0
  a, b = int((1.0 + 2.0) / dt), int((1.0 + 8.0 - 0.5) / dt)     # P29 kin panel timing: idle 1 s, command 8 s, idle 3 s
  L = ["# m1c 참조 품질 (G1 계획기 cand + B2 + 흐름 전체 M1, 열린 고리; P29 kin_baseline_g1m1.npz, 30 Hz)", "",
       "| 명령 | 흔드는 발 최고점 중앙 cm | 받침발 미끄럼 비 중앙 (발 속도 / 뿌리 속도) | 걸음/s | 걸음 폭 m | 지나칠 때 간격 중앙 m |",
       "|---|---|---|---|---|---|"]
  res = {}
  for n in ("fwd05", "fwd03", "side03", "back03", "turn08", "walkturn"):
    if n not in z.files:
      continue
    Q = z[n].astype(float)
    with torch.no_grad():
      o = fk(torch.tensor(Q[:, :3], dtype=torch.float32), torch.tensor(Q[:, 3:7], dtype=torch.float32), torch.tensor(Q[:, 7:], dtype=torch.float32))
    sole = o["sole"].numpy()[a:b]
    hmin = sole[..., 2].min(-1)
    st = hmin < 0.005
    peaks = []
    for f in range(2):
      for s0, s1 in spans(~st[:, f], 2):
        peaks.append(float(hmin[s0:s1, f].max()))
    cen = sole[..., :2].mean(2)
    rv = np.linalg.norm(np.gradient(Q[a:b, :2], dt, axis=0), axis=1)
    fv = np.linalg.norm(np.gradient(cen, dt, axis=0), axis=-1)
    st3 = st.copy()
    st3[1:-1] = st[1:-1] & st[:-2] & st[2:]
    ratio = [fv[k, f] / rv[k] for k in range(len(rv)) for f in range(2) if st3[k, f] and rv[k] > 0.05]
    g = gait(Q[a:b, :3], Q[a:b, 3:7], sole, None, dt)
    res[n] = dict(swing_peak_cm=100 * float(np.median(peaks)) if peaks else None, slip_ratio=float(np.median(ratio)) if ratio else None, **g)
    r_ = res[n]
    f = lambda x, nd=3: "—" if x is None else f"{x:.{nd}f}"  # noqa: E731
    L.append(f"| {n} | {f(r_['swing_peak_cm'], 1)} | {f(r_['slip_ratio'], 2)} | {f(r_['steps_per_s'], 2)} | {f(r_['step_width'])} | {f(r_['passing_gap_med'])} |")
  Path(out).write_text("\n".join(L) + "\n")
  json.dump(res, open(str(out).replace(".md", ".json"), "w"), indent=1)
  print("\n".join(L))


if __name__ == "__main__":
  if sys.argv[1] == "robot":
    robot(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
  else:
    ref(sys.argv[2])
