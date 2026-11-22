"""Negative 30: '바른자세' posture card (rounds/R04/posture_card_v1.md, S1-S5 / W1-W5) from neg30_pass.py npz records.

Windows: standing = stand_push [2, 20] s (planner idle reference; manufacturer: zero command) and stand_static [3, 10] s
(symmetric static reference); walking 0.5 = walk_stop [t0+3, t0+13]; walking 0.3 = moves 'fwd0.3' segment from +2 s to its
end. Per episode (time means over the window, degrees unless noted):
  S1/W1 torso roll, pitch mean (|.|, max of the two)      S2/W2 waist roll, yaw mean (|.|, max)
  S3 leg mirror asymmetry |mean q_L - M(mean q_R)| per joint type (hip pitch / roll / yaw, knee, ankle pitch / roll; M flips
     roll / yaw), max over types                           S4/W3 arm |mean q - default| per joint (rad), max over 10 joints
  S5 foot placement in the pelvis heading frame: max(|y_L + y_R|, |x_L - x_R|) (m) and toe angle |psi_L + psi_R|
  W4 left / right step length and step time ratios (|ratio - 1|, max of the two) and swing apex height ratio; gait events
     from the vertical foot force (off = Fz < 5 N for >= 2 ticks), so low-clearance (shuffling) gaits are counted
  W5 heading drift |yaw change| / distance travelled (deg / m)
Thresholds: max(1.5 x manufacturer median, card floor) per view type (standing / walk 0.5 / walk 0.3; the manufacturer
has no reference input, so the symmetric static reference view uses its standing values); a checkpoint passes an item when
>= 90 % of its episodes do.

  python -I neg30_posture.py OUT.md OUT.json SPECS_GLOB RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DT = 0.02
UP, DOWN = 0.015, 0.005
LEGJ = (("hip_pitch", 1), ("hip_roll", -1), ("hip_yaw", -1), ("knee", 1), ("ankle_pitch", 1), ("ankle_roll", -1))
ARMJ = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
DEFAULT = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.8,
           "left_wrist_roll_joint": 0.0, "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2,
           "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.8, "right_wrist_roll_joint": 0.0}   # R1 contract default
FLOOR = {"S1": 0.5, "S2": 0.5, "S3": 1.0, "S4": 0.10, "S5a": 0.01, "S5b": 3.0, "W1": 0.5, "W2": 0.5, "W3": 0.15, "W4a": 0.05,
         "W4b": 0.20, "W5": 2.0}
NAMES = {"S1": "서기 몸통 roll·pitch 평균 치우침 (°)", "S2": "서기 허리 roll·yaw 평균 치우침 (°)", "S3": "서기 다리 거울 비대칭 최대 (°)",
         "S4": "서기 팔 기본 대비 평균 편차 최대 (rad)", "S5a": "서기 발 자리 비대칭 (m)", "S5b": "서기 발끝 각 비대칭 (°)",
         "W1": "걷기 몸통 roll·pitch 평균 치우침 (°)", "W2": "걷기 허리 roll·yaw 평균 치우침 (°)", "W3": "걷기 팔 기본 대비 평균 편차 최대 (rad)",
         "W4a": "걷기 좌우 걸음 길이·시간 비 |비−1|", "W4b": "걷기 좌우 흔드는 발 높이 비 |비−1|", "W5": "곧게 걷기 방향 치우침 (°/m)"}


def roll_pitch(q):
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))), np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))


def yaw_of(q):
  return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def landing_idx(z):
  idx, up = [], z[0] > UP
  for k in range(1, len(z)):
    if up and z[k] < DOWN:
      idx.append(k)
      up = False
    elif not up and z[k] > UP:
      up = True
  return idx


def stand_card(z, names, k0, k1):
  ix = {n: i for i, n in enumerate(names)}
  q = z["q"][k0:k1].astype(float)
  qm = q.mean(0)
  r, p = roll_pitch(z["torso_quat"][k0:k1].astype(float))
  out = dict(S1=max(abs(r.mean()), abs(p.mean())), S1_roll=float(r.mean()), S1_pitch=float(p.mean()))
  wr, wy = math.degrees(qm[ix["waist_roll_joint"]]), math.degrees(qm[ix["waist_yaw_joint"]])
  out.update(S2=max(abs(wr), abs(wy)), S2_roll=wr, S2_yaw=wy)
  leg = {}
  for j, sg in LEGJ:
    leg[j] = abs(math.degrees(qm[ix[f"left_{j}_joint"]] - sg * qm[ix[f"right_{j}_joint"]]))
  jw = max(leg, key=leg.get)
  out.update(S3=leg[jw], S3_joint=jw, S3_all=leg)
  arm = {n: abs(float(qm[ix[n]]) - DEFAULT[n]) for n in DEFAULT}
  aw = max(arm, key=arm.get)
  out.update(S4=arm[aw], S4_joint=aw)
  so = z["sole"][k0:k1].astype(float)                        # (n, 2, 4, 3)
  pos, quat = z["pos"][k0:k1].astype(float), z["quat"][k0:k1].astype(float)
  yw = yaw_of(quat)
  c, s = np.cos(yw), np.sin(yw)
  ctr = so[..., :2].mean(2)                                   # (n, 2, 2)
  rel = ctr - pos[:, None, :2]
  xr = c[:, None] * rel[..., 0] + s[:, None] * rel[..., 1]
  yr = -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
  xm, ym = xr.mean(0), yr.mean(0)
  heel, toe = so[:, :, 0:2, :2].mean(2), so[:, :, 2:4, :2].mean(2)
  fy = np.arctan2(toe[..., 1] - heel[..., 1], toe[..., 0] - heel[..., 0]) - yw[:, None]
  fy = (fy + np.pi) % (2 * np.pi) - np.pi
  psi = np.degrees(fy.mean(0))
  out.update(S5a=max(abs(ym[0] + ym[1]), abs(xm[0] - xm[1])), S5b=abs(psi[0] + psi[1]), S5_xy=[float(xm[0]), float(ym[0]), float(xm[1]), float(ym[1])],
             S5_psi=[float(psi[0]), float(psi[1])])
  return out


def walk_card(z, names, k0, k1):
  ix = {n: i for i, n in enumerate(names)}
  q = z["q"][k0:k1].astype(float)
  qm = q.mean(0)
  r, p = roll_pitch(z["torso_quat"][k0:k1].astype(float))
  out = dict(W1=max(abs(r.mean()), abs(p.mean())), W1_roll=float(r.mean()), W1_pitch=float(p.mean()))
  wr, wy = math.degrees(qm[ix["waist_roll_joint"]]), math.degrees(qm[ix["waist_yaw_joint"]])
  out.update(W2=max(abs(wr), abs(wy)), W2_roll=wr, W2_yaw=wy)
  arm = {n: abs(float(qm[ix[n]]) - DEFAULT[n]) for n in DEFAULT}
  aw = max(arm, key=arm.get)
  out.update(W3=arm[aw], W3_joint=aw)
  so = z["sole"][k0:k1].astype(float)
  sz = so[..., 2].min(2)
  ctr = so[..., :2].mean(2)
  pos, quat = z["pos"][k0:k1].astype(float), z["quat"][k0:k1].astype(float)
  yw = np.unwrap(yaw_of(quat))
  mh = math.atan2(np.sin(yw).mean(), np.cos(yw).mean())
  u = np.array([math.cos(mh), math.sin(mh)])
  # gait events from the vertical foot force (buridan rule: off = Fz < 5 N for >= 2 ticks), so low-clearance gaits count
  fz = z["force"][k0:k1, :, 2].astype(float)
  td, swings = {0: [], 1: []}, {0: [], 1: []}
  for f in range(2):
    off = fz[:, f] < 5.0
    k = 0
    while k < len(off):
      if off[k]:
        j = k
        while j < len(off) and off[j]:
          j += 1
        if j - k >= 2 and j < len(off) and k > 0:
          td[f].append(j)
          swings[f].append(float(sz[k:j, f].max()))
        k = j
      else:
        k += 1
  ev = sorted([(k, f) for f in range(2) for k in td[f]])
  L = {0: [], 1: []}
  Tm = {0: [], 1: []}
  for (k_prev, f_prev), (k, f) in zip(ev[:-1], ev[1:]):
    if f == f_prev:
      continue
    L[f].append(float((ctr[k, f] - ctr[k, f_prev]) @ u))
    Tm[f].append((k - k_prev) * DT)
  apex = swings
  if min(len(L[0]), len(L[1])) >= 2 and np.mean(L[1]) > 1e-3 and np.mean(Tm[1]) > 0:
    rl, rt = np.mean(L[0]) / np.mean(L[1]), np.mean(Tm[0]) / np.mean(Tm[1])
    out.update(W4a=max(abs(rl - 1), abs(rt - 1)), W4_len=[float(np.mean(L[0])), float(np.mean(L[1]))], W4_time=[float(np.mean(Tm[0])), float(np.mean(Tm[1]))])
  else:
    out.update(W4a=None, W4_steps=[len(L[0]), len(L[1])])
  if min(len(apex[0]), len(apex[1])) >= 2 and np.mean(apex[1]) > 1e-3:
    out.update(W4b=abs(np.mean(apex[0]) / np.mean(apex[1]) - 1), W4_apex=[float(np.mean(apex[0])), float(np.mean(apex[1]))])
  else:
    out.update(W4b=None)
  dist = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
  out.update(W5=(abs(math.degrees(yw[-1] - yw[0])) / dist) if dist > 0.3 else None, W5_dist=dist)
  return out


def episodes(spec_glob, dirs):
  specs = {}
  for fn in glob.glob(spec_glob):
    for l in open(fn):
      if l.strip():
        s = json.loads(l)
        specs[s["spec_sha256"]] = s
  out = []
  for d in dirs:
    p = Path(d) / "rows.jsonl"
    if not p.exists():
      continue
    seen = set()
    for l in p.read_text().splitlines():
      r = json.loads(l)
      if "error" in r or (r["spec_sha256"], r["policy"]) in seen:
        continue
      seen.add((r["spec_sha256"], r["policy"]))
      sp = specs.get(r["spec_sha256"])
      npz = Path(d) / "npz" / r["policy"] / f"{r['spec_sha256'][:16]}.npz"
      if sp is None or not npz.exists():
        continue
      out.append((sp, r, npz))
  return out


def card_rows(spec_glob, dirs):
  res = defaultdict(lambda: defaultdict(list))                # (policy, view) -> key -> per-episode values
  for sp, r, npz in episodes(spec_glob, dirs):
    if r.get("F"):
      continue
    fam, cell, pol = sp["family"], sp.get("cell", ""), r["policy"]
    z = np.load(npz)
    names = json.loads(str(z["meta"]))["names"]
    n = len(z["pos"])
    if fam == "stand_push":
      c = stand_card(z, names, int(2.0 / DT), min(n, int(20.0 / DT)))
      view = f"stand_idle|{cell}"
    elif fam == "stand_static":
      c = stand_card(z, names, int(3.0 / DT), n)
      view = f"stand_static|{cell}"
    elif fam == "walk_stop":
      t0 = float(sp["mission"]["settle_s"])
      if not r.get("G_walk"):
        res[(pol, f"walk05|{cell}")]["noG"].append(1)
        continue
      c = walk_card(z, names, int((t0 + 3) / DT), min(n, int((t0 + 13) / DT)))
      view = f"walk05|{cell}"
    elif fam == "moves":
      seg = next((s for s in r.get("segments", []) if s.get("label") == "fwd0.3"), None)
      if seg is None or seg.get("realised") is None or seg["realised"] < 0.7:
        res[(pol, f"walk03|{cell}")]["noG"].append(1)
        continue
      c = walk_card(z, names, int((2.0 + 2.0) / DT), int(8.0 / DT))
      view = f"walk03|{cell}"
    else:
      continue
    for k, v in c.items():
      res[(pol, view)][k].append(v)
  return res


def main():
  out_md, out_json, spec_glob, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
  res = card_rows(spec_glob, dirs)
  pols = sorted({p for p, _v in res})
  views = sorted({v for _p, v in res})
  # thresholds from the manufacturer (same loop, same windows): its standing = stand_idle, walking = walk05 / walk03
  # thresholds per view type (stand_idle; stand_static uses the manufacturer's stand_idle, it has no reference input;
  # walk05; walk03), from the manufacturer's episodes in the primary cell (L01_d00) of the run dirs given
  THR = {}
  for vt in ("stand_idle", "walk05", "walk03"):
    for key in FLOOR:
      if key.startswith("S") != (vt == "stand_idle"):
        continue
      vals = []
      for (p, v), d in res.items():
        if p == "mf" and v.startswith(vt + "|") and "L01_d00" in v:
          vals += [x for x in d.get(key, []) if x is not None]
      mfv = float(np.median(vals)) if vals else None
      THR[(vt, key)] = dict(mf=mfv, thr=max(FLOOR[key], 1.5 * mfv) if mfv is not None else FLOOR[key], floor=FLOOR[key])
  thr_of = lambda view, key: THR[("stand_idle" if view.startswith("stand") else view.split("|")[0], key)]  # noqa: E731
  thr = {f"{vt}:{k}": t for (vt, k), t in THR.items()}
  L = ["| 칸 | 바닥 | 제조사 중앙 (같은 판) | 문턱 = max(1.5 × 제조사, 바닥) |", "|---|---|---|---|"]
  for (vt, k), t in THR.items():
    mfs = "—" if t["mf"] is None else f"{t['mf']:.3f}"
    L.append(f"| {vt} {k} {NAMES[k]} | {t['floor']} | {mfs} | {t['thr']:.3f} |")
  summary = {}
  for view in views:
    keys = [k for k in FLOOR if (k.startswith("S") and view.startswith("stand")) or (k.startswith("W") and view.startswith("walk"))]
    L += ["", f"### {view}", "| 정책 | n | " + " | ".join(keys) + " | 바른자세 칸 통과 |", "|---|---|" + "---|" * (len(keys) + 1)]
    for p in pols:
      d = res.get((p, view))
      if not d:
        continue
      cells, npass = [], 0
      nn = max((len(d.get(k, [])) for k in keys), default=0)
      for k in keys:
        xs = d.get(k, [])
        ok = [x is not None and x <= thr_of(view, k)["thr"] for x in xs]
        sh = sum(ok) / len(ok) if ok else None
        vv = [x for x in xs if x is not None]
        m = float(np.median(vv)) if vv else None
        if sh is not None and sh >= 0.9:
          npass += 1
        cells.append("—" if sh is None else f"{'✓' if sh >= 0.9 else '✗'} {m:.3f}" if m is not None else "✗ 없음")
        summary.setdefault(p, {}).setdefault(view, {})[k] = dict(median=m, share=sh, n=len(xs))
      extra = f" (G 없음 {len(d.get('noG', []))})" if d.get("noG") else ""
      L.append(f"| {p} | {nn}{extra} | " + " | ".join(cells) + f" | {npass}/{len(keys)} |")
  # worst-joint detail for S3 / S4 / W3 (most frequent)
  L += ["", "### 가장 큰 치우침이 나온 관절 (판 최빈값)"]
  for (p, v), d in sorted(res.items()):
    det = []
    for k in ("S3_joint", "S4_joint", "W3_joint"):
      if d.get(k):
        vals, cnt = np.unique(d[k], return_counts=True)
        det.append(f"{k[:2]} {vals[cnt.argmax()]}")
    for k in ("S1_roll", "S1_pitch", "S2_roll", "S2_yaw", "W1_roll", "W1_pitch", "W2_roll", "W2_yaw"):
      if d.get(k):
        det.append(f"{k} {np.median(d[k]):+.2f}")
    if det:
      L.append(f"- {p} {v}: " + ", ".join(det))
  Path(out_md).write_text("\n".join(L) + "\n")
  json.dump(dict(thresholds=thr, summary=summary), open(out_json, "w"), indent=1, default=str)
  print("\n".join(L))


if __name__ == "__main__":
  main()
