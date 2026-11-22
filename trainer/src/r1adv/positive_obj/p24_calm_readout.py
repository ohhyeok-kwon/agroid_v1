"""Positive 24 E3 calm reference readout (rounds/R04/positive/e3_calm_ref/rules.md 2-3).

  python p24_calm_readout.py <out.json> <out.md> manufacturer=<npz> b0=<npz> e114=<npz>
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

UP, DOWN = 0.015, 0.005
AIR = 0.010


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


def landings(z):
  """z: (T,) sole minimum height; count down-crossings after being up (hysteresis)."""
  state_up, count = z[0] > UP, 0
  for v in z[1:]:
    if state_up and v < DOWN:
      count += 1
      state_up = False
    elif not state_up and v > UP:
      state_up = True
  return count


def yaw_of(q):
  return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def wrap(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


def analyse(path):
  z = np.load(path)
  meta = json.loads(str(z["meta"]))
  dt = meta["dt"]
  T = z["q"].shape[0]
  t = (np.arange(T) + 1) * dt
  names = meta["joint_names"]
  arm = [i for i, nme in enumerate(names) if any(k in nme for k in ("shoulder", "elbow", "wrist"))]
  default = np.array(meta["default"])
  rows = meta["rows"]
  out = dict(meta={k: meta[k] for k in meta if k in ("policy", "device", "wall_s", "weights_sha256", "b0_sha256", "selection_sha256",
                                                    "driver_sha256", "calm_fraction", "events")})
  # stand (robot 0), window [2, 20]
  W = t >= 2.0 - 1e-9
  r = 0
  sole = z["sole_z"][W, r]                                   # (n, 2)
  steps = landings(sole[:, 0]) + landings(sole[:, 1])
  minutes = W.sum() * dt / 60.0
  p = z["pos"][W, r]
  yaw = yaw_of(z["quat"][W, r])
  dq, tau, q, qt = z["dq"][W, r], z["tau"][W, r], z["q"][W, r], z["qt"][W, r]
  power = sum(bands(tau[:, j] * dq[:, j], 1.0 / dt, 4.0, 25.0) for j in range(dq.shape[1]))
  arm_range = q[:, arm].max(0) - q[:, arm].min(0)
  arm_act = np.sqrt(((qt[:, arm] - default[arm]) ** 2).mean(1)).mean()
  out["stand"] = dict(steps_per_min=steps / minutes, landings=steps,
                      drift_end_m=float(np.linalg.norm(p[-1, :2] - p[0, :2])),
                      drift_max_m=float(np.linalg.norm(p[:, :2] - p[0, :2], axis=1).max()),
                      heading_end_deg=float(abs(np.degrees(wrap(yaw[-1] - yaw[0])))),
                      heading_max_deg=float(np.degrees(np.abs(wrap(yaw - yaw[0]))).max()),
                      two_feet_off_share=float(((sole[:, 0] > AIR) & (sole[:, 1] > AIR)).mean()),
                      power_4_25_W=float(power), arm_range_max_rad=float(arm_range.max()), arm_range_mean_rad=float(arm_range.mean()),
                      arm_action_rms_rad=float(arm_act), arm_qd_rms=float(np.sqrt((dq[:, arm] ** 2).mean())),
                      pelvis_z_mean=float(p[:, 2].mean()))
  # pushes
  k0 = meta["k_push"]
  tp = (k0 + 1) * dt - dt                                    # 4.0 s
  pres = (t >= tp - 0.2 - 1e-9) & (t < tp - 1e-9)
  w1 = (t > tp + 1e-9) & (t <= tp + 1.0 + 1e-9)
  w6 = (t > tp + 1e-9) & (t <= tp + 6.0 + 1e-9)
  after = t > tp + 1e-9
  pushes = []
  for i, row in enumerate(rows):
    if row[0] != "push":
      continue
    qq = z["quat"][:, i]
    up = 1 - 2 * (qq[:, 1] ** 2 + qq[:, 2] ** 2)
    tilt = np.degrees(np.arccos(np.clip(up, -1, 1)))
    fell = bool(((z["pos"][after, i, 2] < 0.45) | (tilt[after] > 60.0)).any())
    dqi = z["dq"][:, i]
    vmax = float(np.abs(dqi[w1]).max())
    vmax_arm = float(np.abs(dqi[w1][:, arm]).max())
    v = np.linalg.norm(z["v"][:, i, :2], axis=1)
    idx = np.nonzero(after)[0]
    rec = float("nan")
    calm = v < 0.10
    for k in idx:
      if k + 25 <= len(v) and calm[k:k + 25].all():
        rec = float(t[k] - tp)
        break
    so = z["sole_z"][w6, i]
    nsteps = landings(so[:, 0]) + landings(so[:, 1])
    qpre = z["q"][pres, i].mean(0)
    moved = int((np.abs(z["q"][w1, i] - qpre).max(0) >= 0.10).sum())
    pushes.append(dict(J=row[1], dir=row[2], fell=fell, max_qd=vmax, max_qd_arm=vmax_arm, recovery_s=rec, steps=nsteps,
                       joints_moved=moved))
  out["push"] = pushes
  summ = {}
  for J in sorted({p_["J"] for p_ in pushes}):
    ps = [p_ for p_ in pushes if p_["J"] == J]
    rec = [p_["recovery_s"] for p_ in ps if not math.isnan(p_["recovery_s"])]
    summ[str(int(J))] = dict(falls=sum(p_["fell"] for p_ in ps), n=len(ps),
                             max_qd_med=float(np.median([p_["max_qd"] for p_ in ps])),
                             max_qd_arm_med=float(np.median([p_["max_qd_arm"] for p_ in ps])),
                             recovery_med=float(np.median(rec)) if rec else float("nan"), unrecovered=len(ps) - len(rec),
                             steps_med=float(np.median([p_["steps"] for p_ in ps])),
                             joints_moved_med=float(np.median([p_["joints_moved"] for p_ in ps])))
  out["push_by_J"] = summ
  return out


def main(argv):
  res = {}
  for a in argv[2:]:
    k, v = a.split("=", 1)
    res[k] = analyse(v)
  json.dump(res, open(argv[0], "w"), indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
  pols = list(res)
  L = ["### 서기 20 s (0 명령, 창 [2, 20] s)", "| 값 | " + " | ".join(pols) + " |", "|---|" + "---|" * len(pols)]
  keys = [("steps_per_min", "디딤/분", 1), ("drift_end_m", "떠밀림 끝 (m)", 3), ("drift_max_m", "떠밀림 최대 (m)", 3),
          ("heading_end_deg", "방향 변화 끝 (°)", 1), ("heading_max_deg", "방향 변화 최대 (°)", 1),
          ("two_feet_off_share", "두 발 이륙 몫", 3), ("power_4_25_W", "4–25 Hz 관절 파워 (W)", 1),
          ("arm_range_max_rad", "팔 관절각 범위 최대 (rad)", 3), ("arm_range_mean_rad", "팔 관절각 범위 평균 (rad)", 3),
          ("arm_action_rms_rad", "팔 행동 크기 |목표−기본| RMS (rad)", 3), ("arm_qd_rms", "팔 |q̇| RMS (rad/s)", 3)]
  for k, label, nd in keys:
    L.append(f"| {label} | " + " | ".join(f"{res[p]['stand'][k]:.{nd}f}" for p in pols) + " |")
  L.append("\n### 밂 (4 방향 중앙값; 넘어짐은 4 판 중 수)")
  L.append("| J (N·s) | 값 | " + " | ".join(pols) + " |")
  L.append("|---|---|" + "---|" * len(pols))
  for J in ("12", "15", "20", "25"):
    for k, label, nd in (("falls", "넘어짐", 0), ("max_qd_med", "1 s 최대 관절 속도 (rad/s)", 1), ("max_qd_arm_med", "팔 최고속 (rad/s)", 1),
                         ("recovery_med", "회복 시간 (s)", 2), ("unrecovered", "미회복 판", 0), ("steps_med", "디딤 수", 1),
                         ("joints_moved_med", "움직인 관절 수", 1)):
      vals = []
      for p in pols:
        v = res[p]["push_by_J"].get(J, {}).get(k)
        vals.append("—" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{nd}f}" if nd else str(int(v))))
      L.append(f"| {J} | {label} | " + " | ".join(vals) + " |")
  open(argv[1], "w").write("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main(sys.argv[1:])
