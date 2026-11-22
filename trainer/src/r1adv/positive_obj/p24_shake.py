"""Positive 24 (E1 c09) base shake profiles (rounds/R04/positive/e1_c09/rules.md).

The robot base (pelvis) is fixed to a mocap body. A vertical base motion z_b(t) is applied as the equivalent uniform
inertial load: in the base frame every body feels gravity g_eff(t) = -(9.81 + a_b(t)) z, a_b = z_b''. For a fixed base
this is exact (pure translation; contact forces and relative velocities do not depend on the frame) and it avoids the
mocap-teleport artefact (a moved mocap body has zero velocity in the contact Jacobian).

Panels (coordinator 16:1x, 16_rounds.md 'E1 c09 흔들림 측정 보정'):
  trace   main panel: positive 23 pelvis z trace (e1_c09_shake_trace.npz, 200 Hz = the hold sim step), env whose z RMS is
          the median of the recorded envs (lower median for an even count), the 3 s window (start on a 0.5 s grid) whose
          mean-removed z RMS is the median over the windows. a_k = (z_{k+1} - 2 z_k + z_{k-1}) / dt^2 (exactly reproduces
          z under semi-implicit Euler; the base starts the window at rest: a constant velocity offset is Galilean).
  sin493  side panel: a(t) = 4.93 sin(2 pi 1.1 t) m/s^2 (sel01 literal, p95 |a_z| of e1_c09_shake.json)
  sin196  side panel: a(t) = 0.2 g sin(2 pi 1.1 t) (c09 original assumption)

  python -m r1adv.positive_obj.p24_shake <trace.npz> <out.npz> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

DT = 0.005
T_SHAKE = 3.0
N_SHAKE = int(round(T_SHAKE / DT))           # 600
WIN_STEP = 0.5
F_SIN = 1.1
A_P95 = 4.93
G = 9.81


def pick_env(z: np.ndarray, envs: np.ndarray) -> tuple[int, list]:
  rms = z.std(0)
  order = np.argsort(rms, kind="stable")
  k = order[(len(order) - 1) // 2]            # lower median for an even count
  return int(k), [(int(envs[i]), float(rms[i])) for i in order]


def pick_window(zc: np.ndarray, t: np.ndarray) -> tuple[int, list]:
  starts = np.arange(0.0, t[-1] - T_SHAKE + 1e-9, WIN_STEP)
  rows = []
  for s in starts:
    i0 = int(round(s / DT))
    seg = zc[i0:i0 + N_SHAKE + 2]
    if len(seg) < N_SHAKE + 2:
      continue
    rows.append((i0, float(seg[1:-1].std())))
  vals = np.array([r[1] for r in rows])
  order = np.argsort(vals, kind="stable")
  k = order[(len(order) - 1) // 2]
  return rows[k][0], rows


def profiles(trace_path: str) -> tuple[dict, dict]:
  zf = np.load(trace_path)
  z, t, envs = zf["z"].astype(np.float64), zf["t"].astype(np.float64), zf["envs"]
  assert abs(float(zf["rate_hz"]) - 1.0 / DT) < 1e-9
  ei, env_order = pick_env(z, envs)
  zc = z[:, ei] - z[:, ei].mean()
  i0, wins = pick_window(zc, t)
  seg = zc[i0:i0 + N_SHAKE + 2]
  a_tr = (seg[2:] - 2 * seg[1:-1] + seg[:-2]) / DT ** 2
  tt = np.arange(N_SHAKE) * DT
  a_493 = A_P95 * np.sin(2 * np.pi * F_SIN * tt)
  a_196 = 0.2 * G * np.sin(2 * np.pi * F_SIN * tt)
  prof = {"trace": a_tr, "sin493": a_493, "sin196": a_196}
  # base displacement implied by each profile (semi-implicit Euler from rest)
  disp = {}
  for k, a in prof.items():
    v = np.cumsum(a) * DT
    x = np.cumsum(v) * DT
    disp[k] = dict(a_rms=float(a.std()), a_abs_p95=float(np.percentile(np.abs(a), 95)), a_abs_max=float(np.abs(a).max()),
                   x_pp_mm=float(np.ptp(x - np.polyval(np.polyfit(tt, x, 1), tt)) * 1000.0))
  meta = dict(trace=trace_path, trace_sha256=hashlib.sha256(open(trace_path, "rb").read()).hexdigest(),
              env_order_by_z_rms=env_order, env=int(envs[ei]), env_col=ei,
              window_start_index=i0, window_start_s=float(t[i0 + 1]), window_end_s=float(t[i0 + N_SHAKE]),
              window_rows=[(int(a), b) for a, b in wins], window_z_rms_mm=float(seg[1:-1].std() * 1000.0),
              env_z_rms_mm=float(zc.std() * 1000.0), seg_z_pp_mm=float(np.ptp(seg[1:-1]) * 1000.0), stats=disp,
              rule="env: lower median of z RMS; window: 3 s, starts on a 0.5 s grid, median mean-removed z RMS; "
                   "a_k = second difference of z / dt^2")
  return prof, meta


if __name__ == "__main__":
  prof, meta = profiles(sys.argv[1])
  np.savez(sys.argv[2], **prof)
  meta["npz_sha256"] = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
  json.dump(meta, open(sys.argv[3], "w"), indent=1)
  print(json.dumps({k: v for k, v in meta.items() if k not in ("window_rows",)}, indent=1))
