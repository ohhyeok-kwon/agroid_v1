"""Positive 24 E3 calm reference part 2 readout (rounds/R04/positive/e3_calm_ref/rules_cmd.md).

  python p24_cmd_readout.py <out.json> <out.md> manufacturer=<npz> e114direct=<npz> e114dm=<npz>
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

UP, DOWN, AIR = 0.015, 0.005, 0.010
SMOOTH = 25                                  # 0.5 s centred moving average for the step response


def landing_idx(z):
  idx, up = [], z[0] > UP
  for k in range(1, len(z)):
    if up and z[k] < DOWN:
      idx.append(k)
      up = False
    elif not up and z[k] > UP:
      up = True
  return idx


def yaw_of(q):
  return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def roll_pitch(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  return roll, pitch


def smooth(x):
  k = np.ones(SMOOTH) / SMOOTH
  return np.convolve(x, k, mode="same")


def analyse(path):
  z = np.load(path)
  meta = json.loads(str(z["meta"]))
  dt = meta["dt"]
  T = z["pos"].shape[0]
  t = (np.arange(T) + 1) * dt
  names = meta["joint_names"]
  arm = [i for i, nm in enumerate(names) if any(k in nm for k in ("shoulder", "elbow", "wrist"))]
  sp = [names.index("left_shoulder_pitch_joint"), names.index("right_shoulder_pitch_joint")]
  t0, ts = meta["t0"], meta["t_stop"]
  yaw = yaw_of(z["quat"])                                    # (T, n)
  v = z["v"]
  fwd = v[..., 0] * np.cos(yaw) + v[..., 1] * np.sin(yaw)
  lat = -v[..., 0] * np.sin(yaw) + v[..., 1] * np.cos(yaw)
  wz = z["w"][..., 2]
  out = dict(meta={k: meta.get(k) for k in ("subject", "wall_s", "driver_sha256", "weights_sha256", "selection_sha256", "steer",
                                             "s2_period_steps")})
  # step responses
  resp = {}
  for i, (label, sig, cmd) in {0: ("forward 0.5", fwd[:, 0], 0.5), 1: ("lateral 0.3", lat[:, 1], 0.3), 2: ("turn 0.8", wz[:, 2], 0.8),
                               4: ("forward 0.3", fwd[:, 4], 0.3)}.items():
    s = smooth(sig)
    after = t >= t0 - 1e-9
    win = after & (t <= t0 + 8.0 + 1e-9)
    def first(level):
      k = np.nonzero(after & (s >= level * cmd))[0]
      return float(t[k[0]] - t0) if len(k) else float("nan")
    steady = (t >= t0 + 3.0) & (t <= t0 + 8.0)
    resp[label] = dict(t50_s=first(0.5), t90_s=first(0.9), overshoot_pct=max(0.0, float((s[win].max() / cmd - 1.0) * 100.0)),
                       steady_ratio=float(sig[steady].mean() / cmd))
  out["step"] = resp
  # stop (robot 3)
  sp3 = np.linalg.norm(v[:, 3, :2], axis=1)
  idx = np.nonzero(t >= ts - 1e-9)[0]
  rec = float("nan")
  for k in idx:
    if k + 25 <= T and (sp3[k:k + 25] < 0.10).all():
      rec = float(t[k] - ts)
      break
  end = ts + (rec if not math.isnan(rec) else SECONDS_LEFT(t, ts))
  w = (t >= ts - 1e-9) & (t <= end + 1e-9)
  nst = sum(len([k for k in landing_idx(z["sole_z"][w, 3, f])]) for f in range(2))
  out["stop"] = dict(time_to_still_s=rec, steps_to_still=int(nst), speed_at_stop=float(sp3[np.nonzero(t >= ts - 1e-9)[0][0]]))
  # walking appearance (robot 4 = 0.3, robot 0 = 0.5), window [4, 14] s
  look = {}
  for i, label in ((4, "0.3"), (0, "0.5")):
    W = (t >= 4.0 - 1e-9) & (t <= 14.0 + 1e-9)
    q, dq = z["q"][W, i], z["dq"][W, i]
    p = z["pos"][W, i]
    yw = yaw[W, i]
    mh = math.atan2(np.sin(yw).mean(), np.cos(yw).mean())
    rel = p[:, :2] - p[0, :2]
    lateral = -rel[:, 0] * math.sin(mh) + rel[:, 1] * math.cos(mh)
    tt = t[W]
    lat_d = lateral - np.polyval(np.polyfit(tt, lateral, 1), tt)
    z_d = p[:, 2] - np.polyval(np.polyfit(tt, p[:, 2], 1), tt)
    roll, pitch = roll_pitch(z["torso_quat"][W, i])
    so = z["sole_z"][W, i]
    li = [landing_idx(so[:, f]) for f in range(2)]
    apex = []
    for f in range(2):
      for a, b in zip(li[f][:-1], li[f][1:]):
        apex.append(float(so[a:b, f].max()))
    look[label] = dict(arm_swing_shoulder_pitch_range_rad=float(np.mean([q[:, j].max() - q[:, j].min() for j in sp])),
                       arm_qd_rms=float(np.sqrt((dq[:, arm] ** 2).mean())),
                       pelvis_lateral_pp_m=float(lat_d.max() - lat_d.min()), pelvis_lateral_rms_m=float(lat_d.std()),
                       pelvis_vertical_pp_m=float(z_d.max() - z_d.min()), pelvis_vertical_rms_m=float(z_d.std()),
                       torso_roll_rms_deg=float(np.degrees((roll - roll.mean()).std())),
                       torso_pitch_rms_deg=float(np.degrees((pitch - pitch.mean()).std())),
                       step_freq_hz=float((len(li[0]) + len(li[1])) / (tt[-1] - tt[0])),
                       foot_lift_m=float(np.median(apex)) if apex else float("nan"),
                       two_feet_off_share=float(((so[:, 0] > AIR) & (so[:, 1] > AIR)).mean()),
                       realised_fwd_mps=float(fwd[W, i].mean()))
  out["walk"] = look
  return out


def SECONDS_LEFT(t, ts):
  return float(t[-1] - ts)


def main(argv):
  res = {}
  for a in argv[2:]:
    k, v = a.split("=", 1)
    res[k] = analyse(v)
  json.dump(res, open(argv[0], "w"), indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
  subs = list(res)
  f = lambda x, nd: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"      # noqa: E731
  L = ["### 명령 계단 응답 (0.5 s 이동 평균; 시간은 명령 뒤 초)", "| 명령 | 값 | " + " | ".join(subs) + " |", "|---|---|" + "---|" * len(subs)]
  for lab in ("forward 0.5", "lateral 0.3", "turn 0.8", "forward 0.3"):
    for k, name, nd in (("t50_s", "50 % 도달 (s)", 2), ("t90_s", "90 % 도달 (s)", 2), ("overshoot_pct", "넘침 (%)", 1),
                        ("steady_ratio", "정상 실현/명령 (3–8 s)", 2)):
      L.append(f"| {lab} | {name} | " + " | ".join(f(res[s]['step'][lab][k], nd) for s in subs) + " |")
  L.append(f"| 정지(앞 0.5 → 0) | 0.10 m/s 아래 0.5 s 머물기까지 (s) | " + " | ".join(f(res[s]['stop']['time_to_still_s'], 2) for s in subs) + " |")
  L.append(f"| 정지 | 그때까지 디딤 수 | " + " | ".join(str(res[s]['stop']['steps_to_still']) for s in subs) + " |")
  L.append("\n### 걷기 모습 (앞 0.3·0.5 m/s, 창 [4, 14] s)")
  L.append("| 속도 | 값 | " + " | ".join(subs) + " |")
  L.append("|---|---|" + "---|" * len(subs))
  keys = (("arm_swing_shoulder_pitch_range_rad", "팔 흔들기 폭: 어깨 pitch 범위 (rad)", 3), ("arm_qd_rms", "팔 관절 속도 RMS (rad/s)", 2),
          ("pelvis_lateral_pp_m", "골반 좌우 폭 p-p (m)", 3), ("pelvis_vertical_pp_m", "골반 상하 폭 p-p (m)", 3),
          ("torso_roll_rms_deg", "몸통 roll RMS (°)", 2), ("torso_pitch_rms_deg", "몸통 pitch RMS (°)", 2),
          ("step_freq_hz", "걸음 빈도 (착지/s)", 2), ("foot_lift_m", "발 들기 높이 (m)", 3), ("two_feet_off_share", "두 발 이륙 몫", 3),
          ("realised_fwd_mps", "실현 앞 속도 (m/s)", 3))
  for sp_ in ("0.3", "0.5"):
    for k, name, nd in keys:
      L.append(f"| {sp_} | {name} | " + " | ".join(f(res[s]['walk'][sp_][k], nd) for s in subs) + " |")
  open(argv[1], "w").write("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main(sys.argv[1:])
