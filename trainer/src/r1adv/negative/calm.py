"""Calm success of one probe episode (coordinator 12:30, R2 curriculum probe judgment; position drift is NOT a
criterion, so a recovery by stepping counts). Works on runner arrays recorded with --record-feet (without it, a foot's path length from its speed stands in for its
displacement, which is stricter).

An episode is a calm success when all hold:
  1. no physical fall
  2. calm before the event: over the 1 s before the first hit (no-hit episode: the last 1 s) each foot's sole centroid
     moves < 2 cm, the pelvis horizontal speed stays < 0.05 m/s and the yaw rate stays < 0.2 rad/s
  3. recovery after the event: after the last hit, [pelvis horizontal speed < 0.05 m/s and both feet < 0.02 m/s] holds
     for >= 0.5 s before the episode ends (no-hit episode: holds over the last 0.5 s)
Returns a dict with the flag and the measured quantities (so thresholds can be audited).

Joint tremor (coordinator 14:56, rounds/R02/coordinator_decisions_tremor_sensor.md): over the last 4 s, each body
joint's position with its linear trend removed, RMS per joint, the largest over the 24 joints (needs runner
--record-actions). Split by the coordinator's addendum 2 (15:2x) into two bands of the same detrended window:
  vibration  RMS of the content at >= 2 Hz, largest over the joints, <= 1 mrad
  sway       RMS of the content at 0.2-2 Hz, largest over the joints, <= 3 mrad
They are returned (tremor_rad = the unsplit RMS, kept for continuity; vib_rad, sway_rad) but are NOT part of ok (the
level p of hit levels); calm_still() adds both conditions for the base anchor (level-share rule) and the no-hit
teacher-gate cells."""
from __future__ import annotations

import numpy as np

DT = 0.02
PRE_S, HOLD_S = 1.0, 0.5
FOOT_MOVE_M, PELVIS_MPS, YAW_RADPS = 0.02, 0.05, 0.2
STILL_PELVIS, STILL_FOOT = 0.05, 0.02
TREMOR_S, TREMOR_RAD = 4.0, 1e-3
VIB_HZ, SWAY_HZ, VIB_RAD, SWAY_RAD = 2.0, 0.2, 1e-3, 3e-3


def foot_centroids(a: dict, i: int) -> np.ndarray:
  sole = a["sole"][i].reshape(a["sole"].shape[1], 2, 14, 3)
  return sole[..., :2].mean(axis=2)                                  # (T, 2 feet, xy)


def joint_tremor_rad(a: dict, i: int) -> float | None:
  """Largest detrended RMS over the 24 body joints in the last 4 s (rad); None without qpos records."""
  if "qpos" not in a:
    return None
  end = int(a["end_step"][i])
  q = a["qpos"][i, max(0, end - round(TREMOR_S / DT)):end].astype(np.float64)
  if len(q) < 3:
    return None
  t = np.arange(len(q), dtype=np.float64)
  slope, icept = np.polyfit(t, q, 1)
  return float((q - (np.outer(t, slope) + icept)).std(axis=0).max())


def joint_bands_rad(a: dict, i: int) -> tuple[float | None, float | None]:
  """(vibration, sway): the largest per-joint RMS of the detrended last 4 s at >= 2 Hz and at 0.2-2 Hz (rad)."""
  if "qpos" not in a:
    return None, None
  end = int(a["end_step"][i])
  q = a["qpos"][i, max(0, end - round(TREMOR_S / DT)):end].astype(np.float64)
  if len(q) < 8:
    return None, None
  t = np.arange(len(q), dtype=np.float64)
  slope, icept = np.polyfit(t, q, 1)
  X = np.fft.rfft(q - (np.outer(t, slope) + icept), axis=0)
  f = np.fft.rfftfreq(len(q), DT)
  band = lambda keep: np.fft.irfft(np.where(keep[:, None], X, 0.0), n=len(q), axis=0).std(axis=0).max()  # noqa: E731
  return float(band(f >= VIB_HZ)), float(band((f >= SWAY_HZ) & (f < VIB_HZ)))


def calm_still(r: dict) -> bool:
  """Calm success plus vibration <= 1 mrad and sway <= 3 mrad (missing records count as failing)."""
  return bool(r["ok"] and r.get("vib_rad") is not None and r["vib_rad"] <= VIB_RAD and r.get("sway_rad") is not None
              and r["sway_rad"] <= SWAY_RAD)


def calm_success(a: dict, i: int, spec: dict) -> dict:
  end = int(a["end_step"][i])
  fs = int(a["fell_step"][i])
  fell = 0 <= fs < end
  hits = sorted(round(h["t_s"] / DT) for h in spec["env"]["disturbances"] if h["kind"] == "impulse")
  event, last = (hits[0], hits[-1]) if hits else (end, end)
  pre = slice(max(0, event - round(PRE_S / DT)), event)
  if "sole" in a:
    cent = foot_centroids(a, i)
    foot_move = float(np.linalg.norm(cent[pre.stop - 1] - cent[pre.start], axis=1).max())
  else:                                    # without sole records: the path length of each foot (>= its displacement)
    foot_move = float((a["foot_v"][i, pre] * DT).sum(axis=0).max())
  pelvis_pre = float(a["pelvis_v"][i, pre, 0].max())
  yaw = np.unwrap(a["yaw"][i, pre.start:pre.stop, 0])
  yaw_rate = float(np.abs(np.diff(yaw)).max() / DT) if len(yaw) > 1 else 0.0
  calm_before = foot_move < FOOT_MOVE_M and pelvis_pre < PELVIS_MPS and yaw_rate < YAW_RADPS
  still = (a["pelvis_v"][i, :, 0] < STILL_PELVIS) & (a["foot_v"][i] < STILL_FOOT).all(axis=1)
  need = round(HOLD_S / DT)
  recovered, recovery_s = False, None
  start = last + 1 if hits else end - need
  run = 0
  for t in range(max(start, 0), end):
    run = run + 1 if still[t] else 0
    if run >= need:
      recovered, recovery_s = True, (t + 1 - last) * DT if hits else 0.0
      break
  ok = (not fell) and calm_before and recovered
  return dict(ok=bool(ok), fell=bool(fell), calm_before=bool(calm_before), recovered=bool(recovered),
              foot_move_pre_m=foot_move, pelvis_pre_max_mps=pelvis_pre, yaw_rate_pre_max_radps=yaw_rate,
              recovery_s=recovery_s, hits=len(hits), tremor_rad=joint_tremor_rad(a, i),
              **dict(zip(("vib_rad", "sway_rad"), joint_bands_rad(a, i))))


# -- estimator v2 (coordinator 20:5x; W's leakage finding verified in R03/deliberation/02_negative_measures.md 6) ------
# v1 (joint_bands_rad: rectangular FFT of the detrended last 4 s) leaks slow sway into the >= 2 Hz band: a pure
# 0.3 Hz 10 mrad sway reads 0.18-1.82 mrad 'vibration'. v2 is reported beside v1; the v1 gates stay until round 3
# re-registers the thresholds on v2.
VIB2_HZ = 2.0
SWAY2_BAND_HZ = (0.1, 2.0)
SWAY2_MIN_S = 10.0          # registered window for sway; shorter windows are reported with their length (under-read)


def bands_v2(q: np.ndarray) -> dict | None:
  """q: (T, joints) joint angles of one window (rad). Vibration = RMS of the zero-phase 4th-order Butterworth 2 Hz
  high-pass of the linearly detrended window, 0.5 s trimmed at both ends, largest joint; vib_welch = Hann-Welch
  (2 s segments, 50 % overlap) 2-25 Hz band RMS, largest joint (cross-check); sway = RMS of the zero-phase 2nd-order
  0.1-2 Hz band-pass, same trimming, largest joint. Returns None for windows under 2 s."""
  from scipy import signal
  q = np.asarray(q, dtype=np.float64)
  if len(q) < round(2.0 / DT):
    return None
  t = np.arange(len(q), dtype=np.float64)
  slope, icept = np.polyfit(t, q, 1)
  r = q - (np.outer(t, slope) + icept)
  trim = round(0.5 / DT)
  hp = signal.filtfilt(*signal.butter(4, VIB2_HZ, btype="highpass", fs=1.0 / DT), r, axis=0, padtype="odd")[trim:-trim]
  bp = signal.filtfilt(*signal.butter(2, SWAY2_BAND_HZ, btype="bandpass", fs=1.0 / DT), r, axis=0, padtype="odd")[trim:-trim]
  f, p = signal.welch(r, fs=1.0 / DT, window="hann", nperseg=min(len(r), round(2.0 / DT)), noverlap=round(1.0 / DT),
                      detrend="linear", axis=0)
  band = (f >= VIB2_HZ) & (f <= 25.0)
  return dict(vib=float(hp.std(axis=0).max()), vib_welch=float(np.sqrt(np.trapezoid(p[band], f[band], axis=0)).max()),
              sway=float(bp.std(axis=0).max()), window_s=round(len(q) * DT, 2))
