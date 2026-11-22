"""Human references for stopping and starting (coordinator decision r1b-2), from the train clips only
(shared/motion_train; the held-out split is never read; the manufacturer is never used).

Events are found by what the person does, with one rule for every clip:
  speed     pelvis horizontal speed, 50 Hz central difference, 0.5 s moving average (one step: removes the gait ripple)
  stop      the smoothed speed falls below 0.05 m/s and stays there >= 1 s, after a peak >= 0.3 m/s in the 2.5 s before;
            v_c = that peak; onset t0 = the last time before the stop with speed >= 0.9 v_c (the earliest visible
            sign of stopping: a person has no stop command, so this stands in for it)
  start     the speed has stayed below 0.05 m/s for >= 1 s and reaches >= 0.3 m/s within 2.5 s; v_c = the peak there
  settle    the last time after t0 (until the stop + 2 s) that a sole is >= 1 cm above the clip's floor; settle time =
            that time - t0 (soles from forward kinematics of r1_head26.xml: the lowest point of the 7 foot capsules)
  stop J    typical jerk (human_j: per-frame RMS over the 24 body joints of the 50 Hz third difference, time median)
            over [t0, t0 + 1 s]: the 'stop transition' J type
  accel     0.8 v_c / (time from 0.1 v_c to 0.9 v_c) for a start, and the same from 0.9 v_c down to 0.1 v_c for a stop
            (mean rate of the speed ramp), plus the peak rate of the smoothed speed
Takes are counted once (clips of the same 5-digit take are near copies). Selections: 'category' = walk_stop clips,
'motion' = every train clip with a detected event.
usage: python -I -m r1adv.negative.human_stop --clips <motion_train dir> --model <r1_head26.xml> --out stop.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from r1adv.negative.human_j import typical_jerk

DT = 0.02
SMOOTH = 25
STILL, MOVING, LIFT = 0.05, 0.3, 0.01
FACTOR, BAND, FLOOR = 1.25, 0.73, 20.0


def soles(model_path: str, clip) -> np.ndarray:
  """(frames, 2) lowest sole point (m) of the left and the right foot."""
  import mujoco
  model = mujoco.MjModel.from_xml_path(model_path)
  data = mujoco.MjData(model)
  feet = {side: [g for g in range(model.ngeom)
                 if re.search(rf"{side}_foot[1-7]_collision$", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
          for side in ("left", "right")}
  out = np.zeros((len(clip["joint_pos"]), 2))
  for t in range(len(out)):
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = clip["root_pos"][t], clip["root_quat"][t], clip["joint_pos"][t]
    mujoco.mj_kinematics(model, data)
    for s, side in enumerate(("left", "right")):
      lows = []
      for g in feet[side]:
        axis = data.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g, 1]
        lows.append(min(data.geom_xpos[g, 2] + axis[2], data.geom_xpos[g, 2] - axis[2]) - model.geom_size[g, 0])
      out[t, s] = min(lows)
  return out


def speed_of(root_pos: np.ndarray) -> np.ndarray:
  v = np.zeros(len(root_pos))
  v[1:-1] = np.linalg.norm(root_pos[2:, :2] - root_pos[:-2, :2], axis=1) / (2 * DT)
  v[0], v[-1] = v[1], v[-2]
  pad = np.pad(v, SMOOTH // 2, mode="edge")
  return np.convolve(pad, np.ones(SMOOTH) / SMOOTH, mode="valid")


def stop_events(v: np.ndarray) -> list[tuple[int, int, float]]:
  """(onset, stop, v_c) frame indices."""
  out, n, hold, look = [], len(v), round(1.0 / DT), round(2.5 / DT)
  t = 1
  while t < n - hold:
    if v[t] < STILL <= v[t - 1] and (v[t:t + hold] < STILL).all():
      before = v[max(0, t - look):t]
      if before.size and before.max() >= MOVING:
        vc = float(before.max())
        onset = max(i for i in range(max(0, t - look), t) if v[i] >= 0.9 * vc)
        out.append((onset, t, vc))
      t += hold
    t += 1
  return out


def start_events(v: np.ndarray) -> list[tuple[int, float]]:
  out, n, hold, look = [], len(v), round(1.0 / DT), round(2.5 / DT)
  for t in range(hold, n - 1):
    if v[t - 1] < STILL <= v[t] and (v[t - hold:t] < STILL).all():
      after = v[t:min(n, t + look)]
      if after.max() >= MOVING:
        out.append((t, float(after.max())))
  return out


def ramp_rate(v: np.ndarray, begin: int, vc: float, rising: bool) -> tuple[float | None, float]:
  """Mean rate between 0.1 v_c and 0.9 v_c, and the peak rate of the smoothed speed, over the ramp."""
  idx = range(begin, len(v)) if rising else range(begin, len(v))
  t10 = t90 = None
  for i in idx:
    if rising:
      if t10 is None and v[i] >= 0.1 * vc:
        t10 = i
      if v[i] >= 0.9 * vc:
        t90 = i
        break
    else:
      if t90 is None and v[i] <= 0.9 * vc:
        t90 = i
      if v[i] <= 0.1 * vc:
        t10 = i
        break
  if t10 is None or t90 is None or t10 == t90:
    return None, float("nan")
  a, b = (t10, t90) if rising else (t90, t10)
  rate = np.gradient(v[a:b + 1], DT)
  return float(0.8 * vc / ((b - a) * DT)), float(np.max(rate) if rising else -np.min(rate))


def clip_events(model_path: str, clip, cid: str, category: str) -> dict:
  v = speed_of(clip["root_pos"].astype(np.float64))
  sole = soles(model_path, clip)
  floor = float(np.percentile(sole.min(1), 5))
  lifted = (sole - floor >= LIFT).any(1)
  jt = typical_jerk(clip["joint_pos"].astype(np.float64))      # frame k of jt belongs to frames k..k+3
  stops, starts = [], []
  for onset, stop, vc in stop_events(v):
    end = min(len(v) - 1, stop + round(2.0 / DT))
    lifts = [i for i in range(onset, end + 1) if lifted[i]]
    settle = (lifts[-1] - onset) * DT if lifts else 0.0
    window = jt[onset:min(len(jt), onset + round(1.0 / DT))]
    mean, peak = ramp_rate(v, max(0, onset - 1), vc, rising=False)
    stops.append(dict(clip=cid, category=category, onset_s=round(onset * DT, 2), stop_s=round(stop * DT, 2), v_c=round(vc, 3),
                      settle_s=round(settle, 2), stop_J=round(float(np.median(window)), 2) if window.size else None,
                      decel_mean=None if mean is None else round(mean, 3), decel_peak=round(peak, 3)))
  for begin, vc in start_events(v):
    mean, peak = ramp_rate(v, max(0, begin - 1), vc, rising=True)
    starts.append(dict(clip=cid, category=category, start_s=round(begin * DT, 2), v_c=round(vc, 3),
                       accel_mean=None if mean is None else round(mean, 3), accel_peak=round(peak, 3)))
  return dict(stops=stops, starts=starts, floor_m=round(floor, 4))


def summary(rows: list[dict], key: str) -> dict:
  vals = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])])
  if not vals.size:
    return dict(n=0)
  return dict(n=int(vals.size), p10=round(float(np.percentile(vals, 10)), 3), p50=round(float(np.percentile(vals, 50)), 3),
              p90=round(float(np.percentile(vals, 90)), 3), max=round(float(vals.max()), 3))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clips", type=Path, required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  manifest_path = args.clips / "manifest_v0.json"
  manifest = json.loads(manifest_path.read_text())
  if manifest.get("side") != "train":
    raise SystemExit("train clips only")
  taken, events = set(), dict(stops=[], starts=[])
  for c in sorted(manifest["clips"], key=lambda c: c["id"]):
    take = c["id"][:5]
    if take in taken:
      continue
    clip = np.load(args.clips / c["file"])
    found = clip_events(args.model, clip, c["id"], c["category"])
    if found["stops"] or found["starts"]:
      taken.add(take)
      events["stops"] += found["stops"]
      events["starts"] += found["starts"]
  out = dict(schema="r1adv-human-stop-v1", clips_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rules=__doc__.split("\n\n")[0],
             events=events, selections={})
  for name, pick in (("category", lambda r: r["category"] == "walk_stop"), ("motion", lambda r: True)):
    stops = [r for r in events["stops"] if pick(r)]
    starts = [r for r in events["starts"] if pick(r)]
    sel = dict(stops=len(stops), starts=len(starts), settle_s=summary(stops, "settle_s"), stop_J=summary(stops, "stop_J"),
               decel_mean=summary(stops, "decel_mean"), decel_peak=summary(stops, "decel_peak"),
               accel_mean=summary(starts, "accel_mean"), accel_peak=summary(starts, "accel_peak"))
    if sel["settle_s"].get("n"):
      sel["settle_ref_s"] = round(FACTOR * sel["settle_s"]["p90"], 3)
    if sel["stop_J"].get("n"):
      sel["stop_J_ref"] = round(max(FACTOR * sel["stop_J"]["p90"] / BAND, FLOOR), 2)
    out["selections"][name] = sel
  args.out.write_text(json.dumps(out, indent=1))
  print(json.dumps(out["selections"], indent=1))


if __name__ == "__main__":
  main()
