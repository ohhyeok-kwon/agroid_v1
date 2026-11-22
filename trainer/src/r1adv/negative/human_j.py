"""Human jerk reference for the J criterion (coordinator decision R0-1): J_human per mission type = p90 over 2 s windows
of the typical jerk (time median of the per-frame RMS over the 24 body joints of the 50 Hz third difference, the
same path as runner/score). Windows are typed by what the person actually does in them (realised pelvis speed, yaw
rate, pelvis height), not by the clip's file label. Source: train clips only (shared/motion_train, R class); the
held-out split is never read. Also measures how much a 30 fps -> 50 Hz cubic round trip (the mocap pipeline's
band limit) lowers jerk, on these clips and on robot trajectories (passed as .npz with joint_pos).
usage: python -I -m r1adv.negative.human_j --clips <motion_train dir> --out table.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DT = 0.02
WINDOW, STEP = 100, 50
FLOOR, FACTOR = 20.0, 1.25
GO_BANDS = ((0.05, 0.25), (0.25, 0.45), (0.45, 0.8), (0.8, 3.0))


def typical_jerk(joint_pos: np.ndarray) -> np.ndarray:
  """Per-frame RMS over the 24 body joints of the 50 Hz third difference (rad/s^3)."""
  jerk = np.diff(joint_pos[:, :24], n=3, axis=0) / DT ** 3
  return np.sqrt((jerk ** 2).mean(1))


def yaw_of(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q.T
  return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def mission_type(speed: float, wz: float, z_min: float, z_drop: float) -> str | None:
  if z_min < 0.62 or z_drop > 0.08:
    return "crouch"
  if speed < 0.05 and abs(wz) < 0.15:
    return "hold"
  if speed < 0.15 and abs(wz) >= 0.3:
    return "turn"
  for lo, hi in GO_BANDS:
    if lo <= speed < hi:
      return f"go_{lo:.2f}-{hi:.2f}"
  return None


def cubic_roundtrip(joint_pos: np.ndarray, fps: float = 30.0) -> np.ndarray:
  """Sample at `fps` and interpolate back to 50 Hz with a natural cubic spline (the retarget's band limit)."""
  from scipy.interpolate import CubicSpline
  t = np.arange(len(joint_pos)) * DT
  coarse = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
  spline = CubicSpline(coarse, np.stack([np.interp(coarse, t, joint_pos[:, k]) for k in range(joint_pos.shape[1])], 1),
                       axis=0, bc_type="natural")
  return spline(t)


def windows(clip: dict, category: str) -> list[dict]:
  q, pos, quat = clip["joint_pos"], clip["root_pos"], clip["root_quat"]
  jt = typical_jerk(q)
  yaw = yaw_of(quat)
  out = []
  for start in range(0, len(q) - WINDOW, STEP):
    end = start + WINDOW
    seconds = (end - 1 - start) * DT
    speed = float(np.linalg.norm(pos[end - 1, :2] - pos[start, :2]) / seconds)
    wz = float((yaw[end - 1] - yaw[start]) / seconds)
    z = pos[start:end, 2]
    kind = mission_type(speed, wz, float(z.min()), float(z.max() - z.min()))
    j = jt[max(start - 3, 0):end - 3]
    out.append(dict(type=kind, category=category, start=start, speed=speed, wz=wz, z_min=float(z.min()),
                    jerk=float(np.median(j))))
  return out


def table(clips_dir: Path) -> dict:
  manifest = json.loads((clips_dir / "manifest_v0.json").read_text())
  if manifest.get("side") != "train":
    raise ValueError("only the train side may be read")
  rows, roundtrip, clips = [], [], 0
  for entry in manifest["clips"]:
    path = clips_dir / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
      raise ValueError(f"clip changed: {entry['id']}")
    with np.load(path, allow_pickle=False) as z:
      clip = {k: z[k] for k in ("joint_pos", "root_pos", "root_quat")}
    clips += 1
    for w in windows(clip, entry["category"]):
      w.update(clip=entry["id"], arms_valid=bool(entry["arms_valid"]))
      rows.append(w)
    filtered = cubic_roundtrip(clip["joint_pos"])
    roundtrip.append(dict(clip=entry["id"], category=entry["category"], raw=float(np.median(typical_jerk(clip["joint_pos"]))),
                          roundtrip=float(np.median(typical_jerk(filtered)))))
  types = {}
  for kind in sorted({r["type"] for r in rows if r["type"]}):
    use = [r for r in rows if r["type"] == kind and r["category"] != "hit_stagger"]
    arms = [r for r in use if r["arms_valid"]]
    if not use:
      continue
    vals = np.array([r["jerk"] for r in use])
    types[kind] = dict(windows=len(use), clips=len({r["clip"] for r in use}), median=float(np.median(vals)),
                       p90=float(np.percentile(vals, 90)), p10=float(np.percentile(vals, 10)),
                       p90_arms_valid=float(np.percentile([r["jerk"] for r in arms], 90)) if arms else None,
                       J_ref=float(max(FACTOR * np.percentile(vals, 90), FLOOR)))
  ratio = np.array([r["roundtrip"] / max(r["raw"], 1e-9) for r in roundtrip])
  return dict(schema="r1adv-human-j-v1", source=str(clips_dir), manifest_sha256=hashlib.sha256((clips_dir / "manifest_v0.json").read_bytes()).hexdigest(),
              clips=clips, window_s=WINDOW * DT, step_s=STEP * DT, factor=FACTOR, floor=FLOOR, go_bands=GO_BANDS,
              types=types, untyped_windows=sum(1 for r in rows if not r["type"]),
              hit_stagger_windows=sum(1 for r in rows if r["category"] == "hit_stagger"),
              human_roundtrip_ratio=dict(median=float(np.median(ratio)), p10=float(np.percentile(ratio, 10)),
                                         p90=float(np.percentile(ratio, 90))),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), windows=rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--clips", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  result = table(args.clips)
  args.out.write_text(json.dumps(result, indent=0))
  print(json.dumps({k: result[k] for k in ("clips", "untyped_windows", "hit_stagger_windows", "human_roundtrip_ratio")}))
  for kind, row in result["types"].items():
    print(f"{kind:14s} windows {row['windows']:4d} clips {row['clips']:3d} median {row['median']:7.1f} p90 {row['p90']:7.1f} "
          f"p90(arms valid) {row['p90_arms_valid'] if row['p90_arms_valid'] is None else round(row['p90_arms_valid'], 1)} J_ref {row['J_ref']:7.1f}")


if __name__ == "__main__":
  main()
