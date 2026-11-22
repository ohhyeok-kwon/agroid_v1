"""Motion library for the W1-J tracker (r1wb-library-v2 layout, readable by r1wb.tracking.command).

Input: r1adv-clip-v1 clips (26 joints, 50 Hz). The tracker body is the 24-joint r1wb robot (head fused), so the two head
joints are dropped. Arms: the captured human arms; clips flagged arms_valid=False (two-handed rifle pose) get r1wb's
intent arms. Joints are clamped into mjlab's soft range (as r1wb.library). Optional spectral augmentation (arm B of E1):
one extra copy per clip with band-limited (2-20 Hz) noise on all 24 joints, RMS drawn per clip in [0.004, 0.015] rad,
deterministic from sha256(seed tag + clip id). The source of this builder is the judge's public track_m0c.py
(copied and changed, not imported).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FPS = 50.0
AUG_BAND = (2.0, 20.0)
AUG_RMS = (0.004, 0.015)
# The W1 validation split inside the training side of the r1adv split (01_interfaces.md 4): whole takes
# (MetaHuman: first 5 digits of the id) whose sha256(VAL_SEED + take) first byte < VAL_BELOW (about 19 % of takes).
VAL_SEED, VAL_BELOW = "r1adv-wam-val-2026-11-17", 0x30


def take_of(clip_id: str) -> str:
  return clip_id.split("_", 1)[0]


def is_w1_val(clip_id: str) -> bool:
  return int(hashlib.sha256((VAL_SEED + take_of(clip_id)).encode("utf-8")).hexdigest()[:2], 16) < VAL_BELOW


def load_clip(path: Path) -> dict:
  with np.load(path, allow_pickle=False) as z:
    return dict(root_pos=z["root_pos"].astype(np.float64), root_quat=z["root_quat"].astype(np.float64),
                joint_pos=z["joint_pos"].astype(np.float64), joint_names=[str(x) for x in z["joint_names"]])


def band_noise(n: int, joints: int, key: str) -> np.ndarray:
  """Band-limited noise (AUG_BAND), unit RMS per joint, then one RMS per clip from AUG_RMS."""
  seed = int(hashlib.sha256(("r1adv-wam-e1-aug|" + key).encode()).hexdigest()[:16], 16)
  rng = np.random.default_rng(seed)
  white = rng.standard_normal((n, joints))
  spec = np.fft.rfft(white, axis=0)
  f = np.fft.rfftfreq(n, 1.0 / FPS)
  spec[(f < AUG_BAND[0]) | (f > AUG_BAND[1])] = 0.0
  x = np.fft.irfft(spec, n=n, axis=0)
  x /= np.maximum(x.std(0, keepdims=True), 1e-9)
  return x * rng.uniform(*AUG_RMS)


def build(clips: list[dict], out: Path, augment: bool = False, note: str = "") -> dict:
  """clips: [{"id", "path", "category", "arms_valid"(optional, default True)}]. Writes out (npz) + out.manifest.json."""
  import mujoco
  from r1wb import library as L
  from r1wb import robot
  rest = L.natural_rest_pose(Path("/home/jimmy/workspace/agroid_v1/.agent/trainer/data/motions_r1"), tuple(robot.JOINTS))
  model = robot.get_spec().compile()
  data = mujoco.MjData(model)
  joints = tuple(robot.JOINTS)
  arrays = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
  table, start = [], 0
  low, high = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
  middle, half = 0.5 * (low + high), 0.5 * (high - low) * L.SOFT_LIMIT_FACTOR - L.SOFT_LIMIT_MARGIN
  for c in clips:
    d = load_clip(Path(c["path"]))
    idx = [d["joint_names"].index(j) for j in joints]
    base = np.concatenate([d["root_pos"], d["root_quat"], d["joint_pos"][:, idx]], 1)
    variants = [("", base)]
    if augment:
      q = base.copy()
      q[:, 7:] += band_noise(len(q), len(joints), c["id"])
      variants.append(("__aug", q))
    for suffix, q0 in variants:
      q = q0.copy()
      arms = "captured"
      if c.get("arms_valid", True) is False:
        q = L.intent_arms(q, joints, rest)
        arms = "intent"
      clamped = np.clip(q[:, 7:], middle - half, middle + half)
      moved = float(np.abs(clamped - q[:, 7:]).max())
      q[:, 7:] = clamped
      for key, value in zip(arrays, L.body_states(model, data, q)):
        arrays[key].append(value.astype(np.float32))
      table.append(dict(name=c["id"] + suffix, source=c["path"], mirrored=False, context=L.context_of(c["id"]), arms=arms,
                        augmented=bool(suffix), start=start, length=len(q), core=True,
                        pelvis_z_min=round(float(q[:, 2].min()), 3), soft_limit_clamp_rad=round(moved, 4),
                        category=c.get("category")))
      start += len(q)
  merged = {k: np.concatenate(v) for k, v in arrays.items()}
  manifest = dict(protocol="r1wb-library-v2", subset="r1adv-wam", fps=50, frames=int(start), joint_names=list(joints),
                  body_names=[model.body(i).name for i in range(1, model.nbody)], clips=table, augment=augment,
                  aug_band_hz=list(AUG_BAND), aug_rms_rad=list(AUG_RMS), note=note)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out, fps=np.array([50]), clip_start=np.array([c["start"] for c in table]),
                      clip_length=np.array([c["length"] for c in table]), manifest=json.dumps(manifest), **merged)
  out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
  return manifest
