"""Motion library: retargeted human clips -> one 50 Hz array set with a clip table.

Build (CPU, MuJoCo kinematics):  python -m r1wb.library --set core|full
Each selected clip is resampled to 50 Hz, given "intent" arms (rest pose plus a leg-coupled swing: the arm behaviour
System 1 will have; only pick-up clips keep the captured arms because there the hands carry the task), clamped into
the soft joint range the simulator penalises, mirrored, and stored with analytic body velocities (computed per
clip). The tracking command samples (clip, frame) pairs and never crosses a clip boundary.
  core: standing, walking, turning, sideways, backward and crouches down to 0.5 m pelvis height (first end-to-end pass)
  full: core plus deep crouches and pick-ups
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

FPS = 50
# Not part of the walk / crouch / reach repertoire: acted hit reactions, rolls and dives, sprint bursts.
EXCLUDED = {
  "00344": "hit reaction", "01611": "hit reaction", "01612": "hit reaction", "01613": "hit reaction",
  "01614": "hit reaction", "01615": "hit reaction, airborne 29%", "06130": "hit reaction", "06132": "hit reaction",
  "01660": "airborne 27%, root speed p99 2.8 m/s", "01680": "airborne 19%, pelvis 0.44 m",
  "01882": "root speed p99 2.2 m/s",
  "07241": "seated: CoM 17 cm outside both feet in every frame", "07295": "seated with the legs lifted",
}
SOFT_LIMIT_FACTOR, SOFT_LIMIT_MARGIN = 0.95, 0.01      # mjlab's soft range, and how far inside it references stay
CORE_MIN_PELVIS_Z = 0.50
# Captured arms are not usable (rifle / aiming poses in the 016xx-018xx series, hands on the face or head):
# keep legs, pelvis and waist, replace the arms by the intent-arm model below.
INTENT_ARM_IDS = {"00764", "00765", "07241"}
INTENT_ARM_RANGE = (1600, 1899)
# Arm swing measured on the natural walking captures (checks in the design folder): shoulder pitch follows the
# difference of hip pitch angles with gain about -0.35 at normal walking speed and about 0 in cautious walking.
SWING_GAIN, SWING_SPEED = -0.35, (0.2, 0.5)     # gain ramps between these root speeds (m/s)
SWING_LIMIT, SWING_CUTOFF_HZ = 0.3, 3.0
NATURAL_ARM_IDS = ("00211", "00219", "00220", "00776", "00807", "00230")


def clip_id(name: str) -> str:
  return name.split("_", 1)[0]


def context_of(name: str) -> str:
  low = name.lower()
  if "pickup" in low:
    return "pickup"
  if any(k in low for k in ("crouch", "lowalert", "getlow", "sneak")):
    return "low_posture"
  if any(k in low for k in ("idle", "standpose", "standby")):
    return "stand"
  if "turn" in low:
    return "turn"
  if any(k in low for k in ("side", "strafe", "totheleft", "totheright")):
    return "lateral"
  if any(k in low for k in ("backward", "retreat", "stepback")):
    return "backward"
  return "walk"


def uses_intent_arms(name: str) -> bool:
  """Every clip except pick-ups: the legs come from the human, the arms from the intent model."""
  return context_of(name) != "pickup"


def check_mirror(model, data, qpos: np.ndarray, joints: tuple[str, ...]) -> float:
  """Largest difference between FK(mirror(q)) and the reflection of FK(q) over a few frames (m)."""
  import mujoco
  names = [model.body(i).name for i in range(1, model.nbody)]
  swap = [names.index(n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for n in names]
  worst = 0.0
  for frame in np.linspace(0, len(qpos) - 1, 5).astype(int):
    data.qpos[:] = qpos[frame]
    mujoco.mj_kinematics(model, data)
    original = data.xpos[1:].copy()
    data.qpos[:] = mirror(qpos[frame:frame + 1], joints)[0]
    mujoco.mj_kinematics(model, data)
    reflected = original[swap] * np.array([1.0, -1.0, 1.0])
    worst = max(worst, float(np.abs(data.xpos[1:] - reflected).max()))
  return worst


def resample(qpos: np.ndarray, fps: float) -> np.ndarray:
  from scipy.interpolate import CubicSpline
  from scipy.spatial.transform import Rotation, Slerp
  t = np.arange(len(qpos)) / fps
  new_t = np.minimum(np.arange(int(np.floor(t[-1] * FPS + 1e-6)) + 1) / FPS, t[-1])
  out = np.zeros((len(new_t), qpos.shape[1]))
  out[:, :3] = CubicSpline(t, qpos[:, :3])(new_t)
  out[:, 7:] = CubicSpline(t, qpos[:, 7:])(new_t)
  rotations = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
  out[:, 3:7] = Slerp(t, rotations)(new_t).as_quat()[:, [3, 0, 1, 2]]
  return out


def mirror(qpos: np.ndarray, joints: tuple[str, ...]) -> np.ndarray:
  """Left-right reflection: y -> -y, roll and yaw negated, left and right joints swapped."""
  out = qpos.copy()
  out[:, 1] *= -1
  out[:, 4] *= -1      # quaternion (w, x, y, z): reflect across the x-z plane
  out[:, 6] *= -1
  for index, name in enumerate(joints):
    other = name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
    sign = -1.0 if any(k in name for k in ("roll", "yaw")) else 1.0
    out[:, 7 + index] = sign * qpos[:, 7 + joints.index(other)]
  return out


def intent_arms(qpos: np.ndarray, joints: tuple[str, ...], rest: dict[str, float]) -> np.ndarray:
  """Rest arm pose plus a small shoulder-pitch swing coupled to the legs (speed dependent)."""
  from scipy.signal import butter, filtfilt
  out = qpos.copy()
  index = {name: 7 + i for i, name in enumerate(joints)}
  speed = np.linalg.norm(np.gradient(qpos[:, :2], 1.0 / FPS, axis=0), axis=1)
  b, a = butter(2, 1.0 / (FPS / 2))
  speed = filtfilt(b, a, speed) if len(speed) > 12 else speed
  gain = SWING_GAIN * np.clip((speed - SWING_SPEED[0]) / (SWING_SPEED[1] - SWING_SPEED[0]), 0.0, 1.0)
  b, a = butter(2, SWING_CUTOFF_HZ / (FPS / 2))
  for side, other in (("left", "right"), ("right", "left")):
    difference = qpos[:, index[f"{side}_hip_pitch_joint"]] - qpos[:, index[f"{other}_hip_pitch_joint"]]
    difference = filtfilt(b, a, difference) if len(difference) > 12 else difference
    for name, value in rest.items():
      sign = -1.0 if side == "right" and any(k in name for k in ("roll", "yaw")) else 1.0
      out[:, index[f"{side}_{name}"]] = sign * value
    out[:, index[f"{side}_shoulder_pitch_joint"]] += np.clip(gain * difference, -SWING_LIMIT, SWING_LIMIT)
  return out


def natural_rest_pose(source: Path, joints: tuple[str, ...]) -> dict[str, float]:
  """Median arm joint angles of the natural walking captures (left side; right is its mirror)."""
  names = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint", "wrist_roll_joint")
  values = {name: [] for name in names}
  for path in sorted(source.glob("*.npz")):
    if clip_id(path.name) in NATURAL_ARM_IDS:
      qpos = np.load(path, allow_pickle=True)["qpos"]
      for name in names:
        left = qpos[:, 7 + joints.index(f"left_{name}")]
        right = qpos[:, 7 + joints.index(f"right_{name}")]
        sign = -1.0 if any(k in name for k in ("roll", "yaw")) else 1.0
        values[name].append(np.concatenate([left, sign * right]))
  return {name: round(float(np.median(np.concatenate(v))), 3) for name, v in values.items()}


def body_states(model, data, qpos: np.ndarray):
  """Analytic body-frame poses and world velocities for a 50 Hz qpos trajectory."""
  import mujoco
  frames, bodies = len(qpos), model.nbody - 1
  joint_pos, joint_vel = qpos[:, 7:].copy(), np.zeros((frames, model.nv - 6))
  pos, quat = np.zeros((frames, bodies, 3)), np.zeros((frames, bodies, 4))
  lin, ang = np.zeros((frames, bodies, 3)), np.zeros((frames, bodies, 3))
  velocity, six = np.zeros(model.nv), np.zeros(6)
  for frame in range(frames):
    before, after = max(frame - 1, 0), min(frame + 1, frames - 1)
    mujoco.mj_differentiatePos(model, velocity, (after - before) / FPS, qpos[before], qpos[after])
    data.qpos[:], data.qvel[:] = qpos[frame], velocity
    mujoco.mj_forward(model, data)
    joint_vel[frame] = velocity[6:]
    pos[frame], quat[frame] = data.xpos[1:], data.xquat[1:]
    for body in range(1, model.nbody):
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, six, 0)
      ang[frame, body - 1], lin[frame, body - 1] = six[:3], six[3:]
  return joint_pos, joint_vel, pos, quat, lin, ang


def build(source: Path, out: Path, mirrored: bool = True, subset: str = "full") -> dict:
  import mujoco
  from r1wb import robot
  spec = robot.get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  joints = tuple(robot.JOINTS)
  assert [model.joint(i).name for i in range(1, model.njnt)] == list(joints)
  rest = natural_rest_pose(source, joints)
  arrays = {key: [] for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
  clips, start = [], 0
  for path in sorted(source.glob("*.npz")):
    cid = clip_id(path.name)
    if cid in EXCLUDED:
      continue
    clip = np.load(path, allow_pickle=True)
    assert tuple(clip["joint_names"]) == joints, path
    core = context_of(path.name) != "pickup" and float(clip["qpos"][:, 2].min()) >= CORE_MIN_PELVIS_Z
    if subset == "core" and not core:
      continue
    qpos = resample(clip["qpos"].astype(np.float64), float(clip["fps"]))
    arms = "intent" if uses_intent_arms(path.name) else "captured"
    if arms == "intent":
      qpos = intent_arms(qpos, joints, rest)
    low, high = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    middle, half = 0.5 * (low + high), 0.5 * (high - low) * SOFT_LIMIT_FACTOR - SOFT_LIMIT_MARGIN
    clamped = np.clip(qpos[:, 7:], middle - half, middle + half)
    moved = float(np.abs(clamped - qpos[:, 7:]).max())
    qpos[:, 7:] = clamped
    error = check_mirror(model, data, qpos, joints)
    assert error < 1e-5, (path.name, error)
    for flipped in ((False, True) if mirrored else (False,)):
      trajectory = mirror(qpos, joints) if flipped else qpos
      for key, value in zip(arrays, body_states(model, data, trajectory)):
        arrays[key].append(value.astype(np.float32))
      clips.append(dict(name=path.stem + ("_M" if flipped else ""), source=path.name, mirrored=flipped,
                        context=context_of(path.name), arms=arms, start=start, length=len(trajectory), core=core,
                        pelvis_z_min=round(float(trajectory[:, 2].min()), 3), soft_limit_clamp_rad=round(moved, 4),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
      start += len(trajectory)
  merged = {key: np.concatenate(value) for key, value in arrays.items()}
  manifest = dict(protocol="r1wb-library-v2", subset=subset, fps=FPS, frames=int(start), seconds=round(start / FPS, 1),
                  joint_names=list(joints), body_names=[model.body(i).name for i in range(1, model.nbody)],
                  hand_mass_kg=robot.HAND_MASS, intent_arm_rest_pose=rest,
                  intent_arm_swing=dict(gain=SWING_GAIN, speed_ramp_mps=SWING_SPEED, limit_rad=SWING_LIMIT,
                                        cutoff_hz=SWING_CUTOFF_HZ),
                  excluded=EXCLUDED, clips=clips)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out, fps=np.array([FPS]), clip_start=np.array([c["start"] for c in clips]),
                      clip_length=np.array([c["length"] for c in clips]), manifest=json.dumps(manifest), **merged)
  manifest["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
  out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
  return manifest


class Library:
  """Library tensors on a device; bodies are selected by name so the entity's body order is irrelevant."""

  def __init__(self, path: str | Path, body_names: tuple[str, ...], device: str = "cpu"):
    import torch
    data = np.load(path, allow_pickle=False)
    self.manifest = json.loads(str(data["manifest"]))
    order = [self.manifest["body_names"].index(name) for name in body_names]
    tensor = lambda key: torch.tensor(data[key], dtype=torch.float32, device=device)  # noqa: E731
    self.joint_pos, self.joint_vel = tensor("joint_pos"), tensor("joint_vel")
    self.body_pos_w = tensor("body_pos_w")[:, order]
    self.body_quat_w = tensor("body_quat_w")[:, order]
    self.body_lin_vel_w = tensor("body_lin_vel_w")[:, order]
    self.body_ang_vel_w = tensor("body_ang_vel_w")[:, order]
    self.clip_start = torch.tensor(data["clip_start"], dtype=torch.long, device=device)
    self.clip_length = torch.tensor(data["clip_length"], dtype=torch.long, device=device)
    self.clip_end = self.clip_start + self.clip_length - 1          # last valid frame of each clip
    self.frame_clip = torch.repeat_interleave(torch.arange(len(self.clip_start), device=device), self.clip_length)
    self.time_step_total = int(self.joint_pos.shape[0])
    self.clip_names = [c["name"] for c in self.manifest["clips"]]


def main():
  from r1lab.run_record import work_root
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, default=work_root() / "data/motions_r1")
  parser.add_argument("--set", choices=["core", "full"], default="core")
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--no-mirror", action="store_true")
  args = parser.parse_args()
  args.out = args.out or work_root() / f"data/wb/library_v2_{args.set}.npz"
  manifest = build(args.source, args.out, mirrored=not args.no_mirror, subset=args.set)
  by_context: dict[str, float] = {}
  for clip in manifest["clips"]:
    by_context[clip["context"]] = by_context.get(clip["context"], 0.0) + clip["length"] / FPS
  print(f"{args.out}: {len(manifest['clips'])} clips, {manifest['seconds']} s, sha256 {manifest['sha256'][:12]}")
  print("seconds by context:", {k: round(v) for k, v in sorted(by_context.items())})
  print("intent arm rest pose:", manifest["intent_arm_rest_pose"])
  print("intent arms:", sum(c["arms"] == "intent" for c in manifest["clips"]), "captured arms:",
        sum(c["arms"] == "captured" for c in manifest["clips"]),
        "| largest soft-limit clamp (rad):", max(c["soft_limit_clamp_rad"] for c in manifest["clips"]))
  _ = re


if __name__ == "__main__":
  main()
