"""Server rollout of scenario specs with existing policies (R0 diagnostic, R1 level search; evaluation only).

One process = one policy x one repeat. Every instance is one environment on its bank patch, all in one batch:
standardized delivered command (r1wb.eval.common_command, the native slew for every policy), System 2 h12/v2d plan for
the learned policies, uniform sensor noise of the N1 shape scaled per instance (applied with the shared-sensor
transform at the actor boundary), torso impulses at the spec's time / azimuth (relative to the heading then) /
height / size. Fall = the physical criterion of r1wb.world (or trunk contact). Arrays per step go to <out>/arrays.npz.

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/rl-agent/src python -B -m r1adv.negative.runner --policy v12h \
      --specs specs.jsonl --out <dir> [--permute <seed>] [--limit-steps N]

Head26 policies use the sealed helpers of their evaluation study (head_eval_adapter, clean_dagger) read-only.
R1: sensor model of r1adv.negative.sensors (noise, IMU bias and mounting tilt, latency and jitter), and the deploy-
contract policies of the positive agent (--policy positive --checkpoint model_N.pt: 810/570 frames, HoldSymmetricMLP).
R1 extensions (3)-(5): thrown balls and boxes (r1wb.world throwables, released at the spec's time toward the torso),
body randomization per instance (mass and inertia scale, torso payload, foot friction, PD gain scale), and a vertically
vibrating floor (a shaker plate on slide joints, its top 2 cm above the floor, driven every physics substep).
Actions are always the deterministic policy mean.
Stability V1 options (off by default): --zero-policy (PD targets at the default pose), --record-contacts (contacts, ground
reaction, angular momentum), --freeze-at-hit (diagnostic: policy off from the hit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
# "--warp-run-to-run" asks Warp for run-to-run reproducible atomics (a physics test; it does not compile for
# mujoco_warp 3.x here). It has nothing to do with the policy's actions, which are always the deterministic mean.
RUN_TO_RUN = "--warp-run-to-run" in sys.argv
if RUN_TO_RUN:
  os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"
if RUN_TO_RUN:                                        # Warp RUN_TO_RUN atomics + deterministic torch kernels (test)
  wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
  torch.use_deterministic_algorithms(True, warn_only=True)

from r1adv.negative import bank as B  # noqa: E402
from r1adv.negative.seedtree import spec_hash  # noqa: E402

DT = 0.02
DEFAULT_HEIGHT = 0.731
WORK = Path(os.environ.get("R1_WORK", "/rl-agent/.agent"))
STUDIES = WORK / "result/evaluation_batches"
V12H = "wholebody_20261115-1402_p1t12h-head-perception_n16384_from-p1t10s@2748_stage-h-fixed-v10s-v12h-seed1"
POLICIES = {
  "v12h": dict(kind="teacher", run=V12H, iteration=3248,
               sha256="1a05f23394ec005e59b8556c3122322b58c8ec15365d6f2106c80c1cd7d03019",
               helpers="claude-slopecourse45-evaluation-20261116"),
  "slope45": dict(kind="teacher", run="wholebody_20261116-0742_p1t12h-slopecourse45_n16384_from-p1t10s@2748_v12h-slopecourse45-seed1-claude",
                  iteration=3248, sha256="30e0d12af5d687abb416f392f258ddbde3106a029f8d6494c42149263132dc79",
                  helpers="claude-slopecourse45-evaluation-20261116"),
  "rad10x_student": dict(kind="student", arm="A", run="wholebody_20261116-0108_p1t13-dagger-A_n8192_teacher-p1t12h@3248_claude-dagger-rad10-A-s1",
                         iteration=649, sha256="338f33e9a1f1eb8e900921ef4aaedf448300bad5db35ddeca0f3691c55746e8b",
                         helpers="clean-dagger-rad10-evaluation-20261116"),
  "deployable_D": dict(kind="student", arm="D", run="wholebody_20261116-0400_p1t13-dagger-D_n8192_teacher-p1t12h@3248_claude-dagger-rad10-D-s1-u1000",
                       iteration=999, sha256="9d186668ce9d84467af12fc817d9e1f3f8d32fbfc3cfa6da01c69bd2c910842f",
                       helpers="clean-dagger-rad10-D-evaluation-20261116"),
  "manufacturer": dict(kind="manufacturer", bundle="/tmp/agroid-manufacturer-eval-20261114-162347"),
  "positive": dict(kind="deploy"),                    # checkpoint from --checkpoint (positive agent's Hold snapshots)
  "constant": dict(kind="constant"),                  # zero action = joint targets fixed at the nominal pose (R2 S0-e)
}
FRAME_SENSOR = list(range(0, 6)) + list(range(9, 57))   # gyro, gravity, q - q0, dq of an 81-value deploy frame
CONTACT_SLOTS = 64                                      # --record-contacts slots per env (overflow: ncontact > 64)


def package_sha256() -> str:
  here = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  for path in sorted(here.glob("*.py")):
    digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
  return digest.hexdigest()


def sha256(path) -> str:
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


# the code this process runs, hashed at import (a file synced during a long batch must not relabel earlier runs)
IMPORTED = dict(runner_sha256=sha256(Path(__file__)), package_sha256=package_sha256())


def checkpoint(name: str) -> Path | None:
  info = POLICIES[name]
  if info["kind"] in ("manufacturer", "constant"):
    return None
  if info["kind"] == "deploy":
    return Path(info["checkpoint"])
  path = WORK / "runs/r1_wholebody" / info["run"] / f"model_{info['iteration']}.pt"
  if sha256(path) != info["sha256"]:
    raise ValueError(f"{name}: checkpoint sha differs from the evaluated identity")
  return path


# -- specs -> tensors ----------------------------------------------------------------------------------------------

def load_specs(path: Path, atlas: bool = False) -> list[dict]:
  specs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  for spec in specs:
    if spec_hash(spec) != spec["spec_sha256"]:
      raise ValueError(f"spec hash mismatch: {spec.get('seed_path')}")
    env = spec["env"]
    body = env["body"]
    family = spec["generator"]["family"]
    if atlas:
      if env["terrain"] is not None or (env["scene"] or {}).get("source") != "atlas":
        raise ValueError(f"{spec.get('seed_path')}: --atlas runs scene specs only")
      env = dict(env, terrain={}, scene=None)
    elif family not in ("legacy-v0", "pc-v0"):
      from r1adv.negative.families import FAMILIES
      from r1adv.negative.families_v1 import FAMILIES as FAMILIES_V1
      from r1adv.negative.families_ext import FAMILIES as FAMILIES_EXT
      from r1adv.negative.families_r2 import FAMILIES as FAMILIES_R2
      known = {**FAMILIES, **FAMILIES_V1, **FAMILIES_EXT, **FAMILIES_R2}
      if family not in known or known[family].runnable != "legacy":
        raise ValueError(f"{spec.get('seed_path')}: family {family} is not runnable with the legacy policies")
    vibration_ok = all(h["kind"] != "floor_vibration" or (h["axis"] in ("x", "y", "z") and h["frequency_hz"] <= 30.0
                                                           and h["amplitude_m"] <= 0.008 and env["terrain"]
                                                           and env["terrain"]["kind"] == "flat")
                       for h in env["disturbances"])
    if (env["terrain"] is None or env["scene"] is not None or env["obstacles"]
        or any(h["kind"] not in ("impulse", "throw", "floor_vibration") for h in env["disturbances"]) or not vibration_ok
        or env["sense"].get("dropout_p", 0.0) or not set(body) <= {"mass_scale", "payload_kg", "friction_scale", "kp_scale", "kd_scale"}
        or any(s["type"] == "Go" and s["mode"] != "velocity" for s in spec["mission"]["steps"])):
      raise ValueError(f"{spec.get('seed_path')}: outside what the legacy policies can run")
  return specs


def schedule(spec: dict, horizon: int):
  """Per control tick: (vx, vy, wz, height, height_on) and the mission step index (-1 settle, -2 after the end)."""
  rows = np.zeros((horizon, 5), np.float32)
  rows[:, 3] = DEFAULT_HEIGHT
  step = np.full(horizon, -2, np.int16)
  t = round(spec["mission"]["settle_s"] / DT)
  step[:t] = -1
  previous_pose = False
  for k, s in enumerate(spec["mission"]["steps"]):
    n = round(s["duration_s"] / DT)
    part = slice(t, min(t + n, horizon))
    if s["type"] == "Go":
      rows[part, :3] = (s["vx"], s["vy"], s["wz"])
      rows[part, 3:5] = (DEFAULT_HEIGHT, 1.0 if previous_pose else 0.0)
    elif s["type"] == "Hold":
      rows[part, 3:5] = (DEFAULT_HEIGHT, 1.0 if previous_pose else 0.0)
    elif s["type"] == "Pose":
      rows[part, 3:5] = (s["pelvis_height_m"], 1.0)
    else:
      raise ValueError(s["type"])
    step[part] = k
    previous_pose = s["type"] == "Pose"
    t += n
  return rows, step, t


# -- environment and policies --------------------------------------------------------------------------------------

PLATFORM = dict(half_xy_m=1.0, half_z_m=0.005, top_m=0.02, mass_kg=500.0, park_z_m=-50.0, ramp_s=0.5)


def add_platform(cfg) -> None:
  """Shaker table for vibrating-floor levels: a 2 x 2 x 0.01 m plate whose top is 2 cm above the flat floor (its
  bottom stays >= 2 mm above the floor up to the 8 mm amplitude the loader accepts, so it never touches the floor), on three slide joints
  x, y, z in absolute world coordinates, gravity compensated, 500 kg. PlatformDriver prescribes its position and
  velocity before every physics substep. Geom group 0, so the ground rays see it as the ground."""
  import mujoco
  from mjlab.entity import EntityCfg

  def spec():
    made = mujoco.MjSpec()
    body = made.worldbody.add_body(name="platform", gravcomp=1.0)
    for name, axis in (("platform_x", [1.0, 0.0, 0.0]), ("platform_y", [0.0, 1.0, 0.0]), ("platform_z", [0.0, 0.0, 1.0])):
      body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    body.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, -PLATFORM["half_z_m"]],
                  size=[PLATFORM["half_xy_m"], PLATFORM["half_xy_m"], PLATFORM["half_z_m"]], mass=PLATFORM["mass_kg"],
                  rgba=[0.6, 0.6, 0.65, 1.0], group=0)
    return made
  cfg.scene.entities["platform"] = EntityCfg(spec_fn=spec)


class PlatformDriver:
  """Moves the plate kinematically: before physics substep k it writes q(t_k) and the secant velocity
  (q(t_{k+1}) - q(t_k)) / dt, so integration lands exactly on the next knot (piecewise-linear path at the physics
  rate, contacts see the plate's true velocity, no teleports). Displacement along the spec axis:
  amplitude * w(tau) * sin(2 pi f tau), tau = t - t_s, with a raised-cosine ramp-in w over PLATFORM["ramp_s"] (the
  plate does not jolt at t_s). Hooked into the scene's per-substep write (Scene.write_data_to_sim).
  It also measures the floor -> body transmissibility at the drive frequency (vibration criterion proposal): a lock-in
  of the linear velocity of the pelvis, the torso and (head26 robots) the camera, and of the plate, at every physics
  substep from 1 s after the ramp to the end (steady state). T = |V_body| / |V_plate| along the drive axis (equal to
  the acceleration ratio at one frequency); `leak` = the body's lock-in amplitude off the drive axis / |V_plate|."""

  def __init__(self, env, specs_env: list[dict]):
    self.env = env
    self.platform = env.scene["platform"]
    self.robot = env.scene["robot"]
    names = list(self.robot.body_names)
    self.points = [n for n in ("pelvis", "torso_link", "camera_link") if n in names]
    self.point_ids = [names.index(n) for n in self.points]
    dev = env.device
    vib = [[h for h in s["env"]["disturbances"] if h["kind"] == "floor_vibration"] for s in specs_env]
    self.on = torch.tensor([bool(v) for v in vib], device=dev)
    self.amp = torch.tensor([v[0]["amplitude_m"] if v else 0.0 for v in vib], device=dev)
    self.freq = torch.tensor([v[0]["frequency_hz"] if v else 0.0 for v in vib], device=dev)
    self.start = torch.tensor([v[0]["t_s"] if v else 0.0 for v in vib], device=dev)
    axis = torch.zeros(len(vib), 3, device=dev)
    for e, v in enumerate(vib):
      if v:
        axis[e, "xyz".index(v[0]["axis"])] = 1.0
    self.axis = axis
    self.dt = env.physics_dt
    origin = env.scene.env_origins
    rest = torch.stack([origin[:, 0], origin[:, 1], origin[:, 2] + PLATFORM["top_m"]], dim=1)
    parked = torch.tensor([0.0, 0.0, PLATFORM["park_z_m"]], device=dev).expand_as(rest)
    self.rest = torch.where(self.on[:, None], rest, parked)
    self.k = 0
    self.window = self.start + PLATFORM["ramp_s"] + 1.0
    self.lock_body = torch.zeros(len(vib), len(self.points), 3, 2, device=dev)     # sum v cos, sum v sin
    self.lock_plate = torch.zeros(len(vib), 3, 2, device=dev)
    self.samples = torch.zeros(len(vib), device=dev)
    self.write(0)
    self.platform.write_data_to_sim = self.substep          # instance attribute: the scene calls it every substep

  def offset(self, t: float) -> torch.Tensor:
    tau = (t - self.start).clamp_min(0.0)
    ramp = torch.where(tau < PLATFORM["ramp_s"], 0.5 - 0.5 * torch.cos(math.pi * tau / PLATFORM["ramp_s"]), torch.ones_like(tau))
    return (self.on * self.amp * ramp * torch.sin(2.0 * math.pi * self.freq * tau))[:, None] * self.axis

  def write(self, k: int) -> torch.Tensor:
    now, nxt = self.offset(k * self.dt), self.offset((k + 1) * self.dt)
    velocity = (nxt - now) / self.dt
    self.platform.write_joint_state_to_sim(self.rest + now, velocity)
    return velocity

  def substep(self) -> None:
    velocity = self.write(self.k)
    t = self.k * self.dt
    use = (self.on & (t >= self.window)).float()
    if bool(use.any()):
      phase = 2.0 * math.pi * self.freq * (t - self.start)
      basis = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1) * use[:, None]          # (envs, 2)
      body = self.robot.data.body_link_lin_vel_w[:, self.point_ids]                         # (envs, P, 3)
      self.lock_body += body[..., None] * basis[:, None, None, :]
      self.lock_plate += velocity[..., None] * basis[:, None, :]
      self.samples += use
    self.k += 1

  def transmissibility(self) -> dict:
    """Per point: T along the drive axis and the off-axis leak, both relative to the plate's lock-in amplitude."""
    n = self.samples.clamp_min(1.0)[:, None]
    plate = torch.linalg.vector_norm(self.lock_plate, dim=-1) / n                         # (envs, 3) amplitude / 2
    drive = (plate * self.axis).sum(1).clamp_min(1e-12)
    out = {}
    for j, name in enumerate(self.points):
      amp = torch.linalg.vector_norm(self.lock_body[:, j], dim=-1) / n                     # (envs, 3)
      along = (amp * self.axis).sum(1)
      off = torch.linalg.vector_norm(amp * (1.0 - self.axis), dim=1)
      out[f"T_{name}"] = torch.where(self.on & (self.samples > 0), along / drive, torch.full_like(along, float("nan")))
      out[f"leak_{name}"] = torch.where(self.on & (self.samples > 0), off / drive, torch.full_like(off, float("nan")))
    return out


def apply_body(env, specs_env: list[dict]) -> dict:
  """Per-instance body: all robot bodies' mass and inertia x mass_scale (uniform density), torso + payload (point
  mass), foot capsules' sliding friction x friction_scale, PD stiffness x kp_scale and damping x kd_scale."""
  from mjlab.managers.event_manager import RecomputeLevel
  from r1wb.eval.case_pack import _foot_geoms
  robot = env.scene["robot"]
  dev = env.device
  values = {k: torch.tensor([float(s["env"]["body"].get(k, d)) for s in specs_env], device=dev)
            for k, d in (("mass_scale", 1.0), ("payload_kg", 0.0), ("friction_scale", 1.0), ("kp_scale", 1.0), ("kd_scale", 1.0))}
  env.sim.expand_model_fields(("body_mass", "body_inertia", "geom_friction", "actuator_gainprm", "actuator_biasprm"))
  model = env.sim.model
  bodies = robot.indexing.body_ids.long()
  torso = bodies[robot.body_names.index("torso_link")]
  mass, inertia = env.sim.get_default_field("body_mass"), env.sim.get_default_field("body_inertia")
  model.body_mass[:, bodies] = mass[bodies] * values["mass_scale"][:, None]
  model.body_inertia[:, bodies] = inertia[bodies] * values["mass_scale"][:, None, None]
  model.body_mass[:, torso] = model.body_mass[:, torso] + values["payload_kg"]
  feet = torch.tensor(_foot_geoms(env.sim.mj_model), device=dev)
  friction = env.sim.get_default_field("geom_friction")
  model.geom_friction[:, feet, 0] = friction[feet, 0] * values["friction_scale"][:, None]
  ctrl = robot.indexing.ctrl_ids.long()
  gain, bias = env.sim.get_default_field("actuator_gainprm"), env.sim.get_default_field("actuator_biasprm")
  model.actuator_gainprm[:, ctrl, 0] = gain[ctrl, 0] * values["kp_scale"][:, None]
  model.actuator_biasprm[:, ctrl, 1] = bias[ctrl, 1] * values["kp_scale"][:, None]
  model.actuator_biasprm[:, ctrl, 2] = bias[ctrl, 2] * values["kd_scale"][:, None]
  env.sim.recompute_constants(RecomputeLevel.set_const)
  return {k: [float(x) for x in v.cpu()] for k, v in values.items()}


def set_action_delay(cfg, steps: int) -> None:
  """Command (actuator) delay of exactly `steps` physics steps (5 ms each) for every joint: min = max lag."""
  import dataclasses
  robot = cfg.scene.entities["robot"]
  robot.articulation.actuators = tuple(dataclasses.replace(a, delay_min_lag=steps, delay_max_lag=steps, delay_hold_prob=0.0)
                                       for a in robot.articulation.actuators)


def build(policy: str, count: int, seed: int, atlas: Path | None = None, throws: bool = False, platform: bool = False,
          action_delay_steps: int = 0):
  from r1wb.student.stream import stand_clips, stream_env_cfg
  from r1wb.tracking.env_cfg import library_path
  from r1wb import world
  from r1adv.negative.world_build import apply_bank
  from r1adv.negative.generator import BANK_SEED
  info = POLICIES[policy]
  helpers = None
  if info["kind"] in ("manufacturer", "deploy", "constant"):
    cfg = stream_env_cfg(play=True, clean=True, plain_actions=True, matched=False)
  else:
    helpers = STUDIES / info["helpers"] / "helpers"
    sys.path.insert(0, str(helpers))
    import head_eval_adapter as H
    H.activate(True)                                # head26 robot/actions/observations + the 417 sensor transform
    cfg = H.configure_head_stream(stream_env_cfg(play=True, clean=True, matched=True, teacher=str(checkpoint("v12h"))))
  command = cfg.commands["motion"]
  command.motion_file = str(library_path("corebal"))
  command.clip_list = (stand_clips(command.motion_file)[0],)     # the common first standing frame (proper contract)
  command.pose_range, command.velocity_range, command.joint_position_range = {}, {}, (0.0, 0.0)
  command.pin_clips, command.sampling_mode = True, "uniform"
  for key in ("match_noise", "hold_after_hit", "lag_grace"):
    if hasattr(command, key):
      setattr(command, key, 0.0)
  names = {}
  if throws:
    names = {"ball": world.add_throwables(cfg, "ball", 2), "box": world.add_throwables(cfg, "box", 2)}
  if platform:
    add_platform(cfg)
  if atlas is None:
    apply_bank(cfg, BANK_SEED)
  else:
    from r1adv.negative.scene_world import apply_atlas
    apply_atlas(cfg, str(atlas))
  world.raise_build_pose(cfg)
  if hasattr(cfg, "curriculum"):
    cfg.curriculum = {}
  cfg.scene.num_envs = count
  cfg.seed = seed
  cfg.episode_length_s = 1.0e9
  cfg._negative_throwables = names
  if action_delay_steps:
    set_action_delay(cfg, action_delay_steps)
  return cfg, helpers


def spawn(env, term, yaws: torch.Tensor, slopes: torch.Tensor, clearance: float = 0.002,
          raise_m: torch.Tensor | None = None, start: str = "corebal", poses: dict | None = None) -> torch.Tensor:
  """Re-pin every robot on the common standing frame, turn it about its own vertical by the spawn yaw, then set its
  height so the lowest foot-capsule sample clears its patch surface (plane through the centre, uphill +y, slope
  tan = slopes) by `clearance` (r1wb.eval.case_pack placement; no time integration). Returns the vertical moves."""
  from mjlab.utils.lab_api.math import quat_mul
  from r1wb.eval.case_pack import _foot_geoms
  robot = env.scene["robot"]
  ids = torch.arange(env.num_envs, device=env.device)
  term._resample_command(ids)
  free_q = robot.indexing.free_joint_q_adr.long()
  qpos = env.sim.data.qpos
  origin = env.scene.env_origins
  turn = torch.stack([torch.cos(yaws / 2), torch.zeros_like(yaws), torch.zeros_like(yaws), torch.sin(yaws / 2)], dim=1)
  quat = quat_mul(turn, qpos[:, free_q[3:7]])
  offset = qpos[:, free_q[:2]] - origin[:, :2]
  c, s = torch.cos(yaws), torch.sin(yaws)
  turned = torch.stack([c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1]], dim=1)
  qpos[:, free_q[3:7]] = quat / torch.norm(quat, dim=1, keepdim=True)
  qpos[:, free_q[:2]] = origin[:, :2] + turned
  if start == "default":                      # R2: the deploy contract's nominal pose, pelvis upright, same heading
    q = qpos[:, free_q[3:7]]
    heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    qpos[:, free_q[3:7]] = torch.stack([torch.cos(heading / 2), torch.zeros_like(heading), torch.zeros_like(heading),
                                        torch.sin(heading / 2)], dim=1)
    qpos[:, robot.indexing.joint_q_adr.long()] = robot.data.default_joint_pos
  if poses is not None:                       # per-env start frames (R2 curriculum): joints + pelvis roll/pitch, same heading
    sel = poses["mask"]
    q = qpos[:, free_q[3:7]]
    heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    cr, sr = torch.cos(poses["roll"] / 2), torch.sin(poses["roll"] / 2)
    cp, sp = torch.cos(poses["pitch"] / 2), torch.sin(poses["pitch"] / 2)
    cy, sy = torch.cos(heading / 2), torch.sin(heading / 2)
    framed = torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                          cr * cp * sy - sr * sp * cy], dim=1)
    qpos[:, free_q[3:7]] = torch.where(sel[:, None], framed, qpos[:, free_q[3:7]])
    joint_adr = robot.indexing.joint_q_adr.long()
    qpos[:, joint_adr[poses["entity_index"]]] = torch.where(sel[:, None], poses["joints"], qpos[:, joint_adr[poses["entity_index"]]])
  env.sim.data.qvel.zero_()
  env.sim.data.qacc_warmstart.zero_()
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  if start == "corebal" and not torch.allclose(qpos[:, free_q[:2]], origin[:, :2] + turned, atol=1e-5):
    raise RuntimeError("spawn pose did not reach the simulator state")
  env.sim.forward()
  model = env.sim.mj_model
  feet = _foot_geoms(model)
  ids_feet = torch.tensor(feet, device=env.device)
  centre = env.sim.data.geom_xpos[:, ids_feet]                                   # (envs, 14, 3)
  axis = env.sim.data.geom_xmat[:, ids_feet].reshape(env.num_envs, len(feet), 3, 3)[..., :, 2]
  radius = torch.tensor(model.geom_size[feet, 0], device=env.device, dtype=torch.float)
  half = torch.tensor(model.geom_size[feet, 1], device=env.device, dtype=torch.float)
  along = torch.linspace(-1.0, 1.0, 9, device=env.device)
  points = centre[:, :, None, :] + axis[:, :, None, :] * (half[None, :, None, None] * along[None, None, :, None])
  bottom = points[..., 2] - radius[None, :, None]
  surface = origin[:, 2, None, None] + slopes[:, None, None] * (points[..., 1] - origin[:, 1, None, None])
  if raise_m is not None:                                    # standing on the vibration platform
    surface = surface + raise_m[:, None, None]
  lift = (surface + clearance - bottom).amax(dim=(1, 2))
  qpos[:, free_q[2]] = qpos[:, free_q[2]] + lift
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  env.sim.forward()
  if hasattr(term, "_pending_forward"):
    term._pending_forward = False
  env.scene.update(0.0)
  env.sim.sense()
  if hasattr(term, "update_relative_body_poses"):
    term.update_relative_body_poses()
  env.observation_manager.reset()
  env.obs_buf = env.observation_manager.compute(update_history=True)
  return lift


def load_deploy(path: Path, device: str):
  """The positive agent's deploy-contract actor (810, 570 or 330 inputs, 24 policy-order actions), symmetric by
  construction: its own class is read, never changed. Returns (model, width)."""
  from tensordict import TensorDict
  from r1adv.positive.hold_model import HoldSymmetricMLP
  loaded = torch.load(path, map_location=device, weights_only=False)
  state = loaded["actor_state_dict"]
  if "r1adv_plain" in state:                                  # R4 wave 2 XB: plain MLP actor (no mirror average); built as
    from r1adv.positive.hold_plain import load_actor          # its own class -- HoldSymmetricMLP would symmetrise it
    model, width = load_actor(state, device)
    if width not in (810, 570, 330):
      raise ValueError(f"not a deploy-contract Hold actor: {width} inputs")
    return model, width
  layers = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
  hidden = [state[f"mlp.{i}.weight"].shape[0] for i in layers[:-1]]
  width, actions = state[f"mlp.{layers[0]}.weight"].shape[1], state[f"mlp.{layers[-1]}.weight"].shape[0]
  example = TensorDict({"actor": torch.zeros(1, width, device=device)}, batch_size=[1])
  model = HoldSymmetricMLP(example, {"actor": ["actor"]}, "actor", actions, hidden_dims=hidden, activation="elu",
                           obs_normalization=True, distribution_cfg=None)
  weights = {k: v for k, v in state.items() if not k.startswith("distribution.")}
  if "r1adv_contract" in model.state_dict() and "r1adv_contract" not in weights:
    weights["r1adv_contract"] = model.state_dict()["r1adv_contract"]      # checkpoints saved before the contract buffer
  model.load_state_dict(weights)
  model.to(device).eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  if width - 1 in (810, 570, 330) and actions == 24:          # R3 wave 1: contract + one foot-lead input b (runner --foot-lead)
    return model, width
  if width not in (810, 570, 330) or actions != 24:           # 330: R2 arm V / R2c-V (no joint velocity in the frame)
    raise ValueError(f"not a deploy-contract Hold actor: {width} -> {actions}")
  return model, width


class SymBranch(torch.nn.Module):
  """Diagnostic only (coordinator 20:5x, two-foot hop mechanism): one branch of HoldSymmetricMLP's average
  0.5 * (mlp(o) + M mlp(M o)). 'a' = mlp(o), 'b' = M mlp(M o). Reads the loaded model, never changes it."""

  def __init__(self, model, branch: str):
    super().__init__()
    self.model, self.branch = model, branch

  def forward(self, obs):
    m = self.model
    raw = torch.cat([obs[group] for group in m.obs_groups], dim=-1)
    if self.branch == "a":
      return m.mlp(m.obs_normalizer(raw))
    other = m.mlp(m.obs_normalizer(raw[..., m.mirror_obs_index] * m.mirror_obs_sign))
    return other[..., m.mirror_act_index] * m.mirror_act_sign


class VelocityEstimator:
  """Joint speed as a robot's driver would estimate it: quantized joint positions differenced at the physics rate
  (200 Hz here) and passed through a first-order low-pass (spec sense.quant: resolution_rad, cutoff_hz). Updated
  before every physics substep by wrapping the robot's per-substep write (Scene.write_data_to_sim)."""

  def __init__(self, env, sensors):
    self.robot = env.scene["robot"]
    dev, dt = env.device, env.physics_dt
    params = [q or {} for q in sensors.quant_params]
    self.resolution = torch.tensor([[max(float(q.get("resolution_rad", 1e-9)), 1e-12)] for q in params], device=dev)
    tau = torch.tensor([[1.0 / (2.0 * math.pi * q.get("cutoff_hz", 1e9))] for q in params], device=dev)
    self.alpha = dt / (tau + dt)
    self.dt = dt
    # profile r1-floor-v1: white noise N(0, white_sd_rad) on the joint positions before the quantization, drawn per
    # physics step from the spec's noise seed (stream 13), for the environments whose quant spec carries white_sd_rad
    whites = [float(q.get("white_sd_rad", 0.0)) for q in params]
    rows = [i for i, w in enumerate(whites) if w > 0.0]
    self.white, self.k = None, 0
    if rows:
      joints = self.robot.data.joint_pos.shape[1]
      steps = int(sensors.horizon) * int(round(env.step_dt / dt)) + 16
      tape = np.stack([np.random.Generator(np.random.Philox(np.random.SeedSequence([sensors.noise_seeds[i], 13])))
                       .standard_normal((steps, joints)) * whites[i] for i in rows])
      self.white = torch.as_tensor(tape, device=dev, dtype=torch.float)
      self.white_rows = torch.as_tensor(rows, device=dev)
    self.reset()
    original = self.robot.write_data_to_sim

    def hooked():
      original()
      self.update()
    self.robot.write_data_to_sim = hooked

  def quantized(self) -> torch.Tensor:
    q = self.robot.data.joint_pos
    if self.white is not None:
      q = q.clone()
      q[self.white_rows] += self.white[:, min(self.k, self.white.shape[1] - 1)]
    return torch.round(q / self.resolution) * self.resolution

  def reset(self) -> None:
    self.previous = self.quantized()
    self.value = torch.zeros_like(self.previous)

  def update(self) -> None:
    self.k += 1
    now = self.quantized()
    self.value = self.value + self.alpha * ((now - self.previous) / self.dt - self.value)
    self.previous = now


class Sensing:
  """The actor-boundary sensing of one 81-wide deploy frame (policy order): gyro scale error, estimated joint speed
  (quant_lp), observation latency, additive noise (SharedSensorStream.manufacturer_frame) and the r1lab sensor lag
  blend toward the previous measured frame."""

  def __init__(self, sensors, count: int, device, to_policy: torch.Tensor, estimator, actor_clean: bool = False):
    from r1adv.negative.sensors import DelayLine
    self.sensors, self.to_policy, self.estimator, self.actor_clean = sensors, to_policy, estimator, actor_clean
    self.line = DelayLine(count, 81, device)
    self.previous = None
    self.quantize_pos = bool((getattr(sensors, "pos_lsb", torch.zeros(1)) > 0).any())

  def frame(self, latest: torch.Tensor, step: int) -> torch.Tensor:
    from r1wb.eval import shared_sensor_observation as S
    if self.actor_clean:                     # T-form actors read the noise-free frame (positive hold_r2 actor_clean)
      return latest.clone()
    sensors = self.sensors
    latest = latest.clone()
    latest[:, :3] = latest[:, :3] * sensors.gyro_scale
    if self.estimator is not None:
      latest[:, 33:57] = torch.where(sensors.quant[:, None], self.estimator.value[:, self.to_policy], latest[:, 33:57])
    self.line.push(step, latest)
    late = latest.clone()
    late[:, FRAME_SENSOR] = self.line.get(sensors.measured)[:, FRAME_SENSOR]
    latest = torch.where(sensors.delayed[:, None], late, latest)
    frame = torch.where(sensors.clean[:, None], latest, S.SharedSensorStream.manufacturer_frame(sensors, latest, step, self.to_policy))
    measured = frame.clone()
    if self.previous is not None:
      blend = sensors.lag
      frame[:, FRAME_SENSOR] = (1.0 - blend) * frame[:, FRAME_SENSOR] + blend * self.previous[:, FRAME_SENSOR]
    self.previous = measured
    if self.quantize_pos:                    # profile r1-floor-v1: the encoder reading lies on the quantum lattice
      lsb = sensors.pos_lsb
      frame[:, 9:33] = torch.where(lsb > 0.0, torch.round(frame[:, 9:33] / lsb.clamp_min(1e-12)) * lsb, frame[:, 9:33])
    return frame


class FrameHistory:
  """10 frames, oldest first, of the deploy contract; a restarted environment is refilled with its first frame."""
  def __init__(self):
    self.history, self.episode = None, None

  def push(self, frame: torch.Tensor, episode: torch.Tensor) -> torch.Tensor:
    if self.history is None:
      self.history = frame[:, None].repeat(1, 10, 1)
    else:
      self.history = torch.cat((self.history[:, 1:], frame[:, None]), dim=1)
      restarted = episode < self.episode
      self.history[restarted] = frame[restarted, None].expand(-1, 10, -1)
    self.episode = episode.clone()
    return self.history.reshape(len(frame), -1)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", required=True, choices=tuple(POLICIES))
  parser.add_argument("--checkpoint", type=Path, default=None, help="--policy positive: the deploy-contract checkpoint")
  parser.add_argument("--specs", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--permute", type=int, default=None, help="seed of the environment order (repeat check)")
  parser.add_argument("--limit", type=int, default=None, help="first N specs only (smoke)")
  parser.add_argument("--limit-steps", type=int, default=None)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--warp-run-to-run", action="store_true", help="Warp RUN_TO_RUN + deterministic torch (physics bit-repeat test)")
  parser.add_argument("--atlas", type=Path, default=None, help="scene atlas (r1adv.negative.scenes) instead of the bank")
  parser.add_argument("--action-delay-ms", type=int, default=0, help="command (actuator) delay, a multiple of 5 ms")
  parser.add_argument("--start", choices=("corebal", "default"), default="corebal",
                      help="standing frame of the corebal clip (R1 common start) or the deploy nominal pose (R2)")
  parser.add_argument("--record-actions", action="store_true", help="store actions and the 24 body joints per step")
  parser.add_argument("--actor-clean", action="store_true",
                      help="deploy-frame actors read the noise-free frame (T-form teachers: positive hold_r2 actor_clean); "
                           "the command delay still applies")
  parser.add_argument("--record-bodies", action="store_true",
                      help="with --record-feet: per step the speeds of hands (wrist roll links), forearms (elbow links), feet "
                           "(ankle roll links) and the torso, and the translational kinetic energy of the arms and of the whole "
                           "body (deliberation Q7, first principle: flailing / dangerous speed)")
  parser.add_argument("--no-fall-termination", action="store_true",
                      help="keep simulating after a fall (only the time limit ends an episode): post-fall behaviour probes "
                           "(deliberation Q6); falls are then found in the records (tilt / height), not by fell_step")
  parser.add_argument("--record-feet", action="store_true",
                      help="store the whole-body centre of mass (position, velocity) and the 28 sole points (capsule ends, "
                           "lowest point: x, y, z above the patch surface) per step: stepping and capture-point analyses")
  parser.add_argument("--freeze-after-fall", type=float, default=None,
                      help="diagnostic (coordinator 21:5x, post-fall tremor cause): once an env has been down (tilt > 1 rad or "
                           "height < 0.35 m) for this many seconds, hold its action at the last applied value (PD targets "
                           "frozen, policy off) to the end; never for verdicts")
  parser.add_argument("--foot-lead", type=float, default=None,
                      help="R3 wave 1 (b x posture L): the constant foot-lead input b (+1 / -1) appended after the actor input "
                           "when the checkpoint's input is one wider than its deploy contract (810/570/330 + 1); off by default")
  parser.add_argument("--sym-branch", choices=("avg", "a", "b"), default="avg",
                      help="diagnostic: run one branch of the symmetric deploy actor (a = mlp(o), b = M mlp(M o)) instead of "
                           "their average (two-foot hop mechanism test, coordinator 20:5x); never for verdicts")
  parser.add_argument("--record-root", action="store_true",
                      help="store the pelvis (root link) position minus the env origin and its quaternion (w, x, y, z) per "
                           "step: kinematic re-rendering of the recorded episode, W1-B re-tracking (off by default)")
  parser.add_argument("--record-hit", action="store_true",
                      help="store the external wrench written at the torso per step (force xyz N, torque xyz N m; off by default)")
  parser.add_argument("--zero-policy", action="store_true",
                      help="stability V1 (R03/stability_v1.md): every joint's PD target at the default pose from the first step "
                           "(zero action; the policy is not consulted; the same action as --policy constant); off by default")
  parser.add_argument("--freeze-at-hit", action="store_true",
                      help="diagnostic (stability V1, exploratory Z-hit): from each env's first impulse step on, hold its action at "
                           "the last applied value (policy off at the hit: a passive fall from the same state); never for verdicts")
  parser.add_argument("--record-contacts", action="store_true",
                      help="stability V1: per step the robot-environment contacts (world position, force on the robot in the world "
                           "frame, robot body index) in up to 64 slots per env and their count, the total ground reaction force and "
                           "its centre of pressure, and the whole-body angular momentum about the centre of mass (world frame, from "
                           "the body states); off by default")
  parser.add_argument("--plant-tremor", default=None,
                      help="HZ,RAD: add a sine of RAD (joint target radians) at HZ to every joint target (validity check)")
  args = parser.parse_args()
  started = time.time()
  args.out.mkdir(parents=True, exist_ok=False)
  specs = load_specs(args.specs, atlas=args.atlas is not None)[:args.limit]
  count = len(specs)
  order = np.arange(count) if args.permute is None else np.random.default_rng(args.permute).permutation(count)
  specs_env = [specs[i] for i in order]                 # environment e runs instance order[e]
  horizon = max(round(s["mission"]["duration_s"] / DT) for s in specs)
  if args.limit_steps:
    horizon = min(horizon, args.limit_steps)
  set_digest = hashlib.sha256("\n".join(s["spec_sha256"] for s in specs).encode()).hexdigest()
  seed = int(set_digest[:8], 16)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if args.policy == "positive":
    if args.checkpoint is None:
      parser.error("--policy positive needs --checkpoint")
    POLICIES["positive"] = dict(kind="deploy", checkpoint=str(args.checkpoint.resolve()), sha256=sha256(args.checkpoint))
  info = POLICIES[args.policy]

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.lab_api.math import yaw_quat
  from tensordict import TensorDict
  from r1wb import robot as R, world
  from r1wb.eval.case_pack import apply_ground_reference
  from r1wb.eval.common_command import CommonCommand
  from r1wb.eval import shared_sensor_observation as S
  from r1adv.negative.world_build import assign, geometry_sha256
  from r1adv.negative.sensors import DelayLine, SensorModel, sense_of

  has_throws = any(h["kind"] == "throw" for s in specs for h in s["env"]["disturbances"])
  has_platform = any(h["kind"] == "floor_vibration" for s in specs for h in s["env"]["disturbances"])
  has_body = any(any(v != d for k, d in (("mass_scale", 1.0), ("payload_kg", 0.0), ("friction_scale", 1.0), ("kp_scale", 1.0),
                                         ("kd_scale", 1.0)) for v in [s["env"]["body"].get(k, d)]) for s in specs)
  if args.action_delay_ms % 5:
    raise SystemExit("--action-delay-ms must be a multiple of the 5 ms physics step")
  for s in specs_env:                       # verdict specs name the batch conditions they were made for
    v = s["env"].get("verdict")
    if v and (v["command_delay_ms"] != args.action_delay_ms or v["start"] != args.start):
      raise SystemExit(f"{s['seed_path']}: made for delay {v['command_delay_ms']} ms / start {v['start']}, "
                       f"run with {args.action_delay_ms} ms / {args.start}")
  cfg, helpers = build(args.policy, count, seed, args.atlas, throws=has_throws, platform=has_platform,
                       action_delay_steps=args.action_delay_ms // 5)
  if args.no_fall_termination:                          # only the time limit may end an episode
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if getattr(v, "time_out", False) or "time" in k}
  built = time.time()
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  built = time.time() - built
  wrapped = RslRlVecEnvWrapper(env)
  term = env.command_manager.get_term("motion")
  if args.atlas is None:
    cols = [B.column(s["env"]["terrain"]["kind"]) for s in specs_env]
    rows = [int(s["env"]["terrain"]["row"]) for s in specs_env]
  else:
    from r1adv.negative.scene_world import layout, load_atlas, patch_cell
    patches = load_atlas(args.atlas)[0].shape[0]
    cells = [patch_cell(int(s["env"]["scene"]["index"]) % patches, patches) for s in specs_env]
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
  origins = assign(env, rows, cols, layout(patches) if args.atlas is not None else None)
  scripts, steps_of, ends = [], [], []
  for s in specs_env:
    rows_s, step_s, end = schedule(s, horizon)
    scripts.append(rows_s)
    steps_of.append(step_s)
    ends.append(min(end, horizon))
  raw = torch.as_tensor(np.stack(scripts), device=env.device)               # (envs, T, 5)
  table = torch.zeros(count, 1, 8, device=env.device)
  table[:, 0, 0], table[:, 0, 4] = 1.0e6, DEFAULT_HEIGHT                   # the env's own stream stands (unused)
  term.set_script(table)
  body_values = apply_body(env, specs_env) if has_body else None
  env.reset()
  where = (lambda s: s["env"]["scene"]) if args.atlas is not None else (lambda s: s["env"]["terrain"])
  yaws = torch.tensor([where(s)["spawn_yaw_rad"] for s in specs_env], device=env.device, dtype=torch.float)
  slopes = torch.tensor([0.0 if args.atlas is not None else B.plane_height(s["env"]["terrain"]["kind"], s["env"]["terrain"]["row"], 1.0)
                         for s in specs_env], device=env.device, dtype=torch.float)
  driver = PlatformDriver(env, specs_env) if has_platform else None
  start_poses = None
  if any(s["env"].get("start_pose") for s in specs_env):
    from r1wb import robot as R_
    pose_joints = [s["env"]["start_pose"]["joint_pos_rad"] if s["env"].get("start_pose") else [0.0] * 24 for s in specs_env]
    start_poses = dict(mask=torch.tensor([bool(s["env"].get("start_pose")) for s in specs_env], device=env.device),
                       roll=torch.tensor([(s["env"].get("start_pose") or {}).get("pelvis_roll_rad", 0.0) for s in specs_env], device=env.device),
                       pitch=torch.tensor([(s["env"].get("start_pose") or {}).get("pelvis_pitch_rad", 0.0) for s in specs_env], device=env.device),
                       joints=torch.tensor(pose_joints, device=env.device, dtype=torch.float),
                       entity_index=torch.tensor([list(env.scene["robot"].joint_names).index(n) for n in R_.JOINTS], device=env.device))
  lift = spawn(env, term, yaws, slopes, raise_m=torch.where(driver.on, PLATFORM["top_m"], 0.0) if driver else None,
               start=args.start, poses=start_poses)
  apply_ground_reference(env, term)
  common = CommonCommand(torch.zeros(count, 3, device=env.device), DT)
  sensors = SensorModel(specs_env, horizon, env.device)
  quiet, delayed = sensors.clean, sensors.delayed
  estimator = VelocityEstimator(env, sensors) if bool(sensors.quant.any()) else None
  r1lab_profile = any(sense_of(s)["profile"] == "r1lab-deploy-v1" for s in specs_env)
  if r1lab_profile and info["kind"] in ("teacher", "student"):
    raise SystemExit("the r1lab-deploy-v1 sensor profile is wired for deploy-frame policies (deploy, manufacturer, constant)")
  entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  kind = info["kind"]
  stats = dict(decision_s=0.0)

  if kind == "manufacturer":
    from r1wb.eval.manufacturer import load as load_manufacturer
    _, Manufacturer = load_manufacturer(info["bundle"], reset_history=True)
    manufacturer = Manufacturer(env, torch.zeros(count, 3, device=env.device))
    state = dict(history=None, episode=None, step=None)

    sensing = Sensing(sensors, count, env.device, manufacturer.to_policy, estimator, actor_clean=args.actor_clean)

    def noisy_input(model, inputs):
      observation, clean = inputs[0], inputs[0]["actor"]
      latest = clean.reshape(-1, 10, 81)[:, -1]
      frame = sensing.frame(latest, state["step"])
      episode = env.episode_length_buf.clone()
      history = state["history"]
      history = frame[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], frame[:, None]), dim=1)
      if state["episode"] is not None:
        restarted = episode < state["episode"]
        history[restarted] = frame[restarted, None].expand(-1, 10, -1)
      state["history"], state["episode"] = history, episode
      changed = observation.clone()
      changed["actor"] = history.reshape(-1, 810)
      return (changed, *inputs[1:])
    manufacturer.model.register_forward_pre_hook(noisy_input)

    def act(step, previous_dones):
      common.advance(step, raw[:, step, :3])
      manufacturer.twists = common.delivered
      state["step"] = step
      return manufacturer(None)
  elif kind == "constant":
    width_actions = env.action_manager.action.shape[1]

    def act(step, previous_dones):
      common.advance(step, raw[:, step, :3])
      return torch.zeros(count, width_actions, device=env.device)
  elif kind == "deploy":
    from r1lab import contract as LC
    model, deploy_width = load_deploy(checkpoint(args.policy), args.device)
    lead = None
    if deploy_width - 1 in (810, 570, 330):                   # foot-lead checkpoint: base contract + b
      if args.foot_lead is None:
        raise SystemExit("this checkpoint takes a foot-lead input b: pass --foot-lead +1 or -1")
      deploy_width -= 1
      lead = torch.full((count, 1), float(args.foot_lead), device=args.device)
    if args.sym_branch != "avg":
      model = SymBranch(model, args.sym_branch)
    to_policy = torch.tensor(LC.POLICY_TO_ENTITY, device=env.device)
    to_entity = torch.tensor(LC.ACTION_IDX, device=env.device)
    default = torch.tensor(LC.DEFAULT_POS, device=env.device)
    frames = FrameHistory()
    sensing = Sensing(sensors, count, env.device, to_policy, estimator, actor_clean=args.actor_clean)
    deploy = dict(previous=torch.zeros(count, 24, device=env.device))

    def act(step, previous_dones):
      common.advance(step, raw[:, step, :3])
      data = entity.data
      episode = env.episode_length_buf.clone()
      if previous_dones is not None and previous_dones.any():
        deploy["previous"][previous_dones] = 0.0
      latest = torch.cat([data.root_link_ang_vel_b, data.projected_gravity_b, common.delivered,
                          data.joint_pos[:, to_policy] - default, data.joint_vel[:, to_policy], deploy["previous"]], dim=1)
      frame = sensing.frame(latest, step)
      seen = frames.push(frame if deploy_width == 810 else frame[:, :57] if deploy_width == 570 else frame[:, :33], episode)
      action = model(TensorDict({"actor": seen if lead is None else torch.cat([seen, lead], dim=1)}, batch_size=[count]))
      deploy["previous"] = action
      return action[:, to_entity]
  else:
    from r1wb.s2.runtime import S2View, goal_brain
    from r1wb.s2.small import SmallS2
    from r1wb.s2.tokenizer import MotionTokenizer
    from r1wb.tracking.env_cfg import teacher_observation
    tokenizer = MotionTokenizer.load(WORK / "data/s2/tokenizer_v2d.pt", args.device)
    mind = SmallS2.load(WORK / "data/s2/small_h12.pt", args.device)
    clock = dict(step=0)

    def template():
      out = torch.zeros(count, 5, device=env.device)
      row = raw[:, min(clock["step"], horizon - 1)]
      out[:, 3], out[:, 4] = torch.where(row[:, 4] > 0.5, row[:, 3], torch.full_like(row[:, 3], DEFAULT_HEIGHT)), row[:, 4]
      return out
    view = S2View(env, term, goal_brain(mind, lambda decision: common.goal(template())), tokenizer)
    apply_ground_reference(env, view)
    if kind == "teacher":
      from r1wb.student.observations import load_teacher
      model = load_teacher(str(checkpoint(args.policy)), args.device)
      if model.obs_dim != 417:
        raise ValueError("expected a head26 scan teacher")
      student = None
    else:
      from clean_dagger.evaluation import HistoryEvaluationInput, restored_student
      loaded = torch.load(checkpoint(args.policy), map_location="cpu", weights_only=False)
      if loaded.get("dagger_state", {}).get("arm") != info["arm"]:
        raise ValueError("student arm differs")
      student = HistoryEvaluationInput(restored_student(loaded, args.device))

    from r1wb.head_contract import actor_features
    measured_idx = torch.tensor([i for i, n in enumerate(actor_features(head=True, senses="scan"))
                                 if n.startswith(("anchor.", "base.", "body.q.", "body.dq."))], device=env.device)
    line417 = DelayLine(count, len(measured_idx), env.device)

    def act(step, previous_dones):
      clock["step"] = step
      common.advance(step, raw[:, step, :3])
      t0 = time.perf_counter()
      common.update_view(view)
      stats["decision_s"] += time.perf_counter() - t0
      seen = teacher_observation(view.env, "motion", joint_action.plain_action, local=True, width=417)
      line417.push(step, seen[:, measured_idx])
      late = seen.clone()
      late[:, measured_idx] = line417.get(sensors.measured)
      late = torch.where(delayed[:, None], late, seen)
      noisy = S.SharedSensorStream.teacher(sensors, late, step, 180, root_quaternion=entity.data.root_link_quat_w,
                                           anchor_quaternion=view.robot_anchor_quat_w)
      noisy = torch.where(quiet[:, None], late, noisy)            # exact rows without noise: no transform at all
      if student is None:
        plain = model(TensorDict({"actor": noisy}, batch_size=[count]))
      else:
        plain = student.forward(step=step, noisy417=noisy, clean417=seen, episode_length_before=env.episode_length_buf.clone(),
                                context={}, reset_before=previous_dones)
      return joint_action.from_plain_action(plain)

  if args.zero_policy:                                  # V1: PD targets at the default pose from the first step
    zero_width = env.action_manager.action.shape[1]

    def act(step, previous_dones):
      common.advance(step, raw[:, step, :3])
      return torch.zeros(count, zero_width, device=env.device)

  names = list(entity.joint_names)
  body = [names.index(n) for n in R.JOINTS]
  group_ids = {g: [i for i, n in enumerate(R.JOINTS) if any(k in n for k in keys)]
               for g, keys in (("legs", ("hip", "knee", "ankle")), ("waist", ("waist",)), ("arms", ("shoulder", "elbow", "wrist")))}
  torso = entity.body_names.index("torso_link")
  hits = [[h for h in s["env"]["disturbances"] if h["kind"] == "impulse"] for s in specs_env]
  width = max([len(h) for h in hits] + [1])
  hit = {k: torch.zeros(count, width, device=env.device) for k in ("start", "count", "gap", "impulse", "azimuth", "height")}
  hit["start"][:] = -1
  for e, row in enumerate(hits):
    for k, h in enumerate(row):
      hit["start"][e, k] = round(h["t_s"] / DT)
      hit["count"][e, k] = h["burst_count"]
      hit["gap"][e, k] = round(h["burst_gap_s"] / DT)
      hit["impulse"][e, k], hit["azimuth"][e, k], hit["height"][e, k] = h["impulse_ns"], h["azimuth_rad"], h["height_m"]
  throwers = {}
  if has_throws:
    throwers = {kind: world.Thrower(env, names, kind) for kind, names in cfg._negative_throwables.items()}
  releases = {}
  for e, s in enumerate(specs_env):
    for h in s["env"]["disturbances"]:
      if h["kind"] == "throw":
        releases.setdefault(round(h["t_s"] / DT), []).append((e, h))
  T = horizon
  rec = {k: torch.zeros(T, count, w, device=env.device) for k, w in
         (("xy", 2), ("yaw", 1), ("height", 1), ("tilt", 1), ("jerk", 4), ("delivered", 3), ("action_rate", 1), ("force", 1),
          ("pelvis_v", 1), ("foot_v", 2), ("foot_h", 2))}
  # recovery / settling (R2 gate decision 5): pelvis horizontal speed, each foot's speed (ankle roll link) and each foot's
  # lowest sole point above the patch surface (14 foot capsules, 3 samples each); contact is derived in the analysis
  from r1wb.eval.case_pack import _foot_geoms
  foot_geoms = _foot_geoms(env.sim.mj_model)
  foot_names = [__import__("mujoco").mj_id2name(env.sim.mj_model, __import__("mujoco").mjtObj.mjOBJ_GEOM, g) for g in foot_geoms]
  foot_side = torch.tensor([0 if "left" in n else 1 for n in foot_names], device=env.device)
  foot_ids = torch.tensor(foot_geoms, device=env.device)
  foot_radius = torch.tensor(env.sim.mj_model.geom_size[foot_geoms, 0], device=env.device, dtype=torch.float)
  foot_half = torch.tensor(env.sim.mj_model.geom_size[foot_geoms, 1], device=env.device, dtype=torch.float)
  ankle_ids = [entity.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
  surface_slope = slopes.clone()
  if args.record_feet:
    rec["com"] = torch.zeros(T, count, 6, device=env.device)                 # x, y, z, vx, vy, vz (world, whole body)
    rec["sole"] = torch.zeros(T, count, 2 * len(foot_geoms) * 3, device=env.device)
    body_mass = torch.as_tensor(env.sim.mj_model.body_mass[entity.indexing.body_ids.cpu().numpy()], device=env.device,
                                dtype=torch.float)
    if "body_mass" in getattr(env.sim, "_expanded_fields", set()):
      body_mass = env.sim.model.body_mass[:, entity.indexing.body_ids.long()]
    sole_order = torch.argsort(foot_side, stable=True)                          # left capsules first, then right
    if args.record_bodies:
      parts = ("left_wrist_roll_link", "right_wrist_roll_link", "left_elbow_link", "right_elbow_link",
               "left_ankle_roll_link", "right_ankle_roll_link", "torso_link")
      part_ids = [entity.body_names.index(n) for n in parts]
      arm_ids = [i for i, n in enumerate(entity.body_names) if any(k in n for k in ("shoulder", "elbow", "wrist"))]
      rec["part_v"] = torch.zeros(T, count, len(parts), device=env.device)    # m/s, the order of `parts`
      rec["ke"] = torch.zeros(T, count, 2, device=env.device)                 # J: arms, whole body (translational)
      pelvis_id = entity.body_names.index("pelvis")
      rec["part_p"] = torch.zeros(T, count, (len(parts) + 1) * 3, device=env.device)  # world xyz of `parts` + pelvis
  if driver is not None:
    rec["plate"] = torch.zeros(T, count, 3, device=env.device)          # plate displacement from its rest point (m)
  if driver is not None or args.record_actions:
    rec["qpos"] = torch.zeros(T, count, len(body), device=env.device)   # the 24 body joints (residual jerk, offsets)
  if args.record_actions:
    rec["action"] = torch.zeros(T, count, env.action_manager.action.shape[1], device=env.device)
  if args.record_root:
    rec["root"] = torch.zeros(T, count, 7, device=env.device)         # pelvis xyz (minus origin), quaternion wxyz
  if args.record_hit:
    rec["wrench"] = torch.zeros(T, count, 6, device=env.device)       # torso external force xyz, torque xyz
  contact = None
  if args.record_contacts:                              # V1: contacts, ground reaction, angular momentum (off by default)
    import mujoco_warp as mjw
    wd = env.sim.wp_data
    cap = int(wd.contact.pos.shape[0])
    contact = dict(
      ids=wp.array(np.arange(cap, dtype=np.int32), dtype=int, device=env.sim.wp_device),
      out=wp.zeros(cap, dtype=wp.spatial_vector, device=env.sim.wp_device), cap=cap, peak=0, slots=CONTACT_SLOTS)
    contact.update(pos=wp.to_torch(wd.contact.pos), geom=wp.to_torch(wd.contact.geom), world=wp.to_torch(wd.contact.worldid),
                   type=wp.to_torch(wd.contact.type), efc=wp.to_torch(wd.contact.efc_address), nacon=wp.to_torch(wd.nacon),
                   force=wp.to_torch(contact["out"]), index=torch.arange(cap, device=env.device))
    robot_bodies = entity.indexing.body_ids.long()
    contact["geom_body"] = torch.as_tensor(env.sim.mj_model.geom_bodyid, device=env.device, dtype=torch.long)
    contact["local"] = torch.full((env.sim.mj_model.nbody,), -1, dtype=torch.long, device=env.device)
    contact["local"][robot_bodies] = torch.arange(len(robot_bodies), device=env.device)
    expanded = getattr(env.sim, "expanded_fields", None) or getattr(env.sim, "_expanded_fields", set())
    contact["robot_bodies"] = robot_bodies
    contact["mass"] = (env.sim.model.body_mass[:, robot_bodies] if "body_mass" in expanded else
                       torch.as_tensor(env.sim.mj_model.body_mass[robot_bodies.cpu().numpy()], device=env.device, dtype=torch.float)[None])
    contact["inertia"] = (env.sim.model.body_inertia[:, robot_bodies] if "body_inertia" in expanded else
                          torch.as_tensor(env.sim.mj_model.body_inertia[robot_bodies.cpu().numpy()], device=env.device,
                                      dtype=torch.float)[None])
    contact["max_env"] = torch.zeros(count, dtype=torch.long, device=env.device)
    rec["contact"] = torch.zeros(T, count, CONTACT_SLOTS * 7, device=env.device)  # slot: x y z (world), fx fy fz on robot, body
    rec["ncontact"] = torch.zeros(T, count, 1, device=env.device)                 # robot-environment contacts (all, not capped)
    rec["grf"] = torch.zeros(T, count, 5, device=env.device)                      # total force on the robot xyz, CoP xy (world)
    rec["angmom"] = torch.zeros(T, count, 3, device=env.device)                   # about the whole-body COM, world frame

    def record_contacts(step: int) -> None:
      c = contact
      with wp.ScopedDevice(env.sim.wp_device):
        mjw.contact_force(env.sim.wp_model, wd, c["ids"], True, c["out"])
      live = (c["index"] < c["nacon"][0]) & ((c["type"] & 1) != 0) & (c["efc"][:, 0] >= 0) & (c["geom"].amin(dim=1) >= 0)
      g = c["geom"].long().clamp_min(0)
      b0, b1 = c["local"][c["geom_body"][g[:, 0]]], c["local"][c["geom_body"][g[:, 1]]]
      keep = (live & ((b0 >= 0) ^ (b1 >= 0))).nonzero().squeeze(1)
      w = c["world"][keep].long()
      robot_second = b1[keep] >= 0                      # the frame normal points from geom 0 to geom 1
      force = torch.where(robot_second, 1.0, -1.0)[:, None] * c["force"][keep, :3]
      pos = c["pos"][keep]
      part = torch.where(robot_second, b1[keep], b0[keep]).float()
      order_w = torch.argsort(w, stable=True)
      ws = w[order_w]
      per_env = torch.bincount(ws, minlength=count)
      rank = torch.arange(len(ws), device=env.device) - (torch.cumsum(per_env, 0) - per_env)[ws]
      fit = rank < CONTACT_SLOTS
      slots = torch.zeros(count, CONTACT_SLOTS, 7, device=env.device)
      slots[ws[fit], rank[fit]] = torch.cat([pos[order_w], force[order_w], part[order_w, None]], dim=1)[fit]
      rec["contact"][step] = slots.reshape(count, -1)
      rec["ncontact"][step, :, 0] = per_env.float()
      c["max_env"] = torch.maximum(c["max_env"], per_env)
      total = torch.zeros(count, 3, device=env.device).index_add_(0, w, force)
      fz = force[:, 2].clamp_min(0.0)
      load = torch.zeros(count, device=env.device).index_add_(0, w, fz)
      moment = torch.zeros(count, 2, device=env.device).index_add_(0, w, pos[:, :2] * fz[:, None])
      cop = torch.where(load[:, None] > 1e-6, moment / load.clamp_min(1e-6)[:, None], torch.full_like(moment, float("nan")))
      rec["grf"][step] = torch.cat([total, cop], dim=1)
      c["peak"] = max(c["peak"], int(c["nacon"][0]))
      data_b = entity.data
      mass_b, inertia_b = c["mass"], c["inertia"]
      rot = env.sim.data.ximat[:, c["robot_bodies"]].reshape(count, -1, 3, 3)
      p_b, v_b, w_b = data_b.body_com_pos_w, data_b.body_com_lin_vel_w, data_b.body_com_ang_vel_w
      whole = mass_b.sum(dim=1, keepdim=True)
      p_c = (p_b * mass_b[..., None]).sum(dim=1) / whole
      v_c = (v_b * mass_b[..., None]).sum(dim=1) / whole
      spin = torch.einsum("ebij,ebj->ebi", rot, inertia_b * torch.einsum("ebji,ebj->ebi", rot, w_b))
      orbit = mass_b[..., None] * torch.cross(p_b - p_c[:, None], v_b - v_c[:, None], dim=-1)
      rec["angmom"][step] = (spin + orbit).sum(dim=1)
  # planted tremor (validity check): a sine on every joint target, from --plant-tremor (all envs) or the spec's
  # env.plant_tremor {hz, rad} (per env); joint-target radians are converted to action units by the action scale
  plant = [s["env"].get("plant_tremor") for s in specs_env]
  if args.plant_tremor:
    hz, rad = (float(x) for x in args.plant_tremor.split(","))
    plant = [dict(hz=hz, rad=rad)] * count
  tremor = None
  if any(plant):
    tremor = (torch.tensor([[p["hz"] if p else 0.0] for p in plant], device=env.device),
              torch.tensor([[p["rad"] if p else 0.0] for p in plant], device=env.device) / joint_action._scale)
  alive = torch.ones(count, dtype=torch.bool, device=env.device)
  fell_step = torch.full((count,), -1, dtype=torch.long, device=env.device)
  reason = torch.zeros(count, dtype=torch.long, device=env.device)
  ends_t = torch.as_tensor(ends, device=env.device)
  history = [entity.data.joint_pos[:, body].clone() for _ in range(3)]
  previous_action, previous_dones = None, None
  frozen = torch.zeros(count, dtype=torch.bool, device=env.device)
  down_steps = torch.zeros(count, dtype=torch.long, device=env.device)
  frozen_action = None
  first_hit = torch.where(hit["start"] >= 0, hit["start"], torch.full_like(hit["start"], 1.0e9)).amin(dim=1)
  hit_frozen = torch.zeros(count, dtype=torch.bool, device=env.device)
  hit_action = None
  step_times = []
  for step in range(T):
    t0 = time.perf_counter()
    sensors.begin(step)
    with torch.no_grad():
      action = act(step, previous_dones)
      if tremor is not None:
        action = action + tremor[1] * torch.sin(2.0 * math.pi * tremor[0] * step * DT)
      if args.freeze_after_fall is not None and step > 0 and previous_action is not None:
        down = (rec["tilt"][step - 1, :, 0] > 1.0) | (rec["height"][step - 1, :, 0] < 0.35)
        down_steps = torch.where(down, down_steps + 1, torch.zeros_like(down_steps))
        newly = (~frozen) & (down_steps >= round(args.freeze_after_fall / DT))
        if frozen_action is None:
          frozen_action = previous_action.clone()
        frozen_action[newly] = previous_action[newly]
        frozen |= newly
        action = torch.where(frozen[:, None], frozen_action, action)
      if args.freeze_at_hit and previous_action is not None:     # diagnostic: policy off from the hit step on
        newly = (~hit_frozen) & (step >= first_hit)
        if hit_action is None:
          hit_action = previous_action.clone()
        hit_action[newly] = previous_action[newly]
        hit_frozen |= newly
        action = torch.where(hit_frozen[:, None], hit_action, action)
      since = step - hit["start"]
      active = (hit["start"] >= 0) & (since >= 0) & (since < hit["count"] * hit["gap"]) & (torch.remainder(since, hit["gap"].clamp_min(1)) == 0)
      yaw = 2.0 * torch.atan2(yaw_quat(entity.data.root_link_quat_w)[:, 3], yaw_quat(entity.data.root_link_quat_w)[:, 0])
      angle = yaw[:, None] + hit["azimuth"]
      size = torch.where(active, hit["impulse"] / DT, torch.zeros_like(hit["impulse"]))
      force = torch.stack([size * torch.cos(angle), size * torch.sin(angle), torch.zeros_like(size)], dim=2)   # (envs, K, 3)
      arm = torch.zeros_like(force)
      arm[..., 2] = hit["height"]
      forces = force.sum(dim=1, keepdim=True)
      torques = torch.cross(arm, force, dim=2).sum(dim=1, keepdim=True)
      entity.write_external_wrench_to_sim(forces, torques, body_ids=[torso])
      for thrown in ("ball", "box"):                    # not `kind`: that names the policy kind (meta below)
        due = [(e, h) for e, h in releases.get(step, []) if h["body"] == thrown]
        if due:
          ids = torch.tensor([e for e, _ in due], device=env.device)
          az = torch.tensor([h["azimuth_rad"] for _, h in due], device=env.device)
          along = torch.stack([torch.cos(yaw[ids] + az), torch.sin(yaw[ids] + az), torch.zeros_like(az)], dim=1)
          target = entity.data.body_com_pos_w[ids, torso].clone()
          target[:, 2] += torch.tensor([h["height_m"] for _, h in due], device=env.device)
          throwers[thrown].throw(ids, target, along, torch.tensor([h["speed_mps"] for _, h in due], device=env.device),
                               entity.data.body_com_lin_vel_w[ids, torso])
      _, _, dones, _ = wrapped.step(action)
      for thrower in throwers.values():
        thrower.after_step(DT)
      dones = dones.bool()
      terminated = env.termination_manager.terminated
      failed = alive & dones & terminated
      if failed.any():
        code = torch.zeros_like(reason)
        for i, name in enumerate(("fallen", "trunk_ground"), start=1):
          try:
            code = torch.where(env.termination_manager.get_term(name).bool() & (code == 0), i, code)
          except KeyError:
            pass
        reason = torch.where(failed, code, reason)
        fell_step = torch.where(failed, torch.full_like(fell_step, step), fell_step)
      alive &= ~dones
      data = entity.data
      q = data.joint_pos[:, body]
      jerk = (q - 3.0 * history[0] + 3.0 * history[1] - history[2]) / DT ** 3
      history = [q.clone(), history[0], history[1]]
      rec["jerk"][step, :, 0] = jerk.pow(2).mean(1).sqrt()
      for j, g in enumerate(("legs", "waist", "arms"), start=1):
        rec["jerk"][step, :, j] = jerk[:, group_ids[g]].pow(2).mean(1).sqrt()
      rec["xy"][step] = data.root_link_pos_w[:, :2] - origins[:, :2]
      rec["yaw"][step, :, 0] = 2.0 * torch.atan2(yaw_quat(data.root_link_quat_w)[:, 3], yaw_quat(data.root_link_quat_w)[:, 0])
      rec["height"][step, :, 0] = world.standing_height(env)
      rec["tilt"][step, :, 0] = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
      rec["delivered"][step] = common.delivered
      if previous_action is not None:
        rec["action_rate"][step, :, 0] = (action - previous_action)[:, :24].pow(2).mean(1).sqrt()
      rec["force"][step, :, 0] = torch.norm(forces[:, 0], dim=1)
      if args.record_root:
        rec["root"][step, :, :3] = data.root_link_pos_w - origins
        rec["root"][step, :, 3:] = data.root_link_quat_w
      if args.record_hit:
        rec["wrench"][step, :, :3] = forces[:, 0]
        rec["wrench"][step, :, 3:] = torques[:, 0]
      rec["pelvis_v"][step, :, 0] = torch.norm(data.root_link_lin_vel_w[:, :2], dim=1)
      rec["foot_v"][step] = torch.norm(data.body_link_lin_vel_w[:, ankle_ids], dim=2)
      centre = env.sim.data.geom_xpos[:, foot_ids]
      axis = env.sim.data.geom_xmat[:, foot_ids].reshape(count, len(foot_geoms), 3, 3)[..., :, 2] * foot_half[None, :, None]
      sole = torch.stack([centre - axis, centre, centre + axis], dim=2)                   # (envs, 14, 3, 3)
      surface = origins[:, 2, None, None] + surface_slope[:, None, None] * (sole[..., 1] - origins[:, 1, None, None])
      low = (sole[..., 2] - surface).amin(dim=2) - foot_radius[None, :]                  # (envs, 14)
      for side in (0, 1):
        rec["foot_h"][step, :, side] = low[:, foot_side == side].amin(dim=1)
      if args.record_feet:
        mass = body_mass if body_mass.dim() == 2 else body_mass[None, :]
        total = mass.sum(dim=1, keepdim=True)
        rec["com"][step, :, :3] = (data.body_com_pos_w * mass[..., None]).sum(dim=1) / total
        rec["com"][step, :, 3:] = (data.body_com_lin_vel_w * mass[..., None]).sum(dim=1) / total
        tips = torch.stack([sole[:, :, 0], sole[:, :, 2]], dim=2)[:, sole_order]   # (envs, 14, 2 capsule ends, 3)
        height = tips[..., 2] - (origins[:, 2, None, None] + surface_slope[:, None, None] * (tips[..., 1] - origins[:, 1, None, None])) \
          - foot_radius[sole_order][None, :, None]
        rec["sole"][step] = torch.cat([tips[..., :2], height[..., None]], dim=3).reshape(count, -1)
        if args.record_bodies:
          rec["part_v"][step] = torch.linalg.vector_norm(data.body_link_lin_vel_w[:, part_ids], dim=2)
          ke = 0.5 * mass * (data.body_com_lin_vel_w ** 2).sum(dim=2)
          rec["ke"][step, :, 0] = ke[:, arm_ids].sum(dim=1)
          rec["ke"][step, :, 1] = ke.sum(dim=1)
          pos = data.body_link_pos_w[:, part_ids + [pelvis_id]] - origins[:, None, :]
          rec["part_p"][step] = pos.reshape(count, -1)
      if driver is not None:
        rec["plate"][step] = driver.platform.data.joint_pos - driver.rest
      if "qpos" in rec:
        rec["qpos"][step] = q
      if "action" in rec:
        rec["action"][step] = action
      if contact is not None:
        record_contacts(step)
      previous_action, previous_dones = action.clone(), dones.clone()
    step_times.append(time.perf_counter() - t0)
    if step % 100 == 0:
      print(f"step {step}/{T} alive {int(alive.sum())}/{count} {np.mean(step_times[-100:]) * 1e3:.1f} ms/step", flush=True)
  entity.write_external_wrench_to_sim(torch.zeros(count, 1, 3, device=env.device), torch.zeros(count, 1, 3, device=env.device), body_ids=[torso])
  inverse = np.argsort(order)                          # back to instance order
  arrays = {k: v.permute(1, 0, 2).cpu().numpy()[inverse] for k, v in rec.items()}
  arrays.update(fell_step=fell_step.cpu().numpy()[inverse], reason=reason.cpu().numpy()[inverse],
                end_step=np.asarray(ends)[inverse], step_index=np.stack(steps_of)[inverse],
                spawn_lift=lift.cpu().numpy()[inverse],
                **({k: v.cpu().numpy()[inverse] for k, v in driver.transmissibility().items()} if driver is not None else {}),
                **({"total_mass_kg": (body_mass.sum(dim=-1) * torch.ones(count, device=env.device)).cpu().numpy()[inverse],
                    "origin_z": origins[:, 2].cpu().numpy()[inverse]} if args.record_feet else {}),
                **({"origin_xyz": origins.cpu().numpy()[inverse],
                    "contact_max_per_env": contact["max_env"].cpu().numpy()[inverse]} if contact is not None else {}),
                throw_given_ns=(torch.stack([t.given for t in throwers.values()]).flatten(0, 1).nan_to_num(-1.0).amax(0)
                                if throwers else torch.full((count,), -1.0, device=env.device)).cpu().numpy()[inverse],
                spec_sha256=np.asarray([s["spec_sha256"] for s in specs_env])[inverse])
  np.savez_compressed(args.out / "arrays.npz", **arrays)
  meta = dict(schema="r1adv-negative-rollout-v1", policy=args.policy, identity={k: v for k, v in info.items()},
              checkpoint=str(checkpoint(args.policy)) if checkpoint(args.policy) else None,
              helpers=str(helpers) if helpers else None, specs=str(args.specs.resolve()), specs_set_sha256=set_digest,
              count=count, horizon=T, permute=args.permute, seed=seed, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              env_build_s=round(built, 1), total_s=None, step_ms_mean=round(1e3 * float(np.mean(step_times[5:] or step_times)), 2),
              s2_update_s=round(stats["decision_s"], 1), geometry_sha256=geometry_sha256(env.sim.mj_model),
              versions=dict(python=platform.python_version(), torch=torch.__version__, warp=wp.__version__,
                            mujoco=__import__("mujoco").__version__),
              runner_sha256=IMPORTED["runner_sha256"], package_sha256=IMPORTED["package_sha256"], warp_run_to_run=RUN_TO_RUN,
              sim_device=args.device,
              actions="deterministic policy mean (never sampled)", body=body_values,
              protocol=dict(start=args.start, action_delay_ms=args.action_delay_ms, plant_tremor=args.plant_tremor,
                            sym_branch=args.sym_branch, record_root=args.record_root, record_hit=args.record_hit,
                            foot_lead=args.foot_lead, freeze_after_fall=args.freeze_after_fall,
                            zero_policy=args.zero_policy, freeze_at_hit=args.freeze_at_hit, record_contacts=args.record_contacts,
                            actor_clean=args.actor_clean,
                            sensor_profiles=sorted({sense_of(s)["profile"] for s in specs_env}),
                            joint_vel_models=sorted({sense_of(s)["joint_vel_model"] for s in specs_env})),
              extensions=dict(throws=has_throws, body=has_body, vibrating_floor=has_platform,
                              platform=dict(PLATFORM, drive="kinematic, every physics substep, secant velocity") if has_platform else None),
              atlas=dict(path=str(args.atlas), sha256=sha256(args.atlas)) if args.atlas else None,
              max_memory_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
              physics_determinism="MuJoCo-Warp is not bitwise deterministic run to run; compare repeats as distributions")
  if info["kind"] == "deploy":
    meta["deploy_width"] = int(deploy_width)
  if contact is not None:
    meta["contacts"] = dict(capacity=contact["cap"], peak_nacon=contact["peak"], slots=CONTACT_SLOTS,
                            max_per_env=int(contact["max_env"].max()), body_names=list(entity.body_names),
                            slot_layout="x y z (world) | fx fy fz on the robot (world, N) | robot body index",
                            source="mujoco_warp contact_force after the step's forward(); robot-environment pairs, efc address >= 0",
                            grf="sum of the slot forces over all contacts | CoP = sum(p f_z+) / sum(f_z+), NaN without load",
                            angmom="sum_b R_b I_b R_b^T w_b + m_b (c_b - c) x (v_b - v_c), world frame")
  meta["total_s"] = round(time.time() - started, 1)
  (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
  print(json.dumps({k: meta[k] for k in ("policy", "count", "horizon", "env_build_s", "step_ms_mean", "total_s")}), flush=True)
  env.close()


if __name__ == "__main__":
  main()
