"""R1 Hold mission (round R01, family 1 of the charter curriculum): stand in the human reference posture on flat ground.

Reward: exactly the fixed objective (objective.py)  r = L + beta * D. Falls end the episode (physical only).
  L channels (Hold): posture RMS over the 24 body joints to the target, pelvis height, pelvis tilt to the target
  attitude, planar root speed (0.4 s mean), yaw rate.  D: fixed human-motion density (human_surprise.py).
Input files: pinned by sha256 in objective.HOLD (D d_v0b.npz; Hold target hold_target_v0c.json = coordinator decision 3
  + the static-margin fix), server copy /rl-agent/.agent/r1adv_inputs/, PC originals .agent/adversarial/positive/models/.
Actor (bracket, coordinator decision 4/6):
  prev   the agroid deploy contract: 10 frames x [gyro, gravity, twist command, q - q0, dq, previous action] = 810
         (r1lab frame_stack with its sensor model: gyro bias/scale/spikes, attitude bias drift, encoder noise, latency)
  noprev the same frames without the previous action = 570
Both: left-right symmetric by construction (hold_model.HoldSymmetricMLP). Critic: clean frames + privileged state.
Command (actuator) delay 0-40 ms randomised (0-8 physics steps, redrawn every 2.5 s, held in between): +40 ms made the
R00 teachers oscillate.
Nothing here is a rule in the control path: the deployed network maps its inputs to joint targets.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from r1lab import contract as LC
from r1lab.mdp import observations as r1obs
from r1lab.mdp.actions import PolicyJointPositionActionCfg
from r1wb import robot
from r1wb.tracking.env_cfg import sensors as wb_sensors

from r1adv.positive import objective as OBJ
from r1adv.positive.human_surprise import HumanSurprise

EXPERIMENT = "r1_r1adv_hold"
INPUT_DIRS = (Path("/rl-agent/.agent/r1adv_inputs"), Path(__file__).resolve().parents[4] / ".agent/adversarial/positive/models")
# The mission constants and the pinned input files are part of the formula file (objective.HOLD): its sha256 locks them.
PINNED = {OBJ.HOLD["d_file"]: OBJ.HOLD["d_sha256"], OBJ.HOLD["target_file"]: OBJ.HOLD["target_sha256"],
          OBJ.HOLD_R3["d_file"]: OBJ.HOLD_R3["d_sha256"]}                  # v2 density (R3 on), resolvable now
SIGMA = dict(OBJ.HOLD["sigma"])
FORM = OBJ.HOLD["form"]
DELAY_PHYSICS_STEPS = 8                     # 0-40 ms command delay
DELAY_UPDATE_PHYSICS_STEPS = 500            # the lag is redrawn every 2.5 s (per-env phase) and held in between: a latency
                                            # that varies slowly, not a jitter (mjlab's default redraws every physics step)


def pinned(name: str) -> Path:
  for d in INPUT_DIRS:
    p = d / name
    if p.is_file():
      digest = hashlib.sha256(p.read_bytes()).hexdigest()
      if digest != PINNED[name]:
        raise ValueError(f"{p}: sha256 {digest} differs from the pinned {PINNED[name]}")
      return p
  raise FileNotFoundError(f"{name} not found in {INPUT_DIRS}")


class frame_stack_noprev(r1obs.frame_stack):
  """The deploy frame stack without the previous-action block (57 values a frame, 570 in all)."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.hist = torch.zeros(self.n, LC.HISTORY, 57, device=self.dev)

  def __call__(self, env, command_name: str, noise: bool = True, noise_params: dict | None = None) -> torch.Tensor:
    del noise, noise_params
    step = env.common_step_counter
    ids = self.env_ids if step != self.last_step else self.needs_fill.nonzero(as_tuple=False).flatten()
    if ids.numel():
      measured = self._measure(env, ids)
      prev = torch.where(self.needs_fill[ids, None], measured, self.prev_sensor[ids])
      self.prev_sensor[ids] = measured
      self.sensor[ids] = ((1.0 - self.lag[ids]) * measured + self.lag[ids] * prev if self.noise else measured)
    sensor = self.sensor
    cmd = env.command_manager.get_command(command_name)[:, :3]
    frame = torch.cat([sensor[:, :6], cmd, sensor[:, 6:30], sensor[:, 30:54]], dim=-1)
    if step != self.last_step:
      self.hist = torch.roll(self.hist, shifts=-1, dims=1)
    self.hist[:, -1] = frame
    if self.needs_fill.any():
      fill = self.needs_fill.nonzero(as_tuple=False).flatten()
      self.hist[fill] = frame[fill].unsqueeze(1).expand(-1, LC.HISTORY, -1)
      self.needs_fill[fill] = False
    self.last_step = step
    return self.hist.reshape(self.n, LC.HISTORY * 57)


class HoldObjective:
  """Reward term: r = L + beta * D (objective.py), with the Hold mission channels and the human-motion density."""

  def __init__(self, cfg: RewardTermCfg, env):
    p = cfg.params
    self.robot = env.scene["robot"]
    dev = env.device
    target = json.loads(pinned(OBJ.HOLD["target_file"]).read_text())
    if list(target["joint_names"]) != list(self.robot.joint_names)[:24]:
      raise ValueError("Hold target joint order differs from the robot")
    self.q_star = torch.tensor(target["target_rad"], dtype=torch.float32, device=dev)
    self.h_star = float(target["pelvis_height_m"])
    qw, qx, qy, qz = target["qpos"][3:7]
    # gravity in the pelvis frame of the target attitude: R^T (0, 0, -1)
    self.g_star = torch.tensor([-2 * (qx * qz - qw * qy), -2 * (qy * qz + qw * qx), -(1 - 2 * (qx * qx + qy * qy))],
                               dtype=torch.float32, device=dev)
    self.surprise = HumanSurprise(dict(np.load(pinned(OBJ.HOLD["d_file"]))), env.num_envs, device=dev)
    self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=dev)
    self.v_mean = torch.zeros(env.num_envs, 2, device=dev)
    self.dt = env.step_dt
    self.env = env
    if p.get("formula_sha256") != OBJ.formula_sha256():
      raise ValueError("objective.py differs from the formula recorded in this task's configuration")

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = slice(None)
    self.fresh[env_ids] = True
    self.v_mean[env_ids] = 0.0

  def channels(self) -> dict[str, torch.Tensor]:
    d = self.robot.data
    q = d.joint_pos[:, :24]
    posture = (q - self.q_star).pow(2).mean(1).sqrt()
    height = d.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2] - self.h_star
    g = d.projected_gravity_b
    tilt = torch.acos((g * self.g_star).sum(1).clamp(-1.0, 1.0))
    self.v_mean += (d.root_link_lin_vel_w[:, :2] - self.v_mean) * (self.dt / 0.4)
    speed = self.v_mean.norm(dim=1)
    yaw = d.root_link_ang_vel_w[:, 2].abs()
    return {"posture": posture, "height": height.abs(), "tilt": tilt, "speed": speed, "yaw": yaw}

  def __call__(self, env, beta: float, sigma: dict, form: str, formula_sha256: str = "") -> torch.Tensor:
    d = self.robot.data
    q, quat = d.joint_pos[:, :24], d.root_link_quat_w
    if bool(self.fresh.any()):
      ids = self.fresh.nonzero(as_tuple=False).flatten()
      self.surprise.reset(ids, q[ids], quat[ids])
      self.fresh[ids] = False
    self.surprise.push(q, quat)
    natural, s = self.surprise.naturalness(form)
    ch = self.channels()
    like = OBJ.mission_likelihood(ch, sigma, form)
    r = OBJ.reward(like, natural, beta)
    log = env.extras.setdefault("log", {})
    log["Objective/L"] = like.mean()
    log["Objective/D"] = natural.mean()
    log["Objective/S_median"] = s.median()
    for k, v in ch.items():
      log[f"Objective/err_{k}"] = v.mean()
    return r


def physical_fall(env, limit_angle: float = 1.0, min_height: float = 0.35) -> torch.Tensor:
  d = env.scene["robot"].data
  tilted = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)) > limit_angle
  low = (d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]) < min_height
  trunk = env.scene["trunk_ground"].data.found.reshape(env.num_envs, -1).any(dim=1)
  return tilted | low | trunk


class objective_channels:
  """Critic: the Hold mission state the reward is about (privileged): posture RMS to the target, pelvis height error,
  planar root velocity. Its own copy of the target (independent of the reward term's construction order)."""

  def __init__(self, cfg, env):
    target = json.loads(pinned(OBJ.HOLD["target_file"]).read_text())
    self.q_star = torch.tensor(target["target_rad"], dtype=torch.float32, device=env.device)
    self.h_star = float(target["pelvis_height_m"])

  def __call__(self, env) -> torch.Tensor:
    d = env.scene["robot"].data
    posture = (d.joint_pos[:, :24] - self.q_star).pow(2).mean(1, keepdim=True).sqrt()
    height = (d.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]) - self.h_star
    return torch.cat([posture, height, d.root_link_lin_vel_w[:, :2]], dim=1)


def hold_env_cfg(prev_action: bool = True, beta: float = 1.0, play: bool = False) -> ManagerBasedRlEnvCfg:
  if float(beta) not in tuple(float(v) for v in OBJ.HOLD["beta_bracket"]):
    raise ValueError(f"beta {beta} is not one of the formula's values {OBJ.HOLD['beta_bracket']}")
  cfg = make_velocity_env_cfg()
  cfg.sim.njmax, cfg.sim.nconmax = 300, None
  cfg.sim.mujoco.ccd_iterations, cfg.sim.contact_sensor_maxmatch = 50, 64
  entity = robot.robot_cfg(delay_steps=0 if play else DELAY_PHYSICS_STEPS)
  if not play:
    entity.articulation.actuators = tuple(dataclasses.replace(a, delay_hold_prob=0.0, delay_update_period=DELAY_UPDATE_PHYSICS_STEPS)
                                          for a in entity.articulation.actuators)
  cfg.scene.entities = {"robot": entity}
  cfg.scene.terrain.terrain_type, cfg.scene.terrain.terrain_generator = "plane", None
  cfg.scene.sensors = wb_sensors()                        # self_collision, feet_ground, trunk_ground
  cfg.episode_length_s = 10.0 if not play else 20.0
  # zero twist command (Hold): the frame stack reads it like the deploy contract's command slot
  cfg.commands = {"twist": UniformVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(1.0e9, 1.0e9), rel_standing_envs=1.0, rel_heading_envs=0.0,
    heading_command=False, debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)))}
  frames = r1obs.frame_stack if prev_action else frame_stack_noprev
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={"frames": ObservationTermCfg(func=frames, params={"command_name": "twist", "noise": not play})},
      concatenate_terms=True, enable_corruption=not play),
    "critic": ObservationGroupCfg(
      terms={"frames": ObservationTermCfg(func=frames, params={"command_name": "twist", "noise": False}),
             "base_lin_vel": ObservationTermCfg(func=r1obs.base_lin_vel),
             "base_height": ObservationTermCfg(func=r1obs.base_height),
             "feet": ObservationTermCfg(func=mdp.foot_contact, params={"sensor_name": "feet_ground"}),
             "channels": ObservationTermCfg(func=objective_channels)},
      concatenate_terms=True, enable_corruption=False)}
  cfg.actions = {"joint_pos": PolicyJointPositionActionCfg(entity_name="robot")}
  geoms = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))
  events = {k: cfg.events[k] for k in ("reset_base", "reset_robot_joints", "foot_friction", "encoder_bias", "base_com")}
  events["reset_base"].params["pose_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.02), "yaw": (-math.pi, math.pi)}
  events["reset_base"].params["velocity_range"] = {} if play else {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1)}
  events["reset_robot_joints"].params["position_range"] = (0.0, 0.0) if play else (-0.1, 0.1)
  events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0) if play else (-0.2, 0.2)
  events["foot_friction"].params["asset_cfg"].geom_names = geoms
  events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  if play:
    for k in ("foot_friction", "encoder_bias", "base_com"):
      events.pop(k)
  cfg.events = events
  cfg.rewards = {"objective": RewardTermCfg(func=HoldObjective, weight=1.0,
                                            params={"beta": float(beta), "sigma": dict(SIGMA), "form": FORM,
                                                    "formula_sha256": OBJ.formula_sha256()})}
  cfg.terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
                      "fall": TerminationTermCfg(func=physical_fall)}
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.viewer.body_name = "torso_link"
  return cfg


def hold_rl_cfg(max_iterations: int = 1500) -> RslRlOnPolicyRunnerCfg:
  actor = RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"})
  actor.class_name = "r1adv.positive.hold_model:HoldSymmetricMLP"
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name=EXPERIMENT, save_interval=100, num_steps_per_env=24, max_iterations=max_iterations,
    logger="tensorboard", upload_model=False)


def contract_record(prev_action: bool, beta: float) -> dict:
  return dict(formula=OBJ.FORMULA_VERSION, formula_sha256=OBJ.formula_sha256(), locked=OBJ.HOLD["locked"], form=FORM,
              sigma=dict(SIGMA), beta=beta, actor=("deploy contract 810" if prev_action else "no previous action 570"),
              pinned=dict(PINNED), command_delay_ms=[0, 5 * DELAY_PHYSICS_STEPS],
              delay_update_s=DELAY_UPDATE_PHYSICS_STEPS * 0.005, symmetric="hold_model.HoldSymmetricMLP")
