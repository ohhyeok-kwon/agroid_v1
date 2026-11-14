"""Ceiling test of the design bracket: a walking policy from scratch, with no human motion and no anchor.

One question: what does reinforcement learning alone reach on this robot model, in this simulator, under the same
ball-like hits and bursts? It is the far end of the whole design (everything the human-motion pipeline adds is taken
away), so its result bounds what any variant of that pipeline can be asked for, and a policy that passes is a
candidate source of the recovery behaviour (distilled by our own training, never deployed as it is).

The recipe is mjlab's stock velocity-tracking task (rewards, observations, pushes and learning settings as shipped
for the Unitree G1, flat ground), applied to this robot model with its deploy PD gains and action scale. The only
addition is the impulse generator of r1wb.harden as a second command term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from r1wb import robot
from r1wb.harden import DisturbedStreamCommand
from r1wb.student.stream import GoalStreamCommand
from r1wb.tracking.env_cfg import EXPERIMENT, sensors as tracking_sensors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ImpulseCommand(CommandTerm):
  """The hit generator of DisturbedStreamCommand on its own (no command stream, no reference motion)."""
  cfg: ImpulseCommandCfg
  _uniform = GoalStreamCommand._uniform
  _schedule = DisturbedStreamCommand._schedule
  _start_hits = DisturbedStreamCommand._start_hits
  _apply_wrench = DisturbedStreamCommand._apply_wrench

  def __init__(self, cfg: ImpulseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    count = self.num_envs
    self.robot = env.scene[cfg.entity_name]
    self.calm = torch.rand(count, device=self.device) < cfg.calm_fraction
    self.gentle = torch.zeros(count, dtype=torch.bool, device=self.device)
    self.next_hit = torch.full((count,), 1.0e9, device=self.device)
    self.hit_left = torch.zeros(count, dtype=torch.long, device=self.device)
    self.hit_force = torch.zeros(count, 3, device=self.device)
    self.hit_torque = torch.zeros(count, 3, device=self.device)
    self.hit_body = torch.zeros(count, dtype=torch.long, device=self.device)
    self.since_hit = torch.full((count,), 1.0e3, device=self.device)
    self.burst_left = torch.zeros(count, dtype=torch.long, device=self.device)
    self.burst_hard = torch.zeros(count, dtype=torch.bool, device=self.device)
    self.burst_angle = torch.zeros(count, device=self.device)
    self.last_impulse = torch.zeros(count, device=self.device)
    self.hit_impulse_w = torch.zeros(count, 2, device=self.device)
    self.hit_count = torch.zeros(count, dtype=torch.long, device=self.device)
    self.first_impulse = torch.zeros(count, device=self.device)
    self.impulse_scale = 1.0
    self._bodies = [self.robot.body_names.index(name) for name in cfg.hit_bodies]
    self._hits_per_step = int(round(cfg.hit_duration / env.step_dt))
    self.metrics["impulse"] = torch.zeros(count, device=self.device)
    self._nothing = torch.zeros(count, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._nothing

  def _update_metrics(self) -> None:
    self.metrics["impulse"][:] = self.last_impulse

  def _resample_command(self, env_ids: torch.Tensor) -> None:      # only on reset (the resampling time is never reached)
    self.hit_left[env_ids] = 0
    self.burst_left[env_ids] = 0
    self.burst_hard[env_ids] = False
    self.since_hit[env_ids] = 1.0e3
    self.last_impulse[env_ids] = 0.0
    self._schedule(env_ids)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is not None:
      return
    dt = self._env.step_dt
    if self.cfg.impulse_ramp_steps > 0:
      self.impulse_scale = min(1.0, 0.2 + 0.8 * self._env.common_step_counter / self.cfg.impulse_ramp_steps)
    self.next_hit -= dt
    self.since_hit += dt
    due = (self.next_hit <= 0.0).nonzero(as_tuple=False).flatten()
    if len(due):
      self._start_hits(due)
    self._apply_wrench()


@dataclass(kw_only=True)
class ImpulseCommandCfg(CommandTermCfg):
  """Defaults = the "extreme" disturbance set of r1wb.anchored."""
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  calm_fraction: float = 0.2
  impulse_range: tuple[float, float] = (5.0, 45.0)
  impulse_ramp_steps: int = 600 * 24
  hit_interval: tuple[float, float] = (1.5, 4.0)
  hit_duration: float = 0.1
  spike_share: float = 0.7
  hit_height: tuple[float, float] = (-0.1, 0.25)
  hit_bodies: tuple[str, ...] = ("torso_link", "pelvis")
  hit_body_weights: tuple[float, ...] | None = None
  hit_body_scale: tuple[float, ...] | None = None
  burst_prob: float = 0.7
  burst_min: int = 4
  burst_max: int = 12
  burst_gap: tuple[float, float] = (0.08, 0.25)
  burst_impulse: tuple[float, float] = (8.0, 16.0)
  burst_hard_share: float = 0.0
  hard_hits: tuple[int, int] = (6, 9)
  hard_gap: tuple[float, float] = (0.12, 0.22)
  hard_impulse: tuple[float, float] = (10.0, 14.0)
  focus_share: float = 0.0
  focus_half_angle: float = 0.6
  focus_impulse: tuple[float, float] = (18.0, 32.0)
  gentle_impulse: tuple[float, float] = (5.0, 15.0)      # unused here (no gentle environments)
  gentle_interval: tuple[float, float] = (3.0, 6.0)

  def build(self, env: ManagerBasedRlEnv) -> ImpulseCommand:
    return ImpulseCommand(self, env)


def walker_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()
  cfg.sim.njmax, cfg.sim.nconmax = 300, None
  cfg.sim.mujoco.ccd_iterations, cfg.sim.contact_sensor_maxmatch = 50, 64
  cfg.scene.entities = {"robot": robot.robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type, cfg.scene.terrain.terrain_generator = "plane", None
  sites = ("left_foot", "right_foot")
  geoms = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))
  kept = []
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      continue
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in sites)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    kept.append(sensor)
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1,
    track_air_time=True)
  # The whole-body tools' sensors too (feet_ground, trunk_ground, self_collision): same fall criterion and the same
  # balance measures as for every other policy of the family.
  cfg.scene.sensors = tuple(kept) + (feet,) + tracking_sensors()
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.actions["joint_pos"].scale = dict(robot.ACTION_SCALE)
  cfg.viewer.body_name = "torso_link"
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geoms
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15, r".*knee.*": 0.35, r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1, r".*waist_yaw.*": 0.2, r".*waist_roll.*": 0.08, r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15, r".*shoulder_yaw.*": 0.1, r".*elbow.*": 0.15, r".*wrist.*": 0.3}
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.2, r".*hip_yaw.*": 0.2, r".*knee.*": 0.6, r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15, r".*waist_yaw.*": 0.3, r".*waist_roll.*": 0.08, r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2, r".*shoulder_yaw.*": 0.15, r".*elbow.*": 0.35, r".*wrist.*": 0.3}
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  for name in ("foot_clearance", "foot_slip"):
    cfg.rewards[name].params["asset_cfg"].site_names = sites
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0
  cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
                                                 params={"sensor_name": "self_collision", "force_threshold": 10.0})
  cfg.commands["twist"].rel_standing_envs = 0.3              # the hit tests are mostly on a standing or slow robot
  cfg.commands["impulse"] = ImpulseCommandCfg()
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.commands["impulse"].calm_fraction = 1.0
  return cfg


def walker_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=3000,
    logger="tensorboard", upload_model=False)
