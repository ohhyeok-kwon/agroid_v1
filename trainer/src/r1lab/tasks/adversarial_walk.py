"""Human cautious locomotion with a learned, progressively harder environment.

Normal movement should express care through foot placement and weight transfer.
Recovery takes precedence over quietness, exact speed and imitation bonuses.
"""
from dataclasses import dataclass

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from r1lab.mdp.adversarial_env import AdversarialScenario, restore_adversary_material
from r1lab.mdp.human_prior import human_motion_prior
from r1lab.mdp.recovery import RecoveryScaledTerm, recovery_level
from r1lab.mdp.scene_program import CONTINUOUS, EVENTS, OBJECTS, scene_entities
from r1lab.run_record import work_root
from r1lab.tasks import careful_walk, common


PROGRAM_FEATURES = CONTINUOUS + OBJECTS + EVENTS*5 + 3


def scenario_observation(env):
  """Privileged critic input; no generated-scene information reaches the actor."""
  scenario = getattr(env, "_r1_adversary", None)
  if scenario is None or scenario.sample is None:
    return torch.zeros(env.num_envs, PROGRAM_FEATURES, device=env.device)
  sample = scenario.sample
  program = scenario.program
  quat = program["object_quat"]
  w, x, y, z = quat.unbind(-1)
  yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
  pitch = torch.asin((2*(w*y-z*x)).clamp(-1,1))
  robot = env.scene["robot"].data
  rw, rx, ry, rz = robot.root_link_quat_w.unbind(-1)
  robot_yaw = torch.atan2(2*(rw*rz+rx*ry), 1-2*(ry*ry+rz*rz))
  world = program["object_pos"] + program["anchor_offset"][:,None] + env.scene.env_origins[:,None]
  delta = world[..., :2]-robot.root_link_pos_w[:,None,:2]
  cosine, sine = robot_yaw.cos()[:,None], robot_yaw.sin()[:,None]
  local = torch.stack((cosine*delta[...,0]+sine*delta[...,1],
                       -sine*delta[...,0]+cosine*delta[...,1]),-1)
  yaw = yaw-robot_yaw[:,None]
  yaw = torch.atan2(yaw.sin(),yaw.cos())
  objects = torch.cat((local/3.,
      program["object_size"]/.35, yaw[...,None]/torch.pi,
      pitch[...,None]/torch.pi, program["object_friction"][...,None]/1.4), -1)
  objects = objects*program["object_mask"][...,None]
  event_type = program["event_type"].long()
  strength = torch.where(event_type==1,program["ball_speed"]/25.,
      torch.where(event_type==2,program["force_n"]/100.,
      torch.where(event_type==3,program["push_dv"],program["material_friction"]/1.4)))
  events = torch.stack((program["event_time"]/20., program["event_angle"]/torch.pi,
      strength, program["duration"]/.5, program["height_offset"]/.2,
      program["body_choice"].float()/3., program["point_offset"]/.1), -1)
  events = events*(event_type>0)[...,None]
  event = torch.nn.functional.one_hot(event_type, 5).flatten(1)
  clock = torch.stack((scenario.elapsed/20., recovery_level(env),
                       scenario.sample["action"][:, 1]/4.), -1)
  return torch.cat((objects.flatten(1), events.flatten(1),
                    program["object_mask"], event, clock), -1)


@dataclass
class RiskPPOCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "r1lab.learning:RiskPPO"
  warmup_updates: int = 64
  fixed_anchor_constraints: bool = True
  calibration_updates: int = 64
  minimum_anchor_episodes: int = 200
  # Per-control-step rates: fall event, unrecovered seconds, stationary seconds.
  cost_limits: tuple[float, ...] = (0.00001, 0.001, 0.0002)
  initial_multipliers: tuple[float, ...] = (120., 12., 1.)
  dual_lr: float = .02


def adversarial_walk_env_cfg(play=False, t0=0):
  cfg = careful_walk.projectile_walk_env_cfg(play=play, t0=t0, min_scale=.05)
  cfg.events.pop("projectiles")
  cfg.events = {"restore_material": EventTermCfg(func=restore_adversary_material,
                                                mode="reset"), **cfg.events}
  cfg.events["scenarios"] = EventTermCfg(func=AdversarialScenario, mode="step",
                                        params={"count": 8, "clean_fraction": .25})
  cfg.scene.entities.update(scene_entities())
  from r1lab.mdp.contacts import terrain_union
  cfg.scene.sensors = tuple(terrain_union(sensor) if sensor.name in
      ("feet_contact", "illegal_contact", "limb_contact") else sensor
      for sensor in cfg.scene.sensors)
  cfg.scene.sensors += (common.terrain_scan(),)
  cfg.scene.sensors += (ContactSensorCfg(name="adversary_object_contact",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="adversary_scene/scene"),
      fields=("found",), reduce="netforce", num_slots=1),)
  cfg.observations["critic"] = common.critic_obs("twist", scan=True)
  cfg.observations["critic"].terms["scenario"] = ObservationTermCfg(func=scenario_observation)
  cfg.sim.nconmax, cfg.sim.njmax, cfg.sim.contact_sensor_maxmatch = 192, 1536, 384
  command = cfg.commands["twist"]
  command.ranges.lin_vel_x = (-.25, .35)
  command.ranges.lin_vel_y = (-.20, .20)
  command.ranges.ang_vel_z = (-.60, .60)
  command.rel_standing_envs = .20
  # Modest task tracking prevents perpetual standing, without demanding speed
  # when balance requires shorter steps or emergency stopping.
  for name in ("track_lin_vel", "track_ang_vel", "lin_vel_err", "yaw_rate_err"):
    term = cfg.rewards[name]
    term.weight *= .5
    term.params = {**term.params, "base_func": term.func, "min_scale": .05}
    term.func = RecoveryScaledTerm
  cfg.rewards["pose"].weight = .1
  cfg.rewards["air_time"].weight = .5
  cfg.rewards["base_height"].weight *= .1
  for name in ("foot_clearance", "foot_swing_height"):
    # Let demonstrated foot trajectories and physical contact determine the
    # cautious movement, rather than imposing one fixed swing-height template.
    cfg.rewards[name].weight = 0.
  for name in ("power", "joint_vel"):
    cfg.rewards[name].weight *= .25
  cfg.rewards["style"] = RewardTermCfg(func=human_motion_prior, weight=3., params={
      "data_file": str(work_root()/"data/amp/human_careful_fullbody_v3.npz"),
      "stable_locomotion_only": True, "command_name": "twist",
      "batch": 2048, "buffer": 65536, "num_updates": 4,
      "validation_batch": 2048, "lr": 1e-4, "grad_penalty": 5.})
  from r1lab.terrain_observation import add_actor_height_scan
  return add_actor_height_scan(cfg)


def adversarial_walk_rl_cfg():
  cfg = common.ppo_runner_cfg("r1_careful_walk", iters=2000)
  cfg.algorithm = RiskPPOCfg(learning_rate=5e-5, schedule="adaptive", entropy_coef=.002,
      num_learning_epochs=3, num_mini_batches=8, desired_kl=.01)
  cfg.critic.class_name = "r1lab.learning:RiskCritic"
  cfg.actor.distribution_cfg["init_std"] = .2
  cfg.save_interval = 100
  return cfg
