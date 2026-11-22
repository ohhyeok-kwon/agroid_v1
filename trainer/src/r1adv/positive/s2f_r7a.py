"""R7 wave-4 branch A control cell 'A-GMM' of the positive agent (positive session 8; coordinator 2026-11-18 20:2x KST,
.agent/adversarial/rounds/R04/07B_r7_synthesis.md 5.1 and the 20:2x decisions). Exploratory: the parents were trained
with the 31 careful_walk terms.

Question: r6d changed only the hits and kept the gait terms. Does a stepping policy keep stepping when its objective
becomes the Hold objective (no gait terms, no walking practice)?
Parents (the unit of the verdict, one run each): r6d long_s1 / long_s2 model_4000 (s2f lineage, self-collision on,
adapted to option S; std about 0.56, inherited, not widened).

Changed from the parents' environment (s2f_r6d cell 'long' = careful_walk stage 2 + option-S Hold hits + push):
  reward     the 31 careful_walk terms (AMP style, gait and quiet-contact terms, terminated -100, ...) -> the Hold objective
             r = L + D (objective.py v1: cauchy, locked sigma, beta 1, D = d_v0b), with L's target TARGET below
             = hold_target_v0d (legs, waist, pelvis height, attitude) with the arms replaced by the parents' quiet-
             standing arms (mirror-averaged, mean of the two parents; .agent/adversarial/positive/tools/r7a_qstar.py)
  commands   all standing (rel_standing_envs 1.0): L's speed and yaw channels target zero
  floor      flat plane (L's height channel measures the pelvis against the env origin): no terrain curriculum and no
             out-of-bounds termination (both exist only with the terrain generator)
  curricula  removed: smoothness / quiet_and_disturb / stepping_taxes reweighted the removed terms (r1lab robustness_ramp
             stops on a missing reward term), command_vel shaped the walking commands; at the parents' step counter every
             ramp sat at its end value
Unchanged: observations (actor 810 deploy contract, critic 905), actions, robot (self-collision on), events (push
+-0.4 m/s every 3-7 s; option-S torso hits 4-25 N s, 70 % half-sine 0.1-0.8 s / 30 % impulse; body randomisation),
terminations (time-out, fell_over 60 deg, illegal_contact), episode 20 s, simulation settings, PPO settings, mirror
augmentation, the inherited std.
PPO: careful_walk_rl_cfg with algorithm ParentLrPPO: after a resume the adaptive learning rate starts at START_LR
(1.139e-4, the s2f lineage's rate) instead of the restored optimizer's (the critic warm-up run leaves 5e-4 there).
Critic warm-up: a separate run with r1lab.train --freeze-actor (actor, std and actor normaliser frozen; fixed lr 5e-4),
50 updates (precedent 11-08 s7: critic_warmup 0-49, then the main run from @49); the main run resumes from its last
checkpoint.
"""
from __future__ import annotations

import dataclasses
import json
import math

import torch

from mjlab.rl import RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO

from r1adv.positive import hold_r2 as R2
from r1adv.positive import hold_task as HT
from r1adv.positive import objective as OBJ

T0 = 1499
TARGET = dict(file="hold_target_r7a_v1.json", sha256="41a5f5ec3e56e587ccd60f170a036a65cae75a98247dad23a493a378b15bbb9c")
START_LR = 1.1390625e-4
PARENTS = {
  "p1": dict(run="runs/r1_careful_walk/careful_walk_20261118-1234_r6d-long_n4096_from-s2f@3300_r6d-S2Flong_s1",
             checkpoint="model_4000.pt", sha256="732740881551603d61cdbdc1d506b19afd04cf22695a583483c3d5d7765d60a4"),
  "p2": dict(run="runs/r1_careful_walk/careful_walk_20261118-1236_r6d-long_n4096_from-s2f@3300_r6d-S2Flong_s2",
             checkpoint="model_4000.pt", sha256="1805ffb78406e90bdad5b1991f06d18b4e7420118f8b9e7d61c3a7e2b3f28e35"),
}
WARMUP_UPDATES = 50


class HoldObjectiveR7A(HT.HoldObjective):
  """HoldObjective (the locked formula) with L's target read from TARGET (pinned by sha256) instead of hold_target_v0d."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    p = cfg.params
    target = json.loads(R2.pinned(dict(file=p["target_file"], sha256=p["target_sha256"])).read_text())
    if list(target["joint_names"]) != list(self.robot.joint_names)[:24]:
      raise ValueError("r7a target joint order differs from the robot")
    dev = env.device
    self.q_star = torch.tensor(target["target_rad"], dtype=torch.float32, device=dev)
    self.h_star = float(target["pelvis_height_m"])
    qw, qx, qy, qz = target["qpos"][3:7]
    self.g_star = torch.tensor([-2 * (qx * qz - qw * qy), -2 * (qy * qz + qw * qx), -(1 - 2 * (qx * qx + qy * qy))],
                               dtype=torch.float32, device=dev)

  def __call__(self, env, beta: float, sigma: dict, form: str, formula_sha256: str = "", target_file: str = "",
               target_sha256: str = "") -> torch.Tensor:
    del target_file, target_sha256
    return super().__call__(env, beta, sigma, form, formula_sha256)


def r7a_env_cfg(play: bool = False, t0: int = T0):
  """The parents' environment (s2f_r6d 'long') with the changes listed in the module docstring."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.terrains import TerrainEntityCfg
  from r1adv.positive.s2f_r6d import s2f_env_cfg
  cfg = s2f_env_cfg("long", play=play, t0=t0)
  cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
  cfg.terminations.pop("out_of_bounds", None)
  cfg.curriculum = {}
  cfg.commands["twist"].rel_standing_envs = 1.0
  cfg.rewards = {"objective": RewardTermCfg(func=HoldObjectiveR7A, weight=1.0, params=dict(
    beta=1.0, sigma=dict(HT.SIGMA), form=HT.FORM, formula_sha256=OBJ.formula_sha256(),
    target_file=TARGET["file"], target_sha256=TARGET["sha256"]))}
  return cfg


@dataclasses.dataclass
class ParentLrPpoCfg(RslRlPpoAlgorithmCfg):
  start_learning_rate: float | None = None
  """Adaptive schedule: the learning rate after a resume (None keeps the restored optimizer's)."""


class ParentLrPPO(PPO):
  """rsl_rl PPO; after loading a checkpoint with its optimizer, the adaptive schedule starts at start_learning_rate."""

  def __init__(self, *args, start_learning_rate: float | None = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.start_learning_rate = start_learning_rate

  def load(self, loaded_dict, load_cfg, strict):
    out = super().load(loaded_dict, load_cfg, strict)
    restores_optimizer = load_cfg is None or bool(load_cfg.get("optimizer", True))
    if self.start_learning_rate is not None and self.schedule == "adaptive" and restores_optimizer:
      self.learning_rate = float(self.start_learning_rate)
      for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
    return out


def r7a_rl_cfg():
  from r1lab.tasks.careful_walk import careful_walk_rl_cfg
  cfg = careful_walk_rl_cfg()
  base = cfg.algorithm
  cfg.algorithm = ParentLrPpoCfg(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                                 start_learning_rate=START_LR)
  cfg.algorithm.class_name = "r1adv.positive.s2f_r7a:ParentLrPPO"
  return cfg
