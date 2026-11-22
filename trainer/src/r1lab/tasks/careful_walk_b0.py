"""B0 (positive 23, coordinator 2026-11-20 18:2x; rounds/R04/16_rounds.md 'B0 card'). DIAGNOSTIC ONLY.

S1 with only the E1 R2 leaves set back to the anchor's = the E1 R1 setting (anchor trainer + S1 environment leaves) with
the rewards and commands of S1 objective v2. The env is r1s1's S0b-fixed env (r1s1.tasks._env_fixed:
s1_env_cfg(effort="power", post_metrics=True): rewards, goal sampler move / crouch / hold, goal observation, deadline,
terminations, S1 environment leaves); only its representation is the anchor's (careful_walk stage 1):
  actor input   the anchor's 10-frame history (r1lab frame_stack: gyro, gravity, command, q - q0, q_dot, previous action,
                its sensor noise model) with the 9-value S1 goal in the 3 command slots: 10 x 87 = 870 values
  critic input  the anchor's (frames, clean frame, base velocity, base height, feet contact / force, feet positions) with
                the goal in the command slots: 870 + 87 + 14 = 971; the anchor's 'feet_contact' sensor is added for it
  actions       the anchor's policy-order joint-position action (scales and offsets equal to S1's entity-order term)
  trainer       careful_walk_rl_cfg: MLPModel actor and critic with empirical normalisation, std learned from 0.8, PPO,
                gamma 0.99, entropy 0.005, adaptive lr, 4,096 envs; mirror augmentation by r1lab.train with the goal's
                mirror signs (r1s1.model) and the 87-value frame: install_goal_mirror() adds the 870 / 87 widths to
                r1lab.mdp.symmetry.Mirror in the process that builds this env (the 810 / 81 paths are untouched)
No AMP, gait or careful_walk tracking terms (S1 has none)."""
from __future__ import annotations

import dataclasses

import torch

from r1lab import contract as C
from r1lab.mdp import observations as obs

GOAL = 9
FRAME = 6 + GOAL + 3 * 24          # 87
WIDTH = C.HISTORY * FRAME          # 870
GOAL_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)   # r1s1.model.actor_mirror's goal part
CRITIC_LAYOUT = (("actor", WIDTH), ("frame", FRAME), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6))


class goal_frame_stack(obs.frame_stack):
  """r1lab frame_stack with the whole goal command (9 values) where the anchor's frame has v_x, v_y, w_z."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.hist = torch.zeros(self.n, C.HISTORY, FRAME, device=self.dev)

  def __call__(self, env, command_name: str, noise: bool = True, noise_params: dict | None = None) -> torch.Tensor:
    del noise, noise_params
    step = env.common_step_counter
    ids = self.env_ids if step != self.last_step else self.needs_fill.nonzero(as_tuple=False).flatten()
    if ids.numel():
      measured = self._measure(env, ids)
      prev = torch.where(self.needs_fill[ids, None], measured, self.prev_sensor[ids])
      self.prev_sensor[ids] = measured
      self.sensor[ids] = ((1.0 - self.lag[ids]) * measured + self.lag[ids] * prev
                          if self.noise else measured)
    sensor = self.sensor
    goal = env.command_manager.get_command(command_name)
    act = env.action_manager.action
    frame = torch.cat([sensor[:, :6], goal, sensor[:, 6:30], sensor[:, 30:54], act], dim=-1)
    if step != self.last_step:
      self.hist = torch.roll(self.hist, shifts=-1, dims=1)
    self.hist[:, -1] = frame
    if self.needs_fill.any():
      ids = self.needs_fill.nonzero(as_tuple=False).flatten()
      self.hist[ids] = frame[ids].unsqueeze(1).expand(-1, C.HISTORY, -1)
      self.needs_fill[ids] = False
    self.last_step = step
    return self.hist.reshape(self.n, WIDTH)


def goal_true_frame(env, command_name: str) -> torch.Tensor:
  """r1lab true_frame (critic) with the whole goal command."""
  d = env.scene["robot"].data
  perm = torch.tensor(C.POLICY_TO_ENTITY, device=env.device, dtype=torch.long)
  q = (d.joint_pos - d.default_joint_pos)[:, perm]
  dq = d.joint_vel[:, perm]
  goal = env.command_manager.get_command(command_name)
  return torch.cat([d.root_link_ang_vel_b, d.projected_gravity_b, goal, q, dq, env.action_manager.action], dim=-1)


def install_goal_mirror() -> None:
  from r1lab.mdp import symmetry as S
  M = S.Mirror
  if getattr(M, "_b0_goal_frames", False):
    return
  frame0, actor0 = M.frame, M.actor

  def tables(self, device):
    key = ("b0", device)
    if key not in self._t:
      idx = torch.tensor(C.MIRROR_INDEX, device=device)
      sgn = torch.tensor(C.MIRROR_SIGN, device=device, dtype=torch.float32)
      fi = torch.cat([torch.arange(6 + GOAL, device=device), 6 + GOAL + idx, 30 + GOAL + idx, 54 + GOAL + idx])
      fs = torch.cat([torch.tensor(C.ANG_VEL_SIGN, device=device, dtype=torch.float32),
                      torch.tensor(C.GRAVITY_SIGN, device=device, dtype=torch.float32),
                      torch.tensor(GOAL_SIGN, device=device, dtype=torch.float32), sgn, sgn, sgn])
      self._t[key] = (fi, fs)
    return self._t[key]

  def frame(self, x):
    if x.shape[-1] == FRAME:
      fi, fs = tables(self, x.device)
      return x[..., fi] * fs
    return frame0(self, x)

  def actor(self, x):
    if x.shape[-1] == WIDTH:
      b = x.shape[0]
      return frame(self, x.view(b, C.HISTORY, FRAME)).reshape(b, -1)
    return actor0(self, x)

  M.frame, M.actor = frame, actor
  M._b0_goal_frames = True


def anchor_representation(cfg, play: bool = False, t0: int = 0):
  """Put the anchor's actor / critic inputs (goal in the command slots), action term and feet_contact sensor into cfg."""
  from r1lab.tasks.careful_walk import careful_walk_env_cfg
  anchor = careful_walk_env_cfg(play=play, flat=True, stage=1, t0=t0)
  actor, critic = anchor.observations["actor"], anchor.observations["critic"]
  t = actor.terms["frames"]
  actor.terms["frames"] = dataclasses.replace(t, func=goal_frame_stack, params={**t.params, "command_name": "goal"})
  t = critic.terms["frames"]
  critic.terms["frames"] = dataclasses.replace(t, func=goal_frame_stack, params={**t.params, "command_name": "goal"})
  t = critic.terms["true_frame"]
  critic.terms["true_frame"] = dataclasses.replace(t, func=goal_true_frame, params={**t.params, "command_name": "goal"})
  cfg.observations = {"actor": actor, "critic": critic}
  cfg.actions = dict(anchor.actions)
  named = {sensor.name: sensor for sensor in anchor.scene.sensors}
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (named["feet_contact"],)
  install_goal_mirror()
  return cfg


def b0_env_cfg(play: bool = False, t0: int = 0):
  from r1s1.env_cfg import s1_env_cfg
  return anchor_representation(s1_env_cfg(effort="power", play=play, post_metrics=True), play=play, t0=t0)


def b0_rl_cfg():
  from r1lab.tasks import careful_walk
  return careful_walk.careful_walk_rl_cfg()
