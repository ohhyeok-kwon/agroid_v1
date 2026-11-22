"""E1 c13 axis split (positive 23, coordinator 2026-11-20 16:0x; rounds/R04/16_rounds.md 'E1 card'). DIAGNOSTIC ONLY.

Base = today's anchor (careful_walk_s1_repro: careful_walk stage 1, the walking run's recorded tracking terms, random
weights). The reward block stays careful_walk's. The leaves where the S1 (A) run differs from the anchor (cfg_diff of
the two runs' params; list in rounds/R04/positive/r13_posctl/cwr/c13/README.md) are split into two bundles:

  R1 = anchor + S1 environment leaves
       events      S1's resets (yaw only, base velocity +-0.1 m/s, joints +-0.05 rad, no joint velocity) and startup
                   randomisation (foot friction 0.3-1.2 shared, torso CoM / mass, hand loads 0-3 kg + CoM, PD x0.85-1.15,
                   encoder bias +-0.01) and pushes on the trunk (5-25 N s, 20 ms, every 4-8 s, all envs, start to end),
                   replacing the anchor's reset-mode randomisation set
       terminations S1's: a body other than the feet on the floor (1 N), self-collision (10 N), time-out (replacing
                   tilt 60 deg / trunk contact); their two contact sensors are added
       sim         S1's contact capacity (nconmax 64, njmax 400)
  R2 = anchor + S1 representation / optimisation leaves
       actor input S1's 0.5 s history (25 frames) of its proprio frame (gyro / sqrt(g/L), gravity, (q - q0) / half range,
                   q_dot / q_dot_max, with S1's sensor noise and 0 / 20 ms observation delay, hold 0.9) and of the
                   previous (sampled) action as joint-target offsets, plus the anchor's velocity command (3) where S1
                   has its 9-value goal: 25 x 78 + 3 = 1,953 inputs; no observation normaliser on the actor
       actions     S1's entity-order joint-position action (same scales and offsets as the anchor's policy-order term)
       policy std  S1 (A)'s fixed per-joint std sigma_500 x 0.5 (r1s1.distribution.FixedGaussian, not learned)
       actor       mirror-averaged mean (as r1s1.model.S1SymmetricMLPModel) - so the anchor's mirror data augmentation
                   is off for R2 (the augmentation's layouts are the 810 input's)
Everything else keeps the anchor's value: rewards, the velocity command and its distribution, the critic input and its
normaliser (S1's critic reads S1's goal), PPO (class, gamma 0.99, entropy 0.005, lr schedule, epochs, minibatches),
4,096 envs, terrain, robot.
"""
from __future__ import annotations

import torch

import rsl_rl.models.mlp_model as base
from rsl_rl.models import MLPModel

COMMAND = 3
# R1 bundle: sensors added for S1's terminations
R1_SENSORS = ("s1_body_ground", "s1_self")


def _s1(play: bool):
  from r1s1.env_cfg import s1_env_cfg
  return s1_env_cfg(effort="power", play=play)


def r1_env_cfg(play: bool = False, t0: int = 0):
  from r1lab.tasks.careful_walk_repro import repro_env_cfg
  cfg = repro_env_cfg(play=play, t0=t0)
  s1 = _s1(play)
  cfg.events = dict(s1.events)
  cfg.terminations = dict(s1.terminations)
  named = {sensor.name: sensor for sensor in s1.scene.sensors}
  cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(named[name] for name in R1_SENSORS)
  cfg.sim.nconmax, cfg.sim.njmax = s1.sim.nconmax, s1.sim.njmax
  return cfg


def command_obs(env, command_name: str = "twist") -> torch.Tensor:
  """The anchor's velocity command (v_x, v_y, w_z), at its own scale (as the anchor's frames carry it)."""
  return env.command_manager.get_command(command_name)


def r2_representation(cfg, play: bool = False):
  """Swap the actor input and the action term of a careful_walk env cfg for R2's (module docstring)."""
  from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
  s1 = _s1(play)
  terms = {name: term for name, term in s1.observations["actor"].terms.items() if name != "goal"}
  terms["command"] = ObservationTermCfg(func=command_obs, params={"command_name": "twist"})
  cfg.observations["actor"] = ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=False)
  cfg.actions = dict(s1.actions)
  return cfg


def r2_env_cfg(play: bool = False, t0: int = 0):
  from r1lab.tasks.careful_walk_repro import repro_env_cfg
  return r2_representation(repro_env_cfg(play=play, t0=t0), play=play)


def r2_rl_cfg():
  from r1lab.tasks import careful_walk
  from r1s1 import numbers as N
  cfg = careful_walk.careful_walk_rl_cfg()
  cfg.actor.class_name = "r1lab.tasks.careful_walk_c13:C13SymmetricMLPModel"
  cfg.actor.obs_normalization = False
  cfg.actor.distribution_cfg = {"class_name": "r1s1.distribution:FixedGaussian", "std_values": list(N.FIXED_STD),
                                "scale": N.SIGMA_FIXED_SCALE}
  return cfg


def actor_mirror(device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
  """r1s1.model's mirror of the 0.5 s proprio and target histories, then the command (v_x, -v_y, -w_z)."""
  from r1s1 import model as S1M
  index, sign = S1M.actor_mirror("cpu")
  width = S1M.WIDTH - S1M.GOAL
  index, sign = index[:width].tolist(), sign[:width].tolist()
  index += [width + 0, width + 1, width + 2]
  sign += [1.0, -1.0, -1.0]
  return torch.tensor(index, dtype=torch.long, device=device), torch.tensor(sign, device=device)


def width() -> int:
  from r1s1 import model as S1M
  return S1M.WIDTH - S1M.GOAL + COMMAND


class _OnnxC13Symmetric(base._OnnxMLPModel):
  def __init__(self, model: MLPModel, verbose: bool) -> None:
    super().__init__(model, verbose)
    from r1s1.model import action_mirror
    obs_index, obs_sign = actor_mirror()
    act_index, act_sign = action_mirror()
    for name, value in (("obs_index", obs_index), ("obs_sign", obs_sign), ("act_index", act_index),
                        ("act_sign", act_sign)):
      self.register_buffer(name, value)

  def one(self, x: torch.Tensor) -> torch.Tensor:
    return self.deterministic_output(self.mlp(self.obs_normalizer(x)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    flipped = torch.index_select(x, 1, self.obs_index) * self.obs_sign
    return 0.5 * (self.one(x) + torch.index_select(self.one(flipped), 1, self.act_index) * self.act_sign)


class C13SymmetricMLPModel(MLPModel):
  """Mirror-averaged actor mean for R2's 1,953-value input (as r1s1.model.S1SymmetricMLPModel)."""
  symmetrized = True

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.obs_dim != width():
      raise ValueError(f"R2 actor observation must have {width()} values, got {self.obs_dim}")
    from r1s1.model import action_mirror
    obs_index, obs_sign = actor_mirror()
    act_index, act_sign = action_mirror()
    for name, value in (("obs_index", obs_index), ("obs_sign", obs_sign), ("act_index", act_index),
                        ("act_sign", act_sign)):
      self.register_buffer("mirror_" + name, value, persistent=False)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    obs = base.unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    plain = self.mlp(self.obs_normalizer(raw))
    other = self.mlp(self.obs_normalizer(raw[..., self.mirror_obs_index] * self.mirror_obs_sign))
    output = 0.5 * (plain + other[..., self.mirror_act_index] * self.mirror_act_sign)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(output)
    return output

  def as_onnx(self, verbose: bool):
    return _OnnxC13Symmetric(self, verbose)
