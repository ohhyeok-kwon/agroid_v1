"""R2 wave 2 arm A (positive agent; coordinator lock 2026-11-17 10:12 KST, item 4; optional and last): residual AR(1)
exploration with the exact likelihood (panel R1, RL exploration designer C1).

Rollout action:  a_t = mu(o_t) + r_t,  r_t = phi r_{t-1} + sigma sqrt(1 - phi^2) xi_t,  r = a - mu (the residual),
phi = exp(-dt / tau) fixed (tau = 0.5 s, dt = 0.02 s; not learned), sigma the learned std (stationary marginal std).
The policy is Markov in (o_t, o_{t-1}, a_{t-1}): its conditional distribution is
  N( mu(o_t) + phi v (a_{t-1} - mu(o_{t-1})),  sigma^2 (1 - phi^2 v) ),  v = 1 inside an episode, 0 at its first step,
and PPO's ratio, KL and entropy use this conditional with mu recomputed under the current parameters (one extra forward
pass for o_{t-1}), so the likelihood is exact. The previous observation, previous action and v ride along in the
observation TensorDict (keys ar_prev_obs, ar_prev_act, ar_valid) so the rollout storage keeps them per transition.
Deterministic output (deployment, evaluation) is mu(o_t) only: the deployed policy is the same function as before.
Classes are plugged by name: algorithm.class_name = r1adv.positive.ar1:AR1PPO, actor.class_name =
r1adv.positive.ar1:AR1HoldSymmetricMLP, distribution class_name = r1adv.positive.ar1:ScaledGaussian. rsl_rl 5.4.2."""
from __future__ import annotations

import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from r1adv.positive.hold_model import HoldSymmetricMLP

TAU_S = 0.5
DT = 0.02
PHI = math.exp(-DT / TAU_S)
KEYS = ("ar_prev_obs", "ar_prev_act", "ar_valid")


class ScaledGaussian(GaussianDistribution):
  """Gaussian with a per-sample std multiplier (set by the actor before update)."""

  scale: torch.Tensor | None = None

  def update(self, mlp_output: torch.Tensor) -> None:
    super().update(mlp_output)
    if self.scale is not None:
      self._distribution = Normal(self._distribution.mean, self._distribution.stddev * self.scale)


class AR1HoldSymmetricMLP(HoldSymmetricMLP):
  """HoldSymmetricMLP whose stochastic output is the AR(1) conditional (module docstring); same parameters and state
  dict as HoldSymmetricMLP, so evaluation tools load its checkpoints unchanged."""

  def mean_action(self, raw: torch.Tensor) -> torch.Tensor:
    plain = self.mlp(self.obs_normalizer(raw))
    other = self.mlp(self.obs_normalizer(raw[..., self.mirror_obs_index] * self.mirror_obs_sign))
    return 0.5 * (plain + other[..., self.mirror_act_index] * self.mirror_act_sign)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    mu = self.mean_action(raw)
    if self.distribution is None or not stochastic_output:
      return mu if self.distribution is None else self.distribution.deterministic_output(mu)
    if KEYS[0] in obs.keys():
      v = obs["ar_valid"].reshape(-1, 1)
      mu_prev = self.mean_action(obs["ar_prev_obs"])
      mean = mu + PHI * v * (obs["ar_prev_act"] - mu_prev)
      self.distribution.scale = torch.sqrt(1.0 - PHI ** 2 * v)
    else:                                                     # no context (e.g. a plain call): the marginal
      mean = mu
      self.distribution.scale = None
    self.distribution.update(mean)
    return self.distribution.sample()


class AR1PPO(PPO):
  """PPO that carries the AR(1) context (previous actor observation, previous action, validity) in the observations."""

  @staticmethod
  def _augment(obs, prev_obs, prev_act, valid):
    obs = obs.clone()
    obs["ar_prev_obs"] = prev_obs
    obs["ar_prev_act"] = prev_act
    obs["ar_valid"] = valid
    return obs

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    n = obs.batch_size[0]
    width = obs["actor"].shape[-1]
    dev = obs["actor"].device
    aug = AR1PPO._augment(obs, torch.zeros(n, width, device=dev), torch.zeros(n, env.num_actions, device=dev),
                          torch.zeros(n, 1, device=dev))
    return PPO.construct_algorithm(aug, env, cfg, device)

  def act(self, obs):
    if not hasattr(self, "_ar_prev_obs"):
      n, dev = obs["actor"].shape[0], obs["actor"].device
      self._ar_prev_obs = torch.zeros_like(obs["actor"])
      self._ar_prev_act = torch.zeros(n, self.actor.mirror_act_index.numel(), device=dev)
      self._ar_valid = torch.zeros(n, 1, device=dev)
    aug = self._augment(obs, self._ar_prev_obs, self._ar_prev_act, self._ar_valid)
    actions = super().act(aug)
    self._ar_prev_obs = obs["actor"].clone()
    self._ar_prev_act = actions.clone()
    self._ar_valid = torch.ones_like(self._ar_valid)
    return actions

  def process_env_step(self, obs, rewards, dones, extras):
    super().process_env_step(obs, rewards, dones, extras)
    done = dones.reshape(-1, 1).bool()
    self._ar_valid = torch.where(done, torch.zeros_like(self._ar_valid), self._ar_valid)

  def compute_returns(self, obs):
    aug = self._augment(obs, self._ar_prev_obs, self._ar_prev_act, self._ar_valid)
    return super().compute_returns(aug)
