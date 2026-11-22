"""E4 board-3 method arms (positive 25, coordinator 2026-11-20 23:5x KST; rounds/R04/16_rounds.md '3 판 준비'). DIAGNOSTIC.
All on B0 (r1lab.tasks.careful_walk_b0, unchanged); every change is a visible config leaf.

m1m3  M1 (r1lab.tasks.careful_walk_e4m.m1_success: S_t = 1[in goal and at rest] at every step) + the fall termination
      removed (terminations.s1_body_ground: any non-foot body on the floor > 1 N). Self-collision and time-out stay.
      Records only (no reward, no termination): metrics e4n_down = 1[a non-foot body on the floor > 1 N] and
      e4n_down_success = e4n_down and S = 1 (a fallen robot lying still inside a hold goal issued while down would get S).
m5    input bottleneck (R18-a r07 VIB on the actor-loss side, exploration by the bottleneck sample). The actor MLP (B0's
      870 -> 512 -> 256 -> 128) ends in 2 d_z values (mu_z, log sigma_z), d_z = 32; z ~ q(z|o) = N(mu_z, sigma_z^2);
      the action is the linear read-out a = W z + b (+ a fixed floor N(0, 0.05^2) per joint), so the policy is exactly
          pi(a|o) = N(W mu_z + b, W diag(sigma_z^2) W^T + 0.05^2 I)   (torch LowRankMultivariateNormal)
      and PPO's ratio / entropy / adaptive-lr KL are exact. Exploration noise comes from sigma_z through W (no action std).
      Actor loss += beta KL(q(z|o) || N(0, I)), beta 0.005 (= B0's entropy coefficient: one nat of observation
      information priced like one nat of action entropy). Mean action (deploy, ONNX) = W mu_z + b. At the start sigma_z = 1
      and W ~ N(0, 0.8^2 / d_z), so each joint's action std starts at ~0.8 as B0's.
m6    continuous rest success: S_t = 1[in goal] x exp(-|v_xy pelvis| / v0 - mean_j |qdot_j| / w0) at every step (M1's
      structure with the rest indicator made smooth). v0 = 0.05 m/s = the readout's / S1's pelvis rest threshold (REST[0];
      the frozen readout's rest share uses |v| <= 0.05 m/s and |w| <= 0.20 rad/s); w0 = 0.20 rad/s = the readout's angular
      rest threshold applied to the mean joint speed (the readout has no joint-speed threshold). At either threshold the
      factor is e^-1. in goal as judge() (TOL). Other reward terms unchanged.
m7    mission mix: the goal sampler's hold share 0.25 -> 0.40 (locomotion 0.45 -> 0.36, crouch 0.30 -> 0.24, the two kept in
      their 3 : 2 ratio) and hold goals U(2, 6) s -> U(4, 10) s (the anchor careful_walk's command resampling range 4-10 s,
      whose 15 % zero-velocity draws are its standing; the 20 s episode holds it). commands.goal.share and the new field
      commands.goal.hold_only_s.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import LowRankMultivariateNormal, kl_divergence

from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import Distribution

from r1s1.goal import GoalCommand, GoalCommandCfg, _uniform

ARMS3 = ("m1m3", "m5", "m6", "m7")
V0, W0 = 0.05, 0.20
M7_SHARE = {"hold": 0.40, "locomotion": 0.36, "crouch": 0.24}
M7_HOLD_S = (4.0, 10.0)
VIB = {"latent_dim": 32, "beta": 0.005, "floor_std": 0.05, "init_std": 0.8}


# -- m1m3 records ---------------------------------------------------------------------------------------------------
def e4n_down(env, command_name: str = "goal") -> torch.Tensor:
  from r1s1 import mdp as SM
  return SM.s1_body_ground(env).float()


def e4n_down_success(env, command_name: str = "goal") -> torch.Tensor:
  from r1s1 import mdp as SM
  _, in_goal, at_rest = env.command_manager.get_term(command_name).judge()
  return (SM.s1_body_ground(env) & in_goal & at_rest).float()


# -- m6 -------------------------------------------------------------------------------------------------------------
def m6_success(env, command_name: str = "goal", v0: float = V0, w0: float = W0) -> torch.Tensor:
  _, in_goal, _ = env.command_manager.get_term(command_name).judge()
  d = env.scene["robot"].data
  v = d.root_link_lin_vel_w[:, :2].norm(dim=-1)
  w = d.joint_vel.abs().mean(dim=-1)
  return in_goal.float() * torch.exp(-v / v0 - w / w0)


# -- m7 -------------------------------------------------------------------------------------------------------------
class M7GoalCommand(GoalCommand):
  """r1s1 GoalCommand with the hold goals' duration drawn from cfg.hold_only_s (the rest of the draw unchanged)."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    hold = self.kind[env_ids] == 0
    if hold.any():
      ids = env_ids[hold]
      dur = _uniform(len(ids), *self.cfg.hold_only_s, self.device)
      self.hold[ids] = dur
      self.time_left[ids] = self.tau0[ids] + dur          # tau0 = 0 for hold: the deadline stays at issue


@dataclass(kw_only=True)
class M7GoalCommandCfg(GoalCommandCfg):
  hold_only_s: tuple[float, float] = M7_HOLD_S

  def build(self, env) -> M7GoalCommand:
    return M7GoalCommand(self, env)


# -- m5 -------------------------------------------------------------------------------------------------------------
class _VIBMean(nn.Module):
  def __init__(self, dist: "VIBLowRankGaussian"):
    super().__init__()
    self.d = dist.latent_dim
    self.W, self.b = dist.W, dist.b

  def forward(self, x):
    return x[..., : self.d] @ self.W.t() + self.b


class VIBLowRankGaussian(Distribution):
  """Stochastic bottleneck z ~ N(mu_z, sigma_z^2) read out linearly: an exact low-rank-plus-floor Gaussian policy."""

  def __init__(self, output_dim: int, latent_dim: int = 32, beta: float = 0.005, floor_std: float = 0.05,
               init_std: float = 0.8) -> None:
    super().__init__(output_dim)
    self.latent_dim, self.beta, self.floor_std = int(latent_dim), float(beta), float(floor_std)
    self.W = nn.Parameter(torch.randn(output_dim, self.latent_dim) * (init_std / math.sqrt(self.latent_dim)))
    self.b = nn.Parameter(torch.zeros(output_dim))
    self._dist: LowRankMultivariateNormal | None = None
    self.kl_z: torch.Tensor | None = None
    LowRankMultivariateNormal.set_default_validate_args(False)

  @property
  def input_dim(self) -> int:
    return 2 * self.latent_dim

  def init_mlp_weights(self, mlp: nn.Module) -> None:
    last = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():                                  # log sigma_z head starts at 0 (sigma_z = 1)
      last.weight[self.latent_dim:].zero_()
      last.bias[self.latent_dim:].zero_()

  def _split(self, mlp_output):
    return mlp_output[..., : self.latent_dim], mlp_output[..., self.latent_dim:].clamp(-6.0, 2.0)

  def update(self, mlp_output: torch.Tensor) -> None:
    mu, log_sig = self._split(mlp_output)
    sig = log_sig.exp()
    mean = mu @ self.W.t() + self.b
    factor = self.W * sig.unsqueeze(-2)                    # (..., A, d)
    diag = torch.full_like(mean, self.floor_std ** 2)
    self._dist = LowRankMultivariateNormal(mean, factor, diag)
    self.kl_z = 0.5 * (mu.pow(2) + sig.pow(2) - 1.0 - 2.0 * log_sig).sum(-1)

  def sample(self) -> torch.Tensor:
    return self._dist.sample()

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    mu, _ = self._split(mlp_output)
    return mu @ self.W.t() + self.b

  def as_deterministic_output_module(self) -> nn.Module:
    return _VIBMean(self)

  @property
  def mean(self) -> torch.Tensor:
    return self._dist.loc

  @property
  def std(self) -> torch.Tensor:
    return (self._dist.cov_factor.pow(2).sum(-1) + self._dist.cov_diag).sqrt()

  @property
  def entropy(self) -> torch.Tensor:
    return self._dist.entropy()

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    return (self._dist.loc, self._dist.cov_factor, self._dist.cov_diag)

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self._dist.log_prob(outputs)

  def kl_divergence(self, old_params, new_params) -> torch.Tensor:
    return kl_divergence(LowRankMultivariateNormal(*old_params), LowRankMultivariateNormal(*new_params))


class VIBPPO(PPO):
  """rsl_rl 5.4.2 PPO.update with the bottleneck term: loss += beta * mean KL(q(z|o) || N(0, I)) (lines marked M5)."""

  def update(self) -> dict[str, float]:
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    mean_kl_z = 0  # M5
    mean_rnd_loss = 0 if self.rnd else None
    mean_symmetry_loss = 0 if self.symmetry else None
    dist = self._raw_actor.distribution  # M5
    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
      generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)
      self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
      distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
      entropy = self.actor.output_entropy[:original_batch_size]
      kl_z = dist.kl_z[:original_batch_size]  # M5
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()
      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
      loss = loss + dist.beta * kl_z.mean()  # M5
      rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None
      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss
      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()
      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_kl_z += kl_z.mean().item()  # M5
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_kl_z /= num_updates  # M5
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    loss_dict = {"value": mean_value_loss, "surrogate": mean_surrogate_loss, "entropy": mean_entropy,
                 "vib_kl": mean_kl_z}  # M5
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    self.storage.clear()
    return loss_dict


# -- configs ----------------------------------------------------------------------------------------------------------
def e4n_env_cfg(arm: str, play: bool = False, t0: int = 0):
  from mjlab.managers.metrics_manager import MetricsTermCfg
  from r1lab.tasks.careful_walk_b0 import b0_env_cfg
  from r1lab.tasks.careful_walk_e4m import m1_success
  if arm not in ARMS3:
    raise ValueError(f"arm must be one of {ARMS3}")
  cfg = b0_env_cfg(play=play, t0=t0)
  if arm == "m1m3":
    t = cfg.rewards["s1_success"]
    cfg.rewards["s1_success"] = dataclasses.replace(t, func=m1_success)
    cfg.terminations.pop("s1_body_ground")
    cfg.metrics = dict(cfg.metrics or {})
    for name, func in (("e4n_down", e4n_down), ("e4n_down_success", e4n_down_success)):
      cfg.metrics[name] = MetricsTermCfg(func=func, params={"command_name": "goal"}, reduce="mean")
  elif arm == "m6":
    t = cfg.rewards["s1_success"]
    cfg.rewards["s1_success"] = dataclasses.replace(t, func=m6_success, params={**t.params, "v0": V0, "w0": W0})
  elif arm == "m7":
    g = cfg.commands["goal"]
    fields = {f.name: getattr(g, f.name) for f in dataclasses.fields(g) if f.init}
    fields["share"] = dict(M7_SHARE)
    cfg.commands["goal"] = M7GoalCommandCfg(**fields, hold_only_s=M7_HOLD_S)
  return cfg


def e4n_rl_cfg(arm: str):
  from r1lab.tasks.careful_walk_b0 import b0_rl_cfg
  cfg = b0_rl_cfg()
  if arm == "m5":
    cfg.actor.distribution_cfg = {"class_name": "r1lab.tasks.careful_walk_e4n:VIBLowRankGaussian", **VIB}
    cfg.algorithm.class_name = "r1lab.tasks.careful_walk_e4n:VIBPPO"
  return cfg
