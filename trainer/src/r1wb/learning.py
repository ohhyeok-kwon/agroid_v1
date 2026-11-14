"""PPO with an imitation term toward labels carried in the observations.

The environment supplies an observation group `bc_group` = [label action (A), weight (1), driven (1)]. Where
`driven` is set the label (plus a little noise) is executed instead of the sampled action: those samples take part
in the imitation term only, never in the policy-gradient, entropy or value terms (they are not the policy's own
experience). This is the teacher-driven share of DAgger: without it a student that does not start a motion on a
new kind of command only ever sees labels from the states where it is stuck. The loss adds
bc_coef(t) * weighted mean squared error between the policy mean and the label; the coefficient decays linearly
to a floor and never reaches zero. During the first `critic_warmup_updates` updates only the value loss and the
imitation term move the networks (the critic has to exist before the policy-gradient term may act); after that
the policy-gradient and entropy terms are ramped in linearly over `surrogate_ramp_updates` updates instead of being
switched on in one step.
Learning rate. With schedule="fixed" (the setting the student uses) the warm-up runs at `learning_rate` and the
rest of the run decays linearly from `ppo_learning_rate` to `final_learning_rate`. A KL-adaptive rate does not work
next to a supervised term, in either direction: the stock rule let it climb to 1e-2 while the measured KL was small
(first pilot: the imitation loss jumped from 0.085 to 0.47 when the policy-gradient term came on), and with a cap it
fell to its floor in the first update after the warm-up and stayed there (second run, 5e-4 -> 5e-5 at update 50),
because the imitation steps alone move the policy by more than the KL target. schedule="adaptive" is kept, capped
at the configured rate and floored at 5e-5, for runs without a strong imitation term.
The imitation loss is also logged per joint group (legs, waist, arms).

AnchoredPPO (r1wb.anchored) is the same loss with a different source of the label: a frozen copy of the policy the
run started from. The environment only supplies [weight, unsettled]; the label is that frozen network's action on the
very observation the policy sees. The weight is 1 while the robot is steady and 0 from an external impulse until the
body has settled, so calm behaviour stays the start policy's and the reaction to a hit is left to the rewards.
This exists during training only: the exported policy is the mean network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


class ImitationPPO(PPO):
  def __init__(self, *args, bc_group: str = "bc", bc_coef: float = 1.0, bc_floor: float = 0.3,
               bc_decay_updates: int = 1500, critic_warmup_updates: int = 50, surrogate_ramp_updates: int = 100,
               ppo_learning_rate: float | None = None, final_learning_rate: float | None = None,
               lr_decay_updates: int = 4000, driven_noise: float = 0.1, ramp_learning_rate: bool = False, **kwargs):
    super().__init__(*args, **kwargs)
    if self.rnd or self.symmetry or self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("ImitationPPO supports plain feed-forward actor-critic training only")
    self.bc_group, self.bc_coef, self.bc_floor = bc_group, bc_coef, bc_floor
    self.bc_decay_updates, self.critic_warmup_updates = bc_decay_updates, critic_warmup_updates
    self.surrogate_ramp_updates = surrogate_ramp_updates
    self.ppo_learning_rate, self.final_learning_rate = ppo_learning_rate, final_learning_rate
    self.lr_decay_updates = lr_decay_updates
    self.driven_noise = driven_noise
    self.update_count = 0
    self.max_learning_rate = float(self.learning_rate)
    # Adam normalises the size of a gradient away, so a ramp on the loss alone does not soften the first policy
    # steps of a network that has had no gradient during the warm-up. Set by AnchoredPPO, where that matters: the
    # policy starts as a finished policy, and its first policy-gradient update moved it by 0.12 action units (RMS).
    # Without an imitation term the policy network receives no gradient during the critic warm-up, so the optimiser
    # has no scale for it: the first policy-gradient step is then a full learning-rate step in every weight however
    # small the ramped loss is (run of 11-12 04:53 without labels: distance to the labels 0.22 -> 40 in two
    # iterations, then non-finite). Ramping the rate itself with the surrogate keeps those first steps small.
    self.ramp_learning_rate = ramp_learning_rate
    # AnchoredPPO: the policy is not touched at all during the critic warm-up. Its imitation term is zero there only
    # up to rounding (the anchor and a policy with a noise head compute the same mean with different kernels), and
    # Adam turns gradients of 1e-8 into full-size steps: the policy drifted 0.4 action units in one update.
    self.hold_policy_in_warmup = False

  def act(self, obs):
    actions = super().act(obs)
    labels = obs[self.bc_group]
    if labels.shape[1] > actions.shape[1] + 1:
      driven = labels[:, -1] > 0.5
      if driven.any():
        target = labels[:, :actions.shape[1]]
        executed = target + self.driven_noise * torch.randn_like(target)
        actions = torch.where(driven.unsqueeze(1), executed, actions)
        self.transition.actions = actions.detach()
    return actions

  def _prepare(self, observations) -> None:
    """Hook before the policy is evaluated on a batch of observations."""

  def _extra_loss(self, observations):
    """Hook after the policy was evaluated on a batch: (loss term, {log name: value})."""
    return 0.0, {}

  def _labels(self, observations, width: int):
    """(label action, imitation weight, 1 where the sample is the policy's own experience)."""
    labels = observations[self.bc_group]
    target, weight = labels[:, :width], labels[:, width]
    # Samples executed by the label are not the policy's own experience: imitation only.
    own = 1.0 - labels[:, width + 1] if labels.shape[1] > width + 1 else torch.ones_like(weight)
    return target, weight, own

  def imitation_coefficient(self) -> float:
    progress = min(1.0, self.update_count / max(1, self.bc_decay_updates))
    return self.bc_coef + (self.bc_floor - self.bc_coef) * progress

  def update(self) -> dict[str, float]:
    totals = dict(value=0.0, surrogate=0.0, entropy=0.0, imitation=0.0, imitation_legs=0.0, imitation_waist=0.0,
                  imitation_arms=0.0)
    warmup = self.update_count < self.critic_warmup_updates
    ramp = min(1.0, max(0.0, (self.update_count - self.critic_warmup_updates + 1) / max(1, self.surrogate_ramp_updates)))
    coefficient = self.imitation_coefficient()
    if self.schedule != "adaptive" and not warmup:
      start = self.ppo_learning_rate if self.ppo_learning_rate is not None else self.max_learning_rate
      end = self.final_learning_rate if self.final_learning_rate is not None else start
      progress = min(1.0, (self.update_count - self.critic_warmup_updates)
                     / max(1, self.lr_decay_updates - self.critic_warmup_updates))
      self.learning_rate = (start + (end - start) * progress) * (ramp if self.ramp_learning_rate else 1.0)
      for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
      self._prepare(batch.observations)
      self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy
      # During the warm-up the imitation term alone moves the policy quickly; a KL rule tuned for policy-gradient
      # steps would shrink the rate to its floor there, so it only runs afterwards (and never above the start rate).
      if self.desired_kl is not None and self.schedule == "adaptive" and not warmup:
        with torch.inference_mode():
          kl_mean = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params))
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(5e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
      target, weight, own = self._labels(batch.observations, distribution_params[0].shape[1])
      own_count = own.sum().clamp_min(1.0)
      ratio = torch.exp((actions_log_prob - torch.squeeze(batch.old_actions_log_prob)).clamp(-20.0, 20.0))
      advantages = torch.squeeze(batch.advantages)
      surrogate_loss = (torch.max(-advantages * ratio,
                                  -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param))
                        * own).sum() / own_count
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_error = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2))
      else:
        value_error = (batch.returns - values).pow(2)
      value_loss = (value_error.squeeze(-1) * own).sum() / own_count
      entropy = entropy * own * (own.numel() / own_count)        # the mean below then runs over own samples only
      squared = torch.square(distribution_params[0] - target)
      if weight.dim() == 2:      # a weight per sample and joint
        # Divided by the weights of the ordinary (settled) anchor alone: extra arm targets add to the loss and do
        # not dilute the settled anchor (with the plain mean, labels on a quarter of the samples would halve it).
        imitation = (squared * weight).sum() / self._anchor_norm.clamp_min(float(weight.shape[1]))
      else:
        imitation = (squared.mean(dim=1) * weight).sum() / weight.sum().clamp_min(1.0)
      with torch.no_grad():        # joint order of the action: legs (12), waist (2), arms (10)
        groups = {"imitation_legs": squared[:, :12].mean(), "imitation_waist": squared[:, 12:14].mean(),
                  "imitation_arms": squared[:, 14:].mean()}
      extra, extra_logs = self._extra_loss(batch.observations)
      loss = self.value_loss_coef * value_loss + coefficient * imitation + extra
      if not warmup:
        loss = loss + ramp * (surrogate_loss - self.entropy_coef * entropy.mean())
      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      if warmup and self.hold_policy_in_warmup:
        for parameter in self.actor.parameters():
          parameter.grad = None
      self.optimizer.step()
      for key, value in (("value", value_loss), ("surrogate", surrogate_loss), ("entropy", entropy.mean()),
                         ("imitation", imitation), *groups.items(), *extra_logs.items()):
        totals[key] = totals.get(key, 0.0) + value.item()
    updates = self.num_learning_epochs * self.num_mini_batches
    result = {key: value / updates for key, value in totals.items()}
    result["imitation_coefficient"] = coefficient
    result["surrogate_ramp"] = ramp
    result["learning_rate"] = self.learning_rate
    self.update_count += 1
    self.storage.clear()
    return result

  def save(self) -> dict:
    state = super().save()
    state["imitation_update_count"] = self.update_count
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    result = super().load(loaded_dict, load_cfg, strict)
    self.update_count = int(loaded_dict.get("imitation_update_count", 0))
    return result



class SymmetricPPO(ImitationPPO):
  """ImitationPPO plus a mirror loss: mirror_coef x |policy(mirror(o)) - mirror(policy(o))|^2, summed over the
  joints and averaged over the batch (r1wb.symmetry holds the maps; verified in the simulator by
  r1wb.eval.mirror_check).

  The robot is left-right symmetric, so the best policy commutes with the mirror. The policies trained without this
  do not: E99's joint targets differ by 0.041 rad on average between a state and its mirror image (0.049 while
  walking), which is more than its distance to the course (0.02), and which side of a mirrored pair of commands
  stops cleanly flips from checkpoint to checkpoint. The loss says nothing about what to do, only that both sides
  are the same problem: each side's experience then also teaches the other. Training only; the exported policy is
  the mean network as before."""

  def __init__(self, *args, mirror_coef: float = 4.0, **kwargs):
    super().__init__(*args, **kwargs)
    groups = tuple(self.actor.obs_groups)
    if len(groups) != 1:
      raise ValueError(f"SymmetricPPO mirrors one observation group, the policy reads {groups}")
    self.mirror_coef, self._mirror_group, self._mirror = mirror_coef, groups[0], None

  def _extra_loss(self, observations):
    from r1wb import symmetry
    # Nothing during the critic warm-up, then ramped in with the policy-gradient term (and with the learning rate,
    # ramp_learning_rate): acting alone at the full rate from the first update, the term made a finished policy
    # symmetric in eight updates (1.24 -> 0.016) and unusable on the way (value loss 400 against 2 to 6 in the plain
    # run): Adam's first steps are full-size in every weight whatever the gradient is (smoke run 11-13 15:10).
    ramp = min(1.0, max(0.0, (self.update_count - self.critic_warmup_updates + 1) / max(1, self.surrogate_ramp_updates)))
    if ramp <= 0.0:
      return 0.0, {}
    seen = observations[self._mirror_group]
    if self._mirror is None:
      self._mirror = symmetry.ActorMirror(seen.shape[1], seen.device)
    mean = self.actor.output_distribution_params[0]
    flipped = TensorDict({self._mirror_group: self._mirror.observation(seen)}, batch_size=[seen.shape[0]])
    other = self.actor(flipped)                    # deterministic output: the mean on the mirrored observation
    loss = torch.square(other - self._mirror.action(mean)).sum(dim=1).mean()
    return ramp * self.mirror_coef * loss, {"mirror": loss}


class AnchoredPPO(ImitationPPO):
  def __init__(self, *args, anchor_checkpoint: str = "", anchor_groups: tuple[str, ...] = ("actor", "goal"),
               aux_group: str = "", aux_coef: float = 1.0, aux_tap: int = 2, aux_dim: int = 4,
               class_advantages: bool = False, course_group: str = "", course_weight: float = 1.0,
               repeat_checkpoint: str = "", repeat_weight: float = 0.0, **kwargs):
    super().__init__(*args, **kwargs)
    from r1wb.student.observations import load_teacher
    if not anchor_checkpoint:
      raise ValueError("AnchoredPPO needs the checkpoint of the policy the run starts from")
    self.anchor = load_teacher(anchor_checkpoint, self.device)
    self.anchor_groups = tuple(anchor_groups)
    self.ramp_learning_rate = True
    self.hold_policy_in_warmup = True
    # Estimation head (training only): a linear read-out of the policy's hidden features that has to tell the
    # privileged disturbance state. It gives the network a feature that separates "I was hit" from ordinary motion,
    # so that the reaction can change without the anchored, undisturbed behaviour changing with it.
    # PPO scales the advantages by their spread over all samples. The rare falls of the environments under bursts
    # are orders of magnitude larger than anything a moderate hit changes, so in one common scale the samples that
    # could teach a smaller reaction carry almost no gradient. With class_advantages each class of environments
    # (never hit / gentle / the rest; fourth column of the anchor signal) is scaled on its own.
    self.class_advantages = class_advantages
    self.course_group, self.course_weight = course_group, course_weight
    # Once hits repeat within one reaction, the arm joints are anchored to what the checkpoint this run continues
    # from does there (its burst reaction is already learned and measured); see anchored.arm_course.
    self.repeat_weight = repeat_weight
    self.repeat_model = load_teacher(repeat_checkpoint, self.device) if repeat_weight > 0.0 else None
    self.aux_group, self.aux_coef = aux_group, aux_coef
    self.aux_head, self._aux_features = None, None
    if aux_group:
      layers = [module for module in self.actor.mlp.modules() if isinstance(module, nn.Linear)]
      tap = layers[aux_tap]
      tap.register_forward_pre_hook(lambda module, inputs: setattr(self, "_aux_features", inputs[0]))
      self.aux_head = nn.Linear(tap.in_features, aux_dim).to(self.device)
      self.optimizer.add_param_group({"params": list(self.aux_head.parameters()), "lr": self.learning_rate})

  def compute_returns(self, obs) -> None:
    super().compute_returns(obs)
    if not self.class_advantages:
      return
    storage = self.storage
    raw = storage.returns - storage.values
    kind = storage.observations[self.bc_group][..., 3]
    scaled = torch.zeros_like(raw)
    for value in kind.unique():
      mask = (kind == value).unsqueeze(-1)
      part = raw[mask]
      scaled[mask] = (part - part.mean()) / (part.std() + 1e-8)
    storage.advantages = scaled

  def save(self) -> dict:
    state = super().save()
    if self.aux_head is not None:
      state["aux_head_state_dict"] = self.aux_head.state_dict()
    return state

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    result = super().load(loaded_dict, load_cfg, strict)
    if self.aux_head is not None and "aux_head_state_dict" in loaded_dict:
      self.aux_head.load_state_dict(loaded_dict["aux_head_state_dict"])
    return result

  def _extra_loss(self, observations):
    if not self.aux_group:
      return 0.0, {}
    loss = nn.functional.mse_loss(self.aux_head(self._aux_features), observations[self.aux_group])
    return self.aux_coef * loss, {"estimate": loss}

  def anchor_action(self, observations) -> torch.Tensor:
    joined = torch.cat([observations[group] for group in self.anchor_groups], dim=1)
    with torch.no_grad():
      return self.anchor(TensorDict({"actor": joined}, batch_size=[joined.shape[0]]))

  def _labels(self, observations, width: int):
    signal = observations[self.bc_group]
    weight = signal[:, 0]
    # Nominal samples (run without exploration noise) are anchor samples only: they carry no policy-gradient signal.
    own = 1.0 - signal[:, 2] if signal.shape[1] > 2 else torch.ones_like(weight)
    target = self.anchor_action(observations)
    if self.course_group:
      # During a moderate hit the anchor of the arm joints is "as before the hit"; the other joints keep theirs
      # (none while unsettled). Elsewhere nothing changes.
      course = observations[self.course_group]
      hold = course[:, 0:1] > 0.5
      target = torch.cat([target[:, :14], torch.where(hold, course[:, 1:11], target[:, 14:])], dim=1)
      self._anchor_norm = weight.sum() * width
      weight = weight.unsqueeze(1).repeat(1, width)
      weight[:, 14:] = torch.where(hold, torch.full_like(weight[:, 14:], self.course_weight), weight[:, 14:])
      if self.repeat_model is not None:
        repeat = course[:, 11:12] > 0.5
        joined = torch.cat([observations[group] for group in self.anchor_groups], dim=1)
        with torch.no_grad():
          continued = self.repeat_model(TensorDict({"actor": joined}, batch_size=[joined.shape[0]]))
        target = torch.cat([target[:, :14], torch.where(repeat, continued[:, 14:], target[:, 14:])], dim=1)
        weight[:, 14:] = torch.where(repeat, torch.full_like(weight[:, 14:], self.repeat_weight), weight[:, 14:])
    return target, weight, own

  def act(self, obs):
    actions = PPO.act(self, obs)
    signal = obs[self.bc_group]
    if signal.shape[1] > 2:
      nominal = signal[:, 2:3] > 0.5
      actions = torch.where(nominal, self.transition.distribution_params[0], actions)
      self.transition.actions = actions.detach()
      self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
    return actions
