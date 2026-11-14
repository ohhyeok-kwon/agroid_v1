"""One distributed policy with separate risk values and learned opponents.

PPO is extended with task/fall/recovery/speed value heads, adaptive nonnegative
cost multipliers and group risk weighting. These are empirical constraints,
not a certificate of safety. All auxiliary updates occur at rollout barriers.
"""
from __future__ import annotations

import copy
import json
import math
import time
import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel

from r1lab.distributed import all_reduce_mean, global_moments


def reduce_sum(value):
  world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
  return all_reduce_mean(value)*world


class RiskCritic(MLPModel):
  def __init__(self, obs, obs_groups, obs_set, output_dim=1, **kwargs):
    super().__init__(obs, obs_groups, obs_set, 4, **kwargs)


class RiskPPO(PPO):
  """PPO-Lagrangian with a shared four-head critic and five risk groups."""
  def __init__(self, actor, critic, storage, *, warmup_updates=32,
               cost_limits=(.0001, .004, .001), dual_lr=.02,
               initial_multipliers=(20., 2., 5.), fixed_anchor_constraints=False,
               calibration_updates=64, minimum_anchor_episodes=200, **kwargs):
    super().__init__(actor, critic, storage, **kwargs)
    self.maximum_learning_rate = float(self.learning_rate)
    if self.symmetry or actor.is_recurrent or critic.is_recurrent:
      raise ValueError("RiskPPO requires unaugmented feedforward observations")
    shape = (storage.num_transitions_per_env, storage.num_envs)
    storage.values = torch.zeros(*shape, 4, device=self.device)
    storage.returns = torch.zeros_like(storage.values)
    self.costs = torch.zeros(*shape, 3, device=self.device)
    self.groups = torch.zeros(*shape, dtype=torch.long, device=self.device)
    self.raw_costs = torch.zeros_like(self.costs)
    self.is_generated = torch.zeros(*shape, dtype=torch.bool, device=self.device)
    self.is_rsi = torch.zeros(*shape, dtype=torch.bool, device=self.device)
    self.constraint_groups = torch.zeros(*shape, dtype=torch.long, device=self.device)
    self.physics_overflow = torch.zeros((), device=self.device)
    self.anchor_budget = None
    if fixed_anchor_constraints:
      from r1lab.constraints import AnchorBudgets
      self.anchor_budget = AnchorBudgets(self.device, initial_multipliers,
          calibration_updates, minimum_anchor_episodes, dual_lr)
    self.cost_limits = torch.tensor(cost_limits, device=self.device)
    self.multipliers = torch.tensor(initial_multipliers, device=self.device)
    if self.cost_limits.shape != (3,) or (self.cost_limits <= 0).any():
      raise ValueError("Three positive cost-rate limits required")
    self.cost_ema = self.cost_limits.clone()
    self.group_cost_ema = torch.zeros(5, device=self.device)
    self.group_weights = torch.ones(5, device=self.device)
    self.dual_lr, self.warmup_updates = dual_lr, warmup_updates
    self.update_index = 0
    self.env = None
    self.normalize_advantage_per_mini_batch = False
    self.kl_diagnostics = {}

  @staticmethod
  def _global_max(value):
    result = value.detach().clone()
    if torch.distributed.is_initialized():
      torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
    return result

  @torch.no_grad()
  def _rollout_diagnostics(self):
    """Observe ranges without modifying/clipping any observation or action."""
    observations = self.storage.observations.flatten(0, 1)
    maxima = torch.zeros(8, dtype=torch.float64, device=self.device)
    cohort_maxima = torch.zeros(2, 3, dtype=torch.float64, device=self.device)
    is_rsi = self.is_rsi.flatten()
    nonfinite = torch.zeros((), dtype=torch.float64, device=self.device)

    def measure(value, index):
      nonlocal nonfinite
      good = torch.isfinite(value)
      nonfinite += (~good).sum()
      maximum = torch.where(good, value.abs(), 0.).amax().double()
      maxima[index] = torch.maximum(maxima[index], maximum)

    for start in range(0, len(observations), 32768):
      part = observations[start:start+32768]
      for model, index in ((self._raw_actor, 0), (self._raw_critic, 2)):
        raw = torch.cat([part[key] for key in model.obs_groups], -1)
        normalized = model.obs_normalizer(raw)
        measure(raw, index)
        measure(normalized, index+1)
        if index == 0:
          for column,value in enumerate((raw,normalized)):
            per_row = torch.where(torch.isfinite(value),value.abs(),0.).amax(-1).double()
            mask = is_rsi[start:start+len(part)]
            for row,selected in enumerate((~mask,mask)):
              cohort_maxima[row,column] = torch.maximum(cohort_maxima[row,column],
                  torch.where(selected,per_row,0.).amax())
    measure(self.storage.actions, 4)
    actions = self.storage.actions.flatten(0,1)
    per_row = torch.where(torch.isfinite(actions),actions.abs(),0.).amax(-1).double()
    for row,selected in enumerate((~is_rsi,is_rsi)):
      cohort_maxima[row,2] = torch.where(selected,per_row,0.).amax()
    measure(self.storage.distribution_params[0], 5)
    measure(self.storage.returns, 6)
    measure(self.storage.advantages, 7)
    std = self.storage.distribution_params[1]
    good_std = torch.isfinite(std) & (std > 0)
    nonfinite += (~good_std).sum()
    std_max = torch.where(good_std, std, 0.).amax().double()
    std_min = torch.where(torch.isfinite(std), std, 0.).amin().double()
    values = self._global_max(torch.cat((maxima, std_max[None], -std_min[None], cohort_maxima.flatten())))
    names = ('actor_obs_abs_max', 'actor_normalized_obs_abs_max',
             'critic_obs_abs_max', 'critic_normalized_obs_abs_max',
             'action_abs_max', 'old_mean_abs_max', 'return_abs_max', 'advantage_abs_max',
             'old_std_max', 'old_std_min')
    result = {'risk/'+name:float(value) for name,value in zip(names, values.tolist())}
    result['risk/old_std_min'] *= -1
    for row,cohort in enumerate(('non_rsi','rsi')):
      for column,name in enumerate(('actor_obs_abs_max','actor_normalized_obs_abs_max','action_abs_max')):
        result[f'risk/{cohort}/{name}'] = values[10+row*3+column].item()
    result['risk/rollout_invalid_values'] = reduce_sum(nonfinite).item()
    if result['risk/rollout_invalid_values']:
      self._fail_numeric('Nonfinite rollout values or nonpositive action std', result)
    return result

  def _fail_numeric(self, message, diagnostics):
    self.last_metrics = diagnostics
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
      print('[RiskPPO] '+message+' '+json.dumps(diagnostics, allow_nan=False), flush=True)
    raise FloatingPointError(message)

  def _step_snapshot(self):
    return ({key:value.detach().clone() for key,value in self._raw_actor.state_dict().items()},
            {key:value.detach().clone() for key,value in self._raw_critic.state_dict().items()},
            copy.deepcopy(self.optimizer.state_dict()))

  def _restore_step(self, snapshot):
    self._raw_actor.load_state_dict(snapshot[0])
    self._raw_critic.load_state_dict(snapshot[1])
    self.optimizer.load_state_dict(snapshot[2])
    self.optimizer.zero_grad(set_to_none=True)
    # Refresh ephemeral distribution caches without sampling or advancing RNG.
    with torch.no_grad():
      model = self._raw_actor
      model.distribution.update(model.mlp(model.get_latent(self.storage.observations[0])))

  def _set_learning_rate(self, rate):
    self.learning_rate = rate
    for group in self.optimizer.param_groups:
      group['lr'] = rate

  def _invalid_parameters_or_optimizer(self):
    tensors = list(self.actor.parameters())+list(self.critic.parameters())
    tensors += [value for state in self.optimizer.state.values()
                for value in state.values() if torch.is_tensor(value)]
    invalid = torch.stack([(~torch.isfinite(value)).any().to(self.device) for value in tensors]).sum()
    return reduce_sum(invalid).item()

  @torch.no_grad()
  def _rollout_policy_kl(self):
    """Check every collected state, including states outside the trained batch."""
    observations = self.storage.observations.flatten(0, 1)
    old = tuple(value.flatten(0, 1) for value in self.storage.distribution_params)
    totals = torch.zeros(3, dtype=torch.float64, device=self.device)
    maximum = totals.new_zeros(())
    cohort_totals = totals.new_zeros(2, 2)
    cohort_maxima = totals.new_zeros(2)
    is_rsi = self.is_rsi.flatten()
    model = self._raw_actor
    for start in range(0, len(observations), 32768):
      stop = start+32768
      # MLPModel's stochastic forward samples an unused action. Update the
      # distribution directly so validation never consumes policy RNG.
      model.distribution.update(model.mlp(model.get_latent(observations[start:stop])))
      kl = model.get_kl_divergence(tuple(value[start:stop] for value in old),
                                 model.output_distribution_params).double()
      good = torch.isfinite(kl)
      finite_kl = torch.where(good, kl.clamp_min(0), 0.)
      totals += torch.stack((finite_kl.sum(), kl.new_tensor(kl.numel()), (~good).sum()))
      maximum = torch.maximum(maximum, finite_kl.amax())
      masks = torch.stack((~is_rsi[start:stop],is_rsi[start:stop]))
      cohort_totals[:,0] += (finite_kl[None]*masks).sum(-1)
      cohort_totals[:,1] += masks.sum(-1)
      cohort_maxima = torch.maximum(cohort_maxima,
          torch.where(masks,finite_kl[None],0.).amax(-1))
    totals = reduce_sum(totals)
    maximum = self._global_max(maximum)
    # Separate collectives leave the existing acceptance calculation unchanged.
    cohort_totals = reduce_sum(cohort_totals)
    cohort_maxima = self._global_max(cohort_maxima)
    self.kl_diagnostics = {'risk/post_step_kl_last':(totals[0]/totals[1].clamp_min(1)).item()}
    for row,name in enumerate(('non_rsi','rsi')):
      self.kl_diagnostics.update({
          f'risk/{name}/post_step_kl_mean':(cohort_totals[row,0]/cohort_totals[row,1].clamp_min(1)).item(),
          f'risk/{name}/post_step_kl_max':cohort_maxima[row].item(),
          f'risk/{name}/post_step_kl_samples':cohort_totals[row,1].item()})
    return (totals[0]/totals[1].clamp_min(1)).item(), maximum.item(), totals[2].item()

  @staticmethod
  def construct_algorithm(obs, env, cfg, device):
    algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    algorithm.env = env.unwrapped
    return algorithm

  def process_env_step(self, obs, rewards, dones, extras):
    index = self.storage.step
    robust = extras.get("robust")
    if robust is None:
      raise ValueError("RiskPPO requires scored adversarial environment transitions")
    overflow = robust.get("physics_overflow")
    if overflow is not None:
      self.physics_overflow.add_((overflow != 0).sum())
    costs = robust["costs"].detach().clone().to(self.device)
    groups = robust["groups"].to(self.device)
    if costs.shape != (self.storage.num_envs, 3) or not torch.isfinite(costs).all():
      raise ValueError("Invalid fall/recovery/speed costs")
    if ((groups < 0) | (groups >= 5)).any():
      raise ValueError("Unknown risk group")
    self.raw_costs[index] = costs
    self.groups[index] = groups
    rsi = robust.get('rsi_episode',torch.zeros_like(groups,dtype=torch.bool)).to(self.device).bool()
    if rsi.shape != (self.storage.num_envs,):
      raise ValueError('Invalid RSI cohort mask')
    self.is_rsi[index] = rsi
    if self.anchor_budget is not None:
      # RSI may fall while learning reference dynamics. Keep its PPO costs,
      # but never calibrate ordinary-start budgets against reference resets.
      anchor_robust = dict(robust,calibration_mask=robust["calibration_mask"].bool() & ~rsi)
      self.anchor_budget.observe(anchor_robust)
      self.is_generated[index] = (robust["slice_id"] == 2) | rsi
      self.constraint_groups[index] = robust["anchor_group"].long().clamp(0, 4)
    self.transition.rewards = rewards.detach().clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      timeouts = extras["time_outs"].to(self.device)
      self.transition.rewards += self.gamma*self.transition.values[:, 0]*timeouts
      costs += self.gamma*self.transition.values[:, 1:]*timeouts[:, None]
    self.costs[index] = costs
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs):
    st = self.storage
    last = self.critic(obs).detach()
    advantage = torch.zeros_like(last)
    for step in reversed(range(st.num_transitions_per_env)):
      next_value = last if step == st.num_transitions_per_env-1 else st.values[step+1]
      continuation = 1.-st.dones[step].float()
      vector_reward = torch.cat((st.rewards[step], self.costs[step]), -1)
      delta = vector_reward+self.gamma*continuation*next_value-st.values[step]
      advantage = delta+self.gamma*self.lam*continuation*advantage
      st.returns[step] = advantage+st.values[step]
    vector_advantage = st.returns-st.values
    multipliers = self.multipliers
    if self.anchor_budget is not None:
      by_group = self.anchor_budget.multipliers[self.constraint_groups]
      multipliers = torch.where(self.is_generated[..., None],
                                self.anchor_budget.multipliers.amax(0), by_group)
    combined = vector_advantage[..., 0]-(vector_advantage[..., 1:]*multipliers).sum(-1)
    mean, variance, _ = global_moments(combined.reshape(-1))
    combined = (combined-mean)/(variance.sqrt()+1e-8)
    if self.anchor_budget is not None:
      st.advantages.copy_(combined.unsqueeze(-1))
      return
    weights = self.group_weights[self.groups]
    weights = weights/all_reduce_mean(weights.mean()).clamp_min(1e-6)
    st.advantages.copy_((combined*weights).unsqueeze(-1))

  def _normalization_update(self):
    normalizer = self._raw_critic.obs_normalizer
    if not hasattr(normalizer, "count"):
      return
    observations = self.storage.observations.flatten(0, 1)
    samples = torch.cat([observations[key] for key in self._raw_critic.obs_groups], -1)
    mean, variance, count = global_moments(samples)
    total = normalizer.count+count
    fraction = count.float()/total.clamp_min(1)
    delta = mean-normalizer._mean
    normalizer._mean.add_(fraction*delta)
    normalizer._var.add_(fraction*(variance-normalizer._var+delta*(mean-normalizer._mean)))
    normalizer._var.clamp_(min=1e-8)
    normalizer._std.copy_(normalizer._var.sqrt())
    normalizer.count.copy_(total)

  def _risk_update(self):
    masks = torch.stack((~self.is_rsi,self.is_rsi)).double()
    counts = reduce_sum(torch.stack((masks.sum((1,2)),
                                    (masks*self.raw_costs[...,0]).sum((1,2))),-1))
    cohorts = {f'risk/{name}/{metric}':counts[row,column].item()
               for row,name in enumerate(('non_rsi','rsi'))
               for column,metric in enumerate(('world_steps','falls'))}
    if self.anchor_budget is not None:
      result = self.anchor_budget.update(reduce_sum)
      self.multipliers.copy_(self.anchor_budget.multipliers.amax(0))
      measured = all_reduce_mean(self.raw_costs.mean((0, 1)))
      for i, name in enumerate(("fall", "recovery", "stall")):
        result[f"training/{name}_per_step"] = measured[i].item()
      result.update(cohorts)
      return result
    measured = all_reduce_mean(self.raw_costs.mean((0, 1)))
    self.cost_ema.lerp_(measured, .05)
    error = ((self.cost_ema-self.cost_limits)/self.cost_limits).clamp(-1, 5)
    if self.update_index >= self.warmup_updates:
      self.multipliers.add_(self.dual_lr*error).clamp_(0, 200)
    burden = (self.raw_costs*self.multipliers).sum(-1).reshape(-1)
    groups = self.groups.reshape(-1)
    sums, counts = torch.zeros(5, device=self.device), torch.zeros(5, device=self.device)
    sums.scatter_add_(0, groups, burden)
    counts.scatter_add_(0, groups, torch.ones_like(burden))
    totals = reduce_sum(torch.stack((sums, counts)))
    active = totals[1] > 0
    rate = totals[0]/totals[1].clamp_min(1)
    self.group_cost_ema[active] = .95*self.group_cost_ema[active]+.05*rate[active]
    relative = self.group_cost_ema/self.group_cost_ema.mean().clamp_min(.01)
    self.group_weights.copy_(torch.exp(.3*(relative-1)).clamp(.5, 2.))
    result = dict(cohorts)
    for i,name in enumerate(("fall", "recovery", "stall")):
      result[f"risk/{name}_rate"] = measured[i].item()
      result[f"risk/{name}_multiplier"] = self.multipliers[i].item()
    for i in range(5): result[f"risk/group_{i}_weight"] = self.group_weights[i].item()
    return result

  def update(self):
    overflow = reduce_sum(self.physics_overflow).item()
    self.physics_overflow.zero_()
    if overflow:
      raise RuntimeError(f"MuJoCo capacity overflow in {int(overflow)} world-steps; "
                         "no policy/auxiliary update performed. Increase simulation capacity.")
    diagnostics = self._rollout_diagnostics()
    if self._invalid_parameters_or_optimizer():
      self._fail_numeric('Nonfinite parameter or Adam state before optimizer update', diagnostics)
    totals = dict(value=0., surrogate=0., entropy=0., kl=0.)
    updates = 0
    attempts = 0
    rejected = 0
    maximum_kl = 0.
    post_kl = 0.
    accepted_kl = 0.
    sample_kl = 0.
    validation_seconds = 0.
    self.kl_diagnostics = {'risk/post_step_kl_last':0.,
        **{f'risk/{cohort}/post_step_kl_{metric}':0.
           for cohort in ('non_rsi','rsi') for metric in ('mean','max','samples')}}
    cohort_kl_peaks = {f'risk/{cohort}/post_step_kl_{metric}_peak':0.
                      for cohort in ('non_rsi','rsi') for metric in ('mean','max')}
    actor_gradient_norm = 0.
    critic_gradient_norm = 0.
    actor_parameter_delta = 0.
    early_stop = False
    warmup = (self.update_index < self.warmup_updates or
              self.anchor_budget is not None and not self.anchor_budget.ready)
    validate_actor = (not warmup and self.desired_kl is not None
                      and any(parameter.requires_grad for parameter in self.actor.parameters()))
    _, variance, _ = global_moments(self.storage.returns.reshape(-1, 4))
    if not torch.isfinite(variance).all():
      self._fail_numeric('Nonfinite return variance before optimizer update', diagnostics)
    value_scale = variance.sqrt().clamp_min(.1)
    generator = self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
    for batch in generator:
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      entropy = self.actor.output_entropy.mean()
      values = self.critic(batch.observations)
      kl = all_reduce_mean(self.actor.get_kl_divergence(
        batch.old_distribution_params,self.actor.output_distribution_params).mean()).item()
      if not math.isfinite(kl):
        self._fail_numeric('Nonfinite pre-step policy KL', diagnostics)
      maximum_kl = max(maximum_kl, kl)
      if not warmup and self.schedule == "adaptive" and self.desired_kl is not None:
        if kl > 4*self.desired_kl:
          self.learning_rate = max(1.e-6, self.learning_rate/2)
        elif kl > 2*self.desired_kl:
          self.learning_rate = max(1.e-6, self.learning_rate/1.5)
        elif 0 < kl < self.desired_kl/2:
          self.learning_rate = min(self.maximum_learning_rate, self.learning_rate*1.5)
        for group in self.optimizer.param_groups:
          group["lr"] = self.learning_rate
      if not warmup and self.desired_kl is not None and kl > 4*self.desired_kl:
        early_stop = True
        break
      ratio = (log_prob-batch.old_actions_log_prob.squeeze(-1)).exp()
      advantage = batch.advantages.squeeze(-1)
      surrogate = torch.maximum(-advantage*ratio,
        -advantage*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
      clipped = batch.values+(values-batch.values).clamp(-self.clip_param*value_scale,
                                                       self.clip_param*value_scale)
      error = ((values-batch.returns)/value_scale).square()
      if self.use_clipped_value_loss:
        error = torch.maximum(error,((clipped-batch.returns)/value_scale).square())
      value_loss = error.mean()
      loss = self.value_loss_coef*value_loss
      if not warmup: loss = loss+surrogate-self.entropy_coef*entropy
      if reduce_sum((~torch.isfinite(loss)).float()).item():
        self._fail_numeric('Nonfinite PPO loss before optimizer update', diagnostics)
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      if self.is_multi_gpu: self.reduce_parameters()
      gradients = [parameter.grad for model in (self.actor,self.critic)
                   for parameter in model.parameters() if parameter.grad is not None]
      invalid_gradients = torch.stack([(~torch.isfinite(value)).any() for value in gradients]).sum()
      if reduce_sum(invalid_gradients).item():
        self._fail_numeric('Nonfinite gradients before optimizer update', diagnostics)
      actor_norm = nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
      critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
      norms = self._global_max(torch.stack((actor_norm.to(self.device),critic_norm.to(self.device))))
      if not torch.isfinite(norms).all():
        self._fail_numeric('Nonfinite gradient norm before optimizer update', diagnostics)
      actor_gradient_norm = max(actor_gradient_norm, norms[0].item())
      critic_gradient_norm = max(critic_gradient_norm, norms[1].item())
      snapshot = self._step_snapshot()
      attempts += 1
      self.optimizer.step()
      if self._invalid_parameters_or_optimizer():
        self._restore_step(snapshot)
        self._set_learning_rate(max(1.e-6, self.learning_rate/2))
        self._fail_numeric('Nonfinite optimizer proposal; model and Adam restored', diagnostics)
      deltas = torch.stack([(parameter-snapshot[0][name]).abs().amax()
                            for name,parameter in self._raw_actor.named_parameters()])
      actor_parameter_delta = max(actor_parameter_delta, self._global_max(deltas.amax()).item())
      if validate_actor:
        validation_started = time.perf_counter()
        proposal_kl, proposal_max, invalid = self._rollout_policy_kl()
        for cohort in ('non_rsi','rsi'):
          for metric in ('mean','max'):
            key=f'risk/{cohort}/post_step_kl_{metric}'
            cohort_kl_peaks[key+'_peak']=max(cohort_kl_peaks[key+'_peak'],self.kl_diagnostics[key])
        validation_seconds += time.perf_counter()-validation_started
        if invalid:
          self._restore_step(snapshot)
          self._set_learning_rate(max(1.e-6, self.learning_rate/2))
          self._fail_numeric('Nonfinite post-step policy KL; model and Adam restored', diagnostics)
        maximum_kl = max(maximum_kl, proposal_kl)
        post_kl = max(post_kl, proposal_kl)
        sample_kl = max(sample_kl, proposal_max)
        if proposal_kl > 4*self.desired_kl:
          self._restore_step(snapshot)
          self._set_learning_rate(max(1.e-6, self.learning_rate/2))
          rejected += 1
          early_stop = True
          break
        accepted_kl = proposal_kl
      for name,value in (("value",value_loss),("surrogate",surrogate),("entropy",entropy)):
        totals[name] += value.item()
      totals["kl"] += kl
      updates += 1
    # Every rank has the same minibatch/early-stop schedule. Publish the global
    # mean rather than presenting rank zero's local loss as a shared metric.
    names = tuple(totals)
    averaged = all_reduce_mean(torch.tensor(
        [totals[name]/max(1,updates) for name in names], device=self.device))
    result = {name:averaged[i].item() for i,name in enumerate(names)}
    result.update(diagnostics)
    result.update(self.kl_diagnostics)
    result.update(cohort_kl_peaks)
    result.update(self._risk_update())
    with torch.no_grad(): self._normalization_update()
    result.update({"risk/warmup":float(warmup),"risk/optimizer_steps":float(updates),
                   "risk/attempted_optimizer_steps":float(attempts),
                   "risk/rejected_optimizer_steps":float(rejected),
                   "risk/max_observed_kl":maximum_kl,"risk/kl_early_stop":float(early_stop),
                   "risk/post_step_kl":post_kl,"risk/post_step_kl_sample_max":sample_kl,
                   "risk/accepted_rollout_kl":accepted_kl,
                   "risk/post_step_validation_seconds":self._global_max(
                       torch.tensor(validation_seconds, device=self.device)).item(),
                   "risk/actor_gradient_norm_max":actor_gradient_norm,
                   "risk/critic_gradient_norm_max":critic_gradient_norm,
                   "risk/actor_parameter_delta_abs_max":actor_parameter_delta,
                   "risk/learning_rate":float(self.learning_rate)})
    if self.env is not None:
      prior = getattr(self.env,"_r1_human_prior",None)
      adversary = getattr(self.env,"_r1_adversary",None)
      if prior is not None:
        result.update({"human/"+key:value for key,value in prior.update(reduce_sum).items()
                       if isinstance(value,(int,float))})
      if adversary is not None:
        # Fixed-anchor physical measurements gate the current curriculum.
        # The moving training discriminator is never its own quality judge.
        # Keep the old prototype path for checkpoint/test compatibility.
        stats = getattr(prior, "stats", {}) if prior is not None else {}
        if self.anchor_budget is not None:
          anchor = self.anchor_budget
          rates = anchor.ema[..., 0]/anchor.ema[..., 1].clamp_min(1e-9)
          adversary.quality_gate = bool(not warmup and anchor.ready and
              (rates[:, :2] <= 2*anchor.limits[:, :2]).all())
        else:
          adversary.quality_gate = bool(not warmup
              and stats.get("HumanPrior/validation_valid", 0.)
              and stats.get("HumanPrior/reference_accuracy", 0.) >= .6
              and stats.get("HumanPrior/policy_probability", 0.) >= .35
              and stats.get("HumanPrior/probability_gap", 1.) <= .35
              and self.cost_ema[0].item() <= 5*self.cost_limits[0].item())
        result["environment/anchor_safety_gate"] = float(adversary.quality_gate)
        result.update({"environment/"+key:value for key,value in adversary.update(reduce_sum).items()
                       if isinstance(value,(int,float))})
    self.update_index += 1
    self.storage.clear()
    self.last_metrics = result
    if callback := getattr(self, "on_update", None):
      callback(result, self.update_index)
    return result

  def save(self):
    state = super().save()
    state["risk_state"] = {"multipliers":self.multipliers,"cost_ema":self.cost_ema,
        "group_cost_ema":self.group_cost_ema,"group_weights":self.group_weights,
        "update_index":self.update_index,"learning_rate":self.learning_rate}
    if self.anchor_budget is not None:
      state["anchor_budgets"] = self.anchor_budget.state_dict()
      state["risk_state"].update(cost_protocol_version=3,
                                 stall_protocol_version=self.anchor_budget.stall_protocol_version)
    if self.env is not None:
      for key, attribute in (("human_prior","_r1_human_prior"),("environment_agent","_r1_adversary")):
        model = getattr(self.env,attribute,None)
        if model is not None:
          state[key] = model.state_dict() if key=="human_prior" else model.checkpoint()
    return state

  def load(self, state, load_cfg=None, strict=True):
    result = super().load(state,load_cfg,strict)
    resume_auxiliary = load_cfg is None or load_cfg.get("optimizer",False)
    if resume_auxiliary and "risk_state" in state:
      saved = state["risk_state"]
      for key in ("multipliers","cost_ema","group_cost_ema","group_weights"):
        getattr(self,key).copy_(saved[key].to(self.device))
      self.update_index = saved["update_index"]
      self.learning_rate = float(saved.get("learning_rate", self.optimizer.param_groups[0]["lr"]))
      if self.schedule == "adaptive":
        self.learning_rate = min(self.learning_rate, self.maximum_learning_rate)
      for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
    if resume_auxiliary and self.anchor_budget is not None and "anchor_budgets" in state:
      self.anchor_budget.load_state_dict(state["anchor_budgets"])
      # The revised measurement retains the seconds-based stall return and its
      # four-head value layout. Preserve the trained critic and Adam warm start.
      self.anchor_budget.restore_metadata.update(critic_stall_head="preserved",
                                                 ppo_optimizer="preserved")
    if resume_auxiliary and self.env is not None:
      for key,attribute in (("human_prior","_r1_human_prior"),("environment_agent","_r1_adversary")):
        model = getattr(self.env,attribute,None)
        if model is not None and key in state:
          model.load_state_dict(state[key]) if key=="human_prior" else model.restore(state[key])
    return result
