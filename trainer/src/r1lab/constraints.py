"""Fixed-scenario risk budgets, independent of the learned adversary's distribution."""
import torch


class AnchorBudgets:
  """Counts episodes rather than survival steps; moving time normalizes stall."""
  names = ("clean", "ball", "force_push", "terrain", "material")
  costs = ("fall_per_episode", "unrecovered_seconds", "stall_fraction")
  state_version = 2
  stall_protocol_version = 2

  def __init__(self, device, initial=(120., 12., 1.), calibration_updates=64,
               minimum_episodes=200, integral_rate=.02):
    self.device = device
    self.calibration_updates = calibration_updates
    self.minimum_episodes = minimum_episodes
    self.integral_rate = integral_rate
    self.pending = torch.zeros(5, 3, 2, device=device)
    self.baseline = torch.zeros_like(self.pending)
    self.window = torch.zeros_like(self.pending)
    self.ema = torch.zeros_like(self.pending)
    self.limits = torch.tensor([.001, .1, .01], device=device).repeat(5, 1)
    self.integral = torch.tensor(initial, device=device).repeat(5, 1)
    self.multipliers = self.integral.clone()
    self.steps = 0
    self.ready = False
    self.stall_episode_counts = torch.zeros(5, dtype=torch.float64, device=device)
    self.stall_ready = torch.zeros(5, dtype=torch.bool, device=device)
    self.restore_metadata = {}

  def observe(self, robust):
    if robust.get("stall_protocol_version") != self.stall_protocol_version:
      raise ValueError("Fixed-anchor observations require stall_protocol_version=2")
    valid = (robust["calibration_mask"].bool() & robust["completed"].bool()
             & ~robust["episode_censored"].bool())
    groups = robust["anchor_group"].long()[valid]
    if ((groups < 0) | (groups >= 5)).any():
      raise ValueError("Unknown fixed-anchor risk group")
    numerator = torch.stack((robust["episode_fall"].float(),
        robust["episode_recovery_seconds"], robust["episode_stall_seconds"]), -1)[valid]
    moving = robust["episode_commanded_seconds"][valid]
    denominator = torch.stack((torch.ones_like(moving), torch.ones_like(moving), moving), -1)
    values = torch.stack((numerator, denominator), -1)
    if not torch.isfinite(values).all() or (values < 0).any():
      raise ValueError("Invalid fixed-anchor episode counts")
    self.pending.view(5, 6).index_add_(0, groups, values.reshape(-1, 6))

  def update(self, reduce_sum):
    totals = reduce_sum(self.pending).clone()
    self.pending.zero_()
    self.steps += 1
    was_stall_ready = self.stall_ready.clone()
    self.stall_episode_counts += totals[:, 0, 1].double()
    if not self.ready:
      self.baseline += totals
      self.stall_ready |= ((self.stall_episode_counts >= self.minimum_episodes)
                           & (self.baseline[:, 2, 1] > 0))
      enough = (self.baseline[:, 0, 1] >= self.minimum_episodes).all()
      if self.steps >= self.calibration_updates and bool(enough):
        measured = self.baseline[..., 0]/self.baseline[..., 1].clamp_min(1e-9)
        # Fall and stall caps express the task goal, not a claimed certification.
        # Recovery may not regress from the same frozen starting policy; a
        # nonzero floor avoids an impossible zero-second empirical budget.
        self.limits[:, 1] = measured[:, 1].clamp(min=.1, max=1.)
        self.ema.copy_(self.baseline/max(1, self.steps))
        self.ready = True
    else:
      # Fall/recovery retain their exact checkpoint statistics and cadence.
      # A migrated stall stream calibrates independently from new episodes.
      calibrating = ~was_stall_ready
      self.baseline[calibrating, 2] += totals[calibrating, 2]
      newly_ready = (calibrating & (self.stall_episode_counts >= self.minimum_episodes)
                     & (self.baseline[:, 2, 1] > 0))
      self.stall_ready |= newly_ready
      self.ema[newly_ready, 2] = self.baseline[newly_ready, 2]
      self.window[:, :2] += totals[:, :2]
      self.window[was_stall_ready, 2] += totals[was_stall_ready, 2]
      for group in range(5):
        if self.window[group, 0, 1] >= self.minimum_episodes:
          self.ema[group, :2].lerp_(self.window[group, :2], .1)
          self._update_dual(group, slice(0, 2))
          # Standing-only windows do not provide a stall rate denominator.
          if was_stall_ready[group] and self.window[group, 2, 1] > 0:
            self.ema[group, 2].lerp_(self.window[group, 2], .1)
            self._update_dual(group, 2)
          self.window[group].zero_()
        if newly_ready[group]:
          self._update_dual(group, 2)
    result = {"constraints/calibrated":float(self.ready),
              "constraints/stall_calibrated":float(self.stall_ready.all()),
              "constraints/stall_protocol_version":float(self.stall_protocol_version)}
    rate = self.ema[..., 0]/self.ema[..., 1].clamp_min(1e-9)
    for group, name in enumerate(self.names):
      result[f"constraints/{name}/baseline_episodes"] = self.baseline[group, 0, 1].item()
      result[f"constraints/{name}/observed_episodes"] = totals[group, 0, 1].item()
      result[f"constraints/{name}/observed_moving_seconds"] = totals[group, 2, 1].item()
      result[f"constraints/{name}/stall_calibrated"] = float(self.stall_ready[group])
      result[f"constraints/{name}/stall_calibration_episodes"] = self.stall_episode_counts[group].item()
      for cost, label in enumerate(self.costs):
        key = f"constraints/{name}/{label}"
        result[key] = rate[group, cost].item()
        result[key+"_limit"] = self.limits[group, cost].item()
        result[key+"_multiplier"] = self.multipliers[group, cost].item()
    return result

  def _update_dual(self, group, costs):
    moments = self.ema[group, costs]
    rate = moments[..., 0]/moments[..., 1].clamp_min(1e-9)
    error = (rate/self.limits[group, costs]-1).clamp(-1, 5)
    error = torch.where(moments[..., 1] > 0, error, 0.)
    current = self.integral[group, costs]
    windup = ((current >= 200) & (error > 0)) | ((current <= 0) & (error < 0))
    current.add_(self.integral_rate*torch.where(windup, 0., error)).clamp_(0, 200)
    self.multipliers[group, costs] = (current+error.clamp_min(0)).clamp(0, 200)

  def state_dict(self):
    return {"version":self.state_version, "stall_protocol_version":self.stall_protocol_version,
            "steps":self.steps, "ready":self.ready,
            **{key:getattr(self,key).clone() for key in
               ("baseline","window","ema","limits","integral","multipliers",
                "stall_episode_counts","stall_ready")}}

  def load_state_dict(self, state):
    version = state.get("version")
    if version not in (1, self.state_version):
      raise ValueError("Unsupported fixed-anchor budget checkpoint")
    old_stall = state.get("stall_protocol_version", 1)
    if old_stall not in (1, self.stall_protocol_version) or (version == self.state_version and old_stall != self.stall_protocol_version):
      raise ValueError("Unsupported fixed-anchor stall protocol")
    self.steps, self.ready = state["steps"], state["ready"]
    for key in ("baseline","window","ema","limits","integral","multipliers"):
      getattr(self,key).copy_(state[key].to(self.device))
    migrated = old_stall != self.stall_protocol_version
    if migrated:
      for key in ("pending", "baseline", "window", "ema"):
        getattr(self, key)[:, 2].zero_()
      self.stall_episode_counts.zero_()
      self.stall_ready.zero_()
    else:
      for key in ("stall_episode_counts", "stall_ready"):
        if key not in state:
          raise ValueError(f"Missing fixed-anchor checkpoint field: {key}")
        getattr(self, key).copy_(state[key].to(self.device))
    self.restore_metadata = dict(state_version_from=version, state_version_to=self.state_version,
        stall_protocol_from=old_stall, stall_protocol_to=self.stall_protocol_version,
        stall_statistics_reset=migrated, preserved="fall/recovery statistics, all limits/integrals/multipliers",
        stall_adaptation="per-group >=minimum_episodes new completed episodes and positive commanded seconds",
        minimum_episodes=self.minimum_episodes)
