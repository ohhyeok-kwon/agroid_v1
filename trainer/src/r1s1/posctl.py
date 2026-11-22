"""R13 positive controls (positive 23, coordinator 2026-11-20 09:2x-10:1x KST; negative 18 addenda 46-47, freezes 84-85).

DIAGNOSTIC ONLY - NOT ADOPTED INTO S1. Two tasks share one frame: wb_s1_s0b_power_fixed's observations (actor 1,959 =
0.5 s history + the 9-value goal, fixed physical scales), actions, startup randomisation, latency and pushes,
terminations (body other than the feet on the floor, self-collision; self-collision physics on), simulation, episode,
goal distribution, PPO (S1PPO 16,384 x 24 with its in-run gates), mirror-symmetric 512-256-128 actor and fixed std
sigma_500 x 0.5. Only the reward block changes, plus what its terms read (a derived command term, never observed; for
the walking control two contact sensors and the recipe's own penalty ramp). No manufacturer policy, model or data.

  wb_s1_posctl_walk   (A) the verified recipe, walking only: careful_walk's velocity-tracking stage 1 "learn to walk"
                      with the term functions, weights and kernel widths recorded by its walking run
                      (r1_careful_walk/2026-11-08_05-01-55_s1, params/env.yaml sha256 9812463c...; random-init lineage
                      amp_v4@0 -> v5 -> v6@900 -> s1, deterministic 8-direction walking at 0.3 m/s by update 1,400).
                      24 non-zero terms including old-recipe ingredients (action_rate, joint_limits, is_terminated),
                      gait terms (air_time, feet_distance, stand_still, pose) and the AMP style term on retargeted human
                      clips: the one objective with a record of teaching this body to walk from random weights, so a
                      failure here can be read against the frame (addendum 47); never a base.
  wb_s1_posctl_dense  (B) the common base candidate (coordinator 10:1x): S1's own objective v2 unchanged (success,
                      potential difference, power) + one dense tracking term per moving goal kind: planar velocity and
                      turn rate for locomotion goals, pelvis height rate for crouch goals; hold goals keep v2 alone. No
                      ingredient-table name (r1s1.gates.FORBIDDEN), no debt (lag integral, deadline-chasing command),
                      no reference-height termination, no post-hit grace, no gait-rule term, no human data.

The command (term "twist" for A, "pace" for B; reward side only) - direction to the goal x the goal spec's own fixed
rate, never raised by lag (coordinator 10:1x 1):
  locomotion goal: pace p and turn rate w = the goal's own draws (numbers.PACE_MPS, TURN_RATE_RPS), which r1s1.goal
                   draws for the deadline budget tau0 = SETTLE_S + D / p + |dpsi| / w and discards; the goal term is
                   PacedGoalCommand, which records them from those very draws (same distribution, same random stream).
  crouch goal:     height rate r_h = the goal's drawn rate (budget SETTLE_S + |dh| / r_h); planar and turn rates 0.
  hold goal:       all rates 0.
  v = u * p while |e_p| > TOL_p (u = e_p / |e_p|, pelvis heading frame), else 0; w_z = sign(e_psi) * w while |e_psi| >
  TOL_psi, else 0; hdot = sign(e_h) * r_h while |e_h| > TOL_h, else 0. The rates are fixed when the goal is issued.
  The actor's goal observation on the issue step shows D = |e_p|, |dpsi|, |dh| and tau0 = tau: r_h follows exactly,
  p and w only through the one budget equation (issue_budget_residual), so a turning goal leaves the pace partly unseen.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from r1s1 import numbers as N
from r1s1.goal import GoalCommand, GoalCommandCfg

VERSION = "s1-posctl-v3"            # v3: the goal's drawn pace / turn rate (v2's budget-average rates halved the median
                                    # commanded speed to 0.19 m/s; v1's deadline-chasing command was a debt, 10:1x)
TASK = "wb_s1_posctl_walk"
TASK_DENSE = "wb_s1_posctl_dense"
BASE_TASK = "wb_s1_s0b_power_fixed"
MOVING = 0.05                       # m/s: a walking command in the records (the recipe's own walking / command threshold)
FLIGHT_N = 1.0                      # N: a foot with less vertical floor force than this is off the floor (records only)

# --- (A) the verified recipe ------------------------------------------------------------------------------------
SOURCE_RUN = "r1_careful_walk/2026-11-08_05-01-55_s1"
SOURCE_ENV_YAML_SHA256 = "9812463c8445c8d2cc3b638104497c2198ff30393cb717c8bc8222b52638acd6"
AMP_DATA_SHA256 = "3ff0f61e6e34a34c31c88e99f724f143f1a20cc36b9f38e684834e94dd50819f"
TERMS = ("track_lin_vel", "track_ang_vel", "lin_vel_err", "yaw_rate_err", "alive", "upright", "base_height",
         "lin_vel_z", "ang_vel_xy", "torso_ang_vel", "angular_momentum", "feet_distance", "stand_still", "pose",
         "air_time", "action_rate", "action_acc", "joint_acc", "joint_vel", "power", "joint_limits", "limb_contact",
         "terminated", "style")
# the four entries where today's careful_walk_env_cfg(stage=1) differs from the walking run's recorded params/env.yaml:
# that run tracked the instantaneous pelvis velocity (mjlab's stock kernels and r1lab's instantaneous L1 errors), not
# today's 0.4 s EMA terms, with these weights and kernel widths (name: (module, function, weight, params))
RECORDED = {
  "track_lin_vel": ("mjlab.tasks.velocity.mdp.rewards", "track_linear_velocity", 3.0, {"command_name": "twist", "std": 0.15}),
  "track_ang_vel": ("mjlab.tasks.velocity.mdp.rewards", "track_angular_velocity", 1.5, {"command_name": "twist", "std": 0.3}),
  "lin_vel_err": ("r1lab.mdp.rewards", "lin_vel_error_l1", -3.0, {"command_name": "twist"}),
  "yaw_rate_err": ("r1lab.mdp.rewards", "yaw_rate_error_l1", -1.0, {"command_name": "twist"}),
}
SENSORS = ("feet_contact", "limb_contact")

# --- (B) dense tracking terms on top of objective v2 ------------------------------------------------------------
DENSE = {  # name: (weight, kernel width, goal kind)
  "s1_dense_planar": (1.0, 0.15, 1),
  "s1_dense_turn": (0.5, 0.3, 1),
  "s1_dense_height": (1.0, 0.05, 2),
}
DENSE_TERMS = ("s1_success", "s1_progress", "s1_effort_power") + tuple(DENSE)

METRICS = ("cmd_speed", "moving_share", "cmd_speed_moving", "vel_along_moving", "error_vel_xy", "error_vel_yaw",
           "flight_share", "flight_share_moving", "cmd_hdot_abs", "error_hdot")

REASONS = {
  "MOVING": "careful_walk's own walking threshold (pose walking_threshold, air_time and stand_still command threshold)",
  "FLIGHT_N": "the S1 termination's floor-contact detection level (numbers.GROUND_FORCE_N): records only (hopping: both "
              "feet off the floor at once), never a reward",
  "RECORDED": "careful_walk.py was edited after the walking run (tracking moved to 0.4 s EMA terms, track_ang_vel 1.5 "
              "-> 3.0 and std 0.3 -> 0.4, track_lin_vel std 0.15 -> 0.2, lin_vel_err -3 -> -2, yaw_rate_err -1 -> -2); "
              "the run's recorded functions, weights and widths are the verified ones. The run kept no code snapshot: "
              "term bodies are today's mjlab / r1lab ones",
  "AMP_DATA_SHA256": "the AMP reference file every careful_walk run since 11-08 07:01 read (s2..s2f, r6d); the walking "
                     "run (05:01) read the file of that name before its last rebuild, not hashed then",
  "rates": "coordinator 10:1x: command = goal direction x the goal spec's fixed pace; a command that grows when the robot "
           "lags (remaining distance / remaining time) is a debt and is not used. The pace and turn rate are the goal's "
           "own draws (numbers.PACE_MPS, TURN_RATE_RPS), which r1s1.goal draws for the deadline budget (tau0 = SETTLE_S + "
           "D / pace + |dpsi| / rate) and discards; PacedGoalCommand records them from those very draws. Moving and "
           "turning together at them ends both before the budget. (v2's budget averages D / (tau0 - SETTLE_S), "
           "|dpsi| / (tau0 - SETTLE_S) were exactly readable on the issue step but put the median planar command at "
           "0.19 m/s against the drawn 0.40, 32 % of goals under 0.15 m/s, where standing still keeps exp(-1) of the "
           "planar kernel: README r13_posctl 3.1)",
  "s1_dense_planar": "weight 1 = W_SUCCESS, the verified recipe's walking-to-standing ratio: there a robot standing at a "
                     "walking command still collects alive 0.5 + upright 1 + pose 0.4 + yaw tracking 1.5 = 3.4 per "
                     "second and walking on command adds track_lin_vel 3.0 (0.9 of it); here standing collects "
                     "W_SUCCESS = 1 per second (standing at a 0.2-0.6 m/s command scores exp(-(0.2/0.15)^2) = 0.17 "
                     "down to 1e-7 of the term); width 0.15 m/s = the recipe's planar kernel (track_linear_velocity "
                     "std); the kernel is mjlab's (planar error^2 + vertical speed^2)",
  "s1_dense_turn": "weight 0.5 = the verified recipe's yaw / planar weight ratio (1.5 / 3.0); width 0.3 rad/s = its yaw "
                   "kernel (track_angular_velocity std); the kernel is mjlab's (yaw-rate error^2 + roll / pitch rate^2)",
  "s1_dense_height": "weight 1 = the planar term's (one dense term per moving goal kind, same scale); width 0.05 m/s = "
                     "half the slowest crouch rate (numbers.CROUCH_RATE_MPS 0.1 m/s): holding still at a crouch command "
                     "scores exp(-4) = 0.02 at the slowest rate, and moving at it scores 1; inside the height tolerance "
                     "the command is 0 and the term rewards holding the height",
}


class PacedGoalCommand(GoalCommand):
  """r1s1.goal.GoalCommand that also keeps each goal's drawn pace and turn rate. The base class draws them only for the
  locomotion deadline budget and discards them; here its own draws are recorded as they are made (r1s1.goal._uniform
  wrapped for the duration of the base call), so the goal distribution, the random stream, the goal state and the
  observation are exactly the base class's (README r13_posctl 5.2: the same draws, bit for bit)."""

  def __init__(self, cfg: "PacedGoalCommandCfg", env):
    super().__init__(cfg, env)
    self.pace = torch.zeros(self.num_envs, device=self.device)
    self.turn_rate = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    from r1s1 import goal as G
    original, drawn = G._uniform, {}

    def recording(count, low, high, device):
      value = original(count, low, high, device)
      drawn[(float(low), float(high))] = value
      return value

    G._uniform = recording
    try:
      super()._resample_command(env_ids)
    finally:
      G._uniform = original
    self.pace[env_ids] = drawn[tuple(float(x) for x in N.PACE_MPS)]
    self.turn_rate[env_ids] = drawn[tuple(float(x) for x in N.TURN_RATE_RPS)]


@dataclass(kw_only=True)
class PacedGoalCommandCfg(GoalCommandCfg):
  keep_rates: bool = True                                       # marks the recording goal term in the dumped config

  def build(self, env) -> PacedGoalCommand:
    return PacedGoalCommand(self, env)


def paced_goal_cfg(base: GoalCommandCfg) -> PacedGoalCommandCfg:
  """The same goal configuration (every field copied), built as PacedGoalCommand."""
  from dataclasses import fields
  return PacedGoalCommandCfg(**{f.name: getattr(base, f.name) for f in fields(base)})


def goal_rates(goal) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Per-goal fixed (pace, turn rate, height rate): the goal's own draws (module docstring); zero for other kinds."""
  zero = torch.zeros(goal.num_envs, device=goal.device)
  loco, crouch = goal.kind == 1, goal.kind == 2
  pace = torch.where(loco, goal.pace, zero)
  turn = torch.where(loco, goal.turn_rate, zero)
  hrate = torch.where(crouch, goal._crouch_rate, zero)
  return pace, turn, hrate


def command_from_goal(ep, epsi, eh, mask, pace, turn, hrate) -> torch.Tensor:
  """(n, 4) [v_x, v_y (pelvis heading frame), w_z, hdot]: direction to the goal x the fixed rate, 0 inside tolerance."""
  dist = ep.norm(dim=1)
  zero = torch.zeros_like(dist)
  move = (mask[:, 0] > 0.5) & (dist > N.TOL[0])
  v = ep / dist.clamp_min(1.0e-9).unsqueeze(1) * torch.where(move, pace, zero).unsqueeze(1)
  rot = (mask[:, 1] > 0.5) & (epsi.abs() > N.TOL[1])
  wz = torch.where(rot, torch.sign(epsi) * turn, zero)
  lift = (mask[:, 2] > 0.5) & (eh.abs() > N.TOL[2])
  hz = torch.where(lift, torch.sign(eh) * hrate, zero)
  return torch.stack([v[:, 0], v[:, 1], wz, hz], dim=1)


def issue_budget_residual(goal_obs: torch.Tensor, kind, pace, turn, hrate) -> torch.Tensor:
  """On a goal's issue step: |(tau0 - SETTLE_S) - (D / p + |dpsi| / w)| for locomotion goals and |(tau0 - SETTLE_S) -
  |dh| / r_h| for crouch goals (0 for hold goals), with D, |dpsi|, |dh|, tau0 read from the 9-value goal observation
  (r1s1.goal.GoalCommand._observe: e_p / lambda_p, sin / cos e_psi, e_h / lambda_h, mask, tau+ / 4 s): the recorded
  rates are the ones that set the deadline the actor sees."""
  budget = goal_obs[:, 8] * N.TAU_SCALE_S - N.SETTLE_S
  d = (goal_obs[:, 0:2] * N.LAMBDA[0]).norm(dim=1)
  psi = torch.atan2(goal_obs[:, 2], goal_obs[:, 3]).abs()
  h = (goal_obs[:, 4] * N.LAMBDA[2]).abs()
  loco = (budget - d / pace.clamp_min(1.0e-6) - psi / turn.clamp_min(1.0e-6)).abs()
  crouch = (budget - h / hrate.clamp_min(1.0e-6)).abs()
  zero = torch.zeros_like(budget)
  return torch.where(kind == 1, loco, torch.where(kind == 2, crouch, zero))


class GoalRateCommand(CommandTerm):
  """Reward-side command derived every step from the S1 goal term (registered after it). Nothing is drawn. Per-episode
  records (Metrics/<name>/...): mean commanded speed; share of steps with a walking command; over those steps the mean
  commanded speed and the mean pelvis velocity along the command; mean |v_cmd - v|, |w_cmd - w_z|; share of steps with
  both feet off the floor (hopping record) overall and over walking-command steps; mean |hdot_cmd| and |hdot_cmd - hdot|."""
  cfg: "GoalRateCommandCfg"

  def __init__(self, cfg: "GoalRateCommandCfg", env):
    super().__init__(cfg, env)
    n, device = self.num_envs, self.device
    self.robot = env.scene[cfg.entity_name]
    self._full = torch.zeros(n, 4, device=device)
    self._sums = torch.zeros(n, len(METRICS), device=device)
    self._steps = torch.zeros(n, device=device)
    self._moving_steps = torch.zeros(n, device=device)
    for name in METRICS:
      self.metrics[name] = torch.zeros(n, device=device)

  @property
  def command(self) -> torch.Tensor:
    return self._full if self.cfg.height_column else self._full[:, :3]

  def _goal(self):
    return self._env.command_manager.get_term(self.cfg.goal_name)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids                       # a pure function of the current goal state: all envs on every call
    goal = self._goal()
    ep, epsi, eh = goal.errors()
    self._full = command_from_goal(ep, epsi, eh, goal.mask, *goal_rates(goal))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids                       # nothing is drawn: the goal term owns the episode's randomness

  def _update_metrics(self) -> None:
    live = (self._env.episode_length_buf > 0).float()   # a just-reset env still holds the old episode's command
    data = self.robot.data
    v, w, hdot = data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2], data.root_link_lin_vel_w[:, 2]
    cmd = self._full
    speed = cmd[:, :2].norm(dim=1)
    moving = (speed > MOVING).float() * live
    along = (v * cmd[:, :2]).sum(dim=1) / speed.clamp_min(1.0e-9)
    fz = self._env.scene[self.cfg.feet_sensor].data.force.reshape(self.num_envs, -1, 3)[:, :, 2].abs()
    flight = (fz < FLIGHT_N).all(dim=1).float()
    self._steps += live
    self._moving_steps += moving
    s = self._sums
    for i, value in enumerate((speed * live, moving, speed * moving, along * moving, (cmd[:, :2] - v).norm(dim=1) * live,
                               (cmd[:, 2] - w).abs() * live, flight * live, flight * moving, cmd[:, 3].abs() * live,
                               (cmd[:, 3] - hdot).abs() * live)):
      s[:, i] += value
    steps, moving_steps = self._steps.clamp_min(1.0), self._moving_steps.clamp_min(1.0)
    for i, name in enumerate(METRICS):
      per_moving = name in ("cmd_speed_moving", "vel_along_moving", "flight_share_moving")
      self.metrics[name][:] = s[:, i] / (moving_steps if per_moving else steps)

  def reset(self, env_ids):
    extras = super().reset(env_ids)   # logs the finished episodes' means, zeroes self.metrics[env_ids]
    self._sums[env_ids] = 0.0
    self._steps[env_ids] = 0.0
    self._moving_steps[env_ids] = 0.0
    return extras


@dataclass(kw_only=True)
class GoalRateCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  goal_name: str = "goal"
  feet_sensor: str = "s1_feet"
  height_column: bool = False                                   # (A) twist [v_x, v_y, w_z]; (B) [.., hdot] too
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)   # never resampled: derived every step

  def build(self, env) -> GoalRateCommand:
    return GoalRateCommand(self, env)


# --- (A) ----------------------------------------------------------------------------------------------------------
def recipe(play: bool = False):
  """(rewards, sensors, curriculum) of careful_walk stage 1 with the walking run's recorded terms (RECORDED)."""
  import importlib
  from r1lab.tasks import careful_walk, common
  source = careful_walk.careful_walk_env_cfg(play=False, stage=1, t0=0)
  rewards = {}
  for name in TERMS:
    term = source.rewards[name]
    if name in RECORDED:
      module, function, weight, params = RECORDED[name]
      term.func, term.weight, term.params = getattr(importlib.import_module(module), function), weight, dict(params)
    rewards[name] = term
  if play:
    rewards.pop("style")              # evaluation layout: no discriminator (readouts drive the S0 environment anyway)
  dropped = {name: term.weight for name, term in source.rewards.items() if name not in TERMS}
  if any(weight != 0.0 for weight in dropped.values()):
    raise ValueError(f"careful_walk stage 1 has non-zero terms outside TERMS: {dropped}")
  named = {sensor.name: sensor for sensor in common.sensors(self_collision=True, scan=False)}
  sensors = tuple(named[name] for name in SENSORS)
  curriculum = {} if play else {"smoothness": source.curriculum["smoothness"]}
  return rewards, sensors, curriculum


def posctl_env_cfg(play: bool = False, **kw):
  """(A) wb_s1_s0b_power_fixed's environment with the reward block of the verified recipe (module docstring)."""
  from r1s1.tasks import _env_fixed
  cfg = _env_fixed(play=play, **kw)
  rewards, sensors, curriculum = recipe(play=play)
  cfg.rewards = rewards
  cfg.commands = {"goal": paced_goal_cfg(cfg.commands["goal"]), "twist": GoalRateCommandCfg()}
  cfg.scene.sensors = tuple(cfg.scene.sensors) + sensors
  cfg.curriculum = curriculum
  return cfg


# --- (B) ----------------------------------------------------------------------------------------------------------
def _goal_and_command(env, command_name):
  return env.command_manager.get_term("goal"), env.command_manager.get_command(command_name)


def s1_dense_planar(env, command_name: str = "pace", std: float = 0.15, kind: int = 1) -> torch.Tensor:
  """Locomotion goals: exp(-(|v_cmd - v_xy|^2 + v_z^2) / std^2), pelvis body frame (mjlab's planar kernel)."""
  goal, cmd = _goal_and_command(env, command_name)
  v = env.scene["robot"].data.root_link_lin_vel_b
  err = ((cmd[:, :2] - v[:, :2]) ** 2).sum(dim=1) + v[:, 2] ** 2
  return (goal.kind == kind).float() * torch.exp(-err / std ** 2)


def s1_dense_turn(env, command_name: str = "pace", std: float = 0.3, kind: int = 1) -> torch.Tensor:
  """Locomotion goals: exp(-((w_cmd - w_z)^2 + |w_xy|^2) / std^2), pelvis body frame (mjlab's yaw kernel)."""
  goal, cmd = _goal_and_command(env, command_name)
  w = env.scene["robot"].data.root_link_ang_vel_b
  err = (cmd[:, 2] - w[:, 2]) ** 2 + (w[:, :2] ** 2).sum(dim=1)
  return (goal.kind == kind).float() * torch.exp(-err / std ** 2)


def s1_dense_height(env, command_name: str = "pace", std: float = 0.05, kind: int = 2) -> torch.Tensor:
  """Crouch goals: exp(-(hdot_cmd - hdot)^2 / std^2), hdot = the pelvis vertical speed (world)."""
  goal, cmd = _goal_and_command(env, command_name)
  hdot = env.scene["robot"].data.root_link_lin_vel_w[:, 2]
  return (goal.kind == kind).float() * torch.exp(-((cmd[:, 3] - hdot) ** 2) / std ** 2)


def dense_env_cfg(play: bool = False, **kw):
  """(B) wb_s1_s0b_power_fixed's environment: objective v2 unchanged + one dense tracking term per moving goal kind."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from r1s1.tasks import _env_fixed
  cfg = _env_fixed(play=play, **kw)
  rewards = dict(cfg.rewards)
  functions = {"s1_dense_planar": s1_dense_planar, "s1_dense_turn": s1_dense_turn, "s1_dense_height": s1_dense_height}
  for name, (weight, std, kind) in DENSE.items():
    rewards[name] = RewardTermCfg(func=functions[name], weight=weight,
                                  params={"command_name": "pace", "std": std, "kind": kind})
  cfg.rewards = rewards
  cfg.commands = {"goal": paced_goal_cfg(cfg.commands["goal"]), "pace": GoalRateCommandCfg(height_column=True)}
  return cfg
