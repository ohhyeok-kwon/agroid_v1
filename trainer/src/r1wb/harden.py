"""P3 hardening, rounds 1-4: a correction on top of the frozen System 1 policy, and the disturbance generator.

Structure of rounds 1-4. The action is  a = S1-a(observation) + delta(observation).  S1-a is frozen and delta starts
at zero, so undisturbed behaviour is S1-a by construction and "how much did the robot deviate from what it intended"
is the size of delta. The environment adds the two (ResidualOnBaseAction), so the policy that PPO trains outputs
delta only and every price can be written on delta directly:
  - arms: the realised distance from the arm intent; waist: the correction, never relieved; legs: priced at half the
    waist's rate and relieved by urgency (privileged: r1wb.dynamics.urgency_margin, "one ordinary step is not
    enough", held 0.3 s, counted only shortly after an external impulse);
  - while the command is "stand" and nothing urgent is happening: joint speed and feet off the ground;
  - one per step for staying up and a large penalty for a fall (with costs alone a fall was the cheaper outcome).
What these rounds reached (same probe, 160 single hits of 12 and 25 N s / 80 bursts of 8 x 12 N s): falls 23 -> 11
and 80 -> 80. A correction explored with a noise of 0.1 and priced by its size does not find the fast, large leg
reaction a hard hit needs.

A fifth round gave the correction a hand-written hit detector that enlarged it for a fixed time. The user rejected
that on principle: nothing rule-based in the control path, and nothing taken from the manufacturer's policy; the
model has to learn the reaction itself through rewards and the training pipeline. That code is gone. The stage that
replaced it is r1wb.anchored (one network, continued from S1-a). This module keeps what both use:

Disturbances (DisturbedStreamCommand): horizontal impulses on the torso or pelvis at random times. A share is
ball-like (the whole impulse within one control step, `spike_share`), the rest is a 0.1 s push; a share of the
events are bursts, several hits from about the same side a fraction of a second apart. A share of the environments
is never disturbed, so calm behaviour keeps being measured and trained.

Unsettled (privileged bookkeeping for losses, rewards and metrics; nothing of it is deployed): set when an external
impulse acts on the robot, cleared when the body is steady again, i.e. when the planar velocity, the pelvis
roll/pitch rate and the pelvis posture have stayed near their own running averages for `settle_dwell`. It is a
statement about the state, not a timer: a small hit clears in a fraction of a second, a burst only after the robot
has stopped being carried. The posture just before the hit is kept as a reference while unsettled.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from r1wb import dynamics, robot
from r1wb.student.actions import IntentResidualAction, IntentResidualActionCfg
from r1wb.student.observations import load_teacher
from r1wb.student.stream import GoalStreamCommand, GoalStreamCommandCfg, stream_env_cfg
from r1wb.tracking.env_cfg import EXPERIMENT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LEGS, WAIST, ARMS = slice(0, 12), slice(12, 14), slice(14, 24)


# -- action of rounds 1-4: frozen base + correction ----------------------------------------------------------------

class ResidualOnBaseAction(IntentResidualAction):
  cfg: ResidualOnBaseActionCfg

  def __init__(self, cfg: ResidualOnBaseActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self.base = load_teacher(cfg.base_checkpoint, self.device)
    self.base_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self.residual = torch.zeros(self.num_envs, self.action_dim, device=self.device)

  def process_actions(self, actions: torch.Tensor):
    observation = self._env.obs_buf
    if all(group in observation for group in self.cfg.base_groups):
      joined = torch.cat([observation[group] for group in self.cfg.base_groups], dim=1)
      with torch.no_grad():
        self.base_action = self.base(TensorDict({"actor": joined}, batch_size=[joined.shape[0]]))
    self.residual = actions.clone()
    base = self.base_action
    if self.cfg.arm_base_scale != 1.0:
      # Only a share of the base policy's own arm output is passed on. Measured later with a zero correction: this
      # alone costs calm walking (commanded distance 0.92 -> 0.79, forward 0.2 m/s 0.92 -> 0.65), because that
      # output also holds the arms against gravity. Kept for the round-3/4 checkpoints only.
      base = base.clone()
      base[:, self.arm] = self.cfg.arm_base_scale * base[:, self.arm]
    super().process_actions(base + self.cfg.residual_scale * actions)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    index = slice(None) if env_ids is None else env_ids
    self.base_action[index] = 0.0
    self.residual[index] = 0.0


@dataclass(kw_only=True)
class ResidualOnBaseActionCfg(IntentResidualActionCfg):
  base_checkpoint: str = ""
  base_groups: tuple[str, ...] = ("actor", "goal")
  residual_scale: float = 1.0
  arm_base_scale: float = 1.0       # share of the base policy's arm output that is applied

  def build(self, env: ManagerBasedRlEnv) -> ResidualOnBaseAction:
    return ResidualOnBaseAction(self, env)


def applied_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The action that was applied (base + correction): what S1-a's "previous action" input means."""
  return env.action_manager.get_term("joint_pos").raw_action


# -- command stream with impulses, urgency and the unsettled state -------------------------------------------------

class DisturbedStreamCommand(GoalStreamCommand):
  cfg: DisturbedStreamCommandCfg

  def __init__(self, cfg: DisturbedStreamCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    count = self.num_envs
    self.calm = torch.rand(count, device=self.device) < cfg.calm_fraction
    # Gentle environments: single moderate hits far apart. Staying up is easy there, so the size of the reaction is
    # what their return differs by (in the others, bursts make every step a fight to stay up).
    self.gentle = ~self.calm & (torch.rand(count, device=self.device) < cfg.gentle_share)
    self.next_hit = torch.full((count,), 1.0e9, device=self.device)
    self.hit_left = torch.zeros(count, dtype=torch.long, device=self.device)
    self.hit_force = torch.zeros(count, 3, device=self.device)
    self.hit_torque = torch.zeros(count, 3, device=self.device)
    self.hit_body = torch.zeros(count, dtype=torch.long, device=self.device)
    self.since_hit = torch.full((count,), 1.0e3, device=self.device)
    self.burst_left = torch.zeros(count, dtype=torch.long, device=self.device)   # further hits of the running burst
    self.burst_hard = torch.zeros(count, dtype=torch.bool, device=self.device)   # the running burst is a dense one
    self.burst_angle = torch.zeros(count, device=self.device)
    self.last_impulse = torch.zeros(count, device=self.device)
    self.hit_impulse_w = torch.zeros(count, 2, device=self.device)   # the last hit as a planar vector (N s, world)
    self.hit_count = torch.zeros(count, dtype=torch.long, device=self.device)   # hits since the running reaction began
    self.first_impulse = torch.zeros(count, device=self.device)      # size of the first of them (N s)
    self.urgency = torch.zeros(count, device=self.device)        # held, 0..1
    self.urgency_raw = torch.zeros(count, device=self.device)
    self.gate = torch.zeros(count, device=self.device)           # urgency that counts as relief (rounds 1-4)
    self.impulse_scale = 1.0                                      # share of the impulse range in use (see impulse_ramp_steps)
    self.velocity_mean = torch.zeros(count, 2, device=self.device)   # own planar CoM velocity, about 1 s average
    self._mean_valid = torch.zeros(count, dtype=torch.bool, device=self.device)
    # Unsettled bookkeeping.
    self.unsettled = torch.zeros(count, device=self.device)      # 1 from an external impulse until the body is steady
    self.unsettled_time = torch.zeros(count, device=self.device)  # seconds of the running unsettled spell
    self.steady_for = torch.zeros(count, device=self.device)     # seconds the steadiness test has held
    self.gravity_mean = torch.zeros(count, 3, device=self.device)  # running average of gravity in the pelvis frame
    self.gravity_mean[:, 2] = -1.0
    self.posture_before = self.gravity_mean.clone()              # the same, frozen while unsettled
    self.swing = torch.zeros(count, device=self.device)          # rad away from the posture before the hit (unsettled only)
    # Size of the reaction: measured from the impulse until `reaction_tail` after the body is steady again, against
    # the pelvis posture and the joint angles just before the hit (their 0.5 s averages, frozen for that time).
    joints = self.robot.data.joint_pos.shape[1]
    self.reacting = torch.zeros(count, device=self.device)
    self.settled_for = torch.full((count,), 1.0e3, device=self.device)    # s since the last unsettled spell ended
    self.joint_mean = torch.zeros(count, joints, device=self.device)
    self.joints_before = torch.zeros(count, joints, device=self.device)
    self.tilt_dev = torch.zeros(count, device=self.device)                # rad, pelvis posture (0 when not reacting)
    self.joint_dev = torch.zeros(count, joints, device=self.device)       # rad per joint (0 when not reacting)
    self.action_mean = torch.zeros(count, joints, device=self.device)     # running average of the applied action (0.5 s)
    self.action_before = torch.zeros(count, joints, device=self.device)   # the same, frozen while reacting
    self._peak = torch.zeros(count, 2, device=self.device)                # running peaks: tilt, mean waist+arm deviation
    self._last_peak = torch.zeros(count, 2, device=self.device)           # the same of the last finished reaction
    self.unsteadiness = torch.zeros(count, 3, device=self.device)  # velocity (m/s), rate (rad/s), posture (rad) deviations
    self._bodies = [self.robot.body_names.index(name) for name in cfg.hit_bodies]
    self._centroidal = None
    self._hits_per_step = int(round(cfg.hit_duration / env.step_dt))
    for key in ("urgency", "gate", "impulse", "calm_urgent", "calm_residual", "calm_goal_lag", "hit_residual",
                "unsettled", "calm_unsettled", "settle_time", "gentle_settle_time", "gentle_peak_tilt",
                "gentle_peak_upper"):
      self.metrics[key] = torch.zeros(count, device=self.device)
    self._gentle_scale = count / self.gentle.float().sum().clamp_min(1.0)   # so that the logged mean is the gentle mean
    self._calm_scale = count / self.calm.float().sum().clamp_min(1.0)       # so that the logged mean is the calm mean
    self._hit_scale = count / (~self.calm).float().sum().clamp_min(1.0)
    self._settle_time = torch.zeros(count, device=self.device)   # length of each environment's last finished spell

  def _schedule(self, ids: torch.Tensor) -> None:
    cfg, count = self.cfg, len(ids)
    # Inside a burst the next hit follows after a short gap; otherwise the next event is seconds away.
    gap = torch.where(self.burst_hard[ids], self._uniform(count, *cfg.hard_gap), self._uniform(count, *cfg.burst_gap))
    wait = torch.where(self.burst_left[ids] > 0, gap, self._uniform(count, *cfg.hit_interval))
    wait = torch.where(self.gentle[ids], self._uniform(count, *cfg.gentle_interval), wait)
    self.next_hit[ids] = torch.where(self.calm[ids], torch.full_like(wait, 1.0e9), wait)

  def _start_hits(self, ids: torch.Tensor) -> None:
    cfg, count = self.cfg, len(ids)
    low, high = cfg.impulse_range
    impulse = low + (high - low) * self.impulse_scale * torch.rand(count, device=self.device)
    angle = self._uniform(count, -math.pi, math.pi)
    if cfg.focus_share > 0.0:
      # A share of the impulses is drawn where the falls are: strong pushes toward the robot's back.
      focus = torch.rand(count, device=self.device) < cfg.focus_share
      w, x, y, z = self.robot.data.root_link_quat_w[ids].unbind(-1)
      heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
      backward = heading + math.pi + self._uniform(count, -cfg.focus_half_angle, cfg.focus_half_angle)
      angle = torch.where(focus, backward, angle)
      impulse = torch.where(focus, self._uniform(count, *cfg.focus_impulse), impulse)
    # Bursts: several ball-sized hits from about the same side, a fraction of a second apart.
    gentle = self.gentle[ids]
    follow = self.burst_left[ids] > 0                                                    # a further hit of a running burst
    start = ~follow & ~gentle & (torch.rand(count, device=self.device) < cfg.burst_prob)  # this event opens a burst
    burst_low, burst_high = cfg.burst_impulse
    impulse = torch.where(follow | start, self._uniform(count, burst_low, burst_low + (burst_high - burst_low) * self.impulse_scale),
                          impulse)
    # Optional share of long, dense bursts (off by default: drawn at random, one burst in thirty is as long, as
    # dense and as strong as the eight-ball test, and a held-out probe has to exist before training on its shape).
    hard_start = start & (torch.rand(count, device=self.device) < cfg.burst_hard_share)
    self.burst_hard[ids] = torch.where(start, hard_start, self.burst_hard[ids] & follow)
    hard = self.burst_hard[ids]
    impulse = torch.where(hard, self._uniform(count, *cfg.hard_impulse) * (0.5 + 0.5 * self.impulse_scale), impulse)
    impulse = torch.where(gentle, self._uniform(count, *cfg.gentle_impulse), impulse)
    angle = torch.where(follow, self.burst_angle[ids] + self._uniform(count, -0.3, 0.3), angle)
    self.burst_angle[ids] = angle
    further = torch.randint(max(1, cfg.burst_min - 1), max(cfg.burst_min, cfg.burst_max), (count,), device=self.device)
    further = torch.where(hard_start, torch.randint(cfg.hard_hits[0] - 1, cfg.hard_hits[1], (count,), device=self.device), further)
    self.burst_left[ids] = torch.where(follow, self.burst_left[ids] - 1,
                                       torch.where(start, further, torch.zeros_like(further)))
    if cfg.hit_body_weights is None:
      body = torch.randint(len(self._bodies), (count,), device=self.device)
    else:
      weights = torch.tensor(cfg.hit_body_weights, device=self.device)
      body = torch.multinomial(weights / weights.sum(), count, replacement=True)
    if cfg.hit_body_scale is not None:
      # A ball passes on less momentum to a light limb than to the trunk (and a trunk-sized impulse on a forearm
      # within one control step makes the simulation diverge: the first "anywhere" run ended in NaN).
      impulse = impulse * torch.tensor(cfg.hit_body_scale, device=self.device)[body]
    direction = torch.stack([angle.cos(), angle.sin(), torch.zeros(count, device=self.device)], dim=1)
    # A ball delivers its impulse within one control step; a shove takes `hit_duration`.
    spike = torch.rand(count, device=self.device) < cfg.spike_share
    steps = torch.where(spike, torch.ones_like(self.hit_left[ids]), torch.full_like(self.hit_left[ids], self._hits_per_step))
    force = direction * (impulse / (steps * self._env.step_dt)).unsqueeze(1)
    arm = torch.zeros(count, 3, device=self.device)
    arm[:, 2] = self._uniform(count, *cfg.hit_height)
    if cfg.hit_body_scale is not None:
      # The lever arm above the body origin is meant for the trunk; a limb is hit at the limb.
      limb = torch.tensor(cfg.hit_body_scale, device=self.device)[body] < 1.0
      arm[:, 2] = torch.where(limb, self._uniform(count, -0.05, 0.05), arm[:, 2])
    self.hit_force[ids], self.hit_torque[ids] = force, torch.cross(arm, force, dim=1)
    self.hit_body[ids] = body
    self.hit_impulse_w[ids] = direction[:, :2] * impulse.unsqueeze(1)
    self.hit_count[ids] += 1
    self.first_impulse[ids] = torch.where(self.hit_count[ids] == 1, impulse, self.first_impulse[ids])
    self.hit_left[ids] = steps
    self.since_hit[ids] = 0.0
    self.last_impulse[ids] = impulse
    self._schedule(ids)

  def _apply_wrench(self) -> torch.Tensor:
    """Writes the wrench of the next physics step; returns which environments it acts on."""
    acting = self.hit_left > 0
    active = acting.float().unsqueeze(1)
    forces = torch.zeros(self.num_envs, len(self._bodies), 3, device=self.device)
    torques = torch.zeros_like(forces)
    slot = torch.nn.functional.one_hot(self.hit_body, len(self._bodies)).float().unsqueeze(-1)
    forces += slot * (self.hit_force * active).unsqueeze(1)
    torques += slot * (self.hit_torque * active).unsqueeze(1)
    self.robot.write_external_wrench_to_sim(forces, torques, body_ids=self._bodies)
    self.hit_left = (self.hit_left - 1).clamp_min(0)
    return acting

  def _update_urgency(self) -> dict:
    if self._centroidal is None:
      self._centroidal = dynamics.Centroidal(self._env)
    state = self._centroidal.state()
    velocity = state["velocity"][:, :2]
    # "No divergence" = the robot's own recent mean velocity (see dynamics.urgency_margin), not the command.
    blend = self._env.step_dt / self.cfg.velocity_mean_time
    self.velocity_mean = torch.where(self._mean_valid.unsqueeze(1), self.velocity_mean + (velocity - self.velocity_mean) * blend,
                                     velocity)
    self.urgency_raw = self._centroidal.urgency_margin(self.velocity_mean, state=state)["urgency"]
    decay = math.exp(-self._env.step_dt / self.cfg.urgency_hold)
    self.urgency = torch.maximum(self.urgency_raw, self.urgency * decay)
    self.gate = self.urgency * (self.since_hit < self.cfg.relief_window).float()
    return state

  def _update_settling(self, state: dict, acting: torch.Tensor) -> None:
    """Unsettled: set by an external impulse, cleared when the body has been steady for `settle_dwell`."""
    cfg, data, dt = self.cfg, self.robot.data, self._env.step_dt
    gravity = data.projected_gravity_b
    fresh = ~self._mean_valid
    self.gravity_mean = torch.where(fresh.unsqueeze(1), gravity, self.gravity_mean + (gravity - self.gravity_mean) * (dt / cfg.posture_time))
    self.posture_before = torch.where(fresh.unsqueeze(1), gravity, self.posture_before)
    joints = data.joint_pos
    self.joint_mean = torch.where(fresh.unsqueeze(1), joints, self.joint_mean + (joints - self.joint_mean) * (dt / cfg.posture_time))
    self.joints_before = torch.where(fresh.unsqueeze(1), joints, self.joints_before)
    velocity = torch.norm(state["velocity"][:, :2] - self.velocity_mean, dim=1)
    rate = torch.norm(data.root_link_ang_vel_b[:, :2], dim=1)
    posture = _angle(gravity, self.gravity_mean)
    self.unsteadiness = torch.stack([velocity, rate, posture], dim=1)
    steady = (velocity < cfg.settle_velocity) & (rate < cfg.settle_rate) & (posture < cfg.settle_posture) & ~acting
    self.steady_for = torch.where(steady, self.steady_for + dt, torch.zeros_like(self.steady_for))
    was = self.unsettled > 0
    now = acting | (was & (self.steady_for < cfg.settle_dwell))
    finished = was & ~now
    self._settle_time = torch.where(finished, self.unsettled_time, self._settle_time)
    self.unsettled_time = torch.where(now, self.unsettled_time + dt, torch.zeros_like(self.unsettled_time))
    self.unsettled = now.float()
    # The posture before the hit: the running average, frozen from the moment an impulse acts until the reaction is
    # over (the unsettled spell and `reaction_tail` after it; with a tail of 0 that is the spell itself).
    self.settled_for = torch.where(now, torch.zeros_like(self.settled_for), self.settled_for + dt)
    was_reacting = self.reacting > 0
    reacting = now | (self.settled_for < cfg.reaction_tail)
    self.reacting = reacting.float()
    self.hit_count = torch.where(reacting, self.hit_count, torch.zeros_like(self.hit_count))
    self.posture_before = torch.where(reacting.unsqueeze(1), self.posture_before, self.gravity_mean)
    self.joints_before = torch.where(reacting.unsqueeze(1), self.joints_before, self.joint_mean)
    applied = self._env.action_manager.get_term("joint_pos").raw_action
    if applied.shape[1] == self.action_mean.shape[1]:
      self.action_mean = torch.where(fresh.unsqueeze(1), applied, self.action_mean + (applied - self.action_mean) * (dt / cfg.posture_time))
      self.action_before = torch.where(reacting.unsqueeze(1) & ~fresh.unsqueeze(1), self.action_before, self.action_mean)
    self.swing = _angle(gravity, self.posture_before) * self.unsettled
    self.tilt_dev = _angle(gravity, self.posture_before) * self.reacting
    self.joint_dev = (joints - self.joints_before).abs() * self.reacting.unsqueeze(1)
    peak = torch.stack([self.tilt_dev, self.joint_dev[:, 12:].mean(dim=1)], dim=1)
    self._peak = torch.where(reacting.unsqueeze(1), torch.maximum(self._peak, peak), self._peak)
    over = (was_reacting & ~reacting).unsqueeze(1)
    self._last_peak = torch.where(over, self._peak, self._last_peak)
    self._peak = torch.where(reacting.unsqueeze(1), self._peak, torch.zeros_like(self._peak))

  def _update_command(self, env_ids: torch.Tensor | None = None):
    super()._update_command(env_ids)
    if env_ids is not None:
      return
    dt = self._env.step_dt
    if self.cfg.impulse_ramp_steps > 0:
      self.impulse_scale = min(1.0, 0.2 + 0.8 * self._env.common_step_counter / self.cfg.impulse_ramp_steps)
    self.next_hit -= dt
    self.since_hit += dt
    due = (self.next_hit <= 0.0).nonzero(as_tuple=False).flatten()
    if len(due):
      self._start_hits(due)
    acting = self._apply_wrench()
    state = self._update_urgency()
    self._update_settling(state, acting)
    self._mean_valid[:] = True
    self.metrics["urgency"][:] = self.urgency
    self.metrics["gate"][:] = self.gate
    self.metrics["impulse"][:] = self.last_impulse
    # Undisturbed environments only (the stop rules read these): how often urgency fires without any impulse, how
    # large the correction is in calm motion, and how well the commands are followed.
    calm = self.calm.float() * self._calm_scale
    self.metrics["calm_urgent"][:] = (self.urgency_raw > 0).float() * calm
    self.metrics["calm_goal_lag"][:] = torch.norm(self.goal_lag[:, :2], dim=1) * calm
    self.metrics["unsettled"][:] = self.unsettled
    self.metrics["calm_unsettled"][:] = self.unsettled * calm
    self.metrics["settle_time"][:] = self._settle_time
    gentle = self.gentle.float() * self._gentle_scale
    self.metrics["gentle_settle_time"][:] = self._settle_time * gentle
    self.metrics["gentle_peak_tilt"][:] = self._last_peak[:, 0] * gentle
    self.metrics["gentle_peak_upper"][:] = self._last_peak[:, 1] * gentle
    action = self._env.action_manager.get_term("joint_pos")
    if hasattr(action, "residual"):
      residual = action.residual.abs().mean(dim=1)
      self.metrics["calm_residual"][:] = residual * calm
      self.metrics["hit_residual"][:] = residual * (~self.calm).float() * self._hit_scale

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self.hit_left[env_ids] = 0
    self.burst_left[env_ids] = 0
    self.burst_hard[env_ids] = False
    self.since_hit[env_ids] = 1.0e3
    self.last_impulse[env_ids] = 0.0
    self.hit_impulse_w[env_ids] = 0.0
    self.hit_count[env_ids] = 0
    self.first_impulse[env_ids] = 0.0
    self.urgency[env_ids] = 0.0
    self.gate[env_ids] = 0.0
    self.unsettled[env_ids] = 0.0
    self.unsettled_time[env_ids] = 0.0
    self.steady_for[env_ids] = 0.0
    self.swing[env_ids] = 0.0
    self.reacting[env_ids] = 0.0
    self.settled_for[env_ids] = 1.0e3
    self.tilt_dev[env_ids] = 0.0
    self.joint_dev[env_ids] = 0.0
    self._peak[env_ids] = 0.0
    self._mean_valid[env_ids] = False
    self._schedule(env_ids)


def _angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  cosine = (a * b).sum(dim=1) / (torch.norm(a, dim=1) * torch.norm(b, dim=1)).clamp_min(1e-6)
  return torch.acos(cosine.clamp(-1.0, 1.0))


@dataclass(kw_only=True)
class DisturbedStreamCommandCfg(GoalStreamCommandCfg):
  calm_fraction: float = 0.3
  impulse_range: tuple[float, float] = (5.0, 30.0)      # N s (the tests use 12 and 25)
  impulse_ramp_steps: int = 0                            # environment steps over which the range opens (0 = full from the start)
  hit_interval: tuple[float, float] = (3.0, 8.0)        # s between impulses
  hit_duration: float = 0.1                              # s over which a push is applied
  spike_share: float = 0.0                               # share of hits delivered within one control step (ball-like)
  hit_height: tuple[float, float] = (-0.1, 0.25)        # m above the body origin (lever arm of the push)
  hit_bodies: tuple[str, ...] = ("torso_link", "pelvis")
  hit_body_weights: tuple[float, ...] | None = None      # sampling weights of hit_bodies (None: uniform)
  hit_body_scale: tuple[float, ...] | None = None        # share of the drawn impulse that reaches each hit body
  burst_prob: float = 0.0                                # share of impulse events that are bursts
  burst_min: int = 2                                     # hits per burst: burst_min .. burst_max
  burst_max: int = 8
  burst_gap: tuple[float, float] = (0.1, 0.4)            # s between the hits of a burst
  burst_impulse: tuple[float, float] = (6.0, 15.0)       # N s per hit of a burst
  burst_hard_share: float = 0.0                          # share of bursts that are long and dense:
  hard_hits: tuple[int, int] = (6, 9)                    #   hits per burst (inclusive)
  hard_gap: tuple[float, float] = (0.12, 0.22)           #   s between them
  hard_impulse: tuple[float, float] = (10.0, 14.0)       #   N s per hit
  focus_share: float = 0.0                               # share of impulses drawn from the focus set below
  focus_half_angle: float = 0.6                          # rad around "pushed straight backward" (body frame)
  focus_impulse: tuple[float, float] = (18.0, 32.0)      # N s
  urgency_hold: float = 0.3                              # s
  velocity_mean_time: float = 1.0                        # s, averaging time of the own-velocity reference
  relief_window: float = 2.0                             # s after an impulse during which urgency relieves prices
  # Steadiness test of the unsettled state: deviations from the body's own running averages. The limits sit above
  # what S1-a shows in undisturbed command streams under training conditions (measured before use, see the ledger).
  posture_time: float = 0.5                              # s, averaging time of the posture reference
  settle_velocity: float = 0.25                          # m/s, planar CoM velocity against its 1 s average
  settle_rate: float = 1.5                               # rad/s, pelvis roll/pitch rate
  settle_posture: float = 0.12                           # rad, pelvis posture against its 0.5 s average
  settle_dwell: float = 0.3                              # s the test has to hold
  reaction_tail: float = 0.0                             # s after settling during which the reaction is still measured
  gentle_share: float = 0.0                              # share of the disturbed environments with single moderate hits only
  gentle_impulse: tuple[float, float] = (5.0, 15.0)      #   N s
  gentle_interval: tuple[float, float] = (3.0, 6.0)      #   s between them

  def build(self, env: ManagerBasedRlEnv) -> DisturbedStreamCommand:
    return DisturbedStreamCommand(self, env)


# -- rewards and observations --------------------------------------------------------------------------------------

def _term(env):
  return env.command_manager.get_term("motion")


def residual_l1(env: ManagerBasedRlEnv, group: str, relief: float = 0.0) -> torch.Tensor:
  """Mean |correction| of a joint group (action units); `relief` is the share removed at full gated urgency."""
  part = {"legs": LEGS, "waist": WAIST, "arms": ARMS}[group]
  size = env.action_manager.get_term("joint_pos").residual[:, part].abs().mean(dim=1)
  return size * (1.0 - relief * _term(env).gate)


def standing_quiet(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground", lift_cost: float = 5.0) -> torch.Tensor:
  """While asked to stand and nothing is urgent: mean joint speed (rad/s) plus `lift_cost` per foot off the ground."""
  term = _term(env)
  standing = ((torch.norm(term.stream_target[:, :3], dim=1) < 1e-6) & (term.since_change > 1.5)).float()
  speed = env.scene["robot"].data.joint_vel.abs().mean(dim=1)
  airborne = (env.scene[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] <= 0).float().sum(dim=1)
  return standing * (1.0 - term.gate) * (speed + lift_cost * airborne).clamp(max=5.0)


def urgency_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _term(env).urgency


def fell(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One per step that did not end in a fall. Without it every term of this stage is a cost, and in a hard recovery
  (costs of 2-3 per step) ending the episode by falling was the cheaper outcome."""
  return 1.0 - env.termination_manager.terminated.float()


def lag_cost(env: ManagerBasedRlEnv, limit: float = 1.0, yaw_weight: float = 0.3) -> torch.Tensor:
  """Lag behind the commanded twist, bounded: being carried a metre off by a push must not cost more than falling."""
  lag = _term(env).goal_lag
  return (torch.norm(lag[:, :2], dim=1) + yaw_weight * lag[:, 2].abs()).clamp(max=limit)


def urgency_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  return torch.stack([term.urgency, term.gate, (term.since_hit < 2.0).float(), term.last_impulse * 0.04], dim=1)


def residual_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.get_term("joint_pos").residual


# -- configuration of rounds 1-4 -----------------------------------------------------------------------------------

def harden_env_cfg(play: bool = False, teacher: str | None = None, clean: bool | None = None,
                   subset: str = "core",
                   base_shares: tuple[float, ...] = (0.3, 0.5, 0.0, 0.0, 0.2),
                   relief: bool = True, arm_base_scale: float = 1.0,
                   focus_share: float = 0.0, burst_prob: float = 0.0, spike_share: float = 0.0) -> ManagerBasedRlEnvCfg:
  """`teacher` = the frozen S1-a checkpoint the correction is added to."""
  if teacher is None:
    raise ValueError("The hardening task needs the frozen base policy (--teacher <S1-a model_N.pt>)")
  cfg = stream_env_cfg(play=play, clean=clean, subset=subset)
  cfg.commands = {"motion": disturbed_command_cfg(cfg.commands["motion"], base_shares)}
  if not relief:
    cfg.commands["motion"].relief_window = 0.0      # control run: urgency never relieves a price (gate stays 0)
  cfg.commands["motion"].focus_share = focus_share
  cfg.commands["motion"].burst_prob = burst_prob
  cfg.commands["motion"].spike_share = spike_share
  cfg.actions = {"joint_pos": ResidualOnBaseActionCfg(entity_name="robot", actuator_names=(".*",),
                                                      scale=dict(robot.ACTION_SCALE), use_default_offset=True,
                                                      base_checkpoint=str(teacher), arm_base_scale=arm_base_scale)}
  # S1-a's "previous action" input is the action that was applied, not the correction alone.
  cfg.observations["actor"].terms["actions"] = ObservationTermCfg(func=applied_action)
  critic = cfg.observations["critic"].terms
  critic["actions"] = ObservationTermCfg(func=applied_action)
  critic["urgency"] = ObservationTermCfg(func=urgency_state)
  critic["residual"] = ObservationTermCfg(func=residual_state)
  cfg.events.pop("push_robot", None)               # impulses come from the command term
  cfg.rewards["arm_deviation"].weight = -3.0
  # The correction is not paid for following commands. In the first run it was (the student's goal kernels were
  # kept), and after 250 iterations it had become a standing offset: mean |correction| 0.06-0.07 on the legs in
  # every script, standing included, because shifting the posture earned a little more height and twist reward
  # than the leg price cost. Only a bounded linear lag term stays, so that walking on after a hit is still preferred.
  # Staying up is rewarded in itself (alive) so that no sum of costs can make a fall the cheaper outcome.
  for term in ("goal_twist", "goal_twist_l1", "goal_height", "goal_pitch", "goal_lag"):
    cfg.rewards.pop(term, None)
  cfg.rewards.update({
    "residual_legs": RewardTermCfg(func=residual_l1, weight=-1.0, params={"group": "legs", "relief": 1.0}),
    "residual_waist": RewardTermCfg(func=residual_l1, weight=-2.0, params={"group": "waist"}),
    "residual_arms": RewardTermCfg(func=residual_l1, weight=-0.2, params={"group": "arms"}),
    "standing_quiet": RewardTermCfg(func=standing_quiet, weight=-0.2),
    "urgency": RewardTermCfg(func=urgency_cost, weight=-0.5),
    "lag": RewardTermCfg(func=lag_cost, weight=-1.0),
    "alive": RewardTermCfg(func=alive, weight=5.0),
    "fell": RewardTermCfg(func=fell, weight=-200.0),
  })
  return cfg


def disturbed_command_cfg(old, shares: tuple[float, ...] | None = (0.3, 0.5, 0.0, 0.0, 0.2)) -> DisturbedStreamCommandCfg:
  """The stream command with impulses. With `shares` (rounds 1-4): only commands S1-a follows, at sizes above its
  measured command floor (E9 at 2,500: forward from 0.2 m/s, with or without a curve; turning in place from 0.6
  rad/s; standing; lowered height), no pitch goal; a correction must not be paid to do locomotion the base ignores.
  With shares=None the stream keeps its own full command distribution (every kind, pitch goals included)."""
  values = {field.name: getattr(old, field.name) for field in fields(old)}
  if shares is not None:
    values.update(segment_shares=shares, pitch_prob=0.0, min_forward=0.2, max_forward=0.5, min_yaw_rate=0.6,
                  max_yaw_rate=1.0)
  return DisturbedStreamCommandCfg(**values)


def harden_rl_cfg(state_dependent_noise: bool = False) -> RslRlOnPolicyRunnerCfg:
  """state_dependent_noise: the exploration noise is an output of the network instead of one learned number per
  joint. With one number the price on the correction pushed it down everywhere (0.10 -> 0.08 in 500 iterations),
  which also removed the exploration in the rare moments where a large recovery step has to be discovered."""
  distribution = ({"class_name": "HeteroscedasticGaussianDistribution", "init_std": 0.1, "std_type": "log",
                   "std_range": (0.02, 0.6)} if state_dependent_noise else
                  {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"})
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg=distribution),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.001,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    obs_groups={"actor": ("actor", "goal"), "critic": ("critic",)},
    experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=2000,
    logger="tensorboard", upload_model=False)
