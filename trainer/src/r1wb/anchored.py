"""P3, sixth round: one network that learns its own reaction to hits (r1wb.harden holds the disturbance generator).

What is deployed: a single policy with S1-a's inputs and outputs. No base-plus-correction sum, no detector, no
switch: whatever the robot does after a hit is what this network learned to do from its own observation history.

How it is trained (all of this exists during training only):
  - The policy starts as S1-a (same weights, same observation normaliser, which stays frozen) and is trained on
    with PPO on command streams with ball-like hits, pushes and bursts.
  - A frozen copy of S1-a is the anchor. The loss adds  coefficient x weight x |policy mean - anchor action|^2  with
    weight = 1 - unsettled. While the robot is steady the policy is held to S1-a (calm behaviour stays the human-like
    one, sample by sample); from the moment an external impulse acts until the body is steady again the anchor lets
    go and only the rewards speak. "Unsettled" is privileged bookkeeping of the simulator (r1wb.harden): the
    learning algorithm reads it, the policy does not.
  - Rewards. While settled: the command-following terms S1-a was trained with. While unsettled those are switched
    off and only staying up and recovering count: each step is a cost (recover fast), the pelvis turning away from
    its posture before the hit and its roll/pitch rate are priced (the fall in the making). Always: a pay for every
    step that stays up, a large penalty for a fall, and the arms' distance from their intent at three times the
    student-stage price (react with what is needed). The costs that can act while unsettled are bounded and sum to
    less than the pay for staying up: in the first run the unbounded command-following penalties stayed on, a hard
    recovery cost more per step than it earned, and the policy learned to fall sooner (single-hit falls 16 -> 40 of
    160 between iterations 250 and 500, while the anchored calm behaviour stayed within 0.02 action units of S1-a).
  - Exploration is the policy's own learned noise (S1-a ends its training at 0.23-0.49 action units per joint), one
    number per joint. Variant `learned`: the network outputs the noise size itself, per joint and per state, starting
    at S1-a's values, so it can explore more where that pays and less where it costs.
  - Commands: the stream's full distribution (stand, forward, backward, lateral, turn, lowered height, pitch). The
    anchor only holds what the training visits: in a first run restricted to stand / forward / turn, walking backward
    at 0.2 m/s fell from 0.72 to 0.52 of the commanded distance within 250 iterations.
Tried and dropped: enlarging the noise threefold while unsettled (a schedule driven by the privileged state). The
robot then stays unsettled because of its own noise, and after 250 iterations the deterministic policy fell in 78 of
160 single hits (S1-a: 23).
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from r1wb.harden import alive, disturbed_command_cfg, fell, urgency_cost, urgency_state
from r1wb.student.env_cfg import ImitationPpoCfg
from r1wb.student.stream import stream_env_cfg
from r1wb.tracking.env_cfg import EXPERIMENT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _term(env):
  return env.command_manager.get_term("motion")


def arm_course(env: ManagerBasedRlEnv, mode: str = "gentle", limit: float = 15.0) -> torch.Tensor:
  """For the learning algorithm (not an input of the policy): [hold, the ten arm outputs of before the hit (their
  0.5 s average, frozen at the hit), repeat]. With it the anchor says for the arms what "undisturbed" means during a
  hit. See AnchoredPPO._labels.
  mode "gentle": hold = a gentle environment is reacting (E51-E57).
  mode "first_hit": hold = the running reaction has seen one hit so far and that hit was at most `limit` N s, in any
  environment; repeat = a further hit has landed in the same reaction. Measured: keeping the arms on their course
  for the first hit of a burst costs about 1 N s per hit of what the robot withstands, keeping them there while the
  hits go on costs 3-5; and a hold learned only on single hits spreads into the repeated-hit states (arm targets
  0.06-0.12 rad off the twin's during hits 2-5 instead of 0.12-0.24), where nothing said otherwise."""
  term = _term(env)
  if mode == "first_hit":
    hold = term.reacting * (term.hit_count == 1).float() * (term.first_impulse <= limit).float()
    repeat = term.reacting * (term.hit_count >= 2).float()
  else:
    hold, repeat = term.reacting * term.gentle.float(), torch.zeros_like(term.reacting)
  return torch.cat([hold.unsqueeze(1), term.action_before[:, 14:], repeat.unsqueeze(1)], dim=1)


def anchor_signal(env: ManagerBasedRlEnv, domain: str = "settled", release_lateral: bool = False) -> torch.Tensor:
  """For the learning algorithm: [anchor weight, unsettled]. Not an input of the policy.
  domain "settled": the anchor holds every sample outside an unsettled spell. "calm": it holds only the environments
  that are never hit. The difference is the seconds after a hit: once the body is steady the first form ties the
  policy to S1-a again, and what S1-a does there is its own, large reaction."""
  term = _term(env)
  # Fourth column: the environment's class (0 never hit, 1 gentle, 2 the rest), for per-class advantage scaling.
  kind = torch.where(term.calm, 0.0, torch.where(term.gentle, 1.0, 2.0))
  zero = torch.zeros_like(kind)
  # release_lateral: no anchor while a sideways speed is commanded. S1-a cannot do it (1.2 s of sideways walking in
  # the motion library: left 0.06 m/s for 0.2 commanded, right none), so there the anchor would hold a failure.
  keep = (term.twist_goal()[:, 1].abs() < 0.03).float() if release_lateral else 1.0
  if domain == "not_reacting":
    # As "settled", but the anchor stays off for the whole reaction (the unsettled spell and the tail after it).
    return torch.stack([(1.0 - term.reacting) * keep, term.unsettled, zero, kind], dim=1)
  if domain == "nominal":
    # The anchor holds the never-hit environments, and those run without exploration noise (third column, read by
    # AnchoredPPO): what is tied to S1-a is the motion itself, not the response to being shaken. With noise the
    # anchored samples cover the policy's reflexes too (the body is shaken by its own noise all the time), and the
    # reaction to a moderate hit could not change ("calm": unchanged after 400 updates; no anchor: halved).
    calm = term.calm.float()
    return torch.stack([calm * keep, term.unsettled, calm * keep, kind], dim=1)
  weight = term.calm.float() if domain == "calm" else 1.0 - term.unsettled
  return torch.stack([weight * keep, term.unsettled, zero, kind], dim=1)


def settling_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """For the critic: unsettled, seconds of the running spell (scaled), posture swing, and the three deviations."""
  term = _term(env)
  return torch.cat([term.unsettled.unsqueeze(1), (term.unsettled_time * 0.5).clamp(max=2.0).unsqueeze(1),
                    term.swing.unsqueeze(1), term.unsteadiness], dim=1)


def while_settled(func):
  """The same reward term, counted only while the robot is settled."""
  @functools.wraps(func)
  def term(env, *args, **kwargs):
    return func(env, *args, **kwargs) * (1.0 - _term(env).unsettled)
  return term


def while_not_reacting(func):
  """The same reward term, counted only outside a reaction (the unsettled spell and the tail after it)."""
  @functools.wraps(func)
  def term(env, *args, **kwargs):
    return func(env, *args, **kwargs) * (1.0 - _term(env).reacting)
  return term


def in_gentle(func):
  """The same reward term, counted only in the gentle environments (single moderate hits)."""
  @functools.wraps(func)
  def term(env, *args, **kwargs):
    return func(env, *args, **kwargs) * _term(env).gentle.float()
  return term


def unsettled_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One per step from an external impulse until the body is steady again: the time a recovery takes."""
  return _term(env).unsettled


def swing_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """While unsettled: squared angle (rad^2) between the pelvis posture and the posture just before the hit. Bounded,
  so that no sum of costs exceeds the pay for staying up. Zero when settled: intended leaning is never priced."""
  return _term(env).swing.square().clamp(max=0.5)


def spin_cost(env: ManagerBasedRlEnv, limit: float = 25.0) -> torch.Tensor:
  """While unsettled: squared roll/pitch rate of the pelvis (rad^2/s^2), bounded."""
  rate = env.scene["robot"].data.root_link_ang_vel_b[:, :2].square().sum(dim=1).clamp(max=limit)
  return rate * _term(env).unsettled


def reaction_cost(env: ManagerBasedRlEnv, limit: float = 2.0) -> torch.Tensor:
  """While unsettled: mean speed of the waist and arm joints (rad/s), bounded. What a recovery needs is done by the
  legs; the waist and the arms should move only as far as that pays."""
  speed = env.scene["robot"].data.joint_vel[:, 12:].abs().mean(dim=1).clamp(max=limit)
  return speed * _term(env).unsettled


def _over(value: torch.Tensor, dead: float, full: float) -> torch.Tensor:
  """0 below `dead`, 1 from `full` on, linear in between: a bounded price with the same slope at every size."""
  return ((value - dead) / (full - dead)).clamp(0.0, 1.0)


def tilt_reaction(env: ManagerBasedRlEnv, dead: float = 0.03, full: float = 0.28) -> torch.Tensor:
  """While reacting: how far the pelvis has turned from its posture before the hit (rad, 0..1 between dead and full)."""
  return _over(_term(env).tilt_dev, dead, full)


def upper_reaction(env: ManagerBasedRlEnv, dead: float = 0.05, full: float = 0.35) -> torch.Tensor:
  """While reacting: waist and arm joints away from their angles before the hit, mean over the 12 joints (0..1)."""
  return _over(_term(env).joint_dev[:, 12:], dead, full).mean(dim=1)


def stance_reaction(env: ManagerBasedRlEnv, dead: float = 0.05, full: float = 0.35, linear: float = 0.03,
                    yaw: float = 0.05) -> torch.Tensor:
  """While reacting and asked to stand: leg joints away from their angles before the hit (0..1). A walking robot's
  legs have no such reference (the gait goes on), so walking is not priced here."""
  term = _term(env)
  goal = term.twist_goal()
  standing = ((torch.norm(goal[:, :2], dim=1) < linear) & (goal[:, 2].abs() < yaw)).float()
  return standing * _over(term.joint_dev[:, :12], dead, full).mean(dim=1)


def arm_hold(env: ManagerBasedRlEnv, dead: float = 0.1) -> torch.Tensor:
  """While reacting: how far the arm outputs of this step are from what they were before the hit (their 0.5 s
  average, frozen at the hit), squared above a dead band and bounded at 1, mean over the ten arm joints. It prices
  the command itself, in the step it is given: the arms keep doing what they did unless that is worth the price."""
  term = _term(env)
  action = env.action_manager.get_term("joint_pos").raw_action[:, 14:]
  change = ((action - term.action_before[:, 14:]).abs() - dead).clamp(min=0.0)
  return change.square().clamp(max=1.0).mean(dim=1) * term.reacting


def reaction_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """For the critic: reacting, pelvis deviation, mean waist+arm and leg deviations, and the gentle class."""
  term = _term(env)
  return torch.stack([term.reacting, term.tilt_dev, term.joint_dev[:, 12:].mean(dim=1),
                      term.joint_dev[:, :12].mean(dim=1), term.gentle.float()], dim=1)


def disturbance_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """What the policy's estimation head is trained to tell from its own observation history (training only):
  unsettled, reacting, and the last impulse in the robot's heading frame (N s / 20) while reacting."""
  term = _term(env)
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  cos, sin = yaw.cos(), yaw.sin()
  ix, iy = term.hit_impulse_w[:, 0], term.hit_impulse_w[:, 1]
  scale = term.reacting / 20.0
  return torch.stack([term.unsettled, term.reacting, (cos * ix + sin * iy) * scale, (-sin * ix + cos * iy) * scale], dim=1)


def anchored_env_cfg(play: bool = False, teacher: str | None = None, clean: bool | None = None,
                     subset: str = "core", intensity: str = "base", quiet_reaction: bool = False,
                     reaction: str = "plain", anchor_on: str = "settled",
                     release_lateral: bool = False, estimate: bool = False,
                     arms_on_course: bool = False, gentle_share: float | None = None,
                     hard_bursts: float = 0.0, course_mode: str = "gentle") -> ManagerBasedRlEnvCfg:
  """`teacher` = the S1-a checkpoint: the frozen anchor (the algorithm loads it) and, unless the task starts from
  scratch, the start weights of the policy.
  Design axes (tested at their ends first, then bisected; see gates.md):
    intensity  "base": 5-30 N s, 40% of the events bursts of 2-8 hits of 6-15 N s, 0.1-0.4 s apart.
               "extreme": 5-45 N s, 70% bursts of 4-12 hits of 8-16 N s, 0.08-0.25 s apart (well beyond the test of
               eight 12 N s hits 0.15 s apart); the range opens over the first 300 iterations.
               "beyond": the far end of coverage. Hits anywhere on the body (half on the trunk, half on thighs,
               shanks, upper arms and forearms), up to head height, 5-60 N s, 70% bursts of 4-16 hits of 8-18 N s,
               0.06-0.25 s apart. Diverged after four minutes (trunk-sized impulses on limb links).
               "anywhere": "extreme" spread over the body with limb impulses scaled to what a ball can pass on.
    quiet_reaction  prices waist and arm motion while unsettled and raises the price of the pelvis turning away.
                    Result: no effect on the reaction (the mean waist+arm speed while unsettled stayed at 1.5 rad/s
                    from iteration 100 to 700; the deterministic policy's reaction to 12 N s was unchanged at 500).
    reaction   "sharp": the far corner of the reaction axis. (1) Half of the disturbed environments are "gentle":
               single hits of 5-15 N s, 3-6 s apart, where staying up is easy and the reaction is what differs.
               (2) The reaction is priced by its size against the state just before the hit, from the impulse until
               1 s after the body is steady: pelvis posture, waist and arm joints, and the leg joints when asked to
               stand; each price is linear from a dead band up to a bound. (3) Pay for staying up 10, so that the
               bounded prices (at most 8.5) stay below it.
               Result at 500 iterations from E31: falls kept (single 0/160, burst 6/160), reaction unchanged (arm
               speed 0.97 -> 0.94 rad/s, joints moved 21.7 -> 21.4).
    anchor_on  where the anchor holds (see anchor_signal): "settled" (rounds up to E36), "calm" (never-hit
               environments), "nominal" (never-hit environments run without exploration noise)."""
  if intensity not in ("base", "extreme", "beyond", "anywhere"):
    raise ValueError(intensity)
  if reaction not in ("plain", "sharp", "gentle", "hold"):
    raise ValueError(reaction)
  if anchor_on not in ("settled", "calm", "nominal", "not_reacting"):
    raise ValueError(anchor_on)
  cfg = stream_env_cfg(play=play, clean=clean, subset=subset)
  command = disturbed_command_cfg(cfg.commands["motion"], shares=None)
  command.spike_share, command.burst_prob = 0.7, 0.0 if play else 0.4
  command.hit_interval, command.calm_fraction = (1.5, 4.0), 0.2
  if intensity in ("extreme", "anywhere") and not play:
    command.impulse_range, command.burst_prob = (5.0, 45.0), 0.7
    command.burst_min, command.burst_max = 4, 12
    command.burst_impulse, command.burst_gap = (8.0, 16.0), (0.08, 0.25)
    command.impulse_ramp_steps = 300 * 24
  if intensity == "beyond" and not play:
    limbs = tuple(f"{side}_{part}" for side in ("left", "right")
                  for part in ("hip_yaw_link", "knee_link", "shoulder_roll_link", "elbow_link"))
    command.hit_bodies = ("torso_link", "pelvis") + limbs
    command.hit_body_weights = (2.0, 2.0) + (0.5,) * len(limbs)
    command.hit_height = (-0.1, 0.4)
    command.impulse_range, command.burst_prob = (5.0, 60.0), 0.7
    command.burst_min, command.burst_max = 4, 16
    command.burst_impulse, command.burst_gap = (8.0, 18.0), (0.06, 0.25)
    command.impulse_ramp_steps = 300 * 24
  if intensity == "anywhere" and not play:
    # "extreme" with the hits spread over the body: half on the trunk (up to head height), half on thighs, shanks,
    # upper arms and forearms with the share of the impulse a ball can pass on to them (legs 0.6, arms 0.35).
    limbs = tuple(f"{side}_{part}" for side in ("left", "right")
                  for part in ("hip_yaw_link", "knee_link", "shoulder_roll_link", "elbow_link"))
    command.hit_bodies = ("torso_link", "pelvis") + limbs
    command.hit_body_weights = (2.0, 2.0) + (0.5,) * len(limbs)
    command.hit_body_scale = (1.0, 1.0) + (0.6, 0.6, 0.35, 0.35) * 2
    command.hit_height = (-0.1, 0.4)
  if hard_bursts > 0.0 and not play:
    # A share of the bursts is long and dense (6-9 hits of 10-14 N s, 0.12-0.22 s apart, from one side): the shape
    # of the eight-ball test, where the remaining falls are (pushed backward). The held-out burst probes (5 x 14 N s
    # at 0.25 s, 10 x 9 at 0.08, 8 x 12 at 0.3) stay outside this shape.
    command.burst_hard_share = hard_bursts
  cfg.commands = {"motion": command}
  # A diverged simulation ends the episode instead of the run.
  cfg.terminations["nan"] = TerminationTermCfg(func=mdp.nan_detection)
  cfg.events.pop("push_robot", None)               # impulses come from the command term
  cfg.observations["bc"] = ObservationGroupCfg(
    terms={"signal": ObservationTermCfg(func=anchor_signal,
                                        params={"domain": anchor_on, "release_lateral": release_lateral})},
    concatenate_terms=True, enable_corruption=False)
  if arms_on_course:
    # The anchor for the arms during a moderate hit: the outputs of before the hit (see arm_course). S1-a itself
    # swings the arms there (its teacher followed human clips matched to the pushed state: people do), which is the
    # larger half of what moves; with the arms on their course a 12 N s hit moves as many joints as the
    # manufacturer's policy does (measured with the un-hit twin's arm outputs, 17.6 against 17.2 of 24).
    cfg.observations["course"] = ObservationGroupCfg(
      terms={"arms": ObservationTermCfg(func=arm_course, params={"mode": course_mode})},
      concatenate_terms=True, enable_corruption=False)
    if reaction == "plain":
      # The anchor target alone, with E31's rewards: gentle environments (their share of the disturbed ones is
      # `gentle_share`, default a quarter) exist only so that this target has samples.
      if not play:
        command.gentle_share = 0.25 if gentle_share is None else gentle_share
      command.reaction_tail = 1.0
  if estimate:
    # Targets of the estimation head (see AnchoredPPO): the network learns to tell from its own inputs that it was
    # hit, from where and how hard. Nothing reads these at run time; the head is not part of the exported policy.
    cfg.observations["aux"] = ObservationGroupCfg(terms={"disturbance": ObservationTermCfg(func=disturbance_state)},
                                                  concatenate_terms=True, enable_corruption=False)
  critic = cfg.observations["critic"].terms
  critic["urgency"] = ObservationTermCfg(func=urgency_state)
  critic["settling"] = ObservationTermCfg(func=settling_state)
  cfg.rewards["arm_deviation"].weight = -3.0
  for name in ("goal_twist", "goal_twist_l1", "goal_lag", "goal_height", "goal_pitch"):
    cfg.rewards[name].func = while_settled(cfg.rewards[name].func)
  # While unsettled at most: unsettled 1 + swing 2 + spin 0.5 + urgency 0.5 = 4, below the 8 for staying up.
  cfg.rewards.update({
    "alive": RewardTermCfg(func=alive, weight=8.0),
    "fell": RewardTermCfg(func=fell, weight=-200.0),
    "unsettled": RewardTermCfg(func=unsettled_cost, weight=-1.0),
    "swing": RewardTermCfg(func=swing_cost, weight=-4.0),
    "spin": RewardTermCfg(func=spin_cost, weight=-0.02),
    "urgency": RewardTermCfg(func=urgency_cost, weight=-0.5),
  })
  if quiet_reaction:
    # Far end of the reaction-size axis. At most: unsettled 1 + swing 3 + spin 0.5 + urgency 0.5 + reaction 2 = 7,
    # still below the 8 for staying up.
    cfg.rewards["swing"].weight = -6.0
    cfg.rewards["reaction"] = RewardTermCfg(func=reaction_cost, weight=-1.0)
  if reaction == "hold":
    # Third form. The second one (reaction="gentle") left the deterministic reaction unchanged after 400 updates
    # although its prices prefer quiet arms: a price on joint angles reaches the arm commands only through the
    # dynamics. Here the arm command itself is priced, in the step it is given (arm_hold), in the gentle
    # environments; the waist+arm angle price stays at a third. Gentle bounds: 4 + 7.5 + 4 + 12 + 3 = 30.5 < 32.
    if not play:
      command.gentle_share = 0.5
    command.reaction_tail = 1.0
    critic["reaction"] = ObservationTermCfg(func=reaction_state)
    for name in ("goal_twist", "goal_twist_l1", "goal_lag", "goal_height", "goal_pitch"):
      cfg.rewards[name].func = while_not_reacting(cfg.rewards[name].func.__wrapped__)
    cfg.rewards["alive"].weight = 32.0
    cfg.rewards["fell"].weight = -3200.0
    cfg.rewards["tilt_reaction"] = RewardTermCfg(func=in_gentle(tilt_reaction), weight=-7.5)
    cfg.rewards["upper_reaction"] = RewardTermCfg(func=in_gentle(upper_reaction), weight=-4.0)
    cfg.rewards["arm_hold"] = RewardTermCfg(func=in_gentle(arm_hold), weight=-12.0)
    cfg.rewards["stance_reaction"] = RewardTermCfg(func=in_gentle(stance_reaction), weight=-3.0)
  if reaction == "gentle":
    # Second form of the reaction prices, after measuring the first on E31 (reward_check.py): there, keeping the arm
    # outputs as they were before a hit earned LESS than the present reaction (510 -> 504 per second: the
    # command-following terms, back on 0.7 s after the hit, pay for chasing the commanded speed with the whole
    # body). So: (1) command following is not counted during a reaction (spell + 1 s), in every environment;
    # (2) the size of the reaction is priced in the gentle environments only (there staying up is not at stake; in
    # the extreme ones only staying up counts).
    if not play:
      command.gentle_share = 0.5
    command.reaction_tail = 1.0
    critic["reaction"] = ObservationTermCfg(func=reaction_state)
    for name in ("goal_twist", "goal_twist_l1", "goal_lag", "goal_height", "goal_pitch"):
      cfg.rewards[name].func = while_not_reacting(cfg.rewards[name].func.__wrapped__)
    # (3) Relative prices from the same measurement: with the arms kept on their course the waist+arm price falls
    # by a third while the pelvis turns slightly more and the commanded speed returns slightly later; waist+arms 12
    # against tilt 7.5 makes that the better outcome by a clear margin. (4) Staying up 30 per step-second, above the
    # sum of all bounded prices (gentle: 4 + 7.5 + 12 + 3 = 26.5), and a fall costs 3000 (about as much as the pay
    # of the two seconds the value horizon covers): no reduction of a reaction is worth a visible risk of falling.
    cfg.rewards["alive"].weight = 30.0
    cfg.rewards["fell"].weight = -3000.0
    cfg.rewards["tilt_reaction"] = RewardTermCfg(func=in_gentle(tilt_reaction), weight=-7.5)
    cfg.rewards["upper_reaction"] = RewardTermCfg(func=in_gentle(upper_reaction), weight=-12.0)
    cfg.rewards["stance_reaction"] = RewardTermCfg(func=in_gentle(stance_reaction), weight=-3.0)
  if reaction == "sharp":
    if not play:
      command.gentle_share = 0.5
    command.reaction_tail = 1.0
    critic["reaction"] = ObservationTermCfg(func=reaction_state)
    # At most: unsettled 1 + urgency 0.5 + spin 0.5 + swing 1 + tilt 2.5 + upper 2 + stance 1 = 8.5 < 10.
    cfg.rewards["alive"].weight = 10.0
    cfg.rewards["swing"].weight = -2.0
    cfg.rewards["tilt_reaction"] = RewardTermCfg(func=tilt_reaction, weight=-2.5)
    cfg.rewards["upper_reaction"] = RewardTermCfg(func=upper_reaction, weight=-2.0)
    cfg.rewards["stance_reaction"] = RewardTermCfg(func=stance_reaction, weight=-1.0)
  return cfg


@dataclass
class AnchoredPpoCfg(ImitationPpoCfg):
  class_name: str = "r1wb.learning:AnchoredPPO"
  anchor_checkpoint: str = ""                    # set by the training script (the --teacher checkpoint)
  anchor_groups: tuple[str, ...] = ("actor", "goal")
  aux_group: str = ""                            # observation group with the estimation targets ("" = no head)
  aux_coef: float = 1.0
  aux_tap: int = 2                               # the head reads the input of this linear layer of the policy
  aux_dim: int = 4                               # width of that group (see disturbance_state)
  class_advantages: bool = False                 # scale the advantages within each environment class
  course_group: str = ""                         # observation group of arm_course ("" = S1-a's own arm outputs)
  course_weight: float = 1.0                     # weight of an arm joint on its course against a settled sample's 1
  repeat_checkpoint: str = ""                    # set by the training script: the checkpoint the run continues from
  repeat_weight: float = 0.0                     # weight of "arms as that checkpoint does" once hits repeat (0 = none)


def anchored_rl_cfg(learned_noise: bool = False, scratch: bool = False, anchor: float = 20.0,
                    estimate: bool = False, class_advantages: bool = False,
                    arms_on_course: bool = False, course_weight: float = 1.0,
                    repeat_weight: float = 0.0) -> RslRlOnPolicyRunnerCfg:
  """The student's network (the start weights have to fit) and its fixed learning-rate schedule; the imitation
  coefficient is constant (the anchor never fades). learned_noise: the exploration noise is a second output of the
  network (started at the start policy's per-joint values by r1wb.train.start_policy).
  Design axes: scratch = the policy starts from random weights (calm behaviour is then learned from the anchor like
  any imitation, the reaction to hits from nothing; noise 1.0 and learnable, no warm-up, a higher learning rate);
  anchor = the imitation coefficient (0 = no anchor at all)."""
  if scratch:
    return RslRlOnPolicyRunnerCfg(
      actor=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True,
                          distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
      critic=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True),
      algorithm=AnchoredPpoCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="fixed",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        bc_coef=anchor, bc_floor=anchor, critic_warmup_updates=0, surrogate_ramp_updates=1,
        ppo_learning_rate=3.0e-4, final_learning_rate=1.0e-4, lr_decay_updates=2000),
      obs_groups={"actor": ("actor", "goal"), "critic": ("critic",)},
      experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=2000,
      logger="tensorboard", upload_model=False)
  # Own noise: the start policy's per-joint values, kept fixed (left learnable, the single number per joint rose from
  # 0.33 to 0.45 in 200 iterations and calm motion got noisier with it). Learned noise: per joint and per state.
  distribution = ({"class_name": "HeteroscedasticGaussianDistribution", "init_std": 0.3, "std_type": "log",
                   "std_range": (0.05, 1.0)} if learned_noise else
                  {"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar", "learn_std": False})
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True,
                        distribution_cfg=distribution),
    critic=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True),
    algorithm=AnchoredPpoCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=5.0e-4, schedule="fixed",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
      bc_coef=anchor, bc_floor=anchor, critic_warmup_updates=100, surrogate_ramp_updates=100,
      ppo_learning_rate=1.0e-4, final_learning_rate=5.0e-5, lr_decay_updates=2000,
      aux_group="aux" if estimate else "", class_advantages=class_advantages,
      course_group="course" if arms_on_course else "", course_weight=course_weight, repeat_weight=repeat_weight),
    obs_groups={"actor": ("actor", "goal"), "critic": ("critic",)},
    experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=2000,
    logger="tensorboard", upload_model=False)
