"""P1t: the teacher on the task itself.

The earlier teachers learned to replay a clip where the clip is (world-anchored tracking) and were then asked, as
label providers, for something else: follow a command stream from wherever the robot stands, re-anchored. Measured
11-12 07:00 on calm command streams, labels driving the robot: every teacher of the line (P1c, E63, E64, E70) is
ended by the progress limit once per 31 to 37 s, turning in place once per 15 to 18 s. What the teacher could not
do, the student had to find by itself against its labels.

Here the teacher solves the deployed task: the student's environment as it is (clips and command streams, the
reference re-anchored to the robot, progress judged on the twist goal, the price on leaving the course, the
evaluation's impulses), with the teacher's own observation: the course, plus the goal and how far it is behind it.
Teacher and student then solve one problem and differ only in what they see. A teacher is finished when it passes,
driving the robot itself, every test the deployed policy has to pass.

The stream's course is the command's plan (selected from the command and its own frame, never from the robot): a
course that follows the robot can neither lead it nor price what a hit does to it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking import mdp

from r1wb import robot
from r1wb.student import observations as obs
from r1wb.student.env_cfg import ImitationPpoCfg
from r1wb.student.stream import mixed_env_cfg
from r1wb.tracking import mdp as wb
from r1wb.tracking.env_cfg import EXPERIMENT, tracking_env_cfg, use_local_anchor


def teacher_task_env_cfg(play: bool = False, clean: bool | None = None, subset: str = "core",
                         disturbance: str = "impulse_events", joint_course: float = 5.0,
                         hit_grace: float = 2.0, failure_cost: float = 0.0,
                         coherent_course: bool = False, step_cost: float = 0.0,
                         row_share: float | None = None, decisive_course: bool = False,
                         course_travel: float = 0.0, course_cap: float | None = None,
                         steer: str = "", step_rest: bool = False, rest_room: float = 0.0,
                         step_calm: float = 0.0, ground: bool | str = False,
                         plan_ends: bool = True, senses: str = "") -> ManagerBasedRlEnvCfg:
  """failure_cost: charged once when an episode ends on a failure (a fall, a lost pose, the progress limit), in
  seconds of reward weight (the reward manager multiplies by the step). While a row of balls lands the step reward is
  negative (measured 11-12 07:35 at a trunk tilt above 0.3 rad: -1.7 to -5 per step: the velocity the balls inject,
  the recovery steps and the price are all charged), so enduring a long row was worth about as much as ending the
  episode, and the run with rows of balls fell more and more (E76). Staying up has to be worth more than anything
  that can be saved by going down: 1000 = 20 in the learner's units, twice the worst row measured.
  coherent_course: the stream's course plays the window a look-up scored before the next look-up (match_commit):
  pieces of whole steps instead of half-second pieces (without training, E70's labels then turn right in place at
  0.83 instead of 0.02).
  step_cost: charged once per touchdown the course does not have (tracking.mdp.step_unprompted). Together with the
  joint-course price and the failure cost this is one price list of interventions by physical size: leaving the
  course's posture (per rad and step), a step the course does not take (25 = 0.5 in the learner's units, about
  four steps of calm reward), failing (20).
  step_rest: the step price only for steps taken while the course stands still (step_unprompted `rest`).
  step_calm: with step_rest, only steps with no hit in the last `step_calm` seconds are charged.
  rest_room: price per metre and second of the capture point lacking room inside the support while the course
  stands still and nothing has hit the robot (tracking.mdp.rest_room): the condition those steps come from.
  course_travel: price per m/s of the robot's travel leaving the course's travel (tracking.mdp.course_travel_l1), the
  second line of the price list next to the posture price.
  decisive_course: the course generator as System 2's data uses it since 11-13 (r1wb.s2.collect): a look-up asks for
  the height the command is going to, "stand" chooses among motion that comes to rest, a chosen piece is played for
  0.64 s instead of 1.2 s, no random spread. With the 1.2 s pieces a sideways command's course covered half of the
  commanded distance (with 0.64 s: 0.84 to 0.99), and a policy trained on the slow courses lags the faster ones
  (E87 sideways 0.52 to 0.81 of the command against E79's 0.85 to 0.95; gates 11-13 05:30).
  course_cap: the posture price saturates: the mean joint distance to the course is charged up to this many rad per
  step and not beyond. Measured 11-13 10:46 on E94 (135 : 3000): the distance summed over the 2.5 s after a hit is
  0.085 rad s for single hits (price 11) and 0.33 rad s, up to 0.66, for rows of eight that are withstood (price 44,
  up to 90) against 60 for a fall on top of what was paid until then: a hard row is not worth fighting. Capped, small
  departures are priced as before and a fight for balance costs little more than a single reaction.
  steer: a learned System 2 (data/s2/steer_<name>.pt) gives the streams their commands during training
  (student.stream MixedGoalCommandCfg.steer): the two layers are trained on each other's behaviour in turns.
  row_share: share of the hit events that open a row of balls (the disturbance's own value if None). It is the
  policy's prior that a first hit is followed by more: at one half the reaction to any single hit is whole-body
  (all ten arm joints, arm speed 1 to 2 rad/s: E80; rows left out: 4 to 6 arm joints, 0.3 to 0.5 rad/s: E79).
  ground: the floor is a field of shapes (r1wb.world.training_ground: bumps, stones, slopes, a tilted floor, low
  stairs, a share of flat) and every height of the course, the height goal and the fall criterion are taken above
  the ground under the robot's feet. Measured 11-14 without training: read against the start level, the course's
  height alone makes the flat-floor policy fall on a 10 deg slope (6 of 10 runs; 0 of 11 with the height above the
  ground); slopes of 20 deg and standing still on any slope need the experience. ground="feet": each foot's height is
  taken above the ground under that foot (the pelvis and the hands above the mean of the two): two feet a step apart
  on a 25 deg slope stand 14 cm apart in height, which the mean alone reads as a foot-height error (the flat-floor
  policy's episodes on the steep levels end on the foot-height criterion and on lagging progress, 11-14 03:00).
  plan_ends=False: leaving the course's heights (pelvis, feet, hands) or its tilt does not end an episode; a fall, the
  trunk on the ground and lagging progress still do. On shaped ground those departures are the adaptation itself
  (after a hit they were already suspended for `hit_grace` seconds); the course stays a priced preference.
  senses: what the actor reads beyond the plan and its state now (r1wb.senses): "history" (0.2 s of its own body),
  "scan" (the ground around it, also given to the critic) or "both"; appended, so that a policy without them
  continues as the same function (r1wb.train widened)."""
  clean = play if clean is None else clean
  cfg = mixed_env_cfg(play=play, teacher=None, clean=clean, subset=subset, disturbance=disturbance,
                      joint_course=joint_course, hit_grace=hit_grace, stream_course="plan")
  cfg.commands["motion"].match_commit = coherent_course
  if decisive_course:
    assert coherent_course
    motion = cfg.commands["motion"]
    motion.match_height, motion.match_stop, motion.travel_window, motion.match_noise = "target", "arrive", 32, 0.0
  # The teacher's observation and action, as in the tracking environment it comes from (same order: its weights
  # continue), re-anchored to the robot; the goal and the goal lag are appended.
  base = tracking_env_cfg(play=play, subset=subset, hardened=True, clean=clean, slip_excess=True, contact_match=True,
                          disturbance=disturbance, joint_course=joint_course, hit_grace=hit_grace)
  use_local_anchor(base, groups=("actor",))
  actor = base.observations["actor"]
  name = {"command_name": "motion"}
  actor.terms["goal"] = ObservationTermCfg(func=obs.goal, params=name)
  actor.terms["goal_lag"] = ObservationTermCfg(func=obs.goal_lag, params=name)
  cfg.observations = {
    "actor": actor,
    "critic": cfg.observations["critic"],
    "bc": ObservationGroupCfg(terms={"label": ObservationTermCfg(func=obs.no_label)}, concatenate_terms=True,
                              enable_corruption=False),
  }
  cfg.actions = base.actions
  if row_share is not None and "impulse" in cfg.events:      # a play scene has no disturbance event
    cfg.events["impulse"].params["burst_prob"] = row_share
  if course_cap is not None:
    cfg.rewards["joint_course"].params["cap"] = course_cap
  if steer:
    cfg.commands["motion"].steer = steer
  if ground and not play:
    from r1wb import world
    world.training_ground(cfg)
  if ground:
    cfg.commands["motion"].on_support = True
    cfg.commands["motion"].feet_on_ground = ground == "feet"
  if not plan_ends:
    for name in ("anchor_z", "feet_z", "hands_z", "anchor_rot"):
      cfg.terminations.pop(name, None)
  if senses:
    from r1wb import senses as extra
    extra.add_sensor(cfg, senses)
    actor.terms["senses"] = ObservationTermCfg(func=extra.read, params={"kind": senses})
    if senses in ("scan", "both"):
      cfg.observations["critic"].terms["ground_scan"] = ObservationTermCfg(func=extra.read, params={"kind": "scan"})
  if failure_cost > 0.0:
    cfg.rewards["failure"] = RewardTermCfg(func=mdp.is_terminated, weight=-failure_cost)
  if course_travel > 0.0:
    trust = {key: value for key, value in cfg.rewards["joint_course"].params.items() if key in ("command_name", "stream", "grace")}
    cfg.rewards["course_travel"] = RewardTermCfg(func=wb.course_travel_l1, weight=-course_travel, params=trust)
  if rest_room > 0.0:
    cfg.rewards["rest_room"] = RewardTermCfg(func=wb.rest_room, weight=-rest_room, params={"command_name": "motion"})
  if step_cost > 0.0:
    cfg.rewards["step"] = RewardTermCfg(func=wb.step_unprompted, weight=-step_cost,
                                        params={"command_name": "motion", "sensor_name": "feet_ground",
                                                "foot_bodies": robot.FOOT_BODIES, "rest": step_rest, "calm": step_calm})
  return cfg


@dataclass
class SymmetricPpoCfg(ImitationPpoCfg):
  class_name: str = "r1wb.learning:SymmetricPPO"
  mirror_coef: float = 4.0


def teacher_task_rl_cfg(mirror: float = 0.0, symmetric: bool = False) -> RslRlOnPolicyRunnerCfg:
  """The teacher's network continued with the student's learner without labels: the value function starts new for
  the new rewards (50 updates before the policy moves, then policy gradient and rate ramped in over 100).
  mirror: coefficient of the left-right mirror loss (r1wb.learning.SymmetricPPO); 0 = the plain learner. Tried once
  and not used: the loss destroyed a finished policy on its way to symmetry.
  symmetric: the policy is the network averaged with its own mirror image (r1wb.symmetric_model): symmetric by
  construction, the same weights, trained by the plain learner."""
  if symmetric:
    cfg = teacher_task_rl_cfg(mirror=mirror)
    cfg.actor.class_name = "r1wb.symmetric_model:SymmetricMLPModel"
    return cfg
  if mirror > 0.0:
    cfg = teacher_task_rl_cfg()
    plain = cfg.algorithm
    cfg.algorithm = SymmetricPpoCfg(**{f.name: getattr(plain, f.name) for f in fields(plain) if f.name != "class_name"},
                                    mirror_coef=mirror)
    return cfg
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True),
    algorithm=ImitationPpoCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=5.0e-4, schedule="fixed",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
      bc_coef=0.0, bc_floor=0.0, ramp_learning_rate=True),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=4000,
    logger="tensorboard", upload_model=False)
