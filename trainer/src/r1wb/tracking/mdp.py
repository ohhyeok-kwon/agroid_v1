"""Terms added to the stock tracking recipe for a multi-clip library (terminations, penalties, pushes).

Why they exist: with the stock terms a robot that stays where it started while the reference walks away keeps most
of the reward and is never terminated, the orientation test only compares the vertical component of gravity (blind
to heading and to the direction of a lean), and a 0.25 m height tolerance accepts a half crouch.
"""
from __future__ import annotations

import math

from typing import TYPE_CHECKING

import torch
from r1wb.performance import cached_read, device_indices

from mjlab.utils.lab_api.math import (matrix_from_quat, quat_apply, quat_apply_inverse, quat_error_magnitude, quat_inv,
                                      quat_mul, yaw_quat)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _term(env, command_name):
  return env.command_manager.get_term(command_name)


@cached_read
def heading_shift(term) -> torch.Tensor:
  """Planar rotation that gives the reference the robot's heading (the re-anchoring the relative body terms use)."""
  return yaw_quat(quat_mul(term.robot_anchor_quat_w, quat_inv(term.anchor_quat_w)))


@cached_read
def local_anchor_quat(term) -> torch.Tensor:
  """Reference anchor rotation after re-anchoring: the reference's tilt at the robot's heading."""
  return quat_mul(heading_shift(term), term.anchor_quat_w)


def _body_ids(term, body_names):
  ids = [i for i, name in enumerate(term.cfg.body_names) if body_names is None or name in body_names]
  return device_indices(term._env, ids)


def body_position_error_exp(env, command_name, std, body_names=None):
  term = _term(env, command_name)
  ids = _body_ids(term, body_names)
  error = torch.sum(torch.square(term.body_pos_relative_w[:, ids] - term.robot_body_pos_w[:, ids]), dim=-1)
  return torch.exp(-error.mean(-1) / std**2)


def body_orientation_error_exp(env, command_name, std, body_names=None):
  term = _term(env, command_name)
  ids = _body_ids(term, body_names)
  error = quat_error_magnitude(term.body_quat_relative_w[:, ids], term.robot_body_quat_w[:, ids]) ** 2
  return torch.exp(-error.mean(-1) / std**2)


def body_z_error(env, command_name, threshold, body_names=None):
  term = _term(env, command_name)
  ids = _body_ids(term, body_names)
  error = torch.abs(term.body_pos_relative_w[:, ids, -1] - term.robot_body_pos_w[:, ids, -1])
  return torch.any(error > threshold, dim=-1)


# -- observations: the reference as seen from where the robot is (no planar or heading error) -------------------
#
# A policy that receives only a twist in its own heading frame cannot observe how far it has drifted from the
# reference in the plane, so nothing that teaches or scores it may depend on that drift. These terms replace the
# world-frame anchor terms wherever the reference-free student is involved (teacher labels, critic, rewards).

def local_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor position in the robot anchor frame with the planar error removed (height error only)."""
  term = _term(env, command_name)
  offset = torch.zeros_like(term.anchor_pos_w)
  offset[:, 2] = term.anchor_pos_w[:, 2] - term.robot_anchor_pos_w[:, 2]
  return quat_apply_inverse(term.robot_anchor_quat_w, offset)


def local_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor rotation in the robot anchor frame with the heading error removed (tilt only)."""
  term = _term(env, command_name)
  matrix = matrix_from_quat(quat_mul(quat_inv(term.robot_anchor_quat_w), local_anchor_quat(term)))
  return matrix[..., :2].reshape(env.num_envs, 6)


# -- terminations -----------------------------------------------------------------------------------------------

def anchor_xy_error(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  term = _term(env, command_name)
  return torch.norm(term.anchor_pos_w[:, :2] - term.robot_anchor_pos_w[:, :2], dim=1) > threshold


def anchor_rotation_error(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  term = _term(env, command_name)
  return quat_error_magnitude(term.anchor_quat_w, term.robot_anchor_quat_w) > threshold


def anchor_tilt_error(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  """Rotation error after the reference has been given the robot's heading (roll and pitch only)."""
  term = _term(env, command_name)
  return quat_error_magnitude(local_anchor_quat(term), term.robot_anchor_quat_w) > threshold


def body_on_ground(env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 1.0) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  return torch.norm(force.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=1) > force_threshold


# -- rewards ----------------------------------------------------------------------------------------------------

def anchor_xy_error_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = _term(env, command_name)
  return torch.norm(term.anchor_pos_w[:, :2] - term.robot_anchor_pos_w[:, :2], dim=1)


def anchor_z_error_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = _term(env, command_name)
  return (term.anchor_pos_w[:, 2] - term.robot_anchor_pos_w[:, 2]).abs()


def stance_slip(env: ManagerBasedRlEnv, sensor_name: str, foot_bodies: tuple[str, ...],
                command_name: str | None = None, allowance: float = 0.05) -> torch.Tensor:
  """Planar speed of the feet that carry load (m/s, summed over feet).

  With `command_name` only the excess over the reference's own foot speed is charged (minus `allowance`): the
  ankle of a trailing foot that rolls over its toes moves while the foot is loaded, and the plain version charged
  a perfectly tracking robot 14-17% of the reward on walking clips (second training review, 0.1 item 1).
  """
  robot = env.scene["robot"]
  cache = robot.__dict__.setdefault("_r1wb_cache", {})
  if "feet" not in cache:
    cache["feet"] = [robot.body_names.index(name) for name in foot_bodies]
  speed = torch.norm(robot.data.body_link_lin_vel_w[:, device_indices(env, cache["feet"]), :2], dim=-1)
  contact = env.scene[sensor_name].data.found.reshape(env.num_envs, -1)[:, :len(foot_bodies)] > 0
  if command_name is not None:
    term = _term(env, command_name)
    reference = torch.norm(term.body_lin_vel_w[:, _body_ids(term, foot_bodies), :2], dim=-1)
    speed = (speed - reference - allowance).clamp_min(0.0)
  return (speed * contact.float()).sum(dim=1)


def foot_lift_unprompted(env: ManagerBasedRlEnv, command_name: str, sensor_name: str, foot_bodies: tuple[str, ...],
                         height_band: float = 0.02, speed_limit: float = 0.15) -> torch.Tensor:
  """Number of feet that are off the ground while the reference keeps the same foot planted (low and not moving).

  The mirror image of `stance_slip`: that one charges moving a loaded foot, this one charges lifting a foot the
  motion keeps down. Without it the first teacher took steps nobody asked for: 203 of 1,415 touchdowns in the
  deterministic evaluation happened while the reference foot neither lifted nor moved (E3, iteration 1,500).
  A reference foot that slides or pivots on the ground is not "planted", so re-placing the foot there is free.
  """
  term = _term(env, command_name)
  ids = _body_ids(term, foot_bodies)
  height = term.body_pos_w[:, ids, 2]
  planted = ((height < height.amin(dim=1, keepdim=True) + height_band)
             & (torch.norm(term.body_lin_vel_w[:, ids, :2], dim=-1) < speed_limit))
  contact = env.scene[sensor_name].data.found.reshape(env.num_envs, -1)[:, :len(foot_bodies)] > 0
  return (planted & ~contact).float().sum(dim=1)


class step_unprompted:
  """Steps the course does not take: 1 for every foot that lands (after at least `min_air` steps off the ground)
  while the reference kept the same foot planted (low and not moving) for the whole flight. The same definition
  as the evaluation's "unprompted footfalls" (eval_tracking.py), so the price is on what is measured.

  The price on leaving the course was measured in joint angles only, and a short step moves the joints little: the
  teachers trained on the task took three to four times as many unprompted steps as the tracking teacher they
  started from (106 to 138 against 32 over the 134 clips, calm) and 7 to 11 steps after a 25 N s hit against 1 to 6.
  A step is an intervention of its own size (it changes the support), between moving a joint and falling, and is
  charged per step, not per time in the air (a charge on air time is saved by stepping quickly, not by stepping
  less). A step the robot needs to stay up is still worth it: failing costs many steps.

  A flight during which the stream's course changes clips is judged on the piece it started under (the new piece
  may keep planted a foot the old one had told to swing).
  Not used: counting robot touchdowns against reference touchdowns. The reference's own steps are under-counted
  (a foot that skims the ground never leaves the height band: even world-anchored tracking shows 145 footfalls on
  the backward clips against 90 counted in the reference, with 2 to 7 of them unprompted).

  rest: charge only the steps taken while the course stands still (the command stream's `hold`: asked to stand and
  the reference paused on a standing frame) for the whole flight. There the definition has no grey zone (no reference
  foot skims or pivots), which is why the term was shelved for moving courses (gates 11-12 08:23). Measured 11-13
  14:06: after a stop the policy keeps the course's posture to 0.02 rad and steps in place on its heels five times a
  second for three seconds, because that costs next to nothing (`foot_lift` charges time in the air, 0.02 a step)
  while leaning to bring the centre of mass back costs posture price."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    count, feet = env.num_envs, len(cfg.params["foot_bodies"])
    self.air = torch.zeros(count, feet, dtype=torch.long, device=env.device)
    self.kept = torch.zeros(count, feet, dtype=torch.bool, device=env.device)   # the reference foot stayed planted
    self.switches = torch.zeros(count, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, sensor_name: str, foot_bodies: tuple[str, ...],
               height_band: float = 0.02, speed_limit: float = 0.15, min_air: int = 3, rest: bool = False,
               calm: float = 0.0) -> torch.Tensor:
    """Legacy evaluation only: positive calm exempts flights following a recent hit.
    At calm=0 every flight is priced without consulting the hit clock.
    In the scene 86% of the steps taken under a standing course follow a hit within 2 s (reward check 11-13
    14:15: 127 per 100 env-s there against 7 when calm): without this the term is mostly a tax on recovering."""
    term = _term(env, command_name)
    ids = _body_ids(term, foot_bodies)
    contact = env.scene[sensor_name].data.found.reshape(env.num_envs, -1)[:, :len(foot_bodies)] > 0
    if rest:
      hold = getattr(term, "hold", None)
      hold = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) if hold is None else hold
      if calm > 0.0:
        clock = getattr(env, "hit_clock", None)
        if clock is not None:
          hold = hold & (clock >= calm)
      planted = hold.unsqueeze(1).expand_as(contact)
    else:
      height = term.body_pos_w[:, ids, 2]
      planted = ((height < height.amin(dim=1, keepdim=True) + height_band)
                 & (torch.norm(term.body_lin_vel_w[:, ids, :2], dim=-1) < speed_limit))
    fresh = (env.episode_length_buf <= 1).unsqueeze(1)
    self.air = torch.where(fresh, torch.zeros_like(self.air), self.air)
    switches = None if rest else getattr(term, "match_switches", None)
    if switches is not None:            # a foot in the air when the course changes clips: the flight is not charged
      changed = (switches != self.switches).unsqueeze(1)
      self.kept = self.kept & ~(changed & ~contact)
      self.switches = switches.clone()
    lifting = ~contact & (self.air == 0)
    self.kept = torch.where(lifting, planted, self.kept & planted)
    landing = contact & (self.air >= min_air) & self.kept
    self.air = torch.where(contact, torch.zeros_like(self.air), self.air + 1)
    return landing.float().sum(dim=1)


FOOTPRINT = ((-0.050, 0.133), (-0.035, 0.035))      # the sole's collision capsules in the foot frame (model, 11-13)


class rest_room:
  """While the course stands still: how much the capture point
  (centre of mass + velocity / sqrt(g / height), r1wb.dynamics) lacks of `room` metres inside the support of both
  soles, in metres, at most `cap`.

  Standing still is only possible with the capture point inside the support. Measured 11-13 on 336 stops of E99:
  the stops followed by steps a second or more later have it 2.1 cm from the edge at 0.5 s, the others 6.6 cm, and
  nothing about the course's own pose tells the two apart. The posture price holds the body on the course's joint
  angles and says nothing about where the body is over its feet; this term prices exactly that, and only while the
  course itself claims to be at rest (no assumed gait, no step time). The steps are the symptom (`step_unprompted`
  with `rest`), this is the condition they come from. Privileged (simulator state): a reward, nothing deployed.
  Positive grace preserves historical evaluation only; grace=0 never consults the hit clock."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    from r1wb import dynamics
    self.centroidal = dynamics.Centroidal(env, footprint=FOOTPRINT)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, room: float = 0.04, cap: float = 0.10,
               grace: float = 0.0) -> torch.Tensor:
    term = _term(env, command_name)
    hold = getattr(term, "hold", None)
    if hold is None:
      return torch.zeros(env.num_envs, device=env.device)
    calm = hold
    if grace > 0.0:
      clock = getattr(env, "hit_clock", None)
      if clock is not None:
        calm = hold & (clock >= grace)
    lacking = (self.centroidal.support_margin()["d"] + room).clamp(0.0, cap)
    return lacking * calm.float()


def anchor_height_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  term = _term(env, command_name)
  return torch.exp(-torch.square(term.anchor_pos_w[:, 2] - term.robot_anchor_pos_w[:, 2]) / std**2)


def anchor_tilt_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  term = _term(env, command_name)
  return torch.exp(-quat_error_magnitude(local_anchor_quat(term), term.robot_anchor_quat_w) ** 2 / std**2)


def local_body_linear_velocity_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float,
                                         body_names: tuple[str, ...] | None = None) -> torch.Tensor:
  """Body velocity error with the reference velocities turned to the robot's heading."""
  term = _term(env, command_name)
  ids = _body_ids(term, body_names)
  shift = heading_shift(term).unsqueeze(1).expand(-1, len(ids), -1)
  error = torch.sum(torch.square(quat_apply(shift, term.body_lin_vel_w[:, ids]) - term.robot_body_lin_vel_w[:, ids]), dim=-1)
  return torch.exp(-error.mean(-1) / std**2)


def local_body_angular_velocity_error_exp(env: ManagerBasedRlEnv, command_name: str, std: float,
                                          body_names: tuple[str, ...] | None = None) -> torch.Tensor:
  term = _term(env, command_name)
  ids = _body_ids(term, body_names)
  shift = heading_shift(term).unsqueeze(1).expand(-1, len(ids), -1)
  error = torch.sum(torch.square(quat_apply(shift, term.body_ang_vel_w[:, ids]) - term.robot_body_ang_vel_w[:, ids]), dim=-1)
  return torch.exp(-error.mean(-1) / std**2)


def course_is_trusted(env: ManagerBasedRlEnv, command_name: str, stream: str = "always", grace: float = 0.0) -> torch.Tensor:
  """1 where the reference is a valid course, per environment. A clip is one at all times. A matched command stream
  re-selects its reference from the pushed robot's feet (measured 11-12 03:00: the walking reaction then never
  returns to its orbit), so `stream` says how far it is trusted: "always", "calm" (not within `grace` seconds of a
  hit) or "never". Positive grace is retained for historical evaluation; grace=0 does not read hit timing."""
  term = _term(env, command_name)
  ones = torch.ones(env.num_envs, device=env.device)
  streams = getattr(term, "stream_env", None)
  if streams is None or stream == "always":
    return ones
  if stream == "never":
    return (~streams).float()
  if grace <= 0.0:
    return ones
  clock = getattr(env, "hit_clock", None)
  calm = ones.bool() if clock is None else clock >= grace
  return (~streams | calm).float()


def joint_motion_l1(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean measured joint speed (rad/s); the reward manager integrates it into joint path length.

  A pose error or an action-rate cost can stay small while the physical joints oscillate. This term prices actual
  motion at every step, including recovery: task and failure rewards determine which motion is worth taking.
  L1 preserves the cost of a fixed path when its duration changes; no reference-phase subtraction, hit exemption
  or saturation lets a policy hide extra motion. It is a motion regularizer, not an energy or necessity oracle.
  """
  return env.scene["robot"].data.joint_vel.abs().mean(dim=1)


def joint_course_l1(env: ManagerBasedRlEnv, command_name: str, cap: float | None = None, stream: str = "always",
                    grace: float = 0.0) -> torch.Tensor:
  """Mean absolute distance of the joint angles to the course (the reference's joint angles), rad per joint, the
  same price for every joint. The pose kernels are wide (0.3 m, 0.4 rad): with them alone a tracker earns more by
  reacting to a hit with its arms than by keeping them on the course (measured, gates 11-12 01:20). This term prices
  the intervention itself; which joints must move after a hit is left to the learner."""
  term = _term(env, command_name)
  distance = (term.robot_joint_pos - term.joint_pos).abs().mean(dim=1)
  # cap: an untrained policy is far from the course; an unbounded penalty there would make ending the episode pay.
  distance = distance if cap is None else distance.clamp(max=cap)
  return distance * course_is_trusted(env, command_name, stream, grace)


def posture_course_l1(env: ManagerBasedRlEnv, command_name: str, cap: float | None = None, stream: str = "always",
                      grace: float = 0.0) -> torch.Tensor:
  """Mean angle (rad per tracked link) between how the robot's links stand in space and how the course's do: the
  posture price measured against gravity and the ground instead of joint by joint.

  The course is turned to the robot's heading (as for the tracking terms). Every link but the feet is compared as it
  hangs in space (against gravity); a foot's course orientation is first laid on the ground under that foot (the ray
  sensor's surface normal), so a sole flat on a slope is on its course. On a flat floor the feet are compared like
  the rest. Why: the joint-angle price (joint_course_l1) charges what a slope itself demands (20 deg of ankle on a
  20 deg slope, for as long as the robot is on it: up to 162 undiscounted per episode at the cap against 60 for a
  fall; at gamma=.99 the 500-step cap cost is 32.187, so this comparison alone does not prove a failure incentive).
  The terrain adaptation is still priced by joint error (gates 11-14 03:2x). Measured against gravity and the
  surface, an upright body with its soles on the slope pays nothing."""
  from mjlab.utils.lab_api.math import quat_error_magnitude, quat_mul
  term = _term(env, command_name)
  want, have = term.body_quat_relative_w, term.robot_body_quat_w
  if "feet_ground_height" in env.scene.sensors:
    normals = env.scene["feet_ground_height"].data.normals_w.reshape(env.num_envs, 2, -1, 3).mean(dim=2)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    angle = torch.acos(normals[..., 2].clamp(-1.0, 1.0))                              # tilt of the ground under each foot
    axis = torch.stack([-normals[..., 1], normals[..., 0], torch.zeros_like(angle)], dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    tilt = torch.cat([torch.cos(0.5 * angle).unsqueeze(-1), axis * torch.sin(0.5 * angle).unsqueeze(-1)], dim=-1)
    fresh = (env.episode_length_buf < 1).view(-1, 1, 1)                               # the rays are a step old at a reset
    tilt = torch.where(fresh, torch.tensor([1.0, 0.0, 0.0, 0.0], device=tilt.device).expand_as(tilt), tilt)
    if not hasattr(term, "_foot_bodies"):
      term._foot_bodies = [list(term.cfg.body_names).index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    want = want.clone()
    want[:, term._foot_bodies] = quat_mul(tilt, want[:, term._foot_bodies])
  distance = quat_error_magnitude(want.reshape(-1, 4), have.reshape(-1, 4)).reshape(env.num_envs, -1).mean(dim=1)
  distance = distance if cap is None else distance.clamp(max=cap)
  return distance * course_is_trusted(env, command_name, stream, grace)


def course_travel_l1(env: ManagerBasedRlEnv, command_name: str, yaw_scale: float = 0.3, stream: str = "always",
                     grace: float = 0.0) -> torch.Tensor:
  """How far the robot's own travel is from the course's: the root twists (running means over the last 0.4 s of
  what each did, each in its own heading frame) apart in m/s, plus `yaw_scale` m per rad of yaw rate. The joint-course price charges the posture only, and
  under a high price a policy mimed a sideways course's joints while covering 0.67 to 0.75 of its distance with the
  heading drifting 40 to 50 degrees (E94; the goal rewards that would correct it are worth less than 0.01 rad of
  joint distance there). This charges not going where the course goes, with the same trust as the posture price, so
  raising the price list makes the policy follow more closely instead of less."""
  term = _term(env, command_name)
  gap = term.twist_mean - term.course_twist_mean           # not table_twist: that is the mean of the 0.4 s to come
  return (torch.norm(gap[:, :2], dim=1) + yaw_scale * gap[:, 2].abs()) * course_is_trusted(env, command_name, stream, grace)


def tracking_is_judged(env: ManagerBasedRlEnv, grace: float) -> torch.Tensor:
  """At grace=0 tracking is judged at every instant, without reading the hit clock.

  Positive grace is historical evaluation only: a recovery step lifts a foot or lowers the pelvis by more than
  the tracking tolerances without any fall, and with bursts in the disturbances those endings (not falls) were what
  ended nearly every episode (run of 11-12 03:42: foot-height endings 27 -> 141 per window, tracking 128 -> 33 of 134).
  A real fall ends the episode at any time (`fallen`)."""
  if grace <= 0.0:
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  clock = getattr(env, "hit_clock", None)
  if clock is None:
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  return clock > grace


def judged_anchor_z(env, command_name: str, threshold: float, grace: float) -> torch.Tensor:
  from mjlab.tasks.tracking import mdp as stock
  return stock.bad_anchor_pos_z_only(env, command_name=command_name, threshold=threshold) & tracking_is_judged(env, grace)


def judged_anchor_xy(env, command_name: str, threshold: float, grace: float) -> torch.Tensor:
  return anchor_xy_error(env, command_name=command_name, threshold=threshold) & tracking_is_judged(env, grace)


def judged_anchor_rot(env, command_name: str, threshold: float, grace: float) -> torch.Tensor:
  return anchor_rotation_error(env, command_name=command_name, threshold=threshold) & tracking_is_judged(env, grace)


def judged_body_z(env, command_name: str, threshold: float, body_names: tuple[str, ...], grace: float) -> torch.Tensor:
  from mjlab.tasks.tracking import mdp as stock
  function = body_z_error if "_r1wb_read_cache" in env.__dict__ else stock.bad_motion_body_pos_z_only
  return (function(env, command_name=command_name, threshold=threshold, body_names=body_names)
          & tracking_is_judged(env, grace))


def fallen(env: ManagerBasedRlEnv, min_height: float = 0.35, max_tilt: float = 0.9) -> torch.Tensor:
  """Physical fall: the pelvis is low above the surface the robot stands on, or tilted far (the criterion of the
  stream environments). On a flat floor that surface is the start level."""
  from r1wb import world
  robot = env.scene["robot"]
  height = world.standing_height(env)
  gravity = robot.data.projected_gravity_b
  tilt = torch.atan2(torch.norm(gravity[:, :2], dim=1), -gravity[:, 2])
  return (height < min_height) | (tilt > max_tilt)


# -- events -----------------------------------------------------------------------------------------------------

class push_unless_calm:
  """Root velocity kicks at random intervals. A fixed share of environments is never pushed (clean per-clip
  statistics), and the kick is scaled down while the reference pelvis is low (deep postures have little margin)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    count = env.num_envs
    self.calm = torch.rand(count, device=env.device) < cfg.params.get("calm_fraction", 0.2)

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str,
               velocity_range: dict[str, tuple[float, float]], calm_fraction: float = 0.2,
               low_height: float = 0.5, low_scale: float = 0.4) -> None:
    del calm_fraction
    robot = env.scene["robot"]
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids[~self.calm[env_ids]]
    if len(env_ids) == 0:
      return
    term = _term(env, command_name)
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    low = torch.tensor([velocity_range.get(k, (0.0, 0.0))[0] for k in keys], device=env.device)
    high = torch.tensor([velocity_range.get(k, (0.0, 0.0))[1] for k in keys], device=env.device)
    kick = low + (high - low) * torch.rand(len(env_ids), 6, device=env.device)
    scale = torch.where(term.anchor_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2] < low_height, low_scale, 1.0)
    velocity = robot.data.root_link_vel_w[env_ids] + kick * scale.unsqueeze(1)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def uniform_equilibrium_residual(uniform: torch.Tensor, interval_s: tuple[float, float]) -> torch.Tensor:
  """Inverse CDF of the stationary residual of T~U(a,b), for supplied draws uniform in [0,1].

  Density is P(T>r)/E[T], not U(0,b). This pure transform consumes no RNG. It describes an independent observation
  time in an ordinary renewal process; it does not reproduce burst- or failure-conditioned residuals.
  """
  low, high = interval_s
  if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low <= high and high > 0.0):
    raise ValueError("equilibrium reset requires a finite interval 0 <= low <= high with high > 0")
  mean = 0.5 * (low + high)
  tail = high - torch.sqrt((1.0 - uniform) * (high - low) * (high + low))
  return torch.where(uniform <= low / mean, mean * uniform, tail)


class impulse_unless_calm:
  """Force impulses at a point of one body from a horizontal direction drawn uniformly: the disturbance of the
  evaluation (momentum in N s) instead of a root-velocity kick. The impulse is spread over a duration drawn
  log-uniformly from `duration_s` (one policy step = a ball, tenths of a second = a shove). A fixed share of
  environments is never hit, hits are scaled down while the reference pelvis is low, and `ramp_steps` opens the
  impulse range from 20% to 100% over that many environment steps (training from scratch). Use with mode="step".
  calm_streams: environments on a command stream are never hit (a hit can only be priced where the reference is a
  course; the reaction is then learned on the clips alone).
  height / side: where the impulse acts relative to the body's centre of mass, m. A number = always there; a
  (low, high) pair = drawn per hit (up and down the body: thigh to head; sideways of the line of the force: a hit on
  a shoulder turns the robot). A ball does not choose the trunk's centre.
  reset_on_episode: opt in to independent episode schedules and clear unfinished hits on reset. Defaults to False
  to retain the disturbance distribution and RNG sequence of existing tasks; set it in the event's cfg.params.
  reset_wait_mode: "fresh" retains v5's full-interval reset wait; "equilibrium" draws its ordinary-renewal residual.
  Only reset_on_episode=True uses this mode. Construction and subsequent inter-hit/burst schedules are unchanged."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self._reset_on_episode = params.get("reset_on_episode", False)
    self._interval_s = params["interval_s"]
    self._reset_wait_mode = params.get("reset_wait_mode", "fresh")
    if self._reset_on_episode and self._reset_wait_mode not in ("fresh", "equilibrium"):
      raise ValueError("reset_wait_mode must be fresh or equilibrium")
    self.robot = env.scene["robot"]
    self.body = [self.robot.body_names.index(params.get("body", "torso_link"))]
    count, device = env.num_envs, env.device
    self.calm = torch.rand(count, device=device) < params.get("calm_fraction", 0.2)
    low, high = params["interval_s"]
    self.wait = low + (high - low) * torch.rand(count, device=device)
    self.left = torch.zeros(count, dtype=torch.long, device=device)        # policy steps the running hit still lasts
    self.force = torch.zeros(count, 3, device=device)
    self.torque = torch.zeros(count, 3, device=device)
    self.written = False                                                   # a non-zero wrench is in the simulator
    self.burst_left = torch.zeros(count, dtype=torch.long, device=device)   # further hits of the running burst
    self.burst_angle = torch.zeros(count, device=device)
    self.streams_marked = False
    env.hit_clock = torch.full((count,), 1.0e3, device=device)              # seconds since the last hit (privileged)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Called by EventManager after episode reset, including for mode="step" terms."""
    if not self._reset_on_episode:
      return
    if env_ids is None:
      env_ids = torch.arange(len(self.wait), device=self.wait.device)
    if len(env_ids) == 0:
      return
    low, high = self._interval_s
    uniform = torch.rand(len(env_ids), device=self.wait.device)
    self.wait[env_ids] = (uniform_equilibrium_residual(uniform, self._interval_s)
                        if self._reset_wait_mode == "equilibrium" else low + (high - low) * uniform)
    self.left[env_ids] = 0
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.burst_left[env_ids] = 0
    self.burst_angle[env_ids] = 0.0
    self._env.hit_clock[env_ids] = 1.0e3
    zeros = self.force[env_ids].unsqueeze(1)
    self.robot.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids, body_ids=self.body)
    # A non-reset env can still have a final one-step wrench in the simulator although its left is already zero.
    # Keep the shared flag on partial reset so the next call clears that wrench. Manager env_ids are unique.
    if len(env_ids) == len(self.wait):
      self.written = False

  def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str,
               impulse_range: tuple[float, float], interval_s: tuple[float, float], body: str = "torso_link",
               height: float | tuple[float, float] = 0.1, calm_fraction: float = 0.2, low_height: float = 0.5,
               low_scale: float = 0.4, side: float | tuple[float, float] = 0.0,
               duration_s: tuple[float, float] = (0.02, 0.02), ramp_steps: int = 0, burst_prob: float = 0.0,
               burst_hits: tuple[int, int] = (4, 12), burst_gap_s: tuple[float, float] = (0.08, 0.25),
               burst_impulse: tuple[float, float] = (8.0, 16.0), calm_streams: bool = False,
               reset_on_episode: bool = False, reset_wait_mode: str = "fresh") -> None:
    """burst_prob: share of the events that open a burst: `burst_hits` one-step hits of `burst_impulse` N s from
    about the same side, `burst_gap_s` apart (balls in a row)."""
    del env_ids, body, calm_fraction, reset_on_episode, reset_wait_mode
    device, dt = env.device, env.step_dt
    if calm_streams and not self.streams_marked:      # the command term exists only after the managers are built
      streams = getattr(_term(env, command_name), "stream_env", None)
      if streams is not None:
        self.calm = self.calm | streams
      self.streams_marked = True
    self.wait -= dt
    env.hit_clock = torch.where(env.episode_length_buf <= 1, torch.full_like(env.hit_clock, 1.0e3), env.hit_clock + dt)
    env.hit_clock = torch.where(self.left > 0, torch.zeros_like(env.hit_clock), env.hit_clock)
    start = (self.wait <= 0.0) & ~self.calm & (self.left == 0)
    if start.any():
      ids = start.nonzero().squeeze(-1)
      count = len(ids)
      share = 1.0 if ramp_steps <= 0 else min(1.0, 0.2 + 0.8 * env.common_step_counter / ramp_steps)
      low, high = impulse_range
      impulse = (low + (high - low) * torch.rand(count, device=device)) * share
      term = _term(env, command_name)
      impulse = impulse * torch.where(term.anchor_pos_w[ids, 2] - env.scene.env_origins[ids, 2] < low_height, low_scale, 1.0)
      short, long = duration_s
      duration = short * torch.exp(torch.rand(count, device=device) * math.log(max(long / short, 1.0)))
      steps = torch.round(duration / dt).clamp_min(1).long()
      angle = (torch.rand(count, device=device) * 2.0 - 1.0) * math.pi
      wait = interval_s[0] + (interval_s[1] - interval_s[0]) * torch.rand(count, device=device)
      if burst_prob > 0.0:
        follow = self.burst_left[ids] > 0
        opening = ~follow & (torch.rand(count, device=device) < burst_prob)
        ball = follow | opening
        size = (burst_impulse[0] + (burst_impulse[1] - burst_impulse[0]) * torch.rand(count, device=device)) * share
        impulse = torch.where(ball, size, impulse)
        steps = torch.where(ball, torch.ones_like(steps), steps)
        angle = torch.where(follow, self.burst_angle[ids] + (torch.rand(count, device=device) * 2.0 - 1.0) * 0.3, angle)
        self.burst_angle[ids] = angle
        further = torch.randint(burst_hits[0] - 1, burst_hits[1], (count,), device=device)
        self.burst_left[ids] = torch.where(follow, self.burst_left[ids] - 1, torch.where(opening, further, torch.zeros_like(further)))
        gap = burst_gap_s[0] + (burst_gap_s[1] - burst_gap_s[0]) * torch.rand(count, device=device)
        wait = torch.where(self.burst_left[ids] > 0, gap, wait)
      force = torch.stack([angle.cos(), angle.sin(), torch.zeros_like(angle)], dim=1) * (impulse / (steps * dt)).unsqueeze(1)
      def drawn(value):
        if isinstance(value, (tuple, list)):
          return value[0] + (value[1] - value[0]) * torch.rand(count, device=device)
        return torch.full((count,), float(value), device=device)

      arm = torch.zeros(count, 3, device=device)
      arm[:, 2] = drawn(height)
      offset = drawn(side)                                   # across the line of the force, in the horizontal plane
      arm[:, 0], arm[:, 1] = -angle.sin() * offset, angle.cos() * offset
      self.force[ids], self.torque[ids], self.left[ids] = force, torch.cross(arm, force, dim=1), steps
      self.wait[ids] = wait
      env.hit_clock[ids] = 0.0
    acting = self.left > 0
    if not (acting.any() or self.written):
      return
    # The wrench stays in the simulator until it is overwritten: the step after the last hit writes zeros.
    active = acting.float().unsqueeze(1)
    self.robot.write_external_wrench_to_sim((self.force * active).unsqueeze(1), (self.torque * active).unsqueeze(1),
                                            body_ids=self.body)
    self.written = bool(acting.any())
    self.left = (self.left - 1).clamp_min(0)
