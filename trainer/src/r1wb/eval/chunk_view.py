"""The course as System 1 would receive it from a System 2 that re-plans in chunks.

The command stream's planner runs every control step inside the environment. A System 2 (a large model) cannot: it
emits a chunk of the coming reference every `period` steps, the chunk reaches System 1 `latency` steps after the state
it was planned from, it covers `horizon` frames, and its content is not exact. This view stands between the live
course and the tracking policy's observation and applies exactly those four things, so a System 1 can be measured
under them before any System 2 exists:

  period    control steps between two chunks (50 Hz steps)
  latency   steps between planning a chunk and System 1 using it
  horizon   frames a chunk covers; the reference preview is cut there (as it is at a clip's end)
  noise     "jump": every chunk carries its own constant joint-angle error (the reference jumps at every new chunk,
            also while standing); "drift": one slowly wandering joint-angle error (time constant 0.5 s), continuous
            across chunks
  sigma     rad, standard deviation of that error per joint

A chunk here is a run of library frames (the planner's own continuation of its clip at planning time): the view
reads the library at the chunk's frame instead of the planner's current frame. With period 1, latency 0, a horizon
beyond the longest preview and sigma 0 it returns the live course.

dense=True: the chunk is turned into reference frames (r1wb.s2.frames: joint angles, torso height and tilt, torso
step) and the observation is rebuilt from those frames alone, the way it has to be when the reference comes out of
a model. `transform` (frames (n, horizon, 30) -> same shape) is applied to every new chunk: a tokenizer's
encode-decode round trip, for instance.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat

from r1wb.s2 import frames as s2


class ChunkView:
  def __init__(self, env, term, period: int = 10, latency: int = 5, horizon: int = 50, noise: str = "jump",
               sigma: float = 0.0, seed: int = 0, dense: bool = False, transform=None, lead: int = 1):
    assert latency <= period, "one chunk in flight at a time"
    self._env, self._term = env, term
    self.period, self.latency, self.horizon, self.noise, self.sigma = period, latency, horizon, noise, sigma
    count, device = env.num_envs, env.device
    self.dense, self.transform, self.lead = dense or transform is not None, transform, lead
    if self.dense:
      end = term.clip_end[term.frame_clip]
      self._table = s2.library_frames(term.motion, term.motion_anchor_body_index, end)
      self._frames = torch.zeros(count, lead + horizon, s2.FRAME_DIM, device=device)   # [frames before, the run]
      model = env.sim.mj_model
      self._torso_com = torch.tensor(model.body("robot/" + term.cfg.anchor_body_name).ipos, dtype=torch.float, device=device)
      self._reference = None
    self._generator = torch.Generator(device=device).manual_seed(seed)
    self._clock = 0
    zeros = lambda *shape, dtype=torch.long: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
    self._base = {key: zeros(count) for key in ("frame", "last", "born")}       # the chunk System 1 uses
    self._hold = zeros(count, dtype=torch.bool)
    self._goal = torch.zeros(count, term.goal.shape[1], device=device)
    self._offset = torch.zeros(count, term.motion.joint_pos.shape[1], device=device)
    self._pending = None
    self.chunks = 0
    self.env = _EnvView(env, self)
    self._plan(torch.arange(count, device=device), activate=True)

  # -- chunks ---------------------------------------------------------------------------------------------------

  def _snapshot(self):
    term = self._term
    frame = term.time_steps.clone()
    last = torch.minimum(frame + self.horizon - 1, term.clip_end[term.clip_ids])
    return dict(frame=frame, last=last, hold=term.hold.clone(), goal=term.goal.clone(), born=self._clock)

  def _activate(self, chunk, ids):
    self._base["frame"][ids], self._base["last"][ids] = chunk["frame"][ids], chunk["last"][ids]
    self._base["born"][ids] = chunk["born"]
    self._hold[ids], self._goal[ids] = chunk["hold"][ids], chunk["goal"][ids]
    if self.dense:
      ahead = torch.arange(-self.lead, self.horizon, device=ids.device).unsqueeze(0)
      ahead = torch.where(chunk["hold"][ids].unsqueeze(1), torch.zeros_like(ahead), ahead)
      first = self._term.clip_start[self._term.frame_clip[chunk["frame"][ids]]].unsqueeze(1)
      index = torch.minimum(chunk["frame"][ids].unsqueeze(1) + ahead, chunk["last"][ids].unsqueeze(1))
      index = torch.maximum(index, first)
      run = self._table[index].clone()
      stays = torch.cat([index[:, 1:] == index[:, :-1], torch.ones(len(ids), 1, dtype=torch.bool, device=ids.device)], dim=1)
      run[:, :, s2.STEP] = run[:, :, s2.STEP] * (~stays).float().unsqueeze(2)       # a frame that is held does not travel
      if self.transform is not None:
        run = self.transform(run)
      self._frames[ids] = run
      self._reference = None
    if self.noise == "jump" and self.sigma > 0.0:
      self._offset[ids] = self.sigma * torch.randn(len(ids), self._offset.shape[1], generator=self._generator,
                                                   device=self._offset.device)

  def _plan(self, ids, activate: bool = False):
    chunk = self._snapshot()
    if activate:
      self._activate(chunk, ids)
    else:
      self._pending = chunk

  def update(self) -> None:
    """Once per control step, after the environment advanced the planner and before the policy is evaluated."""
    env = self._env
    self._clock += 1
    everyone = torch.arange(env.num_envs, device=env.device)
    if self._clock % self.period == 0:
      self._plan(everyone)
      self.chunks += 1
    if self._pending is not None and self._clock - self._pending["born"] >= self.latency:
      self._activate(self._pending, everyone)
      self._pending = None
    restarted = (env.episode_length_buf <= 1).nonzero(as_tuple=False).flatten()     # a new episode starts on a new chunk
    if len(restarted):
      self._plan(restarted, activate=True)
    if self.noise == "drift" and self.sigma > 0.0:
      rate = env.step_dt / 0.5
      self._offset += -rate * self._offset + self.sigma * math.sqrt(2.0 * rate) * torch.randn(
        self._offset.shape, generator=self._generator, device=self._offset.device)

  # -- the course System 1 sees ---------------------------------------------------------------------------------

  def _frame(self) -> torch.Tensor:
    age = torch.where(self._hold, torch.zeros_like(self._base["born"]), self._clock - self._base["born"])
    return torch.minimum(self._base["frame"] + age, self._base["last"])

  def _dense(self):
    """(reference, index into each environment's run)."""
    if self._reference is None:
      self._reference = s2.DenseReference(self._frames, self._env.step_dt, velocity_point=self._torso_com, lead=self.lead)
    return self._reference, (self._clock - self._base["born"]).clamp(min=0, max=self.horizon - 1)

  @property
  def command(self) -> torch.Tensor:
    if self.dense:
      reference, index = self._dense()
      vector = reference.command(index, self._term.future)
      width = self._offset.shape[1]
      vector[:, :width] += self._offset                        # the same joint-angle error on the current frame
      start = 2 * width
      for _ in self._term.future:                              # ... and on every preview frame
        vector[:, start:start + width] += self._offset
        start += width + 9
      return vector
    term, motion, frame = self._term, self._term.motion, self._frame()
    anchor = term.motion_anchor_body_index
    parts = [motion.joint_pos[frame] + self._offset, motion.joint_vel[frame]]
    now_pos, now_quat = motion.body_pos_w[frame, anchor], motion.body_quat_w[frame, anchor]
    heading = yaw_quat(now_quat)
    for step in term.future:
      index = torch.minimum(frame + step, self._base["last"])
      parts.append(motion.joint_pos[index] + self._offset)
      parts.append(quat_apply_inverse(heading, motion.body_pos_w[index, anchor] - now_pos))
      relative = matrix_from_quat(quat_mul(quat_inv(now_quat), motion.body_quat_w[index, anchor]))
      parts.append(relative[..., :2].reshape(len(frame), 6))
    return torch.cat(parts, dim=1)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    if self.dense:                                             # only the height is read (re-anchored observation)
      reference, index = self._dense()
      position = self._env.scene.env_origins.clone()
      position[:, 2] += reference.anchor_height(index)
      return position
    return self._term.motion.body_pos_w[self._frame(), self._term.motion_anchor_body_index] + self._env.scene.env_origins

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    if self.dense:                                             # the reference's tilt at the robot's own heading
      reference, index = self._dense()
      return quat_mul(yaw_quat(self._term.robot_anchor_quat_w), reference.anchor_tilt(index))
    return self._term.motion.body_quat_w[self._frame(), self._term.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    if self.dense:
      reference, index = self._dense()
      return quat_apply(yaw_quat(self._term.robot_anchor_quat_w), reference.anchor_velocity(index)[0])
    return self._term.motion.body_lin_vel_w[self._frame(), self._term.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    if self.dense:
      reference, index = self._dense()
      return quat_apply(yaw_quat(self._term.robot_anchor_quat_w), reference.anchor_velocity(index)[1])
    return self._term.motion.body_ang_vel_w[self._frame(), self._term.motion_anchor_body_index]

  @property
  def goal(self) -> torch.Tensor:
    return self._goal

  def __getattr__(self, name):                       # everything else is the live term (robot state, goal lag, ...)
    return getattr(self._term, name)


class _EnvView:
  """The environment with the command term replaced by the chunk view (for the observation functions)."""

  def __init__(self, env, view):
    self._env = env
    self.command_manager = SimpleNamespace(get_term=lambda name: view)

  def __getattr__(self, name):
    return getattr(self._env, name)
