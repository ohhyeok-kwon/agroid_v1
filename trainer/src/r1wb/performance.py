"""Reuse read-only state within one manager call, never across simulation phases.

Physics, resets, command advancement and ray updates stay outside these scopes.
In particular, pre-reset rewards must not share a cache with post-reset observations.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import time


def record_iteration_times(runner, path):
  """Include logger/checkpoint overhead in successive rank-zero wall intervals."""
  native = runner.logger.log
  previous = None

  @wraps(native)
  def log(*args, **kwargs):
    nonlocal previous
    result = native(*args, **kwargs)
    now = time.perf_counter()
    row = {"iteration": kwargs["it"], "collection_s": kwargs["collect_time"],
           "learning_s": kwargs["learn_time"],
           "wall_interval_s": None if previous is None else now - previous}
    previous = now
    with path.open("a") as output:
      output.write(json.dumps(row) + "\n")
    return result
  runner.logger.log = log


class ReadCache:
  def __init__(self):
    self.active = False
    self.values = {}
    self.indices = {}

  def read(self, key, compute):
    if not self.active:
      return compute()
    if key not in self.values:
      self.values[key] = compute()
    return self.values[key]

  @contextmanager
  def scope(self):
    if self.active:
      yield
      return
    self.active = True
    try:
      yield
    finally:
      self.active = False
      self.values.clear()

  def wrap(self, function):
    @wraps(function)
    def call(*args, **kwargs):
      with self.scope():
        return function(*args, **kwargs)
    return call


def cached_read(function):
  """Cache a pure env/command query only inside an explicitly read-only scope."""
  @wraps(function)
  def read(owner):
    env = getattr(owner, "_env", owner)
    cache = env.__dict__.get("_r1wb_read_cache")
    if cache is None or not cache.active:
      return function(owner)
    return cache.read((id(owner), function), lambda: function(owner))
  return read


def device_indices(env, indices):
  """Keep static gather indices on device; Python lists otherwise copy each call."""
  cache = env.__dict__.get("_r1wb_read_cache")
  if cache is None:
    return indices
  import torch
  key = tuple(indices)
  if key not in cache.indices:
    cache.indices[key] = torch.tensor(key, dtype=torch.long, device=env.device)
  return cache.indices[key]


def fixed_bincount(indices, size):
  """Integer histogram with known bounds, without bincount's host size query."""
  import torch
  return torch.zeros(size, dtype=torch.long, device=indices.device).scatter_add_(
    0, indices, torch.ones_like(indices))


def yaw_rotation(rot_mat):
  """RayCastSensor's yaw extraction with the same singularity fallback on device."""
  import torch
  x_proj = rot_mat[:, :, 0].clone()
  x_proj[:, 2] = 0
  x_norm = x_proj.norm(dim=1)
  singular = x_norm < 0.1
  y_proj = rot_mat[:, :, 1].clone()
  y_proj[:, 2] = 0
  y_proj = y_proj / y_proj.norm(dim=1).clamp(min=1e-6).unsqueeze(-1)
  x_from_y = torch.zeros_like(y_proj)
  x_from_y[:, 0] = y_proj[:, 1]
  x_from_y[:, 1] = -y_proj[:, 0]
  x_proj = torch.where(singular.unsqueeze(-1), x_from_y, x_proj)
  x_norm = torch.where(singular, torch.ones_like(x_norm), x_norm)
  x_proj = x_proj / x_norm.clamp(min=1e-6).unsqueeze(-1)
  result = torch.zeros_like(rot_mat)
  result[:, 0, 0] = x_proj[:, 0]
  result[:, 1, 0] = x_proj[:, 1]
  result[:, 0, 1] = -x_proj[:, 1]
  result[:, 1, 1] = x_proj[:, 0]
  result[:, 2, 2] = 1
  return result


def install_read_cache(env):
  """Install on a training env after construction. No global mjlab patches."""
  from mjlab.entity.data import EntityData

  if "_r1wb_read_cache" in env.__dict__:
    return env._r1wb_read_cache
  cache = env._r1wb_read_cache = ReadCache()

  def property_reader(name):
    original = getattr(EntityData, name).fget
    return property(lambda data: cache.read((id(data), name), lambda: original(data)))

  # A per-env adapter shares the original tensors and write methods. Only derived
  # reads are memoized, and only while the managers below cannot change the state.
  names = ("joint_pos", "joint_pos_biased", "joint_vel", "joint_acc", "root_link_pose_w",
           "root_link_vel_w", "body_link_pose_w", "body_link_vel_w", "body_link_ang_vel_w",
           "projected_gravity_b", "root_link_lin_vel_b", "root_link_ang_vel_b")
  data_type = type("ReadCachedEntityData", (EntityData,), {name: property_reader(name) for name in names})
  robot = env.scene["robot"]
  data = data_type.__new__(data_type)
  data.__dict__.update(robot.data.__dict__)
  robot._data = data
  for manager in (env.termination_manager, env.reward_manager, env.observation_manager, env.metrics_manager):
    manager.compute = cache.wrap(manager.compute)
  for term in env.command_manager._terms.values():
    term._update_metrics = cache.wrap(term._update_metrics)
    if hasattr(term, "update_relative_body_poses"):
      term.update_relative_body_poses = cache.wrap(term.update_relative_body_poses)
  from mjlab.tasks.tracking import mdp as stock
  from r1wb.tracking import mdp
  replacements = {
    stock.motion_relative_body_position_error_exp: mdp.body_position_error_exp,
    stock.motion_relative_body_orientation_error_exp: mdp.body_orientation_error_exp,
    stock.bad_motion_body_pos_z_only: mdp.body_z_error,
  }
  for manager in (env.reward_manager, env.termination_manager):
    for cfg in manager._term_cfgs:
      cfg.func = replacements.get(cfg.func, cfg.func)
  for sensor in env.scene.sensors.values():
    if getattr(sensor.cfg, "ray_alignment", None) == "yaw":
      sensor._extract_yaw_rotation = yaw_rotation
  return cache


def install_fast_resets(env):
  """Use device scalar fills and one transfer for termination counters.

  Keep the managers' reduction/reset order and the command's random draws.
  Adapt only this environment; unsupported index forms use the native methods.
  """
  import torch

  if env.__dict__.get("_r1wb_fast_resets", False):
    return
  env._r1wb_fast_resets = True

  def rows(ids):
    return isinstance(ids, torch.Tensor) and ids.ndim == 1 and ids.dtype in (torch.int32, torch.int64)

  reward, termination = env.reward_manager, env.termination_manager
  native_reward, native_termination = reward.reset, termination.reset

  def reset_rewards(env_ids=None):
    if not rows(env_ids):
      return native_reward(env_ids)
    extras = {}
    for name, value in reward._episode_sums.items():
      extras["Episode_Reward/" + name] = value[env_ids].mean() / env.max_episode_length_s
      value.index_fill_(0, env_ids, 0.0)
    for cfg in reward._class_term_cfgs:
      cfg.func.reset(env_ids=env_ids)
    return extras

  def reset_terminations(env_ids=None):
    if not rows(env_ids):
      return native_termination(env_ids)
    values = [torch.count_nonzero(value[env_ids]) for value in termination._term_dones.values()]
    counts = torch.stack(values).tolist() if values else []
    extras = {"Episode_Termination/" + name: count for name, count in zip(termination._term_dones, counts)}
    for cfg in termination._class_term_cfgs:
      cfg.func.reset(env_ids=env_ids)
    return extras

  reward.reset, termination.reset = reset_rewards, reset_terminations
  data = env.scene["robot"].data
  native_clear = data.clear_state

  def clear_state(env_ids=None):
    if not rows(env_ids):
      return native_clear(env_ids)
    if data.is_actuated:
      for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target", "tendon_len_target",
                   "tendon_vel_target", "tendon_effort_target", "site_effort_target"):
        getattr(data, name).index_fill_(0, env_ids, 0.0)

  data.clear_state = clear_state


def install_reference_tables(env):
  """Tabulate the immutable reference preview, using the original batch shape.

  LibraryMotionCommand's training paths keep clip_ids aligned with frame_clip.
  Robot state, ground offsets, command streams and sensor history are not tables.
  """
  import copy
  import torch
  from r1wb.tracking.command import LibraryMotionCommand

  for command in env.command_manager._terms.values():
    if not isinstance(command, LibraryMotionCommand) or type(command).command is not LibraryMotionCommand.command:
      continue
    if "_r1wb_reference_table" in command.__dict__ or env.num_envs < 256:
      continue
    shadow = copy.copy(command)
    count, batch = command.motion.time_step_total, env.num_envs
    with torch.inference_mode():
      # Warm TorchScript before storing values; its profiling executor can change
      # the first invocation's rounding. No random operations occur in command.
      for _ in range(3):
        preview = shadow.command
      table = preview.new_empty((count, preview.shape[1]))
      for start in range(0, count, batch):
        shadow.time_steps = torch.arange(start, start + batch, device=env.device).clamp_max(count - 1)
        shadow.clip_ids = shadow.frame_clip[shadow.time_steps]
        stop = min(start + batch, count)
        table[start:stop] = shadow.command[:stop - start]
    command._r1wb_reference_table = table
