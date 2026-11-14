"""The whole-body teacher task (r1wb.teacher_task) behind RLinf's simulator-environment interface.

RLinf's environment worker owns the environment and exchanges observations and actions with a separate rollout
worker (policy inference) and actor worker (learner). The interface is the one of rlinf.envs.sim.genesis: batched
torch tensors, `reset`, `step`, `chunk_step`, episode metrics under infos["episode"], and on a reset the terminal
observation under infos["final_observation"].

mjlab resets finished environments inside `step` and returns the first observation of the new episode, so the true
terminal observation is not available: the last observation before the episode ended is handed over in its place
(what rsl_rl's bootstrap on time-outs uses as well).
"""
from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch


class R1TaskEnv(gym.Env):
  metadata: dict[str, Any] = {"render_modes": []}

  def __init__(self, cfg, num_envs: int, seed_offset: int, total_num_processes: int, worker_info: Any,
               record_metrics: bool = True) -> None:
    import warp as wp
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
      wp.config.ptx_target_arch = 90
      wp.config.cuda_output = "ptx"
    from mjlab.envs import ManagerBasedRlEnv
    from r1wb.tasks import TASKS

    self.cfg = cfg
    self.num_envs = num_envs
    self.seed = int(cfg.seed) + int(seed_offset)
    self.seed_offset, self.total_num_processes, self.worker_info = seed_offset, total_num_processes, worker_info
    self.group_size = int(cfg.group_size)
    self.num_group = num_envs // self.group_size
    self.auto_reset = bool(cfg.auto_reset)
    self.ignore_terminations = bool(cfg.ignore_terminations)
    self.use_fixed_reset_state_ids = bool(cfg.use_fixed_reset_state_ids)
    self.max_episode_steps = int(cfg.max_episode_steps)
    self.video_cfg = cfg.video_cfg
    self._is_start = True
    self._generator = np.random.default_rng(seed=self.seed)
    task = str(cfg.init_params.task_name)
    env_cfg = TASKS[task]["env"]()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = self.seed
    self.device = torch.device("cuda:0")
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    self.obs_group = str(cfg.init_params.get("obs_group", "actor"))
    # Where the tensors handed to the other workers live. RLinf passes GPU tensors between workers on the same GPU
    # through CUDA IPC, which needs a permission (pidfd_getfd) that the GPU server's container does not give:
    # "cpu" sends them as host tensors instead.
    self.ret_device = torch.device(str(cfg.init_params.get("ret_device", "cuda:0")))
    self.action_dim = int(self.env.action_manager.total_action_dim)
    self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
    self.record_metrics = record_metrics
    self.returns = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
    self.failed_once = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    self._states = None
    self.update_reset_state_ids()

  # -- what the environment worker reads ---------------------------------------------------------------------------
  @property
  def elapsed_steps(self) -> torch.Tensor:
    return self._elapsed_steps

  @property
  def total_num_group_envs(self) -> int:
    return int(np.iinfo(np.uint32).max // 2)

  @property
  def is_start(self) -> bool:
    return self._is_start

  @is_start.setter
  def is_start(self, value: bool) -> None:
    self._is_start = value

  def update_reset_state_ids(self) -> None:
    ids = torch.from_numpy(self._generator.integers(0, self.total_num_group_envs, size=(self.num_group,), dtype=np.int64))
    self.reset_state_ids = ids.repeat_interleave(self.group_size).to(self.device)

  def _wrap(self, states: torch.Tensor) -> dict[str, Any]:
    return {"states": states.to(self.ret_device), "task_descriptions": [""] * self.num_envs}

  def _episode(self) -> dict[str, torch.Tensor]:
    length = self._elapsed_steps.clamp(min=1).float()
    out = {"return": self.returns.clone(), "episode_len": self._elapsed_steps.clone(), "reward": self.returns / length,
           "success_once": ~self.failed_once}
    return {key: value.to(self.ret_device) for key, value in out.items()}

  # -- gymnasium ---------------------------------------------------------------------------------------------------
  def reset(self, env_idx=None, *, seed: Optional[int] = None, options: Optional[dict] = None):
    if env_idx is None:
      observation, _ = self.env.reset()
      self._states = observation[self.obs_group]
      self._elapsed_steps.zero_(); self.returns.zero_(); self.failed_once.zero_()
    # a reset of single environments happens inside mjlab's step; nothing to do here
    return self._wrap(self._states), {}

  def step(self, actions, auto_reset: bool = True):
    if isinstance(actions, np.ndarray):
      actions = torch.from_numpy(actions)
    actions = actions.to(self.device, dtype=torch.float32)
    previous = self._states
    observation, reward, terminated, truncated, _ = self.env.step(actions)
    states = observation[self.obs_group]
    terminated, truncated = terminated.bool(), truncated.bool()
    self._elapsed_steps += 1
    self.returns += reward
    self.failed_once |= terminated
    infos: dict[str, Any] = {"episode": self._episode()} if self.record_metrics else {}
    if self.ignore_terminations:
      terminated = torch.zeros_like(terminated)
    dones = terminated | truncated
    if dones.any():
      # mjlab has already reset these environments: `states` is their first observation of the new episode.
      final = self._wrap(torch.where(dones.unsqueeze(1), previous, states))
      final_info = dict(infos)
      self._elapsed_steps[dones] = 0
      self.returns[dones] = 0.0
      self.failed_once[dones] = False
      infos = {"episode": self._episode()} if self.record_metrics else {}
      flags = dones.to(self.ret_device)
      infos.update(final_observation=final, final_info=final_info, _final_info=flags, _final_observation=flags,
                   _elapsed_steps=flags)
    self._states = states
    return (self._wrap(states), reward.to(self.ret_device), terminated.to(self.ret_device),
            truncated.to(self.ret_device), infos)

  def chunk_step(self, chunk_actions: torch.Tensor):
    """(num_envs, chunk, action_dim) -> lists of observations and infos, (num_envs, chunk) rewards and flags."""
    obs_list, infos_list, rewards, terminations, truncations = [], [], [], [], []
    for i in range(chunk_actions.shape[1]):
      obs, reward, terminated, truncated, infos = self.step(chunk_actions[:, i])
      obs_list.append(obs); infos_list.append(infos)
      rewards.append(reward); terminations.append(terminated); truncations.append(truncated)
    return (obs_list, torch.stack(rewards, dim=1), torch.stack(terminations, dim=1), torch.stack(truncations, dim=1),
            infos_list)

  def close(self) -> None:
    env = getattr(self, "env", None)
    if env is not None:
      env.close()
      self.env = None
