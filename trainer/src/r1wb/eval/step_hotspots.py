"""Which functions the Python side of an environment step spends its time in (cProfile over env.step with the policy).
A measurement of the simulator's driver code: nothing is trained. The GPU is synchronised after every step so that
waiting for it is not charged to whichever function happens to read a value next.
usage: python -m r1wb.eval.step_hotspots <task> <checkpoint> [--envs 4096] [--steps 150] [--top 45]
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("checkpoint")
  parser.add_argument("--envs", type=int, default=4096)
  parser.add_argument("--steps", type=int, default=150)
  parser.add_argument("--top", type=int, default=45)
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict

  from r1wb.student.observations import load_teacher
  from r1wb.tasks import TASKS

  torch.manual_seed(0)
  cfg = TASKS[args.task]["env"]()
  cfg.scene.num_envs = args.envs
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  model = load_teacher(args.checkpoint, "cuda:0")
  width = model.obs_dim

  def act(obs):
    with torch.no_grad():
      return model(TensorDict({"actor": obs["actor"][:, :width]}, batch_size=[args.envs]))

  obs = wrapped.get_observations()
  for _ in range(60):
    obs, _, _, _ = wrapped.step(act(obs))
  profile = cProfile.Profile()
  for _ in range(args.steps):
    action = act(obs)
    wp.synchronize_device("cuda:0"); torch.cuda.synchronize()
    profile.enable()
    obs, _, _, _ = wrapped.step(action)
    profile.disable()
    wp.synchronize_device("cuda:0"); torch.cuda.synchronize()
  stats = pstats.Stats(profile)
  stats.sort_stats("cumulative")
  total = stats.total_tt
  print(f"{args.task}: {args.envs} environments, {args.steps} steps; {1000 * total / args.steps:.1f} ms of Python-visible time per step (with the profiler's own cost)")
  rows = []
  for (file, line, name), (calls, _, own, cumulative, _) in stats.stats.items():
    where = file.split("site-packages/")[-1].split("/src/")[-1]
    rows.append((cumulative, own, calls, f"{where}:{line} {name}"))
  rows.sort(reverse=True)
  print(f"{'ms/step':>8s} {'own':>7s} {'calls/step':>10s}  function (by time including what it calls)")
  for cumulative, own, calls, label in rows[:args.top]:
    print(f"{1000 * cumulative / args.steps:8.2f} {1000 * own / args.steps:7.2f} {calls / args.steps:10.1f}  {label[-110:]}")
  rows.sort(key=lambda row: -row[1])
  print(f"{'own ms':>8s} {'calls/step':>10s}  function (by its own time)")
  for cumulative, own, calls, label in rows[:25]:
    print(f"{1000 * own / args.steps:8.2f} {calls / args.steps:10.1f}  {label[-110:]}")


if __name__ == "__main__":
  main()
