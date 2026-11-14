"""Where a training iteration's time goes: one environment step taken apart, for several environment counts.

For a task's training scene, driven by a checkpoint's policy, this prints per control step (ms):
  step      the wall time of env.step as training runs it (no extra synchronisation), and with the policy's inference
  phases    the same step with the GPU synchronised around every phase, so that each phase is charged its own GPU
            work: physics (4 substeps), writing actions, scene update, terminations, rewards, events, resets, forward
            kinematics, commands, sensors (rays, contacts), observations; and the policy's forward pass
and from two environment counts the part that does not shrink with the count (a + b x environments).
Nothing is trained and nothing is recorded: a measurement of the simulator.
usage: python -m r1wb.eval.step_profile <task> <checkpoint> [--envs 4096 16384] [--steps 96] [--warm 48]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

PHASES = ("physics", "write", "scene", "terminations", "rewards", "events", "resets", "forward", "commands", "sensors",
          "observations", "policy")


def sync(device: str) -> None:
  wp.synchronize_device(device)
  torch.cuda.synchronize()


def measure(task: str, checkpoint: str, count: int, steps: int, warm: int, device: str = "cuda:0",
            nconmax: int | None = None, njmax: int | None = None, lift: float = 0.0, noise: float = 0.0) -> dict:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict

  from r1wb.student.observations import load_teacher
  from r1wb.tasks import TASKS

  torch.manual_seed(0)
  cfg = TASKS[task]["env"]()
  cfg.scene.num_envs = count
  cfg.seed = 0
  budget = (cfg.sim.nconmax, cfg.sim.njmax)
  if nconmax is not None:
    cfg.sim.nconmax = nconmax
  if njmax is not None:
    cfg.sim.njmax = njmax
  if lift > 0.0:      # the pose the model is BUILT in (every robot at the world's origin) is raised off the ground
    from r1wb import world
    world.raise_build_pose(cfg, lift)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  model = load_teacher(checkpoint, device)
  width = model.obs_dim

  def act(obs):
    with torch.no_grad():
      action = model(TensorDict({"actor": obs["actor"][:, :width]}, batch_size=[count]))
      return action + noise * torch.randn_like(action) if noise > 0.0 else action       # (training explores with this spread)

  obs = wrapped.get_observations()
  data = env.sim.wp_data
  contacts, rows, rows_mean = 0, 0, 0.0          # what the scene actually uses: contacts of all worlds together, constraint rows per world
  overflowed, overflow_steps = 0, 0              # worlds flagged by the simulator for a budget it ran out of (any kind)
  for _ in range(warm):
    obs, _, _, _ = wrapped.step(act(obs))
    contacts = max(contacts, int(data.nacon.numpy()[0]))
    used = data.nefc.numpy()
    rows, rows_mean = max(rows, int(used.max())), max(rows_mean, float(used.mean()))
    if hasattr(data, "overflow"):
      flags = int((data.overflow.numpy() != 0).sum())
      overflowed, overflow_steps = max(overflowed, flags), overflow_steps + (flags > 0)
  # the state the scene is in after the warm-up, as numbers that two runs can be compared by (same seed, same policy)
  robot = env.scene["robot"].data
  digest = (float(robot.joint_pos.abs().sum()), float(robot.root_link_pos_w[:, 2].sum()), int(env.episode_length_buf.sum()))
  # 1) as training runs it
  sync(device)
  began = time.perf_counter()
  for _ in range(steps):
    obs, _, _, _ = wrapped.step(act(obs))
  sync(device)
  plain = (time.perf_counter() - began) / steps
  sync(device)
  began = time.perf_counter()
  for _ in range(steps):
    act(obs)
  sync(device)
  inference = (time.perf_counter() - began) / steps
  # 2) phase by phase (synchronised)
  spent = dict.fromkeys(PHASES, 0.0)

  def timed(name, function):
    def wrapper(*args, **kwargs):
      sync(device)
      start = time.perf_counter()
      result = function(*args, **kwargs)
      sync(device)
      spent[name] += time.perf_counter() - start
      return result
    return wrapper

  env.sim.step = timed("physics", env.sim.step)
  env.action_manager.process_action = timed("write", env.action_manager.process_action)
  env.action_manager.apply_action = timed("write", env.action_manager.apply_action)
  env.scene.write_data_to_sim = timed("write", env.scene.write_data_to_sim)
  env.scene.update = timed("scene", env.scene.update)
  env.termination_manager.compute = timed("terminations", env.termination_manager.compute)
  env.reward_manager.compute = timed("rewards", env.reward_manager.compute)
  env.event_manager.apply = timed("events", env.event_manager.apply)
  env._reset_idx = timed("resets", env._reset_idx)
  env.sim.forward = timed("forward", env.sim.forward)
  env.command_manager.compute = timed("commands", env.command_manager.compute)
  env.sim.sense = timed("sensors", env.sim.sense)
  env.observation_manager.compute = timed("observations", env.observation_manager.compute)
  sync(device)
  began = time.perf_counter()
  for _ in range(steps):
    sync(device)
    start = time.perf_counter()
    action = act(obs)
    sync(device)
    spent["policy"] += time.perf_counter() - start
    obs, _, _, _ = wrapped.step(action)
  sync(device)
  whole = (time.perf_counter() - began) / steps
  memory = torch.cuda.max_memory_allocated() / 2**30
  used = torch.cuda.mem_get_info()
  env.close()
  del env, wrapped, model
  torch.cuda.empty_cache()
  return dict(count=count, plain=plain, inference=inference, whole=whole, phases={k: v / steps for k, v in spent.items()},
              gpu_gb=(used[1] - used[0]) / 2**30, torch_gb=memory, contacts=contacts / count, rows=rows, rows_mean=rows_mean,
              budget=budget, given=(cfg.sim.nconmax, cfg.sim.njmax), overflowed=overflowed, overflow_steps=overflow_steps,
              pool=int(data.naconmax) if hasattr(data, "naconmax") else -1, digest=digest)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("checkpoint")
  parser.add_argument("--envs", type=int, nargs="+", default=[4096, 16384])
  parser.add_argument("--steps", type=int, default=96)
  parser.add_argument("--warm", type=int, default=48)
  parser.add_argument("--nconmax", type=int, default=None, help="contact budget per world instead of the task's")
  parser.add_argument("--njmax", type=int, default=None, help="constraint rows per world instead of the task's")
  parser.add_argument("--lift", type=float, default=0.0, help="m: the robot's build pose is raised (no contacts when the model is built)")
  parser.add_argument("--noise", type=float, default=0.0, help="spread of the noise added to the actions (training: about 0.55)")
  args = parser.parse_args()
  rows = [measure(args.task, args.checkpoint, count, args.steps, args.warm, nconmax=args.nconmax, njmax=args.njmax,
                  lift=args.lift, noise=args.noise) for count in args.envs]
  print(f"{args.task}   {torch.cuda.get_device_name(0)}   {args.steps} steps after {args.warm}; ms per control step")
  print(f"{'envs':>7s} {'step':>7s} {'+policy':>8s} {'env-steps/s':>12s} | " + " ".join(f"{name[:9]:>9s}" for name in PHASES)
        + f" | {'sum':>7s} {'GPU GB':>7s}")
  for row in rows:
    print(f"  {row['count']} environments: budget per world {row['given'][0]} contacts, {row['given'][1]} rows (the task's: {row['budget']}); "
          f"in use over the warm-up: {row['contacts']:.1f} contacts per world at most (all worlds together), rows per world at most "
          f"{row['rows']}, mean at most {row['rows_mean']:.0f}; worlds flagged for an overflow at most {row['overflowed']} "
          f"(in {row['overflow_steps']} of the warm-up steps); state after the warm-up: joints {row['digest'][0]:.3f}, "
          f"heights {row['digest'][1]:.3f}, episode steps {row['digest'][2]}")
  for row in rows:
    phases = row["phases"]
    print(f"{row['count']:7d} {1000 * (row['plain'] - row['inference']):7.1f} {1000 * row['plain']:8.1f} {row['count'] / row['plain']:12.0f} | "
          + " ".join(f"{1000 * phases[name]:9.2f}" for name in PHASES) + f" | {1000 * sum(phases.values()):7.1f} {row['gpu_gb']:7.1f}")
  if len(rows) >= 2:
    first, last = rows[0], rows[-1]
    slope = (last["plain"] - first["plain"]) / (last["count"] - first["count"])
    fixed = first["plain"] - slope * first["count"]
    print(f"step time = {1000 * fixed:.1f} ms that does not depend on the number of environments + {1e6 * slope:.2f} us per environment "
          f"(from {first['count']} and {last['count']}); the fixed part is {100 * fixed / first['plain']:.0f}% of a step at {first['count']} "
          f"and {100 * fixed / last['plain']:.0f}% at {last['count']}")
    print("per phase, the part that does not depend on the count (ms) and the part per 1,000 environments (ms):")
    for name in PHASES:
      b = (last["phases"][name] - first["phases"][name]) / (last["count"] - first["count"])
      a = first["phases"][name] - b * first["count"]
      print(f"  {name:13s} {1000 * a:7.2f}  {1e6 * b:7.3f}")


if __name__ == "__main__":
  main()
