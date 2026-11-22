"""A policy in its own terrain training scene, read by the kind of ground and its level.  (nothing is trained)

For a task built with teacher_task ground=True (r1wb.world.training_ground): the policy acts in the training scene
and for every kind of patch (flat, bumps, stones, tilted, down, up, stairs) and level band this prints
  seconds     environment-seconds spent there
  endings     episodes ended per 100 environment-seconds, by reason (fall, trunk on the ground, the course's height,
              feet height, hands height, tilt, progress lag); "all" sums flags and can count an episode repeatedly
  unique      separate table: non-timeout terminations and physical falls (fallen OR trunk_ground), counted once
  reward      mean step reward
  climbed     mean absolute height of the ground under the feet against the start level (m): how far the robots got
  standing    landings per minute while the course stands still (a foot that was off the ground for 3 steps or more)
  joints      mean joint distance to the course (rad per joint): what the joint-angle posture price charges
  links       mean link angle to the course against gravity and the ground (rad per link): tracking.mdp.posture_course_l1
The answer to "which ground still ends the episodes, and by which criterion".
usage: python -m r1wb.eval.ground_terms <task> <checkpoint> [--seconds 30] [--envs 4096] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

REASONS = ("fallen", "trunk_ground", "anchor_z", "feet_z", "hands_z", "anchor_rot", "goal_lag")
BANDS = ((0, 2), (3, 5), (6, 7))


def unique_endings(manager) -> torch.Tensor:
  """Per-environment [termination, physical fall] flags from the completed control step.

  mjlab's auto-reset logs but preserves these buffers until the next termination compute. Read them immediately
  after step(), not from the already-reset physical state. A timeout alone is not a termination or a fall.
  """
  physical = torch.zeros_like(manager.terminated)
  for name in ("fallen", "trunk_ground"):
    if name in manager.active_terms:
      physical |= manager.get_term(name)
  return torch.stack((manager.terminated, physical & manager.terminated), dim=1)


def print_unique_endings(kinds: list[str], steps: torch.Tensor, ended: torch.Tensor, dt: float) -> None:
  """Keep the original table parseable; every new row has a distinct 'unique' prefix."""
  print("unique ground levels seconds terminations physical_falls terminations/100s physical_falls/100s")
  for k, kind in enumerate(kinds):
    for b, (low, high) in enumerate(BANDS):
      i = k * len(BANDS) + b
      seconds = float(steps[i]) * dt
      if seconds > 0.0:
        count = ended[i]
        print(f"unique {kind} {low}-{high} {seconds:.6f} {int(count[0])} {int(count[1])} "
              f"{100.0 * float(count[0]) / seconds:.6f} {100.0 * float(count[1]) / seconds:.6f}")
  seconds = float(steps.sum()) * dt
  if seconds > 0.0:
    count = ended.sum(0)
    print(f"unique all - {seconds:.6f} {int(count[0])} {int(count[1])} "
          f"{100.0 * float(count[0]) / seconds:.6f} {100.0 * float(count[1]) / seconds:.6f}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("checkpoint")
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--envs", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict

  from r1wb import world
  from r1wb.student.observations import load_teacher
  from r1wb.tasks import TASKS
  from r1wb.tracking import mdp as tracking

  torch.manual_seed(args.seed)
  cfg = TASKS[args.task]["env"]()
  cfg.seed = args.seed
  cfg.scene.num_envs = args.envs
  kinds = list(cfg.scene.terrain.terrain_generator.sub_terrains)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  model = load_teacher(args.checkpoint, "cuda:0")
  policy = lambda o: model(TensorDict({"actor": o["actor"][:, :model.obs_dim]}, batch_size=[args.envs]))  # noqa: E731
  term = env.command_manager.get_term("motion")
  terrain = env.scene.terrain
  count, device, dt = args.envs, env.device, float(env.step_dt)
  band = torch.zeros(8, dtype=torch.long, device=device)
  for index, (low, high) in enumerate(BANDS):
    band[low:high + 1] = index
  cell = terrain.terrain_types * len(BANDS) + band[terrain.terrain_levels.clamp(0, 7)]          # fixed per environment
  cells = len(kinds) * len(BANDS)

  def zeros(*shape, kind=torch.float):
    return torch.zeros(*shape, device=device, dtype=kind)

  steps, reward, climbed, hold_steps, landings, joints, links = (zeros(cells) for _ in range(7))
  ended = zeros(cells, len(REASONS))
  unique_ended = zeros(cells, 2, kind=torch.long)
  air = zeros(count, 2, kind=torch.long)
  obs = wrapped.get_observations()
  for _ in range(int(args.seconds / dt)):
    with torch.no_grad():
      action = policy(obs)
    obs, gained, dones, _ = wrapped.step(action)
    unique_ended.index_add_(0, cell, unique_endings(env.termination_manager).long())
    steps.index_add_(0, cell, torch.ones(count, device=device))
    reward.index_add_(0, cell, gained.float())
    climbed.index_add_(0, cell, world.support_under(env).abs())
    joints.index_add_(0, cell, (term.robot_joint_pos - term.joint_pos).abs().mean(dim=1))
    links.index_add_(0, cell, tracking.posture_course_l1(env, "motion"))
    for index, name in enumerate(REASONS):
      if name in env.termination_manager.active_terms:
        ended[:, index].index_add_(0, cell, env.termination_manager.get_term(name).float())
    contact = env.scene["feet_ground"].data.found.reshape(count, -1)[:, :2] > 0
    landed = (contact & (air >= 3)).float().sum(1)
    air = torch.where(contact, torch.zeros_like(air), air + 1)
    air[dones.bool()] = 0
    hold = getattr(term, "hold", None)
    if hold is not None:
      still = (hold & ~dones.bool()).float()
      hold_steps.index_add_(0, cell, still)
      landings.index_add_(0, cell, landed * still)

  print(f"{args.task}  {args.checkpoint}\n{args.envs} environments x {args.seconds:g} s, seed {args.seed}; endings per 100 environment-seconds")
  print(f"{'ground':12s} {'levels':>6s} {'seconds':>8s} " + " ".join(f"{name[:9]:>9s}" for name in REASONS)
        + f" {'all':>6s} {'reward':>7s} {'climbed':>8s} {'standing/min':>12s} {'joints':>7s} {'links':>7s}")
  for k, kind in enumerate(kinds):
    for b, (low, high) in enumerate(BANDS):
      i = k * len(BANDS) + b
      seconds = float(steps[i]) * dt
      if seconds <= 0.0:
        continue
      rate = 100.0 * ended[i] / seconds
      held = float(hold_steps[i]) * dt
      print(f"{kind:12s} {f'{low}-{high}':>6s} {seconds:8.0f} " + " ".join(f"{float(v):9.2f}" for v in rate)
            + f" {float(rate.sum()):6.2f} {float(reward[i] / steps[i].clamp_min(1)):7.3f} {float(climbed[i] / steps[i].clamp_min(1)):8.3f}"
            + (f" {60.0 * float(landings[i]) / held:12.1f}" if held > 1.0 else f" {'-':>12s}")
            + f" {float(joints[i] / steps[i].clamp_min(1)):7.4f} {float(links[i] / steps[i].clamp_min(1)):7.4f}")
  total = float(steps.sum()) * dt
  print(f"{'all':12s} {'':>6s} {total:8.0f} " + " ".join(f"{100.0 * float(v) / total:9.2f}" for v in ended.sum(0))
        + f" {100.0 * float(ended.sum()) / total:6.2f} {float(reward.sum() / steps.sum()):7.3f}")
  print_unique_endings(kinds, steps, unique_ended, dt)


if __name__ == "__main__":
  main()
