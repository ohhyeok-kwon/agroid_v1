"""While the course stands still: where is the body over its feet, and how often does it step?  (nothing is trained)

Reward check for the prices at rest (tracking.mdp.rest_room, step_unprompted with rest): a policy acts in its own
training scene, and for the environments whose course is paused on a standing frame (`hold`) this prints, split by
the time since the pause began and by "calm" / "within 2 s of a hit":
  room      distance of the capture point to the edge of the support (cm, positive = inside; the model's real sole)
  lacking   share of the steps with less than `--room` cm, and the mean shortfall there (cm) = what rest_room charges
  landings  per 100 environment-seconds (a foot that was off the ground for 3 steps or more), and how many of the
            shorter lift-offs (1 or 2 steps) there are next to them (the count the step price cannot see)
  slide     distance the soles travel while they are on the ground, cm per second and foot
usage: python -m r1wb.eval.rest_terms <task> <checkpoint> [--seconds 40] [--envs 4096] [--room 4]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

AGES = ((0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 1.0e9))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("checkpoint")
  parser.add_argument("--seconds", type=float, default=40.0)
  parser.add_argument("--envs", type=int, default=4096)
  parser.add_argument("--room", type=float, default=4.0)
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict
  from r1wb import dynamics
  from r1wb.student.observations import load_teacher
  from r1wb.tasks import TASKS
  from r1wb.tracking.mdp import FOOTPRINT

  torch.manual_seed(0)
  cfg = TASKS[args.task]["env"]()
  cfg.scene.num_envs = args.envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  model = load_teacher(args.checkpoint, "cuda:0")
  policy = lambda o: model(TensorDict({"actor": o["actor"][:, :model.obs_dim]}, batch_size=[args.envs]))  # noqa: E731
  term = env.command_manager.get_term("motion")
  centroidal = dynamics.Centroidal(env, footprint=FOOTPRINT)
  robot = env.scene["robot"]
  feet = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
  count, device, dt = args.envs, env.device, float(env.step_dt)
  room = args.room / 100.0

  def zeros(*shape, kind=torch.float):
    return torch.zeros(*shape, device=device, dtype=kind)

  age = zeros(count)
  air = zeros(count, 2, kind=torch.long)
  soles_before = robot.data.body_link_pos_w[:, feet, :2].clone()
  cells = len(AGES) * 2
  steps, margin, short, lack, landings, hops, slide = (zeros(cells) for _ in range(7))
  obs = wrapped.get_observations()
  for _ in range(int(args.seconds / dt)):
    with torch.no_grad():
      action = policy(obs)
    obs, _, dones, _ = wrapped.step(action)
    hold = term.hold & term.stream_env & ~dones.bool()
    age = torch.where(hold, age + dt, torch.zeros_like(age))
    d = -centroidal.support_margin()["d"]                                   # positive inside
    contact = env.scene["feet_ground"].data.found.reshape(count, -1)[:, :2] > 0
    landing = (contact & (air >= 3)).float().sum(1)
    hop = (contact & (air >= 1) & (air < 3)).float().sum(1)
    air = torch.where(contact, torch.zeros_like(air), air + 1)
    soles = robot.data.body_link_pos_w[:, feet, :2]
    moved = (torch.norm(soles - soles_before, dim=2) * contact.float()).sum(1) * (~dones.bool()).float()
    soles_before = soles.clone()
    recent = (env.hit_clock < 2.0).long()
    for k, (low, high) in enumerate(AGES):
      pick = hold & (age > low) & (age <= high)
      for r in (0, 1):
        which = pick & (recent == r)
        cell = 2 * k + r
        steps[cell] += which.float().sum()
        margin[cell] += (d * which.float()).sum()
        short[cell] += ((d < room) & which).float().sum()
        lack[cell] += ((room - d).clamp(0.0, 0.10) * which.float()).sum()
        landings[cell] += (landing * which.float()).sum()
        hops[cell] += (hop * which.float()).sum()
        slide[cell] += (moved * which.float()).sum()
  print(f"task {args.task}   policy {Path(args.checkpoint).parent.name[-34:]}@{Path(args.checkpoint).stem}   {args.seconds:g} s x {count} envs   room {args.room:g} cm")
  print(f"{'since the pause':16s} {'':10s} {'env-s':>8s} {'room cm':>8s} {'< room':>7s} {'lacking cm':>10s} {'charge/s @100':>13s} {'landings/100s':>13s} {'hops/100s':>10s} {'slide cm/s':>10s}")
  for k, (low, high) in enumerate(AGES):
    for r, label in ((0, "calm"), (1, "after hit")):
      cell = 2 * k + r
      n = float(steps[cell])
      if n < 1:
        continue
      seconds = n * dt
      print(f"{low:4.1f} .. {min(high, 99):4.1f} s   {label:10s} {seconds:8.0f} {100 * float(margin[cell]) / n:8.2f} {float(short[cell]) / n:7.3f}"
            f" {100 * float(lack[cell]) / max(float(short[cell]), 1.0):10.2f} {100.0 * float(lack[cell]) / n:13.3f} {100 * float(landings[cell]) / seconds:13.1f}"
            f" {100 * float(hops[cell]) / seconds:10.1f} {100 * float(slide[cell]) / seconds / 2.0:10.2f}")
  calm = [2 * k for k in range(len(AGES))]
  n = float(steps[calm].sum())
  print(f"{'all, calm':27s} {n * dt:8.0f} {100 * float(margin[calm].sum()) / n:8.2f} {float(short[calm].sum()) / n:7.3f} {'':10s} {100.0 * float(lack[calm].sum()) / n:13.3f}"
        f" {100 * float(landings[calm].sum()) / (n * dt):13.1f} {100 * float(hops[calm].sum()) / (n * dt):10.1f} {100 * float(slide[calm].sum()) / (n * dt) / 2.0:10.2f}")
  env.close()


if __name__ == "__main__":
  main()
