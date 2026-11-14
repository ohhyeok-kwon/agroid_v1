"""What the training scene pays a policy during and after a row of hits, term by term.  (nothing is trained)

Question (11-13): rows of eight hits while walking fell the policy in the two seconds after the row, and in the
traces its feet do not keep up with the body. Which reward terms charge travelling with the push, and does
travelling with it go with staying up in the policy's own rows?

The policy acts in its own training scene (hits and rows as trained). For every row on a command stream:
  window     from the first hit until 2 s after the last one, or the end of the episode
  per term   the reward summed over the window, in the learner's units (weight x value x step)
  travel     the root's displacement along the direction of the push at 0.3 / 0.6 s after the first hit and at the
             end of the window (m), its largest speed along the push (m/s)
  steps      landings in the window, and for each landing how far ahead of the other foot (along the push) it lands
  outcome    fallen (fallen / trunk on the ground), another ending, or survived
Printed: the rows by kind of command and outcome, and for the rows that are survived a least-squares fit of each
term's sum on [1, impulse of the row, travel at the end]: the last coefficient is what one more metre of travelling
with the push is paid or charged at the same row. All rows are saved to --out for other cuts.

usage: python -m r1wb.eval.row_terms <task> <checkpoint> [--seconds 60] [--envs 4096] [--out rows.npz]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"


def find_event(env):
  """The disturbance event's object (it carries the rows' bookkeeping)."""
  seen = []
  for value in vars(env.event_manager).values():
    groups = value.values() if isinstance(value, dict) else [value] if isinstance(value, list) else []
    for group in groups:
      for item in (group if isinstance(group, (list, tuple)) else [group]):
        func = getattr(item, "func", item)
        if hasattr(func, "burst_left") and hasattr(func, "burst_angle"):
          return func
        seen.append(type(func).__name__)
  raise RuntimeError(f"no disturbance event with rows in this scene (saw {sorted(set(seen))[:12]})")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("checkpoint")
  parser.add_argument("--seconds", type=float, default=60.0)
  parser.add_argument("--envs", type=int, default=4096)
  parser.add_argument("--after", type=float, default=2.0)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict
  from r1wb.student.observations import load_teacher
  from r1wb.tasks import TASKS

  torch.manual_seed(0)
  cfg = TASKS[args.task]["env"]()
  cfg.scene.num_envs = args.envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  model = load_teacher(args.checkpoint, "cuda:0")
  policy = lambda o: model(TensorDict({"actor": o["actor"][:, :model.obs_dim]}, batch_size=[args.envs]))  # noqa: E731
  names = list(env.reward_manager.active_terms)
  term = env.command_manager.get_term("motion")
  event = find_event(env)
  robot = env.scene["robot"]
  feet = [robot.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
  count, device, dt = args.envs, env.device, float(env.step_dt)
  T = len(names)

  def zeros(*shape, kind=torch.float):
    return torch.zeros(*shape, device=device, dtype=kind)

  active = zeros(count, kind=torch.bool)
  sums, impulse, hits, clock = zeros(count, T), zeros(count), zeros(count), zeros(count)
  origin, push, heading_cos = zeros(count, 2), zeros(count, 2), zeros(count)
  early = zeros(count, 2)                    # travel along the push at 0.3 / 0.6 s
  peak, landings, ahead, longest, tilt = zeros(count), zeros(count), zeros(count), zeros(count), zeros(count)
  kind = zeros(count, kind=torch.long)       # 0 clip, 1 stand, 2 moving
  speed_told = zeros(count)
  air = zeros(count, 2, kind=torch.long)
  rows = []
  obs = wrapped.get_observations()
  for _ in range(int(args.seconds / dt)):
    with torch.no_grad():
      action = policy(obs)
    data = robot.data
    position_before = data.root_link_pos_w[:, :2].clone()
    obs, _, dones, _ = wrapped.step(action)
    data = robot.data
    reward = env.reward_manager._step_reward * dt
    manager = env.termination_manager
    fell = zeros(count, kind=torch.bool)
    for name in ("fallen", "trunk_ground"):
      if name in manager.active_terms:
        fell |= manager.get_term(name).bool()
    ended = dones.bool()
    hit_now = env.hit_clock == 0.0
    # a row opens: the hit that just started leaves further hits to come
    opening = hit_now & (event.burst_left > 0) & ~active & ~ended
    if opening.any():
      ids = opening.nonzero().squeeze(-1)
      angle = event.burst_angle[ids]
      push[ids] = torch.stack([angle.cos(), angle.sin()], dim=1)
      origin[ids] = position_before[ids]
      w, x, y, z = data.root_link_quat_w[ids].unbind(-1)
      yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
      heading_cos[ids] = torch.cos(angle - yaw)
      target = term.stream_target[ids]
      moving = (torch.norm(target[:, :2], dim=1) > 0.05) | (target[:, 2].abs() > 0.1)
      kind[ids] = torch.where(term.stream_env[ids], torch.where(moving, 2, 1), 0)
      speed_told[ids] = torch.norm(target[:, :2], dim=1)
      for buffer in (sums, impulse, hits, clock, early, peak, landings, ahead, longest, tilt):
        buffer[ids] = 0.0
      active[ids] = True
    live = active & ~opening | opening
    travel = torch.sum((data.root_link_pos_w[:, :2] - origin) * push, dim=1)
    velocity = torch.sum(data.root_link_lin_vel_w[:, :2] * push, dim=1)
    w, x, y, z = data.root_link_quat_w.unbind(-1)
    lean = torch.acos((1 - 2 * (x * x + y * y)).clamp(-1.0, 1.0))
    contact = env.scene["feet_ground"].data.found.reshape(count, -1)[:, :2] > 0
    landing = contact & (air >= 3)
    air = torch.where(contact, torch.zeros_like(air), air + 1)
    soles = data.body_link_pos_w[:, feet, :2]
    lead = torch.sum((soles - soles.flip(1)) * push.unsqueeze(1), dim=2)        # each foot ahead of the other, along the push
    sums += reward * live.float().unsqueeze(1) * (~ended).float().unsqueeze(1)
    impulse += torch.norm(event.force, dim=1) * dt * (hit_now & live).float()
    hits += (hit_now & live).float()
    clock += dt * live.float()
    for k, moment in enumerate((0.3, 0.6)):
      reached = live & ~ended & ((clock - moment).abs() < 0.5 * dt)
      early[:, k] = torch.where(reached, travel, early[:, k])
    alive = live & ~ended
    peak = torch.where(alive, torch.maximum(peak, velocity), peak)
    tilt = torch.where(alive, torch.maximum(tilt, lean), tilt)
    landed = (landing & alive.unsqueeze(1)).float()
    landings += landed.sum(1)
    ahead += (lead * landed).sum(1)
    longest = torch.where(alive, torch.maximum(longest, (lead * landed).amax(dim=1)), longest)
    closing = live & (ended | ((event.burst_left == 0) & (env.hit_clock >= args.after)))
    if closing.any():
      ids = closing.nonzero().squeeze(-1)
      outcome = torch.where(fell[ids], 2, torch.where(ended[ids], 1, 0))
      final = torch.where(ended[ids], torch.sum((position_before[ids] - origin[ids]) * push[ids], dim=1), travel[ids])
      rows.append(torch.cat([kind[ids].float().unsqueeze(1), outcome.float().unsqueeze(1), heading_cos[ids].unsqueeze(1),
                             impulse[ids].unsqueeze(1), hits[ids].unsqueeze(1), clock[ids].unsqueeze(1), early[ids],
                             final.unsqueeze(1), peak[ids].unsqueeze(1), landings[ids].unsqueeze(1), ahead[ids].unsqueeze(1),
                             longest[ids].unsqueeze(1), tilt[ids].unsqueeze(1), speed_told[ids].unsqueeze(1), sums[ids]], dim=1).cpu())
      active[ids] = False
  table = torch.cat(rows).numpy() if rows else np.zeros((0, 15 + T))
  columns = ["kind", "outcome", "push_vs_heading", "impulse", "hits", "seconds", "travel_0.3", "travel_0.6", "travel", "peak_speed",
             "landings", "ahead_sum", "longest", "tilt", "told_speed"] + names
  if args.out is not None:
    np.savez_compressed(args.out, rows=table, columns=np.array(columns), task=args.task, checkpoint=str(args.checkpoint))
  col = {name: i for i, name in enumerate(columns)}
  print(f"task {args.task}   policy {Path(args.checkpoint).parent.name[-34:]}@{Path(args.checkpoint).stem}   {args.seconds:g} s x {count} envs   rows {len(table)}")
  print(f"{'command':8s} {'push':8s} {'rows':>5s} {'fell':>5s} {'other':>5s} | survived: {'hits':>5s} {'N s':>6s} {'travel m':>8s} {'peak m/s':>8s} {'landings':>8s} {'ahead m':>8s} {'longest':>8s} {'tilt':>5s}"
        f" | fell: {'N s':>6s} {'travel0.6':>9s} {'ahead m':>8s} {'longest':>8s}   (survived travel0.6 {'':>1s})")
  for k, label in ((0, "clip"), (1, "stand"), (2, "moving")):
    for side, low, high in (("forward", 0.5, 1.01), ("backward", -1.01, -0.5), ("sideways", -0.5, 0.5)):
      pick = (table[:, col["kind"]] == k) & (table[:, col["push_vs_heading"]] > low) & (table[:, col["push_vs_heading"]] <= high)
      if pick.sum() < 5:
        continue
      part = table[pick]
      ok, down = part[part[:, col["outcome"]] == 0], part[part[:, col["outcome"]] == 2]
      mean = lambda block, name: float(block[:, col[name]].mean()) if len(block) else float("nan")  # noqa: E731
      per = lambda block: float((block[:, col["ahead_sum"]] / np.maximum(block[:, col["landings"]], 1)).mean()) if len(block) else float("nan")  # noqa: E731
      print(f"{label:8s} {side:8s} {len(part):5d} {len(down):5d} {int((part[:, col['outcome']] == 1).sum()):5d} |           {mean(ok, 'hits'):5.1f} {mean(ok, 'impulse'):6.1f} {mean(ok, 'travel'):8.2f}"
            f" {mean(ok, 'peak_speed'):8.2f} {mean(ok, 'landings'):8.1f} {per(ok):8.3f} {mean(ok, 'longest'):8.3f} {mean(ok, 'tilt'):5.2f}"
            f" |       {mean(down, 'impulse'):6.1f} {mean(down, 'travel_0.6'):9.2f} {per(down):8.3f} {mean(down, 'longest'):8.3f}   ({mean(ok, 'travel_0.6'):.2f})")
  for k, label in ((1, "stand"), (2, "moving")):
    ok = table[(table[:, col["kind"]] == k) & (table[:, col["outcome"]] == 0)]
    if len(ok) < 30:
      continue
    design = np.stack([np.ones(len(ok)), ok[:, col["impulse"]], ok[:, col["travel"]]], axis=1)
    print(f"\nsurvived rows on a '{label}' command ({len(ok)}): each term's sum over the window, and the fit sum = a + b x impulse + c x travel")
    print(f"{'term':26s} {'mean':>8s} {'c per m of travel':>18s} {'b per 10 N s':>13s}")
    order = []
    for i, name in enumerate(names):
      y = ok[:, 15 + i]
      coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
      order.append((coefficient[2], name, float(y.mean()), coefficient[1] * 10.0))
    for c, name, mean, b in sorted(order):
      if abs(mean) > 0.01 or abs(c) > 0.05:
        print(f"{name:26s} {mean:+8.2f} {c:+18.2f} {b:+13.2f}")
    total = ok[:, 15:].sum(1)
    coefficient, *_ = np.linalg.lstsq(design, total, rcond=None)
    print(f"{'all terms':26s} {total.mean():+8.2f} {coefficient[2]:+18.2f} {coefficient[1] * 10.0:+13.2f}")
  env.close()


if __name__ == "__main__":
  main()
