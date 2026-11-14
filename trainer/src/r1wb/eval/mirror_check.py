"""The left-right mirror maps of r1wb.symmetry, checked in the simulator, and how far a policy is from symmetric.

1) Policy: on the states the policy itself visits in its scene, |policy(mirror(o)) - mirror(policy(o))| per joint,
   in action units and in rad of joint target (action x scale), by joint group. A symmetric policy gives 0.
2) Maps: some environments on clip replay are overwritten with the mirror image of others (root pose and velocity,
   joint state, the mirrored clip at the same frame, goal mask and lag) and both are driven with mirrored actions.
   If the observation map is right and the physics is symmetric, the observation of the copy stays the mirror of
   the original's: printed per block of the observation after 1, 2, 5, 10 and 20 steps (largest and mean absolute
   difference), next to the same numbers for the UNMIRRORED comparison (how large a wrong map would read).
usage: python -m r1wb.eval.mirror_check <task> <checkpoint> [--envs 512] [--pairs 96]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

BLOCKS = (("reference now (joints)", 48), ("previews", 132), ("anchor height/tilt", 9), ("base lin vel", 3), ("base ang vel", 3),
          ("joint pos", 24), ("joint vel", 24), ("last action", 24), ("reference velocity", 6), ("goal", 18), ("goal lag", 3),
          ("senses: history", 780), ("senses: scan", 117))      # (r1wb.senses; a policy without them stops before)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("task")
  parser.add_argument("checkpoint")
  parser.add_argument("--envs", type=int, default=512)
  parser.add_argument("--pairs", type=int, default=96)
  parser.add_argument("--warm", type=int, default=150)
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict
  from r1wb import robot as r1, symmetry
  from r1wb.student.observations import load_teacher
  from r1wb.tasks import TASKS
  from r1wb.tracking.env_cfg import teacher_observation

  torch.manual_seed(0)
  cfg = TASKS[args.task]["env"](play=True)
  cfg.scene.num_envs = args.envs
  dropped = [name for name in list(cfg.events) if name in ("impulse", "push_robot", "kick")]
  for name in dropped:
    cfg.events.pop(name)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  model = load_teacher(args.checkpoint, "cuda:0")
  count, device = args.envs, env.device
  policy = lambda o: model(TensorDict({"actor": o}, batch_size=[o.shape[0]]))  # noqa: E731
  term = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  task = model.obs_dim != 273
  mirror = symmetry.ActorMirror(model.obs_dim, device)
  scale = torch.tensor([r1.ACTION_SCALE[n] for n in r1.JOINTS], device=device)
  print(f"task {args.task} (play)   policy {Path(args.checkpoint).parent.name[-34:]}@{Path(args.checkpoint).stem}   events left: {sorted(cfg.events)}   dropped: {dropped}")

  def seen():
    return teacher_observation(env, "motion", env.action_manager.action, local=True, width=model.obs_dim)

  # 1) the policy's own asymmetry on the states it visits
  wrapped.get_observations()
  gap_action, gap_rad, steps = torch.zeros(24, device=device), torch.zeros(24, device=device), 0
  moving_gap, standing_gap, moving_n, standing_n = 0.0, 0.0, 0, 0
  for step in range(args.warm):
    with torch.no_grad():
      o = seen()
      action = policy(o)
      other = mirror.action(policy(mirror.observation(o)))
    difference = (action - other).abs()
    if step >= 25:
      gap_action += difference.mean(0)
      gap_rad += (difference * scale).mean(0)
      steps += 1
      if hasattr(term, "stream_target"):
        moving = term.stream_env & (torch.norm(term.stream_target[:, :3], dim=1) > 0.05)
        standing = term.stream_env & ~moving
        moving_gap += float((difference * scale).mean(1)[moving].sum()); moving_n += int(moving.sum())
        standing_gap += float((difference * scale).mean(1)[standing].sum()); standing_n += int(standing.sum())
    wrapped.step(action)
  gap_action, gap_rad = gap_action / steps, gap_rad / steps
  print(f"1) policy asymmetry |policy(mirror o) - mirror policy(o)|, mean over {steps} steps x {count} envs:")
  for label, part in (("legs", slice(0, 12)), ("waist", slice(12, 14)), ("arms", slice(14, 24)), ("all", slice(0, 24))):
    print(f"   {label:6s} {float(gap_action[part].mean()):.4f} action units   {float(gap_rad[part].mean()):.4f} rad of joint target (largest joint {float(gap_rad[part].max()):.4f})")
  if moving_n and standing_n:
    print(f"   stream commands: moving {moving_gap / moving_n:.4f} rad, standing {standing_gap / standing_n:.4f} rad")

  # 2) the maps: mirrored copies on clip replay
  manifest = json.loads(Path(str(cfg.commands["motion"].motion_file)).with_suffix(".manifest.json").read_text())
  names = [clip["name"] for clip in manifest["clips"]]
  twin_clip = torch.tensor([names.index(n[:-2]) if n.endswith("_M") else names.index(n + "_M") for n in names], device=device)
  clip_env = ~term.stream_env if hasattr(term, "stream_env") else torch.ones(count, dtype=torch.bool, device=device)
  ids = clip_env.nonzero().squeeze(-1)
  # sources far from their clip's end, so that neither side runs out within the test
  room = term.clip_end[term.clip_ids[ids]] - term.time_steps[ids]
  ids = ids[room > 60]
  pairs = min(args.pairs, len(ids) // 2)
  source, copy = ids[:pairs], ids[pairs:2 * pairs]
  data = robot.data
  origin = env.scene.env_origins
  position = data.root_link_pos_w[source] - origin[source]
  pose = torch.cat([symmetry.mirror_polar(position) + origin[copy], symmetry.mirror_quat(data.root_link_quat_w[source])], dim=1)
  velocity = torch.cat([symmetry.mirror_polar(data.root_link_lin_vel_w[source]), symmetry.mirror_axial(data.root_link_ang_vel_w[source])], dim=1)
  robot.write_root_link_pose_to_sim(pose, env_ids=copy)
  robot.write_root_link_velocity_to_sim(velocity, env_ids=copy)
  robot.write_joint_state_to_sim(symmetry.mirror_joints(data.joint_pos[source]), symmetry.mirror_joints(data.joint_vel[source]), env_ids=copy)
  clip = term.clip_ids[source]
  term.clip_ids[copy] = twin_clip[clip]
  term.time_steps[copy] = term.clip_start[twin_clip[clip]] + (term.time_steps[source] - term.clip_start[clip])
  if hasattr(term, "goal_mask"):
    term.goal_mask[copy] = term.goal_mask[source]
    twist = torch.tensor((1.0, -1.0, -1.0), device=device)
    term.goal_lag[copy] = term.goal_lag[source] * twist
    if hasattr(term, "_goal_history"):        # the lag integrates against the goal issued 0.2 s earlier
      term._goal_history[copy] = term._goal_history[source] * twist
    if hasattr(term, "twist_mean"):
      term.twist_mean[copy] = term.twist_mean[source] * twist
  env.episode_length_buf[copy] = env.episode_length_buf[source]
  alive = torch.ones(pairs, dtype=torch.bool, device=device)
  print(f"2) {pairs} mirrored copies on clip replay; observation of the copy against the mirrored observation of the original"
        f" (and against the unmirrored one), largest / mean absolute difference")
  header = f"{'block':24s}" + "".join(f" | step {k:2d}: mirrored      as is" for k in (1, 5, 20))
  rows = {name: [] for name, _ in BLOCKS}
  with torch.no_grad():
    action = policy(seen())
  for step in range(1, 21):
    action[copy] = mirror.action(action[source])
    _, _, dones, _ = wrapped.step(action)
    alive &= ~dones[source].bool() & ~dones[copy].bool()
    with torch.no_grad():
      o = seen()
      action = policy(o)
    if step in (1, 5, 20):
      expected, plain, got = mirror.observation(o[source])[alive], o[source][alive], o[copy][alive]
      start = 0
      for name, width in BLOCKS:
        if start >= o.shape[1]:
          break
        part = slice(start, start + width)
        a, b = (got[:, part] - expected[:, part]).abs(), (got[:, part] - plain[:, part]).abs()
        rows[name].append(f" | {float(a.max()):8.4f} {float(a.mean()):7.4f}  {float(b.max()):7.3f} {float(b.mean()):6.3f}")
        start += width
  print(header)
  for name, _ in BLOCKS:
    if rows[name]:
      print(f"{name:24s}" + "".join(rows[name]))
  print(f"   pairs still running at step 20: {int(alive.sum())} of {pairs}")
  env.close()


if __name__ == "__main__":
  main()
