"""Things System 2 has to see: coloured boxes standing on the floor around the robot.

A box is a fixed body the simulator lets us place per environment (mjlab wraps a body without a joint into a mocap
body). The boxes do not collide: they are what the camera sees, and "touching a box" is judged from distances, so
System 1's physics and contact sensors are exactly those it was trained with.

  cfg.scene.entities.update(scene.boxes())          # before the environment is built
  layout = scene.place(env, rng)                    # after a reset: boxes around each robot; returns their positions
"""
from __future__ import annotations

import math

import torch

COLOURS = {"red": (0.85, 0.08, 0.08, 1.0), "green": (0.05, 0.60, 0.12, 1.0), "blue": (0.08, 0.18, 0.85, 1.0)}
NAMES = tuple(COLOURS)
HALF = (0.2, 0.2, 0.4)                       # half sizes (m): a box 0.4 x 0.4 wide, 0.8 high
DISTANCE = (1.6, 3.4)                        # from the robot at the start (m)
APART = 1.0                                  # least distance between two boxes (m)
VIEW_HALF_ANGLE = math.radians(33.0)         # the head camera sees a box whose centre is within this of straight ahead


def _box_spec(colour: tuple):
  def spec_fn():
    import mujoco
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    body.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HALF), rgba=list(colour), contype=0,
                  conaffinity=0)
    return spec
  return spec_fn


def boxes() -> dict:
  from mjlab.entity import EntityCfg
  return {f"box_{name}": EntityCfg(spec_fn=_box_spec(colour),
                                   init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)))
          for name, colour in COLOURS.items()}


def place(env, generator: torch.Generator, ids: torch.Tensor | None = None, bearing_range: float = math.pi,
          positions: torch.Tensor | None = None) -> torch.Tensor:
  """Puts the boxes of environments `ids` around the robot: each at DISTANCE from it, at a bearing within
  +-bearing_range of its heading, at least APART from one another. Returns their world positions (len(ids), boxes,
  2). positions: (len(ids), boxes, 2) to place them exactly there instead."""
  from mjlab.utils.lab_api.math import yaw_quat
  device = env.device
  ids = torch.arange(env.num_envs, device=device) if ids is None else ids
  count = len(ids)
  robot = env.scene["robot"]
  root = robot.data.root_link_pos_w[ids, :2]
  quat = yaw_quat(robot.data.root_link_quat_w[ids])
  heading = torch.atan2(2.0 * quat[:, 0] * quat[:, 3], 1.0 - 2.0 * quat[:, 3] ** 2)
  if positions is None:
    positions = torch.zeros(count, len(NAMES), 2, device=device)
    pending = torch.ones(count, dtype=torch.bool, device=device)
    for _ in range(200):                                              # draw again where two boxes are too close
      n = int(pending.sum())
      if n == 0:
        break
      distance = DISTANCE[0] + (DISTANCE[1] - DISTANCE[0]) * torch.rand(n, len(NAMES), generator=generator, device=device)
      bearing = (2.0 * torch.rand(n, len(NAMES), generator=generator, device=device) - 1.0) * bearing_range
      angle = heading[pending].unsqueeze(1) + bearing
      drawn = root[pending].unsqueeze(1) + distance.unsqueeze(2) * torch.stack([torch.cos(angle), torch.sin(angle)], dim=2)
      positions[pending] = drawn
      gap = torch.cdist(positions, positions) + 10.0 * torch.eye(len(NAMES), device=device)
      pending = gap.min(dim=2).values.min(dim=1).values < APART
  for index, name in enumerate(NAMES):
    pose = torch.zeros(count, 7, device=device)
    pose[:, :2], pose[:, 2], pose[:, 3] = positions[:, index], HALF[2], 1.0
    env.scene[f"box_{name}"].write_mocap_pose_to_sim(pose, ids)
  return positions


def seen_from(env, positions: torch.Tensor, ids: torch.Tensor | None = None):
  """Where each box is for the robot now: distance (envs, boxes) from the pelvis, bearing (envs, boxes) from its
  heading (left positive), and whether the head camera has it in view (its centre within VIEW_HALF_ANGLE and no
  nearer box within 8 degrees of the same direction)."""
  from mjlab.utils.lab_api.math import yaw_quat
  robot = env.scene["robot"]
  ids = torch.arange(env.num_envs, device=env.device) if ids is None else ids
  root = robot.data.root_link_pos_w[ids, :2]
  quat = yaw_quat(robot.data.root_link_quat_w[ids])
  heading = torch.atan2(2.0 * quat[:, 0] * quat[:, 3], 1.0 - 2.0 * quat[:, 3] ** 2)
  offset = positions - root.unsqueeze(1)
  distance = offset.norm(dim=2)
  bearing = torch.atan2(offset[:, :, 1], offset[:, :, 0]) - heading.unsqueeze(1)
  bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
  apart = (bearing.unsqueeze(2) - bearing.unsqueeze(1)).abs()
  nearer = distance.unsqueeze(1) < distance.unsqueeze(2)               # [i, j]: box j is nearer than box i
  hidden = ((apart < math.radians(8.0)) & nearer).any(dim=2)
  return distance, bearing, (bearing.abs() < VIEW_HALF_ANGLE) & ~hidden
