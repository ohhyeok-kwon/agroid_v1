"""Left-right mirror of the robot, of what the System 1 policy observes and of what it outputs.

The body is mirror-symmetric (masses, limits, default pose and action scales are equal on twin joints), so the
best policy is too: mirror(policy(o)) = policy(mirror(o)). Nothing in the training enforced it, and the measured
policies are not: on mirror-image courses (joint difference of the plans 0.000 to 0.001 rad) the robot's motion is
0.015 to 0.018 rad off its mirror image, and which side stops cleanly flips from checkpoint to checkpoint
(gates 11-13 14:59, 15:0x). This module is the single place that knows the mirror maps; r1wb.learning.SymmetricPPO
uses them for a mirror loss, r1wb.eval.mirror_check verifies them in the simulator.

Joint map (derived from the model by forward kinematics and verified on 300 random poses, all 27 bodies, positions
and rotations, residual 0): swap left and right, and change the sign of every roll and yaw joint.
Vectors in a body or heading frame (x forward, y left, z up): a position, velocity or displacement mirrors as
(x, -y, z), an angular velocity as (-x, y, -z), a rotation matrix R as M R M with M = diag(1, -1, 1); its first two
columns flattened row by row [r00 r01 r10 r11 r20 r21] take the signs (+ - - + + -).
"""
from __future__ import annotations

import torch

from r1wb import robot

PREVIEWS = 4                      # future frames in the reference command (tracking.command)
POLAR, AXIAL, ROT6 = (1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)


def _twin(name: str) -> str:
  return name.replace("left", "@").replace("right", "left").replace("@", "right")


JOINT_PERM = tuple(robot.JOINTS.index(_twin(name)) for name in robot.JOINTS)
JOINT_SIGN = tuple(-1.0 if ("_roll_" in name or "_yaw_" in name) else 1.0 for name in robot.JOINTS)


class Layout:
  """Index and sign of a mirrored vector built block by block: mirror(x) = x[..., index] * sign."""

  def __init__(self):
    self.index: list[int] = []
    self.sign: list[float] = []

  def _add(self, index, sign):
    start = len(self.index)
    self.index += [start + i for i in index]
    self.sign += list(sign)
    return self

  def joints(self):
    return self._add(JOINT_PERM, JOINT_SIGN)

  def polar(self):
    return self._add(range(3), POLAR)

  def axial(self):
    return self._add(range(3), AXIAL)

  def rot6(self):
    return self._add(range(6), ROT6)

  def same(self, count: int = 1):
    return self._add(range(count), [1.0] * count)

  def goal(self):
    """student.command goal: twist (vx vy wz), height, pitch, [hand on, left wrist xyz, time], [hand on, right wrist
    xyz, time], mask (3). The wrists swap."""
    self._add(range(3), (1.0, -1.0, -1.0)).same(2)
    start = len(self.index)
    self.index += [start + 5, start + 6, start + 7, start + 8, start + 9, start, start + 1, start + 2, start + 3, start + 4]
    self.sign += [1.0, 1.0, -1.0, 1.0, 1.0] * 2
    return self.same(3)

  def lag(self):
    return self._add(range(3), (1.0, -1.0, -1.0))

  def history(self):
    """r1wb.senses history: per step joint angles, joint velocities, action, angular velocity, gravity in the trunk."""
    from r1wb import senses
    for _ in range(senses.HISTORY):
      self.joints().joints().joints().axial().polar()
    return self

  def scan(self):
    """r1wb.senses scan: heights on a grid in the heading frame (x fastest); the mirror swaps left and right rows."""
    from r1wb import senses
    nx, ny = senses.SCAN_GRID
    return self._add([(ny - 1 - iy) * nx + ix for iy in range(ny) for ix in range(nx)], [1.0] * (nx * ny))

  def tensors(self, device):
    assert sorted(self.index) == list(range(len(self.index)))
    return torch.tensor(self.index, dtype=torch.long, device=device), torch.tensor(self.sign, device=device)


def actor_layout(task: bool = True, senses: str = "") -> Layout:
  """The teacher actor's input in the order of tracking.env_cfg.teacher_observation (273, or 294 with goal and lag,
  then the senses of r1wb.senses: "history", "scan" or "both")."""
  layout = Layout().joints().joints()                       # reference joint angles and velocities now
  for _ in range(PREVIEWS):
    layout.joints().polar().rot6()                          # preview: joint angles, anchor displacement, rotation
  layout.polar().rot6()                                     # reference anchor seen from the robot: height, tilt
  layout.polar().axial()                                    # base linear and angular velocity (IMU)
  layout.joints().joints().joints()                         # joint angles, velocities, previous action
  layout.polar().axial()                                    # reference anchor velocity in its own frame
  if task:
    layout.goal().lag()
  if senses in ("history", "both"):
    layout.history()
  if senses in ("scan", "both"):
    layout.scan()
  return layout


class ActorMirror:
  """mirror of the actor's observation (by its width) and of the action."""

  def __init__(self, width: int, device):
    if width == 417:
      from r1wb.head_resume import mirror_tensors
      self.obs_index, self.obs_sign, self.act_index, self.act_sign = mirror_tensors(device)
      return
    from r1wb import senses
    layouts = [actor_layout(False)] + [actor_layout(True, kind) for kind in senses.WIDTH]
    widths = {len(layout.index): layout for layout in layouts}
    if width not in widths:
      raise ValueError(f"no mirror map for an actor observation of {width} values (known: {sorted(widths)})")
    self.obs_index, self.obs_sign = widths[width].tensors(device)
    self.act_index = torch.tensor(JOINT_PERM, dtype=torch.long, device=device)
    self.act_sign = torch.tensor(JOINT_SIGN, device=device)

  def observation(self, obs: torch.Tensor) -> torch.Tensor:
    return obs[..., self.obs_index] * self.obs_sign

  def action(self, action: torch.Tensor) -> torch.Tensor:
    return action[..., self.act_index] * self.act_sign


def mirror_joints(values: torch.Tensor) -> torch.Tensor:
  index = torch.tensor(JOINT_PERM, dtype=torch.long, device=values.device)
  return values[..., index] * torch.tensor(JOINT_SIGN, device=values.device, dtype=values.dtype)


def mirror_quat(quat: torch.Tensor) -> torch.Tensor:
  """(w, x, y, z) of the mirrored rotation M R M."""
  return quat * torch.tensor((1.0, -1.0, 1.0, -1.0), device=quat.device, dtype=quat.dtype)


def mirror_polar(vector: torch.Tensor) -> torch.Tensor:
  return vector * torch.tensor(POLAR, device=vector.device, dtype=vector.dtype)


def mirror_axial(vector: torch.Tensor) -> torch.Tensor:
  return vector * torch.tensor(AXIAL, device=vector.device, dtype=vector.dtype)


class Symmetrized(torch.nn.Module):
  """policy_s(o) = (policy(o) + mirror(policy(mirror(o)))) / 2. It commutes with the mirror for any policy, without
  training: the test of whether a policy's asymmetry is something it needs (r1wb.eval tools, --symmetrize)."""

  def __init__(self, model):
    super().__init__()
    self.model, self.obs_dim, self._mirror = model, model.obs_dim, None

  def forward(self, observations):
    from tensordict import TensorDict
    seen = observations["actor"]
    if self._mirror is None:
      self._mirror = ActorMirror(seen.shape[1], seen.device)
    flipped = TensorDict({"actor": self._mirror.observation(seen)}, batch_size=[seen.shape[0]])
    return 0.5 * (self.model(observations) + self._mirror.action(self.model(flipped)))
