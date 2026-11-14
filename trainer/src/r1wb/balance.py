"""Rest poses of the motion library, given the largest stability margin their stance allows.

The library holds people's motion retargeted joint by joint. Where such a frame is at rest, nothing guarantees that
THIS body's centre of mass stands where ITS soles give it room. Measured 2026-11-13 (gates.md): in the frames a plan
comes to rest on, it stands 2 to 6 cm toward the heels (6.5 cm of room at the heels against 11.8 cm at the toes),
different for every piece of motion, and the stops out of backward walking are the ones that need a late extra
step. A plan that comes to rest should rest where the body has as much room as its stance can give.

What is done, per frame and from the frame's own numbers (no constant angle, no list of clips, no directions):
  which    the frames a plan can come to rest on: exactly the course generator's own test (student.stream
           stand_frame: pelvis, heading and both feet at rest for the next second)
  support  the convex hull of the footprint corners that are on the ground (the collision footprint measured from
           the model); its largest margin is the radius of the largest circle that fits inside
  goal     the point nearest to the centre of mass among those that have that largest margin (for two feet side by
           side: the line halfway between heels and toes; a body already there is not moved sideways)
  move     the pelvis and everything above it is shifted so that the centre of mass is at the goal; the soles stay
           where they are (leg inverse kinematics), the pelvis keeps its height and orientation
  blend    the 0.64 s before and after a run of such frames carry its first / last shift, fading out linearly, so
           that the motion arrives at the new rest pose without a jump
Frames in motion are untouched. Body poses and velocities are then recomputed exactly as r1wb.library builds them.

usage: python -m r1wb.balance [--subset core] [--out corebal] [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from r1wb import robot
from r1wb.library import FPS, SOFT_LIMIT_FACTOR, SOFT_LIMIT_MARGIN, body_states
from r1wb.tracking.env_cfg import library_path

# A frame the plan can come to rest on (student.stream stand_frame): for the next second the pelvis stays below
# 0.08 m/s and 0.2 rad/s and both feet below 0.12 m/s.
REST = dict(root=0.08, turn=0.2, feet=0.12, seconds=1.0)
BLEND = 32                      # frames before and after a run of rest frames that carry its shift, fading (0.64 s)
ON_GROUND = 0.012               # a footprint corner less than this above the frame's lowest corner stands (m)
FEET_STAY = 1.0e-3              # the feet count as held where they were within this (m, rad)
PELVIS_DROP = 0.005             # the pelvis may come down by this much for a shift over straight legs (m), never up
LEG_TURN = 2.5                  # a leg turns by about shift / leg length; beyond this many times that (plus 0.01 rad)
                                # the leg is being rearranged (a straight leg at its joint limits), which is not this
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
LEGS = ("left", "right")


def footprint(model) -> tuple[np.ndarray, np.ndarray, float]:
  """The foot's collision footprint in the frame of its body: x range, y range, height of the underside."""
  foot = model.body(FEET[0]).id
  low, high = np.full(3, np.inf), np.full(3, -np.inf)
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != foot or not model.geom(g).name.endswith("_collision"):
      continue
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
    rotation = rotation.reshape(3, 3)
    radius = float(model.geom_size[g, 0])
    half = float(model.geom_size[g, 1]) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
    for end in (-half, half):
      point = model.geom_pos[g] + rotation @ np.array([0.0, 0.0, end])
      low, high = np.minimum(low, point - radius), np.maximum(high, point + radius)
  assert np.isfinite(low).all(), "no collision geoms on the foot"
  return np.array([low[0], high[0]]), np.array([low[1], high[1]]), float(low[2])


def largest_margin(points: np.ndarray):
  """Ground points (n, 2) -> the hull's edges (unit normals, offsets: inside is normal . p + offset <= 0) and the
  radius of the largest circle inside it; None if the points do not span an area."""
  try:
    hull = ConvexHull(points)
  except QhullError:
    return None
  normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
  answer = linprog([0.0, 0.0, -1.0], A_ub=np.column_stack([normals, np.ones(len(normals))]), b_ub=-offsets,
                   bounds=[(None, None), (None, None), (0.0, None)], method="highs")
  return (normals, offsets, float(answer.x[2])) if answer.success else None


def nearest_with_margin(point: np.ndarray, normals: np.ndarray, offsets: np.ndarray, margin: float) -> np.ndarray:
  """The point nearest to `point` that keeps `margin` to every edge (Dykstra's alternating projections)."""
  x, carried = point.astype(np.float64).copy(), np.zeros((len(normals), 2))
  for _ in range(200):
    start = x.copy()
    for i in range(len(normals)):
      y = x + carried[i]
      over = normals[i] @ y + offsets[i] + margin
      x_new = y - max(over, 0.0) * normals[i]
      carried[i] = y - x_new
      x = x_new
    if np.linalg.norm(x - start) < 1.0e-7:
      break
  return x


class Balancer:
  def __init__(self):
    self.model = robot.get_spec().compile()
    self.data = mujoco.MjData(self.model)
    self.joints = tuple(robot.JOINTS)
    assert [self.model.joint(i).name for i in range(1, self.model.njnt)] == list(self.joints)
    self.feet = [self.model.body(name).id for name in FEET]
    self.x, self.y, self.under = footprint(self.model)
    self.centre = np.array([self.x.mean(), self.y.mean(), self.under])
    self.leg_dofs = [[6 + i for i, name in enumerate(self.joints) if name.startswith(side) and any(k in name for k in ("hip", "knee", "ankle"))]
                     for side in LEGS]
    assert all(len(dofs) == 6 for dofs in self.leg_dofs)
    low, high = self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1]
    middle, half = 0.5 * (low + high), 0.5 * (high - low) * SOFT_LIMIT_FACTOR - SOFT_LIMIT_MARGIN
    self.low, self.high = middle - half, middle + half
    self.leg = float(np.linalg.norm(self.model.body("left_knee_link").pos) + np.linalg.norm(self.model.body("left_ankle_pitch_link").pos))

  def fits(self, left: float, turned: float, shift: np.ndarray) -> bool:
    """The feet stayed, and the legs only leaned as far as the shift asks."""
    return left <= FEET_STAY and turned <= LEG_TURN * float(np.linalg.norm(shift)) / self.leg + 0.01

  def state(self, q: np.ndarray):
    """-> centre of mass (3), per foot: position, rotation matrix, middle of the sole (3), height of its underside."""
    d = self.data
    d.qpos[:] = q
    mujoco.mj_kinematics(self.model, d)
    mujoco.mj_comPos(self.model, d)
    feet = []
    for foot in self.feet:
      rotation = d.xmat[foot].reshape(3, 3).copy()
      corners = [d.xpos[foot] + rotation @ np.array([a, b, self.under]) for a in self.x for b in self.y]
      feet.append((d.xpos[foot].copy(), rotation, d.xpos[foot] + rotation @ self.centre, np.array(corners)))
    return d.subtree_com[1].copy(), feet

  def shifted(self, q: np.ndarray, shift: np.ndarray, feet) -> tuple[np.ndarray, float, float]:
    """The pelvis moved by `shift` (world x, y), the feet held where `feet` has them -> pose, what is left of the
    feet's error (m or rad), the largest change of a leg joint (rad).
    The legs' twelve joints and the height of the pelvis are solved together (a straight leg cannot get longer:
    the pelvis then comes down by the fraction of a millimetre the shift needs, never up and at most PELVIS_DROP);
    joints stay inside their range."""
    d, m = self.data, self.model
    first = q.copy()
    q = q.copy()
    q[:2] += shift
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    columns = [2] + self.leg_dofs[0] + self.leg_dofs[1]
    address = [2] + [7 + (k - 6) for k in self.leg_dofs[0] + self.leg_dofs[1]]
    low = np.concatenate([[q[2] - PELVIS_DROP], self.low[[k - 6 for k in self.leg_dofs[0] + self.leg_dofs[1]]]])
    high = np.concatenate([[q[2]], self.high[[k - 6 for k in self.leg_dofs[0] + self.leg_dofs[1]]]])
    # A joint the capture already has at (or a hair beyond) its range stays where it is; it may not go further out.
    # (Pulling it inside first would move the foot by a fraction of a millimetre, and a straight leg answers that
    # with a tenth of a radian at the knee.)
    low, high = np.minimum(low, q[address]), np.maximum(high, q[address])

    def residual():
      d.qpos[:] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      errors = []
      for foot, (position, rotation, _, _) in zip(self.feet, feet):
        turn = rotation @ d.xmat[foot].reshape(3, 3).T                       # what is left to turn, in the world
        errors.append(np.concatenate([position - d.xpos[foot],
                                      0.5 * np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]])]))
      return np.concatenate(errors)

    error = residual()
    for _ in range(20):
      if np.abs(error).max() < 1.0e-6:
        break
      rows = []
      for foot in self.feet:
        mujoco.mj_jacBody(m, d, jacp, jacr, foot)
        rows.append(np.concatenate([jacp[:, columns], jacr[:, columns]]))
      jacobian = np.concatenate(rows)                                          # (12, 13)
      free = (q[address] > low + 1.0e-9) | (jacobian.T @ error > 0.0)          # a joint at its limit may only come back in
      free &= (q[address] < high - 1.0e-9) | (jacobian.T @ error < 0.0)
      active = jacobian[:, free]
      step = np.zeros(len(columns))
      step[free] = np.linalg.solve(active.T @ active + 1.0e-7 * np.eye(int(free.sum())), active.T @ error)
      step *= min(1.0, 0.2 / max(float(np.abs(step).max()), 1.0e-9))           # at most 0.2 rad (or m) a turn
      q[address] = np.clip(q[address] + step, low, high)
      error = residual()
    return q, float(np.abs(error).max()), float(np.abs(q[address[1:]] - first[address[1:]]).max())

  def rest_frames(self, qpos: np.ndarray, feet_pos: np.ndarray) -> np.ndarray:
    """The frames from which pelvis, heading and both feet stay at rest for the next second (student.stream)."""
    frames = len(qpos)
    rate = lambda series: np.gradient(series, axis=0) * FPS if frames > 1 else np.zeros_like(series)  # noqa: E731
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    yaw = np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    now = np.maximum.reduce([np.linalg.norm(rate(qpos[:, :2]), axis=1) / REST["root"], np.abs(rate(yaw)) / REST["turn"],
                             np.linalg.norm(rate(feet_pos)[..., :2], axis=2).max(axis=1) / REST["feet"]])
    busy = now.copy()
    for step in range(2, int(REST["seconds"] * FPS) + 1, 2):
      busy = np.maximum(busy, now[np.minimum(np.arange(frames) + step, frames - 1)])
    return busy < 1.0

  def clip(self, qpos: np.ndarray) -> tuple[np.ndarray, dict]:
    """One clip (frames, 7 + 24) -> the same with its rest poses where their stance has the most room, and what was done."""
    frames = len(qpos)
    states = [self.state(q) for q in qpos]
    com = np.array([c for c, _ in states])
    corners = np.array([[feet[k][3] for k in range(2)] for _, feet in states])            # (frames, 2, 4, 3)
    rest = self.rest_frames(qpos, np.array([[feet[k][0] for k in range(2)] for _, feet in states]))
    wanted, share, edges_of = np.zeros((frames, 2)), np.zeros(frames), {}
    for f in np.nonzero(rest)[0]:
      ground = corners[f].reshape(8, 3)
      ground = ground[ground[:, 2] < ground[:, 2].min() + ON_GROUND][:, :2]
      found = largest_margin(ground) if len(ground) >= 3 else None
      if found is None or found[2] < 1.0e-3:
        continue
      normals, offsets, margin = found
      goal = nearest_with_margin(com[f, :2], normals, offsets, margin - 1.0e-3)
      edges_of[f] = (normals, offsets, goal)
      # The pelvis shift that puts the centre of mass on the goal (the legs' share of the mass does not come along).
      moved, now = np.zeros(2), com[f, :2]
      for _ in range(6):
        if np.linalg.norm(goal - now) < 2.0e-4:
          break
        moved = moved + (goal - now)
        q, left, turned = self.shifted(qpos[f], moved, states[f][1])
        if not self.fits(left, turned, moved):
          break
        now = self.state(q)[0][:2]
      wanted[f] = moved
      for part in (1.0, 0.75, 0.5, 0.25):                     # the largest part of it this stance can do
        _, left, turned = self.shifted(qpos[f], part * moved, states[f][1])
        if self.fits(left, turned, part * moved):
          share[f] = part
          break
    # One run of rest frames is one pose held: its frames move by a share that does not jump from frame to frame.
    solved = np.array([f in edges_of for f in range(frames)])
    runs = np.nonzero(np.diff(np.concatenate([[False], solved, [False]]).astype(int)))[0]
    window = 15
    for first, last in zip(runs[::2], runs[1::2] - 1):
      part = share[first:last + 1]
      padded = np.pad(part, window, mode="edge")
      least = np.array([padded[i:i + 2 * window + 1].min() for i in range(len(part))])
      share[first:last + 1] = np.convolve(np.pad(least, window, mode="edge"), np.ones(2 * window + 1) / (2 * window + 1), mode="valid")
    # One shift for every frame of the clip: the rest frames have theirs; between two runs it goes from the one to
    # the other, and fades to nothing where the next rest is more than BLEND frames away. Then a mean over 0.2 s.
    shift = np.zeros((frames, 2))
    shift[solved] = share[solved, None] * wanted[solved]
    runs = np.nonzero(np.diff(np.concatenate([[False], solved, [False]]).astype(int)))[0]
    ends = [(-1, None)] + [(int(last), int(first)) for first, last in zip(runs[::2], runs[1::2] - 1)] + [(frames, None)]
    for (last_a, _), (_, first_b) in zip([(e[0], None) for e in ends[:-1]], [(None, e[1]) for e in ends[1:]]):
      gap = range(last_a + 1, first_b if first_b is not None else frames)
      for f in gap:
        k_a, k_b = f - last_a, (first_b - f) if first_b is not None else BLEND
        from_a = shift[last_a] * max(0.0, 1.0 - k_a / BLEND) if last_a >= 0 else np.zeros(2)
        from_b = shift[first_b] * max(0.0, 1.0 - k_b / BLEND) if first_b is not None else np.zeros(2)
        t = k_a / (k_a + k_b)
        shift[f] = (1.0 - t) * from_a + t * from_b
    window = 11
    padded = np.pad(shift, ((window // 2, window // 2), (0, 0)), mode="edge")
    shift = np.stack([np.convolve(padded[:, i], np.ones(window) / window, mode="valid") for i in range(2)], axis=1)
    out = qpos.copy()
    before, after, margin_before, margin_after, unsolved, blended = [], [], [], [], 0, 0
    for f in np.nonzero(np.linalg.norm(shift, axis=1) > 2.0e-4)[0]:
      q, left, turned = self.shifted(qpos[f], shift[f], states[f][1])
      part = 1.0
      while not self.fits(left, turned, part * shift[f]) and part > 0.2:      # rare: the mean asks for more than this frame can do
        part -= 0.25
        q, left, turned = self.shifted(qpos[f], part * shift[f], states[f][1])
      if not self.fits(left, turned, part * shift[f]):
        unsolved += 1
        shift[f] = 0.0
        continue
      out[f], shift[f] = q, part * shift[f]
      blended += int(not solved[f])
      if solved[f]:
        normals, offsets, goal = edges_of[f]
        now = self.state(q)[0][:2]
        before.append(float(np.linalg.norm(goal - com[f, :2]))); after.append(float(np.linalg.norm(goal - now)))
        margin_before.append(float(-(normals @ com[f, :2] + offsets).max())); margin_after.append(float(-(normals @ now + offsets).max()))
    # What was added to the capture (pelvis place and leg joints) is one signal over time: its mean over 0.1 s takes
    # out the frame where a leg near its limits was solved a little differently from its neighbours. The soles then
    # stay within about a millimetre instead of exactly (reported as feet_moved_m).
    added = out - qpos
    added[:, 3:7] = 0.0
    padded = np.pad(added, ((2, 2), (0, 0)), mode="edge")
    added = np.stack([np.convolve(padded[:, i], np.ones(5) / 5.0, mode="valid") for i in range(added.shape[1])], axis=1)
    out = qpos + added
    feet_moved = 0.0
    for f in np.nonzero(np.abs(added).max(axis=1) > 1.0e-6)[0]:
      _, now = self.state(out[f])
      feet_moved = max(feet_moved, max(float(np.linalg.norm(now[k][0] - states[f][1][k][0])) for k in range(2)))
    size = np.linalg.norm(shift[solved], axis=1) if solved.any() else np.zeros(0)
    jump = float(np.abs(np.diff(out[:, 7:], axis=0) - np.diff(qpos[:, 7:], axis=0)).max()) if frames > 1 else 0.0
    done = dict(frames=frames, rest=int(rest.sum()), moved=int(solved.sum()), unsolved=unsolved, blended=blended,
                full=int((share[solved] > 0.99).sum()), added_joint_step_rad=jump, feet_moved_m=feet_moved,
                to_goal_before_m=float(np.mean(before)) if before else None, to_goal_after_m=float(np.mean(after)) if after else None,
                margin_before_m=float(np.mean(margin_before)) if margin_before else None,
                margin_after_m=float(np.mean(margin_after)) if margin_after else None,
                shift_mean_m=float(size.mean()) if len(size) else 0.0, shift_max_m=float(size.max()) if len(size) else 0.0)
    return out, done


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--subset", default="core", help="the library to read (library_v2_<subset>.npz)")
  parser.add_argument("--out", default="corebal", help="the library to write (library_v2_<out>.npz)")
  parser.add_argument("--check", action="store_true", help="only report what would change (no file is written)")
  args = parser.parse_args()
  source = library_path(args.subset)
  data = np.load(source, allow_pickle=False)
  manifest = json.loads(str(data["manifest"]))
  balancer = Balancer()
  assert manifest["joint_names"] == list(balancer.joints)
  names = manifest["body_names"]
  assert names == [balancer.model.body(i).name for i in range(1, balancer.model.nbody)] and names[0] == "pelvis"
  print(f"footprint measured from the model: x {balancer.x.round(4).tolist()}, y {balancer.y.round(4).tolist()}, underside {balancer.under:.4f} m")
  keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
  arrays = {key: [] for key in keys}
  totals, worst_copy = {}, 0.0
  for clip, start, length in zip(manifest["clips"], data["clip_start"], data["clip_length"]):
    rows = slice(int(start), int(start + length))
    qpos = np.concatenate([data["body_pos_w"][rows, 0], data["body_quat_w"][rows, 0], data["joint_pos"][rows]], axis=1).astype(np.float64)
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    balanced, done = balancer.clip(qpos)
    clip["balanced"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in done.items()}
    group = totals.setdefault(clip["context"], dict(clips=0, frames=0, rest=0, moved=0, full=0, unsolved=0, blended=0, pairs=[], shift=0.0, jump=0.0))
    group["clips"] += 1
    for key in ("frames", "rest", "moved", "full", "unsolved", "blended"):
      group[key] += done[key]
    group["jump"] = max(group["jump"], done["added_joint_step_rad"])
    group["feet"] = max(group.get("feet", 0.0), done["feet_moved_m"])
    if done["moved"]:
      group["pairs"].append((done["to_goal_before_m"], done["to_goal_after_m"], done["margin_before_m"], done["margin_after_m"], done["shift_mean_m"], done["moved"]))
    group["shift"] = max(group["shift"], done["shift_max_m"])
    if args.check:
      continue
    states = body_states(balancer.model, balancer.data, balanced)
    if done["moved"] + done["blended"] == 0:                 # an untouched clip has to come back as it was
      worst_copy = max(worst_copy, float(np.abs(states[2] - data["body_pos_w"][rows]).max()))
    for key, value in zip(keys, states):
      arrays[key].append(value.astype(np.float32))
  mean = lambda pairs, k: sum((p[k] or 0.0) * p[5] for p in pairs) / max(sum(p[5] for p in pairs), 1)  # noqa: E731
  print(f"{'context':12s} clips frames   rest  moved (in full) unsolved blended | to the goal cm before -> after | margin cm before -> after | shift cm mean, max | largest added joint step (rad), feet moved (mm)")
  for context, g in totals.items():
    print(f"{context:12s} {g['clips']:5d} {g['frames']:6d} {g['rest']:6d} {g['moved']:6d} {g['full']:9d} {g['unsolved']:8d} {g['blended']:7d} |"
          f" {100 * mean(g['pairs'], 0):17.2f} -> {100 * mean(g['pairs'], 1):5.2f} | {100 * mean(g['pairs'], 2):14.2f} -> {100 * mean(g['pairs'], 3):5.2f} |"
          f" {100 * mean(g['pairs'], 4):8.2f} {100 * g['shift']:5.1f} | {g['jump']:.3f} {1000 * g.get('feet', 0.0):.2f}")
  print(f"untouched clips reproduced to {worst_copy:.2e} m")
  if args.check:
    return
  out = library_path(args.out)
  manifest.update(subset=args.out, balanced_from=dict(subset=args.subset, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                                     rest=REST, blend_frames=BLEND, footprint=dict(x=balancer.x.tolist(), y=balancer.y.tolist())))
  manifest.pop("sha256", None)
  merged = {key: np.concatenate(value) for key, value in arrays.items()}
  assert merged["joint_pos"].shape == data["joint_pos"].shape
  np.savez_compressed(out, fps=data["fps"], clip_start=data["clip_start"], clip_length=data["clip_length"], manifest=json.dumps(manifest), **merged)
  manifest["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
  out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
  print("written", out)


if __name__ == "__main__":
  main()
