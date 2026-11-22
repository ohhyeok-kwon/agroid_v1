"""Explicit, paired physical cases for the opt-in proper-r1-v1 evaluation.

This module never advances time or selects successful initial states. It places a
frozen row once on the actual compiled geometry, and reports unsuccessful setup.
The only random values used are the explicit values in the case manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "r1wb-proper-case-pack-v1"
WORLDS = ("flat", "side10", "side20", "up20", "gravel4", "rough10")
NOISE = ("N0", "N1_shared_sensor_v1")
DIRECTIONS = {"front": (1., 0., 0.), "back": (-1., 0., 0.),
              "left": (0., 1., 0.), "right": (0., -1., 0.)}
DT = .02


def sha256(path):
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def _array(row, name, length):
  value = np.asarray(row.get(name), dtype=np.float64)
  if value.shape != (length,) or not np.isfinite(value).all():
    raise ValueError(f"{row.get('block_id')}: {name} must contain {length} finite numbers")
  return value


def _int(row, name, lower, upper):
  value = row.get(name)
  if type(value) is not int or not lower <= value <= upper:
    raise ValueError(f"{row.get('block_id')}: {name} must be an integer in [{lower}, {upper}]")
  return value


def _quat_mul(a, b):
  w, x, y, z = a
  v, i, j, k = b
  return np.array((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                   w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v))


def _quat_rpy(rpy):
  r, p, y = np.asarray(rpy) / 2
  return _quat_mul(_quat_mul((math.cos(y), 0, 0, math.sin(y)),
                            (math.cos(p), 0, math.sin(p), 0)),
                   (math.cos(r), math.sin(r), 0, 0))


def _rotation(q):
  w, x, y, z = np.asarray(q) / np.linalg.norm(q)
  return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _hash_arrays(*values):
  digest = hashlib.sha256()
  for value in values:
    value = np.ascontiguousarray(value)
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
  return digest.hexdigest()


def apply_ground_reference(env, term):
  """Use current foot rays even at reset; the terrain and episode clock stay intact.

  The frozen training helper assumes reset is on the flat pad. Proper evaluation
  explicitly spawns outside that pad, so its first-frame assumption cannot apply.
  Invoke instead of (not on top of) world.ground_reference on command and S2View.
  """
  base = type(term)
  # S2View delegates unknown instance attributes to the command. Its own wrapper
  # must not be confused with the command's already-installed wrapper.
  if getattr(base, "_proper_ground_reference", False):
    raise ValueError("Proper ground reference must be applied only once")
  def support():
    ray = env.scene["feet_ground_height"].data
    floor = ray.frame_pos_w[:, :, 2] - ray.heights.reshape(env.num_envs, -1)
    return (floor - env.scene.env_origins[:, 2:3]).mean(dim=1)
  members = {"_proper_ground_reference": True}
  if isinstance(getattr(base, "anchor_pos_w", None), property):
    def anchor_pos_w(self):
      value = base.anchor_pos_w.fget(self).clone()
      value[:, 2] += support()
      return value
    members["anchor_pos_w"] = property(anchor_pos_w)
  if isinstance(getattr(base, "body_pos_w", None), property):
    def body_pos_w(self):
      value = base.body_pos_w.fget(self).clone()
      value[..., 2] += support().unsqueeze(1)
      return value
    members["body_pos_w"] = property(body_pos_w)
  if "anchor_pos_w" not in members:
    raise ValueError("Ground-relative term has no anchor_pos_w property")
  term.__class__ = type("ProperGround"+base.__name__, (base,), members)


def validate_manifest(data, *, allow_subset=False):
  if data.get("schema") != SCHEMA:
    raise ValueError(f"Expected {SCHEMA}")
  rows = data.get("rows")
  if not isinstance(rows, list) or not rows:
    raise ValueError("A case pack must contain explicit rows")
  if len(rows) != 64 and not (allow_subset and 1 <= len(rows) <= 64):
    raise ValueError("A production case pack requires exactly 64 blocks; smoke subsets are explicit")
  joint_names = data.get("joint_names")
  if (not isinstance(joint_names, list) or len(joint_names) != 24
      or len(set(joint_names)) != 24 or not all(isinstance(n, str) for n in joint_names)):
    raise ValueError("joint_names must explicitly give the 24 unique entity-order names")
  if len({r.get("block_id") for r in rows}) != len(rows):
    raise ValueError("Duplicate block IDs")
  if len({r.get("initialization_seed") for r in rows}) != len(rows):
    raise ValueError("Initialisation seeds must be unique within a case pack")
  conditions = {(r.get("world"), r.get("noise_mode"), r.get("terrain_seed")) for r in rows}
  if len(conditions) != 1:
    raise ValueError("A case pack is one world/noise/terrain-seed condition")
  world, noise, terrain_seed = next(iter(conditions))
  if world not in WORLDS or noise not in NOISE or terrain_seed != 7:
    raise ValueError(f"Unregistered condition {world}/{noise}/seed={terrain_seed}")
  identities = set()
  for row in rows:
    if not isinstance(row.get("block_id"), str) or not row["block_id"]:
      raise ValueError("Every row requires a block ID")
    if row.get("initial_reference") != {"library": "corebal", "clip_selection": "first_nonmirrored_stand_in_manifest", "frame": 0}:
      raise ValueError("Every case must explicitly select common corebal first standing clip/frame 0")
    for name in ("initialization_seed", "noise_seed", "schedule_seed"):
      _int(row, name, 0, 2**32-1)
    spawn = _array(row, "spawn_xy_m", 2)
    offset = _array(row, "root_offset_xyz_m", 3)
    if not 0 <= offset[2] <= .003 + 1e-12:
      raise ValueError("Root z is an explicit nonnegative lift after geometry placement")
    if world != "flat" and (spawn + offset[:2])[0] < 1.25:
      raise ValueError("Nonflat case must start beyond the unchanged flat/transition pad")
    for name in ("root_offset_rpy_rad", "root_linear_velocity_offset_mps", "root_angular_velocity_offset_radps"):
      _array(row, name, 3)
    _array(row, "joint_position_offset_rad", 24)
    _array(row, "joint_velocity_offset_radps", 24)
    twist = _array(row, "twist_mps_radps", 3)
    if not (np.array_equal(twist, (0., 0., 0.)) or np.array_equal(twist, (.35, 0., 0.))):
      raise ValueError("Impulse case packs currently contain stand and forward 0.35 only")
    if row.get("direction") not in DIRECTIONS:
      raise ValueError("Unknown force direction")
    impulse = row.get("impulse_ns")
    if isinstance(impulse, bool) or not isinstance(impulse, (int, float)) or not math.isfinite(impulse) or impulse <= 0:
      raise ValueError("Impulse must be a finite positive value")
    first = _int(row, "first_hit_control_step", 125, 175)
    count = _int(row, "burst_count", 1, 8)
    gap = _int(row, "burst_gap_control_steps", 8, 8)
    duration = _int(row, "hit_duration_control_steps", 1, 1)
    horizon = _int(row, "episode_control_steps", 600, 600)
    if first + (count-1)*gap + duration + 200 > horizon:
      raise ValueError("Case must retain four seconds after the final force interval")
    identity = json.dumps({k: row[k] for k in row if k.startswith("root_") or k.startswith("joint_") or k == "spawn_xy_m"}, sort_keys=True)
    if identity in identities:
      raise ValueError("Duplicate physical initial-state recipe; a different seed alone is insufficient")
    identities.add(identity)
  return data


def _foot_geoms(model):
  import mujoco
  result = []
  for geom in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
    if re.search(r"(?:^|/)(left|right)_foot[1-7]_collision$", name):
      if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise ValueError(f"Unsupported foot collision geometry: {name}")
      result.append(geom)
  if len(result) != 14:
    raise ValueError(f"Expected 14 unchanged R1 foot capsules, got {len(result)}")
  return result


def _terrain_ray(model, data, xy):
  """Ray against actual compiled group-0 floor, with the triangle normal."""
  import mujoco
  point = np.array((xy[0], xy[1], 20.), dtype=np.float64)
  normal = np.zeros(3, dtype=np.float64)
  geom = np.array([-1], dtype=np.int32)
  distance = mujoco.mj_ray(model, data, point, np.array((0., 0., -1.)),
                           np.array((1, 0, 0, 0, 0, 0), dtype=np.uint8), True, -1, geom, normal)
  if distance < 0 or normal[2] <= 0:
    raise ValueError(f"Terrain ray missed/upward face absent at {xy.tolist()}")
  return float(point[2]-distance), normal, int(geom[0])


def _samples(model, data, foot_geoms):
  """Capsule lower support samples, augmented by exact contact QA below.

  Dense sampled rays are for placement and are not a proof of no interpenetration.
  MuJoCo contact distances, saved separately, check the resulting pose.
  """
  points = []
  for geom in foot_geoms:
    matrix = data.geom_xmat[geom].reshape(3, 3)
    radius, half_length = model.geom_size[geom, :2]
    for fraction in np.linspace(-1., 1., 9):
      center = data.geom_xpos[geom] + matrix[:, 2] * (half_length*fraction)
      points.append((geom, center - np.array((0., 0., radius))))
      for angle in np.arange(8)*math.pi/4:
        # Lower hemisphere at 45 degrees: edge support matters on gravel/slope.
        points.append((geom, center + radius*np.array((math.cos(angle), math.sin(angle), -1.))/math.sqrt(2)))
  return points


def place_on_geometry(model, qpos, qvel, free_q, foot_geoms, *, lift=0., clearance=.002):
  """CPU kinematic/contact query only; no time integration, retries, or rollout."""
  import mujoco
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  samples = _samples(model, data, foot_geoms)
  correction = max(_terrain_ray(model, data, point[:2])[0] + clearance - point[2] for _, point in samples)
  qpos = qpos.copy()
  qpos[free_q[2]] += correction + lift
  # Fixed deterministic geometry correction. A failed geometry check is retained,
  # never used to choose another row or to test a policy before accepting it.
  adjustments = []
  for _ in range(6):
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    needed = 0.
    for contact in data.contact[:data.ncon]:
      geoms = (int(contact.geom[0]), int(contact.geom[1]))
      if not any(g in foot_geoms for g in geoms):
        continue
      if not any(model.geom_group[g] == 0 for g in geoms):
        continue
      vertical = abs(float(contact.frame[2]))
      if contact.dist < clearance and vertical > .2:
        needed = max(needed, (clearance-float(contact.dist))/vertical)
    if needed < 1e-9:
      break
    qpos[free_q[2]] += needed
    adjustments.append(needed)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  contacts = []
  for contact in data.contact[:data.ncon]:
    g0, g1 = map(int, contact.geom)
    contacts.append(dict(geom_ids=[g0, g1], distance_m=float(contact.dist),
                         position_w=contact.pos.tolist(), normal_w=contact.frame[:3].tolist()))
  feet = {}
  for side in ("left", "right"):
    geom = next(g for g in foot_geoms if model.geom(g).name.endswith(f"{side}_foot4_collision"))
    axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
    ends = [data.geom_xpos[geom] + sign*axis*model.geom_size[geom, 1] for sign in (-1., 1.)]
    # Sorting by the capsule/body +x axis is robust to a root yaw perturbation.
    body = int(model.geom_bodyid[geom])
    forward = data.xmat[body].reshape(3, 3)[:, 0]
    ends.sort(key=lambda p: float(np.dot(p, forward)))
    feet[side] = {}
    for label, point in zip(("heel", "toe"), ends):
      point = point - np.array((0., 0., model.geom_size[geom, 0]))
      floor, normal, ground = _terrain_ray(model, data, point[:2])
      feet[side][label] = dict(sole_sample_w=point.tolist(), floor_height_m=floor,
                               normal_w=normal.tolist(), terrain_geom=ground,
                               vertical_clearance_m=float(point[2]-floor))
  finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
  min_distance = min((c["distance_m"] for c in contacts), default=None)
  equality = np.asarray(data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
  eq_position = np.asarray(data.efc_pos)[equality]
  eq_velocity = np.asarray(data.efc_vel)[equality]
  valid = finite and (min_distance is None or min_distance >= -1e-5)
  return qpos, dict(geometry_valid=valid, finite=finite, min_contact_distance_m=min_distance,
                    vertical_placement_m=float(correction+lift+sum(adjustments)),
                    contact_corrections_m=adjustments, contacts=contacts, feet=feet,
                    requested_vertical_clearance_m=clearance, final_positive_lift_m=lift,
                    equality_position_max_abs=float(np.abs(eq_position).max()) if eq_position.size else 0.,
                    equality_velocity_max_abs=float(np.abs(eq_velocity).max()) if eq_velocity.size else 0.,
                    equality_position_residuals=eq_position.tolist(), equality_velocity_residuals=eq_velocity.tolist(),
                    contact_qa="CPU MuJoCo forward on static compiled model; no settling or policy outcome tested")


@dataclass
class CasePack:
  path: Path
  manifest: dict
  digest: str
  _forces_world: np.ndarray | None = field(default=None, repr=False)
  _tensor_cache: dict = field(default_factory=dict, repr=False)

  @property
  def rows(self): return self.manifest["rows"]
  @property
  def world(self): return self.rows[0]["world"]
  @property
  def noise_mode(self): return self.rows[0]["noise_mode"]
  @property
  def paired(self): return True
  @property
  def cases(self): return [(i, r["direction"], r["impulse_ns"], 0) for i, r in enumerate(self.rows)]
  @property
  def clips(self): return list(range(len(self.rows)))
  @property
  def triples(self): return [tuple(r["twist_mps_radps"]) for r in self.rows]
  @property
  def twin_of(self): return list(range(len(self.rows), 2*len(self.rows)))
  @property
  def noise_seeds(self): return [r["noise_seed"] for r in self.rows]*(2 if self.paired else 1)
  @property
  def num_envs(self): return (2 if self.paired else 1)*len(self.rows)
  @property
  def horizon(self): return 600

  def configure_cfg(self, cfg):
    """Call after apply_ground, before constructing the environment."""
    cfg.scene.num_envs = self.num_envs
    cfg.episode_length_s = max(float(cfg.episode_length_s), 30.)
    command = cfg.commands["motion"]
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0., 0.)
    command.pin_clips = True
    command.sampling_mode = "uniform"
    command.clip_list = tuple(command.clip_list[:1])
    if not command.clip_list:
      raise ValueError("Common corebal standing clip must be selected before configuring a case pack")
    motion_manifest = json.loads(Path(command.motion_file).with_suffix(".manifest.json").read_text())
    standing = [i for i, clip in enumerate(motion_manifest["clips"])
                if clip["context"] == "stand" and not clip["mirrored"]]
    if not standing or command.clip_list != (standing[0],):
      raise ValueError("Initializer requires first nonmirrored standing clip in the corebal manifest")
    for key in ("match_noise", "hold_after_hit", "lag_grace"):
      if hasattr(command, key): setattr(command, key, 0.)
    if not math.isclose(float(cfg.sim.mujoco.timestep)*cfg.decimation, DT, abs_tol=1e-12):
      raise ValueError("Proper case-pack control step must be exactly 0.02 seconds")

  def initialize(self, env, term):
    """After env.reset/ground_reference; before S2/policy/history construction.

    Derived simulator/sensor values and observation history are refreshed. Command
    phase and reference are not advanced, nor is this a general state-fork API.
    """
    import torch
    from r1wb.robot import JOINTS
    if list(JOINTS) != self.manifest["joint_names"]:
      raise ValueError("Case-pack joint order differs from the deployed entity contract")
    if env.num_envs != self.num_envs:
      raise ValueError(f"Case-pack worlds {self.num_envs}, environment worlds {env.num_envs}")
    if getattr(env, "_proper_case_initialized", False):
      raise ValueError("Explicit physical cases may only be initialized once")
    robot, model = env.scene["robot"], env.sim.mj_model
    reference_before_repin = term.time_steps.detach().cpu().tolist()
    # env.reset computes commands at dt=0. LibraryMotionCommand nevertheless
    # increments matched references by one frame, whereas frozen manufacturer
    # streams stay at frame 0. Re-run the same deterministic reset for every
    # policy, without compute/physics stepping, before extracting the baseline.
    term._resample_command(torch.arange(self.num_envs, device=env.device))
    free_q = robot.indexing.free_joint_q_adr.detach().cpu().numpy().astype(int)
    free_v = robot.indexing.free_joint_v_adr.detach().cpu().numpy().astype(int)
    if free_q.shape != (7,) or free_v.shape != (6,):
      raise ValueError("Case-pack initializer requires one floating robot root")
    indices = [robot.joint_names.index(name) for name in JOINTS]
    joint_q = robot.indexing.joint_q_adr.detach().cpu().numpy()[indices].astype(int)
    joint_v = robot.indexing.joint_v_adr.detach().cpu().numpy()[indices].astype(int)
    origins = env.scene.env_origins.detach().cpu().numpy()
    baseline_q = env.sim.data.qpos[0].detach().cpu().numpy().copy()
    baseline_v = env.sim.data.qvel[0].detach().cpu().numpy().copy()
    baseline_q[free_q[:3]] -= origins[0]
    original_quat = baseline_q[free_q[3:]].copy()
    original_omega_w = _rotation(original_quat) @ baseline_v[free_v[3:]]
    expected_clip = int(term.cfg.clip_list[0])
    expected_frame = int(term.clip_start[expected_clip])
    if not bool(((term.clip_ids == expected_clip) & (term.time_steps == expected_frame)).all()):
      raise ValueError("Initial command is not on the frozen common standing clip/frame 0")
    foot_geoms = _foot_geoms(model)
    # Group 0 is already the production ray-sensor floor contract. Refuse robot
    # geoms in it rather than silently ray-casting the robot as terrain.
    robot_geoms = robot.indexing.geom_ids.detach().cpu().numpy()
    if np.any(model.geom_group[robot_geoms] == 0):
      raise ValueError("Robot geometry unexpectedly occupies the terrain ray group 0")
    q_all, v_all, details, force_directions = [], [], [], []
    for index, row in enumerate(self.rows):
      q, v = baseline_q.copy(), baseline_v.copy()
      offset = np.asarray(row["root_offset_xyz_m"])
      q[free_q[:3]] += origins[index]
      q[free_q[:2]] = origins[index, :2] + np.asarray(row["spawn_xy_m"]) + offset[:2]
      quat = _quat_mul(_quat_rpy(row["root_offset_rpy_rad"]), original_quat)
      quat /= np.linalg.norm(quat)
      q[free_q[3:]] = quat
      v[free_v[:3]] += row["root_linear_velocity_offset_mps"]
      v[free_v[3:]] = _rotation(quat).T @ (original_omega_w + row["root_angular_velocity_offset_radps"])
      q[joint_q] += row["joint_position_offset_rad"]
      v[joint_v] += row["joint_velocity_offset_radps"]
      q, qa = place_on_geometry(model, q, v, free_q, foot_geoms, lift=float(offset[2]))
      heading = math.atan2(_rotation(quat)[1, 0], _rotation(quat)[0, 0])
      c, s = math.cos(heading), math.sin(heading)
      local = DIRECTIONS[row.get("direction", "front")]
      direction = [c*local[0]-s*local[1], s*local[0]+c*local[1], 0.]
      qa.update(block_id=row["block_id"], initialization_seed=row["initialization_seed"],
                env_index=index, twin_env_index=index+len(self.rows) if self.paired else None, initial_yaw_rad=heading,
                force_world_unit=direction, qpos_sha256=_hash_arrays(q), qvel_sha256=_hash_arrays(v),
                root_position_relative_origin_m=(q[free_q[:3]]-origins[index]).tolist(),
                root_quaternion_wxyz=quat.tolist())
      q_all.append(q); v_all.append(v); details.append(qa); force_directions.append(direction)
    q_actual, v_actual = np.stack(q_all), np.stack(v_all)
    if self.paired:
      q_twin = q_actual.copy()
      q_twin[:, free_q[:3]] += origins[len(self.rows):] - origins[:len(self.rows)]
      all_q, all_v = np.concatenate((q_actual, q_twin)), np.concatenate((v_actual, v_actual))
    else:
      all_q, all_v = q_actual, v_actual
    # Cast once before copying: hash/QA below describe actual float32 sim inputs,
    # not higher precision temporary placement values.
    q_tensor = torch.as_tensor(all_q, device=env.device, dtype=env.sim.data.qpos.dtype)
    v_tensor = torch.as_tensor(all_v, device=env.device, dtype=env.sim.data.qvel.dtype)
    env.sim.data.qpos.copy_(q_tensor)
    env.sim.data.qvel.copy_(v_tensor)
    env.sim.data.qacc_warmstart.zero_()
    env.sim.forward()
    term._pending_forward = False
    env.scene.update(0.)
    env.sim.sense()
    if hasattr(term, "update_relative_body_poses"):
      term.update_relative_body_poses()
    env.observation_manager.reset()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    actual_q, actual_v = q_tensor.detach().cpu().numpy(), v_tensor.detach().cpu().numpy()
    state_hashes = []
    for i, qa in enumerate(details):
      relative_q = actual_q[i].copy(); relative_q[free_q[:3]] -= origins[i]
      qa["normalized_state_sha256"] = _hash_arrays(relative_q, actual_v[i])
      qa["actual_qpos_sha256"] = _hash_arrays(actual_q[i])
      qa["actual_qvel_sha256"] = _hash_arrays(actual_v[i])
      if self.paired:
        twin_q = actual_q[i+len(self.rows)].copy(); twin_q[free_q[:3]] -= origins[i+len(self.rows)]
        qa["pair_qpos_max_abs"] = float(np.abs(relative_q-twin_q).max())
        qa["pair_qvel_max_abs"] = float(np.abs(actual_v[i]-actual_v[i+len(self.rows)]).max())
      state_hashes.append(qa["normalized_state_sha256"])
    if len(set(state_hashes)) != len(state_hashes):
      raise ValueError("Distinct case recipes collapsed to duplicate actual initial physical states")
    self._forces_world = np.asarray(force_directions, dtype=np.float64)
    self._tensor_cache.clear()
    env._proper_case_initialized = True
    geometry_hash = _hash_arrays(model.geom_type, model.geom_pos, model.geom_quat,
                                 model.geom_size, model.hfield_size, model.hfield_data)
    return dict(schema="r1wb-proper-case-initialization-v1", case_pack_sha256=self.digest,
                case_pack_path=str(self.path), world=self.world, terrain_seed=7,
                initial_reference=self.rows[0]["initial_reference"],
                reference_global_frames_before_common_repin=reference_before_repin,
                common_repin="_resample_command(all_ids), once; no dt=0 command compute or time integration",
                initial_reference_clip_ids=term.clip_ids.detach().cpu().tolist(),
                initial_reference_global_frames=term.time_steps.detach().cpu().tolist(),
                initial_reference_library_sha256=sha256(term.cfg.motion_file),
                initial_reference_manifest_sha256=sha256(Path(term.cfg.motion_file).with_suffix(".manifest.json")),
                geometry_sha256=geometry_hash, geometry_generalization=False,
                initialization="common reference + explicit physical offsets + deterministic terrain placement",
                initialization_time_advanced=False, independent_physical_hashes=len(set(state_hashes)),
                paired_world_order="actual rows followed by their own one-to-one no-hit twins" if self.paired else "command rows; no twin worlds",
                all_geometry_valid=all(row["geometry_valid"] for row in details), rows=details,
                limitations=["CPU geometry QA uses static compiled model; verify play has no model randomization",
                             "No guarantee of stable settling; policy pre-hit failures are retained",
                             "Initial qpos/qvel pairing is not a full simulator/hidden-policy-state fork",
                             "Actuated joint offsets are not projected onto passive equality constraints; residuals are reported",
                             "Pre-hit observations/actions must still be checked for each pair"])

  def tensors(self, device):
    import torch
    key = str(device)
    if key not in self._tensor_cache:
      if self._forces_world is None:
        raise RuntimeError("Initialize physical cases before using world-frame force directions")
      def tensor(name, dtype):
        return torch.tensor([r[name] for r in self.rows], device=device, dtype=dtype)
      self._tensor_cache[key] = dict(
        hit_step=tensor("first_hit_control_step", torch.long), impulse=tensor("impulse_ns", torch.float32),
        burst_count=tensor("burst_count", torch.long), gap_steps=tensor("burst_gap_control_steps", torch.long),
        hit_steps=tensor("hit_duration_control_steps", torch.long),
        forces_world_unit=torch.as_tensor(self._forces_world, device=device, dtype=torch.float32),
        hit_height=torch.tensor([r.get("hit_height_m", .1) for r in self.rows], device=device, dtype=torch.float32))
    return self._tensor_cache[key]

  def schedule(self, step, device):
    values = self.tensors(device)
    since = int(step) - values["hit_step"]
    return ((since >= 0) & (since < values["burst_count"]*values["gap_steps"])
            & (since.remainder(values["gap_steps"]) < values["hit_steps"]))


def load_case_pack(path, args=None, *, allow_subset=False):
  path = Path(path).resolve()
  data = validate_manifest(json.loads(path.read_text()), allow_subset=allow_subset)
  pack = CasePack(path, data, sha256(path))
  if args is not None:
    if getattr(args, "comparison_contract", None) != "proper-r1-v1":
      raise ValueError("Explicit case packs require --comparison-contract proper-r1-v1")
    if getattr(args, "world", None) != pack.world:
      raise ValueError("--world differs from the frozen case pack")
    if getattr(args, "observation_profile", None) != pack.noise_mode:
      raise ValueError("--observation-profile differs from the frozen case pack")
    for name in ("steer", "throw", "graft", "reflex_gain", "arm_response_scale", "rate_feedback", "leg_boost", "course", "rate_feedback_hip"):
      value = getattr(args, name, None)
      expected = 1.0 if name == "leg_boost" else None
      if (name == "arm_response_scale" and value is not None) or (value and value != expected):
        raise ValueError(f"Case packs prohibit the diagnostic/action override --{name.replace('_', '-')}")
    if getattr(args, "quiet", "none") != "none" or getattr(args, "hold_after_hit", 0) != 0:
      raise ValueError("Case packs prohibit quiet overrides and hit-time holds")
  return pack


@dataclass
class CommandPack(CasePack):
  """Independent physical starts for the separate no-impulse command suite."""
  @property
  def paired(self): return False
  @property
  def twin_of(self): return []
  @property
  def scripts(self): return [r["segments_control_steps_vx_vy_wz"] for r in self.rows]
  @property
  def horizon(self): return max(r["episode_control_steps"] for r in self.rows)
  @property
  def episode_steps(self): return [r["episode_control_steps"] for r in self.rows]
  def schedule(self, step, device):
    raise ValueError("Command packs have no force schedule")


def load_command_pack(path, args=None, *, allow_subset=False):
  path = Path(path).resolve()
  data = json.loads(path.read_text())
  if data.get("schema") != "r1wb-proper-command-pack-v1":
    raise ValueError("Expected r1wb-proper-command-pack-v1")
  rows = data.get("rows", [])
  if len(rows) != 52 and not (allow_subset and 1 <= len(rows) <= 52):
    raise ValueError("Command suite requires 52 independent blocks per noise condition")
  equivalent = []
  for row in rows:
    segments = row.get("segments_control_steps_vx_vy_wz")
    if not isinstance(segments, list) or not segments:
      raise ValueError("Commands require explicit [duration_control_steps, vx, vy, wz] segments")
    duration = 0
    for segment in segments:
      if (not isinstance(segment, list) or len(segment) != 4 or type(segment[0]) is not int
          or segment[0] <= 0 or not np.isfinite(segment).all() or abs(segment[3]) > .6
          or math.hypot(segment[1], segment[2]) > .35+1e-12):
        raise ValueError("Invalid/unsupported physical command segment")
      duration += segment[0]
    if duration != row.get("episode_control_steps") or duration not in (600, 1000):
      raise ValueError("Command horizon must exactly equal its frozen script duration")
    if row.get("world") != "flat":
      raise ValueError("Frozen independent command suite is flat only")
    equivalent.append(dict(row, twist_mps_radps=[0., 0., 0.], direction="front", impulse_ns=12.,
                           first_hit_control_step=125, burst_count=1, burst_gap_control_steps=8,
                           hit_duration_control_steps=1, episode_control_steps=600))
  validate_manifest(dict(data, schema=SCHEMA, rows=equivalent), allow_subset=True)
  pack = CommandPack(path, data, sha256(path))
  if args is not None:
    if getattr(args, "comparison_contract", None) != "proper-r1-v1":
      raise ValueError("Command packs require proper-r1-v1")
    if getattr(args, "observation_profile", None) != pack.noise_mode:
      raise ValueError("Command pack observation profile mismatch")
    if getattr(args, "world", "flat") not in ("", "flat") or getattr(args, "steer", ""):
      raise ValueError("Command packs prohibit nonflat world/steering")
  return pack
