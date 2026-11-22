"""Worlds beyond the flat floor: ground shapes and thrown bodies, for evaluation and training.

Ground (apply_ground): one height-field patch replaces the plane. Every environment stands on the same patch (the
worlds are separate, the geometry is shared) and starts on a flat disc at its centre, so that the reset pose is valid;
the shape begins around it. Heights that the flat-floor tools read from the world's z are read from a ray under the
pelvis instead (sensor "root_ground"): root_height, fallen, ground_under.

  rough<cm>        bumps up to <cm> high, 0.2 m apart, smooth between them (an uneven floor)
  gravel<cm>       every 5 cm cell at its own height up to <cm> (a fixed bed of stones: the stones do not roll)
  up<deg> / down<deg>   a ramp along +x that begins 0.75 m ahead of the start: walking forward climbs / descends
  side<deg>        the ground ahead tilts about the direction of travel (the left side is higher): walking forward
                   crosses the slope
  plane<deg>       the whole floor is one tilted plane around the flat start (training: forward crosses the slope,
                   left climbs, right descends)

Thrown bodies (throwables, Thrower): balls and boxes with a free joint each, without weight in flight like the
Operation Center's ball test before its launch (gravity compensated: they fly straight), parked high above the scene
until thrown. Their positions are world coordinates (an entity without a robot is not moved to its environment's origin).
"""
from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from r1wb.performance import cached_read, device_indices

SIZE = 24.0                 # m, side of the patch (a 20 s walk at 0.35 m/s stays on it)
START = 0.75                # m, radius of the flat start disc / distance at which a ramp begins
BLEND = 0.30                # m over which the shape fades in around the start
BALL = dict(radius=0.07, mass=1.0)          # the Operation Center's projectile (app/operation_center/simulation.py)
BOX = dict(half=0.15, mass=5.0)             # a 30 cm box
PARK = 60.0                 # m above the start, where the bodies wait (below a plane they would be pushed out of it)


def _fade(distance: np.ndarray) -> np.ndarray:
  """0 inside the start disc, 1 from START + BLEND on (smooth)."""
  t = np.clip((distance - START) / BLEND, 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def heights(name: str, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, size: float = SIZE,
            share: float = 1.0) -> np.ndarray:
  """Ground height (m) at the points (x, y) around the start of a patch `size` m wide; 0 on the start disc.
  share: the shape's size as a fraction of the one the name states (training patches grow with their level)."""
  kind, value = re.fullmatch(r"([a-z]+)(\d+(?:\.\d+)?)", name).groups()
  value = float(value) * share
  if kind == "plane":       # a floor tilted about x (its left side higher), flat on the start disc
    return math.tan(math.radians(value)) * y * _fade(np.hypot(x, y))
  if kind in ("up", "down", "side"):
    slope = math.tan(math.radians(value))
    ahead = np.clip(x - START, 0.0, None)
    if kind == "side":
      return slope * y * _fade(x + START)                    # the tilt is complete START + BLEND ahead of the start
    # a ramp that begins smoothly: the first BLEND metres curve into the slope
    run = np.where(ahead < BLEND, ahead * ahead / (2.0 * BLEND), ahead - BLEND / 2.0)
    return (slope if kind == "up" else -slope) * run
  if kind in ("rough", "gravel"):
    top = value / 100.0
    if kind == "gravel":
      cell = 0.05
      grid = rng.uniform(0.0, top, size=(int(size / cell) + 2, int(size / cell) + 2))
      field = grid[np.clip(((y + size / 2) / cell).astype(int), 0, grid.shape[0] - 1),
                   np.clip(((x + size / 2) / cell).astype(int), 0, grid.shape[1] - 1)]
    else:
      cell = 0.2
      grid = rng.uniform(0.0, top, size=(int(size / cell) + 2, int(size / cell) + 2))
      u, v = (x + size / 2) / cell, (y + size / 2) / cell
      i, j = np.clip(u.astype(int), 0, grid.shape[1] - 2), np.clip(v.astype(int), 0, grid.shape[0] - 2)
      a, b = u - i, v - j
      a, b = a * a * (3.0 - 2.0 * a), b * b * (3.0 - 2.0 * b)
      field = (grid[j, i] * (1 - a) * (1 - b) + grid[j, i + 1] * a * (1 - b)
               + grid[j + 1, i] * (1 - a) * b + grid[j + 1, i + 1] * a * b)
    return (field - top / 2.0) * _fade(np.hypot(x, y))      # zero mean around the start height
  raise ValueError(f"unknown ground: {name}")


@dataclass(kw_only=True)
class GroundCfg(SubTerrainCfg):
  name: str
  resolution: float = 0.1       # m per sample; every sample under a foot is a contact candidate
  grows: bool = False           # the shape's size follows the patch's difficulty (0..1) up to the size the name states

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    body = spec.body("terrain")
    columns, rows = int(self.size[0] / self.resolution), int(self.size[1] / self.resolution)
    # MuJoCo height fields: rows run along y, columns along x; the samples span the whole side
    x = np.linspace(-self.size[0] / 2, self.size[0] / 2, columns)[None, :].repeat(rows, 0)
    y = np.linspace(-self.size[1] / 2, self.size[1] / 2, rows)[:, None].repeat(columns, 1)
    z = heights(self.name, x, y, rng, size=self.size[0], share=max(difficulty, 0.05) if self.grows else 1.0)
    low, high = float(z.min()), float(z.max())
    span = max(high - low, 1.0e-3)
    field = spec.add_hfield(name=f"ground_{uuid.uuid4().hex}", size=[self.size[0] / 2, self.size[1] / 2, span, 0.5],
                            nrow=rows, ncol=columns, userdata=((z - low) / span).astype(np.float32).flatten().tolist())
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                         pos=[self.size[0] / 2, self.size[1] / 2, low], rgba=[0.45, 0.47, 0.5, 1.0])
    return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]),
                         geometries=[TerrainGeometry(geom=geom, hfield=field)])


def apply_ground(cfg, name: str, seed: int = 7) -> None:
  """The scene's plane becomes the ground `name`; "" or "flat" keeps the plane (the ray sensor is added either way,
  so that the same code measures both)."""
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in ("root_ground", "feet_ground_height")) + ray_sensors()
  cfg.terminations["fallen"] = TerminationTermCfg(func=fallen)
  if name in ("", "flat"):
    return
  terrain = cfg.scene.terrain
  terrain.terrain_type = "generator"
  terrain.max_init_terrain_level = None
  terrain.terrain_generator = TerrainGeneratorCfg(
    seed=seed, size=(SIZE, SIZE), border_width=1.0, num_rows=1, num_cols=1, color_scheme="none",
    sub_terrains={name: GroundCfg(name=name, size=(SIZE, SIZE), resolution=0.05 if name.startswith("gravel") else 0.1)})
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 400)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)


@cached_read
def root_height(env) -> torch.Tensor:
  """Pelvis height above the ground under it (m)."""
  if "root_ground" in env.scene.sensors:
    return env.scene["root_ground"].data.heights.reshape(env.num_envs, -1)[:, 0]
  return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


@cached_read
def ground_under(env) -> torch.Tensor:
  """Height of the ground under the pelvis above the start level (m)."""
  root = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return root - root_height(env)


@cached_read
def support_under(env) -> torch.Tensor:
  """Height of the ground under the two feet (their mean) above the start level (m): the surface the robot stands on.
  On a slope it differs from the ground under the pelvis by the slope times the pelvis' lead over the feet.
  Zero in a scene without the feet's ray sensor (the flat floor) and on the step an episode starts (the robot is then
  on the flat start disc and the rays are still those of the episode before)."""
  return feet_under(env).mean(dim=1)


@cached_read
def feet_under(env) -> torch.Tensor:
  """(environments, 2) height of the ground under the left and the right foot above the start level (m); zero where
  support_under is zero (no ray sensor, the step an episode starts)."""
  if "feet_ground_height" not in env.scene.sensors:
    return torch.zeros(env.num_envs, 2, device=env.device)
  robot = env.scene["robot"]
  ids = [robot.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
  foot_z = robot.data.body_link_pos_w[:, device_indices(env, ids), 2]
  clearance = env.scene["feet_ground_height"].data.heights.reshape(env.num_envs, -1)[:, :2]
  level = foot_z - clearance - env.scene.env_origins[:, 2:3]
  return torch.where((env.episode_length_buf < 1).unsqueeze(1), torch.zeros_like(level), level)


@cached_read
def standing_height(env) -> torch.Tensor:
  """Pelvis height above the surface the robot stands on (m). On the flat floor: its height above the start level."""
  root = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return root - support_under(env)


def ray_sensors() -> tuple:
  """The rays that measure the ground: under the pelvis (root_ground) and under each foot (feet_ground_height)."""
  common = dict(ray_alignment="world", pattern=RingPatternCfg.single_ring(radius=0.02, num_samples=4), max_distance=3.0,
                exclude_parent_body=True, include_geom_groups=(0,), reduction="mean", debug_vis=False)
  return (TerrainHeightSensorCfg(name="root_ground", frame=(ObjRef(type="body", name="pelvis", entity="robot"),), **common),
          TerrainHeightSensorCfg(name="feet_ground_height",
                                 frame=tuple(ObjRef(type="body", name=f"{side}_ankle_roll_link", entity="robot")
                                             for side in ("left", "right")), **common))


def training_ground(cfg, seed: int = 11) -> None:
  """The training scene's floor becomes a field of patches (10 m, eight levels each, the shape growing with the
  level): flat, bumps up to 10 cm, a bed of stones up to 4 cm, slopes up and down in every direction up to 25 deg, a
  floor tilted about the direction of travel up to 25 deg, low stairs up to 6 cm. Every patch has a flat start in
  its middle. The environments are spread over the kinds by the shares below and over the levels at random."""
  from mjlab.terrains import config as presets
  size = (10.0, 10.0)
  slope = math.tan(math.radians(25.0))
  kinds = {
    "flat": presets.flat(proportion=0.15, size=size),
    "bumps": GroundCfg(name="rough10", grows=True, proportion=0.2, size=size),
    "stones": GroundCfg(name="gravel4", grows=True, proportion=0.1, size=size, resolution=0.05),
    "tilted": GroundCfg(name="plane25", grows=True, proportion=0.1, size=size),
    "down": presets.hf_pyramid_slope(slope_range=(0.0, slope), platform_width=1.5, proportion=0.175, size=size),
    "up": presets.hf_pyramid_slope_inv(slope_range=(0.0, slope), platform_width=1.5, proportion=0.175, size=size),
    "stairs_down": presets.pyramid_stairs(step_height_range=(0.0, 0.06), step_width=0.4, platform_width=1.5,
                                          proportion=0.05, size=size),
    "stairs_up": presets.pyramid_stairs_inv(step_height_range=(0.0, 0.06), step_width=0.4, platform_width=1.5,
                                            proportion=0.05, size=size),
  }
  terrain = cfg.scene.terrain
  terrain.terrain_type = "generator"
  terrain.max_init_terrain_level = None
  terrain.terrain_generator = TerrainGeneratorCfg(seed=seed, size=size, border_width=4.0, num_rows=8, num_cols=len(kinds),
                                                  curriculum=True, color_scheme="none", sub_terrains=kinds)
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in ("root_ground", "feet_ground_height")) + ray_sensors()
  # The contact budget also has to hold the model's own start state: the robot at the world's origin, where four
  # patches meet (125 contacts counted there on 11-14).
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 250)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)


def raise_build_pose(cfg, height: float = 8.0) -> None:
  """The pose the model is BUILT in (qpos0: every robot at the world's origin, where terrain patches meet and the edge
  of a slope stands above the robot) is raised into the air. No episode starts there (the command places the robot;
  the entity's own initial state is untouched), but the simulator refuses a contact budget below what that pose holds
  and the constraint budget followed it: 125 contacts and 530 rows on the training field (measured 11-14), against
  14 contacts per world and 230 rows at most in use while training."""
  from dataclasses import replace
  robot = cfg.scene.entities["robot"]
  base = robot.spec_fn

  def spec_fn():
    spec = base()
    root = spec.body("pelvis")
    root.pos = [float(root.pos[0]), float(root.pos[1]), float(root.pos[2]) + height]
    return spec

  cfg.scene.entities["robot"] = replace(robot, spec_fn=spec_fn)


def fallen(env, min_height: float = 0.35, max_tilt: float = 0.9) -> torch.Tensor:
  """student.stream.fallen with the height taken above the ground under the pelvis."""
  tilt = torch.acos((-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
  return (root_height(env) < min_height) | (tilt > max_tilt)


def ground_reference(env, term, at: str = "pelvis") -> None:
  """DIAGNOSTIC (evaluation only): the reference that `term` gives the policy stands on the ground under the robot's
  pelvis instead of on the start level. `term` is whatever the policy's observation reads: the motion command, or
  the view that stands in for it when System 2 writes the plan (eval.chunk_view.ChunkView, s2.runtime.S2View).
  Asks whether a policy that reads the reference's height fails on a slope because of that reading alone.
  at: "pelvis" = the ground under the pelvis; "feet" = the ground under the feet (the surface it stands on)."""
  base = type(term)
  members = {}
  lift = ground_under if at == "pelvis" else support_under
  if isinstance(getattr(base, "anchor_pos_w", None), property):
    def anchor_pos_w(self):
      position = base.anchor_pos_w.fget(self).clone()
      position[:, 2] += lift(env)
      return position
    members["anchor_pos_w"] = property(anchor_pos_w)
  if isinstance(getattr(base, "body_pos_w", None), property):
    def body_pos_w(self):
      position = base.body_pos_w.fget(self).clone()
      position[..., 2] += lift(env).unsqueeze(1)
      return position
    members["body_pos_w"] = property(body_pos_w)
  assert "anchor_pos_w" in members, f"{base.__name__} has no reference height to raise"
  term.__class__ = type("OnTheGround" + base.__name__, (base,), members)


def _body(kind: str, index: int):
  def spec() -> mujoco.MjSpec:
    made = mujoco.MjSpec()
    body = made.worldbody.add_body(name="body", pos=[0.0, 0.0, PARK + index], gravcomp=1.0)
    body.add_freejoint(name="free")
    if kind == "ball":
      body.add_geom(name="shape", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL["radius"], 0, 0], mass=BALL["mass"],
                    rgba=[0.9, 0.35, 0.1, 1.0], group=2)
    else:
      body.add_geom(name="shape", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX["half"]] * 3, mass=BOX["mass"],
                    rgba=[0.7, 0.55, 0.3, 1.0], group=2)
    return made
  return spec


def add_throwables(cfg, kind: str, count: int) -> tuple[str, ...]:
  """`count` bodies of one kind ("ball" or "box") in every world; returns their entity names."""
  names = tuple(f"{kind}_{i}" for i in range(count))
  for i, name in enumerate(names):
    # the entity's initial state places the free joint: without it the body would start at the environment's origin
    cfg.scene.entities[name] = EntityCfg(spec_fn=_body(kind, i),
                                         init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, PARK + i)))
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 160)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 800)
  return names


class Thrower:
  """Throws the bodies added by add_throwables at a point of the robot, from `distance` m away."""

  def __init__(self, env, names: tuple[str, ...], kind: str, distance: float = 1.5):
    self.env, self.kind, self.distance = env, kind, distance
    self.bodies = [env.scene[name] for name in names]
    self.mass = (BALL if kind == "ball" else BOX)["mass"]
    self.reach = BALL["radius"] if kind == "ball" else BOX["half"]
    self.used = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.since = torch.full((len(names), env.num_envs), 1.0e3, device=env.device)
    self.flight = torch.zeros(len(names), env.num_envs, 3, device=env.device)        # thrown velocity
    # what each throw gave away: mass x (thrown velocity - velocity `measure` s later) along the flight (N s)
    self.given = torch.full((len(names), env.num_envs), float("nan"), device=env.device)
    self.measure = torch.zeros(len(names), env.num_envs, device=env.device)          # s after the throw

  def lead_steps(self, speed: torch.Tensor, dt: float) -> torch.Tensor:
    """Policy steps between the throw and the contact (the body starts `distance` from the target's surface)."""
    return torch.round((self.distance - self.reach) / speed.clamp_min(0.1) / dt).long().clamp_min(1)

  def throw(self, envs: torch.Tensor, target: torch.Tensor, along: torch.Tensor, speed: torch.Tensor,
            target_velocity: torch.Tensor | None = None) -> None:
    """envs: environment ids; target (n, 3) world point to hit; along (n, 3) unit direction of flight; speed (n,).
    The aim leads a moving target by its velocity over the flight time."""
    if envs.numel() == 0:
      return
    flight = (self.distance / speed.clamp_min(0.1)).unsqueeze(1)
    aim = target if target_velocity is None else target + target_velocity * flight
    start = aim - along * self.distance
    pose = torch.cat([start, torch.tensor([1.0, 0.0, 0.0, 0.0], device=start.device).expand(len(envs), 4)], dim=1)
    velocity = torch.cat([along * speed.unsqueeze(1), torch.zeros(len(envs), 3, device=start.device)], dim=1)
    which = self.used[envs] % len(self.bodies)
    for index, body in enumerate(self.bodies):
      chosen = which == index
      if chosen.any():
        body.write_root_link_pose_to_sim(pose[chosen], envs[chosen])
        body.write_root_link_velocity_to_sim(velocity[chosen], envs[chosen])
        self.since[index, envs[chosen]] = 0.0
        self.flight[index, envs[chosen]] = velocity[chosen, :3]
        self.given[index, envs[chosen]] = float("nan")
        self.measure[index, envs[chosen]] = self.distance / speed[chosen].clamp_min(0.1) + 0.15
    self.used[envs] += 1

  def after_step(self, dt: float, life: float = 0.6) -> None:
    """Bodies go back to their parking place `life` seconds after their contact was measured."""
    self.since += dt
    for index, body in enumerate(self.bodies):
      due = (self.since[index] >= self.measure[index]) & (self.since[index] < 1.0e3) & torch.isnan(self.given[index])
      if due.any():
        speed = torch.norm(self.flight[index], dim=1).clamp_min(1e-6)
        kept = (body.data.root_link_lin_vel_w * self.flight[index]).sum(1) / speed
        self.given[index] = torch.where(due, self.mass * (speed - kept), self.given[index])
      old = ((self.since[index] > self.measure[index] + life) & (self.since[index] < 1.0e3)).nonzero().flatten()
      if old.numel():
        pose = torch.zeros(len(old), 7, device=self.env.device)
        pose[:, 2], pose[:, 3] = PARK + index, 1.0
        body.write_root_link_pose_to_sim(pose, old)
        body.write_root_link_velocity_to_sim(torch.zeros(len(old), 6, device=self.env.device), old)
        self.since[index, old] = 1.0e3
