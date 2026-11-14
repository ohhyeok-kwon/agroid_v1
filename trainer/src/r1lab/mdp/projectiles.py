"""Batched physical projectiles; contact flags are reward bookkeeping only."""
from __future__ import annotations

import math
import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg


def _ball_spec():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="free")
  body.add_geom(name="sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(.07, 0, 0), mass=1., friction=(.6, .02, .01),
                contype=1, conaffinity=1)
  return spec


def projectile_entities(count=8):
  return {f"ball{i}": EntityCfg(spec_fn=_ball_spec,
          init_state=EntityCfg.InitialStateCfg(pos=(25.+2*i, 0., 5.)))
          for i in range(count)}


def projectile_sensors(count=8, history_length=4):
  return tuple(ContactSensorCfg(name=f"projectile_hit_{i}",
      primary=ContactMatch(mode="body", pattern="ball", entity=f"ball{i}"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"), reduce="netforce", history_length=history_length)
      for i in range(count))


def ball_contact_until(env):
  """Per-env policy step until which a recent ball contact counts (-1: none)."""
  if not hasattr(env, "_r1_ball_until"):
    env._r1_ball_until = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
  return env._r1_ball_until


def ballistic_velocity(origin, target, speed):
  """Simulation.launch_at's lower arc, with speed in m/s and world positions."""
  delta = target - origin
  gravity = delta.new_tensor([0., 0., -9.81])
  b = speed.square() + (delta * gravity).sum(-1)
  disc = b.square() - gravity.square().sum() * delta.square().sum(-1)
  if (disc < 0).any():
    raise ValueError("Projectile target is unreachable at this speed")
  flight = (2 * delta.square().sum(-1) / (b + disc.sqrt())).sqrt()
  return delta / flight[:, None] - gravity * flight[:, None] / 2


class ProjectileBurst:
  """Step event; free slots are never stolen from in-flight projectiles.

  Curriculum starts at the first event call, including resumed training. Clean
  episodes contain no projectiles; other disturbance events remain independent.
  """
  def __init__(self, cfg, env):
    self.env = env
    self.count = cfg.params.get("count", 8)
    self.clean_fraction = cfg.params.get("clean_fraction", .25)
    if self.count < 1 or not 0 <= self.clean_fraction <= 1:
      raise ValueError("Invalid projectile count or clean fraction")
    self.balls = [env.scene[f"ball{i}"] for i in range(self.count)]
    self.body = env.scene["robot"].find_bodies("torso_link")[0][0]
    shape = (env.num_envs, self.count)
    self.active = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.hit = torch.zeros_like(self.active)
    self.age = torch.zeros(shape, device=env.device)
    self.wait = torch.zeros(env.num_envs, device=env.device)
    self.left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.clean = torch.zeros_like(self.wait, dtype=torch.bool)
    self.angle = torch.zeros_like(self.wait)
    self.height = torch.zeros_like(self.wait)
    self.spacing = torch.zeros_like(self.wait)
    self.speed = torch.zeros_like(self.wait)
    self.launched = torch.zeros_like(self.wait)
    self.hits = torch.zeros_like(self.wait)
    self.origin = None
    self.last_observed = -1
    self.last_called = -1
    env._r1_projectile_event = self
    self.reset()

  def _ids(self, ids=None):
    all_ids = torch.arange(self.env.num_envs, device=self.env.device)
    return all_ids if ids is None else all_ids[ids] if isinstance(ids, slice) else ids

  def _park(self, i, ids):
    if len(ids) == 0:
      return
    state = self.balls[i].data.default_root_state[ids].clone()
    state[:, :3] = self.env.scene.env_origins[ids] + state.new_tensor([25.+2*i, 0., 5.])
    state[:, 7:] = 0
    self.balls[i].write_root_state_to_sim(state, env_ids=ids)

  def reset(self, env_ids=None):
    ids = self._ids(env_ids)
    self.active[ids] = False
    self.hit[ids] = False
    self.age[ids] = 0
    self.left[ids] = 0
    self.wait[ids] = 2. + 2. * torch.rand(len(ids), device=self.env.device)
    self.clean[ids] = torch.rand(len(ids), device=self.env.device) < self.clean_fraction
    self.launched[ids] = 0
    self.hits[ids] = 0
    for i in range(self.count):
      self._park(i, ids)

  def observe_contacts(self):
    env = self.env
    if self.last_observed == env.common_step_counter:
      return
    self.last_observed = env.common_step_counter
    touched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i in range(self.count):
      force = env.scene[f"projectile_hit_{i}"].data.force_history
      contact = force.norm(dim=-1).amax(dim=(1, 2)) > 1e-3
      contact &= self.active[:, i]
      new = contact & ~self.hit[:, i]
      self.hits += new.float()
      self.hit[:, i] |= contact
      touched |= contact
    if touched.any():
      from r1lab.mdp.recovery import mark_disturbance
      ids = touched.nonzero(as_tuple=False).flatten()
      mark_disturbance(env, ids)
      # Ball-only mark (bookkeeping, never observed): consumers that must not react to pushes.
      ball_contact_until(env)[ids] = env.common_step_counter + round(1.5 / env.step_dt)

  def __call__(self, env, env_ids=None, count=8, clean_fraction=.25,
               speed_range=(6., 18.), warmup_steps=7200, spacing=(.3, .5),
               cooldown=(3., 5.), lifetime=5.):
    del env_ids, clean_fraction
    if count != self.count or speed_range[0] <= 0 or speed_range[1] < speed_range[0]:
      raise ValueError("Invalid projectile parameters")
    if self.last_called == env.common_step_counter:
      return
    self.last_called = env.common_step_counter
    # Consume previous physics contacts before expiring/reusing slots.
    self.observe_contacts()
    if self.origin is None:
      self.origin = env.common_step_counter
    fraction = min(1., max(0., (env.common_step_counter-self.origin)/max(1, warmup_steps)))
    max_speed = min(speed_range[1], max(speed_range[0], 9.) +
                    max(0., speed_range[1]-max(speed_range[0], 9.))*fraction)
    shot_limit = min(3, 1 + int(3*fraction))
    self.age += self.active * env.step_dt
    self.active &= self.age < lifetime
    self.wait -= env.step_dt
    ready = (self.wait <= 1e-6) & ~self.clean & (~self.active).any(-1)
    fresh = ready & (self.left == 0)
    ids = fresh.nonzero(as_tuple=False).flatten()
    n = len(ids)
    self.left[ids] = torch.randint(1, shot_limit+1, (n,), device=env.device)
    self.angle[ids] = 2*math.pi*torch.rand(n, device=env.device)
    self.height[ids] = -.05 + .1*torch.rand(n, device=env.device)
    self.spacing[ids] = spacing[0] + (spacing[1]-spacing[0])*torch.rand(n, device=env.device)
    self.speed[ids] = speed_range[0] + (max_speed-speed_range[0])*torch.rand(n, device=env.device)
    ids = ready.nonzero(as_tuple=False).flatten()
    if len(ids):
      slot = (~self.active[ids]).long().argmax(-1)
      robot = env.scene["robot"].data
      q = robot.root_link_quat_w[ids]
      yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                        1-2*(q[:, 2].square()+q[:, 3].square()))
      theta = yaw + self.angle[ids]
      target = robot.body_com_pos_w[ids, self.body].clone()
      target[:, 2] += self.height[ids]
      origin = target + torch.stack((2*theta.cos(), 2*theta.sin(), torch.full_like(theta, .1)), -1)
      velocity = ballistic_velocity(origin, target, self.speed[ids])
      for i in range(self.count):
        mask = slot == i
        selected = ids[mask]
        if len(selected):
          state = self.balls[i].data.default_root_state[selected].clone()
          state[:, :3] = origin[mask]
          state[:, 7:] = 0
          state[:, 7:10] = velocity[mask]
          self.balls[i].write_root_state_to_sim(state, env_ids=selected)
          self.active[selected, i] = True
          self.hit[selected, i] = False
          self.age[selected, i] = 0
      self.launched[ids] += 1
      self.left[ids] -= 1
      rest = cooldown[0] + (cooldown[1]-cooldown[0])*torch.rand(len(ids), device=env.device)
      self.wait[ids] = torch.where(self.left[ids] > 0, self.spacing[ids], rest)
    for i in range(self.count):
      self._park(i, (~self.active[:, i]).nonzero(as_tuple=False).flatten())
    log = env.extras.setdefault("log", {})
    log["Metrics/projectile_launched"] = self.launched.sum()
    log["Metrics/projectile_hit_fraction"] = self.hits.sum()/self.launched.sum().clamp_min(1)
    log["Metrics/projectile_max_speed"] = torch.tensor(max_speed, device=env.device)
