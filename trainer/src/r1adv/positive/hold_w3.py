"""R5 wave 3 of the positive agent: the friction-world far end, EXPLORATORY (coordinator 2026-11-18 07:1x KST,
.agent/adversarial/rounds/R04/06_wave3_plan.md, its '관문 검토 반영' section first). Only the world changes: objective,
reward, termination, exploration, hits and observations are those of R3 wave 1 C0 / R4 wave 2 XB.

  XBF  XB  (C0 environment, plain MLP actor hold_plain.HoldPlainMLP)      + friction world   task adv_hold_w3_xbf
  XBN  XB-null: XBF's code and draws, each hazard cell's mu = its environment's off-cell mu   task adv_hold_w3_xbn
  C0F  C0  (symmetric actor hold_model.HoldSymmetricMLP)                  + friction world   task adv_hold_w3_c0f

Friction world: the negative's frozen spec .agent/adversarial/rounds/R04/r5w3_world_spec.md (sections 1-4 sha256 SPEC),
its numbers copied verbatim into WORLD:
  - per environment and episode one square grid of n x n cells: cell side c ~ U(0.30, 0.60) m, n = ceil(8 / c), rotated by
    theta ~ U(0, 90 deg) about the environment origin (the robot's start), grid centre shifted by U(-c/2, c/2) on each
    axis; each cell hazardous with probability 0.5, a hazardous cell's sliding friction mu ~ U(0.15, 0.35); outside the
    field and on the other cells the floor keeps the environment's own foot friction (the training environment's startup
    randomisation U(0.3, 1.2), unchanged). No safe zone: the start footprint can be covered.
  - drawn whenever an environment (re)starts an episode and fixed in world coordinates during it; the draws read only the
    environment index and the restart order (never the robot, the hits or the policy): one dedicated torch.Generator on
    the environment device, seed child_seed(child_seed(0, "R05w3-train-world-v1"), f"seed{run_seed}") % 2**63
    (r1adv.negative.seedtree), every restarting batch drawn in environment-index order, the same number of draws in every
    cell, so XBF and XBN of one seed take the same layouts (until their training diverges).
  - applied before every physics substep (hooked after the robot's write_data_to_sim, as the negative's evaluation world
    r1adv/negative/hazards.py f8116c67 with safe_r_m = 0): each of the 14 foot capsules gets as its sliding friction
    (geom_friction axis 0; torsional / rolling friction and every other geom unchanged) the value of the cell under its
    centre. The R1 foot capsules out-rank the floor in contact priority, so their value is the contact's (checked at
    construction). geom_xpos is one substep (5 ms) stale, as in the evaluation world.
  - not observed: the actor and critic observation configurations are C0's / XB's unchanged; nothing reads the friction,
    the cells or a contact parameter.
Runs train from scratch, 1200 updates (labels 0 ... 1199), a checkpoint every SAVE_INTERVAL = 50 updates.
Logged (training diagnostics in extras["log"], never observed): World/feet_on_hazard (share of the foot capsules over a
hazard cell), World/mu_feet (mean sliding friction written to the foot capsules), World/hazard_share (share of hazardous
cells inside the fields drawn in that step), World/restarts (layouts drawn so far per environment).
"""
from __future__ import annotations

import math

import torch
from mjlab.managers.event_manager import EventTermCfg

from r1adv.negative.seedtree import child_seed
from r1adv.positive import hold_w1 as W1
from r1adv.positive import hold_w2 as W2

SPEC = dict(file=".agent/adversarial/rounds/R04/r5w3_world_spec.md", sections="1-4",
            sha256="dea59d5aa945fb328a1e78f26a73aca1af6e56d23cedcfada23555841abbddd3", frozen="2026-11-18 07:10 KST")
WORLD = dict(cell_m=(0.30, 0.60), theta_deg=(0.0, 90.0), shift="U(-c/2, c/2) per axis", field_m=8.0, p_hazard=0.5,
             mu=(0.15, 0.35), safe_r_m=0.0, off_cell="environment foot friction (startup U(0.3, 1.2))",
             seed_root="R05w3-train-world-v1", reference="r1adv/negative/hazards.py f8116c67 cells (friction)")
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))
SAVE_INTERVAL = 50
CELLS = {
  "XBF": dict(base="C0", plain=True, null=False),
  "XBN": dict(base="C0", plain=True, null=True),
  "C0F": dict(base="C0", plain=False, null=False),
}


def world_seed(run_seed: int) -> int:
  """Spec section 2: the seed of the dedicated world generator of a run."""
  return child_seed(child_seed(0, WORLD["seed_root"]), f"seed{int(run_seed)}") % 2 ** 63


class friction_world:
  """Reset event (mode 'reset', see the module docstring): draws the friction cells of every (re)starting environment and,
  through a hook after the robot's write_data_to_sim, writes the foot capsules' sliding friction before every physics
  substep. null=True (XBN): the same draws and the same computation, every capsule keeps its environment's floor value."""

  model_fields = ("geom_friction",)          # mjlab's EventManager expands the field per environment (already by foot DR)

  def __init__(self, cfg, env):
    import mujoco
    p = cfg.params
    if p.get("spec_sha256") != SPEC["sha256"]:
      raise ValueError("friction world configured from a different world spec")
    self.env, dev, count = env, env.device, env.num_envs
    mj = env.sim.mj_model
    ids = []
    for name in FOOT_GEOMS:
      gid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}")
      if gid < 0 or mj.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        raise ValueError(f"foot capsule robot/{name} not found as a capsule geom")
      ids.append(gid)
    others = [g for g in range(mj.ngeom) if not (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("robot/")]
    if not others or int(mj.geom_priority[others].max()) >= int(mj.geom_priority[ids].min()):
      raise ValueError("the foot capsules must out-rank the floor in contact priority (their friction is the contact's)")
    self.foot_ids = torch.tensor(ids, dtype=torch.long, device=dev)
    self.null = bool(p["null"])
    self.write_cells = torch.tensor(not self.null, device=dev)
    self.origins = env.scene.env_origins
    self.g = torch.Generator(device=dev)
    self.seed = world_seed(int(env.cfg.seed))
    self.g.manual_seed(self.seed)
    self.c_lo, self.c_hi = WORLD["cell_m"]
    self.th_lo, self.th_hi = (math.radians(v) for v in WORLD["theta_deg"])
    self.mu_lo, self.mu_hi = WORLD["mu"]
    self.nmax = math.ceil(WORLD["field_m"] / self.c_lo)
    self.c = torch.ones(count, device=dev)
    self.theta = torch.zeros(count, device=dev)
    self.n = torch.zeros(count, dtype=torch.long, device=dev)
    self.x0 = torch.zeros(count, device=dev)
    self.y0 = torch.zeros(count, device=dev)
    self.values = torch.zeros(count, self.nmax, self.nmax, device=dev)      # hazard mu of each cell, 0 = no hazard
    self.restarts = torch.zeros(count, dtype=torch.long, device=dev)
    self.grid = torch.arange(self.nmax, device=dev)
    self.friction = self.xpos = self.nominal = None                           # bound at the first physics substep
    self.last_substep = 0
    self.writes = 0
    self.draw(torch.arange(count, device=dev))                               # the first episodes (restart 0)
    robot = env.scene["robot"]
    original = robot.write_data_to_sim

    def hooked():
      original()
      self.apply()
    robot.write_data_to_sim = hooked

  def reset(self, env_ids=None):
    pass

  # ------------------------------------------------------------------------------------------------------------- layout
  def draw(self, env_ids: torch.Tensor) -> torch.Tensor:
    """New cells for env_ids, in environment-index order; returns the share of hazardous cells inside each field."""
    ids = torch.sort(env_ids).values
    k, dev, nmax = len(ids), ids.device, self.nmax
    u = torch.rand(k, 4, device=dev, generator=self.g)                       # c, theta, shift x, shift y
    hazard = torch.rand(k, nmax, nmax, device=dev, generator=self.g) < WORLD["p_hazard"]
    mu = self.mu_lo + (self.mu_hi - self.mu_lo) * torch.rand(k, nmax, nmax, device=dev, generator=self.g)
    c = self.c_lo + (self.c_hi - self.c_lo) * u[:, 0]
    n = torch.ceil(WORLD["field_m"] / c).long().clamp(max=nmax)
    inside = (self.grid[None, :, None] < n[:, None, None]) & (self.grid[None, None, :] < n[:, None, None])
    self.c[ids] = c
    self.theta[ids] = self.th_lo + (self.th_hi - self.th_lo) * u[:, 1]
    self.n[ids] = n
    self.x0[ids] = -n * c / 2 + (u[:, 2] - 0.5) * c
    self.y0[ids] = -n * c / 2 + (u[:, 3] - 0.5) * c
    self.values[ids] = torch.where(hazard & inside, mu, torch.zeros_like(mu))
    self.restarts[ids] += 1
    return (hazard & inside).float().sum((1, 2)) / (n * n).float()

  def cell_value(self, xy: torch.Tensor) -> torch.Tensor:
    """(envs, P, 2) world points -> (envs, P) hazard mu of the cell under each point (0: no hazard or outside the field)."""
    d = xy - self.origins[:, None, :2]
    cos, sin = torch.cos(-self.theta)[:, None], torch.sin(-self.theta)[:, None]
    lx, ly = cos * d[..., 0] - sin * d[..., 1], sin * d[..., 0] + cos * d[..., 1]
    i = torch.floor((lx - self.x0[:, None]) / self.c[:, None]).long()
    j = torch.floor((ly - self.y0[:, None]) / self.c[:, None]).long()
    n = self.n[:, None]
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    rows = torch.arange(len(xy), device=xy.device)[:, None].expand_as(i)
    v = self.values[rows, i.clamp(0, self.nmax - 1), j.clamp(0, self.nmax - 1)]
    return torch.where(ok, v, torch.zeros_like(v))

  # ------------------------------------------------------------------------------------------------------------- apply
  def apply(self) -> None:
    """Before a physics substep: the foot capsules' sliding friction from the cells under their centres (once per substep;
    the extra write_data_to_sim after an auto-reset and the calls before the first step are skipped)."""
    env = self.env
    substep = int(env._sim_step_counter)
    if substep == 0 or substep == self.last_substep:
      return
    self.last_substep = substep
    if self.friction is None:
      self.friction = env.sim.model.geom_friction
      self.xpos = env.sim.data.geom_xpos
      if self.friction.dim() != 3 or self.friction.shape[0] != env.num_envs:
        raise RuntimeError("geom_friction is not expanded per environment")
      self.nominal = self.friction[:, self.foot_ids, 0].clone()             # each environment's own floor friction
    v = self.cell_value(self.xpos[:, self.foot_ids, :2])
    hazard = v > 0
    mu = torch.where(hazard & self.write_cells, v, self.nominal)
    self.friction[:, self.foot_ids, 0] = mu
    self.writes += 1
    if substep % env.cfg.decimation == 0:
      log = env.extras.setdefault("log", {})
      log["World/feet_on_hazard"] = hazard.float().mean()
      log["World/mu_feet"] = mu.mean()
      log["World/restarts"] = self.restarts.float().mean()

  def __call__(self, env, env_ids, null: bool, spec_sha256: str):
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
      return
    share = self.draw(env_ids)
    env.extras.setdefault("log", {})["World/hazard_share"] = share.mean()


# ---------------------------------------------------------------------------------------------------------- configurations
def env_cfg(null: bool, play: bool = False):
  """C0's environment (also XB's) plus the friction world (training only; play / evaluation environments are C0's)."""
  cfg = W1.hold_w1_env_cfg(lead=False, posture="point", play=play)
  if play:
    return cfg
  cfg.events["friction_world"] = EventTermCfg(func=friction_world, mode="reset",
                                              params=dict(null=bool(null), spec_sha256=SPEC["sha256"]))
  return cfg


def rl_cfg(plain: bool, max_iterations: int = 1500):
  """XB's (plain=True) or C0's PPO recipe, a checkpoint every 50 updates (hold_w2.rl_cfg without the XA / XAn options)."""
  return W2.rl_cfg(floor=False, plain=plain, ar1=False, save_interval=SAVE_INTERVAL, max_iterations=max_iterations)


def cell_env_cfg(cell: str, play: bool = False):
  return env_cfg(CELLS[cell]["null"], play=play)


def cell_rl_cfg(cell: str, max_iterations: int = 1500):
  return rl_cfg(CELLS[cell]["plain"], max_iterations=max_iterations)


def record() -> dict:
  return dict(spec=dict(SPEC), world=dict(WORLD), cells=CELLS, foot_geoms=list(FOOT_GEOMS), save_interval=SAVE_INTERVAL,
              base_tasks=dict(XB="adv_hold_w2_xb", C0="adv_hold_w1_c0"))
