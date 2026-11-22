"""Signal-dependent execution noise (SDN) body model -- simulated TRAINING / EVALUATION WORLD only (positive 12, R8 W0;
rounds/R04/08B_r8_synthesis.md 2 (b) and 4.4; gate rules rounds/R04/positive/r8_body/sdn_gate_rules.md).

Principle (Harris & Wolpert 1998, minimum-variance motor control): the torque a joint actually produces scatters in
proportion to the torque it is asked for.  The R1 "torque command" is the PD torque of its MuJoCo <position> actuator
    u = kp (q_target - q) - kd dq        (gainprm[0] = kp, biasprm = [0, -kp, -kd], ctrllimited False in mjlab)
and with this module the simulated body produces
    sdn    tau = clip(m u),   m = max(0, 1 + c eps)     gainprm[0], biasprm[1], biasprm[2] of every actuator scaled by m
    white  tau = clip(u + s eps)                        biasprm[0] = s eps (additive, state independent; s in N m)
    none   tau = clip(u)
eps is a stationary N(0, 1) process per environment and actuator, advanced once per control step (one fresh innovation
xi ~ N(0, 1) per environment, actuator and policy step, 20 ms) and held over that step's physics substeps:
    eps_t = rho eps_{t-1} + sqrt(1 - rho^2) xi_t,   rho = exp(-dt / tau_c)   (tau_c = 0: rho = 0, independent steps)
clip = the actuator force range (the effort limit), applied by MuJoCo after the noise exactly as before.  The max(0, .)
keeps a gain from changing sign (P(1 + c eps < 0) = 4e-4 at c = 0.3); it is the only change to the Gaussian.

Correlation time (the gate's choice and reasons are in the rules file): the minimum-variance argument needs successive
commands' noise to decorrelate within a reaction (0.1-0.3 s), so that spreading an impulse over several steps cuts the
scatter of its sum while one big step keeps all of it; the human model it borrows injects white noise into the neural
command and low-passes it through the muscle (30-40 ms), which removes the unphysiological fast content.  tau_c = 0.04 s
(rho = 0.61 per 20 ms step) keeps the first and gives the second; tau_c = 0 (independent steps) is the limit case.

It is a property of the simulated body: no reward term, no filter or change of the policy's actions, no time window,
nothing keyed to hits.  Nothing of it is on the deploy path: the real R1, its deploy contract and its controller never
import this module.

Use
  evaluation   SdnBody(env, seed) after the environments exist; set_levels(kind, c, white_std, tau_c); call step() once
               per control step before env.step (it advances eps and writes the per-environment actuator parameters).
  training     EventTermCfg(func=sdn_step_event, mode="step", params=dict(c=..., kind="sdn", seed=..., tau_c=...)) in the env's
               events: mjlab calls it once per env step after the physics substeps (its eps acts on the next step) and
               its reset() after the reset events (domain randomisation of the PD gains is kept as the base and the
               noise multiplies it).  Not used by any registered task yet (W1 decides).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

KINDS = {"none": 0, "sdn": 1, "white": 2}
GAIN_FIELDS = ("actuator_gainprm", "actuator_biasprm")


def module_sha256() -> str:
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SdnBody:
  """Per-environment execution noise on the position actuators of one entity (see module docstring)."""

  def __init__(self, env, seed: int, entity_name: str = "robot"):
    import mujoco
    self.env = env
    self.device = env.device
    self.n = env.num_envs
    entity = env.scene[entity_name]
    self.ctrl = entity.indexing.ctrl_ids.long().to(self.device)
    mjm = env.sim.mj_model
    ids = self.ctrl.cpu().numpy()
    if not ((mjm.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED).all()
            and (mjm.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE).all()):
      raise ValueError("SdnBody expects MuJoCo <position> actuators (fixed gain, affine bias)")
    if (abs(mjm.actuator_gainprm[ids, 0] + mjm.actuator_biasprm[ids, 1]) > 1e-9).any() or (mjm.actuator_biasprm[ids, 0] != 0).any():
      raise ValueError("actuator gain / bias are not kp, (0, -kp, -kd)")
    self.joint_names = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(mjm.actuator_trnid[i, 0])) for i in ids]
    self.nu = len(ids)
    missing = [f for f in GAIN_FIELDS if f not in getattr(env.sim, "_expanded_fields", set())]
    if missing:
      env.sim.expand_model_fields(tuple(missing))     # re-captures the CUDA graphs (mjlab)
    model = env.sim.model
    self.base_g0 = model.actuator_gainprm[:, self.ctrl, 0].clone()
    self.base_b0 = model.actuator_biasprm[:, self.ctrl, 0].clone()
    self.base_b1 = model.actuator_biasprm[:, self.ctrl, 1].clone()
    self.base_b2 = model.actuator_biasprm[:, self.ctrl, 2].clone()
    self.kind = torch.zeros(self.n, dtype=torch.long, device=self.device)
    self.c = torch.zeros(self.n, device=self.device)
    self.white_std = torch.zeros(self.n, self.nu, device=self.device)
    self.rho = torch.zeros(self.n, device=self.device)
    self.dt = float(env.step_dt)
    self.gen = torch.Generator(device=self.device)
    self.gen.manual_seed(int(seed))
    self.eps = torch.zeros(self.n, self.nu, device=self.device)
    self.m = torch.ones(self.n, self.nu, device=self.device)
    self.w = torch.zeros(self.n, self.nu, device=self.device)
    self.calls = 0
    self.eps_sum = 0.0
    self.eps_sq = 0.0
    self.eps_lag = 0.0                                    # sum of eps_t eps_(t-1): the realised lag-1 correlation
    self.m_negative = 0                                   # draws where 1 + c eps < 0 (clamped to 0)

  def set_levels(self, kind: torch.Tensor, c: torch.Tensor, white_std: torch.Tensor | None = None,
                 tau_c: torch.Tensor | float = 0.0) -> None:
    self.kind = kind.to(self.device).long()
    self.c = c.to(self.device).float()
    tau = torch.as_tensor(tau_c, dtype=torch.float32, device=self.device).expand(self.n)
    if bool((tau < 0).any()):
      raise ValueError("tau_c must be >= 0")
    self.rho = torch.where(tau > 0, torch.exp(-self.dt / tau.clamp_min(1e-9)), torch.zeros_like(tau))
    if white_std is not None:
      if tuple(white_std.shape) != (self.n, self.nu):
        raise ValueError(f"white_std shape {tuple(white_std.shape)} != {(self.n, self.nu)}")
      self.white_std = white_std.to(self.device).float()
    if bool(((self.kind == KINDS["white"]) & (self.white_std.amax(dim=1) <= 0)).any()):
      raise ValueError("white environments need a positive white_std")

  def capture_base(self, env_ids: torch.Tensor | None = None) -> None:
    """Take the current actuator parameters as the noise-free base (call after the reset randomisation)."""
    ids = slice(None) if env_ids is None else env_ids.long().to(self.device)
    model = self.env.sim.model
    self.base_g0[ids] = model.actuator_gainprm[ids][:, self.ctrl, 0]
    self.base_b0[ids] = model.actuator_biasprm[ids][:, self.ctrl, 0]
    self.base_b1[ids] = model.actuator_biasprm[ids][:, self.ctrl, 1]
    self.base_b2[ids] = model.actuator_biasprm[ids][:, self.ctrl, 2]

  def write(self, env_ids: torch.Tensor | None = None) -> None:
    """Write base x m (sdn) and base + w (white) for the given environments (all by default)."""
    model = self.env.sim.model
    if env_ids is None:
      g0, b0, b1, b2, m, w = self.base_g0, self.base_b0, self.base_b1, self.base_b2, self.m, self.w
      model.actuator_gainprm[:, self.ctrl, 0] = g0 * m
      model.actuator_biasprm[:, self.ctrl, 0] = b0 + w
      model.actuator_biasprm[:, self.ctrl, 1] = b1 * m
      model.actuator_biasprm[:, self.ctrl, 2] = b2 * m
      return
    ids = env_ids.long().to(self.device)
    rows, cols = ids[:, None], self.ctrl[None, :]
    model.actuator_gainprm[rows, cols, 0] = self.base_g0[ids] * self.m[ids]
    model.actuator_biasprm[rows, cols, 0] = self.base_b0[ids] + self.w[ids]
    model.actuator_biasprm[rows, cols, 1] = self.base_b1[ids] * self.m[ids]
    model.actuator_biasprm[rows, cols, 2] = self.base_b2[ids] * self.m[ids]

  def step(self) -> None:
    """One innovation per environment and actuator; eps (AR(1), stationary N(0, 1)), m and w for the next control
    period; write them."""
    xi = torch.randn((self.n, self.nu), generator=self.gen, device=self.device)
    if self.calls == 0:
      eps = xi                                           # stationary start
    else:
      rho = self.rho[:, None]
      eps = rho * self.eps + torch.sqrt(1.0 - rho * rho) * xi
    sdn = (self.kind == KINDS["sdn"])[:, None]
    white = (self.kind == KINDS["white"])[:, None]
    raw = 1.0 + self.c[:, None] * eps
    self.m_negative += int((sdn & (raw < 0)).sum())
    self.m = torch.where(sdn, raw.clamp_min(0.0), torch.ones_like(raw))
    self.w = torch.where(white, self.white_std * eps, torch.zeros_like(eps))
    if self.calls > 0:
      self.eps_lag += float((eps.double() * self.eps.double()).sum())
    self.eps = eps
    self.calls += 1
    self.eps_sum += float(eps.double().sum())
    self.eps_sq += float((eps.double() ** 2).sum())
    self.write()

  def restore(self) -> None:
    self.m = torch.ones_like(self.m)
    self.w = torch.zeros_like(self.w)
    self.write()

  def nominal_torque(self, ctrl: torch.Tensor, length: torch.Tensor, velocity: torch.Tensor,
                     limit: torch.Tensor) -> torch.Tensor:
    """clip(u) of the noise-free base actuator for given ctrl / actuator length / velocity (N, nu)."""
    u = self.base_g0 * ctrl + self.base_b0 + self.base_b1 * length + self.base_b2 * velocity
    return torch.maximum(torch.minimum(u, limit), -limit)


class sdn_step_event:
  """mjlab class-based event term (mode "step") putting SdnBody into a training environment (W1; not registered yet).

  params: c (float), kind ("sdn" | "white" | "none"), seed (int), tau_c (s, default 0.04), white_std (sequence of nu
          floats, kind "white" only), entity_name ("robot").
  """

  model_fields = GAIN_FIELDS            # mjlab expands these per environment before the CUDA graphs are captured

  def __init__(self, cfg, env):
    p = dict(cfg.params)
    if p.get("kind", "sdn") not in KINDS:
      raise ValueError(f"kind {p.get('kind')!r} not in {tuple(KINDS)}")
    self.env = env
    self.params = p
    self.body = None

  def _ensure(self) -> SdnBody:
    if self.body is None:
      p = self.params
      body = SdnBody(self.env, seed=int(p.get("seed", 0)), entity_name=p.get("entity_name", "robot"))
      n = self.env.num_envs
      kind = torch.full((n,), KINDS[p.get("kind", "sdn")], dtype=torch.long)
      c = torch.full((n,), float(p.get("c", 0.0)))
      std = None
      if p.get("kind") == "white":
        std = torch.as_tensor(p["white_std"], dtype=torch.float32)[None, :].expand(n, -1).clone()
      body.set_levels(kind, c, std, tau_c=float(p.get("tau_c", 0.04)))
      self.body = body
    return self.body

  def reset(self, env_ids=None) -> None:
    body = self._ensure()
    ids = None if env_ids is None or isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=body.device)
    body.capture_base(ids)               # after the reset events: the episode's randomised gains are the base
    body.write(ids)                      # the current period's noise applies to the reset environments too

  def __call__(self, env, env_ids, c: float = 0.0, kind: str = "sdn", seed: int = 0, tau_c: float = 0.04,
               white_std=None, entity_name: str = "robot") -> None:
    del env, env_ids, c, kind, seed, tau_c, white_std, entity_name
    self._ensure().step()
