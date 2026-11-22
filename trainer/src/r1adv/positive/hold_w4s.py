"""R6 wave-4 option S of the positive agent: the human-push disturbance for the training environment (coordinator
2026-11-18 11:0x KST; .agent/adversarial/rounds/R04/06B_r6_synthesis.md, lever S; evidence r6_a3_pushdur.md and
r6_suddenness.md). Only the force-time curve of the torso hits changes; every hit keeps its schedule and its draws.

  - schedule and draws: R2.torso_hits' own generator, unchanged: first hit U(first_s), then every U(gap_s) s; impulse
    J ~ U(impulse_ns) (4-25 N s), azimuth U(0, 2 pi) from the heading, height U(height_m). Same seed -> same hit stream.
  - with probability IMPULSE_SHARE (0.3) a hit stays the 0.02 s impulse (force J / 0.02 s for one control step: the
    current training disturbance, bit for bit) -- the 'in every case' principle keeps sudden impacts in the distribution
  - otherwise a half-sine push F(t) = (pi J / 2T) sin(pi t / T), 0 <= t <= T, T ~ U(PUSH_S) = U(0.1, 0.8) s:
      human pushes last 0.3-0.8 s; dimensionless tau = omega0 T: DeCIMo 1.88 [1.81, 1.95] (measured), UTwente 1.6-2.5
      (estimated) (r6_suddenness.md); the R1 (omega0 = 3.9 /s) matches tau 1.6-2.5 at T = 0.41-0.64 s, the same seconds
      0.3-0.8 s are tau 1.2-3.1; 0.1-0.3 s keeps brisk pushes; the same J distribution, so a long push has a small peak
      (20 N s over 0.6 s: 52 N)
  - applied as a 50 Hz staircase (the negative's A3 rule, r6_a3_pushdur.md 1): control step k of a push carries the
    impulse J_k = (J/2) (cos(pi k dt / T) - cos(pi min((k+1) dt, T) / T)), so sum_k J_k = J exactly; force J_k / dt during
    that step; direction fixed in the world at the push start (heading + azimuth, as the impulse) and the same torso point
    (height h: torque = arm x force), as the impulse
  - the impulse-or-push choice and T come from the term's OWN generator (seed env.seed * 7919 + PUSH_SEED), so every
    other draw (and every other term's stream) is that of the current environment for the same seed
  - gap_s >= max T + one step: pushes never overlap; an episode reset ends a push (its force is cleared)
  - not observed: no force, duration or push state in any observation (actor or critic), as now
Cells (diagnostic / building block): XBS = XB (adv_hold_w2_xb: C0 environment, plain MLP actor) + S, task adv_hold_w4_xbs
(tag r6w4-XBS). with_human_push(cfg) adds S to any environment whose hits are R2.torso_hits.
Logged (training diagnostics, never observed): Push/pushing (share of environments under a push force this step).
"""
from __future__ import annotations

import math

import torch

from r1adv.positive import hold_r2 as R2
from r1adv.positive import hold_w1 as W1
from r1adv.positive import hold_w2 as W2

IMPULSE_SHARE = 0.3
PUSH_S = (0.1, 0.8)
PUSH_SEED = 606
PUSH_SHAPE = "half-sine, 50 Hz staircase, J_k = (J/2)(cos(pi k dt/T) - cos(pi min((k+1)dt, T)/T))"
SAVE_INTERVAL = 50
SOURCES = dict(human_push_s="0.3-0.8 s (r6_suddenness.md 2)", decimo_tau="1.88 [1.81, 1.95]", utwente_tau="1.6-2.5 (est.)",
               robot_omega0_per_s=3.9, a3_rule="r6_a3_pushdur.md 1 (1561053b)")


class torso_pushes(R2.torso_hits):
  """R2.torso_hits with option S (see the module docstring). The parent's draws are made in the parent's order from the
  parent's generator; the impulse-or-push choice and the duration from self.gd."""

  def __init__(self, cfg, env):
    p = cfg.params
    n, dev = env.num_envs, env.device
    self.share = float(p["impulse_share"])
    self.t_lo, self.t_hi = (float(x) for x in p["push_s"])
    if not (0.0 <= self.share <= 1.0 and 0.02 <= self.t_lo <= self.t_hi):
      raise ValueError("bad push parameters")
    if float(p["gap_s"][0]) < self.t_hi + env.step_dt:
      raise ValueError("pushes could overlap: the smallest gap is shorter than the longest push")
    self.push_on = torch.zeros(n, dtype=torch.bool, device=dev)
    self.push_k = torch.zeros(n, dtype=torch.long, device=dev)
    self.push_n = torch.ones(n, dtype=torch.long, device=dev)
    self.push_T = torch.full((n,), env.step_dt, device=dev)
    self.push_J = torch.zeros(n, device=dev)
    self.push_dir = torch.zeros(n, 3, device=dev)
    self.push_arm = torch.zeros(n, 3, device=dev)
    self.gd = torch.Generator(device=dev)
    self.gd.manual_seed(int(env.cfg.seed) * 7919 + int(p.get("push_seed", PUSH_SEED)))
    super().__init__(cfg, env)                                      # the parent's state, generator and first schedule

  def reset(self, env_ids=None):
    ids = torch.arange(len(self.next), device=self.next.device) if env_ids is None else env_ids
    if len(ids):
      self.push_on[ids] = False
      self.push_k[ids] = 0
    super().reset(env_ids)                                           # parent: new first-hit time, clears an applied force

  def durations(self, k: int):
    """(is impulse, T) for k new hits, from the term's own generator (two uniforms per hit, always)."""
    u = torch.rand(k, 2, device=self.next.device, generator=self.gd)
    is_imp = u[:, 0] < self.share
    T = torch.where(is_imp, torch.full_like(u[:, 1], self.dt), self.t_lo + (self.t_hi - self.t_lo) * u[:, 1])
    return is_imp, T

  def __call__(self, env, env_ids, impulse_ns, height_m, first_s, gap_s, impulse_share=IMPULSE_SHARE, push_s=PUSH_S,
               push_shape: str = PUSH_SHAPE, push_seed: int = PUSH_SEED, code_sha256: str = "", distribution_sha256: str = ""):
    t = env.episode_length_buf.float() * self.dt
    fire = t >= self.next
    cont = self.push_on & ~fire
    clear = self.active & ~fire & ~cont
    if not (fire.any() or cont.any() or clear.any()):
      return
    dev = self.next.device
    ids = fire.nonzero().flatten()
    if len(ids):                                                      # the parent's draws, order and generator
      imp = self._u(*impulse_ns, len(ids))
      azimuth = self._u(0.0, 2.0 * math.pi, len(ids))
      h = self._u(*height_m, len(ids))
      q = self.asset.data.root_link_quat_w[ids]
      heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      ang = heading + azimuth
      self.push_dir[ids] = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], 1)
      arm = torch.zeros(len(ids), 3, device=dev)
      arm[:, 2] = h
      self.push_arm[ids] = arm
      self.next[ids] = t[ids] + self._u(*self.gap, len(ids))
      self.count[ids] += 1
      self.last[ids] = torch.stack([t[ids], imp, azimuth, h], 1)
      is_imp, T = self.durations(len(ids))
      self.push_J[ids] = imp
      self.push_T[ids] = T
      self.push_n[ids] = torch.where(is_imp, torch.ones_like(self.push_n[ids]),
                                     torch.ceil(T / self.dt - 1e-6).long().clamp(min=1))
      self.push_k[ids] = 0
      self.push_on[ids] = True
    upd = (fire | cont | clear).nonzero().flatten()
    on = self.push_on[upd]
    forces = torch.zeros(len(upd), 1, 3, device=dev)
    torques = torch.zeros(len(upd), 1, 3, device=dev)
    pe = upd[on]
    if len(pe):
      k = self.push_k[pe].float()
      T, J = self.push_T[pe], self.push_J[pe]
      a = k * self.dt
      b = torch.minimum((k + 1.0) * self.dt, T)
      Jk = torch.where(self.push_n[pe] == 1, J, 0.5 * J * (torch.cos(math.pi * a / T) - torch.cos(math.pi * b / T)))
      f = self.push_dir[pe] * (Jk / self.dt)[:, None]
      forces[on, 0] = f
      torques[on, 0] = torch.cross(self.push_arm[pe], f, dim=1)
      self.push_k[pe] += 1
      done = pe[self.push_k[pe] >= self.push_n[pe]]
      self.push_on[done] = False
    self.asset.write_external_wrench_to_sim(forces, torques, env_ids=upd, body_ids=[self.torso])
    self.active = fire | cont
    env.extras.setdefault("log", {})["Push/pushing"] = self.active.float().mean()


def with_human_push(cfg, impulse_share: float = IMPULSE_SHARE, push_s: tuple = PUSH_S):
  """Option S on an environment configuration whose hits are R2.torso_hits (training only)."""
  term = cfg.events.get("torso_hits")
  if term is None or term.func is not R2.torso_hits:
    raise ValueError("option S applies to the plain R2.torso_hits term")
  term.func = torso_pushes
  term.params.update(impulse_share=float(impulse_share), push_s=tuple(float(x) for x in push_s), push_shape=PUSH_SHAPE,
                     push_seed=PUSH_SEED)
  return cfg


CELLS = {"XBS": dict(base="XB", plain=True)}


def cell_env_cfg(cell: str, play: bool = False):
  cfg = W1.hold_w1_env_cfg(lead=False, posture="point", play=play)          # C0's environment = XB's
  return cfg if play else with_human_push(cfg)


def cell_rl_cfg(cell: str, max_iterations: int = 1500):
  return W2.rl_cfg(floor=False, plain=CELLS[cell]["plain"], ar1=False, save_interval=SAVE_INTERVAL,
                   max_iterations=max_iterations)


def record() -> dict:
  return dict(impulse_share=IMPULSE_SHARE, push_s=PUSH_S, push_seed=PUSH_SEED, push_shape=PUSH_SHAPE, sources=SOURCES,
              cells=CELLS, base_tasks=dict(XB="adv_hold_w2_xb"))
