"""E4 board-2 method arms (positive 25, coordinator 2026-11-20 23:0x-23:2x KST; rounds/R04/16_rounds.md, user 23:0x / 23:1x).
DIAGNOSTIC. Each arm is a new task on top of B0 (r1lab.tasks.careful_walk_b0, unchanged: S1 objective v2 env + the anchor's
representation and trainer). Every change is a visible config leaf (a swapped func / class with new fields / removed key).

M1   objective structure. B0's success reward is S1 v2's S = 1 before the deadline (tau > 0) and [in goal and at rest] after
     it (r1s1.goal.GoalCommand.judge()[0]). M1 replaces only the reward term's func (rewards.s1_success.func):
         S_t = 1[in_goal(s_t) and at_rest(s_t)]           at every step, before and after the deadline (weight W_SUCCESS 1)
     in_goal / at_rest exactly as judge() (TOL, REST of r1s1.numbers). Hold goals already had this S from issue (their
     deadline is at issue: tau0 = 0, tau <= 0), so M1 changes locomotion and crouch goals before their deadline. The deadline
     stays in the goal (the tau channel of the goal observation, the goal duration tau0 + hold), in the end judgement (the
     command's own judge() / metrics 'success' and the s1_post_* records) and the progress potential term is unchanged.
M2'  gait / contact human prior only (user 23:1x: human reference motion only to turn two-foot hops into one-foot stepping).
     A Wasserstein critic (WGAN-GP, Gulrajani et al. 2017: two-sided gradient penalty on interpolates, lambda 10, plus a
     drift term 1e-3 E[D_ref^2]) on lower-body gait features only, over a window of K = 6 policy frames (0.1 s):
       per frame   foot heights above the lower foot (L, R)                                   2 x K
       per step    foot vertical velocities (L, R), foot horizontal speeds (L, R), pelvis vertical velocity   5 x (K - 1)
     (foot = the model's left_foot / right_foot sites, pelvis = root link; differences over 20 ms; yaw / position / crouch
     height invariant; no arm, trunk, joint-angle or reaction-dynamics feature). The same function builds the reference from
     the retargeted R1 clips (tools/e4m_gait_ref.py: walk / turn, stand idle, hit-reaction steps, crouch; mirrored).
     Reward (bounded, self-normalised, never flat): r = exp(min(0, (D(x) - m_ref) / g)), m_ref = running mean of D on
     reference windows, g = running mean gap m_ref - m_pi (> 0); 0 until the critic's first update. Weight 0.25 (low: the
     success term pays 1 per second). Reward key 'style' so r1lab.train sets save_path (amp_disc.pt) / load_path.
M4   actuator signal-dependent noise (R18-a r06): the torque the robot's joint receives is
         tau = clip(tau_PD (1 + k eps) + sigma0 eta, -tau_max, tau_max)
     tau_PD = the builtin position actuator's own force this physics substep (gain ctrl + bias0 + bias1 q + bias2 qdot with
     the per-env randomised gains, clamped to its forcerange), eps / eta per env and joint stationary Ornstein-Uhlenbeck
     N(0, 1) sequences at the physics rate (5 ms) with correlation time tau_c ~ U(20, 40) ms per episode, k ~ U(0, 0.15)
     per episode and env, sigma0 = 0.02 tau_max per joint. Applied as qfrc_applied = tau - tau_PD on the 24 joint dofs right
     after the controls are written, before every sim.step (training only; play has none). Physics, not a reward term: the
     effort price still reads qfrc_actuator (the PD torque), i.e. it does not see the noise torque.
J-lim joint targets cut to the physical joint range (MJCF range = entity default_joint_pos_limits): the target written to the
     actuator is clip(q0 + a x scale - encoder_bias, lo, hi). The policy output (and its PPO log-probability, the
     'previous action' observation) is unchanged; only the environment's target is cut.
MX   M1 + M2' + M4 + J-lim.
Bases: B0 (b0_rl_cfg) or B0 + sigma cap 0.8 (r1lab.tasks.careful_walk_e3x.cap_rl_cfg, imported) - task suffix '_cap'.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from r1lab.mdp.actions import PolicyJointPositionAction, PolicyJointPositionActionCfg

STYLE_WEIGHT = 0.25
GAIT_K = 6
GAIT_DT = 0.02
GAIT_FEAT = 2 * GAIT_K + 5 * (GAIT_K - 1)          # 37
GAIT_REF = "/rl-agent/.agent/tmp/positive_e4m/data/e4m_gait_ref_k6.npz"   # built by tools/e4m_gait_ref.py (sha256 in README)
ARMS = ("m1", "m2p", "m4", "mx", "jl")


# -- M1 -------------------------------------------------------------------------------------------------------------
def m1_success(env, command_name: str = "goal") -> torch.Tensor:
  """S_t = 1[in_goal and at_rest] at every step (no 'S = 1 before the deadline')."""
  _, in_goal, at_rest = env.command_manager.get_term(command_name).judge()
  return (in_goal & at_rest).float()


# -- J-lim and M4: the action term ----------------------------------------------------------------------------------
@dataclass(kw_only=True)
class E4mJointPositionActionCfg(PolicyJointPositionActionCfg):
  """The anchor's policy-order joint position action + J-lim (target cut to the joint range) + M4 (actuator noise)."""

  clip_to_joint_range: bool = False
  sdn: bool = False
  sdn_k_range: tuple[float, float] = (0.0, 0.15)
  sdn_sigma0_frac: float = 0.02
  sdn_tau_s: tuple[float, float] = (0.02, 0.04)

  def build(self, env):
    return E4mJointPositionAction(self, env)


class E4mJointPositionAction(PolicyJointPositionAction):
  def __init__(self, cfg: E4mJointPositionActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    lim = self._entity.data.default_joint_pos_limits[0, self._target_ids].clone()      # (24, 2) MJCF range
    self._lo, self._hi = lim[:, 0], lim[:, 1]
    self._sdn = None
    if cfg.sdn:
      self._sdn = _ActuatorNoise(cfg, env, self._entity)

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    if self.cfg.clip_to_joint_range:
      target = torch.maximum(torch.minimum(target, self._hi), self._lo)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if self._sdn is not None:
      self._sdn.redraw(env_ids)


class _ActuatorNoise:
  """M4 noise state and the hook that writes qfrc_applied after the robot's controls each physics substep."""

  def __init__(self, cfg: E4mJointPositionActionCfg, env, robot):
    import mujoco
    self.cfg, self.env, self.robot = cfg, env, robot
    m = env.sim.mj_model
    ctrl = robot.indexing.ctrl_ids.detach().cpu().long()
    if len(ctrl) != 24:
      raise ValueError(f"M4 expects 24 robot actuators, got {len(ctrl)}")
    jnt = torch.as_tensor(m.actuator_trnid[ctrl.numpy(), 0], dtype=torch.long)
    if not all(int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) for a in ctrl.tolist()):
      raise ValueError("M4: every robot actuator must drive a joint")
    if not all(int(m.actuator_gaintype[a]) == 0 and int(m.actuator_biastype[a]) == 1 for a in ctrl.tolist()):
      raise ValueError("M4: expected fixed gain / affine bias (position) actuators")
    dev = env.device
    self.ctrl = ctrl.to(dev)
    self.dof = torch.as_tensor(m.jnt_dofadr[jnt.numpy()], dtype=torch.long, device=dev)
    self.qadr = torch.as_tensor(m.jnt_qposadr[jnt.numpy()], dtype=torch.long, device=dev)
    self.limited = torch.as_tensor(m.actuator_forcelimited[ctrl.numpy()] != 0, device=dev)
    n = env.num_envs
    self.k = torch.zeros(n, 1, device=dev)
    self.rho = torch.zeros(n, 1, device=dev)
    self.eps = torch.zeros(n, 24, device=dev)
    self.eta = torch.zeros(n, 24, device=dev)
    self.last_counter = -1
    self.redraw(None)
    original = robot.write_data_to_sim

    def write_data_to_sim():
      original()
      self.apply()

    robot.write_data_to_sim = write_data_to_sim          # instance attribute: scene.write_data_to_sim() calls it

  def redraw(self, env_ids) -> None:
    whole = env_ids is None or isinstance(env_ids, slice)          # ActionManager.reset passes slice(None) for all
    ids = slice(None) if whole else env_ids
    count = self.k.shape[0] if whole else len(env_ids)
    if count == 0:
      return
    dev = self.k.device
    lo, hi = self.cfg.sdn_k_range
    self.k[ids] = lo + (hi - lo) * torch.rand(count, 1, device=dev)
    t0, t1 = self.cfg.sdn_tau_s
    tau_c = t0 + (t1 - t0) * torch.rand(count, 1, device=dev)
    self.rho[ids] = torch.exp(-self.env.physics_dt / tau_c)
    self.eps[ids] = torch.randn(count, 24, device=dev)
    self.eta[ids] = torch.randn(count, 24, device=dev)

  def _rows(self, field):
    return field if field.shape[0] == self.env.num_envs else field.expand(self.env.num_envs, *field.shape[1:])

  def pd_torque(self):
    """The position actuators' force at the current state and ctrl (MuJoCo: gain ctrl + bias0 + bias1 q + bias2 qdot)."""
    model, data = self.env.sim.model, self.env.sim.data
    gain = self._rows(model.actuator_gainprm)[:, self.ctrl]
    bias = self._rows(model.actuator_biasprm)[:, self.ctrl]
    frange = self._rows(model.actuator_forcerange)[:, self.ctrl]
    u = data.ctrl[:, self.ctrl]
    q = data.qpos[:, self.qadr]
    qd = data.qvel[:, self.dof]
    tau = gain[..., 0] * u + bias[..., 0] + bias[..., 1] * q + bias[..., 2] * qd
    lo = torch.where(self.limited, frange[..., 0], torch.full_like(tau, -1e9))
    hi = torch.where(self.limited, frange[..., 1], torch.full_like(tau, 1e9))
    return torch.maximum(torch.minimum(tau, hi), lo), lo, hi

  def apply(self) -> None:
    counter = int(self.env._sim_step_counter)
    if counter == self.last_counter:                     # the post-reset write of the same substep: no second draw
      return
    self.last_counter = counter
    a = self.rho
    b = torch.sqrt(1.0 - self.rho ** 2)
    self.eps = a * self.eps + b * torch.randn_like(self.eps)
    self.eta = a * self.eta + b * torch.randn_like(self.eta)
    tau_pd, lo, hi = self.pd_torque()
    tau_max = torch.maximum(hi.abs(), lo.abs()).clamp_max(1e6)
    sigma0 = self.cfg.sdn_sigma0_frac * tau_max
    total = torch.maximum(torch.minimum(tau_pd * (1.0 + self.k * self.eps) + sigma0 * self.eta, hi), lo)
    delta = total - tau_pd
    self.env.sim.data.qfrc_applied[:, self.dof] = delta
    self.last = (tau_pd, delta)
    log = self.env.extras.setdefault("log", {})
    log["Metrics/m4_noise_rms_nm"] = delta.pow(2).mean().sqrt()
    log["Metrics/m4_pd_rms_nm"] = tau_pd.pow(2).mean().sqrt()
    log["Metrics/m4_k_mean"] = self.k.mean()


# -- M2': the gait / contact critic ---------------------------------------------------------------------------------
def gait_features(win: torch.Tensor, dt: float = GAIT_DT) -> torch.Tensor:
  """win (B, K, 7) raw frames [z_L, z_R, x_L, y_L, x_R, y_R, z_pelvis] (world, m) -> (B, 2K + 5(K-1)) features."""
  zl, zr = win[..., 0], win[..., 1]
  zmin = torch.minimum(zl, zr)
  d = (win[:, 1:] - win[:, :-1]) / dt
  return torch.cat([zl - zmin, zr - zmin, d[..., 0], d[..., 1], d[..., 2:4].norm(dim=-1), d[..., 4:6].norm(dim=-1),
                    d[..., 6]], dim=-1)


def gait_mirror_index(k: int = GAIT_K) -> list[int]:
  """Feature permutation of the left/right mirror (feet swapped; the pelvis vertical velocity stays)."""
  n = k - 1
  blocks = [list(range(0, k)), list(range(k, 2 * k))]
  o = 2 * k
  vz = [list(range(o, o + n)), list(range(o + n, o + 2 * n))]
  sp = [list(range(o + 2 * n, o + 3 * n)), list(range(o + 3 * n, o + 4 * n))]
  pz = list(range(o + 4 * n, o + 5 * n))
  return blocks[1] + blocks[0] + vz[1] + vz[0] + sp[1] + sp[0] + pz


class GaitCritic(nn.Module):
  def __init__(self, dims=(256, 256)):
    super().__init__()
    layers, i = [], GAIT_FEAT
    for h in dims:
      layers += [nn.Linear(i, h), nn.ELU()]
      i = h
    layers.append(nn.Linear(i, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class gait_style:
  """Reward term: WGAN-GP critic on lower-body gait windows, bounded self-normalised reward (module docstring, M2')."""

  def __init__(self, cfg, env):
    import numpy as np
    p = cfg.params
    dev = env.device
    data = np.load(p["data_file"])
    ref = torch.from_numpy(data["windows"]).float().to(dev)          # (M, GAIT_FEAT), mirrored included
    if ref.shape[1] != GAIT_FEAT:
      raise ValueError(f"gait reference has {ref.shape[1]} features, expected {GAIT_FEAT}")
    self.ref = ref
    self.ref_w = torch.from_numpy(data["weights"]).float().to(dev)
    self.mu, self.sd = ref.mean(0), ref.std(0).clamp_min(p.get("std_floor", 0.02))
    self.critic = GaitCritic().to(dev)
    self.opt = torch.optim.Adam(self.critic.parameters(), lr=p.get("lr", 1e-4), betas=(0.5, 0.9))
    self.buf = torch.zeros(p.get("buffer", 200_000), GAIT_FEAT, device=dev)
    self.buf_n, self.buf_i = 0, 0
    self.robot = env.scene["robot"]
    self.sites, _ = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    n = env.num_envs
    self.win = torch.zeros(n, GAIT_K, 7, device=dev)
    self.age = torch.zeros(n, dtype=torch.long, device=dev)
    self.steps = 0
    self.update_every = p.get("update_every", 24)
    self.n_updates = p.get("num_updates", 4)
    self.batch = p.get("batch", 4096)
    self.gp = p.get("grad_penalty", 10.0)
    self.drift = p.get("drift", 1e-3)
    self.ema = p.get("ema", 0.2)
    self.m_ref, self.gap, self.trained = 0.0, 1.0, False
    self.ckpt = p.get("save_path")
    if p.get("load_path") and Path(p["load_path"]).exists():
      state = torch.load(p["load_path"], map_location=dev)
      self.critic.load_state_dict(state["critic"])
      self.m_ref, self.gap, self.trained = float(state["m_ref"]), float(state["gap"]), True
      print(f"[GAIT] loaded critic {p['load_path']}")

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      self.age[:] = 0
    else:
      self.age[env_ids] = 0

  def _frame(self):
    d = self.robot.data
    feet = d.site_pos_w[:, self.sites]                                  # (n, 2, 3)
    return torch.stack([feet[:, 0, 2], feet[:, 1, 2], feet[:, 0, 0], feet[:, 0, 1], feet[:, 1, 0], feet[:, 1, 1],
                        d.root_link_pos_w[:, 2]], dim=-1)

  def _norm(self, x):
    return (x - self.mu) / self.sd

  def _push(self, x):
    n = x.shape[0]
    if n == 0:
      return
    cap = self.buf.shape[0]
    idx = (torch.arange(n, device=x.device) + self.buf_i) % cap
    self.buf[idx] = x
    self.buf_i = int((self.buf_i + n) % cap)
    self.buf_n = min(cap, self.buf_n + n)

  def _update(self, env):
    with torch.inference_mode(False), torch.enable_grad():
      for _ in range(self.n_updates):
        pi = torch.randint(0, self.buf_n, (self.batch,), device=self.ref.device)
        ri = torch.multinomial(self.ref_w, self.batch, replacement=True)
        x_pi = self._norm(self.buf[pi].clone())
        x_ref = self._norm(self.ref[ri].clone())
        alpha = torch.rand(self.batch, 1, device=x_pi.device)
        x_hat = (alpha * x_ref + (1.0 - alpha) * x_pi).requires_grad_(True)
        d_pi = self.critic(x_pi)
        d_ref = self.critic(x_ref)
        d_hat = self.critic(x_hat)
        g = torch.autograd.grad(d_hat.sum(), x_hat, create_graph=True)[0]
        gp = (g.norm(dim=-1) - 1.0).pow(2).mean()
        loss = d_pi.mean() - d_ref.mean() + self.gp * gp + self.drift * d_ref.pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    m_ref, m_pi = float(d_ref.detach().mean()), float(d_pi.detach().mean())
    if not self.trained:
      self.m_ref, self.gap = m_ref, max(m_ref - m_pi, 1e-3)
    else:
      self.m_ref += self.ema * (m_ref - self.m_ref)
      self.gap += self.ema * (max(m_ref - m_pi, 1e-3) - self.gap)
    self.trained = True
    log = env.extras.setdefault("log", {})
    log["Metrics/gait_d_ref"] = torch.tensor(m_ref)
    log["Metrics/gait_d_policy"] = torch.tensor(m_pi)
    log["Metrics/gait_gap"] = torch.tensor(self.gap)
    log["Metrics/gait_gp"] = gp.detach()

  def __call__(self, env, data_file: str, **kwargs) -> torch.Tensor:
    del data_file, kwargs
    frame = self._frame()
    fresh = self.age == 0
    self.win = torch.roll(self.win, shifts=-1, dims=1)
    self.win[:, -1] = frame
    if fresh.any():
      self.win[fresh] = frame[fresh].unsqueeze(1).expand(-1, GAIT_K, -1)
    self.age += 1
    valid = self.age >= GAIT_K
    x = gait_features(self.win)
    r = torch.zeros(env.num_envs, device=env.device)
    if self.trained:
      with torch.no_grad():
        d = self.critic(self._norm(x)).squeeze(-1)
      r = torch.exp(((d - self.m_ref) / self.gap).clamp_max(0.0))
    r = r * valid.float()
    self._push(x[valid])
    self.steps += 1
    if self.steps % self.update_every == 0 and self.buf_n >= self.batch:
      self._update(env)
      if self.ckpt and self.steps % (self.update_every * 100) == 0:
        Path(self.ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"critic": self.critic.state_dict(), "m_ref": self.m_ref, "gap": self.gap}, self.ckpt)
    log = env.extras.setdefault("log", {})
    if valid.any():
      log["Metrics/gait_style"] = r[valid].mean()
      log["Metrics/gait_style_std"] = r[valid].std() if int(valid.sum()) > 1 else torch.tensor(0.0)
    return r


# -- configs ----------------------------------------------------------------------------------------------------------
def e4m_env_cfg(arm: str, play: bool = False, t0: int = 0, jl: bool = False):
  """B0 env + the arm's leaves. arm in ARMS; jl adds J-lim (always on in mx; the arm 'jl' is B0 + J-lim only)."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from r1lab.tasks.careful_walk_b0 import b0_env_cfg
  if arm not in ARMS:
    raise ValueError(f"arm must be one of {ARMS}")
  cfg = b0_env_cfg(play=play, t0=t0)
  m1 = arm in ("m1", "mx")
  m2 = arm in ("m2p", "mx")
  m4 = arm in ("m4", "mx")
  jl = jl or arm in ("mx", "jl")
  if m1:
    t = cfg.rewards["s1_success"]
    cfg.rewards["s1_success"] = dataclasses.replace(t, func=m1_success)
  if m2 and not play:
    cfg.rewards["style"] = RewardTermCfg(func=gait_style, weight=STYLE_WEIGHT,
                                         params={"data_file": GAIT_REF, "save_path": None, "load_path": None,
                                                 "grad_penalty": 10.0, "drift": 1e-3, "lr": 1e-4, "batch": 4096,
                                                 "num_updates": 4, "update_every": 24, "ema": 0.2,
                                                 "std_floor": 0.02, "window": GAIT_K})
  if jl or m4:
    a = cfg.actions["joint_pos"]
    fields = {f.name: getattr(a, f.name) for f in dataclasses.fields(a) if f.init}
    cfg.actions["joint_pos"] = E4mJointPositionActionCfg(**fields, clip_to_joint_range=jl, sdn=m4 and not play)
  return cfg


def e4m_rl_cfg(cap: bool = False):
  if cap:
    from r1lab.tasks.careful_walk_e3x import cap_rl_cfg
    return cap_rl_cfg()
  from r1lab.tasks.careful_walk_b0 import b0_rl_cfg
  return b0_rl_cfg()


def task_names() -> list[tuple[str, str, bool, bool]]:
  """(task name, arm, cap, jl) for every registered E4 task."""
  out = []
  for arm in ARMS:
    for cap in (False, True):
      jls = (True,) if arm in ("mx", "jl") else (False, True)
      for jl in jls:
        name = f"careful_walk_e4{arm}" + ("_cap" if cap else "") + ("_jl" if jl and arm not in ("mx", "jl") else "")
        out.append((name, arm, cap, jl))
  return out
