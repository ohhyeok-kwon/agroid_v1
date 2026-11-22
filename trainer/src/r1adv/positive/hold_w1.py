"""R3 wave 1 of the positive agent (coordinator decisions 2026-11-17 21:41 KST, .agent/adversarial/rounds/R03/
coordinator_decisions_r3_wave1.md; fixes 22:1x): a 2x2 from scratch on the P-e0 recipe -- task adv_hold_r2_of_pe0 =
the R2 Hold mission with uniform 4-25 N s torso hits, default-pose starts only, T form (the actor reads the noise-free
570 frames), entropy bonus 0, falls end the episode, no curriculum, objective v1 (r = L + D, D = d_v0b in every cell) --
differing ONLY in
  b        the foot-lead input: absent (C0, L1) / present (B1, BL)
  posture  the posture channel of L: point (C0, B1) = (posture RMS to q* / 0.02)^2 = |q - q*|^2 / (24 * 0.02^2) as in
           P-e0 / distribution (L1, BL) = g(q) |q - q*|^2 / (24 * 0.02^2): the same term switched off inside the human
           neutral-standing support (g = 0: joint likelihood at least the median human frame's, feet width and stagger
           within the human p99 of the stance gate) and exactly the point term beyond a short ramp (g = 1;
           posture_prior.py); every other channel and sigma unchanged
Cells: C0 = point, no b (positive control, expected to hop); B1 = point + b; L1 = distribution; BL = distribution + b.

Observation layouts (joints in agroid POLICY order inside the frames, see hold_model):
  actor   C0, L1: 570 = 10 frames x [gyro 3, gravity 3, twist command 3, q - q0 24, dq 24] (noise-free, T form)
          B1, BL: 571 = the same 570 followed by the foot-lead constant b at index 570 (the last value)
  critic  C0, L1: 580 = the 570 frames + base linear velocity 3 + base height 1 + foot contact 2 + mission channels 4
          B1, BL: 581 = the same 580 followed by the same b at index 580 (the last value)
  mission channels (critic, privileged): [posture, height error, planar root velocity 2]; the posture entry is the RMS
          to q* in the point cells and sqrt(g) times it (the reward's channel, 0.02 sqrt(P)) in the distribution cells.
b in {-1, +1} is drawn uniformly per episode at reset (one draw per reset, its own generator seeded from the env seed,
so the global random stream of the other terms is the same in every cell), the same value in the actor and the critic,
and held constant within the episode. At deployment b is a constant chosen once (e.g. +1; the negative runner's
--foot-lead); the play / evaluation environment of this module uses b = +1 unless told otherwise. The actor is
r1adv.positive.hold_model.HoldSymmetricMLP for every cell: with 571 inputs its mirror is (M o, -b) and the first-layer
weights from b start at zero.
"""
from __future__ import annotations

import torch

from mjlab.managers.observation_manager import ObservationTermCfg

from r1adv.positive import hold_r2 as R2
from r1adv.positive import hold_task as HT
from r1adv.positive.posture_prior import PosturePrior, stance

# The human neutral-standing support (.agent/adversarial/positive/tools/w1_posture_gmm.py v3: the stance gate's neutral
# quiet frames = the negative's stance_human.py rule, reproduced by w1_stance_data.py; mirror-averaged mixture; joint
# threshold = the median held-out max-component NLL of those frames; width / stagger bounds = their p99), pinned by
# sha256; server copy /rl-agent/.agent/r1adv_inputs/, PC original .agent/adversarial/positive/models/.
POSTURE_GMM = dict(file="w1_posture_gmm_v3.npz", sha256="7d2d003dce0ed93ce416752d95cd66f1e063955b8c8db6deb839806502897843")
POSTURE_FORM = ("g * |q - q*|^2 / (24 * 0.02^2), g = min(1, max((NLL*-c)+/T_J, (width-w99)+/ramp, (|stagger|-s99)+/ramp)): "
                "0 inside the human neutral-standing support, the point term beyond the ramps")
POINT_SIGMA = HT.SIGMA["posture"]      # 0.02 rad: the channel keeps its sigma; e = 0.02 sqrt(P) makes (e / sigma)^2 = P
LEAD_SEED = 404
CELLS = {"C0": dict(lead=False, posture="point"), "B1": dict(lead=True, posture="point"),
         "L1": dict(lead=False, posture="dist"), "BL": dict(lead=True, posture="dist")}


class _LeadState:
  """Per-episode foot-lead values b, shared by the actor and critic terms of one environment."""

  def __init__(self, env, value):
    self.value = None if value is None else float(value)
    dev = env.device
    self.b = torch.ones(env.num_envs, device=dev)
    self.ids = torch.arange(env.num_envs, device=dev)
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(int(env.cfg.seed) * 7919 + LEAD_SEED)
    self.reset(None)

  def reset(self, env_ids):
    ids = self.ids if env_ids is None else (self.ids[env_ids] if isinstance(env_ids, slice) else env_ids)
    if len(ids) == 0:
      return
    if self.value is not None:
      self.b[ids] = self.value
      return
    u = torch.rand(len(ids), device=self.b.device, generator=self.g)
    self.b[ids] = torch.where(u < 0.5, -1.0, 1.0)


class foot_lead:
  """Observation term: the foot-lead input b (one value, appended as the LAST term of the actor and critic groups).
  params: value = None (training: b uniform in {-1, +1} per episode) or a fixed +1 / -1 (play, evaluation)."""

  def __init__(self, cfg, env):
    value = cfg.params.get("value")
    if value is not None and float(value) not in (-1.0, 1.0):
      raise ValueError(f"foot-lead value must be +1 or -1, got {value}")
    state = getattr(env, "_r1adv_foot_lead", None)
    self.owner = state is None
    if state is None:
      state = _LeadState(env, value)
      env._r1adv_foot_lead = state
    elif state.value != (None if value is None else float(value)):
      raise ValueError("the actor and critic foot-lead terms must use the same value")
    self.state = state

  def reset(self, env_ids=None):
    if self.owner:                                      # one draw per reset; the other group reads the same values
      self.state.reset(env_ids)

  def __call__(self, env, value=None):
    return self.state.b[:, None]


def _prior(params, device) -> PosturePrior:
  path = R2.pinned(dict(file=params["posture_file"], sha256=params["posture_sha256"]))
  return PosturePrior(path, device=device)


class HoldObjectiveW1(HT.HoldObjective):
  """The Hold objective r = L + beta D with the posture channel of L = P = g |q - q*|^2 / (24 * 0.02^2) (posture_prior.py):
  L = 1 / (1 + 1/2 [P + (h / 0.02)^2 + (tilt / 0.035)^2 + (v / 0.05)^2 + (yaw rate / 0.2)^2])."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    p = cfg.params
    if float(p["sigma"]["posture"]) != POINT_SIGMA:
      raise ValueError("the posture channel's sigma differs from the formula's")
    self.prior = _prior(p, env.device)
    if self.prior.joint_names != list(self.robot.joint_names)[:24]:
      raise ValueError("posture support joint order differs from the robot")
    c = self.prior.constants()
    for k in ("c", "T_J", "width_p99", "stagger_p99", "ramp"):
      if abs(c[k] - float(p[f"posture_{k}"])) > 1e-12:
        raise ValueError(f"posture constant {k} of the configuration differs from the pinned file")
    self.feet = [self.robot.body_names.index(b) for b in self.prior.stance_bodies]

  def channels(self) -> dict[str, torch.Tensor]:
    ch = super().channels()                             # the R1 channels (updates the 0.4 s speed mean once per step)
    d = self.robot.data
    q = d.joint_pos[:, :24]
    width, stagger = stance(d.body_link_pos_w[:, self.feet, :2], d.root_link_quat_w)
    g = self.prior.gate(q, width, stagger)
    rms = ch["posture"]                                 # RMS to q*: (rms / 0.02)^2 is the point term
    ch["posture"] = g.sqrt() * rms                      # (e / sigma)^2 = g * point term inside the likelihood's sum
    log = self.env.extras.setdefault("log", {})
    log["Objective/posture_gate"] = g.mean()
    log["Objective/posture_free"] = (g == 0).float().mean()
    log["Objective/posture_rms_to_target"] = rms.mean()
    log["Objective/stance_width"] = width.mean()
    log["Objective/stance_stagger"] = stagger.mean()
    return ch

  def __call__(self, env, beta: float, sigma: dict, form: str, formula_sha256: str = "", posture_file: str = "",
               posture_sha256: str = "", posture_c: float = 0.0, posture_T_J: float = 0.0, posture_width_p99: float = 0.0,
               posture_stagger_p99: float = 0.0, posture_ramp: float = 0.0, posture_form: str = "") -> torch.Tensor:
    return super().__call__(env, beta, sigma, form, formula_sha256)


class objective_channels_dist(HT.objective_channels):
  """Critic mission channels of the distribution cells: the posture entry is the reward's channel sqrt(g) * RMS to q*."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.prior = _prior(cfg.params, env.device)
    self.feet = [env.scene["robot"].body_names.index(b) for b in self.prior.stance_bodies]

  def __call__(self, env, posture_file: str = "", posture_sha256: str = "") -> torch.Tensor:
    out = super().__call__(env)
    d = env.scene["robot"].data
    width, stagger = stance(d.body_link_pos_w[:, self.feet, :2], d.root_link_quat_w)
    out[:, 0] = self.prior.gate(d.joint_pos[:, :24], width, stagger).sqrt() * out[:, 0]
    return out


def posture_params() -> dict:
  """The reward-term parameters of the distribution posture channel (constants read from the pinned file)."""
  c = PosturePrior(R2.pinned(POSTURE_GMM)).constants()
  return dict(posture_file=POSTURE_GMM["file"], posture_sha256=POSTURE_GMM["sha256"], posture_c=c["c"],
              posture_T_J=c["T_J"], posture_width_p99=c["width_p99"], posture_stagger_p99=c["stagger_p99"],
              posture_ramp=c["ramp"], posture_form=POSTURE_FORM)


def hold_w1_env_cfg(lead: bool, posture: str, play: bool = False, lead_value: float | None = None):
  """R3 wave-1 cell: the P-e0 environment (hold_r2_env_cfg(actor_clean=True, human_starts=False, hit_impulse="dist"),
  exactly adv_hold_r2_of_pe0) + the foot-lead input (lead) and / or the distribution posture channel (posture="dist")."""
  if posture not in ("point", "dist"):
    raise ValueError(f"posture must be 'point' or 'dist', got {posture!r}")
  cfg = R2.hold_r2_env_cfg(actor_clean=True, human_starts=False, hit_impulse="dist", play=play)
  if lead:
    value = float(lead_value) if lead_value is not None else (1.0 if play else None)
    for group in ("actor", "critic"):                  # appended last: b is the last input of both networks
      cfg.observations[group].terms["foot_lead"] = ObservationTermCfg(func=foot_lead, params={"value": value})
  elif lead_value is not None:
    raise ValueError("lead_value is only for cells with the foot-lead input")
  if posture == "dist":
    extra = posture_params()
    term = cfg.rewards["objective"]
    term.func = HoldObjectiveW1
    term.params.update(extra)
    channels = cfg.observations["critic"].terms["channels"]
    channels.func = objective_channels_dist
    channels.params = {k: extra[k] for k in ("posture_file", "posture_sha256")}
  return cfg


def hold_w1_rl_cfg(max_iterations: int = 1500):
  """The P-e0 PPO recipe for every cell (entropy bonus 0; actor HoldSymmetricMLP, which reads the input width)."""
  return R2.hold_r2_rl_cfg(max_iterations, entropy_coef=0.0)


def record() -> dict:
  return dict(code_sha256=R2.CODE_SHA256, cells=CELLS, posture=posture_params(), lead_seed=LEAD_SEED,
              base_task="adv_hold_r2_of_pe0")
