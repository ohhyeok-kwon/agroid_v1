"""R4 wave 2 of the positive agent: the training-side arms (coordinator 2026-11-18 01:4x KST, .agent/adversarial/rounds/
R04/04_r4_state.md section 3, candidates X-a / X-b / X-c of rl_exploration_researcher and program_strategist in
04A_panel_answers.md). Objective, reward, mission and every other setting are R3 wave 1's C0 / B1 (r1adv.positive.hold_w1:
the P-e0 recipe -- uniform 4-25 N s torso hits, default-pose starts, T form, entropy 0, falls end the episode, point
posture, r = L + D with D = d_v0b); each cell changes ONE axis of its base:

  XA   = B1 (foot-lead input b, 571 / 581 inputs) + (i) stepped starts + (ii) action-std floor 0.25   [max dose of X-a]
         (i)  40 % of the resets start from a bank of mirror-paired stepped states: one-foot support or split stance after
              a 10-40 cm step, whole-body CoM speed 0.3-0.7 m/s along the step, the load on one foot; inverse kinematics
              on the R1 model of this environment, no motion data, no manufacturer data
              (.agent/adversarial/positive/tools/w2_xa_starts.py, file pinned by sha256 in START_BANK); uniform heading;
              own random generator, so the other terms' random streams are those of B1
         (ii) the Gaussian action std never below 0.25 (FlooredGaussian: the std parameter is projected onto [0.25, inf)
              before its use in each training forward pass -- projected gradient, the parameter is never stuck below the
              floor -- and every use, rollouts included, clamps the std to >= 0.25)
         (iii) time-correlated (pink) exploration noise: left out of XA, see PINK_NOTE; the exact-likelihood residual
              AR(1) of R2 is wired as the optional variant XAn (not part of the early verdict)
  XAs  = B1 + (i) only;  XAf = B1 + (ii) only   (one-factor follow-ups of X-a)
  XB   = C0 with a plain MLP actor (hold_plain.HoldPlainMLP: the mirror average removed, no b; 570 / 580 inputs)
  XC   = C0 + azimuth curriculum: during the first AZ_UNTIL_UPDATE = 400 PPO updates (update index =
         env.common_step_counter // 24, the runner's 24 environment steps per update) no hit has an azimuth within
         0 +- 60 deg of the heading (azimuth 0 = force along the heading, pushing the robot forward = the readout's 0 deg):
         each hit's azimuth is the SAME single uniform draw mapped onto the remaining 240 deg (hit times, impulses and
         heights follow C0's stream); from update 400 on the original uniform 360 deg. A schedule of the training
         distribution keyed to the learning-update count; nothing in the control path.
Every cell saves a checkpoint every SAVE_INTERVAL = 50 updates (the negative's wave-2 readout: 50, 100, ... 300).
Evaluation / play environments (play=True) are those of hold_w1 (no starts bank, no hits, b = +1 in the lead cells).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from mjlab.managers.event_manager import EventTermCfg

from r1adv.positive import hold_r2 as R2
from r1adv.positive import hold_w1 as W1

START_BANK = dict(file="w2_xa_starts_v1.npz", sha256="cec7b776da1fc19de8befa8a7e5b748039a69a4a46549999ee0d7187c40bbe90")
START_FRACTION = 0.40
START_SEED = 505
STD_FLOOR = 0.25
AZ_EXCLUDE_HALF_DEG = 60.0
AZ_UNTIL_UPDATE = 400
STEPS_PER_UPDATE = 24
SAVE_INTERVAL = 50
PINK_NOTE = ("(iii) pink (1/f) exploration noise is left out of XA: it has no finite Markov state, so PPO's per-step "
             "likelihood ratio of a sampled action cannot be evaluated (the sample depends on the whole noise history); "
             "the white-noise likelihood of a coloured sample biases the ratio and its clipping. The exact one-time-"
             "constant alternative, residual AR(1) with the conditional likelihood (r1adv.positive.ar1, R2 arm A), was "
             "only smoke-tested (never trained), and its innovation std is 0.28 x its marginal std, so it would add an "
             "unvalidated third component with a different exploration scale to the max-dose arm; it is wired as the "
             "optional variant XAn instead.")
CELLS = {
  "XA": dict(base="B1", lead=True, starts=True, floor=True, plain=False, azimuth=False, ar1=False),
  "XAs": dict(base="B1", lead=True, starts=True, floor=False, plain=False, azimuth=False, ar1=False),
  "XAf": dict(base="B1", lead=True, starts=False, floor=True, plain=False, azimuth=False, ar1=False),
  "XAn": dict(base="B1", lead=True, starts=True, floor=True, plain=False, azimuth=False, ar1=True),
  "XB": dict(base="C0", lead=False, starts=False, floor=False, plain=True, azimuth=False, ar1=False),
  "XC": dict(base="C0", lead=False, starts=False, floor=False, plain=False, azimuth=True, ar1=False),
}


# ------------------------------------------------------------------------------------------------------- (i) stepped starts
class stepped_start:
  """Reset event, applied after the R1 reset terms (it overrides them for the chosen environments): each resetting
  environment starts with probability `fraction` from a uniformly drawn row of the pinned stepped-start bank, turned to a
  uniform heading: root position = environment origin + the row's pelvis height, root orientation / linear / angular
  velocity turned by the heading, joint positions and velocities as stored (planted soles at rest). The bank holds every
  state next to its mirror image, so the start distribution is left-right symmetric."""

  def __init__(self, cfg, env):
    p = cfg.params
    data = np.load(R2.pinned(dict(file=p["bank_file"], sha256=p["bank_sha256"])))
    self.asset = env.scene["robot"]
    if list(data["joint_names"]) != list(self.asset.joint_names)[:24] or len(self.asset.joint_names) != 24:
      raise ValueError("stepped-start bank joint order differs from the robot")
    mirror = torch.as_tensor(data["mirror_of"], dtype=torch.long)
    if not torch.equal(mirror[mirror], torch.arange(len(mirror))) or bool((mirror == torch.arange(len(mirror))).any()):
      raise ValueError("stepped-start bank rows are not mirror pairs")
    dev = env.device

    def f(key):
      return torch.as_tensor(np.asarray(data[key]), dtype=torch.float, device=dev)

    self.q, self.dq = f("joint_pos"), f("joint_vel")
    self.z, self.quat = f("root_pos")[:, 2], f("root_quat")
    self.v, self.w = f("root_lin_vel"), f("root_ang_vel")
    self.rows = len(self.q)
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(int(env.cfg.seed) * 7919 + START_SEED)
    self.stepped = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)      # current episode started stepped
    self.row = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)    # its bank row (-1: default start)
    self.resets = 0
    self.picked = 0

  def reset(self, env_ids=None):
    pass

  def __call__(self, env, env_ids, bank_file: str, bank_sha256: str, fraction: float):
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
      return
    dev = env.device
    pick = torch.rand(len(env_ids), device=dev, generator=self.g) < fraction
    ids = env_ids[pick]
    self.stepped[env_ids] = pick
    self.row[env_ids] = -1
    self.resets += int(len(env_ids))
    self.picked += int(pick.sum())
    log = env.extras.setdefault("log", {})
    log["Start/stepped_reset_share"] = self.picked / max(self.resets, 1)
    log["Start/stepped_episodes"] = self.stepped.float().mean()
    if len(ids) == 0:
      return
    k = torch.randint(0, self.rows, (len(ids),), device=dev, generator=self.g)
    yaw = (torch.rand(len(ids), device=dev, generator=self.g) * 2.0 - 1.0) * math.pi
    self.row[ids] = k
    c, s = torch.cos(yaw), torch.sin(yaw)

    def turn_vec(v):
      return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)

    zero = torch.zeros_like(yaw)
    turn = torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], 1)
    quat = R2._quat_mul(turn, self.quat[k])
    state = torch.zeros(len(ids), 13, device=dev)
    state[:, :3] = env.scene.env_origins[ids]
    state[:, 2] += self.z[k]
    state[:, 3:7] = quat / quat.norm(dim=1, keepdim=True)
    state[:, 7:10] = turn_vec(self.v[k])
    state[:, 10:13] = turn_vec(self.w[k])
    self.asset.write_root_state_to_sim(state, env_ids=ids)
    self.asset.write_joint_state_to_sim(self.q[k], self.dq[k], env_ids=ids)


# ----------------------------------------------------------------------------------------------------------- (ii) std floor
class FlooredGaussian(GaussianDistribution):
  """rsl_rl's state-independent Gaussian (scalar std parameter) with a floor on the std: before its use in a training
  forward pass the parameter is projected onto [std_floor, inf) (only when some entry is below, i.e. right after an
  optimizer step pushed it there -- before the new graph uses it), and every use clamps the std to >= std_floor
  (rsl_rl's std_range), so rollouts under inference mode see the floor as well. Above the floor nothing changes."""

  def __init__(self, output_dim: int, init_std: float = 1.0, std_floor: float = STD_FLOOR, std_type: str = "scalar",
               learn_std: bool = True) -> None:
    if std_type != "scalar":
      raise ValueError("FlooredGaussian supports the scalar std parameter only")
    if not 0.0 < float(std_floor) <= float(init_std):
      raise ValueError(f"std floor {std_floor} must lie in (0, init_std {init_std}]")
    super().__init__(output_dim, init_std=init_std, std_range=(float(std_floor), 1.0e6), std_type="scalar",
                     learn_std=learn_std)
    self.std_floor = float(std_floor)

  def project(self) -> None:
    if self.std_param.requires_grad and not torch.is_inference_mode_enabled():
      with torch.no_grad():
        if bool((self.std_param < self.std_floor).any()):
          self.std_param.clamp_(min=self.std_floor)

  def update(self, mlp_output: torch.Tensor) -> None:
    self.project()
    super().update(mlp_output)


class FlooredScaledGaussian(FlooredGaussian):
  """FlooredGaussian with ar1.ScaledGaussian's per-sample std multiplier (the AR(1) conditional; variant XAn only)."""

  scale: torch.Tensor | None = None

  def update(self, mlp_output: torch.Tensor) -> None:
    super().update(mlp_output)
    if self.scale is not None:
      self._distribution = Normal(self._distribution.mean, self._distribution.stddev * self.scale)


# ------------------------------------------------------------------------------------------------------ azimuth curriculum
class torso_hits_azimuth(R2.torso_hits):
  """R2.torso_hits with the XC curriculum: while the update index (env.common_step_counter // steps_per_update) is below
  until_update, a hit's azimuth is the same single uniform draw mapped onto [exclude, 2 pi - exclude] (no hit within
  0 +- exclude of the heading); afterwards the parent's uniform [0, 2 pi). Everything else is the parent's code path."""

  def __init__(self, cfg, env):
    self._excluding = False                       # read by _u, which the parent's constructor already calls
    p = cfg.params
    self.half = math.radians(float(p["exclude_half_deg"]))
    self.until = int(p["until_update"])
    self.spu = int(p["steps_per_update"])
    if not 0.0 < self.half < math.pi or self.until < 0 or self.spu <= 0:
      raise ValueError("bad azimuth curriculum parameters")
    for key in ("impulse_ns", "height_m", "gap_s", "first_s"):     # only the azimuth draw may use (0, 2 pi)
      if tuple(float(x) for x in p[key]) == (0.0, 2.0 * math.pi):
        raise ValueError(f"hit range {key} equals the azimuth range")
    super().__init__(cfg, env)

  def _u(self, lo, hi, n):
    if self._excluding and lo == 0.0 and hi == 2.0 * math.pi:      # the azimuth draw of the parent's __call__
      return super()._u(self.half, 2.0 * math.pi - self.half, n)
    return super()._u(lo, hi, n)

  def __call__(self, env, env_ids, impulse_ns, height_m, first_s, gap_s, exclude_half_deg: float = AZ_EXCLUDE_HALF_DEG,
               until_update: int = AZ_UNTIL_UPDATE, steps_per_update: int = STEPS_PER_UPDATE, code_sha256: str = "",
               distribution_sha256: str = ""):
    update = int(env.common_step_counter) // self.spu
    self._excluding = update < self.until
    log = env.extras.setdefault("log", {})
    log["Curriculum/frontal_hits_excluded"] = float(self._excluding)
    try:
      return super().__call__(env, env_ids, impulse_ns, height_m, first_s, gap_s, code_sha256, distribution_sha256)
    finally:
      self._excluding = False


# ---------------------------------------------------------------------------------------------------------- configurations
def env_cfg(lead: bool, starts: bool = False, azimuth: bool = False, play: bool = False, lead_value: float | None = None):
  """The W1 point-posture environment (C0 without b, B1 with b) plus the stepped starts and / or the azimuth curriculum
  (training only). With starts=azimuth=False this is exactly adv_hold_w1_c0 / adv_hold_w1_b1."""
  kwargs = {} if lead_value is None else dict(lead_value=lead_value)
  cfg = W1.hold_w1_env_cfg(lead=lead, posture="point", play=play, **kwargs)
  if play:
    return cfg
  if starts:
    if not START_BANK["sha256"]:
      raise ValueError("the stepped-start bank is not pinned yet")
    cfg.events["stepped_start"] = EventTermCfg(func=stepped_start, mode="reset", params=dict(
      bank_file=START_BANK["file"], bank_sha256=START_BANK["sha256"], fraction=START_FRACTION))
  if azimuth:
    term = cfg.events["torso_hits"]
    term.func = torso_hits_azimuth
    term.params.update(exclude_half_deg=AZ_EXCLUDE_HALF_DEG, until_update=AZ_UNTIL_UPDATE, steps_per_update=STEPS_PER_UPDATE)
  return cfg


def rl_cfg(floor: bool = False, plain: bool = False, ar1: bool = False, save_interval: int = SAVE_INTERVAL,
           max_iterations: int = 1500):
  """The W1 PPO recipe (P-e0: entropy 0, HoldSymmetricMLP) plus the std floor / the plain actor / the AR(1) variant, and
  the checkpoint interval. floor=plain=ar1=False with save_interval=100 is exactly hold_w1_rl_cfg()."""
  if plain and (floor or ar1):
    raise ValueError("the plain actor is its own cell")
  cfg = W1.hold_w1_rl_cfg(max_iterations)
  cfg.save_interval = int(save_interval)
  if floor:
    cfg.actor.distribution_cfg = dict(class_name=("r1adv.positive.hold_w2:FlooredScaledGaussian" if ar1 else
                                                  "r1adv.positive.hold_w2:FlooredGaussian"),
                                      init_std=1.0, std_type="scalar", std_floor=STD_FLOOR)
  if ar1:
    if not floor:
      cfg.actor.distribution_cfg = dict(cfg.actor.distribution_cfg, class_name="r1adv.positive.ar1:ScaledGaussian")
    cfg.actor.class_name = "r1adv.positive.ar1:AR1HoldSymmetricMLP"
    cfg.algorithm.class_name = "r1adv.positive.ar1:AR1PPO"
  if plain:
    cfg.actor.class_name = "r1adv.positive.hold_plain:HoldPlainMLP"
  return cfg


def cell_env_cfg(cell: str, play: bool = False, lead_value: float | None = None):
  c = CELLS[cell]
  return env_cfg(c["lead"], starts=c["starts"], azimuth=c["azimuth"], play=play, lead_value=lead_value)


def cell_rl_cfg(cell: str, max_iterations: int = 1500):
  c = CELLS[cell]
  return rl_cfg(floor=c["floor"], plain=c["plain"], ar1=c["ar1"], max_iterations=max_iterations)


if W1.hold_w1_rl_cfg().num_steps_per_env != STEPS_PER_UPDATE:
  raise RuntimeError("the azimuth curriculum's update index assumes 24 environment steps per PPO update")


def record() -> dict:
  return dict(code_sha256=R2.CODE_SHA256, cells=CELLS, start_bank=dict(START_BANK), start_fraction=START_FRACTION,
              start_seed=START_SEED, std_floor=STD_FLOOR, azimuth=dict(exclude_half_deg=AZ_EXCLUDE_HALF_DEG,
              until_update=AZ_UNTIL_UPDATE, steps_per_update=STEPS_PER_UPDATE), save_interval=SAVE_INTERVAL,
              pink_note=PINK_NOTE, base_tasks=dict(C0="adv_hold_w1_c0", B1="adv_hold_w1_b1"))
