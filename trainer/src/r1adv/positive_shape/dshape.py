"""D_shape minimal version (positive session 15; coordinator 2026-11-19 02:4x KST, rounds/R04/09B_r9_synthesis.md 2 W1 and
4.5): positive 10's online discriminator v1 (r1adv.positive.amp_online.OnlineAMP: LSGAN + gradient penalty, AMP map
max(0, 1 - (d - 1)^2 / 4), no stimulus context, no abstention, same banks, rings, cadence, augmentation) with ONLY its
features changed: the 27 position-free phi channels (D*-noPos) plus ten dimensionless shape channels per window sample.
It takes the D slot of B0's Hold objective (careful_walk_r7a_gmm_o6: r = L + beta D, beta 1, L unchanged), so the GMM
density D (HumanSurprise) drops out; no reward term is added.

Shape channels (one definition for human skeletons and the robot, shape_raw below; positions after the phi chain:
zero-phase 2nd-order Butterworth 6 Hz)
  torso frame  z_t = unit(shoulder midpoint - hip midpoint), y_t = left shoulder - right shoulder made orthogonal to z_t,
               x_t = y_t x z_t (forward)
  0-2 / 3-5    (wrist - shoulder) of the left / right arm in the torso frame / arm length (upper arm + forearm)
  6 / 7        upper-arm elevation: angle between (elbow - shoulder) and -z_t (0 = hanging, pi/2 = horizontal), rad
  8 / 9        trunk tilt against gravity in the heading frame of the hip axis (phi's heading): forward / leftward, rad
  every channel minus the subject's own rest value (the phi body-tell rule, r6_a1_phi 'quiet offset'): human = median over
  the clip's quiet slice (phi's Q: [onset - 0.8 s, onset - 0.1 s) with a push onset, else the whole clip); robot = the
  value at q*_r7a (the parents' quiet standing, hold_target_r7a_v1.json; L's own target). Reason: without it the robot's
  resting arm (elbow 0.73 rad, upper arm 0.19 m vs forearm 0.12 m) and torso landmarks differ from a person's at rest,
  L pins that posture (sigma 0.02 rad), and the discriminator would separate every robot window on rest geometry.
  Landmarks: human concept_v1 canon shoulder_l/r, elbow_l/r, wrist_l/r, hip_l/r; robot the link origins of
  left/right_shoulder_roll_link (the most lateral shoulder joint), _elbow_link, _wrist_roll_link (as phi's wrists),
  _hip_pitch_link (as phi's hips). Arm length: human per clip (median over frames, mean of both sides), robot constant.
Robot side (ShapeStream): landmark positions each control step, the steady-state impulse response of filtfilt (6 Hz,
2nd order, 50 Hz; +-LAG taps, sum 1) centred on frame f = L - LAG (the frame phi emits), channels, a ring, the window by
phi's tau-grid interpolation; a window is valid where phi's is and every sample frame is >= LAG (no padded taps).
Human side: .agent/adversarial/positive/tools/r11_dshape_bank.py (per-frame channels aligned with amp_real_v1.npz).
Masks: the 'hands missing / masked' flag of v1 also zeroes the ten shape channels (in the human bank the upper-body
landmarks are missing together: dos Santos quiet stance and UTwente have no arms or shoulders, 22.5 % of the window ends;
a masked real window must not carry a trunk value that a masked robot window lacks); mirror augmentation swaps the arms
and flips the lateral components (y, tilt_l).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from r1adv.positive import amp_online as AO
from r1adv.positive import phi_dstar as PD

K = 10
SHAPE_NAMES = ("hsL_x", "hsL_y", "hsL_z", "hsR_x", "hsR_y", "hsR_z", "elevL", "elevR", "tilt_f", "tilt_l")
MASKED_SHAPE = tuple(range(K))                                              # zeroed with the hand flag
NCS = PD.NC + K
SHAPE_CH = tuple(range(PD.NC, NCS))
KEEP = tuple(AO.NOPOS) + SHAPE_CH
HAND_CH_S = tuple(AO.HAND_CH) + tuple(PD.NC + c for c in MASKED_SHAPE)
M_PERM = (3, 4, 5, 0, 1, 2, 7, 6, 8, 9)
M_SIGN = (1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0)
LAG = 14
POINTS = ("sh_l", "sh_r", "el_l", "el_r", "wr_l", "wr_r", "hip_l", "hip_r")
ROBOT_BODIES = dict(sh_l="left_shoulder_roll_link", sh_r="right_shoulder_roll_link", el_l="left_elbow_link",
                    el_r="right_elbow_link", wr_l="left_wrist_roll_link", wr_r="right_wrist_roll_link",
                    hip_l="left_hip_pitch_link", hip_r="right_hip_pitch_link")
HUMAN_JOINTS = dict(sh_l="shoulder_l", sh_r="shoulder_r", el_l="elbow_l", el_r="elbow_r", wr_l="wrist_l", wr_r="wrist_r",
                    hip_l="hip_l", hip_r="hip_r")
REST_TARGET = dict(file="hold_target_r7a_v1.json", sha256="41a5f5ec3e56e587ccd60f170a036a65cae75a98247dad23a493a378b15bbb9c")
SHAPE_FILES = dict(real_file="amp_real_v1.npz", real_sha256="60ba21fb99a4df2beb48317cfc7c2ed2c66119ada52379266bfc042eeaad987a",
                   shape_file="amp_shape_v1.npz", shape_sha256="33dff7a18b057cbd93f08353a78aca4c0df7c813c1c4f05cc127c30954114b4d",
                   fail_file="amp_fail_v1.npz", fail_sha256="66d4f70e75fd5a89b3afef663f77f7d2433d32470b058db466b27268508e06a0",
                   fail_shape_file="amp_fail_shape_v1.npz",
                   fail_shape_sha256="9c875913bac1def4f78b7b813446cef199ccc754b1164681b89b09fa51c7daf8")


# --------------------------------------------------------------------------------------------------------- definitions
def _unit(v: torch.Tensor) -> torch.Tensor:
  return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def arm_length(P: dict) -> torch.Tensor:
  """upper arm + forearm, mean of both sides (..,)."""
  lo = (P["el_l"] - P["sh_l"]).norm(dim=-1) + (P["wr_l"] - P["el_l"]).norm(dim=-1)
  ro = (P["el_r"] - P["sh_r"]).norm(dim=-1) + (P["wr_r"] - P["el_r"]).norm(dim=-1)
  return 0.5 * (lo + ro)


def shape_raw(P: dict, arm_len) -> torch.Tensor:
  """landmarks P[name] (..., 3) (z up), arm length (scalar or (...)) -> (..., 10) channels before the rest offset."""
  S = 0.5 * (P["sh_l"] + P["sh_r"])
  H = 0.5 * (P["hip_l"] + P["hip_r"])
  z = _unit(S - H)
  yv = P["sh_l"] - P["sh_r"]
  y = _unit(yv - (yv * z).sum(-1, keepdim=True) * z)
  x = torch.cross(y, z, dim=-1)
  L = torch.as_tensor(arm_len, dtype=S.dtype, device=S.device)
  L = L[..., None] if L.dim() else L

  def tf(v):
    return torch.stack([(v * x).sum(-1), (v * y).sum(-1), (v * z).sum(-1)], -1)
  hs_l = tf(P["wr_l"] - P["sh_l"]) / L
  hs_r = tf(P["wr_r"] - P["sh_r"]) / L

  def elev(e, s):
    u = e - s
    return torch.acos(((u * -z).sum(-1) / u.norm(dim=-1).clamp_min(1e-9)).clamp(-1.0, 1.0))
  lv = _unit(P["hip_l"][..., :2] - P["hip_r"][..., :2])
  fwd = torch.stack([lv[..., 1], -lv[..., 0]], -1)
  tf_ = torch.atan2((z[..., :2] * fwd).sum(-1), z[..., 2])
  tl_ = torch.atan2((z[..., :2] * lv).sum(-1), z[..., 2])
  return torch.cat([hs_l, hs_r, elev(P["el_l"], P["sh_l"])[..., None], elev(P["el_r"], P["sh_r"])[..., None],
                    tf_[..., None], tl_[..., None]], -1)


def mirror_shape(S: torch.Tensor) -> torch.Tensor:
  return S[..., list(M_PERM)] * torch.tensor(M_SIGN, dtype=S.dtype, device=S.device)


def mirror_window_s(W: torch.Tensor) -> torch.Tensor:
  """(..., 58) windows: phi part by phi_dstar.mirror_window, shape part by mirror_shape."""
  return torch.cat([PD.mirror_window(W[..., :PD.NC]), mirror_shape(W[..., PD.NC:])], -1)


def fir_kernel(lag: int = LAG) -> np.ndarray:
  """steady-state impulse response of scipy filtfilt(butter(2, 6 / 25)) at 50 Hz, taps -lag..lag, normalised to sum 1."""
  from scipy.signal import filtfilt
  b, a, _ = PD.butter2()
  n = 8 * lag + 1
  x = np.zeros(n)
  x[n // 2] = 1.0
  y = filtfilt(b, a, x)
  h = y[n // 2 - lag:n // 2 + lag + 1]
  return h / h.sum()


def robot_rest(input_dirs) -> tuple[np.ndarray, float, dict]:
  """(rest channels (10,), robot arm length, landmarks) at q*_r7a: r1lab O6 body (the B0 body's geometry) posed with the
  pinned target's qpos (root + 24 joints), MuJoCo kinematics on the CPU."""
  import mujoco
  from r1lab.robot import get_spec_o6
  t = json.loads(AO.pinned_path(REST_TARGET["file"], REST_TARGET["sha256"], input_dirs).read_text())
  m = get_spec_o6().compile()
  d = mujoco.MjData(m)
  q = np.asarray(t["qpos"], float)
  if len(q) != m.nq:
    raise ValueError(f"target qpos has {len(q)} entries, the O6 body {m.nq}")
  d.qpos[:] = q
  mujoco.mj_kinematics(m, d)
  P = {k: torch.tensor(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy(), dtype=torch.float64)
       for k, b in ROBOT_BODIES.items()}
  L = float(arm_length(P))
  return shape_raw(P, L).numpy(), L, {k: v.tolist() for k, v in P.items()}


# --------------------------------------------------------------------------------------------------------- robot stream
class ShapeStream:
  """Shape channels of every environment, aligned with PhiStream (frame f = L - lag, global step s - lag)."""

  def __init__(self, env, rest: np.ndarray, arm_len: float, lag: int = LAG, w0: float | None = None):
    self.env, self.lag = env, int(lag)
    self.ent = env.scene["robot"]
    names = list(self.ent.body_names)
    self.bidx = torch.tensor([names.index(ROBOT_BODIES[k]) for k in POINTS], device=env.device)
    self.N, self.dev = env.num_envs, torch.device(env.device)
    self.R, self.RP = 4 * self.lag + 8, 40
    self.raw = torch.zeros(self.R, self.N, len(POINTS), 3, dtype=torch.float32, device=self.dev)
    self.ch = torch.full((self.RP, self.N, K), math.nan, dtype=torch.float32, device=self.dev)
    self.h = torch.tensor(fir_kernel(self.lag), dtype=torch.float32, device=self.dev)
    self.taps = torch.arange(2 * self.lag + 1, device=self.dev)
    self.rest = torch.tensor(rest, dtype=torch.float32, device=self.dev)
    self.arm_len = float(arm_len)
    w0 = math.sqrt(PD.G / AO.R1_FIXED_REFERENCE["z0"]) if w0 is None else float(w0)
    self.step_frames = PD.DTAU / w0 * PD.FS
    k = torch.arange(-(PD.WIN - 1), 1, device=self.dev, dtype=torch.float32)
    fi = k * self.step_frames
    i1 = torch.ceil(fi)
    self.a = (fi - (i1 - 1.0))
    self.i0 = (i1 - 1.0).long()
    self.min_off = int(self.i0.min())
    self.n = torch.zeros(self.N, dtype=torch.long, device=self.dev)
    self.s = 0
    self.rows = torch.arange(self.N, device=self.dev)

  def reset(self, env_ids=None):
    if env_ids is None:
      self.n[:] = 0
    else:
      self.n[torch.as_tensor(env_ids, device=self.dev).long()] = 0

  @torch.no_grad()
  def update(self) -> tuple[torch.Tensor, torch.Tensor]:
    """one sample per environment; -> (window (N, 9, 10) ending at frame L - lag, valid (N,): every sample frame >= lag)."""
    s, lag = self.s, self.lag
    o = self.env.scene.env_origins
    pts = self.ent.data.body_link_pos_w[:, self.bidx] - o[:, None, :]
    L = self.n.clone()
    self.raw[s % self.R] = pts.float()
    self.n += 1
    frame = (L[None, :] - 2 * lag + self.taps[:, None]).clamp_min(0)          # (taps, N) episode frames, edge replicate
    gstep = s - L[None, :] + frame
    X = self.raw[gstep % self.R, self.rows[None, :]]                         # (taps, N, 8, 3)
    F = (self.h[:, None, None, None] * X).sum(0).double()                   # filtered landmarks of frame L - lag
    P = {k: F[:, i] for i, k in enumerate(POINTS)}
    c = (shape_raw(P, self.arm_len).float() - self.rest)
    f = L - lag
    self.ch[(s - lag) % self.RP] = torch.where((f >= 0)[:, None], c, torch.full_like(c, math.nan))
    sf = s - lag
    g0 = self.ch[(sf + self.i0) % self.RP]                                   # (9, N, 10)
    g1 = self.ch[(sf + self.i0 + 1) % self.RP]
    W = ((1 - self.a)[:, None, None] * g0 + self.a[:, None, None] * g1).transpose(0, 1)
    ok = (f + self.min_off) >= lag
    self.s += 1
    return torch.where(ok[:, None, None], W, torch.full_like(W, math.nan)), ok


# --------------------------------------------------------------------------------------------------------- banks / rings
class ShapeRealBank(AO.RealBank):
  """v1 human bank (amp_real_v1.npz) + its per-frame shape channels (amp_shape_v1.npz, same frame order): windows of 58
  channels; ends whose shape window has a non-finite value count as 'hands missing' (shape channels masked)."""

  def __init__(self, path, shape_path, device, generator, source_weights="sqrt_hours", exclude=("stylized",)):
    super().__init__(path, device, generator, source_weights, exclude)
    z = np.load(path, allow_pickle=False)
    s = np.load(shape_path, allow_pickle=False)
    meta = json.loads(str(s["meta"]))
    if s["shape"].shape[0] != self.phi.shape[0] or list(s["clip_id"]) != list(z["clip_id"]):
      raise ValueError("shape file not aligned with the v1 bank")
    if list(meta["names"]) != list(SHAPE_NAMES):
      raise ValueError("shape channel names differ")
    self.shape_meta = meta
    self.phi = torch.cat([self.phi, torch.from_numpy(s["shape"].astype(np.float16)).to(self.dev)], 1)
    self.mu = torch.cat([self.mu, torch.from_numpy(s["mu"].astype(np.float32)).to(self.dev)])
    self.sd = torch.cat([self.sd, torch.from_numpy(s["sd"].astype(np.float32)).to(self.dev)])
    src = z["end_src"].astype(np.int64)
    keep = np.array([self.share[x] for x in self.sources])[src] > 0
    miss = torch.from_numpy(s["end_shape_missing"][keep].astype(bool)).to(self.dev)
    if miss.shape[0] != self.end.shape[0]:
      raise ValueError("shape end flags not aligned with the kept ends")
    self.hand_miss = self.hand_miss | miss
    self.shape_missing_share = float(miss.float().mean())


class ShapeFailBank(AO.FailBank):
  """v1 failure bank windows (amp_fail_v1.npz) + their shape windows (amp_fail_shape_v1.npz, same order)."""

  def __init__(self, path, shape_path, device, generator, lying: bool):
    z = np.load(path, allow_pickle=False)
    s = np.load(shape_path, allow_pickle=False)
    if s["windows"].shape[0] != z["windows"].shape[0] or not np.array_equal(s["kind"], z["kind"]):
      raise ValueError("fail shape windows not aligned with the v1 failure bank")
    kind = z["kind"].astype(np.int64)
    keep = kind < (3 if lying else 2)
    W = np.concatenate([z["windows"][keep], s["windows"][keep].astype(np.float16)], -1)
    self.W = torch.from_numpy(W).to(device)
    self.meta = json.loads(str(z["meta"]))
    self.kinds = {int(k): int((kind[keep] == k).sum()) for k in np.unique(kind[keep])}
    self.g, self.dev = generator, torch.device(device)
    if any(p == "manufacturer" for p in self.meta.get("policies", [])):
      raise ValueError("the failure bank holds manufacturer windows")


class ShapeRing(AO.WindowRing):
  def __init__(self, capacity: int, device):
    self.W = torch.zeros(int(capacity), PD.WIN, NCS, dtype=torch.float16, device=device)
    self.ok = torch.zeros(int(capacity), dtype=torch.bool, device=device)
    self.i, self.full = 0, False


# --------------------------------------------------------------------------------------------------------- discriminator
def prep_s(W, mu, sd, keep, hand_mask=None, head_mask=None):
  """AO.prep on 58-channel windows: the hand mask also zeroes the ten shape channels."""
  n = W.shape[0]
  Z = torch.nan_to_num((W - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
  hm = torch.zeros(n, dtype=torch.bool, device=W.device) if hand_mask is None else hand_mask
  dm = torch.zeros(n, dtype=torch.bool, device=W.device) if head_mask is None else head_mask
  if hand_mask is not None:
    Z[:, :, list(HAND_CH_S)] = Z[:, :, list(HAND_CH_S)] * (~hm)[:, None, None]
  if head_mask is not None:
    Z[:, :, list(AO.HEAD_CH)] = Z[:, :, list(AO.HEAD_CH)] * (~dm)[:, None, None]
  Z = Z[:, :, list(keep)]
  return torch.cat([Z.reshape(n, -1), hm.float()[:, None], dm.float()[:, None]], 1)


class OnlineDiscShape(AO.OnlineDisc):
  """v1 discriminator (same net, optimiser, LSGAN + gradient penalty, mirror / mask augmentation) on KEEP channels."""

  def score(self, W):
    with torch.no_grad():
      return self.net(prep_s(W, self.mu, self.sd, self.keep))

  def _aug(self, W, g, miss_hand=None, miss_head=None):
    n = W.shape[0]
    m = torch.rand(n, device=W.device, generator=g) < self.mirror_p
    W = torch.where(m[:, None, None], mirror_window_s(W), W)
    hm = torch.rand(n, device=W.device, generator=g) < self.mask_p
    dm = torch.rand(n, device=W.device, generator=g) < self.mask_p
    if miss_hand is not None:
      hm = hm | miss_hand
    if miss_head is not None:
      dm = dm | miss_head
    return prep_s(W, self.mu, self.sd, self.keep, hm, dm)


# --------------------------------------------------------------------------------------------------------- engine
DEFAULTS_S = dict(AO.DEFAULTS, shape_file="", shape_sha256="", fail_shape_file="", fail_shape_sha256="", load_path="")


def resolve_s(params: dict) -> dict:
  p = dict(DEFAULTS_S)
  unknown = set(params) - set(DEFAULTS_S)
  if unknown:
    raise ValueError(f"unknown D_shape parameters {sorted(unknown)}")
  p.update(params)
  base = AO.resolve({k: v for k, v in p.items() if k in AO.DEFAULTS})
  if base["channels"] != "nopos" or base["reference"] != "fixed":
    raise ValueError("D_shape uses v1's nopos channels and the fixed phi reference")
  p.update(base)
  return p


class OnlineAMPShape(AO.OnlineAMP):
  """OnlineAMP v1 with 58-channel windows (phi 48 + shape 10) and the KEEP feature set; everything else is v1's."""

  def __init__(self, env, params: dict, input_dirs):
    p = resolve_s(params)
    self.p, self.env = p, env
    dev = env.device
    self.N, self.dev = env.num_envs, torch.device(dev)
    seed = int(env.cfg.seed) * 7919 + AO.AMP_SEED
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(seed)
    self.keep = KEEP
    self.reader = PD.PhiEnvReader(env)
    self.stream = PD.PhiStream(self.N, len(self.reader.seg_names), self.reader.seg_mass, self.reader.foot_radius,
                               lag=int(p["lag"]), reference=("fixed", AO.R1_FIXED_REFERENCE), scorer=None, device=dev)
    rest, arm_len, marks = robot_rest(input_dirs)
    self.rest, self.arm_len, self.rest_marks = rest, arm_len, marks
    self.shape = ShapeStream(env, rest, arm_len, lag=int(p["lag"]))
    self.real = ShapeRealBank(AO.pinned_path(p["real_file"], p["real_sha256"], input_dirs),
                              AO.pinned_path(p["shape_file"], p["shape_sha256"], input_dirs), dev, self.g,
                              p["source_weights"], tuple(p["exclude_sources"]))
    lying = False if p["fail_lying"] == "auto" else bool(p["fail_lying"])
    self.fail = (ShapeFailBank(AO.pinned_path(p["fail_file"], p["fail_sha256"], input_dirs),
                               AO.pinned_path(p["fail_shape_file"], p["fail_shape_sha256"], input_dirs), dev, self.g, lying)
                 if float(p["fail_share"]) > 0 else None)
    self.disc = OnlineDiscShape(self.keep, self.real.mu, self.real.sd, dev, seed, hidden=p["hidden"], lr=p["lr"],
                                weight_decay=p["weight_decay"], grad_penalty=p["grad_penalty"], grad_clip=p["grad_clip"],
                                mirror_p=p["mirror_p"], mask_p=p["mask_p"])
    self.buf = ShapeRing(p["buffer"], dev)
    self.fallbuf = ShapeRing(p["fall_buffer"], dev)
    self.recent = torch.zeros(AO.RECENT, self.N, PD.WIN, NCS, dtype=torch.float16, device=dev)
    self.recent_ok = torch.zeros(AO.RECENT, self.N, dtype=torch.bool, device=dev)
    self.push_n = int(min(self.N, p["push_per_step"]))
    self.steps, self.updates, self.skipped = 0, 0, 0
    self.fake_still = None
    self.ceiling = float(p["ceiling"]) if isinstance(p["ceiling"], (int, float)) else None
    self.hits = None
    self.last_log = {}
    if p["load_path"]:                       # continue a saved discriminator (main run after the critic warm-up): weights,
      self.load(p["load_path"])              # optimiser, update count, quiet-share EMA, ceiling

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self.shape.reset(env_ids)

  @torch.no_grad()
  def step(self, env) -> torch.Tensor:
    """v1's step on the combined window (phi window and shape window of the same frame)."""
    out = self.stream.update(self.reader.read())
    Ws, oks = self.shape.update()
    W = torch.cat([out["window"], Ws], -1)
    ok = out["valid"] & oks
    d = self.disc.score(torch.nan_to_num(W, nan=0.0))
    r = AO.amp_reward(d, self.ceiling)
    D = torch.where(ok, r, torch.zeros_like(r))
    idx = torch.randint(0, self.N, (self.push_n,), device=self.dev, generator=self.g)
    self.buf.push(W[idx], ok[idx])
    slot = self.steps % AO.RECENT
    self.recent[slot] = torch.nan_to_num(W, nan=0.0).half()
    self.recent_ok[slot] = ok
    if float(self.p["fall_share"]) > 0:
      fell = env.reset_terminated
      if bool(fell.any()):
        ids = fell.nonzero().flatten()
        self.fallbuf.push(self.recent[:, ids].reshape(-1, PD.WIN, NCS), self.recent_ok[:, ids].reshape(-1))
    self.steps += 1
    log = env.extras.setdefault("log", {})
    okf = ok.float()
    nv = okf.sum().clamp_min(1.0)
    log["AMP/valid_share"] = okf.mean()
    log["AMP/D_mean_valid"] = (D * okf).sum() / nv
    log["AMP/d_mean_valid"] = (torch.nan_to_num(d) * okf).sum() / nv
    log["AMP/D_lt_005_valid"] = ((r < 0.05).float() * okf).sum() / nv
    log["AMP/D_gt_095_valid"] = ((r > 0.95).float() * okf).sum() / nv
    log["AMP/acc_fake_fresh"] = ((d < 0).float() * okf).sum() / nv
    zs = torch.nan_to_num(Ws[:, -1]) * okf[:, None]                          # newest sample's shape channels (valid rows)
    log["Shape/hand_z_above_rest_max"] = torch.maximum(zs[:, 2], zs[:, 5]).max()
    log["Shape/elev_mean_valid"] = 0.5 * (zs[:, 6] + zs[:, 7]).sum() / nv
    log["Shape/tilt_f_mean_valid"] = zs[:, 8].sum() / nv
    ht = self._hit_term(env)
    if ht is not None:
      onset = torch.round(ht.last[:, 0] / env.step_dt).long()
      age = (env.episode_length_buf - int(self.p["lag"])) - (onset + 1)
      react = ok & (ht.count > 0) & (age >= 0) & (age < int(round(6.0 / math.sqrt(PD.G / AO.R1_FIXED_REFERENCE["z0"]) * 50)))
      rf = react.float()
      nr = rf.sum()
      log["AMP/react_share_valid"] = nr / nv
      log["AMP/react_n"] = nr
      log["AMP/D_lt_005_react"] = ((r < 0.05).float() * rf).sum() / nr.clamp_min(1.0)
      log["AMP/D_mean_react"] = (D * rf).sum() / nr.clamp_min(1.0)
    if self.steps % int(self.p["update_every"]) == 0:
      self.update(log)
    for k, v in self.last_log.items():
      log[k] = v
    return D


# --------------------------------------------------------------------------------------------------------- objective
def _objective_base():
  from r1adv.positive import s2f_r7a as R7A
  return R7A.HoldObjectiveR7A


class HoldObjectiveR7AShape(_objective_base()):
  """B0's objective term (HoldObjectiveR7A: L with target q*_r7a) with D = the D_shape discriminator's reward; the
  HumanSurprise density is built by the base class but never evaluated. Everything this term builds or draws stays off
  the global RNG (as positive 10's HoldObjectiveAMP)."""

  def __init__(self, cfg, env):
    from r1adv.positive import hold_task as HT
    dev = torch.device(env.device)
    forked = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=forked):
      super().__init__(cfg, env)
      self.surprise = None
      self.amp = OnlineAMPShape(env, cfg.params["amp"], HT.INPUT_DIRS)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self.amp.reset(None if env_ids is None or isinstance(env_ids, slice) else env_ids)

  def __call__(self, env, beta: float, sigma: dict, form: str, formula_sha256: str = "", target_file: str = "",
               target_sha256: str = "", amp: dict | None = None) -> torch.Tensor:
    from r1adv.positive import objective as OBJ
    del target_file, target_sha256, amp, formula_sha256
    ch = self.channels()
    like = OBJ.mission_likelihood(ch, sigma, form)
    natural = self.amp.step(env)
    r = OBJ.reward(like, natural, beta)
    log = env.extras.setdefault("log", {})
    log["Objective/L"] = like.mean()
    log["Objective/D"] = natural.mean()
    for k, v in ch.items():
      log[f"Objective/err_{k}"] = v.mean()
    return r


def amp_params(files: dict | None = None) -> dict:
  """v1's defaults (AO.DEFAULTS) with the D_shape files. The discriminator state is saved every 50 updates to
  $R1ADV_DSHAPE_SAVE and loaded at the start from $R1ADV_DSHAPE_LOAD when those variables are set (r1lab.train links only
  a term named 'style' to the run directory): one save path per run, and the main run loads its warm-up's state. The
  first two update calls of a run find no valid window yet and do not count, so a 50-iteration warm-up ends at update 48:
  its launcher sets $R1ADV_DSHAPE_SAVE_EVERY = 1 (default 50)."""
  f = dict(SHAPE_FILES, **(files or {}))
  p = resolve_s(dict(f, save_path=os.environ.get("R1ADV_DSHAPE_SAVE", ""), load_path=os.environ.get("R1ADV_DSHAPE_LOAD", ""),
                     save_every_updates=int(os.environ.get("R1ADV_DSHAPE_SAVE_EVERY", AO.DEFAULTS["save_every_updates"]))))
  return {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}


def dshape_env_cfg(play: bool = False, t0: int = 1499, files: dict | None = None):
  """B0's environment (careful_walk_r7a_gmm_o6) with only the objective's D slot replaced (training env only; the play /
  evaluation env stays B0's)."""
  from r1lab.tasks import TASKS
  cfg = TASKS["careful_walk_r7a_gmm_o6"]["env"](play=play, t0=t0)
  if play:
    return cfg
  from r1adv.positive import s2f_r7a as R7A
  term = cfg.rewards["objective"]
  if term.func is not R7A.HoldObjectiveR7A or set(cfg.rewards) != {"objective"}:
    raise ValueError("B0's reward is not the single HoldObjectiveR7A term")
  term.func = HoldObjectiveR7AShape
  term.params["amp"] = amp_params(files)
  return cfg
