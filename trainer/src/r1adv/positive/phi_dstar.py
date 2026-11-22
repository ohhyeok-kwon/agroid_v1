"""In-environment phi (R6 A1 abstract balance features) and the D* window score, vectorised over environments (torch).
Positive session 9, 2026-11-18. A building block only: no task, registry entry or reward uses it (the coordinator decides
r = L + D*). Offline reference: .agent/adversarial/positive/tools/r6_a1_phi.py (phi_track, tau_grid) and r6_a1_dstar.py
(Disc, prep_input, score_record); gate rounds/R04/r6_a1_prior_gate.md.

What is computed per control step (50 Hz) and environment
  inputs (PhiEnvReader.read, local frame = world - env origin): whole-body CoM (sim masses, hands and randomisation
          included, as the runner's record), the 28 sole points (capsule-end centres; z = height above the env origin
          plane minus the capsule radius, as the runner), pelvis, hip-pitch links, head point (torso + (0.015, 0, 0.35)),
          wrist-roll links, and the stock r1.xml body CoMs with their nominal masses (no O6 hands, nominal CoM offsets) for
          the orbital angular momentum -- the same hybrid as the offline robot track (r6_a1_phi.track_robot)
  chain   offline: zero-phase Butterworth 2nd order 6 Hz (scipy filtfilt, odd padding 9) -> central differences
          (np.gradient) -> 48 channels -> 3-frame median of the contact flags. All three are non-causal, so phi of frame f is
          computed when sample f + lag arrives (lag >= 4 steps). The forward pass is scipy's lfilter run causally (same
          start padding and initial state as filtfilt: exact); the backward pass treats the newest sample as the record end
          (filtfilt of the record truncated there, end padding included). The truncation is the only difference from the
          whole-record offline phi and decays as |pole| = 0.59 per frame. phi of the last `lag` frames before an episode
          ends is never computed (so a window never covers them).
  reference  z0 (quiet CoM height), the own-quiet offsets of channels 2:6 and of the endpoint positions, foot length ell and
          leg length lam (offline: per record, from the quiet window [onset - 0.8 s, onset - 0.1 s)). Modes:
            ('window', qa, qb)  medians over episode frames [qa, qb) (offline replica: (160, 195) for records with the
                                event at frame 200; hit-agnostic training default: (40, 75), 0.8-1.5 s after the reset)
            ('fixed', values)   constants for every environment (z0, ell, lam, q24 (4), qep (15))
  window  9 tau-grid samples ending at frame f (tau = omega0 t, step 0.25: about 0.5 s for the R1), linear interpolation,
          contact flags from the nearest frame (r6_a1_phi.tau_grid anchored at f)
  score   D* checkpoints (r6_a1_dstar.train output): standardise with the checkpoint's mu / sd, NaN -> 0, hand flag 0,
          MLP 433-512-256-1, r = max(0, 1 - (d - 1)^2 / 4) per seed, mean over seeds
  valid   False until the reference is set and every window sample lies inside this episode's computed phi; invalid
          windows return NaN scores (never a filler value)

Call site in training: inside a reward term (before the auto-reset), update() once per step; reset(env_ids) from the term's
reset. mjlab's reward stage sees derived quantities one physics substep stale (consistently), which only shifts time.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

G = 9.81
FS = 50.0
DT = 1.0 / FS
FC = 6.0
PAD = 9                          # scipy filtfilt default padlen for a 2nd-order filter
H_C, V_C, HAND_C = 0.03, 0.20, 0.10
DTAU, WIN, NC = 0.25, 9, 48
EP = ("footL", "footR", "handL", "handR", "head")
NAMES = (["s_xi", "ds_xi", "xi_f", "xi_l", "com_f", "com_l", "s_com", "vc_f", "vc_l", "vc_z", "fz", "dfz", "L_f", "L_l", "L_z",
          "supL", "supR", "hand"] + [f"{e}_{q}" for e in EP for q in ("px", "py", "pz", "vx", "vy", "vz")])
FLAGS = (15, 16, 17)
HEAD_IN_TORSO = (0.015, 0.0, 0.35)
FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))
HAND_BODIES = ("left_o6_hand", "right_o6_hand")
# packed raw channels (filtered): com 3 | pelvis 3 | hip_l 3 | hip_r 3 | head 3 | wrist_l 3 | wrist_r 3 | sole 28x3 | segs nb x 3
O_COM, O_PEL, O_HL, O_HR, O_HEAD, O_WL, O_WR, O_SOLE = 0, 3, 6, 9, 12, 15, 18, 21
O_SEG = O_SOLE + 84
# intermediates per frame
I_COM, I_VC, I_ACZ, I_JZ, I_P, I_SZ, I_SV, I_LV, I_WZ, I_L, I_EPOS, I_EVEL = 0, 3, 6, 7, 8, 64, 92, 120, 122, 123, 126, 141
NI = 156


def butter2():
  from scipy.signal import butter, lfilter_zi
  b, a = butter(2, FC / (FS / 2.0))
  return np.asarray(b, float), np.asarray(a, float), np.asarray(lfilter_zi(b, a), float)


def _lf(b, a, zi, xs):
  """scipy.signal.lfilter (transposed direct form II) on a 1-D float64 list with state zi -> (y, z)."""
  z0, z1 = float(zi[0]), float(zi[1])
  out = []
  for x in xs:
    y = b[0] * x + z0
    z0 = b[1] * x - a[1] * y + z1
    z1 = b[2] * x - a[2] * y
    out.append(y)
  return out, (z0, z1)


def init_operator():
  """x[0..9] -> (forward outputs at frames 0..9, state after frame 9) of filtfilt's forward pass (odd start padding,
  initial state lfilter_zi * first padded sample). (12, 10) float64."""
  b, a, zi = butter2()
  B = np.zeros((12, 10))
  for i in range(10):
    x = np.zeros(10)
    x[i] = 1.0
    left = [2 * x[0] - x[k] for k in range(PAD, 0, -1)]
    y, z = _lf(b, a, zi * left[0], left + list(x))
    B[:10, i] = y[PAD:]
    B[10:, i] = z
  return B


def tail_operator(lag: int):
  """(z0, z1 after x[L]; x[L-9..L]; forward outputs y[L-lag-2..L]) -> filtfilt of x[0..L] at frames L-lag-2 .. L-lag+4
  (7 rows). The end padding (2 x[L] - x[L-k], k = 1..9) is continued from the forward state; the backward pass starts
  with lfilter_zi * its first sample, as filtfilt. (7, 2 + 10 + lag + 3) float64."""
  if lag < 4:
    raise ValueError("lag must be >= 4 (3rd derivative stencil + flag median)")
  b, a, zi = butter2()
  ny = lag + 3
  n_in = 2 + 10 + ny
  B = np.zeros((7, n_in))
  for i in range(n_in):
    v = np.zeros(n_in)
    v[i] = 1.0
    z, x, y = v[:2], v[2:12], v[12:]                      # x[k] = x[L-9+k]; y[k] = y[L-lag-2+k]
    right = [2 * x[9] - x[9 - k] for k in range(1, PAD + 1)]
    yext, _ = _lf(b, a, z, right)
    seq = list(reversed(yext)) + list(reversed(list(y)))
    back, _ = _lf(b, a, zi * seq[0], seq)
    # y[k] sits at position PAD + (ny - 1 - k) of seq; outputs for k = 0..6 (frames L-lag-2 .. L-lag+4)
    B[:, i] = [back[PAD + ny - 1 - k] for k in range(7)]
  return B


def np_median(x: torch.Tensor, dim: int) -> torch.Tensor:
  """numpy median semantics (mean of the two middle values for an even count)."""
  s = torch.sort(x, dim=dim).values
  n = x.shape[dim]
  return 0.5 * (s.select(dim, (n - 1) // 2) + s.select(dim, n // 2))


_PAIRS = torch.triu_indices(28, 28, offset=1)


def hull_sd(P: torch.Tensor, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
  """Signed distance (outside +) of 2-D points X (B, Q, 2) to the convex hull of P (B, n, 2), exact: sd(x) =
  max_{|u|=1} (u.x - h_P(u)); the maximiser is a hull-edge normal (inside, or nearest point on an edge) or the direction
  from the nearest vertex (outside), so the max over every point-pair normal (both signs) and every vertex direction is
  exact (r5_dprime_lib.hull_sd). Chunked over B."""
  B, n, _ = P.shape
  out = torch.empty(X.shape[:2], dtype=P.dtype, device=P.device)
  i, j = _PAIRS.to(P.device)
  for s in range(0, B, chunk):
    p, x = P[s:s + chunk], X[s:s + chunk]
    e = p[:, j] - p[:, i]
    nrm = torch.stack([e[..., 1], -e[..., 0]], -1)
    ln = nrm.norm(dim=-1, keepdim=True)
    nrm = torch.where(ln > 1e-12, nrm / ln.clamp_min(1e-12), torch.tensor([1.0, 0.0], dtype=p.dtype, device=p.device))
    U = torch.cat([nrm, -nrm], 1)                                       # (b, D, 2)
    h = torch.bmm(U, p.transpose(1, 2)).amax(-1)                        # support function (b, D)
    ev = (torch.bmm(U, x.transpose(1, 2)) - h[..., None]).amax(1)       # (b, Q)
    d = x[:, :, None, :] - p[:, None, :, :]                             # (b, Q, n, 2)
    dl = d.norm(dim=-1, keepdim=True)
    u = d / dl.clamp_min(1e-12)
    hv = torch.einsum("bqkc,bmc->bqkm", u, p).amax(-1)                  # (b, Q, n)
    vv = (u * x[:, :, None, :]).sum(-1) - hv
    vv = torch.where(dl[..., 0] > 1e-12, vv, torch.full_like(vv, -math.inf))
    out[s:s + chunk] = torch.maximum(ev, vv.amax(-1))
  return out


def mirror_window(W: torch.Tensor) -> torch.Tensor:
  """r6_a1_phi.mirror on (..., 48)."""
  perm = list(range(NC))
  sign = torch.ones(NC, dtype=W.dtype, device=W.device)
  perm[15], perm[16] = 16, 15
  for q in range(6):
    perm[18 + q], perm[24 + q] = 24 + q, 18 + q
    perm[30 + q], perm[36 + q] = 36 + q, 30 + q
  for i in (3, 5, 8, 12, 14):
    sign[i] = -1.0
  for e in range(5):
    sign[18 + 6 * e + 1] = -1.0
    sign[18 + 6 * e + 4] = -1.0
  return W[..., perm] * sign


# ------------------------------------------------------------------------------------------------------------ D* scorer
class DStar(torch.nn.Module):
  """The D* checkpoints of one fold (r6_a1_dstar.train): 433 -> 512 -> 256 -> 1 per seed; r = max(0, 1 - (d-1)^2/4),
  mean over seeds (gate doc 5.1 / r6_a1_gate2: seed mean of r)."""

  def __init__(self, paths, device="cpu"):
    super().__init__()
    import hashlib
    nets, mus, sds, self.meta = [], [], [], []
    self.keep = None                                                    # channel subset (r6_a1_dstar amendment A4 'keep')
    for p in sorted(Path(x) for x in paths):
      ck = torch.load(p, map_location="cpu", weights_only=False)
      keep = None if ck.get("keep") is None else [int(i) for i in ck["keep"]]
      nch = NC if keep is None else len(keep)
      if int(ck["d_in"]) != WIN * nch + 1 or list(ck["names"]) != list(NAMES) or int(ck["win"]) != WIN \
         or float(ck["dtau"]) != DTAU or bool(ck.get("no_contact", False)):
        raise ValueError(f"{p}: not a 9 x 48 (+ hand flag) phi window discriminator of this module's layout")
      if nets and keep != self.keep:
        raise ValueError("checkpoints of one scorer must share the channel subset")
      self.keep = keep
      net = torch.nn.Sequential(torch.nn.Linear(ck["d_in"], 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 1))
      net.load_state_dict({k.replace("net.", "", 1): v for k, v in ck["state"].items()})
      nets.append(net.eval())
      mus.append(torch.as_tensor(np.asarray(ck["mu"], np.float64)))
      sds.append(torch.as_tensor(np.asarray(ck["sd"], np.float64)))
      self.meta.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), fold=ck["fold"], seed=int(ck["seed"])))
    self.nets = torch.nn.ModuleList(nets)
    self.register_buffer("mu", torch.stack(mus).float())
    self.register_buffer("sd", torch.stack(sds).float())
    for prm in self.parameters():
      prm.requires_grad_(False)
    self.to(device)

  @torch.no_grad()
  def forward(self, W: torch.Tensor, mirror_mean: bool = False):
    """W (N, 9, 48) raw phi windows -> (r (N,) seed mean, d (S, N) per seed raw outputs, r_seed (S, N))."""
    W = W.float()
    if mirror_mean:
      ra, da, sa = self.forward(W)
      rb, db, sb = self.forward(mirror_window(W))
      return 0.5 * (ra + rb), torch.stack([da, db]), torch.stack([sa, sb])
    ds = []
    flag = torch.zeros(W.shape[0], 1, device=W.device)
    for k, net in enumerate(self.nets):
      z = torch.nan_to_num((W - self.mu[k]) / self.sd[k], nan=0.0, posinf=0.0, neginf=0.0)
      if self.keep is not None:
        z = z[:, :, self.keep]
      ds.append(net(torch.cat([z.reshape(W.shape[0], -1), flag], 1)).squeeze(-1))
    d = torch.stack(ds)
    rs = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
    return rs.mean(0), d, rs


# ------------------------------------------------------------------------------------------------------------ phi stream
class PhiStream:
  """phi and its tau-grid windows from one sample per control step and environment (see the module docstring).

  update(inp) takes the reader's dict (float64 tensors, local frame); after it, `self.out` holds for every environment the
  newest phi frame (episode frame f = L - lag, where L is the newest sample's frame), its window, `valid`, and -- when a
  scorer is attached -- the D* score of the window (NaN where invalid)."""

  def __init__(self, num_envs: int, n_segs: int, seg_mass, foot_radius: float, lag: int = 14, reference=("window", 40, 75),
               scorer: DStar | None = None, device="cpu", dtype=torch.float64, backlog: int = 34, hull_chunk: int = 4096,
               mirror_mean: bool = False, hull_dtype=torch.float32):
    self.N, self.dev, self.dt_ = int(num_envs), torch.device(device), dtype
    self.lag = int(lag)
    self.nseg = int(n_segs)
    self.C = O_SEG + 3 * self.nseg
    mu = torch.as_tensor(np.asarray(seg_mass, float), dtype=dtype, device=self.dev)
    self.mu_seg = mu / mu.sum()
    self.rad = float(foot_radius)
    self.ref_mode = reference[0]
    self.g_first = max(3, 10 - self.lag)                                # first episode frame with intermediates
    if self.ref_mode == "window":
      self.qa, self.qb = int(reference[1]), int(reference[2])
      if not (self.g_first <= self.qa < self.qb):
        raise ValueError(f"reference window must satisfy {self.g_first} <= qa < qb")
    elif self.ref_mode != "fixed":
      raise ValueError("reference must be ('window', qa, qb) or ('fixed', values)")
    self.backlog = int(backlog)
    self.RX = 16
    self.RY = max(self.lag + 3, 10)
    span = (self.qb - self.qa) if self.ref_mode == "window" else 0
    self.RI = max(40, span + 2, self.backlog + 3)
    self.RQ = max(16, span + self.lag + 2)                              # raw quiet values are stored at arrival
    self.RP = self.RI
    b, a, zi = butter2()
    self.b, self.a = [float(v) for v in b], [float(v) for v in a]
    self.Binit = torch.as_tensor(init_operator(), dtype=dtype, device=self.dev)
    self.Btail = torch.as_tensor(tail_operator(self.lag), dtype=dtype, device=self.dev)
    N, C = self.N, self.C
    kw = dict(dtype=dtype, device=self.dev)
    self.xr = torch.zeros(self.RX, N, C, **kw)
    self.yf = torch.zeros(self.RY, N, C, **kw)
    self.z0 = torch.zeros(N, C, **kw)
    self.z1 = torch.zeros(N, C, **kw)
    self.I = torch.full((self.RI, N, NI), math.nan, **kw)
    self.Q = torch.full((self.RQ, N, 4), math.nan, **kw)              # raw: leg length L / R, foot extent L / R
    self.S = torch.full((self.RI, N, 2), math.nan, **kw)              # s_xi, s_com
    self.PH = torch.full((self.RP, N, NC), math.nan, dtype=torch.float32, device=self.dev)
    self.step = 0
    self.n = torch.zeros(N, dtype=torch.long, device=self.dev)          # samples in the current episode
    self.first_phi = torch.full((N,), 1 << 30, dtype=torch.long, device=self.dev)   # first episode frame with phi
    self.ref_ok = torch.zeros(N, dtype=torch.bool, device=self.dev)
    self.z0r = torch.full((N,), math.nan, **kw)
    self.ell = torch.full((N,), math.nan, **kw)
    self.lam = torch.full((N,), math.nan, **kw)
    self.q24 = torch.full((N, 4), math.nan, **kw)
    self.qep = torch.full((N, 15), math.nan, **kw)
    self.fixed = None
    if self.ref_mode == "fixed":
      self.fixed = {k: torch.as_tensor(np.asarray(v, float), **kw) for k, v in reference[1].items()}
    self.scorer = scorer
    self.mirror_mean = bool(mirror_mean)
    self.hull_chunk = int(hull_chunk)
    self.hull_dtype = hull_dtype
    self.ar10 = torch.arange(10, device=self.dev)
    self.arY = torch.arange(self.lag + 3, device=self.dev)
    self.out = {}
    self.reset(None)

  # ---------------------------------------------------------------------------------------------------------- episodes
  def reset(self, env_ids=None):
    """The next sample of env_ids is frame 0 of a new episode."""
    ids = torch.arange(self.N, device=self.dev) if env_ids is None else torch.as_tensor(env_ids, device=self.dev).long()
    if ids.numel() == 0:
      return
    self.n[ids] = 0
    self.first_phi[ids] = 1 << 30
    self.ref_ok[ids] = False
    if self.fixed is not None:
      f = self.fixed
      self.z0r[ids] = f["z0"]
      self.ell[ids] = f["ell"]
      self.lam[ids] = f["lam"]
      self.q24[ids] = f["q24"]
      self.qep[ids] = f["qep"]

  def _w0(self, ids=None):
    z = self.z0r if ids is None else self.z0r[ids]
    return torch.sqrt(G / z), torch.sqrt(G * z)

  # ---------------------------------------------------------------------------------------------------------- packing
  def _pack(self, inp) -> torch.Tensor:
    N = self.N
    parts = [inp["com"], inp["pelvis"], inp["hip_l"], inp["hip_r"], inp["head"], inp["wrist_l"], inp["wrist_r"],
             inp["sole"].reshape(N, 84), inp["segs"].reshape(N, 3 * self.nseg)]
    return torch.cat([p.to(self.dt_) for p in parts], 1)

  @staticmethod
  def _raw_quiet(inp) -> torch.Tensor:
    """raw (unfiltered) per-frame leg lengths (thigh + shank + ankle height above the foot's lowest sole point) and foot
    extents along each foot's principal axis (r6_a1_phi._r1_track: lam, ell)."""
    sole = inp["sole"]                                                  # (N, 2, 14, 3)
    out = []
    for h, k, a, f in (("hip_l", "knee_l", "ankle_l", 0), ("hip_r", "knee_r", "ankle_r", 1)):
      th = (inp[h] - inp[k]).norm(dim=-1)
      sh = (inp[k] - inp[a]).norm(dim=-1)
      ah = inp[a][:, 2] - sole[:, f, :, 2].amin(-1)
      out.append(th + sh + ah)
    for f in (0, 1):
      p = sole[:, f, :, :2] - sole[:, f, :, :2].mean(1, keepdim=True)  # (N, 14, 2)
      cxx, cyy, cxy = (p[..., 0] ** 2).sum(1), (p[..., 1] ** 2).sum(1), (p[..., 0] * p[..., 1]).sum(1)
      th = 0.5 * torch.atan2(2 * cxy, cxx - cyy)                       # principal axis of the 2x2 scatter
      ax = torch.stack([torch.cos(th), torch.sin(th)], -1)
      pr = (p * ax[:, None, :]).sum(-1)
      out.append(pr.amax(-1) - pr.amin(-1))
    return torch.stack(out, 1)

  # ---------------------------------------------------------------------------------------------------------- update
  @torch.no_grad()
  def update(self, inp) -> dict:
    """One sample per environment (frame L = samples so far in the episode). One host synchronisation per call (the
    reference-capture check); every other branch is a mask."""
    s, N, lag = self.step, self.N, self.lag
    x = self._pack(inp)
    L = self.n.clone()
    self.xr[s % self.RX] = x
    self.Q[s % self.RQ] = self._raw_quiet(inp)
    # forward pass (exact lfilter state from the episode start)
    b, a = self.b, self.a
    y = b[0] * x + self.z0
    z0n = b[1] * x - a[1] * y + self.z1
    z1n = b[2] * x - a[2] * y
    run = (L >= 10)[:, None]
    self.z0 = torch.where(run, z0n, self.z0)
    self.z1 = torch.where(run, z1n, self.z1)
    self.yf[s % self.RY] = torch.where(run, y, self.yf[s % self.RY])
    if s >= 9:                                                          # episodes at frame 9: filtfilt start (masked)
      init = (L == 9)[:, None]
      X10 = self.xr[(s - 9 + self.ar10) % self.RX]                      # (10, N, C)
      o = (self.Binit @ X10.reshape(10, -1)).reshape(12, N, self.C)
      for k in range(10):
        sl = (s - 9 + k) % self.RY
        self.yf[sl] = torch.where(init, o[k], self.yf[sl])
      self.z0 = torch.where(init, o[10], self.z0)
      self.z1 = torch.where(init, o[11], self.z1)
    self.n += 1
    # backward pass of the truncated record -> filtered frames g-3 .. g+3, g = L - lag + 1 (global step sg)
    ready = L >= max(lag + 2, 9)
    sg = s - lag + 1
    g = L - lag + 1
    if s >= max(lag + 2, 9):
      Xin = torch.cat([self.z0[None], self.z1[None], self.xr[(s - 9 + self.ar10) % self.RX],
                       self.yf[(s - lag - 2 + self.arY) % self.RY]], 0)
      R = (self.Btail @ Xin.reshape(Xin.shape[0], -1)).reshape(7, N, self.C)
      Iv = self._intermediates(R)
      self.I[sg % self.RI] = torch.where(ready[:, None], Iv, torch.full_like(Iv, math.nan))
      # regular phi of frame g - 1 for every environment (meaningless rows where no reference: never read, see valid)
      self._phi_frames(None, sg, nb=1)
      if self.ref_mode == "window":
        cap = ready & ~self.ref_ok & (g == self.qb)
      else:
        cap = ready & ~self.ref_ok & (g == self.g_first + 2)            # fixed: first phi frame g_first + 1
      if bool(cap.any()):                                               # the one host synchronisation
        ids = cap.nonzero().flatten()
        if self.ref_mode == "window":
          self._capture(ids, sg)
        else:
          self.ref_ok[ids] = True
          self._phi_frames(ids, sg, nb=1, full_hull=True)
          self.first_phi[ids] = self.g_first + 1
    else:
      self.I[sg % self.RI] = math.nan
    self.step += 1
    return self._emit(L - lag, s - lag)

  # ---------------------------------------------------------------------------------------------------------- frames
  def _intermediates(self, R: torch.Tensor) -> torch.Tensor:
    """7 filtered rows (frames g-3..g+3, (7, N, C)) -> per-frame quantities of frame g that do not depend on the
    reference (np.gradient central differences)."""
    N = self.N
    com = R[:, :, O_COM:O_COM + 3]
    vc = (com[2:] - com[:-2]) / (2 * DT)                               # frames g-2..g+2 (np.gradient, interior)
    ac = (vc[2:] - vc[:-2]) / (2 * DT)                                 # g-1..g+1
    jz = (ac[2, :, 2] - ac[0, :, 2]) / (2 * DT)
    sole = R[:, :, O_SOLE:O_SOLE + 84].reshape(7, N, 28, 3)
    P = sole[3, :, :, :2]
    sz = sole[3, :, :, 2]
    sv = ((sole[4, :, :, :2] - sole[2, :, :, :2]) / (2 * DT)).norm(dim=-1)
    hl, hr = R[3, :, O_HL:O_HL + 3], R[3, :, O_HR:O_HR + 3]
    lv = (hl - hr)[:, :2]
    lv = lv / lv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fwd = torch.stack([lv[:, 1], -lv[:, 0]], -1)
    wz = torch.minimum(R[3, :, O_WL + 2], R[3, :, O_WR + 2])
    seg = R[2:5, :, O_SEG:].reshape(3, N, self.nseg, 3)
    V = (seg[2] - seg[0]) / (2 * DT)
    mu = self.mu_seg[None, :, None]
    c = (mu * seg[1]).sum(1, keepdim=True)
    vcs = (mu * V).sum(1, keepdim=True)
    Lm = (mu * torch.cross(seg[1] - c, V - vcs, dim=-1)).sum(1)
    pel = R[3, :, O_PEL:O_PEL + 3]
    vcom = vc[2]
    fc = sole.reshape(7, N, 2, 14, 3).mean(3)                          # foot centres (7, N, 2, 3)
    pts = [fc[3, :, 0], fc[3, :, 1], R[3, :, O_WL:O_WL + 3], R[3, :, O_WR:O_WR + 3], R[3, :, O_HEAD:O_HEAD + 3]]
    vel = [(fc[4, :, 0] - fc[2, :, 0]) / (2 * DT), (fc[4, :, 1] - fc[2, :, 1]) / (2 * DT),
           (R[4, :, O_WL:O_WL + 3] - R[2, :, O_WL:O_WL + 3]) / (2 * DT) - vcom,
           (R[4, :, O_WR:O_WR + 3] - R[2, :, O_WR:O_WR + 3]) / (2 * DT) - vcom,
           (R[4, :, O_HEAD:O_HEAD + 3] - R[2, :, O_HEAD:O_HEAD + 3]) / (2 * DT) - vcom]
    rot = lambda v: torch.stack([(v[:, :2] * fwd).sum(-1), (v[:, :2] * lv).sum(-1), v[:, 2]], -1)   # noqa: E731
    epos = torch.cat([rot(p - pel) for p in pts], 1)
    evel = torch.cat([rot(v) for v in vel], 1)
    return torch.cat([com[3], vcom, ac[1, :, 2:3], jz[:, None], P.reshape(N, 56), sz, sv, lv, wz[:, None], Lm, epos, evel], 1)

  def _capture(self, ids: torch.Tensor, sg: int) -> None:
    """Reference of ids from episode frames [qa, qb) (frame qb is global step sg), then phi of the backlog frames."""
    qa, qb = self.qa, self.qb
    I = self.I[[(sg - qb + k) % self.RI for k in range(qa, qb)]][:, ids]   # (F, n, NI)
    # every ring is indexed by the frame's global step (frame k of these envs = step sg - qb + k); the raw values were
    # written when the frame arrived, lag - 1 steps before its intermediates (RQ covers span + lag)
    Q = self.Q[[(sg - qb + k) % self.RQ for k in range(qa, qb)]][:, ids]
    z0 = torch.clamp(np_median(I[:, :, I_COM + 2], 0), min=0.3)
    ext = Q[::5, :, 2:4]                                               # frames qa, qa+5, ... (r6_a1_phi._r1_track)
    ell = np_median(ext.permute(1, 0, 2).reshape(len(ids), -1), 1) + 2 * self.rad
    lam = 0.5 * (np_median(Q[:, :, 0], 0) + np_median(Q[:, :, 1], 0))
    w0 = torch.sqrt(G / z0)
    raw = self._balance_raw(I, w0[None], ell[None])                   # (F, n, 4)
    self.z0r[ids], self.ell[ids], self.lam[ids] = z0, ell, lam
    self.q24[ids] = np_median(raw, 0)
    self.qep[ids] = np_median(I[:, :, I_EPOS:I_EPOS + 15], 0)
    self.ref_ok[ids] = True
    nb = min(self.backlog, qb - self.g_first - 1)
    self._phi_frames(ids, sg, nb=nb, full_hull=True)
    self.first_phi[ids] = qb - nb

  @staticmethod
  def _balance_raw(I, w0, ell):
    """channels 2:6 before the own-quiet offset: (xi - hull centre), (c - hull centre) in the heading frame / ell."""
    com, vc = I[..., I_COM:I_COM + 2], I[..., I_VC:I_VC + 2]
    P = I[..., I_P:I_P + 56].reshape(I.shape[:-1] + (28, 2))
    ctr = P.mean(-2)
    lv = I[..., I_LV:I_LV + 2]
    fwd = torch.stack([lv[..., 1], -lv[..., 0]], -1)
    xi = com + vc / w0[..., None]
    rot = lambda v: torch.stack([(v * fwd).sum(-1), (v * lv).sum(-1)], -1)   # noqa: E731
    return torch.cat([rot(xi - ctr), rot(com - ctr)], -1) / ell[..., None]

  def _phi_frames(self, ids: torch.Tensor, sg: int, nb: int, full_hull: bool = False) -> None:
    """phi of frames f = g-nb .. g-1 (global steps sg-nb .. sg-1) for ids. Hull distances: frames g-nb-1 .. g when
    full_hull (first frames of an episode), else only frame g (the earlier ones were stored by the previous updates)."""
    hs = [(sg - nb - 1 + k) % self.RI for k in range(nb + 2)]
    self._hull(ids, hs if full_hull else hs[-1:])
    fs = [(sg - nb + k) for k in range(nb)]
    E = slice(None) if ids is None else ids
    if nb == 1:
      f = fs[0]
      Im, I0, Ip = (self.I[(f + o) % self.RI][E][None] for o in (-1, 0, 1))
      Sm, S0, Sp = (self.S[(f + o) % self.RI][E][None] for o in (-1, 0, 1))
    else:
      Im = self.I[[(f - 1) % self.RI for f in fs]][:, E]
      I0 = self.I[[f % self.RI for f in fs]][:, E]
      Ip = self.I[[(f + 1) % self.RI for f in fs]][:, E]
      Sm = self.S[[(f - 1) % self.RI for f in fs]][:, E]
      S0 = self.S[[f % self.RI for f in fs]][:, E]
      Sp = self.S[[(f + 1) % self.RI for f in fs]][:, E]
    z0 = self.z0r[E][None]
    w0, v0 = torch.sqrt(G / z0), torch.sqrt(G * z0)
    ell, lam = self.ell[E][None], self.lam[E][None]
    K, n = I0.shape[:2]
    ph = torch.empty(K, n, NC, dtype=self.dt_, device=self.dev)
    ph[..., 0] = S0[..., 0]
    ph[..., 1] = (Sp[..., 0] - Sm[..., 0]) / (2 * DT) / w0
    ph[..., 2:6] = self._balance_raw(I0, w0, ell) - self.q24[E][None]
    ph[..., 6] = S0[..., 1]
    lv = I0[..., I_LV:I_LV + 2]
    fwd = torch.stack([lv[..., 1], -lv[..., 0]], -1)
    rot2 = lambda v: torch.stack([(v[..., :2] * fwd).sum(-1), (v[..., :2] * lv).sum(-1)], -1)   # noqa: E731
    vc = I0[..., I_VC:I_VC + 3]
    ph[..., 7:9] = rot2(vc) / v0[..., None]
    ph[..., 9] = vc[..., 2] / v0
    ph[..., 10] = 1.0 + I0[..., I_ACZ] / G
    ph[..., 11] = I0[..., I_JZ] / (G * w0)
    Lm = I0[..., I_L:I_L + 3]
    ph[..., 12:14] = rot2(Lm) / (z0 * v0)[..., None]
    ph[..., 14] = Lm[..., 2] / (z0 * v0)

    def sup(Ix):
      ok = (Ix[..., I_SZ:I_SZ + 28] < H_C * z0[..., None]) & (Ix[..., I_SV:I_SV + 28] < V_C * v0[..., None])
      return torch.stack([ok[..., :14].any(-1), ok[..., 14:].any(-1)], -1).to(self.dt_)

    sm, s0, sp = sup(Im), sup(I0), sup(Ip)
    ph[..., 15:17] = ((sm + s0 + sp) >= 2).to(self.dt_)               # 3-frame median of 0/1 = majority
    hand = lambda Ix: (Ix[..., I_WZ] < HAND_C * z0).to(self.dt_)       # noqa: E731
    ph[..., 17] = ((hand(Im) + hand(I0) + hand(Ip)) >= 2).to(self.dt_)
    ep = (I0[..., I_EPOS:I_EPOS + 15] - self.qep[E][None]) / lam[..., None]
    ev = I0[..., I_EVEL:I_EVEL + 15] / v0[..., None]
    ph[..., 18:48] = torch.cat([ep.reshape(K, n, 5, 3), ev.reshape(K, n, 5, 3)], -1).reshape(K, n, 30)
    if ids is None:
      self.PH[fs[0] % self.RP] = ph[0].float()
    else:
      for k, f in enumerate(fs):
        self.PH[f % self.RP, ids] = ph[k].float()

  def _hull(self, ids, slots) -> None:
    E = slice(None) if ids is None else ids
    I = self.I[slots][:, E] if len(slots) > 1 else self.I[slots[0]][E][None]   # (K, n, NI)
    K, n = I.shape[:2]
    w0 = torch.sqrt(G / self.z0r[E])[None, :, None]
    hd = self.hull_dtype or self.dt_
    P = I[..., I_P:I_P + 56].reshape(K * n, 28, 2).to(hd)
    xi = (I[..., I_COM:I_COM + 2] + I[..., I_VC:I_VC + 2] / w0).reshape(K * n, 1, 2)
    cm = I[..., I_COM:I_COM + 2].reshape(K * n, 1, 2)
    sd = hull_sd(P, torch.cat([xi, cm], 1).to(hd), self.hull_chunk).to(self.dt_).reshape(K, n, 2)
    ell = self.ell[E][None, :, None]
    val = (sd - self.rad) / ell
    for k, sl in enumerate(slots):
      if ids is None:
        self.S[sl] = val[k]
      else:
        self.S[sl, ids] = val[k]

  # ---------------------------------------------------------------------------------------------------------- output
  def _emit(self, f: torch.Tensor, sf: int) -> dict:
    """window ending at frame f (global step sf) for every environment; D* score where valid."""
    N = self.N
    w0 = torch.sqrt(G / self.z0r)                                       # NaN where no reference
    step = DTAU / w0 * FS                                               # frames per tau step
    k = torch.arange(-(WIN - 1), 1, device=self.dev, dtype=self.dt_)
    fi = k[None, :] * step[:, None]                                     # offsets (<= 0) from f, (N, 9)
    ok = self.ref_ok & (f >= self.first_phi)
    fin = torch.where(ok[:, None], fi, torch.zeros_like(fi))
    i1 = torch.ceil(fin)                                                # i0 = ceil - 1, a in (0, 1]: never reads frame f + 1
    a = (fin - (i1 - 1)).float()
    i0 = (i1 - 1).long()
    near = torch.round(fin).long()
    ok = ok & ((f[:, None] + i0) >= self.first_phi[:, None]).all(1)
    rows = torch.arange(N, device=self.dev)[:, None]
    g0 = self.PH[(sf + i0) % self.RP, rows]                            # (N, 9, 48)
    g1 = self.PH[(sf + i0 + 1) % self.RP, rows]
    W = (1 - a)[..., None] * g0 + a[..., None] * g1
    W[..., list(FLAGS)] = self.PH[(sf + near) % self.RP, rows][..., list(FLAGS)]
    W = torch.where(ok[:, None, None], W, torch.full_like(W, math.nan))
    out = dict(frame=f, valid=ok, window=W, phi=torch.where(ok[:, None], self.PH[sf % self.RP], torch.full_like(self.PH[0], math.nan)))
    if self.scorer is not None:
      rv, dv, _ = self.scorer(W, mirror_mean=self.mirror_mean)          # every environment (no host sync), NaN if invalid
      r = torch.where(ok, rv, torch.full_like(rv, math.nan))
      d = torch.where(ok[None], dv, torch.full_like(dv, math.nan)) if dv.dim() == 2 else dv
      out.update(r=r, d=d)
    self.out = out
    return out


# ------------------------------------------------------------------------------------------------------------ env reader
class PhiEnvReader:
  """The PhiStream inputs from an mjlab ManagerBasedRlEnv with the R1 entity 'robot' (see the module docstring)."""

  def __init__(self, env, entity: str = "robot"):
    import mujoco
    from r1wb.robot import R1_XML
    self.env, self.ent = env, env.scene[entity]
    names = list(self.ent.body_names)
    bid = lambda n: names.index(n)                                      # noqa: E731
    self.b_pel, self.b_hl, self.b_hr = bid("pelvis"), bid("left_hip_pitch_link"), bid("right_hip_pitch_link")
    self.b_kl, self.b_kr = bid("left_knee_link"), bid("right_knee_link")
    self.b_al, self.b_ar = bid("left_ankle_pitch_link"), bid("right_ankle_pitch_link")
    self.b_wl, self.b_wr, self.b_torso = bid("left_wrist_roll_link"), bid("right_wrist_roll_link"), bid("torso_link")
    stock = mujoco.MjModel.from_xml_path(str(R1_XML))
    seg_names, seg_mass, seg_ipos = [], [], []
    for b in range(1, stock.nbody):
      if stock.body_mass[b] > 0:
        seg_names.append(mujoco.mj_id2name(stock, mujoco.mjtObj.mjOBJ_BODY, b))
        seg_mass.append(float(stock.body_mass[b]))
        seg_ipos.append(stock.body_ipos[b].copy())
    self.seg_names, self.seg_mass = seg_names, np.array(seg_mass)
    dev = env.device
    self.seg_idx = torch.tensor([bid(n) for n in seg_names], device=dev)
    self.seg_ipos = torch.tensor(np.array(seg_ipos), dtype=torch.float64, device=dev)
    mj = env.sim.mj_model
    gids = []
    for n in FOOT_GEOMS:
      g = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, f"{entity}/{n}")
      if g < 0:
        g = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, n)
      if g < 0:
        raise ValueError(f"foot geom {n} not found")
      gids.append(g)
    self.foot = torch.tensor(gids, device=dev)
    self.foot_r = torch.tensor(mj.geom_size[gids, 0], dtype=torch.float64, device=dev)
    self.foot_h = torch.tensor(mj.geom_size[gids, 1], dtype=torch.float64, device=dev)
    self.foot_radius = float(np.median(mj.geom_size[gids, 0]))
    self.body_ids = self.ent.indexing.body_ids.long()
    self.head = torch.tensor(HEAD_IN_TORSO, dtype=torch.float64, device=dev)

  @staticmethod
  def _rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """rotate v (..., 3) by unit quaternions q (..., 4) wxyz."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)

  def mass(self) -> torch.Tensor:
    env = self.env
    expanded = getattr(env.sim, "expanded_fields", None) or getattr(env.sim, "_expanded_fields", set())
    if "body_mass" in expanded:
      return env.sim.model.body_mass[:, self.body_ids].to(torch.float64)
    return torch.as_tensor(env.sim.mj_model.body_mass[self.body_ids.cpu().numpy()], dtype=torch.float64, device=env.device)[None]

  @torch.no_grad()
  def read(self) -> dict:
    env, d = self.env, self.ent.data
    o = env.scene.env_origins.to(torch.float64)
    N = env.num_envs
    lp = d.body_link_pos_w.to(torch.float64) - o[:, None, :]
    lq = d.body_link_quat_w.to(torch.float64)
    cp = d.body_com_pos_w.to(torch.float64) - o[:, None, :]
    m = self.mass()
    com = (cp * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    gx = env.sim.data.geom_xpos[:, self.foot].to(torch.float64) - o[:, None, :]
    ax = env.sim.data.geom_xmat[:, self.foot].reshape(N, len(self.foot), 3, 3)[..., :, 2].to(torch.float64) * self.foot_h[None, :, None]
    tips = torch.stack([gx - ax, gx + ax], 2)                          # (N, 14 capsules, 2 ends, 3)
    tips = torch.cat([tips[..., :2], (tips[..., 2] - self.foot_r[None, :, None])[..., None]], -1)
    sole = tips.reshape(N, 2, 14, 3)                                    # left capsules first (7 x 2 ends), then right
    segs = lp[:, self.seg_idx] + self._rot(lq[:, self.seg_idx], self.seg_ipos[None].expand(N, -1, -1))
    head = lp[:, self.b_torso] + self._rot(lq[:, self.b_torso], self.head[None].expand(N, -1))
    return dict(com=com, sole=sole, pelvis=lp[:, self.b_pel], hip_l=lp[:, self.b_hl], hip_r=lp[:, self.b_hr],
                knee_l=lp[:, self.b_kl], knee_r=lp[:, self.b_kr], ankle_l=lp[:, self.b_al], ankle_r=lp[:, self.b_ar],
                wrist_l=lp[:, self.b_wl], wrist_r=lp[:, self.b_wr], head=head, segs=segs)


def build_for_env(env, lag: int = 14, reference=("window", 40, 75), dstar_paths=None, mirror_mean: bool = False,
                  hull_chunk: int = 4096):
  """(reader, stream) for an mjlab env; dstar_paths = the checkpoints of one fold (e.g. models_a2/F2_BD_s*.pt)."""
  rd = PhiEnvReader(env)
  sc = DStar(dstar_paths, device=env.device) if dstar_paths else None
  st = PhiStream(env.num_envs, len(rd.seg_names), rd.seg_mass, rd.foot_radius, lag=lag, reference=reference, scorer=sc,
                 device=env.device, mirror_mean=mirror_mean, hull_chunk=hull_chunk)
  return rd, st
