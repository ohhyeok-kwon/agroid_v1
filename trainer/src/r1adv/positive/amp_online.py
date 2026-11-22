"""Online human-motion discriminator on phi windows (AMP form, Peng et al. 2021) for the D slot of the Hold objective.
Positive session 10, 2026-11-18 (coordinator order 13:4x, resumed 19:3x KST). Building block of r1adv.positive.hold_w4amp
(tasks adv_hold_w4_xbs_amp*); offline gate history: rounds/R04/r6_a1_prior_gate.md 10-15 (why the FIXED D* failed:
passive collapse, falls and staged stagger extrapolated as human; quiet < reaction; no failure shapes in human data).

  windows  phi_dstar.PhiStream (in env) / the same tau-grid interpolation on the human corpus (take_windows): 9 samples,
           dtau = 0.25 (0.51 s for the R1, about 0.64 s for an adult), 48 channels; the discriminator reads KEEP channels
           (default NOPOS = the 27 position-free channels of D*-noPos; 'all' = 48) standardised by the human bank's
           mean / sd, NaN -> 0, plus two mask flags (hands, head missing or masked)
  real     concept_v1 split_v4 'train' side (bank file built by .agent/adversarial/positive/tools/r10_amp_bank.py, sha256
           pinned): sources = datasets, with moore_belt (Moore 2015 belt-perturbation trials) and stylized (100STYLE
           non-Neutral, Bandai non-normal, Xia non-neutral, Hosei Noh) as their own sources; staged clips (Eyes accident /
           fighting / karate / pain / joke, cinamon hit / stagger) and LAFAN1 fall / ground takes are not in the bank;
           windows touching run_or_jump / jump_in_place / low (sitting, crouching) / a double flight (supL = supR = 0) /
           hand-on-floor / leaning frames or the 0.2 s clip edges are not windows of the bank
  fake     the policy's own windows from the stream (a replay ring, push_per_step random envs a step), its windows of the
           last RECENT steps before every fall termination (own-fall ring, mixed at fall_share), and a fixed bank of
           recorded falls of OUR policies (fail_share; never the manufacturer's): falling (window end in [f - 0.5 s, f)),
           impact / settle ([f, f + 1 s)) and lying ([f + 1 s, end), only when position channels are used: without
           them a lying-still window is the same point as quiet standing, D*-noPos scored lying 0.81-0.98)
  loss     LSGAN (real +1, fake -1) + grad_penalty * E_real |grad_x d|^2, Adam; update_every env steps, n_updates
           minibatches of batch real + batch fake each (s2f's AMP cadence by default: 24 steps, 4 x 4096)
  reward   D = max(0, 1 - (d - 1)^2 / 4) (AMP standard), optionally capped: min(r, c) / c (ceiling); 0 where the stream
           has no valid window (episode start: about 0.9 s with the fixed reference). Used in place of the current D
           (HumanSurprise); r = L + beta D unchanged.
  still    two options, no default (panel decision): still_balance ('ratio', k, cap): the real batch's share of
           quiet-standing windows = min(cap, k x the fake batch's share) -- where robot and human quiet windows are
           indistinguishable the LSGAN optimum is d = (k - 1) / (k + 1) (k = real : fake still density ratio), so k = 1
           gives r = 0.75 there and r >= 0.95 needs k >= 3.5; ceiling ('real_still_q', q): c = running q-quantile of r
           over the real still windows of each update (or a fixed float). 'Quiet standing' is one phi-only rule for both
           sides (still_windows; thresholds from human quiet standing, stored in the bank file).
Random draws: every draw of this module uses its own torch.Generator (seed = env seed * 7919 + AMP_SEED) and the
discriminator is initialised inside torch.random.fork_rng, so the global stream (actor initialisation, sensor noise,
resets) is the one the environment would have without it.
Diagnostic mode (cue probe): fit_probe / probe_* below; driver .agent/adversarial/positive/tools/r10_amp_probe.py.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from r1adv.positive import phi_dstar as PD

WIN, NC = PD.WIN, PD.NC
POS = (0, 2, 3, 4, 5, 6) + tuple(18 + 6 * e + q for e in range(5) for q in range(3))
NOPOS = tuple(c for c in range(NC) if c not in POS)                       # 27 channels (D*-noPos)
HAND_CH = (17,) + tuple(range(30, 42))
HEAD_CH = tuple(range(42, 48))
AMP_SEED = 707
RECENT = 12                                                                # own-fall ring: last 12 emitted windows
# R1 reference for the fixed phi mode (rounds/R04/positive/r9_phi_env/reference_modes.json 'fixed_reference': medians of
# ten recorded rows' quiet windows); z0 / ell set the tau grid and the normalisation, lam / q24 / qep only position channels
R1_FIXED_REFERENCE = dict(
  z0=0.6457127253215328, ell=0.18299939448560787, lam=0.6569765190082103,
  q24=[-0.08770167354609829, 0.00016525833977443718, -0.08780537285951326, 0.00020496696297219618],
  qep=[0.06801876049409858, 0.11408645707554385, -0.7338743491010512, 0.06471661669352563, -0.11103751101484491,
       -0.7338750152895281, 0.08935336365866897, 0.18207592528722133, -0.05280490715788788, 0.08370837491963026,
       -0.18939850488950843, -0.056127411043797204, 0.0512595164100047, -0.0001868701524032699, 0.39853413809754307])
GROUPS = dict(balance=(1,), com_vel=(7, 8, 9), vert_force=(10, 11), ang_mom=(12, 13, 14), contacts=(15, 16, 17),
              foot_vel=(21, 22, 23, 27, 28, 29), hand_vel=(33, 34, 35, 39, 40, 41), head_vel=(45, 46, 47),
              positions=POS)


# ----------------------------------------------------------------------------------------------------------- windows
def take_windows(phi: torch.Tensor, end: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
  """phi (M, 48) 50 Hz frames, end (n,) frame indices, step (n,) frames per tau step -> (n, 9, 48) float32 windows ending
  at frame `end` (samples end + k step, k = -8..0, linear interpolation; contact flags from the nearest frame), the
  interpolation of PhiStream._emit. The caller guarantees that every index lies inside the window's own clip."""
  k = torch.arange(-(WIN - 1), 1, device=phi.device, dtype=torch.float32)
  fi = k[None, :] * step[:, None].float()
  i1 = torch.ceil(fi)
  a = (fi - (i1 - 1.0))[..., None]
  i0 = end[:, None] + (i1 - 1.0).long()
  W = (1.0 - a) * phi[i0].float() + a * phi[i0 + 1].float()
  near = end[:, None] + torch.round(fi).long()
  W[..., list(PD.FLAGS)] = phi[near][..., list(PD.FLAGS)].float()
  return W


def still_windows(W: torch.Tensor, thr: dict) -> torch.Tensor:
  """Quiet standing (one rule for human and robot windows, phi channels only): at every sample both feet supported and
  no hand contact, and CoM speed / v0, |F_z / mg - 1|, foot speeds / v0, head and wrist speeds relative to the CoM / v0
  below the thresholds (NaN channels -- no wrists / head in the data -- do not count)."""
  n0 = lambda x: torch.nan_to_num(x, nan=0.0)                              # noqa: E731
  sup = (W[..., 15] > 0.5) & (W[..., 16] > 0.5) & (n0(W[..., 17]) < 0.5)
  vc = W[..., 7:10].norm(dim=-1)
  fz = (W[..., 10] - 1.0).abs()
  feet = torch.maximum(W[..., 21:24].norm(dim=-1), W[..., 27:30].norm(dim=-1))
  head = n0(W[..., 45:48]).norm(dim=-1)
  hands = torch.maximum(n0(W[..., 33:36]).norm(dim=-1), n0(W[..., 39:42]).norm(dim=-1))
  ok = sup & (vc < thr["vc"]) & (fz < thr["fz"]) & (feet < thr["feet"]) & (head < thr["head"]) & (hands < thr["hands"])
  return ok.all(dim=1)


def amp_reward(d: torch.Tensor, ceiling: float | None = None) -> torch.Tensor:
  """AMP standard r = max(0, 1 - (d - 1)^2 / 4); with a ceiling c in (0, 1]: min(r, c) / c."""
  r = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)
  if ceiling is None:
    return r
  c = float(min(max(ceiling, 1e-3), 1.0))
  return torch.clamp(r, max=c) / c


def pinned_path(file: str, sha256: str, dirs) -> Path:
  """First copy of `file` in `dirs` whose sha256 equals `sha256` (an empty sha256 is refused)."""
  if not sha256:
    raise ValueError(f"{file}: no pinned sha256")
  for d in dirs:
    p = Path(d) / file
    if p.is_file():
      h = hashlib.sha256()
      with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
          h.update(chunk)
      if h.hexdigest() != sha256:
        raise ValueError(f"{p}: sha256 {h.hexdigest()} != pinned {sha256}")
      return p
  raise FileNotFoundError(f"{file} not found in {list(map(str, dirs))}")


# ----------------------------------------------------------------------------------------------------------- context (v2)
# Stimulus context c (panel design, R7 07B 2 / 5.2): ONLY what the environment set -- the most recent stimulus of the
# episode whose onset is at or before the window end (none before the first): |azimuth| (push direction relative to the
# heading, 0 = pushed from behind; folded left / right, so the mirror augmentation leaves c unchanged), J* = J / (m
# sqrt(g z0)) and the push length T. No elapsed time, no phase, no policy state.
AZ_EDGES = (22.5, 67.5, 112.5, 157.5)                                       # 5 |azimuth| bins (deg)
J_EDGES = (0.12, 0.20, 0.27)                                                # 4 J* bins
T_EDGES = (0.05, 0.25)                                                      # 3 push-length bins (s): impulse / short / long
N_AZ, N_J, N_T = len(AZ_EDGES) + 1, len(J_EDGES) + 1, len(T_EDGES) + 1
N_CTX = 1 + N_AZ * N_J * N_T                                                # bin 0 = no stimulus yet
CTX_DIM = 5
R1_M_SQRT_GZ = 75.0                                                         # N s: 29.8 kg x sqrt(9.81 x 0.646 m)


def ctx_bin(has, az_abs_deg, jstar, T):
  """-> long tensor of context bins (0 = none)."""
  ia = torch.bucketize(az_abs_deg.float(), torch.tensor(AZ_EDGES, device=az_abs_deg.device))
  ij = torch.bucketize(jstar.float(), torch.tensor(J_EDGES, device=jstar.device))
  it = torch.bucketize(T.float(), torch.tensor(T_EDGES, device=T.device))
  b = 1 + (ia * N_J + ij) * N_T + it
  return torch.where(has, b, torch.zeros_like(b))


def ctx_features(has, az_abs_deg, jstar, T):
  """-> (n, 5) [has, cos|az|, sin|az|, J*, log(T / 0.02) / log 40], zeros where there is no stimulus."""
  a = torch.deg2rad(az_abs_deg.float())
  f = torch.stack([torch.ones_like(a), torch.cos(a), torch.sin(a), jstar.float(),
                   torch.log(T.float().clamp_min(0.02) / 0.02) / math.log(40.0)], -1)
  return f * has.float()[:, None]


def neighbor_counts(counts: torch.Tensor) -> torch.Tensor:
  """counts per bin (N_CTX,) -> n_c: own bin + neighbours (|azimuth| +-1, J* +-1, same T); bin 0 keeps its own count."""
  c = counts.double()
  S = c[1:].reshape(N_AZ, N_J, N_T)
  P = torch.nn.functional.pad(S.permute(2, 0, 1)[None], (1, 1, 1, 1))[0]   # (T, AZ + 2, J + 2)
  out = sum(P[:, 1 + da:1 + da + N_AZ, 1 + dj:1 + dj + N_J] for da in (-1, 0, 1) for dj in (-1, 0, 1))
  return torch.cat([c[:1], out.permute(1, 2, 0).reshape(-1)])


def panel_reward(logit: torch.Tensor, nc: torch.Tensor, n0: float = 50.0) -> torch.Tensor:
  """r_D = exp(min(0, l n_c / (n_c + n0))): 1 when indistinguishable or without human evidence; no bonus above 1."""
  return torch.exp(torch.clamp(logit * (nc / (nc + float(n0))), max=0.0))


# ----------------------------------------------------------------------------------------------------------- data
class RealBank:
  """Human windows (bank file of r10_amp_bank.py). Sampling: ends drawn by source share (uniform within a source),
  optionally split into a quiet-standing part of a given share; returns raw windows + mask flags."""

  def __init__(self, path, device, generator: torch.Generator, source_weights: dict | str = "sqrt_hours",
               exclude: tuple = ("stylized",), clip_split: tuple | None = None):
    """clip_split = (mod, rem, 'train' | 'test'): diagnostic hold-out by clip (sha256 of the clip id mod `mod`), so that
    overlapping windows of one clip never sit on both sides (probe only; training uses every end)."""
    z = np.load(path, allow_pickle=False)
    dev = torch.device(device)
    self.dev, self.g = dev, generator
    self.meta = json.loads(str(z["meta"]))
    self.sources = list(self.meta["sources"])
    self.thr = dict(self.meta["still_thr"])
    self.phi = torch.from_numpy(z["phi"]).to(dev)                          # (M, 48) float16
    self.mu = torch.from_numpy(z["mu"].astype(np.float32)).to(dev)
    self.sd = torch.from_numpy(z["sd"].astype(np.float32)).to(dev)
    end = z["end"].astype(np.int64)
    src = z["end_src"].astype(np.int64)
    hours = {s: float(self.meta["source_end_hours"].get(s, 0.0)) for s in self.sources}
    if isinstance(source_weights, str):
      base = {s: (1.0 if source_weights == "equal" else math.sqrt(h) if source_weights == "sqrt_hours" else h)
              for s, h in hours.items()}
    else:
      base = {s: float(source_weights.get(s, 0.0)) for s in self.sources}
    for s in exclude:
      base[s] = 0.0
    tot = sum(v for s, v in base.items() if hours[s] > 0)
    if tot <= 0:
      raise ValueError("every real source has zero weight")
    self.share = {s: (base[s] / tot if hours[s] > 0 else 0.0) for s in self.sources}
    cnt = np.bincount(src, minlength=len(self.sources)).astype(np.float64)
    w = np.array([self.share[s] for s in self.sources])[src] / np.maximum(cnt[src], 1.0)
    keep = w > 0
    if clip_split is not None:
      mod, rem, side = clip_split
      cs = z["clip_start"]
      cid = np.searchsorted(cs, end, side="right") - 1
      hold = np.array([int(hashlib.sha256(str(c).encode()).hexdigest()[:8], 16) % int(mod) == int(rem) for c in z["clip_id"]])
      keep = keep & (hold[cid] if side == "test" else ~hold[cid])
    self.end = torch.from_numpy(end[keep]).to(dev)
    self.src = torch.from_numpy(src[keep]).to(dev)
    self.step = torch.from_numpy(z["end_step"][keep].astype(np.float32)).to(dev)
    self.still = torch.from_numpy(z["end_still"][keep].astype(bool)).to(dev)
    self.hand_miss = torch.from_numpy(z["end_hand_missing"][keep].astype(bool)).to(dev)
    self.head_miss = torch.from_numpy(z["end_head_missing"][keep].astype(bool)).to(dev)
    wt = torch.from_numpy(w[keep]).to(dev, torch.float64)
    self.cdf_all = self._cdf(wt)
    self.cdf_still = self._cdf(torch.where(self.still, wt, torch.zeros_like(wt)))
    self.cdf_move = self._cdf(torch.where(self.still, torch.zeros_like(wt), wt))
    self.natural_still_share = float((wt * self.still).sum() / wt.sum())
    self.n_ends = int(keep.sum())
    self.has_ctx = "end_ctx_bin" in z.files
    if self.has_ctx:                                                         # v2 bank: stimulus context per window end
      self.cbin = torch.from_numpy(z["end_ctx_bin"][keep].astype(np.int64)).to(dev)
      self.cfeat = torch.from_numpy(z["end_ctx_feat"][keep].astype(np.float32)).to(dev)
      self.counts = torch.bincount(self.cbin, minlength=N_CTX)
      self.nc = neighbor_counts(self.counts).float()
      # matched-cell sampling: cell = 2 bin + quiet flag; G[j] = cell + within-cell cumulative weight in (0, 1]
      cell = self.cbin * 2 + self.still.long()
      order = torch.argsort(cell, stable=True)
      cs, ws = cell[order], wt[order]
      self.order = order
      ccount = torch.bincount(cs, minlength=2 * N_CTX)
      start = torch.cumsum(ccount, 0) - ccount
      cum = torch.cumsum(ws, 0)
      base = torch.where(start > 0, cum[(start - 1).clamp_min(0)], torch.zeros_like(cum[:1]).expand_as(start.double()))
      tot = torch.where(ccount > 0, cum[(start + ccount - 1).clamp_min(0)] - base, torch.ones_like(base))
      self.G = cs.double() + (cum - base[cs]) / tot[cs]
      self.cell_count = ccount

  def sample_cells(self, cells: torch.Tensor):
    """one real end per requested cell (cell = 2 bin + quiet); -> (windows, hand-missing, head-missing, quiet, ctx
    features, supported). Unsupported cells (no human window) return a dummy draw with supported = False."""
    sup = self.cell_count[cells] > 0
    u = torch.rand(len(cells), device=self.dev, generator=self.g, dtype=torch.float64) * (1.0 - 1e-9)
    j = torch.searchsorted(self.G, cells.double() + u, right=True).clamp(max=len(self.G) - 1)
    i = self.order[j]
    W = take_windows(self.phi, self.end[i], self.step[i])
    return W, self.hand_miss[i], self.head_miss[i], self.still[i], self.cfeat[i], sup

  def sample_quiet_none(self, n: int):
    """human quiet-standing windows without a stimulus context (the human half of the invariance set I)."""
    cells = torch.full((n,), 1, dtype=torch.long, device=self.dev)           # bin 0, quiet
    return self.sample_cells(cells)[0]

  @staticmethod
  def _cdf(w):
    c = torch.cumsum(w, 0)
    return c / c[-1].clamp_min(1e-300)

  def _draw(self, cdf, n):
    u = torch.rand(n, device=self.dev, generator=self.g, dtype=torch.float64)
    return torch.searchsorted(cdf, u, right=True).clamp(max=len(cdf) - 1)

  def sample(self, n: int, still_share: float | None = None):
    """-> (raw windows (n, 9, 48), hand-missing (n,), head-missing (n,), quiet-standing (n,))."""
    if still_share is None:
      i = self._draw(self.cdf_all, n)
    else:
      ns = int(round(min(max(still_share, 0.0), 1.0) * n))
      i = torch.cat([self._draw(self.cdf_still, ns), self._draw(self.cdf_move, n - ns)])
    W = take_windows(self.phi, self.end[i], self.step[i])
    return W, self.hand_miss[i], self.head_miss[i], self.still[i]


class FailBank:
  """Fixed fake windows from recorded falls of our own policies (r10_amp_failbank.py): kind 0 falling, 1 impact /
  settle, 2 lying. Uniform over the allowed windows."""

  def __init__(self, path, device, generator, lying: bool):
    z = np.load(path, allow_pickle=False)
    kind = z["kind"].astype(np.int64)
    keep = kind < (3 if lying else 2)
    self.W = torch.from_numpy(z["windows"][keep]).to(device)              # float16
    self.meta = json.loads(str(z["meta"]))
    self.kinds = {int(k): int((kind[keep] == k).sum()) for k in np.unique(kind[keep])}
    self.g, self.dev = generator, torch.device(device)
    if any(p == "manufacturer" for p in self.meta.get("policies", [])):
      raise ValueError("the failure bank holds manufacturer windows")

  def sample(self, n):
    i = torch.randint(0, len(self.W), (n,), device=self.dev, generator=self.g)
    return self.W[i].float()


class WindowRing:
  """Ring of raw windows (float16) with a validity flag; sampling is uniform over the valid entries."""

  def __init__(self, capacity: int, device):
    self.W = torch.zeros(int(capacity), WIN, NC, dtype=torch.float16, device=device)
    self.ok = torch.zeros(int(capacity), dtype=torch.bool, device=device)
    self.i, self.full = 0, False

  def push(self, W, ok):
    n = W.shape[0]
    if n == 0:
      return
    cap = self.W.shape[0]
    idx = (torch.arange(n, device=W.device) + self.i) % cap
    self.W[idx] = torch.nan_to_num(W, nan=0.0).half()
    self.ok[idx] = ok
    self.full = self.full or self.i + n >= cap
    self.i = int((self.i + n) % cap)

  def count(self) -> int:
    return int(self.ok.sum())                                               # host sync: callers use it once per update

  def sample(self, n, g):
    i = torch.multinomial(self.ok.float(), n, replacement=True, generator=g)
    return self.W[i].float()


# ----------------------------------------------------------------------------------------------------------- model
class Disc(torch.nn.Module):
  def __init__(self, d_in: int, hidden=(512, 256)):
    super().__init__()
    layers, i = [], d_in
    for h in hidden:
      layers += [torch.nn.Linear(i, h), torch.nn.ReLU()]
      i = h
    layers.append(torch.nn.Linear(i, 1))
    self.net = torch.nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x).squeeze(-1)


def prep(W, mu, sd, keep, hand_mask=None, head_mask=None, ctx=None):
  """raw windows (n, 9, 48) -> (n, 9 len(keep) + 2 [+ 5]): standardised, NaN -> 0, masked groups -> 0, + two mask flags
  [+ the stimulus-context features (v2)]."""
  n = W.shape[0]
  Z = torch.nan_to_num((W - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
  hm = torch.zeros(n, dtype=torch.bool, device=W.device) if hand_mask is None else hand_mask
  dm = torch.zeros(n, dtype=torch.bool, device=W.device) if head_mask is None else head_mask
  if hand_mask is not None:
    Z[:, :, list(HAND_CH)] = Z[:, :, list(HAND_CH)] * (~hm)[:, None, None]
  if head_mask is not None:
    Z[:, :, list(HEAD_CH)] = Z[:, :, list(HEAD_CH)] * (~dm)[:, None, None]
  Z = Z[:, :, list(keep)]
  parts = [Z.reshape(n, -1), hm.float()[:, None], dm.float()[:, None]]
  if ctx is not None:
    parts.append(ctx.float())
  return torch.cat(parts, 1)


class OnlineDisc:
  """Discriminator + optimiser + the augmentation of one training minibatch (mirror, mask flags)."""

  def __init__(self, keep, mu, sd, device, seed: int, hidden=(512, 256), lr=1e-4, weight_decay=1e-4, grad_penalty=5.0,
               grad_clip=1.0, mirror_p=0.5, mask_p=0.3, ctx_dim: int = 0):
    self.keep = tuple(int(c) for c in keep)
    self.mu, self.sd = mu, sd
    self.ctx_dim = int(ctx_dim)
    self.d_in = WIN * len(self.keep) + 2 + self.ctx_dim
    self.lam = 0.0                                                          # v2: invariance multiplier (dual variable)
    with torch.random.fork_rng(devices=[]):                                 # the global RNG stream is left untouched
      torch.manual_seed(int(seed))
      self.net = Disc(self.d_in, tuple(hidden))
    self.net.to(device)
    self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    self.gp, self.clip, self.mirror_p, self.mask_p = float(grad_penalty), float(grad_clip), float(mirror_p), float(mask_p)

  def score(self, W):
    """raw windows -> d (no masks: a robot window has hands and head)."""
    with torch.no_grad():
      return self.net(prep(W, self.mu, self.sd, self.keep))

  def _aug(self, W, g, miss_hand=None, miss_head=None):
    n = W.shape[0]
    m = torch.rand(n, device=W.device, generator=g) < self.mirror_p
    W = torch.where(m[:, None, None], PD.mirror_window(W), W)
    hm = torch.rand(n, device=W.device, generator=g) < self.mask_p
    dm = torch.rand(n, device=W.device, generator=g) < self.mask_p
    if miss_hand is not None:
      hm = hm | miss_hand
    if miss_head is not None:
      dm = dm | miss_head
    return prep(W, self.mu, self.sd, self.keep, hm, dm)

  def train_step(self, Wr, miss_hand, miss_head, Wf, g):
    """one LSGAN + gradient-penalty step; returns (d_real, d_fake) of the minibatch before the step (detached)."""
    xr = self._aug(Wr, g, miss_hand, miss_head).requires_grad_(True)
    xf = self._aug(Wf, g)
    dr, df = self.net(xr), self.net(xf)
    loss_d = ((dr - 1.0) ** 2).mean() + ((df + 1.0) ** 2).mean()
    grad = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0]
    gp = grad.pow(2).sum(1).mean()
    loss = loss_d + self.gp * gp
    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    if self.clip > 0:
      torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
    self.opt.step()
    return dr.detach(), df.detach(), loss_d.detach(), gp.detach()

  # ------------------------------------------------------------------------------------------------- v2 (panel design)
  def logit(self, W, ctx):
    """raw windows + context features -> logit l ~ log(p_human / p_policy) (no masks: a robot window has hands, head)."""
    with torch.no_grad():
      return self.net(prep(W, self.mu, self.sd, self.keep, ctx=ctx))

  def _aug_ctx(self, W, ctx, g, miss_hand=None, miss_head=None):
    n = W.shape[0]
    m = torch.rand(n, device=W.device, generator=g) < self.mirror_p
    W = torch.where(m[:, None, None], PD.mirror_window(W), W)               # |azimuth| context: mirror-invariant
    hm = torch.rand(n, device=W.device, generator=g) < self.mask_p
    dm = torch.rand(n, device=W.device, generator=g) < self.mask_p
    if miss_hand is not None:
      hm = hm | miss_hand
    if miss_head is not None:
      dm = dm | miss_head
    return prep(W, self.mu, self.sd, self.keep, hm, dm, ctx)

  def train_step_bce(self, Wr, mh, md, cr, Wf, cf, wpair, g, WIh=None, WIr=None, acc_target=0.55, lam_lr=1.0,
                     lam_max=100.0):
    """one BCE (human 1 / policy 0) + gradient-penalty step on matched pairs (wpair: 1 where the pair's context cell has
    human support, else 0 -- no evidence, no opinion), plus the quiet-invariance term lam E_I[l^2] on I = human quiet
    standing (WIh) and robot PD-fixed standing (WIr); lam by dual ascent on acc_I - acc_target. Returns detached
    (l_real, l_fake, loss_bce, gp, acc_I or nan, lam)."""
    F = torch.nn.functional
    xr = self._aug_ctx(Wr, cr, g, mh, md).requires_grad_(True)
    xf = self._aug_ctx(Wf, cf, g)
    lr_, lf_ = self.net(xr), self.net(xf)
    w = wpair.float()
    nw = w.sum().clamp_min(1.0)
    loss_b = ((F.softplus(-lr_) + F.softplus(lf_)) * w).sum() / nw
    grad = torch.autograd.grad((lr_ * w).sum(), xr, create_graph=True)[0]
    gp = (grad.pow(2).sum(1) * w).sum() / nw
    loss = loss_b + self.gp * gp
    acc_i = torch.tensor(float("nan"), device=Wr.device)
    if WIh is not None and WIr is not None:
      z0 = torch.zeros(len(WIh), CTX_DIM, device=Wr.device)
      lh = self.net(self._aug_ctx(WIh, z0, g))
      lr2 = self.net(self._aug_ctx(WIr, torch.zeros(len(WIr), CTX_DIM, device=Wr.device), g))
      acc_i = 0.5 * ((lh > 0).float().mean() + (lr2 < 0).float().mean())
      loss = loss + self.lam * 0.5 * (lh.pow(2).mean() + lr2.pow(2).mean())
    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    if self.clip > 0:
      torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
    self.opt.step()
    if WIh is not None and WIr is not None:
      self.lam = float(min(max(self.lam + lam_lr * (float(acc_i) - acc_target), 0.0), lam_max))
    return lr_.detach(), lf_.detach(), loss_b.detach(), gp.detach(), acc_i.detach(), self.lam

  def state_dict(self):
    return dict(net=self.net.state_dict(), opt=self.opt.state_dict(), keep=list(self.keep), d_in=self.d_in,
                ctx_dim=self.ctx_dim, lam=self.lam)

  def load_state_dict(self, sd):
    if list(sd["keep"]) != list(self.keep):
      raise ValueError("discriminator channel subset differs")
    self.net.load_state_dict(sd["net"])
    self.opt.load_state_dict(sd["opt"])


# ----------------------------------------------------------------------------------------------------------- engine
DEFAULTS = dict(
  channels="nopos", reference="fixed", lag=14,
  real_file="", real_sha256="", source_weights="sqrt_hours", exclude_sources=["stylized"],
  fail_file="", fail_sha256="", fail_share=0.10, fail_lying="auto", fall_share=0.05, fall_buffer=65536,
  buffer=262144, push_per_step=1024, update_every=24, n_updates=4, batch=4096,
  hidden=[512, 256], lr=1.0e-4, weight_decay=1.0e-4, grad_penalty=5.0, grad_clip=1.0, mirror_p=0.5, mask_p=0.3,
  still_balance=None, ceiling=None, ema=0.05, save_path="", save_every_updates=50)


def resolve(params: dict) -> dict:
  p = dict(DEFAULTS)
  unknown = set(params) - set(DEFAULTS)
  if unknown:
    raise ValueError(f"unknown AMP parameters {sorted(unknown)}")
  p.update(params)
  if p["channels"] not in ("nopos", "all"):
    raise ValueError("channels must be 'nopos' or 'all'")
  if p["reference"] not in ("fixed", "window"):
    raise ValueError("reference must be 'fixed' or 'window'")
  sb, ce = p["still_balance"], p["ceiling"]
  if sb is not None and not (isinstance(sb, (list, tuple)) and sb[0] in ("ratio", "fixed")):
    raise ValueError("still_balance must be None, ('ratio', k, cap) or ('fixed', share)")
  if ce is not None and not (isinstance(ce, (int, float)) or (isinstance(ce, (list, tuple)) and ce[0] == "real_still_q")):
    raise ValueError("ceiling must be None, a float or ('real_still_q', q)")
  return p


class OnlineAMP:
  """The D_online engine of one environment: phi stream, real / fake data, discriminator, reward map, updates."""

  def __init__(self, env, params: dict, input_dirs):
    p = resolve(params)
    self.p, self.env = p, env
    dev = env.device
    self.N, self.dev = env.num_envs, torch.device(dev)
    seed = int(env.cfg.seed) * 7919 + AMP_SEED
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(seed)
    self.keep = NOPOS if p["channels"] == "nopos" else tuple(range(NC))
    ref = ("fixed", R1_FIXED_REFERENCE) if p["reference"] == "fixed" else ("window", 40, 75)
    self.reader = PD.PhiEnvReader(env)
    self.stream = PD.PhiStream(self.N, len(self.reader.seg_names), self.reader.seg_mass, self.reader.foot_radius,
                               lag=int(p["lag"]), reference=ref, scorer=None, device=dev)
    self.real = RealBank(pinned_path(p["real_file"], p["real_sha256"], input_dirs), dev, self.g, p["source_weights"],
                         tuple(p["exclude_sources"]))
    lying = (p["channels"] == "all") if p["fail_lying"] == "auto" else bool(p["fail_lying"])
    self.fail = (FailBank(pinned_path(p["fail_file"], p["fail_sha256"], input_dirs), dev, self.g, lying)
                 if float(p["fail_share"]) > 0 else None)
    self.disc = OnlineDisc(self.keep, self.real.mu, self.real.sd, dev, seed, hidden=p["hidden"], lr=p["lr"],
                           weight_decay=p["weight_decay"], grad_penalty=p["grad_penalty"], grad_clip=p["grad_clip"],
                           mirror_p=p["mirror_p"], mask_p=p["mask_p"])
    self.buf = WindowRing(p["buffer"], dev)
    self.fallbuf = WindowRing(p["fall_buffer"], dev)
    self.recent = torch.zeros(RECENT, self.N, WIN, NC, dtype=torch.float16, device=dev)
    self.recent_ok = torch.zeros(RECENT, self.N, dtype=torch.bool, device=dev)
    self.push_n = int(min(self.N, p["push_per_step"]))
    self.steps, self.updates, self.skipped = 0, 0, 0
    self.fake_still = None                                                  # EMA of the policy's quiet-standing share
    self.ceiling = float(p["ceiling"]) if isinstance(p["ceiling"], (int, float)) else None
    self.hits = None
    self.last_log = {}

  # ------------------------------------------------------------------------------------------------------- episodes
  def reset(self, env_ids=None):
    self.stream.reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.recent_ok[:, ids] = False

  def _hit_term(self, env):
    if self.hits is None:
      try:
        self.hits = env.event_manager.get_term_cfg("torso_hits").func
      except Exception:                                                     # noqa: BLE001  (logging only)
        self.hits = False
    return self.hits or None

  # ------------------------------------------------------------------------------------------------------- step
  @torch.no_grad()
  def step(self, env) -> torch.Tensor:
    """called once per control step from the reward term (before the auto-reset): D_online per environment."""
    out = self.stream.update(self.reader.read())
    W, ok = out["window"], out["valid"]
    d = self.disc.score(torch.nan_to_num(W, nan=0.0))
    r = amp_reward(d, self.ceiling)
    D = torch.where(ok, r, torch.zeros_like(r))
    idx = torch.randint(0, self.N, (self.push_n,), device=self.dev, generator=self.g)
    self.buf.push(W[idx], ok[idx])
    slot = self.steps % RECENT
    self.recent[slot] = torch.nan_to_num(W, nan=0.0).half()
    self.recent_ok[slot] = ok
    if float(self.p["fall_share"]) > 0:
      fell = env.reset_terminated
      if bool(fell.any()):                                                  # second host sync of the step (falls only)
        ids = fell.nonzero().flatten()
        self.fallbuf.push(self.recent[:, ids].reshape(-1, WIN, NC), self.recent_ok[:, ids].reshape(-1))
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
    ht = self._hit_term(env)
    if ht is not None:
      onset = torch.round(ht.last[:, 0] / env.step_dt).long()             # episode step of the firing; the push acts
      age = (env.episode_length_buf - int(self.p["lag"])) - (onset + 1)    # from the next step (episode clock: also
                                                                           # right with a randomised first-episode clock)
      react = ok & (ht.count > 0) & (age >= 0) & (age < int(round(6.0 / math.sqrt(PD.G / R1_FIXED_REFERENCE["z0"]) * 50)))
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

  # ------------------------------------------------------------------------------------------------------- update
  def _real_share(self):
    sb = self.p["still_balance"]
    if sb is None:
      return None
    if sb[0] == "fixed":
      return float(sb[1])
    if self.fake_still is None:
      return None
    return min(float(sb[2]), float(sb[1]) * self.fake_still)

  def update(self, log):
    n_ok = self.buf.count()
    if n_ok < 1:
      self.skipped += 1
      self.last_log = {"AMP/update_skipped": torch.tensor(float(self.skipped), device=self.dev)}
      return
    p, B = self.p, int(self.p["batch"])
    nf = int(round(float(p["fall_share"]) * B)) if self.fallbuf.count() > 0 else 0
    nb = int(round(float(p["fail_share"]) * B)) if self.fail is not None else 0
    nm = B - nf - nb
    agg = {}
    with torch.inference_mode(False), torch.enable_grad():
      for u in range(int(p["n_updates"])):
        share = self._real_share()
        Wr, mh, md, sr = self.real.sample(B, share)
        Wm = self.buf.sample(nm, self.g)
        parts = [Wm]
        if nf:
          parts.append(self.fallbuf.sample(nf, self.g))
        if nb:
          parts.append(self.fail.sample(nb))
        Wf = torch.cat(parts).clone()
        Wr = Wr.clone()
        sm = still_windows(Wm, self.real.thr)
        dr, df, ld, gp = self.disc.train_step(Wr, mh.clone(), md.clone(), Wf, self.g)
        fs = sm.float().mean()
        self.fake_still = float(fs) if self.fake_still is None else (1 - p["ema"]) * self.fake_still + p["ema"] * float(fs)
        if u == 0:                                                          # fresh minibatch, before this update's steps
          dm = df[:nm]
          agg = {"AMP/acc_real": (dr > 0).float().mean(), "AMP/acc_fake": (dm < 0).float().mean(),
                 "AMP/acc": 0.5 * ((dr > 0).float().mean() + (dm < 0).float().mean()),
                 "AMP/d_real": dr.mean(), "AMP/d_fake": dm.mean(),
                 "AMP/d_real_still": (dr * sr).sum() / sr.sum().clamp_min(1),
                 "AMP/d_real_move": (dr * ~sr).sum() / (~sr).sum().clamp_min(1),
                 "AMP/d_fake_still": (dm * sm).sum() / sm.sum().clamp_min(1),
                 "AMP/d_fake_move": (dm * ~sm).sum() / (~sm).sum().clamp_min(1),
                 "AMP/real_still_share": sr.float().mean(), "AMP/fake_still_share": fs,
                 "AMP/loss_lsgan": ld, "AMP/grad_penalty": gp}
          if nf:
            agg["AMP/d_fake_ownfall"] = df[nm:nm + nf].mean()
          if nb:
            agg["AMP/d_fake_failbank"] = df[nm + nf:].mean()
          ce = p["ceiling"]
          if isinstance(ce, (list, tuple)) and bool(sr.any()):
            rq = torch.quantile(amp_reward(dr[sr]), float(ce[1]))
            self.ceiling = float(rq) if self.ceiling is None else (1 - p["ema"]) * self.ceiling + p["ema"] * float(rq)
    self.updates += 1
    agg["AMP/updates"] = torch.tensor(float(self.updates), device=self.dev)
    agg["AMP/ceiling"] = torch.tensor(float(self.ceiling if self.ceiling is not None else 1.0), device=self.dev)
    agg["AMP/buffer_valid"] = torch.tensor(float(n_ok), device=self.dev)
    self.last_log = agg
    if p["save_path"] and self.updates % int(p["save_every_updates"]) == 0:
      self.save(p["save_path"])

  # ------------------------------------------------------------------------------------------------------- state
  def state_dict(self):
    return dict(disc=self.disc.state_dict(), updates=self.updates, steps=self.steps, fake_still=self.fake_still,
                ceiling=self.ceiling, params=self.p, generator=self.g.get_state())

  def save(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".part")
    torch.save(self.state_dict(), tmp)
    tmp.replace(path)

  def load(self, path):
    sd = torch.load(path, map_location=self.dev, weights_only=False)
    self.disc.load_state_dict(sd["disc"])
    self.updates, self.fake_still, self.ceiling = sd["updates"], sd["fake_still"], sd["ceiling"]


# ----------------------------------------------------------------------------------------------------------- engine v2
class HardBank:
  """Fixed hard-negative fake windows with their stimulus context (r10_amp_hardbank.py): category 0 falling / impact of
  our policies' falls, 1 passive collapse (policy off at the hit), 2 synthetic fidget, 3 human clips replayed 2-3x faster.
  Equal share per enabled category; `exclude` drops categories (the (b') refits)."""

  CATS = ("fall", "collapse", "fidget", "speed")

  def __init__(self, path, device, generator, exclude=()):
    z = np.load(path, allow_pickle=False)
    self.meta = json.loads(str(z["meta"]))
    if any("manufacturer" in str(p) for p in self.meta.get("policies", [])):
      raise ValueError("the hard-negative bank holds manufacturer windows")
    cat = z["category"].astype(np.int64)
    keep = ~np.isin(cat, [self.CATS.index(c) for c in exclude])
    self.W = torch.from_numpy(z["windows"][keep]).to(device)
    self.cbin = torch.from_numpy(z["ctx_bin"][keep].astype(np.int64)).to(device)
    self.cfeat = torch.from_numpy(z["ctx_feat"][keep].astype(np.float32)).to(device)
    self.cat = torch.from_numpy(cat[keep]).to(device)
    self.cats = sorted(int(c) for c in np.unique(cat[keep]))
    cnt = torch.bincount(self.cat, minlength=len(self.CATS)).double()
    w = (1.0 / cnt.clamp_min(1.0))[self.cat]
    self.cdf = torch.cumsum(w, 0) / w.sum()
    self.g, self.dev = generator, torch.device(device)

  def sample(self, n):
    u = torch.rand(n, device=self.dev, generator=self.g, dtype=torch.float64)
    i = torch.searchsorted(self.cdf, u, right=True).clamp(max=len(self.cdf) - 1)
    return self.W[i].float(), self.cbin[i], self.cfeat[i], self.cat[i]


class CtxRing(WindowRing):
  """WindowRing that also keeps each window's context bin and features."""

  def __init__(self, capacity: int, device):
    super().__init__(capacity, device)
    self.cbin = torch.zeros(int(capacity), dtype=torch.long, device=device)
    self.cfeat = torch.zeros(int(capacity), CTX_DIM, dtype=torch.float16, device=device)

  def push_ctx(self, W, ok, cbin, cfeat):
    n = W.shape[0]
    if n == 0:
      return
    idx = (torch.arange(n, device=W.device) + self.i) % self.W.shape[0]
    self.cbin[idx] = cbin
    self.cfeat[idx] = cfeat.half()
    self.push(W, ok)

  def sample_ctx(self, n, g):
    i = torch.multinomial(self.ok.float(), n, replacement=True, generator=g)
    return self.W[i].float(), self.cbin[i], self.cfeat[i].float()


DEFAULTS2 = dict(
  mode="panel", channels="nopos", reference="fixed", lag=14,
  real_file="", real_sha256="", source_weights="sqrt_hours", exclude_sources=["stylized"],
  hard_file="", hard_sha256="", hard_share=0.40, hard_exclude=[], irob_file="", irob_sha256="", fall_share=0.05,
  fall_buffer=65536, buffer=262144, push_per_step=1024, update_every=24, n_updates=4, batch=4096, inv_batch=1024,
  hidden=[512, 256], lr=1.0e-4, weight_decay=1.0e-4, grad_penalty=5.0, grad_clip=1.0, mirror_p=0.5, mask_p=0.3,
  n0=50.0, acc_target=0.55, lam_lr=1.0, lam_max=100.0, save_path="", save_every_updates=50)


def resolve2(params: dict) -> dict:
  p = dict(DEFAULTS2)
  unknown = set(params) - set(DEFAULTS2)
  if unknown:
    raise ValueError(f"unknown D_online parameters {sorted(unknown)}")
  p.update(params)
  if p["mode"] != "panel" or p["channels"] not in ("nopos", "all") or p["reference"] not in ("fixed", "window"):
    raise ValueError("bad D_online mode / channels / reference")
  return p


NO_STIM = 10 ** 9                                                           # onset of 'no stimulus yet' (a slot never used)


class OnlineD2:
  """D_online of the R7 panel design (07B 2 / 5): BCE logit l ~ log(p_human / p_policy) on (phi window, stimulus context),
  matched priors per (context, quiet) cell, quiet-invariance term on I, r_D = exp(min(0, l n_c / (n_c + n0)))."""

  def __init__(self, env, params: dict, input_dirs):
    p = resolve2(params)
    self.p, self.env = p, env
    dev = env.device
    self.N, self.dev = env.num_envs, torch.device(dev)
    seed = int(env.cfg.seed) * 7919 + AMP_SEED
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(seed)
    self.keep = NOPOS if p["channels"] == "nopos" else tuple(range(NC))
    ref = ("fixed", R1_FIXED_REFERENCE) if p["reference"] == "fixed" else ("window", 40, 75)
    self.reader = PD.PhiEnvReader(env)
    self.stream = PD.PhiStream(self.N, len(self.reader.seg_names), self.reader.seg_mass, self.reader.foot_radius,
                               lag=int(p["lag"]), reference=ref, scorer=None, device=dev)
    self.real = RealBank(pinned_path(p["real_file"], p["real_sha256"], input_dirs), dev, self.g, p["source_weights"],
                         tuple(p["exclude_sources"]))
    if not self.real.has_ctx:
      raise ValueError("the panel design needs a real bank with stimulus contexts (amp_real_v2)")
    self.hard = (HardBank(pinned_path(p["hard_file"], p["hard_sha256"], input_dirs), dev, self.g, tuple(p["hard_exclude"]))
                 if float(p["hard_share"]) > 0 else None)
    z = np.load(pinned_path(p["irob_file"], p["irob_sha256"], input_dirs), allow_pickle=False)
    self.pd = torch.from_numpy(z["windows"]).to(dev)                       # robot half of I: recorded quiet standing of
                                                                            # our policies (the PD-fixed R1 falls, t1 A1)
    self.disc = OnlineDisc(self.keep, self.real.mu, self.real.sd, dev, seed, hidden=p["hidden"], lr=p["lr"],
                           weight_decay=p["weight_decay"], grad_penalty=p["grad_penalty"], grad_clip=p["grad_clip"],
                           mirror_p=p["mirror_p"], mask_p=p["mask_p"], ctx_dim=CTX_DIM)
    self.buf = CtxRing(p["buffer"], dev)
    self.fallbuf = CtxRing(p["fall_buffer"], dev)
    self.recent = torch.zeros(RECENT, self.N, WIN, NC, dtype=torch.float16, device=dev)
    self.recent_ok = torch.zeros(RECENT, self.N, dtype=torch.bool, device=dev)
    self.recent_bin = torch.zeros(RECENT, self.N, dtype=torch.long, device=dev)
    self.recent_feat = torch.zeros(RECENT, self.N, CTX_DIM, dtype=torch.float16, device=dev)
    # stimulus history per env: two slots (previous, last): onset episode step, |azimuth| deg, J*, T
    self.h_on = torch.full((self.N, 2), NO_STIM, dtype=torch.long, device=dev)   # NO_STIM: never <= a window end
    self.h_az = torch.zeros(self.N, 2, device=dev)
    self.h_j = torch.zeros(self.N, 2, device=dev)
    self.h_t = torch.full((self.N, 2), 0.02, device=dev)
    self.h_count = torch.zeros(self.N, dtype=torch.long, device=dev)
    self.push_n = int(min(self.N, p["push_per_step"]))
    self.steps, self.updates, self.skipped, self.gate_allowed = 0, 0, 0, 0
    self.hits = None
    self.last_log = {}

  def reset(self, env_ids=None):
    self.stream.reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.recent_ok[:, ids] = False
    self.h_on[ids] = NO_STIM
    self.h_count[ids] = 0

  def _hit_term(self, env):
    if self.hits is None:
      try:
        self.hits = env.event_manager.get_term_cfg("torso_hits").func
      except Exception:                                                     # noqa: BLE001
        self.hits = False
    return self.hits or None

  def _context(self, env):
    """context of the window ending lag steps ago: the most recent stimulus with onset sample <= the window end."""
    ht = self._hit_term(env)
    if ht is not None:
      new = ht.count > self.h_count                                         # a hit fired in last step's event stage
      if bool(new.any()):                                                   # host sync only on steps with new hits
        ids = new.nonzero().flatten()
        self.h_on[ids, 0], self.h_az[ids, 0] = self.h_on[ids, 1], self.h_az[ids, 1]
        self.h_j[ids, 0], self.h_t[ids, 0] = self.h_j[ids, 1], self.h_t[ids, 1]
        self.h_on[ids, 1] = torch.round(ht.last[ids, 0] / env.step_dt).long()
        az = torch.rad2deg(torch.remainder(ht.last[ids, 2], 2 * math.pi))
        self.h_az[ids, 1] = torch.where(az > 180.0, 360.0 - az, az)
        self.h_j[ids, 1] = ht.last[ids, 1] / R1_M_SQRT_GZ
        self.h_t[ids, 1] = ht.push_T[ids] if hasattr(ht, "push_T") else torch.full_like(ht.last[ids, 1], 0.02)
      self.h_count = ht.count.clone()
    end_step = env.episode_length_buf - int(self.p["lag"])                  # episode step of the window-end sample
    use_last = (self.h_on[:, 1] + 1) <= end_step
    use_prev = ~use_last & ((self.h_on[:, 0] + 1) <= end_step)
    k = torch.where(use_last, 1, 0)
    has = use_last | use_prev
    rows = torch.arange(self.N, device=self.dev)
    az, j, t = self.h_az[rows, k], self.h_j[rows, k], self.h_t[rows, k]
    return ctx_bin(has, az, j, t), ctx_features(has, az, j, t)

  @torch.no_grad()
  def step(self, env) -> torch.Tensor:
    out = self.stream.update(self.reader.read())
    W, ok = out["window"], out["valid"]
    cb, cf = self._context(env)
    W0 = torch.nan_to_num(W, nan=0.0)
    lg = self.disc.logit(W0, cf)
    nc = self.real.nc[cb]
    r = panel_reward(lg, nc, self.p["n0"])
    D = torch.where(ok, r, torch.zeros_like(r))
    idx = torch.randint(0, self.N, (self.push_n,), device=self.dev, generator=self.g)
    self.buf.push_ctx(W[idx], ok[idx], cb[idx], cf[idx])
    slot = self.steps % RECENT
    self.recent[slot], self.recent_ok[slot] = W0.half(), ok
    self.recent_bin[slot], self.recent_feat[slot] = cb, cf.half()
    if float(self.p["fall_share"]) > 0:
      fell = env.reset_terminated
      if bool(fell.any()):
        ids = fell.nonzero().flatten()
        self.fallbuf.push_ctx(self.recent[:, ids].reshape(-1, WIN, NC), self.recent_ok[:, ids].reshape(-1),
                              self.recent_bin[:, ids].reshape(-1), self.recent_feat[:, ids].reshape(-1, CTX_DIM).float())
    self.steps += 1
    log = env.extras.setdefault("log", {})
    okf = ok.float()
    nv = okf.sum().clamp_min(1.0)
    shrink = nc / (nc + float(self.p["n0"]))
    log["Donline/valid_share"] = okf.mean()
    log["Donline/rD_mean_valid"] = (D * okf).sum() / nv
    log["Donline/logit_mean_valid"] = (lg * okf).sum() / nv
    log["Donline/rD_lt_005_valid"] = ((r < 0.05).float() * okf).sum() / nv
    log["Donline/abstain_share_valid"] = ((shrink < 0.5).float() * okf).sum() / nv
    log["Donline/stim_ctx_share_valid"] = ((cb > 0).float() * okf).sum() / nv
    stim = ok & (cb > 0)
    ns = stim.float().sum()
    log["Donline/rD_mean_stim"] = (D * stim.float()).sum() / ns.clamp_min(1.0)
    log["Donline/rD_lt_005_stim"] = ((r < 0.05).float() * stim.float()).sum() / ns.clamp_min(1.0)
    log["Donline/acc_fake_fresh"] = ((lg < 0).float() * okf).sum() / nv
    if self.steps % int(self.p["update_every"]) == 0:
      self.update()
    for k_, v in self.last_log.items():
      log[k_] = v
    return D

  def fake_batch(self, B):
    p = self.p
    nf = int(round(float(p["fall_share"]) * B)) if self.fallbuf.count() > 0 else 0
    nh = int(round(float(p["hard_share"]) * B)) if self.hard is not None else 0
    nm = B - nf - nh
    W, cb, cf = self.buf.sample_ctx(nm, self.g)
    parts = [(W, cb, cf)]
    if nf:
      parts.append(self.fallbuf.sample_ctx(nf, self.g))
    if nh:
      Wh, cbh, cfh, _ = self.hard.sample(nh)
      parts.append((Wh, cbh, cfh))
    return (torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts]), torch.cat([x[2] for x in parts]), nm)

  def update(self):
    if self.buf.count() < 1:
      self.skipped += 1
      self.last_log = {"Donline/update_skipped": torch.tensor(float(self.skipped), device=self.dev)}
      return
    p, B = self.p, int(self.p["batch"])
    agg = {}
    with torch.inference_mode(False), torch.enable_grad():
      for u in range(int(p["n_updates"])):
        Wf, cbf, cff, nm = self.fake_batch(B)
        Wf, cff = Wf.clone(), cff.clone()
        qf = still_windows(Wf, self.real.thr)
        Wr, mh, md, sr, cfr, sup = self.real.sample_cells(cbf * 2 + qf.long())
        WIh = self.real.sample_quiet_none(int(p["inv_batch"])).clone()
        WIr = self.pd[torch.randint(0, len(self.pd), (int(p["inv_batch"]),), device=self.dev, generator=self.g)].float()
        lr_, lf_, lb, gp, acc_i, lam = self.disc.train_step_bce(
          Wr.clone(), mh.clone(), md.clone(), cfr.clone(), Wf, cff, sup, self.g, WIh, WIr, p["acc_target"],
          p["lam_lr"], p["lam_max"])
        self.gate_allowed += int(float(acc_i) <= float(p["acc_target"]))
        if u == 0:
          s = sup.float()
          ns = s.sum().clamp_min(1.0)
          sm = s[:nm]
          agg = {"Donline/acc": 0.5 * (((lr_ > 0).float() * s).sum() / ns + ((lf_ < 0).float() * s).sum() / ns),
                 "Donline/acc_policy_part": ((lf_[:nm] < 0).float() * sm).sum() / sm.sum().clamp_min(1.0),
                 "Donline/logit_real": (lr_ * s).sum() / ns, "Donline/logit_fake": (lf_ * s).sum() / ns,
                 "Donline/supported_share": s.mean(), "Donline/fake_quiet_share": qf.float().mean(),
                 "Donline/acc_I": acc_i, "Donline/lambda": torch.tensor(lam, device=self.dev),
                 "Donline/loss_bce": lb, "Donline/grad_penalty": gp}
    self.updates += 1
    agg["Donline/updates"] = torch.tensor(float(self.updates), device=self.dev)
    agg["Donline/literal_gate_allowed_share"] = torch.tensor(
      self.gate_allowed / max(1, self.updates * int(p["n_updates"])), device=self.dev)
    agg["Donline/buffer_valid"] = torch.tensor(float(self.buf.count()), device=self.dev)
    self.last_log = agg
    if p["save_path"] and self.updates % int(p["save_every_updates"]) == 0:
      self.save(p["save_path"])

  def state_dict(self):
    return dict(disc=self.disc.state_dict(), updates=self.updates, steps=self.steps, params=self.p,
                gate_allowed=self.gate_allowed, generator=self.g.get_state())

  def save(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".part")
    torch.save(self.state_dict(), tmp)
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------------------- probe
def fit_probe(real: RealBank, fake_W: torch.Tensor, keep, seed: int, steps: int = 3000, batch: int = 2048,
              still_share=None, **kw) -> OnlineDisc:
  """Diagnostic mode: the same discriminator, data and augmentation as the online engine, refitted offline against
  a FIXED fake set (e.g. one recorded policy). fake_W: raw windows (n, 9, 48)."""
  od = OnlineDisc(keep, real.mu, real.sd, fake_W.device, seed, **kw)
  for _ in range(int(steps)):
    Wr, mh, md, _ = real.sample(batch, still_share)
    i = torch.randint(0, len(fake_W), (batch,), device=fake_W.device, generator=real.g)
    od.train_step(Wr, mh, md, fake_W[i].float(), real.g)
  return od


def channel_saliency(od: OnlineDisc, W: torch.Tensor) -> dict:
  """mean |d d / d z| over windows and samples per phi channel (z = standardised input, so units are per sd)."""
  x = prep(W, od.mu, od.sd, od.keep).requires_grad_(True)
  with torch.enable_grad():
    g = torch.autograd.grad(od.net(x).sum(), x)[0]
  g = g[:, :-2].reshape(len(W), WIN, len(od.keep)).abs().mean((0, 1))
  return {PD.NAMES[c]: float(v) for c, v in zip(od.keep, g)}


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
  """P(pos > neg) + 0.5 P(=), by ranks."""
  x = torch.cat([pos, neg]).double().cpu().numpy()
  from scipy.stats import rankdata
  rk = rankdata(x)
  n1, n0 = len(pos), len(neg)
  return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
