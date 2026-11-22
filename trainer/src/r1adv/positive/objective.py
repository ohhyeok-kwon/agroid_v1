"""The positive agent's fixed objective (coordinator decision 2026-11-17 01:35 KST, panel2_strategy_synthesis.md 1):

    r_t = L_t + beta * D_t          falls end the episode (physical termination only; no tracking ends, no graces)
    L_t  mission success likelihood of the command errors e_k with widths sigma_k (fixed per mission by the coordinator)
    D_t  human-motion score in (0, 1] from the surprise S_t of the last 1 s of motion under the fixed density D

Both are likelihoods with a heavy tail (Student-t, nu = 1, "cauchy"): near zero error they curve like a Gaussian, far
from it they fall as 1/e^2 instead of exp(-e^2), so they never saturate to a zero gradient. Measured reason (R01
sigma_beta_proposal.md): with a Gaussian no single sigma both separates a 0.01 rad standing offset (dL >= 0.05) and keeps
a gradient at the 0.12-0.2 rad posture errors of slopes and of the start of training; the exponential D mapping pressed
every slope sway and every hit recovery (also the manufacturer's) to 0.004-0.012.
Nothing else enters the reward. sigma_k, beta and the form are locked by the coordinator; D's features, the human data
mix, the command curriculum and distillation settings are the positive agent's knobs. The file is frozen by its sha256
(formula_sha256() is written into every run record); a changed file is a new formula version.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

FORMULA_VERSION = "r1adv-positive-objective-v1"          # R1-R2: HOLD below (D = d_v0b); every R2 run keeps this
FORMULA_VERSION_R3 = "r1adv-positive-objective-v2"       # R3 on: HOLD_R3 (D = d_v0c), switched between waves only
FORMS = ("cauchy", "gauss")

# Every number the reward of a mission depends on lives here, so that the sha256 of this file locks them all.
# Locked by the coordinator 2026-11-17 02:55 KST (rounds/R01/coordinator_lock_hold_r1.md); record in
# rounds/R01/positive/lock_record.md. Any change of this file is a new formula version.
HOLD = dict(
  locked=True,
  form="cauchy",
  sigma=dict(posture=0.02, height=0.02, tilt=0.035, speed=0.05, yaw=0.2),   # rad RMS (24 joints), m, rad, m/s, rad/s
  beta_bracket=(0.0, 1.0),
  d_file="d_v0b.npz", d_sha256="a4aefdad71dca66c516d0ef890576c1ab85efeecc362526d8ee1224e6684de1f",
  target_file="hold_target_v0d.json", target_sha256="885bb4d1febc428a9e7e168f2f90d9625fcb38bc17009b125d77f8f6f6051725",
)

# Version 2 (coordinator 2026-11-17 17:0x KST, after rounds/R02/positive/d_v0_data_defect.md): the same formula, sigma,
# form, beta bracket and target; only the density file changes to d_v0c = d_v0b refitted with the same procedure
# WITHOUT the six lying idle clips and the fall-landing clip that were filed under 'stand' / 'hit_stagger' in
# shared/motion_train (7.4 % of the fit windows). Used from R3 on; R2 runs (C1, resumed R2d) keep HOLD / d_v0b.
# Record: rounds/R01/positive/lock_record.md (addendum v2).
HOLD_R3 = dict(HOLD, d_file="d_v0c.npz", d_sha256="4b2bc118e465af08c6f423cd91774c8f1d084a5c80c09bdc69701455b56f0542")


def formula_sha256() -> str:
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def mission_likelihood(errors: dict[str, torch.Tensor], sigma: dict[str, float], form: str = "cauchy") -> torch.Tensor:
  """cauchy: 1 / (1 + 1/2 sum_k (e_k / sigma_k)^2);  gauss: exp(-1/2 sum_k (e_k / sigma_k)^2).
  Only the command fields present in `errors` count (masked fields are absent)."""
  if form not in FORMS:
    raise ValueError(f"unknown likelihood form {form!r}")
  missing = [k for k in errors if sigma.get(k) is None]
  if missing:
    raise ValueError(f"sigma not fixed by the coordinator for fields: {missing}")
  total = None
  for k, e in errors.items():
    term = (e / float(sigma[k])).pow(2)
    total = term if total is None else total + term
  if total is None:
    raise ValueError("a mission step needs at least one command field")
  return 1.0 / (1.0 + 0.5 * total) if form == "cauchy" else torch.exp(-0.5 * total)


def naturalness(surprise: torch.Tensor, s_ref: float, s_scale: float, form: str = "cauchy") -> torch.Tensor:
  """D in (0, 1]: 1 up to the median human surprise s_ref; beyond it 1 / (1 + excess / s) (cauchy) or exp(-excess / s)."""
  if form not in FORMS:
    raise ValueError(f"unknown mapping form {form!r}")
  excess = (surprise - s_ref).clamp_min(0.0) / s_scale
  return 1.0 / (1.0 + excess) if form == "cauchy" else torch.exp(-excess)


def reward(likelihood: torch.Tensor, natural: torch.Tensor, beta: float) -> torch.Tensor:
  """r_t = L_t + beta * D_t (both in (0, 1]); beta >= 0."""
  if beta < 0:
    raise ValueError("beta must be >= 0")
  return likelihood + beta * natural
