"""R7 wave-4 branch B building block of the positive agent (positive session 10, coordinator orders 2026-11-18 13:4x,
19:3x, 20:0x KST; design rounds/R04/07B_r7_synthesis.md 2 / 5): XBS (hold_w4s: the XB environment + option S human
pushes) with ONLY the D slot of the fixed objective replaced:

    r_t = L_t + beta * D_t,   beta = 1,   L unchanged (Cauchy likelihood of the Hold channels),
    D_t = r_D of the online discriminator on the policy's newest phi window and its stimulus context
          (r1adv.positive.amp_online.OnlineD2: BCE logit l ~ log(p_human / p_policy), r_D = exp(min(0, l n_c / (n_c + n0)));
          1 when indistinguishable or without human evidence, never above 1), instead of the HumanSurprise density score.
No term is added; termination, hits, pushes, observations, actor / critic and PPO settings are XBS's (config diff:
rewards.objective.func and rewards.objective.params.amp).

Variants: DON = the panel design (registered task adv_hold_w4_xbs_amp, tag r7b-DON); AMP1 = the first implementation
(LSGAN, AMP map max(0, 1 - (d - 1)^2 / 4), no context; amp_online.OnlineAMP) kept reachable for comparison
(cell_env_cfg("AMP1")), not registered.
Run tokens: --run-name r7b-DON_s<seed> -> run id r1adv_hold_<KST stamp>_r7b-DON_n<envs>_r7b-DON_s<seed>
(rounds/R04/positive/r10_amp/run_tokens.md; 'r7a-' belongs to branch A). Use only after gate T1 passes
(rounds/R04/positive/r10_amp/t1_gate.md). The evaluation / play environment (play=True) is XBS's: no bank, no discriminator.
"""
from __future__ import annotations

import torch

from r1adv.positive import amp_online as AO
from r1adv.positive import hold_task as HT
from r1adv.positive import hold_w4s as W4S
from r1adv.positive import objective as OBJ

DON_FILES = dict(real_file="amp_real_v2.npz", real_sha256="3c284d428de0ed1e8f947173ef1f4c02c7955e9fb64d4987cf39903f15181f51",
                 hard_file="amp_hard_v1.npz", hard_sha256="7d6cb3198d99eeb25c3f1b2b99dc588a807ff5358821247448803cee78c35ec1",
                 irob_file="amp_irob_v1.npz", irob_sha256="b7e3fc6a1b55a39c8990fbb70c0ed1ee6ca740dd94819e0f444ed02d846c71ac")
AMP1_FILES = dict(real_file="amp_real_v1.npz", real_sha256="60ba21fb99a4df2beb48317cfc7c2ed2c66119ada52379266bfc042eeaad987a",
                  fail_file="amp_fail_v1.npz", fail_sha256="66d4f70e75fd5a89b3afef663f77f7d2433d32470b058db466b27268508e06a0")
VARIANTS = {"DON": ("panel", DON_FILES), "AMP1": ("lsgan", AMP1_FILES)}
TASKS = {"DON": "adv_hold_w4_xbs_amp"}


class HoldObjectiveAMP(HT.HoldObjective):
  """HoldObjective (r = L + beta D, same channels, sigma, form and formula hash) with D = the online discriminator's
  reward. The HumanSurprise density is not evaluated. Everything this term builds or draws stays off the global RNG."""

  def __init__(self, cfg, env):
    dev = torch.device(env.device)
    forked = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=forked):
      super().__init__(cfg, env)
      self.surprise = None
      p = cfg.params["amp"]
      self.amp = AO.OnlineD2(env, p, HT.INPUT_DIRS) if p.get("mode") == "panel" else AO.OnlineAMP(env, p, HT.INPUT_DIRS)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    self.amp.reset(None if env_ids is None or isinstance(env_ids, slice) else env_ids)

  def __call__(self, env, beta: float, sigma: dict, form: str, formula_sha256: str = "", amp: dict | None = None):
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


def amp_params(variant: str) -> dict:
  mode, files = VARIANTS[variant]
  p = AO.resolve2(dict(files)) if mode == "panel" else AO.resolve(dict(files))
  return {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}


def cell_env_cfg(variant: str, play: bool = False):
  cfg = W4S.cell_env_cfg("XBS", play=play)
  if play:
    return cfg                                      # evaluation: XBS's play environment, no discriminator
  term = cfg.rewards["objective"]
  if term.func is not HT.HoldObjective:
    raise ValueError("the XBS objective term is not the point-posture HoldObjective")
  term.func = HoldObjectiveAMP
  term.params["amp"] = amp_params(variant)
  return cfg


def cell_rl_cfg(variant: str, max_iterations: int = 1500):
  if variant not in VARIANTS:
    raise ValueError(f"unknown D_online variant {variant!r}")
  return W4S.cell_rl_cfg("XBS", max_iterations=max_iterations)


def record() -> dict:
  return dict(variants={k: v[0] for k, v in VARIANTS.items()}, tasks=TASKS, files=dict(DON=DON_FILES, AMP1=AMP1_FILES),
              defaults=dict(panel=AO.DEFAULTS2, lsgan=AO.DEFAULTS), base_task="adv_hold_w4_xbs", tags={"DON": "r7b-DON"})
