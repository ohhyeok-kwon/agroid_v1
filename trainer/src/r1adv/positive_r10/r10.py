"""R10 / W7 candidate cells (positive session 16; coordinator 2026-11-19 09:2x KST, .agent/adversarial/rounds/R04/10_r10_facts.md)
that need code beyond a one-line task override. Both build on the S0 task (r1adv.positive_s0.scene_s0, unchanged).

(b') S0-ssched  (W7 cell C1) S0 with the actor std not learned (rsl_rl GaussianDistribution learn_std=False) but SET every
                update to B0's recorded trajectory, entropy 0. Table SIGMA_TABLE (pinned by sha256; .agent/adversarial/positive/
                tools/r10_sigma_table.py -> v1 knots, r10_sigma_table2.py -> v2 rows): per cell (the same parent and seed as the
                B0 run r9-GMMo6_p<k>s<j>, R1ADV_SIGMA_CELL) and per update L = 4049..4800 the 24-joint std vector
                  std_j(L) = per-joint linear interpolation between B0's checkpoint vectors at 4049 (the warm-up's last = the
                  r6d parent's /4000 std) and 4100, 4200, ..., 4800 (coordinator 09:5x), stored as float32.
                Set at the first act() of each iteration (rollout and update use it), never learned;
                inactive while the actor is frozen (critic warm-up). Labels outside 4049-4800 raise (fail closed). Every update
                checks the std is still the set vector and logs it ('Mean r10_std_set loss' in train.log).
                Early-warning proxy (no extra simulation): from the rollout storage's mean actions mu (T 24 x envs x 24) the
                share of sign flips of consecutive mu changes (arms 14-23 / legs+waist 0-13) and the share of mu power at
                >= 20 Hz over the 24-step window (envs without a reset in it), logged every update as 'Mean r10_alt_arm loss'
                etc. These are training-rollout numbers (hits, pushes, scenes, sampled-action history), not quiet standing.
(f)  SHA-s      S0-ssched with the Hold objective's D (GMM human density) replaced by the s2f walking AMP style reward:
                r = L + D_AMP, D_AMP = clamp(1 - (d - 1)^2 / 4, 0) in [0, 1] from the parents' discriminator (careful_walk.npz
                walking clips, 44 leg / waist / pelvis features; --amp-disc = the parent run's amp_disc.pt), weight 1.0 (= beta
                of D), the objective term's beta 0 (it returns L; its D is still computed and logged as Objective/D). The AMP
                term logs the discriminator accuracies Metrics/amp_acc_real (d_ref > 0) and amp_acc_fake (d_policy < 0)."""
from __future__ import annotations

import dataclasses
import json
import os

import torch
from torch import nn

from r1adv.positive.s2f_r7a import ParentLrPPO, ParentLrPpoCfg
from r1adv.positive_scene import scene_r8 as R8
from r1lab.mdp import amp as AMP

SIGMA_TABLE = dict(file="r10_sigma_b0_v2.json", sha256="4cfb7f117191978d794f1ed316b3702f10524957c3dc18ed47bb7b2e39b480e4")
ARMS = tuple(range(14, 24))
LEGS = tuple(range(0, 14))
_TABLES: dict = {}


def sigma_table(file: str = SIGMA_TABLE["file"], sha256: str = SIGMA_TABLE["sha256"]) -> dict:
  key = (file, sha256)
  if key not in _TABLES:
    _TABLES[key] = json.loads(R8.pinned(dict(file=file, sha256=sha256)).read_text())
  return _TABLES[key]


def sigma_at(table: dict, cell: str, label: int) -> torch.Tensor:
  """The scheduled std vector (24,) at iteration label `label`: the table row (float32 values), CPU, deterministic."""
  lo, hi = table["labels"]
  if not lo <= label <= hi:
    raise ValueError(f"label {label} outside the B0 table {lo}-{hi}")
  return torch.tensor(table["cells"][cell]["std"][label - lo], dtype=torch.float32)


@dataclasses.dataclass
class SigmaSchedPpoCfg(R8.ScenePpoCfg):
  sigma_table: str = SIGMA_TABLE["file"]
  sigma_table_sha256: str = SIGMA_TABLE["sha256"]


@dataclasses.dataclass
class SigmaSchedFlatPpoCfg(ParentLrPpoCfg):
  sigma_table: str = SIGMA_TABLE["file"]
  sigma_table_sha256: str = SIGMA_TABLE["sha256"]


class _SigmaSched:
  """Mixin: the actor std follows the B0 table (see the module docstring)."""

  def __init__(self, *args, sigma_table: str = SIGMA_TABLE["file"], sigma_table_sha256: str = SIGMA_TABLE["sha256"], **kwargs):
    super().__init__(*args, **kwargs)
    self.r10_table = globals()["sigma_table"](sigma_table, sigma_table_sha256)
    self.r10_cell = os.environ.get("R1ADV_SIGMA_CELL", "")
    if self.r10_cell not in self.r10_table["cells"]:
      raise ValueError(f"R1ADV_SIGMA_CELL={self.r10_cell!r} not in the table cells {sorted(self.r10_table['cells'])}")
    if self._raw_actor.distribution.std_param.requires_grad:
      raise ValueError("the std schedule needs an actor with learn_std=False")
    self.r10_label, self.r10_need, self.r10_active, self.r10_set = None, True, None, None
    print(f"[R10 sigma] table {sigma_table} sha {sigma_table_sha256[:16]} cell {self.r10_cell} labels {self.r10_table['labels']}",
          flush=True)

  def load(self, loaded_dict, load_cfg, strict):
    out = super().load(loaded_dict, load_cfg, strict)
    self.r10_label, self.r10_need = int(loaded_dict.get("iter", 0)), True
    return out

  def _r10_apply(self) -> None:
    if not self.r10_need:
      return
    self.r10_need = False
    active = any(p.requires_grad for p in self._raw_actor.mlp.parameters())
    if self.r10_active is None:
      self.r10_active = active
      print(f"[R10 sigma] schedule {'on' if active else 'off (actor frozen: critic warm-up)'} at label {self.r10_label}",
            flush=True)
    if not active:
      return
    if self.r10_label is None:
      raise RuntimeError("the std schedule needs a resumed iteration label")
    s = sigma_at(self.r10_table, self.r10_cell, self.r10_label)
    p = self._raw_actor.distribution.std_param
    with torch.no_grad():
      p.copy_(s.to(p.device))
    self.r10_set = s

  def act(self, obs):
    self._r10_apply()
    return super().act(obs)

  def _r10_monitor(self) -> dict:
    st = self.storage
    params = getattr(st, "distribution_params", None)
    if not params:
      return {}
    with torch.no_grad():
      mu = params[0].float()                                         # (T, N, A) mean actions of this rollout
      done = st.dones.squeeze(-1).bool()                             # (T, N): reset after step t
      d = mu[1:] - mu[:-1]
      ok = ~done[:-1]
      both = (ok[:-1] & ok[1:]).unsqueeze(-1)
      nz = (d[:-1].abs() > 1e-6) & (d[1:].abs() > 1e-6) & both
      flip = (d[:-1] * d[1:] < 0) & nz
      out = {}
      for name, idx in (("arm", ARMS), ("leg", LEGS)):
        out[f"r10_alt_{name}"] = float(flip[..., idx].sum() / nz[..., idx].sum().clamp_min(1))
      clean = ~done[:-1].any(0)
      if bool(clean.any()):
        x = mu[:, clean]
        x = x - x.mean(0, keepdim=True)
        P = torch.fft.rfft(x, dim=0).abs() ** 2
        f = torch.arange(P.shape[0], device=P.device) * (50.0 / x.shape[0])
        hi, tot = P[f >= 20.0].sum(0), P[1:].sum(0)
        for name, idx in (("arm", ARMS), ("leg", LEGS)):
          out[f"r10_hf_{name}"] = float(hi[:, idx].sum() / tot[:, idx].sum().clamp_min(1e-12))
    return out

  def update(self):
    mon = self._r10_monitor()
    out = super().update()
    if self.r10_active:
      p = self._raw_actor.distribution.std_param.detach().float().cpu()
      if not torch.equal(p, self.r10_set):
        raise RuntimeError(f"std moved during the update at label {self.r10_label}")
      out["r10_std_set"] = float(self.r10_set.mean())
    out.update(mon)
    if self.r10_label is not None:
      self.r10_label += 1
    self.r10_need = True
    return out


class AmpStyleAcc(AMP.amp_style):
  """r1lab.mdp.amp.amp_style (unchanged reward, discriminator, buffer, schedule) + accuracy logs per discriminator update."""

  def _update(self, env):
    with torch.inference_mode(False), torch.enable_grad():
      for _ in range(self.n_updates):
        pi = torch.randint(0, self.buf_n, (self.batch,), device=self.ref.device)
        if self.ref_w is not None:
          ri = torch.multinomial(self.ref_w, self.batch, replacement=True)
        else:
          ri = torch.randint(0, self.ref.shape[0], (self.batch,), device=self.ref.device)
        x_pi = self._norm(self.buf[pi].clone())
        x_ref = self._norm(self.ref[ri].clone()).requires_grad_(True)
        d_pi = self.disc(x_pi)
        d_ref = self.disc(x_ref)
        loss = ((d_ref - 1) ** 2).mean() + ((d_pi + 1) ** 2).mean()
        g = torch.autograd.grad(d_ref.sum(), x_ref, create_graph=True)[0]
        loss = loss + self.gp * g.pow(2).sum(-1).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
        self.opt.step()
    log = env.extras["log"]
    log["Metrics/amp_d_ref"] = d_ref.detach().mean()
    log["Metrics/amp_d_policy"] = d_pi.detach().mean()
    log["Metrics/amp_acc_real"] = (d_ref.detach() > 0).float().mean()
    log["Metrics/amp_acc_fake"] = (d_pi.detach() < 0).float().mean()


class SigmaSchedScenePPO(_SigmaSched, R8.ScenePPO):
  """ScenePPO (scene widening + scene mirror) whose actor std follows the B0 table: cells C1 / C2."""


class SigmaSchedPPO(_SigmaSched, ParentLrPPO):
  """ParentLrPPO (B0's PPO, no scene inputs) whose actor std follows the B0 table: contrast cell C2-flat."""


AMP_WEIGHT = 1.0                         # coordinator 09:5x: D_AMP (amp_style's step value, in [0, 1]) with weight 1.0 = D's in B0


def ssched_rl(cfg, scene: bool = True):
  """A ScenePPO (scene=True) or ParentLrPPO (scene=False) runner cfg -> the std-schedule variant, entropy 0, learn_std False."""
  base = cfg.algorithm
  cls = SigmaSchedPpoCfg if scene else SigmaSchedFlatPpoCfg
  cfg.algorithm = cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
  cfg.algorithm.class_name = "r1adv.positive_r10.r10:" + ("SigmaSchedScenePPO" if scene else "SigmaSchedPPO")
  cfg.algorithm.entropy_coef = 0.0
  cfg.actor.distribution_cfg = dict(cfg.actor.distribution_cfg, learn_std=False)
  return cfg


def use_d_amp(cfg, play: bool = False, t0: int = 1499):
  """r = L + D_AMP: the Hold objective term with beta 0 (returns L) + style = AmpStyleAcc, weight 1, s2f 'long' params."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from r1adv.positive.s2f_r6d import s2f_env_cfg
  cfg.rewards["objective"].params["beta"] = 0.0
  ref = s2f_env_cfg("long", play=play, t0=t0).rewards["style"]
  cfg.rewards["style"] = RewardTermCfg(func=AmpStyleAcc, weight=AMP_WEIGHT, params=dict(ref.params))
  return cfg


def ssched_rl_cfg():
  """C1 / C2: S0's PPO (ScenePPO, start lr 1.139e-4) with SigmaSchedScenePPO, entropy 0, learn_std False."""
  from r1adv.positive_s0.scene_s0 import s0_rl_cfg
  return ssched_rl(s0_rl_cfg(), scene=True)


def shas_env_cfg(play: bool = False, t0: int = 1499):
  """C2: S0's environment with r = L + D_AMP."""
  from r1adv.positive_s0.scene_s0 import s0_env_cfg
  return use_d_amp(s0_env_cfg(play=play, t0=t0), play=play, t0=t0)
