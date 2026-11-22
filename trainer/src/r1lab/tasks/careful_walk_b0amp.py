"""B0+AMP (positive 23, coordinator 2026-11-20 18:4x; rounds/R04/16_rounds.md 'R17 feedback' and 'E2 card'). DIAGNOSTIC.
B0 (r1lab.tasks.careful_walk_b0: S1 objective v2 env + the anchor's representation and trainer) plus the anchor's AMP style
reward term exactly as careful_walk stage 1 has it: r1lab.mdp.amp.amp_style, weight 1.5, data AMP_DATA (careful_walk.npz),
training only (r1lab.train sets save_path = <run>/amp_disc.pt). Nothing else changes."""
from __future__ import annotations


def b0amp_env_cfg(play: bool = False, t0: int = 0):
  from r1lab.tasks.careful_walk import careful_walk_env_cfg
  from r1lab.tasks.careful_walk_b0 import b0_env_cfg
  cfg = b0_env_cfg(play=play, t0=t0)
  if not play:
    cfg.rewards["style"] = careful_walk_env_cfg(play=False, flat=True, stage=1, t0=t0).rewards["style"]
  return cfg


def b0amp_rl_cfg():
  from r1lab.tasks.careful_walk_b0 import b0_rl_cfg
  return b0_rl_cfg()
