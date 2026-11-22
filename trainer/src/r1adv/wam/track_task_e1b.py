"""E1b candidate (coordinator 17:00: approved as a candidate for after the E1 6000 read-out; not registered, not run):
the E1 arms (track_task.env_cfg / rl_cfg, unchanged) on feasibility-projected references with the reference-conditioned
trunk ending (r1adv.wam.feasible):
  (a) libraries e1b_library_<arm>.npz = the E1 training clips, the 5 whose bodies go > 3 cm into the floor lifted
      (wam/tools/e1b_prepare.py), with <library>.trunk_clear.npy beside them;
  (b) trunk_ground ends an episode only where the reference trunk is more than `margin` above the floor.
Rewards, sampler, PPO and budget are E1's, so a difference against E1 at the same checkpoint is the fix's."""
from __future__ import annotations

from pathlib import Path

from mjlab.managers.termination_manager import TerminationTermCfg

from r1lab.run_record import work_root
from r1adv.wam import feasible, track_task


def library_path(arm: str) -> Path:
  return work_root() / f"data/wam/e1b_library_{arm}.npz"


def env_cfg(arm: str, play: bool = False, first_clip: int = 0, library: str | None = None, margin: float = 0.05, **kw):
  lib = str(library or library_path(arm))
  cfg = track_task.env_cfg(arm, play, first_clip, library=lib, **kw)
  cfg.terminations["trunk_ground"] = TerminationTermCfg(
    func=feasible.unexpected_trunk_contact,
    params={"sensor_name": "trunk_ground", "command_name": "motion", "clear_file": lib + ".trunk_clear.npy", "margin": margin})
  return cfg


def rl_cfg(init_std: float = 1.0):
  return track_task.rl_cfg(init_std)
