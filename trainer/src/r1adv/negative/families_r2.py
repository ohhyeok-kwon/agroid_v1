"""R2 curriculum family (coordinator 12:30): a level's genes are its hits and its start frame only. Terrain is flat,
sensors are the locked training model (r1lab-deploy-v1, 1x), command delay is fixed per probe batch; none of these is
searched. The base training distribution (no hit + start spread) is the calm anchor; hits come only from levels.

  start     "default" (the deploy nominal pose) or one of the 1,132 standing frames of r2_train_distribution_v1.json
  hits      1-2 torso impulses: size U(4, 25) N s, side U(-pi, pi) relative to the heading, height U(-0.1, 0.3) m above
            the torso centre of mass; the first at U(2.5, 5) s (later than the training file's 1.5 s, so that the 1 s
            before the first hit is past the 1 s settle and can be judged calm), the second U(2, 6) s later if it still
            leaves >= 2.5 s to recover before the 10 s episode ends
  mission   settle 1 s + Hold 9 s (the positive's 10 s training episode)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from r1adv.negative.families import NEUTRAL_BODY, Family, _terrain
from r1adv.negative.generator import SETTLE_S, ticks
from r1adv.negative.seedtree import Draw, child_seed, r6

HOLD_S = 9.0
EPISODE_S = SETTLE_S + HOLD_S
DISTRIBUTION = Path(__file__).resolve().parents[4] / ".agent/adversarial/shared/r2/r2_train_distribution_v1.json"
SERVER_DISTRIBUTION = Path("/rl-agent/.agent/adversarial_negative/specs/r2/r2_train_distribution_v1.json")
_FRAMES = None


def frames() -> list[dict]:
  global _FRAMES
  if _FRAMES is None:
    path = DISTRIBUTION if DISTRIBUTION.exists() else SERVER_DISTRIBUTION
    _FRAMES = json.loads(path.read_text())["start"]["human_frame"]["frames"]
  return _FRAMES


def mission_r2(d: Draw) -> dict:
  del d
  return dict(grammar="mission-v1", settle_s=SETTLE_S, duration_s=r6(EPISODE_S), steps=[dict(type="Hold", duration_s=ticks(HOLD_S))])


PARAM_SPACE = {
  "start": ("choice", ("default", "frame")),
  "frame": ("int", 0, 1131),
  "hits": ("int", 1, 2),
  "j1": ("float", 4.0, 25.0), "az1": ("float", -180.0, 180.0), "h1": ("float", -0.1, 0.3), "t1": ("float", 2.5, 5.0),
  "j2": ("float", 4.0, 25.0), "az2": ("float", -180.0, 180.0), "h2": ("float", -0.1, 0.3), "dt2": ("float", 2.0, 6.0),
}


def hits_of(p: dict) -> list[dict]:
  if p.get("hits", 1) == 0:                         # the base anchor (no hit)
    return []
  out = [dict(kind="impulse", t_s=ticks(p["t1"]), body="torso_link", impulse_ns=r6(p["j1"]), burst_count=1, burst_gap_s=0.02,
              duration_s=0.02, azimuth_rad=r6(math.radians(p["az1"])), height_m=r6(p["h1"]))]
  t2 = p["t1"] + p["dt2"]
  if p["hits"] >= 2 and t2 <= EPISODE_S - 2.5:
    out.append(dict(kind="impulse", t_s=ticks(t2), body="torso_link", impulse_ns=r6(p["j2"]), burst_count=1, burst_gap_s=0.02,
                    duration_s=0.02, azimuth_rad=r6(math.radians(p["az2"])), height_m=r6(p["h2"])))
  return out


def start_of(p: dict) -> dict | None:
  if p["start"] == "default":
    return None
  f = frames()[int(p["frame"]) % len(frames())]
  return dict(source=f.get("clip"), frame=f.get("frame"), joint_pos_rad=f["joint_pos_rad"],
              pelvis_roll_rad=f["pelvis_roll_rad"], pelvis_pitch_rad=f["pelvis_pitch_rad"])


def _make(p: dict, d: Draw, m: dict) -> dict:
  return dict(terrain=_terrain("flat", 0, d.uniform(-math.pi, math.pi)), scene=None, obstacles=[], disturbances=hits_of(p),
              sense=dict(profile="r1lab-deploy-v1", scale=1.0, delay_steps=0, delay_jitter_steps=0, dropout_p=0.0,
                         joint_vel_model="white", noise_seed=child_seed(int(d.u() * (1 << 53)), "sense/noise")),
              body=dict(NEUTRAL_BODY), start_pose=start_of(p))


FAMILIES = {f.name: f for f in (
  Family("r2-hit-v1", "trap", "legacy", "physical", "stand calmly, take one or two hits, settle again (stepping allowed)",
         PARAM_SPACE, _make),
)}


def source_sha256() -> str:
  here = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  for name in ("families_r2.py", "families.py", "levels.py", "generator.py", "bank.py", "seedtree.py", "__init__.py"):
    digest.update(name.encode() + b"\0" + (here / name).read_bytes() + b"\0")
  return digest.hexdigest()
