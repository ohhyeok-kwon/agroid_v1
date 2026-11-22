"""Environment families v1: the R1 Hold set (coordinator decisions 02:45, items 4, 6 and 7).

Physics in R1 is boxes and planes only (no height fields, no scans). Missions come from the coordinator's R1 range,
Hold: settle 1 s, then stand still for 8 s with a zero twist (an assumption written in level_stream.md). Every family
is PARAM_SPACE + make(params, rng, mission) as in families.py; the witness rule is decided per level:
  "observation"  the level degrades what the robot senses (noise, IMU bias or mounting tilt, latency): the witness is
                 the best policy with the learner's own observations
  "physical"     otherwise: the calmest successful policy of our population (teachers, students, champion snapshots)
The manufacturer is never a witness. baseline v1 is noise-free (v0's baseline drew noise but asked a teacher).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from r1adv.negative import bank as B
from r1adv.negative.families import NEUTRAL_BODY, Family, _nearest_row, _terrain
from r1adv.negative.generator import DT, SETTLE_S, ticks
from r1adv.negative.seedtree import Draw, child_seed, r6

HOLD_S = 8.0


def mission_hold(d: Draw) -> dict:
  """The coordinator's R1 Hold mission (fixed length; d is kept for future ranges)."""
  del d
  return dict(grammar="mission-v1", settle_s=SETTLE_S, duration_s=r6(SETTLE_S + HOLD_S),
              steps=[dict(type="Hold", duration_s=ticks(HOLD_S))])


def _sense(d: Draw, scale=0.0, gyro=0.0, tilt=0.0, delay=0, jitter=0) -> dict:
  sense = dict(profile="uniform-n1shape-v1", scale=r6(scale), delay_steps=int(delay), delay_jitter_steps=int(jitter),
               dropout_p=0.0, noise_seed=child_seed(int(d.u() * (1 << 53)), "sense/noise"))
  if gyro > 0.0:
    a, b = d.uniform(-math.pi, math.pi), d.uniform(-1.0, 1.0)
    sense["gyro_bias_radps"] = [r6(gyro * math.sqrt(1 - b * b) * math.cos(a)), r6(gyro * math.sqrt(1 - b * b) * math.sin(a)), r6(gyro * b)]
  if tilt > 0.0:
    a = d.uniform(-math.pi, math.pi)
    sense["attitude_bias_rad"] = [r6(tilt * math.cos(a)), r6(tilt * math.sin(a))]
  return sense


def _slope(slope_deg: float) -> tuple[str, int]:
  kind = "tilt" if slope_deg <= 16.0 else "steep"
  return kind, _nearest_row(kind, slope_deg)


def _env(terrain: dict, sense: dict) -> dict:
  return dict(terrain=terrain, scene=None, obstacles=[], disturbances=[], sense=sense, body=dict(NEUTRAL_BODY))


def _flat(p, d, m):
  return _env(_terrain("flat", 0, math.radians(p["yaw_deg"])), _sense(d))


def _hold_slope(p, d, m):
  kind, row = _slope(p["slope_deg"])
  return _env(_terrain(kind, row, math.pi / 2 + math.radians(p["heading_deg"])), _sense(d))


def _noise(p, d, m):
  return _env(_terrain("flat", 0, d.uniform(-math.pi, math.pi)), _sense(d, scale=p["noise_scale"]))


def _bias(p, d, m):
  return _env(_terrain("flat", 0, d.uniform(-math.pi, math.pi)),
              _sense(d, scale=p["noise_scale"], gyro=p["gyro_bias_radps"], tilt=p["mount_tilt_rad"]))


def _delay(p, d, m):
  return _env(_terrain("flat", 0, d.uniform(-math.pi, math.pi)), _sense(d, delay=p["delay_steps"], jitter=p["jitter_steps"]))


def _slope_sense(p, d, m):
  kind, row = _slope(p["slope_deg"])
  return _env(_terrain(kind, row, math.pi / 2 + math.radians(p["heading_deg"])),
              _sense(d, scale=p["noise_scale"], delay=p["delay_steps"]))


def _slope_limits(p):
  return ["static slope beyond 0.8 of the friction cone"] if math.tan(math.radians(p["slope_deg"])) >= 0.8 else []


FAMILIES = {f.name: f for f in (
  Family("hold-flat-v1", "baseline", "legacy", "physical", "stand still on a level floor",
         {"yaw_deg": ("float", -180.0, 180.0)}, _flat),
  Family("hold-slope-v1", "trap", "legacy", "physical", "stand still on a ramp, facing up, down or across",
         {"slope_deg": ("float", 3.0, 25.0), "heading_deg": ("float", -180.0, 180.0)}, _hold_slope, _slope_limits),
  Family("hold-noise-v1", "trap", "legacy", "observation", "stand still with noisy IMU and encoders",
         {"noise_scale": ("float", 0.2, 1.5)}, _noise),
  Family("hold-bias-v1", "trap", "legacy", "observation", "stand still with a biased gyro and a tilted IMU mount",
         {"gyro_bias_radps": ("float", 0.0, 0.15), "mount_tilt_rad": ("float", 0.0, 0.08), "noise_scale": ("float", 0.0, 0.5)}, _bias),
  Family("hold-delay-v1", "trap", "legacy", "observation", "stand still with late and jittering sensor data",
         {"delay_steps": ("int", 0, 4), "jitter_steps": ("int", 0, 2)}, _delay,
         lambda p: ["latency beyond the kept history"] if p["delay_steps"] + p["jitter_steps"] >= 8 else []),
  Family("hold-slope-sense-v1", "trap", "legacy", "observation", "stand still on a ramp with noisy, late sensors",
         {"slope_deg": ("float", 3.0, 16.0), "heading_deg": ("float", -180.0, 180.0), "noise_scale": ("float", 0.0, 1.0),
          "delay_steps": ("int", 0, 2)}, _slope_sense, _slope_limits),
)}


def sensor_degraded(env: dict) -> bool:
  s = env["sense"]
  return bool(s.get("scale", 0.0) > 0 or any(s.get("gyro_bias_radps", [0.0])) or any(s.get("attitude_bias_rad", [0.0]))
              or s.get("delay_steps", 0) or s.get("delay_jitter_steps", 0))


def witness_rule(env: dict) -> str:
  return "observation" if sensor_degraded(env) else "physical"


def source_sha256() -> str:
  here = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  for name in ("families_v1.py", "families.py", "levels.py", "generator.py", "bank.py", "seedtree.py", "__init__.py"):
    digest.update(name.encode() + b"\0" + (here / name).read_bytes() + b"\0")
  return digest.hexdigest()
