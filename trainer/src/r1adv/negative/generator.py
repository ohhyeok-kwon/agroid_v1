"""Seeded scenario generator v0 -> r1adv-scenario-v1 specs (01_interfaces.md section 1).

A spec is a pure function of (family rules, instance seed, generator version and source hash). Families:
  legacy-v0  what the existing r1wb policies can run: Hold / Go(velocity) / Pose(pelvis height, then a rise Hold),
             torso impulses at any time, direction, height and size (single or bursts), uniform sensor noise of the
             N1 shape scaled 0..1.25, terrain from the bank (kind, level, spawn yaw). Body, delay and dropout neutral.
  full-v0    for policies still to be trained: adds Go(waypoint + arrival time, then stop), thrown balls and boxes,
             sensor delay and dropout, body mass / payload / friction / PD-gain errors and indoor scenes (by id).
  pc-v0      fixed positive-control cells (side-slope stop, uphill walk, noisy standing) for the runner check.
usage: python -m r1adv.negative.generator --family legacy-v0 --count 1000 --round R00 --out specs.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from r1adv.negative import NAME, VERSION, bank
from r1adv.negative.seedtree import Draw, canonical, child_seed, r6, sha256_text, source_sha256, spec_hash

ROOT_SEED = 20261117
DT = 0.02
DEFAULT_HEIGHT = 0.731
TRAVEL_BUDGET_M = 3.5        # the robot stays inside its 12 m patch
MISSION_CAP_S = 16.0
SETTLE_S = 1.0
BANK_SEED = child_seed(ROOT_SEED, bank.NAME)


def round_seed(name: str) -> int:
  return child_seed(ROOT_SEED, f"round/{name}")


def family_seed(round_name: str, family: str) -> int:
  return child_seed(round_seed(round_name), f"family/{family}")


def instance_seed(round_name: str, family: str, index: int) -> int:
  return child_seed(family_seed(round_name, family), f"instance/{index}")


def ticks(seconds: float) -> float:
  return r6(max(1, round(seconds / DT)) * DT)


# -- d_goal --------------------------------------------------------------------------------------------------------

def _go_velocity(d: Draw) -> dict:
  mode = d.choice(("straight", "turn", "arc"), (0.60, 0.15, 0.25))
  if mode == "straight":
    speed, azimuth = d.uniform(0.15, 0.45), d.uniform(-math.pi, math.pi)
    vx, vy, wz = speed * math.cos(azimuth), speed * math.sin(azimuth), 0.0
  elif mode == "turn":
    vx = vy = 0.0
    wz = d.choice((-1.0, 1.0)) * d.uniform(0.4, 1.0)
  else:
    vx, vy = d.uniform(0.2, 0.4), 0.0
    wz = d.choice((-1.0, 1.0)) * d.uniform(0.15, 0.5)
  return dict(type="Go", mode="velocity", shape=mode, vx=r6(vx), vy=r6(vy), wz=r6(wz), duration_s=ticks(d.uniform(2.0, 5.0)))


def _go_waypoint(d: Draw) -> dict:
  """Waypoint in the robot's start-of-step frame, arrival time, then the robot is to stand there."""
  distance, bearing = d.uniform(0.4, 2.5), d.uniform(-math.pi, math.pi)
  speed = d.uniform(0.15, 0.45)
  return dict(type="Go", mode="waypoint", x=r6(distance * math.cos(bearing)), y=r6(distance * math.sin(bearing)),
              heading_rad=r6(d.uniform(-math.pi, math.pi)) if d.bernoulli(0.3) else None,
              arrive_s=ticks(distance / speed + 0.5), duration_s=ticks(distance / speed + 0.5 + d.uniform(1.0, 3.0)))


def _steps(d: Draw, k: int, waypoints: bool) -> list[dict]:
  steps, previous = [], None
  for i in range(k):
    last = i == k - 1
    if previous == "Pose":
      kind = "Hold"                                   # rise from the crouch while standing (in the S2 vocabulary)
    else:
      weights = {"Hold": 0.30, "Go": 0.50, "Pose": 0.20}
      if previous == "Hold":
        weights["Hold"] = 0.10
      if last and previous == "Go":
        weights = {"Hold": 0.65, "Go": 0.20, "Pose": 0.15}
      kind = d.choice(tuple(weights), tuple(weights.values()))
    if kind == "Hold":
      steps.append(dict(type="Hold", duration_s=ticks(d.uniform(2.0, 5.0))))
    elif kind == "Pose":
      steps.append(dict(type="Pose", pelvis_height_m=r6(d.uniform(0.54, 0.70)), torso_pitch_rad=0.0,
                        duration_s=ticks(d.uniform(2.5, 5.0))))
    else:
      steps.append(_go_waypoint(d) if waypoints and d.bernoulli(0.4) else _go_velocity(d))
    previous = kind
  if steps[-1]["type"] == "Pose":                     # never end crouched without the rise
    steps.append(dict(type="Hold", duration_s=ticks(d.uniform(2.0, 4.0))))
  return steps


def _fit(steps: list[dict]) -> list[dict]:
  """Bound the commanded travel (the robot stays inside its patch) and the mission length."""
  travel = 0.0
  for step in steps:
    if step["type"] == "Go" and step["mode"] == "velocity":
      speed = math.hypot(step["vx"], step["vy"])
      room = TRAVEL_BUDGET_M - travel
      if speed * step["duration_s"] > room:
        step["duration_s"] = ticks(max(room / max(speed, 1e-6), 1.0))
        if speed * step["duration_s"] > room + 1e-9:           # no room left: walk becomes a turn in place
          step.update(vx=0.0, vy=0.0, wz=r6(0.6 if step["wz"] >= 0 else -0.6), shape="turn")
      travel += math.hypot(step["vx"], step["vy"]) * step["duration_s"]
    elif step["type"] == "Go":
      travel += math.hypot(step["x"], step["y"])
  total = sum(s["duration_s"] for s in steps)
  if total > MISSION_CAP_S:
    scale = MISSION_CAP_S / total
    for step in steps:
      step["duration_s"] = ticks(max(step["duration_s"] * scale, 1.0))
  return steps


def mission(d: Draw, family: str) -> dict:
  k = d.choice((1, 2, 3, 4), (0.15, 0.35, 0.30, 0.20))
  steps = _fit(_steps(d, k, waypoints=family == "full-v0"))
  return dict(grammar="mission-v1", settle_s=SETTLE_S, steps=steps,
              duration_s=r6(SETTLE_S + sum(s["duration_s"] for s in steps)))


# -- d_env ---------------------------------------------------------------------------------------------------------

def terrain(d: Draw) -> dict:
  weights = dict(flat=0.20, rough=0.12, gravel=0.08, tilt=0.20, steep=0.12, stairs_up=0.08, stairs_down=0.08,
                 tilt_rough=0.12)
  kind = d.choice(tuple(weights), tuple(weights.values()))
  row = d.integer(0, bank.ROWS - 1)
  value = bank.level(kind, row)
  return dict(bank=bank.NAME, bank_seed=BANK_SEED, bank_sha256=bank.bank_sha256(BANK_SEED), kind=kind, row=row,
              level=list(value) if isinstance(value, tuple) else value, slope_deg=r6(bank.slope_deg(kind, row)),
              spawn_yaw_rad=r6(d.uniform(-math.pi, math.pi)))


def impulses(d: Draw, mission_s: float) -> list[dict]:
  if not d.bernoulli(0.5):
    return []
  count = d.choice((1, 2, 3), (0.6, 0.3, 0.1))
  begin, end = SETTLE_S + 0.5, mission_s - 2.0
  times = []
  for _ in range(count):
    if end - begin < 0.5:
      break
    for _attempt in range(8):
      t = d.uniform(begin, end)
      if all(abs(t - other) >= 1.5 for other in times):
        times.append(t)
        break
  hits = []
  for t in sorted(times):
    burst = d.bernoulli(0.25)
    if burst:
      count, size, gap = d.integer(2, 4), d.loguniform(4.0, 14.0), d.integer(8, 15)
    else:
      count, size, gap = 1, d.loguniform(5.0, 30.0), 8
    hits.append(dict(kind="impulse", t_s=ticks(t), body="torso_link", impulse_ns=r6(size), burst_count=count,
                     burst_gap_s=r6(gap * DT), duration_s=DT, azimuth_rad=r6(d.uniform(-math.pi, math.pi)),
                     height_m=r6(d.uniform(-0.15, 0.25))))
  return hits


def throws(d: Draw, mission_s: float) -> list[dict]:
  if not d.bernoulli(0.25):
    return []
  kind = d.choice(("ball", "box"), (0.7, 0.3))
  return [dict(kind="throw", body=kind, t_s=ticks(d.uniform(SETTLE_S + 0.5, max(SETTLE_S + 0.6, mission_s - 2.0))),
               speed_mps=r6(d.uniform(5.0, 20.0) if kind == "ball" else d.uniform(2.0, 6.0)),
               azimuth_rad=r6(d.uniform(-math.pi, math.pi)), height_m=r6(d.uniform(-0.2, 0.3)))]


def sense(d: Draw, seed: int, full: bool) -> dict:
  band = d.choice(("N0", "N1", "mid"), (0.4, 0.4, 0.2))
  scale = 0.0 if band == "N0" else d.uniform(0.75, 1.25) if band == "N1" else d.uniform(0.2, 0.75)
  return dict(profile="uniform-n1shape-v1", scale=r6(scale),
              delay_steps=d.integer(0, 2) if full and d.bernoulli(0.3) else 0,
              dropout_p=r6(d.uniform(0.0, 0.05)) if full and d.bernoulli(0.2) else 0.0,
              noise_seed=child_seed(seed, "sense/noise"))


def body(d: Draw, full: bool) -> dict:
  if not full:
    return dict(mass_scale=1.0, payload_kg=0.0, friction_scale=1.0, kp_scale=1.0, kd_scale=1.0)
  return dict(mass_scale=r6(d.uniform(0.9, 1.1)), payload_kg=r6(d.uniform(0.0, 3.0) if d.bernoulli(0.3) else 0.0),
              friction_scale=r6(d.loguniform(0.4, 1.4)), kp_scale=r6(d.uniform(0.85, 1.15)),
              kd_scale=r6(d.uniform(0.85, 1.15)))


def scene(d: Draw, full: bool):
  if not full or not d.bernoulli(0.3):
    return None
  return dict(source="indoor-bank-v0", index=d.integer(0, 1 << 20))   # resolved against the converted scene bank


# -- assembly ------------------------------------------------------------------------------------------------------

def generator_record(family: str) -> dict:
  return dict(name=NAME, version=VERSION, family=family, source_sha256=source_sha256())


def finish(spec: dict) -> dict:
  spec["spec_sha256"] = spec_hash(spec)
  return spec


def generate(family: str, seed: int, *, path: str = "") -> dict:
  if family not in ("legacy-v0", "full-v0"):
    raise ValueError(f"unknown family {family}")
  full = family == "full-v0"
  d = Draw(seed)
  goal = mission(d, family)
  env = dict(terrain=terrain(d), scene=scene(d, full), obstacles=[],
             disturbances=impulses(d, goal["duration_s"]) + (throws(d, goal["duration_s"]) if full else []),
             sense=sense(d, seed, full), body=body(d, full))
  return finish(dict(schema="r1adv-scenario-v1", generator=generator_record(family), seed=int(seed), seed_path=path,
                     mission=goal, env=env, solvability_evidence=None))


def positive_controls(replicates: int = 4) -> list[dict]:
  """Fixed cells with known outcomes (11-16 general evaluation); identical setups across replicates except the
  sensor noise seed, so N0 replicates measure the simulator's run-to-run spread."""
  cells = {
    # the W protocol of 11-15/16 (300 steps forward 0.35 m/s, then 400 steps standing), placed on the slope
    "side10_stop": dict(kind="tilt", row=4, yaw=0.0, steps=[("Go", 0.35, 6.0), ("Hold", 8.0)], scale=0.0),
    "side20_stop": dict(kind="steep", row=3, yaw=0.0, steps=[("Go", 0.35, 6.0), ("Hold", 8.0)], scale=0.0),
    "up20_walk": dict(kind="steep", row=3, yaw=math.pi / 2, steps=[("Go", 0.35, 6.0), ("Hold", 8.0)], scale=0.0),
    "flat_walk_stop": dict(kind="flat", row=0, yaw=0.0, steps=[("Go", 0.35, 6.0), ("Hold", 8.0)], scale=0.0),
    "flat_stand_N1": dict(kind="flat", row=0, yaw=0.0, steps=[("Hold", 13.0)], scale=1.0),
    "flat_stand_N0": dict(kind="flat", row=0, yaw=0.0, steps=[("Hold", 13.0)], scale=0.0),
  }
  specs = []
  for name, cell in cells.items():
    for rep in range(replicates):
      seed = child_seed(round_seed("R00"), f"pc-v0/{name}/{rep}")
      steps = []
      for item in cell["steps"]:
        if item[0] == "Go":
          steps.append(dict(type="Go", mode="velocity", shape="straight", vx=item[1], vy=0.0, wz=0.0, duration_s=ticks(item[2])))
        else:
          steps.append(dict(type="Hold", duration_s=ticks(item[1])))
      goal = dict(grammar="mission-v1", settle_s=SETTLE_S, steps=steps,
                  duration_s=r6(SETTLE_S + sum(s["duration_s"] for s in steps)))
      value = bank.level(cell["kind"], cell["row"])
      env = dict(terrain=dict(bank=bank.NAME, bank_seed=BANK_SEED, bank_sha256=bank.bank_sha256(BANK_SEED),
                              kind=cell["kind"], row=cell["row"], level=list(value) if isinstance(value, tuple) else value,
                              slope_deg=r6(bank.slope_deg(cell["kind"], cell["row"])), spawn_yaw_rad=r6(cell["yaw"])),
                 scene=None, obstacles=[], disturbances=[],
                 sense=dict(profile="uniform-n1shape-v1", scale=cell["scale"], delay_steps=0, dropout_p=0.0,
                            noise_seed=child_seed(seed, "sense/noise")),
                 body=dict(mass_scale=1.0, payload_kg=0.0, friction_scale=1.0, kp_scale=1.0, kd_scale=1.0))
      specs.append(finish(dict(schema="r1adv-scenario-v1", generator=generator_record("pc-v0"), seed=seed,
                               seed_path=f"R00/pc-v0/{name}/{rep}", cell=name, mission=goal, env=env,
                               solvability_evidence=None)))
  return specs


def generate_set(family: str, count: int, round_name: str) -> list[dict]:
  if family == "pc-v0":
    return positive_controls()
  return [generate(family, instance_seed(round_name, family, i), path=f"{round_name}/{family}/{i}") for i in range(count)]


def manifest(specs: list[dict], family: str, count: int, round_name: str) -> dict:
  return dict(schema="r1adv-scenario-set-v1", generator=generator_record(family), round=round_name,
              round_seed=round_seed(round_name), family_seed=family_seed(round_name, family) if family != "pc-v0" else None,
              count=len(specs), requested=count, bank=bank.description(BANK_SEED), bank_sha256=bank.bank_sha256(BANK_SEED),
              set_sha256=sha256_text("\n".join(s["spec_sha256"] for s in specs)))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--family", default="legacy-v0", choices=("legacy-v0", "full-v0", "pc-v0"))
  parser.add_argument("--count", type=int, default=1000)
  parser.add_argument("--round", default="R00")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  specs = generate_set(args.family, args.count, args.round)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with args.out.open("x") as stream:
    for spec in specs:
      stream.write(canonical(spec) + "\n")
  info = manifest(specs, args.family, args.count, args.round)
  args.out.with_suffix(".manifest.json").write_text(json.dumps(info, indent=1))
  print(json.dumps({k: info[k] for k in ("count", "set_sha256", "bank_sha256")} | {"source": info["generator"]["source_sha256"]}))


if __name__ == "__main__":
  main()
