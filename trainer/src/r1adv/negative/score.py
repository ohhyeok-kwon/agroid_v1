"""Charter v1 scores per instance from runner arrays: F (physical fall), G (normalised goal error, time mean), J (body
joint jerk, time median per mission step, against the manufacturer on the same instance and step where it actually
did that step, else the manufacturer's class median; floor 20 rad/s^3). No composite score; complete success =
no F, G <= TAU_G, J deficit <= 1. Thresholds marked provisional are R0 choices, not charter values.

G per tick of a step (the command's own frame, re-anchored at each step like r1wb general_eval_v2):
  e = max(|p - p_cmd| / (0.25 m + 0.1 * commanded path so far), |yaw - yaw_cmd| / (0.25 rad + 0.1 * commanded turn),
          |pelvis height - target| / 0.05 m  [height steps, after their first second])
usage: python -m r1adv.negative.score --specs specs.jsonl --runs <dir with <policy>/arrays.npz> --out rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from r1adv.negative import bank as B

DT = 0.02
TAU_G = 1.0                       # provisional (charter gives no G tolerance yet)
J_FLOOR = 20.0                    # rad/s^3, charter's provisional absolute floor
J_FACTOR = 1.25
R_POS, R_YAW, R_H = 0.25, 0.25, 0.05
MIN_TICKS = 25
FLAT = ("flat", "rough", "gravel")
STAIRS = ("stairs_up", "stairs_down")


def terrain_group(kind: str) -> str:
  return "flat" if kind in FLAT else "stairs" if kind in STAIRS else "slope"


def step_class(step: dict) -> str:
  if step["type"] == "Go":
    return "turn" if math.hypot(step["vx"], step["vy"]) < 1e-6 else "walk"
  return step["type"].lower()


def wrap(angle):
  return (angle + np.pi) % (2 * np.pi) - np.pi


def measure(spec: dict, a: dict, i: int) -> dict:
  """Per-step quantities of instance i in one policy's arrays (no reference needed)."""
  steps = spec["mission"]["steps"]
  terrain = spec["env"]["terrain"]
  fell, end = int(a["fell_step"][i]), int(a["end_step"][i])
  falls = 0 <= fell < end
  stop = fell if falls else end
  xy, yaw = a["xy"][i].astype(np.float64), np.unwrap(a["yaw"][i, :, 0].astype(np.float64))
  height, jerk = a["height"][i, :, 0], a["jerk"][i]
  t = round(spec["mission"]["settle_s"] / DT)
  out, errors, previous_pose = [], [], False
  for k, step in enumerate(steps):
    n = round(step["duration_s"] / DT)
    begin, finish = t, t + n
    t = finish
    u = min(finish, stop)
    rec = dict(k=k, type=step["type"], cls=step_class(step), ticks=n, scored=max(0, u - begin), fell_in=bool(falls and begin <= fell < finish))
    pose_on = step["type"] == "Pose" or previous_pose
    previous_pose = step["type"] == "Pose"
    if u - begin < 2:
      out.append(rec)
      continue
    ts = np.arange(begin, u)
    vx, vy, wz = (step.get("vx", 0.0), step.get("vy", 0.0), step.get("wz", 0.0))
    psi = yaw[begin:u]
    c, s = np.cos(psi), np.sin(psi)
    vw = np.stack([c * vx - s * vy, s * vx + c * vy], 1)
    p_cmd = xy[begin] + np.concatenate([np.zeros((1, 2)), np.cumsum(vw[:-1] * DT, 0)])
    elapsed = (ts - begin) * DT
    e_p = np.linalg.norm(xy[begin:u] - p_cmd, axis=1) / (R_POS + 0.1 * math.hypot(vx, vy) * elapsed)
    e_y = np.abs(wrap(psi - (yaw[begin] + wz * elapsed))) / (R_YAW + 0.1 * abs(wz) * elapsed)
    e = np.maximum(e_p, e_y)
    if pose_on:
      target = step["pelvis_height_m"] if step["type"] == "Pose" else 0.731
      late = elapsed >= 1.0
      e_h = np.where(late, np.abs(height[begin:u] - target) / R_H, 0.0)
      e = np.maximum(e, e_h)
      rec["height_err_m"] = float(np.abs(height[begin:u][late] - target).mean()) if late.any() else None
      hold_part = elapsed >= 0.5 * step["duration_s"]
      rec["height_err_median_hold"] = float(np.median(np.abs(height[begin:u][hold_part] - target))) if hold_part.any() else None
    errors.append(e)
    rec["G"] = float(e.mean())
    rec["path_err_m"] = float(np.linalg.norm(xy[begin:u] - p_cmd, axis=1).mean())
    rec["yaw_err_rad"] = float(np.abs(wrap(psi - (yaw[begin] + wz * elapsed))).mean())
    if rec["scored"] >= MIN_TICKS:
      j = jerk[begin + 3:u] if u - begin > 6 else jerk[begin:u]
      rec.update(J=float(np.median(j[:, 0])), J_legs=float(np.median(j[:, 1])), J_waist=float(np.median(j[:, 2])),
                 J_arms=float(np.median(j[:, 3])))
    speed = math.hypot(vx, vy)
    rec["cmd_distance_m"] = speed * step["duration_s"]
    if speed > 1e-6:
      direction = np.array([math.cos(yaw[begin]) * vx - math.sin(yaw[begin]) * vy,
                            math.sin(yaw[begin]) * vx + math.cos(yaw[begin]) * vy]) / speed
      rec["progress_ratio"] = float((xy[u - 1] - xy[begin]) @ direction / (speed * (u - begin) * DT))
      rec["travel_dir_world"] = direction.tolist()
    if abs(wz) > 1e-6:
      rec["turn_ratio"] = float((yaw[u - 1] - yaw[begin]) / (wz * (u - begin) * DT))
    radius = np.linalg.norm(xy[begin:u], axis=1)
    rec["r_max"], rec["r_med"] = float(radius.max()), float(np.median(radius))
    rec["out_of_patch"] = bool(np.abs(xy[begin:u]).max() > B.PATCH / 2 - 0.5)
    if terrain["kind"] in B.SLOPED:
      mid = (begin + u) // 2
      angle, uphill = B.slope_at(terrain["kind"], terrain["row"], float(xy[mid, 0]), float(xy[mid, 1]))
      rec["slope_deg_here"] = math.degrees(angle)
      heading = np.array([math.cos(yaw[mid]), math.sin(yaw[mid])])
      rec["heading_uphill_cos"] = float(heading @ np.asarray(uphill)) if angle > 0 else 0.0
      if "travel_dir_world" in rec:
        rec["travel_uphill_cos"] = float(np.asarray(rec["travel_dir_world"]) @ np.asarray(uphill)) if angle > 0 else 0.0
      # the slope the robot meets along the step: steepest position reached
      far = int(np.argmax(radius)) + begin
      angle_far, uphill_far = B.slope_at(terrain["kind"], terrain["row"], float(xy[far, 0]), float(xy[far, 1]))
      rec["slope_deg_max"] = math.degrees(angle_far)
      if "travel_dir_world" in rec and angle_far > 0:
        rec["travel_uphill_cos_far"] = float(np.asarray(rec["travel_dir_world"]) @ np.asarray(uphill_far))
    out.append(rec)
  scored = np.concatenate(errors) if errors else np.zeros(0)
  return dict(F=bool(falls), fell_step=fell if falls else None, reason=int(a["reason"][i]) if falls else 0,
              G=float(scored.mean()) if scored.size else None,
              J_med=float(np.median(jerk[round(spec["mission"]["settle_s"] / DT) + 3:stop, 0])) if stop - round(spec["mission"]["settle_s"] / DT) > 6 else None,
              tilt_max=float(a["tilt"][i, :stop, 0].max()) if stop > 0 else None,
              force_max=float(a["force"][i, :stop, 0].max()) if stop > 0 else None,
              steps=out)


def performed(rec: dict) -> bool:
  """The manufacturer actually did this step (a valid same-instance J reference)."""
  if rec.get("fell_in") or rec.get("J") is None:
    return False
  if rec["cls"] == "walk":
    return rec.get("progress_ratio", 0.0) >= 0.5
  if rec["cls"] == "turn":
    return rec.get("turn_ratio", 0.0) >= 0.5
  if rec["cls"] == "pose":
    return False                                   # the manufacturer has no height input
  return True


def references(specs: list[dict], manufacturer: list[dict]) -> dict:
  """Class medians of the manufacturer's performed steps: (class, terrain group, noisy?)."""
  pools = {}
  for spec, m in zip(specs, manufacturer):
    noisy = spec["env"]["sense"]["scale"] > 0
    group = terrain_group(spec["env"]["terrain"]["kind"])
    for rec in m["steps"]:
      if performed(rec):
        pools.setdefault((rec["cls"], group, noisy), []).append(rec["J"])
  return {key: float(np.median(v)) for key, v in pools.items()}


def judge(specs: list[dict], measured: dict[str, list[dict]]) -> dict[str, list[dict]]:
  """Adds J references, deficits and complete success to every policy's measurements."""
  man = measured.get("manufacturer")
  classes = references(specs, man) if man is not None else {}
  for policy, rows in measured.items():
    for i, (spec, row) in enumerate(zip(specs, rows)):
      noisy = spec["env"]["sense"]["scale"] > 0
      group = terrain_group(spec["env"]["terrain"]["kind"])
      deficits = []
      for rec in row["steps"]:
        if rec.get("J") is None:
          continue
        ref, source = None, None
        if man is not None:
          mrec = man[i]["steps"][rec["k"]]
          if performed(mrec):
            ref, source = mrec["J"], "same"
        if ref is None:
          cls = "hold" if rec["cls"] == "pose" else rec["cls"]
          ref = classes.get((cls, group, noisy)) or classes.get((cls, "flat", noisy))
          source = "class" if ref is not None else "floor"
        allowed = max(J_FACTOR * ref, J_FLOOR) if ref is not None else J_FLOOR
        rec.update(J_ref=ref, J_ref_source=source, J_deficit=rec["J"] / allowed)
        deficits.append(rec["J_deficit"])
      row["J_deficit"] = max(deficits) if deficits else None
      row["complete"] = bool(not row["F"] and row["G"] is not None and row["G"] <= TAU_G
                             and row["J_deficit"] is not None and row["J_deficit"] <= 1.0)
  return measured


def predicates(spec: dict, row: dict) -> dict:
  """Known-defect conditions of the pre-registration (status.md 01:00), from the spec and the trajectory."""
  terrain, sense = spec["env"]["terrain"], spec["env"]["sense"]
  steps = spec["mission"]["steps"]
  side = up = False
  for step, rec in zip(steps, row["steps"]):
    if terrain["kind"] not in B.SLOPED or rec.get("scored", 0) < MIN_TICKS:
      continue
    # sloped patches are planes through the start (bank v0.1): the robot is on the slope from the first step
    if (step["type"] == "Hold" and step["duration_s"] >= 3.0 and rec.get("slope_deg_here", 0.0) >= 8.0
        and abs(rec.get("heading_uphill_cos", 1.0)) <= math.sqrt(0.5)):
      side = True
    if (step["type"] == "Go" and math.hypot(step["vx"], step["vy"]) >= 0.2 and step["duration_s"] >= 3.0
        and B.slope_deg(terrain["kind"], terrain["row"]) >= 15.0 and rec.get("travel_uphill_cos", -1.0) >= math.sqrt(0.5)):
      up = True
  hold_s = sum(s["duration_s"] for s in steps if s["type"] == "Hold")
  return dict(P_side=side, P_up=up, P_noise=bool(sense["scale"] >= 0.75 and hold_s >= 3.0),
              P_noise_flat=bool(sense["scale"] >= 0.75 and hold_s >= 3.0 and terrain["kind"] == "flat"))


def load(run_dir: Path, specs: list[dict]) -> dict:
  with np.load(run_dir / "arrays.npz", allow_pickle=False) as z:
    a = {k: z[k] for k in z.files}
  if list(a["spec_sha256"]) != [s["spec_sha256"] for s in specs]:
    raise ValueError(f"{run_dir}: instance order or specs differ")
  return a


def score_runs(specs: list[dict], runs: dict[str, Path]) -> dict[str, list[dict]]:
  measured = {}
  for policy, path in runs.items():
    a = load(path, specs)
    measured[policy] = [measure(spec, a, i) for i, spec in enumerate(specs)]
  return judge(specs, measured)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--specs", type=Path, required=True)
  parser.add_argument("--runs", type=Path, required=True, help="directory holding <policy>/arrays.npz")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  specs = [json.loads(line) for line in args.specs.read_text().splitlines() if line.strip()]
  runs = {p.name: p for p in sorted(args.runs.iterdir()) if (p / "arrays.npz").exists()}
  scored = score_runs(specs, runs)
  with args.out.open("w") as stream:
    for policy, rows in scored.items():
      for spec, row in zip(specs, rows):
        stream.write(json.dumps(dict(policy=policy, seed_path=spec["seed_path"], spec_sha256=spec["spec_sha256"],
                                     cell=spec.get("cell"), **predicates(spec, row), **row)) + "\n")
  print(f"wrote {sum(len(r) for r in scored.values())} rows for {list(scored)}")


if __name__ == "__main__":
  main()


# -- criteria r1 (coordinator decisions 02:45): no manufacturer value anywhere on the training side -------------------

G_TOL_R1 = dict(hold_m=0.10, hold_rad=0.10, go_m=0.10, go_per_m=0.10, go_rad=0.15, pose_m=0.03)


def j_type(steps: list[dict], k: int) -> str:
  step, previous = steps[k], (steps[k - 1] if k else None)
  if step["type"] == "Pose" or (step["type"] == "Hold" and previous is not None and previous["type"] == "Pose"):
    return "crouch"
  if step["type"] == "Hold":
    return "hold"
  v = math.hypot(step["vx"], step["vy"])
  if v < 0.05:
    return "turn"
  for lo, hi in ((0.05, 0.25), (0.25, 0.45), (0.45, 0.8), (0.8, 3.0)):
    if lo <= v < hi:
      return f"go_{lo:.2f}-{hi:.2f}"
  return "go_0.80-3.00"


def judge_r1(specs: list[dict], measured: dict[str, list[dict]], human: dict) -> dict[str, list[dict]]:
  """Complete success r1: no fall, every step within its fixed G tolerance and J <= max(1.25 J_human[type], 20).
  J_human is the locked table (human_j.py); a context without a human value (terrain, noise, hit, latency) uses its
  type's flat noise-free value. Adds per step G_ok, J_allowed, J_ok and per instance G_ratio, J_ratio, complete."""
  types = human["types"]
  for rows in measured.values():
    for spec, row in zip(specs, rows):
      steps = spec["mission"]["steps"]
      g_ratios, j_ratios, ok = [], [], True
      for k, rec in enumerate(row["steps"]):
        if rec.get("scored", 0) < 2:
          ok = False
          continue
        step = steps[k]
        if step["type"] == "Hold":
          ratio = max(rec["path_err_m"] / G_TOL_R1["hold_m"], rec["yaw_err_rad"] / G_TOL_R1["hold_rad"])
        elif step["type"] == "Go":
          ratio = max(rec["path_err_m"] / (G_TOL_R1["go_m"] + G_TOL_R1["go_per_m"] * rec["cmd_distance_m"]),
                      rec["yaw_err_rad"] / G_TOL_R1["go_rad"])
        else:
          err = rec.get("height_err_median_hold")
          ratio = err / G_TOL_R1["pose_m"] if err is not None else float("inf")
        rec["G_ratio_r1"] = ratio
        g_ratios.append(ratio)
        if rec.get("J") is not None:
          kind = j_type(steps, k)
          # the locked table (R01_j_locked_v1) carries J_ref itself; a draft table only p90
          allowed = types[kind]["J_ref"] if "J_ref" in types[kind] and human.get("status", "").startswith("LOCKED") \
            else max(1.25 * types[kind]["p90"], 20.0)
          rec.update(J_type=kind, J_allowed=allowed, J_ratio_r1=rec["J"] / allowed)
          j_ratios.append(rec["J"] / allowed)
        ok &= ratio <= 1.0 and (rec.get("J_ratio_r1", 0.0) <= 1.0)
      row["G_ratio"] = max(g_ratios) if g_ratios else None
      row["J_ratio"] = max(j_ratios) if j_ratios else None
      row["complete_r1"] = bool(ok and not row["F"] and g_ratios)
  return measured


def score_runs_r1(specs: list[dict], runs: dict[str, Path], human: dict) -> dict[str, list[dict]]:
  measured = {}
  for policy, path in runs.items():
    a = load(path, specs)
    measured[policy] = [measure(spec, a, i) for i, spec in enumerate(specs)]
  return judge_r1(specs, measured, human)
