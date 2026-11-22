"""Paired motion amounts over an identical observation window, without time-wise pose matching.

These are descriptive measurements, not a score or a recovery/necessity test. Total variation and work can still
depend on the window boundaries; a negative excess can mean reduced task performance. Keep both raw amounts.
The caller must pair the same command/window and report falls, task performance and observation duration too.
"""
from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import numpy as np


GROUPS = {"legs": slice(0, 12), "waist": slice(12, 14), "arms": slice(14, 24), "all": slice(0, 24)}
MIN_AIR_S = 0.06


def _series(value, columns: int, name: str, samples: int | None = None) -> np.ndarray:
  result = np.asarray(value, dtype=np.float64)
  if result.ndim != 2 or result.shape[1] != columns or (samples is not None and len(result) != samples):
    raise ValueError(f"{name} must have shape ({'T' if samples is None else samples}, {columns})")
  if not np.isfinite(result).all():
    raise ValueError(f"{name} contains non-finite values")
  return result


def _amounts(actual: np.ndarray, twin: np.ndarray) -> dict:
  def amounts(values):
    return {**{name: float(values[index].sum()) for name, index in GROUPS.items()},
            "per_joint": values.tolist()}
  return {"actual": amounts(actual), "twin": amounts(twin), "excess": amounts(actual - twin)}


def _touchdowns(contact: np.ndarray, dt: float) -> np.ndarray:
  # Count only observed airborne samples: an unknown pre-window flight contributes nothing.
  air = np.zeros(2, dtype=np.int64)
  landed = np.zeros(2, dtype=np.int64)
  minimum = math.ceil(MIN_AIR_S / dt - 1.0e-12)
  for touching in contact:
    landed += touching & (air >= minimum)
    air = np.where(touching, 0, air + 1)
  return landed


def pair_summary(q, dq, contact, twin_q, twin_dq, twin_contact, dt: float,
                 torque=None, twin_torque=None) -> dict:
  """Measure equal-length [T,24] joint series and [T,2] binary foot contacts at spacing ``dt``.

  ``duration_s`` is (T-1)*dt. Work integrates sample-wise |torque*dq| by the trapezoid rule over those intervals,
  and is omitted (None) unless both torque series are supplied. A touchdown needs at least 0.06 s of observed
  airborne samples, so contact chatter and unknown pre-window flight are excluded. Contacts are sample-based.
  Fewer than two samples return valid=False and None metrics, not evidence of zero motion. valid=True only
  means the time window can be measured; it does not mean the robot stayed upright or performed the command.
  """
  if not math.isfinite(dt) or dt <= 0:
    raise ValueError("dt must be finite and positive")
  q = _series(q, 24, "q")
  samples = len(q)
  dq = _series(dq, 24, "dq", samples)
  twin_q = _series(twin_q, 24, "twin_q", samples)
  twin_dq = _series(twin_dq, 24, "twin_dq", samples)
  contact = _series(contact, 2, "contact", samples)
  twin_contact = _series(twin_contact, 2, "twin_contact", samples)
  if not (((contact == 0) | (contact == 1)).all() and ((twin_contact == 0) | (twin_contact == 1)).all()):
    raise ValueError("contacts must be binary (0/1 or bool)")
  if (torque is None) != (twin_torque is None):
    raise ValueError("supply both torque and twin_torque, or neither")
  if torque is not None:
    torque = _series(torque, 24, "torque", samples)
    twin_torque = _series(twin_torque, 24, "twin_torque", samples)
  result = {"samples": samples, "duration_s": max(samples - 1, 0) * float(dt), "valid": samples >= 2,
            "joint_variation_rad": None, "mechanical_work_j": None, "touchdowns": None}
  if samples < 2:
    return result
  result["joint_variation_rad"] = _amounts(np.abs(np.diff(q, axis=0)).sum(axis=0),
                                             np.abs(np.diff(twin_q, axis=0)).sum(axis=0))
  if torque is not None:
    def work(force, velocity):
      power = np.abs(force * velocity)
      return ((power[:-1] + power[1:]) * (0.5 * dt)).sum(axis=0)
    result["mechanical_work_j"] = _amounts(work(torque, dq), work(twin_torque, twin_dq))
  actual = _touchdowns(contact.astype(bool), dt)
  twin = _touchdowns(twin_contact.astype(bool), dt)
  result["touchdowns"] = {
    "minimum_air_s": MIN_AIR_S,
    **{name: {"left": int(values[0]), "right": int(values[1]), "all": int(values.sum())}
       for name, values in (("actual", actual), ("twin", twin), ("excess", actual - twin))},
  }
  return result


def fixed_horizon_summary(series: dict, dt: float, *, physical_fall=False, body_names=()) -> dict:
  """Describe one pre-action trajectory; alive_before is an irreversible prefix mask.

  No recovery threshold or combined success score is invented. Incomplete survival
  is visible beside every amount; no extrapolation fills the remaining horizon.
  Last-two-second summaries use the same fixed nominal window for every policy.
  Actuator work is sampled |tau*dq|, not motor electrical energy or contact impulse.
  """
  if not math.isfinite(dt) or dt <= 0:
    raise ValueError("dt must be positive and finite")
  q = np.asarray(series["joint_pos"], dtype=np.float64)
  if q.ndim != 2 or q.shape[1] != 24:
    raise ValueError("joint_pos must be [T,24]")
  samples = len(q)
  alive = np.asarray(series["alive_before"], dtype=bool)
  if alive.shape != (samples,) or np.any(alive[1:] & ~alive[:-1]):
    raise ValueError("alive_before must be a [T] prefix; reset episodes cannot re-enter metrics")
  data = {key: np.asarray(value) for key, value in series.items()}
  if any(len(value) != samples for value in data.values()):
    raise ValueError("All trajectory fields must share the fixed frame axis")

  def groups(values):
    return {**{key: float(values[index].sum()) for key, index in GROUPS.items()}, "per_joint": values.tolist()}

  def window(start, stop):
    selected = alive[start:stop]
    frame = {key: np.asarray(value[start:stop], dtype=np.float64) for key, value in data.items() if key != "alive_before"}
    valid_frames = int(selected.sum())
    intervals = selected[1:] & selected[:-1]
    n_intervals = int(intervals.sum())
    report = dict(nominal_start_s=start*dt, nominal_stop_s=stop*dt, nominal_frames=stop-start,
                  alive_frames=valid_frames, observed_sample_intervals=n_intervals,
                  observed_interval_duration_s=n_intervals*dt,
                  complete_survival=bool(selected.all()) and not physical_fall,
                  measurement_valid=False, nonfinite_in_alive_frames=0)
    for values in frame.values():
      report["nonfinite_in_alive_frames"] += int((~np.isfinite(values[selected])).sum())
    if n_intervals == 0 or report["nonfinite_in_alive_frames"]:
      return report
    report["measurement_valid"] = True
    duration = n_intervals * dt

    def integrate(values):
      return ((values[:-1][intervals] + values[1:][intervals]) * (.5 * dt)).sum(axis=0)

    def rms(values):
      return np.sqrt(integrate(values**2) / duration)

    velocity = frame["joint_vel"]
    report["joint_variation_rad"] = groups(np.abs(np.diff(frame["joint_pos"], axis=0))[intervals].sum(axis=0))
    report["joint_speed_integral_rad"] = groups(integrate(np.abs(velocity)))
    report["joint_speed_squared_integral_rad2_s"] = groups(integrate(velocity**2))
    report["joint_speed_rms_rad_s_per_joint"] = rms(velocity).tolist()
    report["mechanical_work_j"] = groups(integrate(np.abs(frame["actuator_torque"] * velocity)))
    report["actuator_torque_abs_integral_nm_s"] = groups(integrate(np.abs(frame["actuator_torque"])))
    root = frame["root_pos_w"]
    report["root_path_m"] = float(np.linalg.norm(np.diff(root, axis=0), axis=1)[intervals].sum())
    indices = np.flatnonzero(selected)
    report["root_displacement_xyz_m"] = (root[indices[-1]] - root[indices[0]]).tolist()
    body = frame["body_pos_w"]
    body_path = np.linalg.norm(np.diff(body, axis=0), axis=-1)[intervals].sum(axis=0)
    relative_body = body - root[:, None]
    relative_path = np.linalg.norm(np.diff(relative_body, axis=0), axis=-1)[intervals].sum(axis=0)
    names = tuple(body_names) or tuple(str(i) for i in range(body.shape[1]))
    if len(names) != body.shape[1]:
      raise ValueError("body_names does not match body_pos_w")
    report["body_path_world_m"] = dict(zip(names, body_path.tolist(), strict=True))
    report["body_path_root_translated_m"] = dict(zip(names, relative_path.tolist(), strict=True))
    report["body_path_note"] = "actual link paths; root_translated removes translation, not base rotation; locomotion/phase remain relevant"
    gyro = frame["root_ang_vel_b"]
    tilt = np.arccos(np.clip(-frame["projected_gravity_b"][:, 2], -1., 1.))
    report["pelvis_tilt_integral_rad_s"] = float(integrate(tilt))
    report["pelvis_tilt_rms_rad"] = float(rms(tilt))
    report["pelvis_angular_velocity_rms_rad_s_xyz"] = rms(gyro).tolist()
    average_gyro = integrate(gyro) / duration
    report["pelvis_angular_velocity_centered_rms_rad_s_xyz"] = rms(gyro-average_gyro).tolist()
    if "body_ang_vel_w" in frame:
      report["body_angular_velocity_rms_rad_s"] = dict(zip(names, rms(np.linalg.norm(frame["body_ang_vel_w"], axis=-1)).tolist(), strict=True))
    if "requested_twist" in frame:
      actual = np.concatenate((frame["root_lin_vel_b"][:, :2], gyro[:, 2:3]), axis=1)
      error = actual-frame["requested_twist"]
      report["raw_command_error_abs_integral"] = integrate(np.abs(error)).tolist()
      report["raw_command_error_rms"] = rms(error).tolist()
      report["raw_command_error_units"] = ["vx:m/s", "vy:m/s", "yaw:rad/s"]
      report["command_pass"] = None
      report["command_pass_note"] = "frozen external command criteria must judge task completion; small motion is not success"
    if "foot_contact" in frame:
      report["touchdowns"] = _touchdowns(frame["foot_contact"][selected].astype(bool), dt).tolist()
    return report

  return dict(fixed_horizon_frames=samples, fixed_horizon_s=samples*dt,
              unique_physical_fall=bool(physical_fall),
              full=window(0, samples), last_2s=window(max(0, samples-round(2./dt)), samples),
              interpretation="descriptive amounts only; compare motion with survival and command completion; no necessity/recovery verdict",
              recovery_state="continuous tilt/angular velocity/command error and fixed final 2s; no new threshold or post-hit grace",
              limitations=["sampled pre-action states omit terminal pre-reset physics intervals; first fall outcome is retained",
                           "finite-window paths/variation and twin differences retain gait-phase and boundary effects",
                           "partial episodes are not zero-filled or extrapolated, and cannot earn an efficiency win"])


def summarize_actor_trace(manifest_path) -> dict:
  """Read only metric arrays from each committed chunk; verify all chunk bytes and time coverage."""
  path = Path(manifest_path).resolve()
  manifest = json.loads(path.read_text())
  if manifest.get("schema") != "r1wb-actor-input-trace-v1" or manifest.get("state") != "complete":
    raise ValueError("A complete actor input trace is required")
  keys = ("joint_pos", "joint_vel", "actuator_torque", "alive_before", "body_pos_w", "root_pos_w",
          "root_lin_vel_b", "root_ang_vel_b", "projected_gravity_b", "requested_twist", "foot_contact", "body_ang_vel_w")
  records = {key: [] for key in keys}
  expected_step = 0

  def artifact(item):
    file = (path.parent / item["path"]).resolve()
    if not file.is_relative_to(path.parent) or not file.is_file():
      raise ValueError("Trace artifact is missing or escapes its archive")
    with file.open("rb") as stream:
      digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
      raise ValueError("Trace artifact SHA mismatch")
    return file

  for chunk in manifest["chunks"]:
    with np.load(artifact(chunk), allow_pickle=False) as values:
      steps = values["step"]
      if not np.array_equal(steps, np.arange(expected_step, expected_step+len(steps))):
        raise ValueError("Trace frame gap/repetition")
      expected_step += len(steps)
      for key in keys:
        if key in values:
          records[key].append(values[key])
  if expected_step != manifest["frames"] or expected_step != manifest["expected_steps"]:
    raise ValueError("Trace frame count mismatch")
  with np.load(artifact(manifest["outcomes"]), allow_pickle=False) as outcome:
    if not np.array_equal(outcome["step"], np.arange(expected_step)):
      raise ValueError("Outcome frame gap/repetition")
    falls = outcome["first_physical_fall"].sum(axis=0)
    if np.any(falls > 1):
      raise ValueError("Auto-reset episode was counted as a second physical fall")
  values = {key: np.concatenate(parts) for key, parts in records.items() if parts}
  if any(len(value) != expected_step for value in values.values()):
    raise ValueError("Metric field omitted from a trace chunk")
  metadata = manifest["metadata"]
  horizons = metadata.get("episode_control_steps", [expected_step] * values["joint_pos"].shape[1])
  if len(horizons) != values["joint_pos"].shape[1] or any(type(h) is not int or not 1 <= h <= expected_step for h in horizons):
    raise ValueError("Invalid frozen per-environment horizon")
  rows = [fixed_horizon_summary({key: value[:horizons[env], env] for key, value in values.items()}, metadata["dt_s"],
                                physical_fall=bool(falls[env]), body_names=metadata["body_names"])
          for env in range(values["joint_pos"].shape[1])]
  return dict(schema="r1wb-fixed-horizon-motion-v1", trace_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              frames=expected_step, all_environments=True, environments=rows,
              source_scope="all actual actor control frames and immutable ground-truth reads; not extrapolated from a chosen subset")
