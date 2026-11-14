"""Auditable physical quietness proxies and fixed-case stability scores.

These deterministic summaries are not acoustic dB or evidence of statistical
superiority. The disclosed weights are engineering choices, not learned facts.
Only Python's standard library is needed; no simulation or inference is run.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


QUIET_METRICS = {
  "touchdown_speed_mps": {"unit": "m/s", "weight": 0.25, "aggregation": "mean of both feet"},
  "foot_normal_force_p95_bw": {"unit": "body weight", "weight": 0.25, "aggregation": "mean of both feet"},
  "positive_loading_rate_p95_bw_s": {"unit": "body weight/s", "weight": 0.25, "aggregation": "mean of both feet"},
  "joint_acceleration_rms_rad_s2": {"unit": "rad/s^2", "weight": 0.25, "aggregation": "reported joint RMS"},
}
STABILITY_WEIGHTS = {"survival": 0.70, "fixed_horizon_recovery": 0.20, "tilt": 0.10}
CASE_FIELDS = ("moving", "speed", "at", "angle", "body", "shots", "interval", "height_offset")


def _finite(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe(value: Any) -> Any:
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, Mapping):
    return {key: _safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_safe(item) for item in value]
  return value


def _mean(values: Sequence[float]) -> float:
  return math.fsum(values) / len(values)


def _quiet_index(log_ratio: float) -> float:
  # Stable form of 100 / (1 + exp(log_ratio)); baseline ratio 1 gives 50.
  return 50.0 * (1.0 - math.tanh(log_ratio / 2.0))


def quiet_pair(candidate: Mapping, manufacturer: Mapping, *, target_speed_mps: float,
               speed_tolerance_mps: float = 0.02, minimum_time_s: float = 10.0) -> dict:
  """Compare measured rows at one actual speed; missing/failed rows score null."""
  if (not _finite(target_speed_mps) or target_speed_mps <= 0
      or not _finite(speed_tolerance_mps) or speed_tolerance_mps <= 0
      or not _finite(minimum_time_s) or minimum_time_s <= 0):
    raise ValueError("Quiet scoring requires positive finite speed, tolerance, and duration")
  reasons, measured = [], {}
  for label, row in (("candidate", candidate), ("manufacturer", manufacturer)):
    measured[label] = {}
    if row.get("fallen") is not False:
      reasons.append(f"{label}: fallen or missing fall status")
    if row.get("invalid_state") is not False:
      reasons.append(f"{label}: invalid or missing state status")
    duration = row.get("valid_time_s")
    if not _finite(duration) or duration < minimum_time_s - 0.001:
      reasons.append(f"{label}: incomplete measured duration")
    speed = row.get("realized_forward_mps")
    if not _finite(speed) or abs(speed - target_speed_mps) > speed_tolerance_mps + 1e-9:
      reasons.append(f"{label}: actual speed is outside target tolerance")
    counts = row.get("touchdown_count")
    if (not isinstance(counts, (list, tuple)) or len(counts) != 2
        or any(not _finite(count) or count < 2 or int(count) != count for count in counts)):
      reasons.append(f"{label}: fewer than two measured touchdowns per foot")
    for key in QUIET_METRICS:
      value = row.get(key)
      values = [value] if key == "joint_acceleration_rms_rad_s2" else value
      expected = 1 if key == "joint_acceleration_rms_rad_s2" else 2
      if (not isinstance(values, (list, tuple)) or len(values) != expected
          or any(not _finite(item) or item <= 0 for item in values)):
        reasons.append(f"{label}: missing or nonpositive {key}")
      else:
        measured[label][key] = _mean(values)
  speeds = [row.get("realized_forward_mps") for row in (candidate, manufacturer)]
  gap = abs(speeds[0] - speeds[1]) if all(_finite(speed) for speed in speeds) else None
  if gap is not None and gap > speed_tolerance_mps + 1e-9:
    reasons.append("candidate/manufacturer actual speed gap exceeds tolerance")
  ratios = {key: measured["candidate"][key] / measured["manufacturer"][key]
            for key in QUIET_METRICS if key in measured["candidate"] and key in measured["manufacturer"]}
  log_ratio = math.fsum(QUIET_METRICS[key]["weight"] * (math.log(measured["candidate"][key])
                       - math.log(measured["manufacturer"][key])) for key in QUIET_METRICS) if not reasons else None
  return _safe({"valid": not reasons, "score": _quiet_index(log_ratio) if log_ratio is not None else None,
          "manufacturer_score": 50.0 if not reasons else None, "reasons": reasons,
          "target_speed_mps": target_speed_mps, "realized_speed_gap_mps": gap,
          "weighted_log_ratio": log_ratio, "ratios": ratios, "aggregated_measurements": measured,
          "raw": {"candidate": dict(candidate), "manufacturer": dict(manufacturer)}})


def score_gait_report(report: Mapping, *, candidate: str = "candidate",
                      targets_mps: Sequence[float] = (0.2, 0.3),
                      speed_tolerance_mps: float = 0.02, issues: Sequence[str] = ()) -> dict:
  """Use recorded matched-case indices, verifying raw values again; never interpolate."""
  if not targets_mps or any(not _finite(target) or target <= 0 for target in targets_mps):
    raise ValueError("Quiet targets must be positive and finite")
  if len(set(targets_mps)) != len(targets_mps):
    raise ValueError("Quiet targets must be unique")
  problems = list(issues)
  if report.get("protocol_version") != 2:
    problems.append("Requires gait protocol_version=2")
  contract = report.get("control_contract_equal", {})
  if not contract or not all(value is True for value in contract.values()):
    problems.append("Manufacturer/candidate control contract is missing or differs")
  seconds, warmup = report.get("seconds"), report.get("warmup")
  duration = seconds - warmup if _finite(seconds) and _finite(warmup) and seconds > warmup >= 0 else None
  if duration is None:
    problems.append("Missing gait measurement duration")
  rows = report.get("results", {})
  candidate_rows = rows.get(candidate, {}).get("cases", [])
  reference_rows = rows.get("manufacturer", {}).get("cases", [])
  pairs = []
  for target in targets_mps:
    matching = [row for row in report.get("speed_comparisons", [])
                if row.get("candidate") == candidate and row.get("reference") == "manufacturer"
                and _finite(row.get("target_mps")) and abs(row["target_mps"] - target) < 1e-9]
    selection = matching[0] if len(matching) == 1 else {}
    ci, ri = selection.get("candidate_case"), selection.get("reference_case")
    valid_indices = (selection.get("matched") is True and type(ci) is int and type(ri) is int
                     and 0 <= ci < len(candidate_rows) and 0 <= ri < len(reference_rows))
    if valid_indices:
      pair = quiet_pair(candidate_rows[ci], reference_rows[ri], target_speed_mps=target,
                        speed_tolerance_mps=speed_tolerance_mps, minimum_time_s=max(10., duration or 10.))
    else:
      pair = {"valid": False, "score": None, "manufacturer_score": None,
              "target_speed_mps": target, "weighted_log_ratio": None,
              "reasons": ["No unique recorded measured speed-matched pair"], "raw": None}
    pair.update(candidate_case=ci, manufacturer_case=ri)
    pair["reasons"].extend(problems)
    if pair["reasons"]:
      pair.update(valid=False, score=None, manufacturer_score=None)
    pairs.append(pair)
  valid = bool(pairs) and all(pair["valid"] for pair in pairs)
  return {"valid": valid, "score": _quiet_index(_mean([pair["weighted_log_ratio"] for pair in pairs])) if valid else None,
          "manufacturer_score": 50.0 if valid else None, "pairs": pairs,
          "coverage": {"requested_targets": len(pairs), "scored_targets": sum(pair["valid"] for pair in pairs),
                       "candidate_rows": len(candidate_rows),
                       "candidate_fallen_rows": sum(row.get("fallen") is True for row in candidate_rows)},
          "speed_tolerance_mps": speed_tolerance_mps, "minimum_measured_time_s": max(10., duration or 10.),
          "formula": "Q=100/(1+exp(mean_target(sum_metric(weight*ln(candidate/manufacturer)))))",
          "metrics": {key: dict(value) for key, value in QUIET_METRICS.items()},
          "weights_are_subjective": True, "is_acoustic_db": False}


def _case_key(row: Mapping) -> tuple | None:
  if any(key not in row for key in CASE_FIELDS):
    return None
  if type(row["moving"]) is not bool or not isinstance(row["body"], str):
    return None
  if any(not _finite(row[key]) for key in CASE_FIELDS if key not in ("moving", "body")):
    return None
  return tuple(round(row[key], 9) if _finite(row[key]) else row[key] for key in CASE_FIELDS)


def stability_summary(rows: Sequence[Mapping], *, expected_cases: int | None = None,
                      recovery_horizon_s: float = 3.0, tilt_limit_deg: float = 30.0) -> dict:
  """Score protocol-3 cases; failures/unrecovered trials receive the full horizon.

  The common recovery origin is the final scheduled shot, not each policy's
  variable final contact. Protocol 3 observes until first-shot time + 4 s.
  """
  if not _finite(recovery_horizon_s) or recovery_horizon_s <= 0 or not _finite(tilt_limit_deg) or tilt_limit_deg <= 0:
    raise ValueError("Recovery horizon and tilt limit must be finite and positive")
  expected = len(rows) if expected_cases is None else expected_cases
  if type(expected) is not int or expected < 0:
    raise ValueError("expected_cases must be a nonnegative integer")
  cases, delays, tilt_costs = [], [], []
  for row in rows:
    reasons = []
    if _case_key(row) is None:
      reasons.append("missing or invalid scenario conditions")
    if row.get("invalid_state") is True:
      reasons.append("invalid simulation state")
    failed = row.get("fell") is True
    if type(row.get("fell")) is not bool:
      reasons.append("missing fall status")
    if row.get("prefallen") is True and not failed:
      reasons.append("inconsistent prefall status")
    tilt = row.get("max_tilt")
    if not _finite(tilt) or not 0 <= tilt <= 180:
      reasons.append("missing or invalid maximum tilt")
    shots, at, interval = row.get("shots"), row.get("at"), row.get("interval")
    schedule_valid = (type(shots) is int and 1 <= shots <= 8 and _finite(at) and at >= 0
                      and _finite(interval) and interval >= 0)
    last_shot = at + (shots - 1) * interval if schedule_valid else None
    if not schedule_valid:
      reasons.append("invalid shot schedule")
    contacts = row.get("contacts")
    contact_hits = (sum(_finite(contact.get("samples")) and contact["samples"] > 0 for contact in contacts)
                    if isinstance(contacts, list) and all(isinstance(contact, Mapping) for contact in contacts) else None)
    hits, fired = row.get("hit_count"), row.get("fired")
    if (type(hits) is not int or type(fired) is not int or contact_hits is None
        or not 0 <= hits <= fired or hits != contact_hits or (schedule_valid and fired > shots)):
      reasons.append("missing or inconsistent contact evidence")
    missed = schedule_valid and type(hits) is int and hits < shots
    if not failed and (missed or fired != shots):
      reasons.append("surviving trial did not receive every scheduled impact")
    if not failed and schedule_valid and 4.0 - (shots - 1) * interval < recovery_horizon_s - 1e-9:
      reasons.append("follow-up is shorter than the fixed recovery horizon")
    end = row.get("recovery_window_end_s")
    raw_delay = row.get("recovery_delay_s")
    if not failed and ("recovery_window_end_s" not in row or "recovery_delay_s" not in row):
      reasons.append("missing recovery observation")
    if end is not None and (not _finite(end) or last_shot is None or end < last_shot or end > at + 4.0 + 1e-3):
      reasons.append("invalid recovery completion time")
    if raw_delay is not None and (not _finite(raw_delay) or raw_delay < 0):
      reasons.append("invalid raw recovery delay")
    if (end is None) != (raw_delay is None):
      reasons.append("inconsistent recovery measurements")
    elapsed = end - last_shot if _finite(end) and last_shot is not None else None
    recovered = not failed and elapsed is not None and 0 <= elapsed <= recovery_horizon_s
    penalty = min(elapsed, recovery_horizon_s) if recovered else recovery_horizon_s
    cases.append({"condition": _safe({key: row.get(key) for key in CASE_FIELDS}),
                  "valid": not reasons, "reasons": reasons, "fell": row.get("fell"),
                  "prefallen": row.get("prefallen"), "max_tilt_deg": _safe(tilt),
                  "scheduled_shots": shots, "fired": fired, "contact_hits": hits,
                  "contact_miss": bool(missed), "last_scheduled_shot_s": last_shot,
                  "raw_recovery_delay_from_last_contact_s": _safe(raw_delay),
                  "recovery_window_end_s": _safe(end), "recovery_completion_from_last_shot_s": elapsed,
                  "recovered_by_horizon": recovered, "penalized_recovery_s": penalty})
    delays.append(penalty)
    tilt_costs.append(1.0 if failed or not _finite(tilt) else min(tilt / tilt_limit_deg, 1.0))
  invalid = sum(not row["valid"] for row in cases)
  count = len(cases)
  valid = count > 0 and count == expected and invalid == 0
  failures = sum(row["fell"] is True for row in cases)
  survival = (count - failures) / count if count else None
  recovery = 1 - _mean(delays) / recovery_horizon_s if count else None
  tilt_quality = 1 - _mean(tilt_costs) if count else None
  score = 100 * (0.70 * survival + 0.20 * recovery + 0.10 * tilt_quality) if valid else None
  finite_tilts = [row["max_tilt_deg"] for row in cases if _finite(row["max_tilt_deg"])]
  return {"valid": valid, "score": score, "coverage": {"expected_cases": expected, "observed_cases": count,
          "invalid_cases": invalid, "complete": count == expected},
          "raw_counts": {"failures": failures, "survivals": sum(row["fell"] is False for row in cases),
                         "prefalls": sum(row["prefallen"] is True for row in cases),
                         "contact_miss_trials": sum(row["contact_miss"] for row in cases),
                         "recovered_by_horizon": sum(row["recovered_by_horizon"] for row in cases),
                         "failed_unrecovered_or_late": sum(not row["recovered_by_horizon"] for row in cases)},
          "mean_penalized_recovery_s": _mean(delays) if valid else None,
          "observed_max_tilt_deg": max(finite_tilts) if finite_tilts else None,
          "tilt_measurement_count": len(finite_tilts), "cases": cases,
          "components": {"survival": survival if valid else None, "fixed_horizon_recovery": recovery if valid else None,
                         "tilt": tilt_quality if valid else None}, "weights": dict(STABILITY_WEIGHTS),
          "formula": "S=100*(0.70*survival+0.20*(1-mean(penalized_recovery)/H)+0.10*(1-mean(tilt_cost)))",
          "recovery_horizon_s": recovery_horizon_s, "recovery_origin": "last scheduled shot",
          "recovery_completion": "end of measured sustained recovery window",
          "tilt_limit_deg": tilt_limit_deg, "failed_tilt_cost": 1.0, "weights_are_subjective": True}


def compare_stability(candidate_rows: Sequence[Mapping], manufacturer_rows: Sequence[Mapping],
                      *, recovery_horizon_s: float = 3.0, issues: Sequence[str] = ()) -> dict:
  keys = [[_case_key(row) for row in rows] for rows in (candidate_rows, manufacturer_rows)]
  problems = list(issues)
  if any(key is None for group in keys for key in group):
    problems.append("Missing scenario conditions")
  elif any(len(set(group)) != len(group) for group in keys) or set(keys[0]) != set(keys[1]):
    problems.append("Candidate/manufacturer scenario sets differ or contain duplicates")
  summaries = {label: stability_summary(rows, expected_cases=len(manufacturer_rows), recovery_horizon_s=recovery_horizon_s)
               for label, rows in (("candidate", candidate_rows), ("manufacturer", manufacturer_rows))}
  for summary in summaries.values():
    if problems:
      summary.update(valid=False, score=None)
  valid = not problems and all(summary["valid"] for summary in summaries.values())
  failures = [summaries[label]["raw_counts"]["failures"] for label in ("candidate", "manufacturer")]
  return {"valid": valid, **summaries, "reasons": problems,
          "score_delta": summaries["candidate"]["score"] - summaries["manufacturer"]["score"] if valid else None,
          "fall_outcome": "both_zero_failures" if failures == [0, 0] and valid else "see_raw_counts",
          "comparison_conditions": {"same_projectile_and_command_schedule": not problems,
                                    "preimpact_realized_speed_matching": "not recorded in projectile protocol 3"},
          "superiority_established": False,
          "interpretation": "Fixed commanded cases do not certify actual pre-impact speed equality or statistical superiority."}


def score_results(gait: Mapping, projectiles: Mapping, manufacturer_projectiles: Mapping, *,
                  candidate: str = "candidate", targets_mps: Sequence[float] = (0.2, 0.3),
                  projectile_speeds_mps: Sequence[float] = (12., 18., 25.),
                  recovery_horizon_s: float = 3.0) -> dict:
  """One JSON protocol for existing gait-v2/projectile-v3 reports, with raw evidence."""
  identity_issues = []
  if not projectile_speeds_mps or len(set(projectile_speeds_mps)) != len(projectile_speeds_mps):
    raise ValueError("Projectile speed regimes must be nonempty and unique")
  for label, report in ((candidate, projectiles), ("manufacturer", manufacturer_projectiles)):
    gait_hash = gait.get("results", {}).get(label, {}).get("sha256")
    impact_hash = report.get("sha256", {}).get(label)
    if not gait_hash or not impact_hash or gait_hash != impact_hash:
      identity_issues.append(f"Gait/projectile policy SHA256 is missing or differs for {label}")
  quiet = score_gait_report(gait, candidate=candidate, targets_mps=targets_mps, issues=identity_issues)
  impact_issues = list(identity_issues)
  if any(report.get("protocol_version") != 3 for report in (projectiles, manufacturer_projectiles)):
    impact_issues.append("Requires projectile protocol_version=3")
  if (not projectiles.get("recovery_definition")
      or projectiles.get("recovery_definition") != manufacturer_projectiles.get("recovery_definition")):
    impact_issues.append("Recovery definitions are missing or differ")
  contact_modes = [report.get("settings", {}).get("contacts") for report in (projectiles, manufacturer_projectiles)]
  if not contact_modes[0] or contact_modes[0] != contact_modes[1]:
    impact_issues.append("Contact physics modes are missing or differ")
  impacts = []
  for speed in projectile_speeds_mps:
    if not _finite(speed) or speed <= 0:
      raise ValueError("Projectile speeds must be positive and finite")
    selected = [[row for row in report.get("results", {}).get(label, [])
                 if _finite(row.get("speed")) and abs(row["speed"] - speed) < 1e-9]
                for report, label in ((projectiles, candidate), (manufacturer_projectiles, "manufacturer"))]
    impacts.append({"projectile_speed_mps": speed,
                    **compare_stability(*selected, recovery_horizon_s=recovery_horizon_s, issues=impact_issues)})
  return _safe({"schema_version": 1, "protocol": "r1lab.physical_proxy_scores.v1", "candidate": candidate,
                "reference": "manufacturer", "quietness": quiet, "stability": impacts,
                "validation_issues": identity_issues,
                "evaluation_protocols": {"gait_version": gait.get("protocol_version"),
                  "projectile_version": projectiles.get("protocol_version"),
                  "recovery_definition": projectiles.get("recovery_definition"),
                  "contact_mode": contact_modes[0],
                  "protocol_3_observation_end": "first scheduled shot + 4 s",
                  "candidate_sha256": projectiles.get("sha256", {}).get(candidate),
                  "manufacturer_sha256": manufacturer_projectiles.get("sha256", {}).get("manufacturer")},
                "notes": ["Quietness is a physical proxy, not measured acoustic dB.",
                          "Weights and tilt threshold are disclosed engineering choices.",
                          "Null scores indicate missing, invalid, or incomparable evidence; they are never zero.",
                          "Recovery penalizes failures and absent/late recovery with the full fixed horizon.",
                          "Quiet targets and projectile speed regimes are reported separately.",
                          "Equal zero-failure counts do not establish stability superiority over the manufacturer."]})
