#!/usr/bin/env bash
# Local only: periodically score and render current training checkpoints; preserve latest
# and best videos with replay evidence under .agent/trainer/result/video/<ID>/.
set -euo pipefail

rl_scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rl_root="$(cd -- "$rl_scripts_dir/.." && pwd)"
if [ -d "$rl_root/.agent" ]; then rl_agent="$rl_root/.agent"; else rl_agent="$(dirname -- "$rl_root")/.agent/trainer"; fi
rl_agent_tmp="$rl_agent/tmp"
mkdir -p -- "$rl_agent_tmp"
export TMPDIR="$rl_agent_tmp" TMP="$rl_agent_tmp" TEMP="$rl_agent_tmp"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL="${MUJOCO_GL:-egl}"

exec python3 -c '
import os, sys
script = sys.argv.pop(1)
sys.argv[0] = script
with os.fdopen(3) as source:
    code = compile(source.read(), script, "exec")
exec(code, {"__name__": "__main__", "__file__": script})
' "$rl_scripts_dir/05_results.sh" "$@" 3<<'RL_RESULTS_PYTHON'
"""학습 run(ID)별 최신·최고 결과 렌더링. 로컬 전용이며 추론(평가·렌더링)은 이 PC에서 한다.

  scripts/05_results.sh --owner-pid <PID>  앱 내부용: 실행 중인 앱에 종속된 주기 작업
  scripts/05_results.sh --once             한 번만
  scripts/05_results.sh --id <ID>          해당 ID만 (한 번)
  scripts/05_results.sh --install-service  앱 자동 관리 안내 (상시 서비스 등록하지 않음)

작업 영역 <work> = agroid_v1/.agent/trainer. 대상: <work>/runs/<실험>/<run>/run_info.json 중 render=true.
점수(높을수록 좋음, 같은 조건이라 run끼리 비교 가능):
  보행     노이즈+거친 지형+밀기·타격, 88 env 10초: 100(1-넘어짐) - 10·속도오차 - 5·회전오차 - 10·착지속도
  웅크리기 노이즈+교란 시작+밀기·타격, 48 env 12초: 100(1-넘어짐) - 100·최종높이오차 - 20·max(0, 최대하강속도-0.12)
  S7(v4) 보행 점수에서 실제 공 24사례 넘어짐 비율×100을 추가 차감한다. 결측/미접촉은 미산출이다.
  S8(v5) 고정 nominal/hard 사례의 유효 노출·넘어짐만 안정성으로 집계한다. 목적성은 별도 미검증이다.
영상: <work>/result/video/<ID>/<촬영시각>_<정책>_<사례>.mp4, 전체 모션·환경·재현 메타데이터와 함께 보존.
결과: <work>/result/render/<ID>/평가 JSON·점수,
      <work>/result/render/<ID>/scores.tsv, <work>/result/index.tsv
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import statistics as st
import signal
import shutil
import threading
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]  # trainer/
AGENT = ROOT / ".agent" if (ROOT / ".agent").is_dir() else ROOT.parent / ".agent" / "trainer"
PY = str(AGENT / ".venv/bin/python")
TOOLS = AGENT / "tools"
sys.path.insert(0, str(TOOLS))
RUNS = AGENT / "runs"
OUT = AGENT / "result/render"
TMP = AGENT / "tmp"
ACTIVE_S = 1200     # active if the newest checkpoint is younger than this
SETTLE_S = 60       # skip checkpoints younger than this (may still be syncing)
STATUS = AGENT / "result/status.json"
WATCHED = AGENT / "result/watched.json"
S8_PROFILE = "careful_adversarial_v1"
TEACHER_PROFILE = "reference_teacher_v1"
STUDENT_PROFILE = "command_student_v1"
STUDENT_COLS = ["score_kind", "survived_idle", "survived_default", "evaluation_cases", "physical_falls",
                "action_rmse_walk", "action_rmse_change"]
S8_PREVIEW = ("nominal_00", "nominal_02", "hard_irregular_crossfire_01",
              "hard_turning_crossfire_01", "hard_rock_field_01", "hard_friction_change_01")
S8_COLS = ["safety_score", "safety_cases", "safety_invalid", "safety_exposure_s",
           "evaluation_scope", "score_kind", "purpose_status"]

EVAL = {
    "careful_walk": ["--task", "careful_walk", "--num-envs", "88", "--seconds", "10", "--noise",
                     "--rough", "--disturb", "--seed", "5"],
    "safe_crouch": ["--task", "safe_crouch", "--num-envs", "48", "--seconds", "12", "--noise",
                    "--disturb", "--seed", "5"],
}
RENDER = {
    "careful_walk": ["--task", "careful_walk", "--rough", "--disturb"],
    "safe_crouch": ["--task", "safe_crouch", "--crouch-disturbed", "--disturb"],
}
MOTION = {  # clean rollout for 3D playback and video: flat ground, no disturbances
    "careful_walk": ["--task", "careful_walk"],
    "safe_crouch": ["--task", "safe_crouch"],
}
COLS = {"careful_walk": ["fall", "vel_err", "yaw_err", "touchdown", "peak_bw", "ball_fall", "ball_cases", "ball_hits"],
        "safe_crouch": ["fall", "final_err", "vmax", "tilt_deg"]}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(text)
    part.replace(path)


def status(**values):
    atomic_write(STATUS, json.dumps({"updated": time.time(), **values}, ensure_ascii=False))


def finite_score(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def score(action, r):
    def mean_required(values):
        values = list(values)
        return st.mean(values) if values and all(finite_score(v) for v in values) else None

    if action == "careful_walk":
        lin = [v for k, v in r.items() if not k.startswith("turn") and k != "stand"]
        turn = [v for k, v in r.items() if k.startswith("turn")]
        m = {"fall": mean_required(v.get("fall_rate") for v in r.values()),
             "vel_err": mean_required(v.get("vel_err") for v in lin),
             "yaw_err": mean_required(v.get("yaw_err") for v in turn),
             "touchdown": mean_required(v.get("touchdown_speed") for v in lin),
             "peak_bw": mean_required(v.get("foot_force_p95_bw") for v in lin)}
        # No touchdown is missing data, not perfectly quiet walking. Zero-speed
        # touchdown events remain valid when the evaluator reports their count.
        if any(v.get("touchdown_count") == 0 for v in lin):
            m["touchdown"] = None
        s = (100 * (1 - m["fall"]) - 10 * m["vel_err"] - 5 * m["yaw_err"] - 10 * m["touchdown"]
             if all(finite_score(v) for v in m.values()) else None)
    else:
        m = {"fall": r["fall_rate"], "final_err": r["final_height_err"],
             "vmax": r["max_vertical_speed"], "tilt_deg": r["max_tilt_deg"]}
        s = (100 * (1 - m["fall"]) - 100 * m["final_err"] - 20 * max(0.0, m["vmax"] - 0.12)
             if all(finite_score(v) for v in m.values()) else None)
    return (round(s, 2) if s is not None else None), {
        k: round(v, 3) if finite_score(v) else None for k, v in m.items()}


def runs(only=None):
    out = []
    for f in sorted(RUNS.glob("*/*/run_info.json")):
        info = json.loads(f.read_text())
        if info.get("render") and info.get("action") in EVAL and (only is None or info["id"] == only):
            out.append((info, f.parent))
    return out


def checkpoints(run):
    now, out = time.time(), {}
    try:
        settled = 0 if json.loads((run / "run_info.json").read_text()).get("atomic_checkpoints") is True else SETTLE_S
    except (OSError, ValueError):
        settled = SETTLE_S
    for p in run.glob("policy_*.onnx"):
        m = re.fullmatch(r"policy_(\d+)\.onnx", p.name)
        if m and now - p.stat().st_mtime > settled:
            out[int(m.group(1))] = p
    return out


def read_rows(d):
    f = d / "scores.tsv"
    if not f.exists():
        return []
    lines = f.read_text().splitlines()
    hdr = lines[0].split("\t")
    return [dict(zip(hdr, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def write_rows(d, action, rows):
    extra = S8_COLS if any(int(row.get("statistics_version", 1)) == 5 for row in rows) else []
    if any(int(row.get("statistics_version", 1)) == 6 for row in rows):
        extra += ["score_kind", "reference_complete", "evaluation_cases", "physical_falls", "tracking_terminations"]
    if any(int(row.get("statistics_version", 1)) == 7 for row in rows):
        extra = list(dict.fromkeys(extra + STUDENT_COLS))
    hdr = ["iter", "score", "statistics_version"] + COLS[action] + extra + ["recording_version", "video", "evaluated"]
    rows = sorted(rows, key=lambda r: int(r["iter"]))
    text = "\n".join(["\t".join(hdr)] + ["\t".join(str(r.get(k, 1 if k == "statistics_version" else "")) for k in hdr) for r in rows])
    atomic_write(d / "scores.tsv", text + "\n")


def pick(ck, rows, active, directory=None):
    if not ck:
        return None
    newest = max(ck)
    done = {int(r["iter"]) for r in rows if r.get("video") and r.get("score") != "fail"
            and (not r.get('recording_version')
                 or directory is not None and recording_audit(directory, int(r['iter'])).get('all_complete'))}
    return None if newest in done else newest


def run_tool(args, timeout):
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def video_destination(run_id, label):
    from evaluation_video import new_video_path
    return new_video_path(run_id, label, 'candidate')


def video_coverage(payload, expected_cases=None):
    from recording_integrity import report_coverage
    cases = expected_cases if expected_cases is not None else (payload or {}).get('selected_case_ids', [])
    result = report_coverage(payload or {}, AGENT / 'result',
                             expected_policies=['manufacturer', 'candidate'], expected_cases=cases)
    result['expected_case_ids'] = list(cases)
    return result


def recording_audit(directory, iteration, reports=None, videos=None):
    from recording_integrity import report_coverage, validate_video
    path = directory / f'recordings_it{iteration}.json'
    if reports is None:
        try:
            previous = json.loads(path.read_text())
            reports, videos = previous['reports'], previous['videos']
        except (OSError, ValueError, KeyError):
            return {'all_complete': False, 'reason': 'Recording inventory is missing'}
    checks = []
    for report in reports:
        try:
            payload = json.loads(Path(report['path']).read_text())
            value = report_coverage(payload, AGENT / 'result', expected_policies=report.get('policies'))
        except (OSError, ValueError, KeyError) as error:
            value = {'all_complete': False, 'reason': str(error)}
        checks.append({'source': report['path'], **value})
    for video in videos or []:
        try:
            payload = json.loads(Path(video).with_suffix('.video.json').read_text())
            value = validate_video(payload, AGENT / 'result')
        except (OSError, ValueError, KeyError) as error:
            value = {'complete': False, 'reason': str(error)}
        checks.append({'source': video, **value, 'all_complete': bool(value.get('complete'))})
    return {'version': 1, 'reports': reports, 'videos': videos or [], 'checks': checks,
            'all_complete': bool(reports) and bool(videos) and all(row.get('all_complete') for row in checks)}


def evaluate(action, onnx):
    rid = os.environ.get('RL_EVALUATION_RUN_ID')
    iteration = os.environ.get('RL_EVALUATION_ITERATION')
    persistent = bool(rid and Path(rid).name == rid and rid not in ('.', '..') and iteration and iteration.isdigit())
    if persistent:
        out = OUT / rid / f'eval_it{iteration}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, out = tempfile.mkstemp(suffix=".json", dir=TMP)
        os.close(fd)
    try:
        r = run_tool([str(TOOLS / "eval_policy.py"), *EVAL[action], "--onnx", str(onnx), "--out", str(out)], 3600)
        if r.returncode != 0:
            log(f"  eval failed: {r.stderr.strip().splitlines()[-1:] }")
            return None
        return json.loads(Path(out).read_text())["result"]
    finally:
        if not persistent:
            Path(out).unlink(missing_ok=True)


def evaluate_projectiles(onnx, dst):
    part = dst.with_name(dst.stem + ".part.json")
    try:
        result = run_tool([str(TOOLS / "eval_projectiles.py"), "--policy", f"candidate={onnx}",
                           "--speeds", "12", "18", "25", "--shots", "3", "--jobs", "1",
                           "--out", str(part)], 1200)
        if result.returncode != 0:
            log(f"  projectile eval failed: {result.stderr.strip().splitlines()[-1:]}")
            return None
        payload = json.loads(part.read_text())
        rows = payload["results"]["candidate"]
        part.replace(dst)
        if (payload.get("protocol_version") != 3 or len(rows) != 24
                or any(r["prefallen"] or not r["hit"] for r in rows)):
            log("  projectile eval incomplete, missing contact or failed before impact; raw evidence retained")
            return None
        return {"ball_fall": sum(r["fell"] for r in rows)/len(rows),
                "ball_cases": len(rows), "ball_hits": sum(r["hit"] for r in rows)}
    finally:
        part.unlink(missing_ok=True)


def evaluate_gait(onnx, dst):
    manufacturer = ROOT.parent / 'src/motion_manager/walking_action_manager/config/walk.onnx'
    with tempfile.TemporaryDirectory(prefix='gait-', dir=TMP) as temporary:
        result = run_tool([str(TOOLS / 'eval_gait.py'), '--policy', f'manufacturer={manufacturer}',
                           '--policy', f'candidate={onnx}', '--reference', 'manufacturer',
                           '--targets', '.2', '.3', '--speeds', '0', '.15', '.2', '.25', '.3', '.35', '.4', '.45', '.5',
                           '--out', temporary], 1200)
        report = Path(temporary) / 'summary.json'
        if result.returncode != 0 or not report.exists():
            log(f"  gait eval failed: {result.stderr.strip().splitlines()[-1:]}")
            return False
        payload = json.loads(report.read_text())
        if payload.get('protocol_version') != 2:
            raise ValueError('Unexpected gait evaluation protocol')
        report.replace(dst)
        return True


def safety_v5(payload, cases, scope):
    """Safety only: no speed/quietness penalty, and no missing exposure credit."""
    issues = []
    payload = payload or {}
    if payload.get("protocol_version") != 1 or payload.get("evaluation_profile") != "hard_walk_v1":
        issues.append("Missing or incompatible hard-case evaluation")
    if set(payload.get("selected_case_ids", [])) != {case["id"] for case in cases}:
        issues.append("Requested and observed case sets differ")
    groups = {}
    for group in ("nominal", "hard"):
        expected = {case["id"]: case for case in cases if case["group"] == group}
        group_result = {}
        for label in ("manufacturer", "candidate"):
            observed = [row for row in payload.get("results", {}).get(label, []) if row.get("group") == group]
            rows = {row.get("case_id"): row for row in observed}
            errors = []
            if not expected or set(rows) != set(expected) or len(rows) != len(observed):
                errors.append("Incomplete or duplicate case coverage")
            exposure, failures, invalid = 0., 0, 0
            for case_id, case in expected.items():
                row = rows.get(case_id, {})
                failed = row.get("fell") is True
                failures += int(failed)
                measured = row.get("valid_exposure_s")
                requested = case["duration_s"] - case["warmup_s"]
                case_hash = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                complete = (row.get("valid") is True and type(row.get("fell")) is bool
                            and row.get("case_sha256") == case_hash and finite_score(measured)
                            and not isinstance(measured, bool)
                            and 0 < float(measured) <= requested + .03
                            and (failed or float(measured) >= requested - .03))
                if complete:
                    exposure += float(measured)
                else:
                    invalid += 1
            if invalid:
                errors.append("Invalid, missing, or incomplete physical exposure")
            group_result[label] = {"expected_cases": len(expected), "observed_cases": len(observed),
                "raw_failures": sum(row.get("fell") is True for row in observed),
                "invalid_cases": invalid, "valid_exposure_s": exposure,
                "valid": not errors, "reasons": errors,
                "fall_rate": failures / len(expected) if expected and not errors else None,
                "safety_score": round(100 * (1 - failures / len(expected)), 3) if expected and not errors else None}
        groups[group] = group_result
    valid = not issues and all(value[who]["valid"] for value in groups.values() for who in ("manufacturer", "candidate"))
    score = min(group["candidate"]["safety_score"] for group in groups.values()) if valid else None
    return {"schema_version": 1, "statistics_version": 5, "score_kind": "safety_only",
            "evaluation_profile": S8_PROFILE, "evaluation_scope": scope,
            "valid": valid, "safety_score": score, "groups": groups, "reasons": issues,
            "generic_score_rankable": valid and scope == "full",
            "formula": "100*(1-fall_fraction) within each complete group; minimum of nominal and hard groups",
            "exposure_rule": "Every requested case must have verified exposure; survivors require the whole measurement interval.",
            "purpose_status": "unverified", "combined_goal_pass": None,
            "purpose_note": "Stability alone does not validate cautious human-like foot placement, weight shift, rhythm, or arm/body coordination.",
            "kinematic_windows": [row["kinematic_windows"] for rows in payload.get("results", {}).values()
                                  for row in rows if row.get("kinematic_windows")],
            "videos": video_coverage(payload, [case["id"] for case in cases]),
            "casebank": payload.get("casebank"),
            "notes": ["Preview and full case sets are not ranked against each other.",
                      "Slowing or stopping during recovery has no speed penalty.",
                      "Nominal and hard groups stay separate; extreme cases are excluded.",
                      "Purpose must be validated independently from the saved 50Hz motion windows."]}


def evaluate_adversarial(onnx, dst, full=False):
    bank_path = AGENT / "data/eval/hard_walk_cases.json"
    bank = json.loads(bank_path.read_text())
    cases = [case for case in bank["cases"] if case["group"] in ("nominal", "hard")
             and (full or case["id"] in S8_PREVIEW)]
    args = [str(TOOLS / "eval_hard_cases.py"), "--casebank", str(bank_path),
            "--candidate", str(onnx), "--groups", "nominal", "hard", "--save-kinematic-windows",
            "--workers", "4" if full else "2", "--out", str(dst)]
    if not full:
        for case_id in S8_PREVIEW:
            args.extend(("--case", case_id))
    result = run_tool(args, 7200 if full else 1800)
    payload = json.loads(dst.read_text()) if dst.is_file() else None
    if result.returncode != 0:
        log(f"  S8 hard-case evaluation incomplete: {result.stderr.strip().splitlines()[-1:]}")
    safety = safety_v5(payload, cases, "full" if full else "preview")
    if result.returncode != 0:
        safety.update(valid=False, safety_score=None, generic_score_rankable=False)
        safety["reasons"].append("Hard-case evaluator did not complete successfully; partial raw evidence retained")
    safety["raw_report"] = dst.name
    return safety


def process_adversarial(info, run, full=False):
    rid, action = info["id"], info["action"]
    os.environ['RL_EVALUATION_RUN_ID'] = rid
    directory = OUT / rid
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints_found = checkpoints(run)
    if not checkpoints_found:
        return False
    iteration = max(checkpoints_found)
    os.environ['RL_EVALUATION_ITERATION'] = str(iteration)
    scope = "full" if full or info.get("state") == "completed" else "preview"
    rows = read_rows(directory)
    previous = next((row for row in rows if int(row["iter"]) == iteration), {})
    previous_evidence_path = directory / f"safety_it{iteration}.json"
    previous_evidence = json.loads(previous_evidence_path.read_text()) if previous_evidence_path.is_file() else {}
    archived = False
    if previous_evidence.get('videos', {}).get('all_complete'):
        try:
            raw = json.loads((directory / f'hard_it{iteration}.json').read_text())
            archived = video_coverage(raw, previous_evidence['videos'].get('expected_case_ids')).get('all_complete')
        except (OSError, ValueError):
            pass
    if (previous.get("statistics_version") == "5" and previous.get("evaluation_scope") == scope
            and archived):
        return False
    if (previous_evidence.get('videos', {}).get('disk_limited', 0)
            and shutil.disk_usage(directory).free < 512 * 1024**2):
        log(f"{rid}: video/replay recording waits for free disk space; prior results preserved")
        return False
    status(state="evaluating_hard_cases", current=rid, iteration=iteration, evaluation_scope=scope)
    evidence = evaluate_adversarial(checkpoints_found[iteration], directory / f"hard_it{iteration}.json", full=scope == "full")
    atomic_write(directory / f"safety_it{iteration}.json", json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    candidate_groups = [group["candidate"] for group in evidence["groups"].values()]
    count = sum(group["expected_cases"] for group in candidate_groups)
    failures = sum(group["raw_failures"] for group in candidate_groups)
    row = {"iter": iteration, "statistics_version": 5, "score_kind": "safety_only",
           "score": evidence["safety_score"] if evidence["generic_score_rankable"] else "missing",
           "safety_score": evidence["safety_score"] if evidence["safety_score"] is not None else "missing",
           "evaluation_scope": scope, "purpose_status": "unverified", "safety_cases": count,
           "safety_invalid": sum(group["invalid_cases"] for group in candidate_groups),
           "safety_exposure_s": round(sum(group["valid_exposure_s"] for group in candidate_groups), 3),
           "fall": round(failures / count, 4) if count else None,
           "evaluated": time.strftime("%Y-%m-%d %H:%M")}
    video = video_destination(rid, f"checkpoint_it{iteration}")
    if previous.get("video") and (directory / previous["video"]).is_file():
        row["video"] = previous["video"]
    else:
        status(state="rendering", current=rid, iteration=iteration, score_kind="safety_only", purpose_status="unverified")
        if render(action, checkpoints_found[iteration], video):
            row["video"] = os.path.relpath(video, directory)
    write_rows(directory, action, [entry for entry in rows if int(entry["iter"]) != iteration] + [row])
    write_index()
    return True


def render(action, onnx, dst, projectiles=False):
    scene = ["--task", "careful_walk", "--projectiles"] if projectiles else RENDER[action]
    r = run_tool([str(TOOLS / "record_video.py"), *scene, "--onnx", str(onnx), "--out", str(dst)], 3600)
    if r.returncode == 0 and dst.exists():
        mo = dst.with_suffix(".motion.json")
        flat_video = video_destination(os.environ['RL_EVALUATION_RUN_ID'], 'flat_motion')
        run_tool([str(TOOLS / "record_video.py"), *MOTION[action], "--onnx", str(onnx), "--out", str(flat_video),
                  "--motion-out", str(mo)], 1800)
        return True
    log(f"  render failed: {r.stderr.strip().splitlines()[-1:]}")
    return False


def process_teacher(info, run):
    """Reference-conditioned actors require their own observations and evidence."""
    directory = OUT / info['id']
    directory.mkdir(parents=True, exist_ok=True)
    found = checkpoints(run)
    if not found:
        return False
    iteration = max(found)
    report = directory / f'teacher_it{iteration}.json'
    comparison_path = directory / f'teacher_comparison_v2_it{iteration}.json'
    heldout_report = directory / f'teacher_heldout_it{iteration}.json'
    def complete(path):
        if not path.is_file():
            return False
        result = json.loads(path.read_text())
        return (result.get('status') == 'complete' and result.get('video', {}).get('status') == 'written'
                and ('standing_v2' not in path.name
                     or result.get('comparison_contract', {}).get('protocol_version') == 2))
    evidence = None
    if report.is_file():
        previous = json.loads(report.read_text())
        if previous.get('status') == 'complete' and previous.get('video', {}).get('status') == 'written':
            if (any(int(row['iter']) == iteration for row in read_rows(directory))
                    and comparison_path.is_file()
                    and json.loads(comparison_path.read_text()).get('status') == 'complete'
                    and (info.get('state') != 'completed' or complete(heldout_report))):
                return False
            evidence = previous
    if evidence is None:
        status(state='evaluating_reference_teacher', current=info['id'], iteration=iteration)
        result = run_tool([str(TOOLS / 'eval_motion_teacher.py'), '--onnx', str(found[iteration]),
                           '--out', str(report), '--num-envs', '4', '--seed', '23'], timeout=1800)
        if result.returncode:
            atomic_write(directory / f'teacher_it{iteration}.log', result.stdout + '\n' + result.stderr)
            log(f"{info['id']}: teacher evaluation incomplete; see {report.name}")
            return False
        evidence = json.loads(report.read_text())
    factory_report = directory / 'manufacturer_standing_v2.json'
    candidate_report = directory / f'teacher_standing_v2_it{iteration}.json'
    factory_policy = ROOT.parent / 'src/motion_manager/walking_action_manager/config/walk.onnx'
    evaluations = [(factory_report, factory_policy, ['--policy-role', 'manufacturer', '--teacher-run', str(run), '--initial-state', 'standing'], '1'),
                   (candidate_report, found[iteration], ['--initial-state', 'standing'], '1')]
    if info.get('state') == 'completed':
        evaluations.append((heldout_report, found[iteration], ['--reference-split', 'heldout'], '4'))
    for path, policy, extra, count in evaluations:
        if complete(path):
            continue
        result = run_tool([str(TOOLS / 'eval_motion_teacher.py'), '--onnx', str(policy),
                           '--out', str(path), '--num-envs', count, '--seed', '23', *extra], timeout=1800)
        if result.returncode or not complete(path):
            atomic_write(path.with_suffix('.log'), result.stdout + '\n' + result.stderr)
            log(f"{info['id']}: comparison recording incomplete: {path.name}")
            return False
    factory = json.loads(factory_report.read_text())
    candidate = json.loads(candidate_report.read_text())
    import numpy as np
    def initial_state(payload):
        with np.load(Path(payload['video']['state_path']) / '00000000.npz', allow_pickle=False) as state:
            return {key: state[key].copy() for key in ('qpos', 'qvel', 'ctrl')}
    first, second = initial_state(factory), initial_state(candidate)
    shared = (factory['video']['model_sha256'] == candidate['video']['model_sha256']
              and factory['reference_sha256'] == candidate['reference_sha256']
              and factory['comparison_contract'].get('protocol_version') == 2
              and candidate['comparison_contract'].get('protocol_version') == 2
              and factory['comparison_contract'].get('reference_alignment') == candidate['comparison_contract'].get('reference_alignment')
              and all(np.array_equal(first[key], second[key]) for key in first))
    comparison = {'status': 'complete' if shared else 'incompatible',
                  'iteration': iteration, 'same_model_reference_initial_state': shared,
                  'manufacturer': str(factory_report), 'candidate': str(candidate_report),
                  'manufacturer_summary': factory['summary'], 'candidate_summary': candidate['summary'],
                  'quality_conclusion': ('below_manufacturer_stability_in_this_case'
                      if shared and candidate['summary']['physical_falls'] > factory['summary']['physical_falls']
                      else 'overall_motion_quality_not_established'),
                  'protocol_version': 2,
                  'scope': 'One common standing-start episode with fixed initial reference XY/yaw alignment; teacher has reference inputs; no interactive command ramp. Human style requires separate trajectory/video review.'}
    atomic_write(comparison_path, json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
    if not shared:
        raise ValueError('Teacher/manufacturer evaluation model or initial-state mismatch')
    summary = evidence.get('summary', {})
    row = {'iter': iteration, 'score': 'unranked', 'statistics_version': 6,
           'score_kind': 'reference_tracking', 'reference_complete': summary.get('reference_complete'),
           'evaluation_cases': summary.get('expected_cases'), 'physical_falls': summary.get('physical_falls'),
           'tracking_terminations': summary.get('tracking_terminations'),
           'evaluated': time.strftime('%Y-%m-%d %H:%M')}
    for environment in evidence.get('video', {}).get('environments', []):
        for video in environment.get('videos', []):
            path = video.get('path') or video.get('video_path')
            if video.get('status') == 'written' and path and Path(path).is_file():
                row['video'] = os.path.relpath(path, directory)
                break
        if row.get('video'):
            break
    write_rows(directory, info['action'], [r for r in read_rows(directory) if int(r['iter']) != iteration] + [row])
    write_index()
    return True


def process_student(info, run):
    """810 students distilled from a matching teacher: scripted commands in the student scene, both start
    modes (library idle frame, R1 default pose), teacher ONNX shadow labels; never eval_policy.py."""
    directory = OUT / info['id']
    directory.mkdir(parents=True, exist_ok=True)
    found = checkpoints(run)
    if not found:
        return False
    iteration = max(found)
    os.environ['RL_EVALUATION_ITERATION'] = str(iteration)
    report = directory / f'student_scripted_it{iteration}.json'
    rows = read_rows(directory)

    def complete(path):
        if not path.is_file():
            return False
        result = json.loads(path.read_text())
        return result.get('status') == 'complete' and result.get('video', {}).get('status') == 'written'
    if complete(report) and any(int(row['iter']) == iteration for row in rows):
        return False
    distill = info.get('distillation') or {}
    teacher = AGENT / str(distill.get('teacher_policy', ''))
    if (not distill.get('teacher_policy') or not teacher.is_file()
            or hashlib.sha256(teacher.read_bytes()).hexdigest() != distill.get('teacher_policy_sha256')):
        log(f"{info['id']}: student evaluation needs the recorded teacher ONNX (missing or hash differs)")
        return False
    if not complete(report):
        status(state='evaluating_command_student', current=info['id'], iteration=iteration)
        result = run_tool([str(TOOLS / 'eval_matching_student.py'), '--student-onnx', str(found[iteration]),
                           '--teacher-onnx', str(teacher), '--teacher-env', distill.get('teacher_env') or
                           'matching_soft_teacher_env_cfg', '--out', str(report), '--seed', '11'], timeout=3600)
        if result.returncode or not complete(report):
            atomic_write(directory / f'student_it{iteration}.log', result.stdout + '\n' + result.stderr)
            log(f"{info['id']}: student evaluation incomplete; see {report.name}")
            return False
    evidence = json.loads(report.read_text())
    summary = evidence.get('summary', {})
    starts = [summary.get(key, {}) for key in ('idle', 'default')]
    rmse = summary.get('action_rmse') or {}
    row = {'iter': iteration, 'score': 'unranked', 'statistics_version': 7, 'score_kind': 'command_student',
           'survived_idle': starts[0].get('survived_full'), 'survived_default': starts[1].get('survived_full'),
           'evaluation_cases': sum(s.get('cases', 0) for s in starts),
           'physical_falls': sum(s.get('physical_falls', 0) for s in starts),
           'action_rmse_walk': (rmse.get('walk_steady') or {}).get('all'),
           'action_rmse_change': (rmse.get('after_change_1s') or {}).get('all'),
           'evaluated': time.strftime('%Y-%m-%d %H:%M')}
    for environment in evidence.get('video', {}).get('environments', []):
        for video in environment.get('videos', []):
            path = video.get('path') or video.get('video_path')
            if video.get('status') == 'written' and path and Path(path).is_file():
                row['video'] = os.path.relpath(path, directory)
                break
        if row.get('video'):
            break
    write_rows(directory, info['action'], [r for r in rows if int(r['iter']) != iteration] + [row])
    write_index()
    return True


def process(info, run, full_s8=False):
    os.environ['RL_EVALUATION_RUN_ID'] = info['id']
    if info.get('evaluation_profile') == TEACHER_PROFILE:
        return process_teacher(info, run)
    if info.get('evaluation_profile') == STUDENT_PROFILE:
        return process_student(info, run)
    if info.get("evaluation_profile") == S8_PROFILE:
        return process_adversarial(info, run, full=full_s8)
    rid, action = info["id"], info["action"]
    d = OUT / rid
    d.mkdir(parents=True, exist_ok=True)
    ck = checkpoints(run)
    if not ck:
        return False
    rows = read_rows(d)
    active = time.time() - max(p.stat().st_mtime for p in ck.values()) < ACTIVE_S
    it = pick(ck, rows, active, directory=d)
    if it is None:
        return False
    os.environ['RL_EVALUATION_ITERATION'] = str(it)
    status(state="evaluating", current=rid, iteration=it)
    log(f"{rid}: evaluate it{it}")
    physical_balls = info.get("task", "").startswith("careful_walk_s7")
    res = evaluate(action, ck[it])
    row = {"iter": it, "recording_version": 1, "evaluated": time.strftime("%Y-%m-%d %H:%M")}
    if res is None and not physical_balls:
        row["score"] = "fail"
    else:
        s, m = score(action, res) if res is not None else (None, {})
        version = (max((v.get("statistics_version", 1) for v in res.values()), default=1)
                   if action == "careful_walk" and res is not None else 1)
        if physical_balls:
            status(state="evaluating_projectiles", current=rid, iteration=it)
            balls = evaluate_projectiles(ck[it], d / f"projectile_it{it}.json")
            version = 4
            if balls is None:
                s = None
                m.update(ball_fall=None, ball_cases=None, ball_hits=None)
            else:
                m.update(balls)
                s = round(s - 100*balls["ball_fall"], 2) if s is not None else None
            status(state="evaluating_gait", current=rid, iteration=it)
            evaluate_gait(ck[it], d / f"gait_it{it}.json")
        row["statistics_version"] = version
        best = max((float(r["score"]) for r in rows
                    if finite_score(r["score"]) and int(r.get("statistics_version", 1)) == version), default=None)
        row.update(score=s if s is not None else "missing", **m)
        log(f"  score {s} {m}" + ("" if best is None else f" (best so far {best})"))
        status(state="rendering", current=rid, iteration=it, score=s)
        video = video_destination(rid, f"checkpoint_it{it}")
        rendered = (render(action, ck[it], video, projectiles=True) if physical_balls
                    else render(action, ck[it], video))
        if rendered:
            row["video"] = os.path.relpath(video, d)
            log(f"  {'new best' if s is not None and (best is None or s > best) else 'latest'} -> {video.relative_to(AGENT)}")
    reports = [{'path': str(d / f'eval_it{it}.json')}]
    if physical_balls:
        reports += [{'path': str(d / f'projectile_it{it}.json'), 'policies': ['candidate']},
                    {'path': str(d / f'gait_it{it}.json'), 'policies': ['manufacturer', 'candidate']}]
    inventory = recording_audit(d, it, reports=reports,
                                videos=[str(d / row['video'])] if row.get('video') else [])
    atomic_write(d / f'recordings_it{it}.json', json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    write_rows(d, action, [r for r in rows if int(r["iter"]) != it] + [row])
    write_index()
    return True


def write_index():
    lines = ["id\tbest_iter\tbest_score\tevaluated\tlast_eval\tbest_video\trun_dir\tlatest_iter\tlatest_score\tlatest_video"]
    for info, run in runs():
        rows = [r for r in read_rows(OUT / info["id"]) if r["score"] != "fail"]
        if not rows:
            continue
        version = int(max(rows, key=lambda r: int(r["iter"])).get("statistics_version", 1))
        ranked = [r for r in rows if finite_score(r["score"])
                  and int(r.get("statistics_version", 1)) == version]
        b = max(ranked, key=lambda r: float(r["score"])) if ranked else {}
        newest = max(rows, key=lambda r: int(r["iter"]))
        lines.append("\t".join([info["id"], b.get("iter", ""), b.get("score", ""), str(len(rows)),
                                max(r["evaluated"] for r in rows), b.get("video", ""),
                                str(run.relative_to(AGENT)), newest["iter"], newest["score"], newest.get("video", "")]))
    atomic_write(AGENT / "result/index.tsv", "\n".join(lines) + "\n")


def cycle(only=None, full_s8=False):
    if os.environ.get('RL_EVALUATION_APP_PID'):
        source = ('import json,sys; sys.path.insert(0,sys.argv[1]); '
                  'from vector_evaluation_video import resume_pending_archives; '
                  'print(json.dumps(resume_pending_archives(sys.argv[2],run_id=sys.argv[3] or None)))')
        recovery = run_tool(['-c', source, str(TOOLS), str(AGENT / 'result'), only or ''], 7200)
        if recovery.returncode:
            log(f"  saved-state video recovery pending: {recovery.stderr.strip().splitlines()[-1:]}")
        elif recovery.stdout.strip():
            try:
                recovered_items = json.loads(recovery.stdout.strip().splitlines()[-1])
            except ValueError:
                recovered_items = []
                log('  saved-state video recovery returned an unreadable report; original archives retained')
            for recovered in recovered_items:
                log(f"  saved-state video {recovered['status']}: {recovered['manifest']}")
    todo = runs(only)
    if only is None:
        watched = set(json.loads(WATCHED.read_text())) if WATCHED.exists() else set()
        watched.update(info["id"] for info, run in todo
                       if info.get("schema_version") == 1
                       or any(time.time() - p.stat().st_mtime < ACTIVE_S for p in run.glob("policy_*.onnx")))
        atomic_write(WATCHED, json.dumps(sorted(watched)))
        todo = [(info, run) for info, run in todo if info["id"] in watched]
    # active runs first (newest checkpoint first)
    todo.sort(key=lambda x: -max((p.stat().st_mtime for p in x[1].glob("policy_*.onnx")), default=0))
    for info, run in todo:
        try:
            if full_s8:
                process(info, run, full_s8=True)
            else:
                process(info, run)
        except Exception as e:  # keep going with the other runs
            log(f"{info['id']}: error {e}")
    write_index()


def install_service():
    print("결과 평가·렌더링은 Operation Center 실행 중에만 앱이 자동 관리합니다. "
          "상시 systemd 서비스는 등록하지 않습니다.")


def process_identity(pid):
    """Linux PID birth time and state; comm can contain spaces/parentheses."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19], fields[0]
    except (OSError, IndexError):
        return None


def watch_owner(owner_pid):
    """Independent of app cleanup: stop the entire session if its owner dies."""
    if os.getpgrp() != os.getpid():
        raise RuntimeError("--owner-pid requires a dedicated process group")
    identity = process_identity(owner_pid)
    initial_parent = os.getppid()

    def terminate_group(signum=None, frame=None):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)
        time.sleep(.3)
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, terminate_group)
    signal.signal(signal.SIGINT, terminate_group)
    if identity is None or identity[1] == "Z" or initial_parent != owner_pid:
        terminate_group()

    def monitor():
        while True:
            current = process_identity(owner_pid)
            if (current is None or current[0] != identity[0] or current[1] == "Z"
                    or os.getppid() != initial_parent):
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(1)

    threading.Thread(target=monitor, name="results-owner-watchdog", daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-pid", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--id", default=None)
    ap.add_argument("--full", action="store_true", help="S8: run the full nominal/hard bank; --id also requests full S8 evaluation")
    ap.add_argument("--interval", type=float, default=600.0)
    ap.add_argument("--install-service", action="store_true")
    a = ap.parse_args()
    if a.interval <= 0:
        ap.error("--interval은 양수여야 합니다.")
    if a.full and not (a.once or a.id):
        ap.error("--full은 --once 또는 --id와 함께 사용하세요.")
    if a.install_service:
        return install_service()
    if a.owner_pid is None and not (a.once or a.id):
        ap.error("자동 평가·렌더링은 Operation Center에서 실행합니다. 수동 실행은 --once 또는 --id를 사용하세요.")
    if a.owner_pid is not None:
        os.environ['RL_EVALUATION_APP_PID'] = str(a.owner_pid)
        watch_owner(a.owner_pid)
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.once or a.id:
        with open(TMP / "results.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cycle(a.id, full_s8=bool(a.full or a.id))
            status(state="idle")
        return
    daemon = open(TMP / "results-daemon.lock", "w")
    try:
        fcntl.flock(daemon, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("자동 렌더링이 이미 실행 중입니다.")
        return
    log(f"render latest checkpoints every {a.interval:g}s -> {OUT}")
    while True:
        started = time.monotonic()
        with open(TMP / "results.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cycle()
            delay = max(0, a.interval - (time.monotonic() - started))
            status(state="idle", next_run=time.time() + delay, interval=a.interval)
        time.sleep(delay)


if __name__ == "__main__":
    main()
RL_RESULTS_PYTHON
