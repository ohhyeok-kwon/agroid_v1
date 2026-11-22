#!/usr/bin/env bash
# Evaluation battery of a whole-body checkpoint (r1wb.eval), run on the GPU server; results come back to the workstation.
#   scripts/06_evaluate.sh <run folder name> <iteration> [what...]     what: suite react clip burst heldout lag track
#                                                                      (default: suite react clip lag)
#   scripts/06_evaluate.sh --pull <run folder name>                    fetch that run's results (numbers, logs)
#   scripts/06_evaluate.sh --pull-archives                             fetch the packed state archives (slow link; also started in the background after a battery)
#   GPU=<n> (default 3)   WAIT=0: start it and return (log: .agent/runs/eval_*.log on both sides)
#   COMMIT=1    the stream's course plays the window a look-up scored (teachers trained with coherent_course)
#   DECISIVE=1  ... and the corrected course generator (teachers trained with decisive_course; implies COMMIT=1)
#   LAG_TASK=   scene for "lag" (default wb_task_teacher)      ORBIT=0   skip the orbit-aligned walking reaction
# The checkpoint's label drives the robot (a tracking or task teacher); the stream's course is the command's plan.
#   suite   18 calm command scripts: robot and course progress per script, falls, torque peak
#   react   single hits 12 / 25 N s on command streams (stand, 0.35 m/s), synchronised twins (+ orbit-aligned)
#   clip    the same hits on time-indexed clips (stand, cautious walk)
#   burst   eight-hit rows, 9 to 16 N s per hit: falls of 10 per cell
#   heldout held-out probe set (632 cases)
#   lag     training-distribution streams of LAG_TASK: progress-limit ends and falls per command kind
#   track   every library clip replayed in the student's scene, judged like a student
#   STEER=<name>  the push stages run with a learned System 2 (data/s2/steer_<name>.pt) steering the course generator,
#               or, with S2PUSH=1, standing between the goal and the code head
#   LIBRARY=<name>  the motion library of the scene (library_v2_<name>.npz, e.g. corebal) in every stage
#   STAGGER=0.32 REPEATS=12  stage "s2": every repeat stops at another moment of the gait (independent stops)
#   S2PUSH=1    the push stages (react wide burst heldout) run with System 2 writing the reference: HEAD, TOKENS and
#               (the teacher) GOAL_MODEL + BACKBONE as for "s2"; commands are then taken from the instruction
#               vocabulary (held-out probe: turn 1.0 instead of 0.8 rad/s, forward 0.35 instead of 0.5 m/s, height 0.62
#               instead of 0.6 m)
#   wide    the rest of the case list recorded for the manufacturer: 160 single hits, 240 at 15 / 18 / 21 N s,
#           80 hits of 0.1 s at 25 / 35 N s, 160 eight-hit rows at 6 / 12 N s
#   s2      System 2 in the loop on the compass scripts (instruction -> goal -> motion codes -> this checkpoint as
#           System 1): HEAD=<code head> TOKENS=<tokenizer> [GOAL_MODEL=<folder> BACKBONE=<model folder>: a
#           vision-language model states the goal; without them the instruction's own goal is used]
#           LANGUAGES="ko en" PHRASES="train heldout"
# On the workstation this forwards to the server (scripts/04_remote.sh) and then pulls. Nothing runs on the
# workstation's GPU. State archives of the rollouts are captured on the server, packed, pulled in the background and
# rendered on the workstation only (by the Operation Center). Originals remain until explicitly removed.
set -uo pipefail
cd "$(dirname "$0")/.."
host="${RL_REMOTE:-super-via-ailab}"; rdir="${RL_REMOTE_DIR:-/rl-agent}"

pull() {    # $1 = run folder name: the results (small) now
  local run=$1 work=../.agent/trainer kind incoming
  mkdir -p "$work/result/transit"
  incoming=$(mktemp -d "$work/result/transit/results.XXXXXXXX") || return 1
  for kind in render logs; do
    ssh "$host" "[ -d $rdir/.agent/result/$kind/$run ]" < /dev/null || continue
    mkdir -p "$incoming/$kind/$run"
    rsync -a "$host:$rdir/.agent/result/$kind/$run/" "$incoming/$kind/$run/" || { echo "pull failed: $kind (partial files retained in $incoming)"; return 1; }
  done
  PYTHONPATH="$PWD/src" "$work/.venv/bin/python" -B -m r1wb.eval.vector_evaluation_video \
    --import-results "$incoming" --result-root "$work/result"
}

pull_archives() {    # the state archives of the rollouts (packed on the server, about 1 MB/s through the relay)
  local work=../.agent/trainer name incoming remote_sha
  mkdir -p "$work/result/transit" "$work/result/video"
  ssh "$host" "ls $rdir/.agent/result/transit/*.tar.zst 2>/dev/null" < /dev/null | while read -r remote; do
    name=$(basename "$remote")
    remote_sha=$(ssh "$host" "sha256sum -- $(printf '%q' "$remote")" < /dev/null) || { echo "archive hash failed: $name"; continue; }
    remote_sha=${remote_sha%% *}
    [[ "$remote_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid archive hash: $name"; continue; }
    [ -f "$work/result/transit/received-$remote_sha.json" ] && continue
    incoming=$(mktemp -d "$work/result/transit/received.XXXXXXXX") || return 1
    rsync -a --partial "$host:$remote" "$incoming/$name" < /dev/null || { echo "archive pull failed; partial retained: $incoming"; continue; }
    if PYTHONPATH="$PWD/src" "$work/.venv/bin/python" -B -m r1wb.eval.vector_evaluation_video \
        --import-bundle "$incoming/$name" --sha256 "$remote_sha" --result-root "$work/result" > "$incoming/import.json"; then
      cp -n "$incoming/import.json" "$work/result/transit/received-$remote_sha.json"
      echo "verified $(TZ=Asia/Seoul date '+%T') $name (server and local originals retained)"
    else
      echo "archive verification/import failed; all files retained: $incoming"
    fi
  done
}

if [ "${1:-}" = "--pull" ]; then pull "$2"; exit; fi
if [ "${1:-}" = "--pull-archives" ]; then pull_archives; exit; fi
[ $# -ge 2 ] || { sed -n '2,8p' "$0"; exit 2; }

if [ ! -d .agent/runs ]; then                                  # workstation: run on the server, wait, pull
  run=$1; it=$2
  name="eval_$(printf '%s' "$run" | sed -E 's/^wholebody_[0-9]+-([0-9]+)_.*_([^_]+)$/\1_\2/')_$it"
  out=$(GPU="${GPU:-3}" scripts/04_remote.sh -d "$name" env COMMIT="${COMMIT:-}" DECISIVE="${DECISIVE:-}" LAG_TASK="${LAG_TASK:-}" ORBIT="${ORBIT:-1}" \
          HEAD="${HEAD:-}" TOKENS="${TOKENS:-}" GOAL_MODEL="${GOAL_MODEL:-}" BACKBONE="${BACKBONE:-}" LANGUAGES="${LANGUAGES:-}" \
          S2PUSH="${S2PUSH:-}" STEER="${STEER:-}" LIBRARY="${LIBRARY:-}" STAGGER="${STAGGER:-}" REPEATS="${REPEATS:-}" \
          PHRASES="${PHRASES:-}" scripts/06_evaluate.sh "$@") || { echo "$out"; exit 1; }
  echo "$out"
  log=$(printf '%s\n' "$out" | sed -n 's/.* log \(.*\)$/\1/p' | tail -1)
  [ "${WAIT:-1}" = 1 ] || exit 0
  until ssh "$host" "tail -1 $log 2>/dev/null | grep -q '^battery done'" < /dev/null; do sleep 15; done
  ssh "$host" "cat $log" < /dev/null | grep -v "Warning\|warn\|^Module \|INFO"
  pull "$run"
  mkdir -p ../.agent/trainer/result/transit
  ( flock -n 9 || exit 0; pull_archives ) 9> ../.agent/trainer/result/transit/.lock >> ../.agent/trainer/result/transit/pull.log 2>&1 < /dev/null &
  disown
  exit
fi

# ---- on the server ----
RUN=.agent/runs/r1_wholebody/$1; IT=$2; shift 2
[ $# -gt 0 ] || set -- suite react clip lag
PY=".agent/.venv/bin/python -B"
KEY=$(basename "$RUN" | sed -E 's/^wholebody_[0-9]+-([0-9]+)_.*_([^_]+)$/\1_\2/')
CK=$RUN/model_$IT.pt
[ -f "$CK" ] || { echo "missing $CK"; echo "battery done (failed)"; exit 1; }
NAME=$(basename "$RUN"); RENDER=.agent/result/render/$NAME; LOGS=.agent/result/logs/$NAME; mkdir -p "$LOGS"
OWNER=$$                                                       # the rollouts end with this script
[ -n "${DECISIVE:-}" ] && COMMIT=1
DECIDED="${DECISIVE:+--match-height target --match-stop arrive --course-set travel_window=32 match_noise=0}"
COURSE="--match-pose reference ${COMMIT:+--commit} $DECIDED ${LIBRARY:+--library $LIBRARY}"
[ -n "${LIBRARY:-}" ] && KEY="${KEY}_$LIBRARY"
PROBE_PY=$PY; H4_YAW=0.8; H4_LOW=0.6; H4_FWD=0.5
if [ -n "${STEER:-}" ]; then                                   # a learned System 2 steers the course generator (r1wb.s2.steer)
  COURSE="$COURSE --steer $STEER"; KEY="${KEY}_steer-$STEER"
fi
if [ -n "${S2PUSH:-}" ]; then                                  # System 2 (and the teacher's goal model) in the push tests
  PROBE_PY="env PYTHONPATH=$PWD/src .agent/rlinf/.venv/bin/python -B"
  COURSE="--match-pose reference --commit --s2-head $HEAD --tokens ${TOKENS:-v2c} ${GOAL_MODEL:+--goal-model $GOAL_MODEL --backbone .agent/models/$BACKBONE} ${STEER:+--steer $STEER} ${LIBRARY:+--library $LIBRARY}"
  KEY="${KEY}_s2-${HEAD}${GOAL_MODEL:+-$GOAL_MODEL}${STEER:+_steer-$STEER}"; H4_YAW=1.0; H4_LOW=0.62; H4_FWD=0.35
fi
export OMP_NUM_THREADS=8
stamp() { TZ=Asia/Seoul date '+%T'; }
tab() { grep -v "Warning\|warn" "$1" | awk '/context\|clip/ {print; next} /^anatomy/ {exit} /^(stand|walk)\|/ {print}' | cut -c1-128; }
trace() { grep -E "Traceback" -A 12 "$1" | head -14; }

probe() {   # $1 = log name, rest = probe arguments
  local L=$LOGS/${KEY}_${IT}_$1.log; shift
  $PROBE_PY -m r1wb.eval.impulse_probe --run "$RUN" --checkpoint "$CK" "$@" --app-pid "$OWNER" > "$L" 2>&1
  local code=$?; trace "$L"; return $code
}

for what in "$@"; do
  case $what in
    suite)
      L=$LOGS/${KEY}_${IT}_suite.log
      $PY -m r1wb.eval.eval_stream --run "$RUN" --checkpoint "$CK" --policy label $COURSE --app-pid "$OWNER" > "$L" 2>&1
      echo "== $KEY $IT labels drive the 18 scripts, course = the command's plan${COMMIT:+, windows played through}${DECISIVE:+, corrected generator}: exit $? $(stamp)"; trace "$L"
      python3 - "$(ls -t "$RENDER"/stream_eval_train_label_${IT}_course-reference*.json | head -1)" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1])); s = a["summary"]
f = lambda x: "   - " if x is None else f"{x:5.2f}"
print(f"   falls {s['falls']}  distance {s['distance_ratio']}  yaw {s['yaw_ratio']}  torque peak {s['torque_ratio_max']}  quiet-stand touchdowns/min {s['quiet_touchdowns_per_min']}  quiet arm speed {s['quiet_arm_speed_rps']}")
print(f"   {'script':16s} robot d / yaw   course d / yaw   switches/s  torque peak (actuator)")
for k, v in a["by_script"].items():
    if v.get("distance_ratio") is None and v.get("yaw_ratio") is None: continue
    print(f"   {k:16s} {f(v['distance_ratio'])} / {f(v['yaw_ratio'])}   {f(v.get('course_distance_ratio'))} / {f(v.get('course_yaw_ratio'))}   {f(v.get('match_switches_per_s'))}       {f(v.get('torque_ratio_peak'))} {v.get('torque_peak_actuator', '')}")
PYEOF
      ;;
    react)
      probe react --task stream_label $COURSE --twists 0 0 0 0.35 0 0 --impulses 12 25 --repeats 5 --sync-twins --trace "$LOGS/${KEY}_${IT}_react.npz"
      echo "== $KEY $IT labels on planned command streams: exit $? $(stamp)"; tab "$LOGS/${KEY}_${IT}_react.log"; grep "total falls" "$LOGS/${KEY}_${IT}_react.log"
      [ "${ORBIT:-1}" = 1 ] && $PY -m r1wb.eval.orbit_distance "$KEY@$IT=$LOGS/${KEY}_${IT}_react.npz" 2>&1 | grep -E "walk" | cut -c1-200
      ;;
    clip)
      probe clip --task clip_label --clips 00227_A_P0_StandIdle_230213 00230_A_P0_WalkCautious_230213 --impulses 12 25 --repeats 5
      echo "== $KEY $IT labels on fixed clips: exit $? $(stamp)"; tab "$LOGS/${KEY}_${IT}_clip.log"; grep "total falls" "$LOGS/${KEY}_${IT}_clip.log"
      ;;
    burst)
      probe burst --task stream_label $COURSE --twists 0 0 0 0.35 0 0 --directions back front --impulses 9 10 11 12 13 14 15 16 --burst 8 --burst-gap 0.15
      python3 - "$LOGS/${KEY}_${IT}_burst.log" "$KEY@$IT" "$?" <<'PYEOF'
import re, sys
rows = {}
for line in open(sys.argv[1], errors="ignore"):
    m = re.match(r"(stand|walk)\|[^|]*\|(front|back)\|(\d+)\s+(\d+)\s+(\d+)", line)
    if m: rows.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = (int(m.group(5)), int(m.group(4)))
print(f"== {sys.argv[2]} rows of eight hits (exit {sys.argv[3]}): falls of 10 at 9..16 N s per hit")
total = 0
for (ctx, d), cells in rows.items():
    sizes = sorted(cells); falls = [cells[s][0] for s in sizes]; total += sum(falls)
    limit = next((s for s in sizes if cells[s][0] * 2 >= cells[s][1]), None)
    print(f"   {ctx:5s} pushed {d:5s}: " + " ".join(f"{f:2d}" for f in falls) + f"   first size with half or more falling: {limit}")
print(f"   falls in the sweep: {total} / {sum(c[1] for cells in rows.values() for c in cells.values())}")
PYEOF
      ;;
    wide)
      falls() { printf "%-10s exit %s  %s\n" "$1" "$2" "$(grep -h 'total falls' "$LOGS/${KEY}_${IT}_$1.log" | sed 's/.*total falls: //')"; }
      probe wide_single --task stream_label $COURSE --twists 0 0 0 0.35 0 0 --impulses 12 25 --repeats 10; falls wide_single $?
      probe wide_mid --task stream_label $COURSE --twists 0 0 0 0.35 0 0 --impulses 15 18 21 --repeats 10; falls wide_mid $?
      probe wide_long --task stream_label $COURSE --twists 0 0 0 0.35 0 0 --impulses 25 35 --hit-duration 0.1 --repeats 5; falls wide_long $?
      probe wide_rows --task stream_label $COURSE --twists 0 0 0 0.35 0 0 --impulses 6 12 --burst 8 --burst-gap 0.15 --repeats 10; falls wide_rows $?
      echo "== $KEY $IT wide list: $(grep -h 'total falls' $LOGS/${KEY}_${IT}_wide_*.log | sed 's/.*total falls: //' | awk '{f += $1; n += $3} END {print f " / " n}') falls $(stamp)"
      ;;
    heldout)
      ALL="front back left right front_left front_right back_left back_right"
      held() {
        local name=$1; shift
        probe "ho_$name" --task stream_label $COURSE "$@"
        local code=$? L=$LOGS/${KEY}_${IT}_ho_$name.log
        printf "%-10s exit %s  %s  fallen cells: %s\n" "$name" "$code" "$(grep -h 'total falls' "$L" | sed 's/.*total falls: //')" \
          "$(grep -v 'Warning\|warn' "$L" | awk '/^anatomy/ {exit} /^(stand|walk)\|/ {if ($3 > 0) printf "%s %s/%s, ", $1, $3, $2}' | sed 's/twist_//g; s/twist|//g' | cut -c1-300)"
      }
      held H1shoulder --hit-body left_shoulder_roll_link --hit-height 0 --directions $ALL --impulses 12 25 --twists 0 0 0 0.35 0 0 --repeats 3
      held H2thigh --hit-body right_hip_yaw_link --hit-height 0 --directions $ALL --impulses 12 25 --twists 0 0 0 0.35 0 0 --repeats 3
      held H3head --hit-height 0.35 --directions $ALL --impulses 12 25 --twists 0 0 0 0.35 0 0 --repeats 3
      held H4cmd --directions $ALL --impulses 25 --twists 0 0 $H4_YAW -0.2 0 0 $H4_FWD 0 0 0 0.2 0 --repeats 3
      held H4low --directions $ALL --impulses 12 25 --twists 0 0 0 --height $H4_LOW --repeats 3
      held H5a --impulses 14 --burst 5 --burst-gap 0.25 --twists 0 0 0 0.35 0 0 --repeats 5
      held H5b --impulses 9 --burst 10 --burst-gap 0.08 --twists 0 0 0 0.35 0 0 --repeats 5
      held H5c --impulses 12 --burst 8 --burst-gap 0.3 --twists 0 0 0 0.35 0 0 --repeats 5
      held H6strong --impulses 30 35 --twists 0 0 0 0.35 0 0 --repeats 5
      echo "held-out total falls: $(grep -h 'total falls' $LOGS/${KEY}_${IT}_ho_*.log | sed 's/.*total falls: //' | awk '{f += $1; n += $3} END {print f " / " n}')"
      ;;
    lag)
      L=$LOGS/${KEY}_${IT}_lag.log
      $PY -m r1wb.eval.stream_reward_by_command "${LAG_TASK:-wb_task_teacher}" "$CK" teacher 20 > "$L" 2>&1
      echo "== ${LAG_TASK:-wb_task_teacher} driven by $KEY@$IT: exit $? $(stamp)"; trace "$L"; awk '/^task /{p=1} p' "$L" | cut -c1-270
      ;;
    track)
      L=$LOGS/${KEY}_${IT}_track.log
      $PY -m r1wb.eval.eval_tracking --run "$RUN" --checkpoint "$CK" --task wb_goal_student --label --app-pid "$OWNER" > "$L" 2>&1
      echo "== $KEY $IT labels replay every clip (student's scene and criteria): exit $? $(stamp)"; trace "$L"
      python3 - "$RENDER/clip_eval_train_label_$IT.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1])); s = a["summary"]
f = lambda x: "  -  " if x is None else f"{x:5.2f}"
t = s.get("trained_clips", {})
print(f"   clips {s['clips']}  completed {s['completed']}  strict {s['passed_strict']}  unprompted footfalls {sum(v['unprompted_footfalls'] for v in s['by_context'].values())}  | trained subset: {t.get('clips')} clips, completed {t.get('completed')}, strict {t.get('passed_strict')}")
print(f"   {'context':12s} clips done strict  distance  yaw   twist err  height p95  footfalls/ref  unprompted")
for k, v in s["by_context"].items():
    print(f"   {k:12s} {v['clips']:5d} {v['completed']:4d} {v['passed_strict']:6d}  {f(v.get('distance_ratio'))}  {f(v.get('yaw_ratio'))}  {f(v.get('twist_error_mean_mps'))}      {v['height_error_p95_m']:.3f}     {v['footfalls']}/{v['reference_footfalls']}   {v['unprompted_footfalls']}")
PYEOF
      ;;
    s2)
      PY2=".agent/rlinf/.venv/bin/python -B"                    # the environment with the vision-language stack
      for language in ${LANGUAGES:-ko en}; do
        for phrases in ${PHRASES:-train heldout}; do
          [ -z "${GOAL_MODEL:-}" ] && [ "$phrases" = heldout ] && continue     # the table goal does not read the words
          L=$LOGS/${KEY}_${IT}_s2_${HEAD}_${GOAL_MODEL:-table}${STEER:+_steer-$STEER}_${language}_${phrases}.log
          PYTHONPATH=$PWD/src $PY2 -m r1wb.eval.eval_stream --run "$RUN" --checkpoint "$CK" --policy s2 --head "$HEAD" \
            --tokens "${TOKENS:-v2c}" ${GOAL_MODEL:+--goal-model "$GOAL_MODEL" --backbone ".agent/models/$BACKBONE"} \
            ${STEER:+--steer "$STEER"} ${LIBRARY:+--library "$LIBRARY"} ${STAGGER:+--stagger "$STAGGER"} ${REPEATS:+--repeats "$REPEATS"} \
            --language "$language" --phrases "$phrases" --match-pose reference --commit --suite compass --seconds 19.84 \
            --app-pid "$OWNER" > "$L" 2>&1
          echo "== $KEY $IT System 2 (head $HEAD, goal ${GOAL_MODEL:-table}, loop ${STEER:-none}, $language, $phrases phrasings): exit $? $(stamp)"; trace "$L"
          sed -n '/^script /,$p' "$L" | cut -c1-135
          grep -E '"(falls|path_ratio|direction_error_deg|heading_drift_deg|quiet_touchdowns_per_min|quiet_arm_speed_rps|quiet_leg_speed_rps|height_error_m)"' "$L" | tr -d '\n' | cut -c1-400; echo
        done
      done
      ;;
    *) echo "unknown: $what" ;;
  esac
done
# Pack self-contained archives plus their original results/traces. Never delete server originals here.
pack_failed=0
if [ -d ".agent/result/video/$NAME" ]; then
  mkdir -p .agent/result/transit
  if $PY -m r1wb.eval.vector_evaluation_video --localize-tree ".agent/result/video/$NAME"; then
    packed=("video/$NAME")
    for kind in render logs; do [ ! -d ".agent/result/$kind/$NAME" ] || packed+=("$kind/$NAME"); done
    bundle=".agent/result/transit/${NAME}__$(date +%Y%m%d-%H%M%S-%N).tar.zst"
    if tar -C .agent/result -cf - "${packed[@]}" | zstd -T8 --long=27 -6 -q -o "$bundle"; then
      sha256sum "$bundle" > "$bundle.sha256"
    else
      echo "archive packing failed; source and partial bundle retained: $bundle"; pack_failed=1
    fi
  else
    echo "archive asset verification failed; no source files removed"; pack_failed=1
  fi
fi
echo "battery done $KEY $IT $(stamp) archive_pack_status=$pack_failed"
exit "$pack_failed"
