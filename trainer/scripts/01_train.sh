#!/usr/bin/env bash
# Start a training run on the GPU server (super-via-ailab:/rl-agent), detached; logs land in .agent/runs/.
#   scripts/01_train.sh <task> [r1lab.train args...]       e.g. scripts/01_train.sh careful_walk_s1 --iters 1500
#   scripts/01_train.sh careful_walk --gpus 4 --num-envs 4096  (one shared policy)
#   GPU=2 scripts/01_train.sh safe_crouch --num-envs 4096      (default: least-used GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
usage() { printf 'usage: scripts/01_train.sh <task> [--gpus N] [r1lab.train args...]\n'; }
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then usage; exit 0; fi
[ $# -gt 0 ] || { usage >&2; exit 2; }
task="$1"
shift
gpus=1
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)
            [ $# -ge 2 ] || { printf '%s\n' '--gpus requires a positive count' >&2; exit 2; }
            gpus="$2"; shift 2 ;;
        --gpus=*) gpus="${1#*=}"; shift ;;
        *) args+=("$1"); shift ;;
    esac
done
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--gpus requires a positive count' >&2; exit 2; }
if ! systemctl --user is-active --quiet trainer-sync.service; then
    printf '학습 등록을 PC에 동기화하려면 trainer-sync가 실행 중이어야 합니다. trainer/scripts/00_initialize.sh --install-service로 시작하세요.\n' >&2
    exit 1
fi
# Run ID = <action>_<local date-time>_<key settings> (r1lab.train); run folder .agent/runs/<exp>/<ID>/
command=(.agent/.venv/bin/python -m r1lab.train)
if [ "$gpus" != 1 ]; then
    command=(.agent/.venv/bin/torchrun --standalone "--nproc-per-node=$gpus" -m r1lab.train)
fi
# Rank zero publishes its run record under this shared, unique launch identity.
launch_id="$(date +%Y%m%d-%H%M%S)-${BASHPID}-${RANDOM}${RANDOM}"
scripts/04_remote.sh --gpus "$gpus" -d "$task" env \
  RL_STAMP="$(date +%Y%m%d-%H%M)" RL_LAUNCH_ID="$launch_id" \
  "${command[@]}" --task "$task" "${args[@]}"
echo "학습 ID·상태·로그는 자동 등록되어 Operation Center의 Trainer에 표시됩니다. 상태: scripts/03_status.sh"
