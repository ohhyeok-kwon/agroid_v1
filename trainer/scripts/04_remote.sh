#!/usr/bin/env bash
# Run a command on the GPU server in /rl-agent with the GPU + headless EGL env (eval, rendering, checks).
#   scripts/04_remote.sh <command...>              foreground   e.g. scripts/04_remote.sh nvidia-smi
#   scripts/04_remote.sh -d <name> <command...>    detached; log .agent/runs/<name>_<stamp>_<pid>.log
#   scripts/04_remote.sh --gpus 4 ...              select four least-used GPUs
#   GPU=3,1,0,2 scripts/04_remote.sh --gpus 4 ...   explicit visible-device order
set -euo pipefail
cd "$(dirname "$0")/.."
host="${RL_REMOTE:-super-via-ailab}"
rdir="${RL_REMOTE_DIR:-/rl-agent}"
name=""
gpus=1
while [ $# -gt 0 ]; do
  case "$1" in
    -d)
      [ $# -ge 2 ] || { printf '%s\n' '-d needs a name' >&2; exit 2; }
      name="$2"; shift 2 ;;
    --gpus)
      [ $# -ge 2 ] || { printf '%s\n' '--gpus requires a positive count' >&2; exit 2; }
      gpus="$2"; shift 2 ;;
    --gpus=*) gpus="${1#*=}"; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--gpus requires a positive count' >&2; exit 2; }
[ $# -gt 0 ] || { sed -n '2,6p' "$0"; exit 2; }

scripts/00_initialize.sh --once >/dev/null   # server has the latest sources
log=""
[ -z "$name" ] || log=".agent/runs/${name//[^[:alnum:]_.-]/_}_$(date +%Y%m%d-%H%M%S)_${BASHPID}.log"
{
printf 'set -euo pipefail\n'
printf 'rdir=%q\nrequested_gpus=%q\ngpu_count=%q\nlog=%q\n' "$rdir" "${GPU:-}" "$gpus" "$log"
printf 'command=('
printf ' %q' "$@"
printf ' )\n'
cat <<'REMOTE'
cd -- "$rdir"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ -f /opt/nvidia-egl/10_nvidia.json ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=/opt/nvidia-egl/10_nvidia.json
  export LD_LIBRARY_PATH=/opt/nvidia-egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
inventory="$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)"
selected=()
if [ -n "$requested_gpus" ]; then
  [[ "$requested_gpus" != ,* && "$requested_gpus" != *, && "$requested_gpus" != *,,* ]] || {
    printf 'GPU must be a comma-separated list without empty entries\n' >&2; exit 2;
  }
  IFS=, read -r -a requested <<< "$requested_gpus"
  for gpu in "${requested[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    resolved=""
    while IFS=, read -r index uuid memory; do
      index="${index//[[:space:]]/}"; uuid="${uuid//[[:space:]]/}"
      if [ "$gpu" = "$index" ] || [ "$gpu" = "$uuid" ]; then resolved="$index"; break; fi
    done <<< "$inventory"
    [ -n "$resolved" ] || { printf 'GPU device is unavailable: %s\n' "$gpu" >&2; exit 2; }
    for existing in "${selected[@]}"; do
      [ "$existing" != "$resolved" ] || { printf 'GPU device is repeated: %s\n' "$gpu" >&2; exit 2; }
    done
    selected+=("$resolved")
  done
  [ "${#selected[@]}" = "$gpu_count" ] || {
    printf 'GPU selects %s devices but --gpus requests %s\n' "${#selected[@]}" "$gpu_count" >&2; exit 2;
  }
else
  mapfile -t available < <(printf '%s\n' "$inventory" | sort -t, -k3,3n -k1,1n | awk -F, '{gsub(/[[:space:]]/, "", $1); if ($1 != "") print $1}')
  [ "${#available[@]}" -ge "$gpu_count" ] || {
    printf 'Requested %s GPUs, but server exposes only %s\n' "$gpu_count" "${#available[@]}" >&2; exit 2;
  }
  selected=("${available[@]:0:gpu_count}")
fi
gpu_list="$(IFS=,; printf '%s' "${selected[*]}")"
export CUDA_VISIBLE_DEVICES="$gpu_list"
if [ -n "$log" ]; then
  mkdir -p .agent/runs
  nohup "${command[@]}" > "$log" 2>&1 < /dev/null &
  printf 'server pid %s gpu %s log %s/%s\n' "$!" "$gpu_list" "$rdir" "$log"
else
  exec "${command[@]}" < /dev/null
fi
REMOTE
} | ssh "$host" bash -s
