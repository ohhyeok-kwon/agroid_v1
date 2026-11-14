#!/usr/bin/env bash
# Build and open Operation Center. All generated files remain under .agent/.
set -euo pipefail
oc_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
oc_app="$oc_root/app/operation_center"
oc_work="$oc_root/.agent/operation_center"
oc_mode=desktop
oc_args=()
for oc_arg in "$@"; do
    case "$oc_arg" in
        --headless) oc_mode=headless ;;
        --build-only) oc_mode=build ;;
        --help|-h)
            printf 'Usage: %s [--headless [--port PORT] [--with-results] | --build-only]\n' "$0"
            exit 0 ;;
        *) oc_args+=("$oc_arg") ;;
    esac
done
mkdir -p "$oc_work/node" "$oc_root/.agent/tmp"
export TMPDIR="$oc_root/.agent/tmp" TMP="$oc_root/.agent/tmp" TEMP="$oc_root/.agent/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OPERATION_CENTER_ROOT="$oc_app"
export CARGO_TARGET_DIR="$oc_work/target"
export CARGO_HOME="$oc_work/cargo"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

for oc_tool in python3 node npm; do
    if ! command -v "$oc_tool" >/dev/null 2>&1; then
        printf '필요한 도구가 없습니다: %s\n' "$oc_tool" >&2
        exit 1
    fi
done

# Serialize builds; the running application never holds the build lock.
exec 9>"$oc_work/build.lock"
flock 9
if ! cmp -s "$oc_app/package-lock.json" "$oc_work/node/package-lock.json" || \
   [ ! -d "$oc_work/node/node_modules/three" ] || [ ! -d "$oc_work/node/node_modules/esbuild" ]; then
    cp "$oc_app/package.json" "$oc_app/package-lock.json" "$oc_work/node/"
    npm ci --prefix "$oc_work/node" --cache "$oc_work/npm-cache" --no-audit --no-fund
fi
oc_hash="$(python3 - "$oc_app" <<'PY'
import hashlib, sys
from pathlib import Path
app = Path(sys.argv[1])
digest = hashlib.sha256()
paths = sorted((app/'ui').rglob('*')) + [app/'build.mjs', app/'package-lock.json', app/'startup.json', app/'assets/robot/sensors.json']
for path in paths:
    if path.is_file():
        digest.update(str(path.relative_to(app)).encode())
        digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)"
if [ ! -f "$oc_work/ui/app.js" ] || [ ! -f "$oc_work/ui.sha256" ] || \
   [ "$(cat "$oc_work/ui.sha256")" != "$oc_hash" ]; then
    node "$oc_app/build.mjs"
    printf '%s\n' "$oc_hash" > "$oc_work/ui.sha256"
fi
if [ "$oc_mode" != headless ]; then
    mkdir -p "$oc_work/tauri-gen"
    if [ ! -e "$oc_app/gen" ] && [ ! -L "$oc_app/gen" ]; then
        ln -s ../../.agent/operation_center/tauri-gen "$oc_app/gen"
    fi
    oc_cargo="$(command -v cargo || true)"
    if [ -z "$oc_cargo" ] && [ -x "$HOME/.cargo/bin/cargo" ]; then
        oc_cargo="$HOME/.cargo/bin/cargo"
    fi
    if [ -z "$oc_cargo" ]; then
        printf '데스크톱 창 빌드에 Rust가 필요합니다. --headless로 브라우저 화면을 실행할 수 있습니다.\n' >&2
        exit 1
    fi
    PATH="$(dirname -- "$oc_cargo"):$PATH" "$oc_cargo" build --manifest-path "$oc_app/Cargo.toml"
fi
oc_ros_setup="${AGROID_ROS_SETUP:-/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash}"
if [ -f "$oc_ros_setup" ]; then
    set +u
    source "$oc_ros_setup"
    set -u
else
    printf 'ROS 2 환경이 없어 3D 모델 화면만 실행합니다.\n' >&2
fi
export AGROID_PROJECT_ROOT="$oc_root"
export ROS_HOME="$oc_root/.agent/ros2/runtime" ROS_LOG_DIR="$oc_root/.agent/ros2/runtime/log"
export ROS_DOMAIN_ID="${AGROID_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
mkdir -p "$ROS_LOG_DIR"
flock -u 9
exec 9>&-
case "$oc_mode" in
    build) printf 'Operation Center 빌드 완료\n' ;;
    headless) exec python3 "$oc_app/server.py" "${oc_args[@]}" ;;
    desktop)
        if [ "${#oc_args[@]}" -ne 0 ]; then
            printf '데스크톱 모드에는 추가 인자가 없습니다. --help를 확인하세요.\n' >&2
            exit 1
        fi
        exec "$CARGO_TARGET_DIR/debug/operation_center" ;;
esac
