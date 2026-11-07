#!/usr/bin/env bash
# R1에서 실행: 동기화된 src/의 ROS 2 패키지 전체를 Release(최적화)로 빌드해 install/에 설치한다.
#   Rust(ament_cargo): cargo --release, LTO·codegen-units=1. CMake 패키지: CMAKE_BUILD_TYPE=Release.
#   --restart  빌드 뒤 boot_manager.service를 다시 시작한다.
# 동기화로 받은 파일은 수정 시각이 지난 빌드보다 과거일 수 있어 cargo·make가 놓친다:
# 마지막 성공 빌드와 내용이 다른 파일은 빌드 전에 touch 한다. colcon은 이 스크립트로만 부른다.
# 빌드 도구(rustup, ROS Python의 pip·colcon·cargo-ament-build)가 없으면 먼저 설치한다.
set -euo pipefail

# OS나 빌드 결과를 바꾸기 전에 실제 R1 하드웨어인지 확인합니다.
require_r1() {
    local architecture model device_tree
    architecture=$(/usr/bin/uname -m) || architecture=unknown
    if [[ $architecture == aarch64 ]]; then
        for device_tree in /sys/firmware/devicetree/base /proc/device-tree; do
            [[ -r $device_tree/model ]] || continue
            model=
            IFS= read -r -d "" model < "$device_tree/model" || true
            [[ $model == "Unitree G1 Board" ]] && return 0
        done
    fi
    printf "오류: 빌드 스크립트는 R1에서만 실행할 수 있습니다 (아키텍처: %s).\n" "$architecture" >&2
    return 1
}

usage() {
    cat <<'EOF'
사용법: ./scripts/03_build.sh [--restart]

R1에서 src/ 전체를 Release로 빌드합니다(colcon, 결과: build/ install/ log/).
  --restart  빌드 뒤 boot_manager.service를 다시 시작
ROS_DISTRO가 없으면 /opt/ros 아래 setup.bash가 있는 단일 배포판을 선택합니다.
EOF
}

restart=false
case ${1:-} in
    '') ;;
    --restart) restart=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
(( $# <= 1 )) || { usage >&2; exit 2; }
require_r1 || exit 1

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_dir"

if [[ -n ${ROS_DISTRO:-} ]]; then
    [[ $ROS_DISTRO =~ ^[a-zA-Z0-9_-]+$ ]] || { printf '오류: 올바르지 않은 ROS_DISTRO입니다.\n' >&2; exit 2; }
    ros_setup="/opt/ros/$ROS_DISTRO/setup.bash"
else
    shopt -s nullglob
    ros_setups=(/opt/ros/*/setup.bash)
    shopt -u nullglob
    if (( ${#ros_setups[@]} != 1 )); then
        printf '오류: ROS 2 설치를 하나로 결정할 수 없습니다. ROS_DISTRO를 지정하세요.\n' >&2
        exit 1
    fi
    ros_setup=${ros_setups[0]}
fi
[[ -r $ros_setup ]] || { printf '오류: ROS 환경 파일을 읽을 수 없습니다: %s\n' "$ros_setup" >&2; exit 1; }
# ROS 환경 스크립트는 정의되지 않은 셸 변수를 참조할 수 있습니다.
set +u
# shellcheck disable=SC1090
source "$ros_setup"
set -u
# ROS 환경의 python3 = rclpy가 있는 ROS Python. colcon과 확장도 여기에 둡니다.
python=$(command -v python3)

# ==================== 빌드 도구 (없을 때만 설치) ====================
# root로 실행하는 R1 전용 Python이므로 pip의 root 경고와 새 버전 알림을 끕니다.
export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
    printf '[build] Rust 설치 (rustup, stable, minimal)\n'
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
fi
if ! "$python" -m pip --version >/dev/null 2>&1; then
    printf '[build] ROS Python에 pip 설치\n'
    # ensurepip은 PIP_* 설정을 무시해 root 경고를 냅니다: 내장 pip 휠을 직접 실행합니다.
    pip_wheel=$("$python" -c 'import ensurepip, pathlib; print(next(pathlib.Path(ensurepip.__file__).with_name("_bundled").glob("pip-*.whl")))')
    "$python" "$pip_wheel/pip" install --no-index "$pip_wheel"
fi
if ! "$python" -c 'import colcon_core, colcon_ros, colcon_cmake, colcon_ros_cargo' >/dev/null 2>&1 ||
    ! command -v colcon cargo-ament-build >/dev/null 2>&1; then
    printf '[build] colcon·Rust 확장 설치 (ROS Python)\n'
    "$python" -m pip install --upgrade \
        colcon-common-extensions colcon-cargo colcon-ros-cargo
fi
# Lyrical interface generation requires CMake >= 3.20 (R1 Ubuntu 20.04 ships 3.16).
export PATH="$("$python" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
if ! "$python" - <<'PY'
import subprocess
try:
    version = subprocess.check_output(['cmake', '--version'], text=True).split()[2]
    raise SystemExit(0 if tuple(map(int, version.split('.')[:2])) >= (3, 20) else 1)
except (OSError, ValueError):
    raise SystemExit(1)
PY
then
    printf '[build] ROS 메시지 생성용 CMake 설치\n'
    "$python" -m pip install 'cmake>=3.20,<4'
    export PATH="$("$python" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
fi
# ===================================================================

state=build/.build_inputs.sha256
mkdir -p build
find src -type f ! -path '*/target/*' ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 |
    LC_ALL=C sort -z | xargs -0r sha256sum > "$state.new"
if [[ -f $state ]]; then
    LC_ALL=C comm -13 <(LC_ALL=C sort "$state") <(LC_ALL=C sort "$state.new") |
        cut -c67- | xargs -r -d '\n' touch --
fi

# 모든 Rust 패키지에 같은 최적화를 적용합니다(패키지별 Cargo.toml 설정보다 우선).
export CARGO_PROFILE_RELEASE_LTO=true CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
# R1의 GNU ld는 ROS 공유 라이브러리 심볼 탐색이 매우 느립니다.
# Rust에 포함된 네이티브 LLVM 링커를 사용하되 명시한 링커 설정은 보존합니다.
rust_linkers="$(dirname -- "$(rustc --print target-libdir)")/bin/gcc-ld"
if [[ -x "$rust_linkers/ld.lld" && ${RUSTFLAGS:-} != *-fuse-ld=* && ${RUSTFLAGS:-} != *linker=* &&
      ! -v CARGO_ENCODED_RUSTFLAGS && ! -v CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER ]]; then
    export PATH="$rust_linkers:$PATH"
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-fuse-ld=lld"
    printf '[build] Rust LLVM 링커 사용\n'
fi
# Rust 패키지는 순서대로 빌드합니다: colcon-ros-cargo가 .cargo/config.toml을 공유합니다.
# 인코딩된 Rust 플래그는 공백이 있는 프로젝트 경로도 그대로 전달합니다.
export CARGO_ENCODED_RUSTFLAGS
CARGO_ENCODED_RUSTFLAGS=$("$python" - <<'PY'
import os
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET
encoded = os.environ.get('CARGO_ENCODED_RUSTFLAGS')
flags = encoded.split('\x1f') if encoded else shlex.split(os.environ.get('RUSTFLAGS', ''))
# ros-env includes generated Rust source, so dependency build.rs scripts do not run.
# Interface libraries are generated earlier in colcon's dependency order.
for manifest in Path('src').rglob('package.xml'):
    package = ET.parse(manifest).getroot()
    if any(group.text == 'rosidl_interface_packages' for group in package.findall('member_of_group')):
        library = (Path('install') / package.findtext('name') / 'lib').resolve()
        flags.append('-Lnative=' + str(library))
print('\x1f'.join(flags))
PY
)
colcon build --base-paths src --executor sequential \
    --cargo-args " --release" \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
mv -- "$state.new" "$state"
printf '[build] 빌드 완료 %s\n' "$(date '+%F %T')"

"$restart" || exit 0
"$project_dir/scripts/daemon/03_restart_service.sh"
