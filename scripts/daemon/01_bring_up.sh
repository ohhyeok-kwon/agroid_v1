#!/usr/bin/env bash
set -euo pipefail

# OS나 서비스를 변경하기 전에 실제 R1 하드웨어인지 확인합니다.
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
    printf "오류: daemon 스크립트는 R1에서만 실행할 수 있습니다 (아키텍처: %s).\n" "$architecture" >&2
    return 1
}
require_r1 || exit 1

# 설치된 boot_manager ROS 2 패키지를 런치합니다. src/는 참조하지 않습니다.
BRINGUP_COMMAND=(ros2 launch boot_manager boot_manager.launch.xml)

usage() {
    cat <<'USAGE'
사용법: ./scripts/daemon/01_bring_up.sh [--check] [-- 실행파일 인자...]

/opt/ros/<배포판>/setup.bash와 이 프로젝트의 install/local_setup.bash를 불러온 뒤 실행합니다.
--check는 ROS 환경·설치된 패키지 확인만 하고 실행하지 않습니다.
최초 준비: PC에서 ./scripts/03_deploy.sh로 빌드·배포·서비스 등록
ROS_DISTRO가 없으면 /opt/ros 아래 setup.bash가 있는 단일 배포판을 선택합니다.
USAGE
}

check_only=false
while (( $# )); do
    case $1 in
        --help|-h) usage; exit 0 ;;
        --check) check_only=true; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if (( $# )); then
    BRINGUP_COMMAND=("$@")
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_dir=$(cd -- "$script_dir/../.." && pwd -P)
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
local_setup="$project_dir/install/local_setup.bash"
[[ -r $local_setup ]] || { printf '오류: 프로젝트 설치본이 없습니다: %s (PC에서 빌드·배포하세요)\n' "$local_setup" >&2; exit 1; }

# ROS 환경 스크립트는 정의되지 않은 셸 변수를 참조할 수 있습니다.
set +u
# shellcheck disable=SC1090,SC1091
source "$ros_setup"
source "$local_setup"
set -u
[[ ${ROS_VERSION:-} == 2 ]] || { printf '오류: ROS 2 환경이 필요합니다.\n' >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ROS_LOG_DIR="$project_dir/log/bringup"

# 새 패키지도 검사하도록 배포 명세 전체를 사용합니다. 외부 설치본으로 대체하지 않습니다.
package_names=$(python3 - "$project_dir/manifest.json" <<'R1_PACKAGES_PYTHON'
import json
from pathlib import Path
import re
import sys

manifest = Path(sys.argv[1])
packages = json.loads(manifest.read_text())['packages']
if not isinstance(packages, dict) or 'boot_manager' not in packages:
    sys.exit('오류: 배포 명세의 패키지 목록이 잘못되었습니다.')
for name, binaries in sorted(packages.items()):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', name):
        sys.exit('오류: 잘못된 배포 패키지 이름: ' + str(name))
    for binary in binaries:
        path = manifest.parent / 'install/lib' / name / binary
        if not path.is_file() or not path.stat().st_mode & 0o111:
            sys.exit('오류: 배포 실행 파일이 없습니다: ' + str(path))
    print(name)
R1_PACKAGES_PYTHON
)
while IFS= read -r package; do
    prefix=$(ros2 pkg prefix "$package" 2>/dev/null) || prefix=
    if [[ $prefix != "$project_dir/install" && $prefix != "$project_dir/install/"* ]]; then
        printf '오류: %s 설치본이 %s/install에 없습니다.\n' "$package" "$project_dir" >&2
        exit 1
    fi
done <<< "$package_names"

command_path=$(type -P -- "${BRINGUP_COMMAND[0]}") || command_path=
[[ -f $command_path && -x $command_path ]] || {
    printf '오류: 실행파일을 찾을 수 없습니다: %s\n' "${BRINGUP_COMMAND[0]}" >&2
    exit 1
}
if "$check_only"; then
    printf '실행 환경 점검 완료: ROS_DISTRO=%s\n실행 명령:' "${ROS_DISTRO:-unknown}"
    printf ' %q' "${BRINGUP_COMMAND[@]}"
    printf '\n'
    exit 0
fi

# Fast DDS는 노드 생성 시점의 네트워크 인터페이스로 고정됩니다. PC 직결(R1 AP) 주소가
# 생긴 뒤 시작해야 PC와 ROS 통신이 됩니다. 최대 30초만 기다리고, 없으면 경고 후 시작합니다.
ap_interface=wlan1
if [[ -r /etc/agroid/network/hostapd.conf ]]; then
    configured=$(sed -n 's/^interface=//p' /etc/agroid/network/hostapd.conf | head -n 1)
    [[ $configured =~ ^[A-Za-z0-9_.-]+$ ]] && ap_interface=$configured
fi
ap_ready() {
    [[ $(cat "/sys/class/net/$ap_interface/operstate" 2>/dev/null) == up ]] &&
        ip -4 -o address show dev "$ap_interface" 2>/dev/null | grep -q ' inet '
}
for (( waited = 0; waited < 300; waited++ )); do
    ap_ready && break
    sleep 0.1
done
if ap_ready; then
    printf 'R1 AP 인터페이스 준비: %s (%d.%d초 대기)\n' "$ap_interface" $((waited / 10)) $((waited % 10))
else
    printf '경고: %s IPv4 주소 없이 시작합니다. PC와 ROS 통신이 되지 않을 수 있습니다.\n' "$ap_interface" >&2
fi

printf 'AGROID bringup 시작:'
printf ' %q' "${BRINGUP_COMMAND[@]}"
printf '\n'
# 셸을 교체하여 systemd의 종료 신호와 종료 상태를 실제 프로세스에 전달합니다.
exec "${BRINGUP_COMMAND[@]}"
