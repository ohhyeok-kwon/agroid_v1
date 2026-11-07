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

case ${1:-} in
    -h|--help)
        printf '사용법: ./scripts/daemon/03_restart_service.sh\n'
        printf 'boot_manager.service를 재시작합니다. R1에서 실행하세요.\n'
        exit 0 ;;
    '') ;;
    *) printf '오류: 인자를 받지 않습니다 (--help 참고).\n' >&2; exit 2 ;;
esac
service=boot_manager.service
action=restart
if [[ $action == restart ]] && ! systemctl is-active --quiet "$service"; then action=start; fi
if (( EUID == 0 )); then
    systemctl "$action" "$service"
else
    sudo -- systemctl "$action" "$service"
fi
printf '%s 재시작 완료\n' "$service"
systemctl status "$service" --no-pager --full || true
