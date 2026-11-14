#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
사용법: ./scripts/02_connect_ssh.sh

R1의 최신 배포(current) 폴더에서 root SSH 셸을 엽니다. 파일을 업로드하지 않습니다.
아직 배포하지 않았다면 프로젝트 폴더에서 엽니다.
접속 정보(현재 IP·고정 호스트 키)는 00_initialize.sh가 만든 .agent/.r1-setup/ssh_config를 씁니다.
비밀번호는 SSH 프롬프트에서 입력하며, 이 스크립트에 저장하지 않습니다.

환경변수: AGROID_R1_PROJECT_DIR (기본: /root/workspace/agroid_v1)
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
config="$project_dir/.agent/.r1-setup/ssh_config"
remote_dir=${AGROID_R1_PROJECT_DIR:-/root/workspace/agroid_v1}

if [[ ! -r $config ]]; then
    printf '오류: %s가 없습니다. 먼저 ./scripts/00_initialize.sh로 R1에 연결하세요.\n' "$config" >&2
    exit 1
fi
if [[ $remote_dir != /* || $remote_dir == *$'\n'* || $remote_dir == *$'\r'* ]]; then
    printf '오류: AGROID_R1_PROJECT_DIR는 R1의 절대 경로여야 합니다.\n' >&2
    exit 2
fi
if ! command -v ssh >/dev/null 2>&1; then
    printf '오류: 로컬 PC에 OpenSSH 클라이언트가 필요합니다.\n' >&2
    exit 1
fi

# POSIX shell quoting: the remote path may contain spaces or apostrophes.
quoted_dir="'${remote_dir//\'/\'\\\'\'}'"
rc=0
ssh -F "$config" -t \
    -o ForwardAgent=no \
    -o ForwardX11=no \
    -o ClearAllForwardings=yes \
    r1 "cd -- $quoted_dir && { if [ -d current ]; then cd -- current; fi; } && exec bash -l" || rc=$?
if (( rc == 255 )); then
    # The direct link (wlr1) disappears when the PC reboots; the init script recreates it.
    printf 'R1 SSH 연결 실패. R1 직결(wlr1)이 없거나 끊겼다면 ./scripts/00_initialize.sh를 실행하세요.\n' >&2
fi
exit "$rc"
