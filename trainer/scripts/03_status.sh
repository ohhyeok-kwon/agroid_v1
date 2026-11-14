#!/usr/bin/env bash
# GPU use, running trainings and the latest log lines on the GPU server.
set -euo pipefail
host="${RL_REMOTE:-super-via-ailab}"
rdir="${RL_REMOTE_DIR:-/rl-agent}"
ssh "$host" bash -s <<EOF
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- runs"
pgrep -af "r1lab.train" || echo "(실행 중인 학습 없음)"
cd "$rdir"
for f in \$(ls -t .agent/runs/*.log 2>/dev/null | head -3); do
  echo "--- \$f"
  grep -aE "Learning iteration|Mean reward|Mean episode length|Traceback|Error" "\$f" | tail -3
done
EOF
systemctl --user is-active trainer-sync >/dev/null 2>&1 && echo "trainer-sync: 서비스 실행 중" || echo "trainer-sync: 서비스 꺼짐 (scripts/00_initialize.sh)"
