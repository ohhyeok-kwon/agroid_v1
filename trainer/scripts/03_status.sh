#!/usr/bin/env bash
# GPU use, running trainings and the latest log lines on the GPU server.
set -euo pipefail
host="${RL_REMOTE:-super-via-ailab}"
rdir="${RL_REMOTE_DIR:-/rl-agent}"
ssh "$host" bash -s <<EOF
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- runs"
r1lab=\$(pgrep -af "r1lab.train")
[ -n "\$r1lab" ] && echo "\$r1lab"
# Registered runs (.agent/runs/*/*/run_info.json, e.g. p31 tracker rounds): running when the state is starting|running
# and the recorded pid or a worker pid is alive and started before the record's last write (a reused pid starts later).
R1LAB="\$r1lab" RUNS="$rdir/.agent/runs" PYTHONIOENCODING=utf-8 python3 - <<'PY'
import json, os, time
from pathlib import Path
boot = int(next(l for l in open('/proc/stat') if l.startswith('btime')).split()[1])
def alive(info):
    last = (info.get('updated_at') or info.get('created_at') or 0) + 5
    for q in [info.get('pid')] + [w.get('pid') for w in info.get('workers') or [] if isinstance(w, dict)]:
        try:
            s = open('/proc/%d/stat' % q).read()
            if boot + int(s[s.rfind(')') + 2:].split()[19]) / os.sysconf('SC_CLK_TCK') <= last:
                return True
        except (OSError, TypeError, ValueError):
            pass
    return False
rows = []
for f in Path(os.environ['RUNS']).glob('*/*/run_info.json'):
    try:
        info = json.loads(f.read_text())
    except (OSError, ValueError):
        continue
    if isinstance(info, dict) and isinstance(info.get('id'), str):
        rows.append(info)
listed = {line.split()[0] for line in os.environ.get('R1LAB', '').splitlines() if line.strip()}
live = [i for i in rows if i.get('state') in ('starting', 'running') and alive(i)]
for i in live:
    if str(i.get('pid')) not in listed:
        print('%s [등록] %s 반복 %s/%s' % (i.get('pid'), i['id'], i.get('iteration'), i.get('total_iterations')))
if not live and not listed:
    print('(실행 중인 학습 없음)')
print('--- 등록 판 (최근 3)')
names = {'completed': '끝남', 'stopped': '멈춤', 'failed': '실패'}
for i in sorted(rows, key=lambda i: (i in live, i.get('finished_at') or i.get('updated_at') or 0), reverse=True)[:3]:
    state = i.get('state')
    label = '도는 중' if i in live else names.get(state) or '기록 %s, 살아 있는 프로세스 없음' % state
    end = i.get('finished_at')
    why = (i.get('p31') or {}).get('stop_line') or i.get('error') or ''
    print('%s | %s | 반복 %s/%s%s%s' % (label, i['id'], i.get('iteration'), i.get('total_iterations'),
          time.strftime(' | 끝 %m-%d %H:%M KST', time.gmtime(end + 9 * 3600)) if isinstance(end, (int, float)) else '',
          ' | ' + str(why) if why and state != 'running' else ''))
PY
cd "$rdir"
for f in \$(ls -t .agent/runs/*.log 2>/dev/null | head -3); do
  echo "--- \$f"
  grep -aE "Learning iteration|Mean reward|Mean episode length|Traceback|Error" "\$f" | tail -3
done
EOF
systemctl --user is-active trainer-sync >/dev/null 2>&1 && echo "trainer-sync: 서비스 실행 중" || echo "trainer-sync: 서비스 꺼짐 (scripts/00_initialize.sh)"
