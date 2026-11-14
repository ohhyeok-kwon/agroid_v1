#!/usr/bin/env bash
# R1 초기 부팅에 boot_manager.service(휴머노이드 기동)를 등록·실행·복원한다.
# 서비스 파일 템플릿과 설치 로직을 이 스크립트에 포함한다. --help 참고.
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
case ${1:-} in --print-unit|-h|--help) ;; *) require_r1 || exit 1 ;; esac

script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
exec /usr/bin/python3 -B -c '
import sys
script = sys.argv.pop(1)
sys.argv[0] = script
with open(3) as source:
    code = compile(source.read(), script, "exec")
exec(code, {"__name__": "__main__", "__file__": script})
' "$script" "$@" 3<<'HUMANOID_SERVICE_PYTHON'
"""R1 service installation using GreenFarm's snapshot/verify/rollback pattern.

Only boot_manager.service is managed. Network, boot guard and vendor units are
left to their existing owners. This script requires no third-party modules.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

SERVICE = 'boot_manager.service'
MARKER = '# Managed by AGROID boot_manager.'
PROJECT = Path(__file__).resolve().parents[2]
UNIT_TEMPLATE = '# Managed by AGROID boot_manager.\n# Runtime SHA256: @RUNTIME_SHA256@\n[Unit]\nDescription=AGROID R1 humanoid bringup (boot_manager)\nAfter=network.target\nRequiresMountsFor=@PROJECT_DIR@\nStartLimitIntervalSec=30\nStartLimitBurst=3\n\n[Service]\nType=exec\nUser=root\nWorkingDirectory=@WORKING_DIR@\nExecStartPre=-/bin/bash @RT_TUNING_SCRIPT@\nLimitRTPRIO=95\nLimitMEMLOCK=infinity\nCPUAffinity=0-6\nOOMScoreAdjust=-900\nExecStart=/bin/bash @BRINGUP_SCRIPT@\nEnvironment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=AGROID_BOOT_REVISION=@RUNTIME_SHA256@\nRestart=on-failure\nRestartSec=2\nTimeoutStartSec=5\nTimeoutStopSec=5\nKillMode=control-group\nKillSignal=SIGTERM\nStandardOutput=journal\nStandardError=journal\n\n[Install]\nWantedBy=multi-user.target\n'


def command(*args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError('{}: {}'.format(' '.join(map(str, args)), result.stderr.strip() or result.stdout.strip()))
    return result.stdout.strip()


def unit_quote(value, executable=False):
    value = str(value)
    if any(ord(char) < 32 for char in value):
        raise RuntimeError('제어 문자가 포함된 경로는 사용할 수 없습니다.')
    value = value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
    if executable:
        value = value.replace('$', '$$')
    return '"' + value + '"'


def render_unit(project):
    daemon = project / 'scripts/daemon'
    # 01_bring_up.sh가 설치된 boot_manager 패키지를 런치하고, 04_realtime_tuning.sh가 기동 전 실시간 설정을 합니다.
    # 둘 중 하나라도 바뀌면 재시작 판단에 씁니다.
    digest = hashlib.sha256((daemon / '01_bring_up.sh').read_bytes() + (daemon / '04_realtime_tuning.sh').read_bytes())
    unit = UNIT_TEMPLATE.replace(
        '@RUNTIME_SHA256@', digest.hexdigest()).replace(
        '@PROJECT_DIR@', unit_quote(project)).replace(
        '@WORKING_DIR@', str(project).replace('%', '%%')).replace(
        '@BRINGUP_SCRIPT@', unit_quote(daemon / '01_bring_up.sh', executable=True)).replace(
        '@RT_TUNING_SCRIPT@', unit_quote(daemon / '04_realtime_tuning.sh', executable=True))
    if Path('/etc/systemd/system/agroid-minimal.target').is_file():
        unit = unit.replace('WantedBy=multi-user.target\n',
                            'WantedBy=multi-user.target agroid-minimal.target\n')
    return unit


def enabled_targets(state):
    # Snapshots made before target-aware registration only recorded the boolean.
    return state.get('targets', ['multi-user.target'] if state['enabled'] else [])


def wanted_targets(unit):
    return sorted(next(line.split('=', 1)[1].split() for line in unit.splitlines()
                       if line.startswith('WantedBy=')))


def require_r1():
    if platform.machine() == 'aarch64':
        for root in ('/sys/firmware/devicetree/base', '/proc/device-tree'):
            path = Path(root) / 'model'
            if path.is_file() and path.read_bytes().rstrip(b'\0\n') == b'Unitree G1 Board':
                return
    raise RuntimeError('서비스 등록은 R1 내부에서 실행하세요. PC에서는 --print-unit으로 확인할 수 있습니다.')


def atomic_write(path, text, mode=0o600):
    if path.is_file() and not path.is_symlink() and path.read_text() == text and path.stat().st_mode & 0o777 == mode:
        return False
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return True


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Installer:
    def __init__(self, project=PROJECT, system_root=Path('/')):
        self.project = project
        self.root = system_root
        self.unit = system_root / 'etc/systemd/system' / SERVICE
        self.link = self.unit.parent / 'multi-user.target.wants' / SERVICE
        self.links = {target: self.unit.parent / (target + '.wants') / SERVICE
                      for target in ('multi-user.target', 'agroid-minimal.target')}
        self.state = system_root / 'var/lib/agroid-boot-manager'
        self.pending = self.state / 'pending.json'
        self.original = self.state / 'original.json'
        self.lock = system_root / 'run/lock/agroid-boot-manager.lock'
        self.machine = (system_root / 'etc/machine-id').read_text().strip()
        self.boot = (system_root / 'proc/sys/kernel/random/boot_id').read_text().strip()

    def ctl(self, *args):
        return command('systemctl', *args)

    def inspect(self):
        values = self.ctl('show', SERVICE, '--property=LoadState,ActiveState,UnitFileState,MainPID,FragmentPath,DropInPaths,NeedDaemonReload')
        props = dict(line.split('=', 1) for line in values.splitlines() if '=' in line)
        if not props.get('LoadState'):
            raise RuntimeError('systemd 서비스 상태를 읽지 못했습니다.')
        if self.unit.is_symlink() or (self.unit.exists() and not self.unit.is_file()):
            raise RuntimeError('기존 서비스 경로가 일반 파일이 아닙니다: ' + str(self.unit))
        text = self.unit.read_text() if self.unit.is_file() else None
        if text is not None and not text.startswith(MARKER + '\n'):
            raise RuntimeError('다른 소유자의 boot_manager.service가 있습니다. 덮어쓰지 않습니다.')
        fragment = props.get('FragmentPath', '')
        if props.get('DropInPaths') or (fragment and fragment != str(self.unit)):
            raise RuntimeError('별도 서비스 파일 또는 drop-in이 있습니다. 자동 변경하지 않습니다.')
        enabled = props.get('UnitFileState') or 'disabled'
        if enabled not in ('enabled', 'disabled'):
            raise RuntimeError('지원하지 않는 서비스 등록 상태: ' + enabled)
        for target, link in self.links.items():
            if link.is_symlink() and link.resolve() != self.unit.resolve():
                raise RuntimeError(target + ' 연결이 다른 서비스 파일을 가리킵니다.')
            if link.exists() and not link.is_symlink():
                raise RuntimeError(target + ' 등록 경로가 심볼릭 링크가 아닙니다.')
        targets = sorted(target for target, link in self.links.items() if link.is_symlink())
        linked = bool(targets)
        if enabled == 'enabled' and not linked:
            raise RuntimeError('지원하지 않는 부팅 대상으로 등록된 서비스입니다. 자동 변경하지 않습니다.')
        active = props.get('ActiveState')
        if active not in ('active', 'inactive', 'failed'):
            raise RuntimeError('서비스가 상태 전환 중입니다: ' + str(active))
        pid = int(props.get('MainPID') or 0)
        revision = ''
        if pid:
            try:
                environment = (self.root / 'proc' / str(pid) / 'environ').read_bytes().split(b'\0')
                revision = next((item.split(b'=', 1)[1].decode() for item in environment
                                 if item.startswith(b'AGROID_BOOT_REVISION=')), '')
            except OSError:
                pass
        return {'unit': text, 'enabled': linked, 'targets': targets, 'active': active == 'active',
                'pid': pid, 'boot': self.boot, 'revision': revision,
                'reload': props.get('NeedDaemonReload') == 'yes'}

    def save(self, path, data):
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_write(path, json.dumps({'version': 1, 'machine': self.machine, **data}, indent=2) + '\n')

    def load(self, path):
        try:
            data = json.loads(path.read_text())
            if data['version'] != 1 or data['machine'] != self.machine:
                raise ValueError('schema/device mismatch')
            before = data['before']
            if (not isinstance(before['enabled'], bool) or not isinstance(before['active'], bool)
                    or not isinstance(before['pid'], int) or not isinstance(before['boot'], str)
                    or not (before['unit'] is None or isinstance(before['unit'], str))):
                raise ValueError('invalid snapshot')
            if before['unit'] is not None and not before['unit'].startswith(MARKER + '\n'):
                raise ValueError('unmanaged unit in snapshot')
            targets = enabled_targets(before)
            if (not isinstance(targets, list) or any(target not in self.links for target in targets)
                    or bool(targets) != before['enabled']):
                raise ValueError('invalid boot targets')
            return data
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError('복구 기록을 확인할 수 없습니다: ' + str(path)) from error

    @staticmethod
    def desired(current, unit, start):
        revision = next(line.split(': ', 1)[1] for line in unit.splitlines() if line.startswith('# Runtime SHA256: '))
        return (current['unit'] == unit and enabled_targets(current) == wanted_targets(unit) and not current['reload']
                and (not start or (current['active'] and current['pid'] > 0 and current['revision'] == revision)))

    @staticmethod
    def same_state(current, before):
        return (all(current[key] == before[key] for key in ('unit', 'enabled', 'active'))
                and enabled_targets(current) == enabled_targets(before) and not current['reload'])

    def replace_unit(self, text):
        if text is None:
            if self.unit.exists():
                self.unit.unlink()
                sync_directory(self.unit.parent)
                return True
            return False
        return atomic_write(self.unit, text, 0o644)

    def restore_state(self, before, allowed_unit):
        current = self.inspect()
        if current['unit'] not in (before['unit'], allowed_unit):
            raise RuntimeError('기록 후 서비스 파일이 외부에서 변경되었습니다. 복구 기록을 보존합니다.')
        if self.same_state(current, before):
            return
        unchanged_process = (current['active'] and before['active']
                             and current['boot'] == before['boot'] and current['pid'] == before['pid'])
        if current['active'] and not unchanged_process:
            self.ctl('stop', SERVICE)
        targets_changed = enabled_targets(current) != enabled_targets(before)
        if current['enabled'] and targets_changed:
            self.ctl('disable', SERVICE)
        changed = self.replace_unit(before['unit'])
        if changed or current['reload']:
            self.ctl('daemon-reload')
        if targets_changed:
            for target in enabled_targets(before):
                self.ctl('add-wants', target, SERVICE)
        if before['active'] and not unchanged_process:
            self.ctl('start', SERVICE)
        if not self.same_state(self.inspect(), before):
            raise RuntimeError('이전 서비스 상태의 복구 검증에 실패했습니다.')

    def clear_pending(self):
        self.pending.unlink()
        sync_directory(self.state)

    def recover(self):
        if not self.pending.exists():
            return
        data = self.load(self.pending)
        current = self.inspect()
        if data.get('action') == 'install' and self.desired(current, data['installed_unit'], data['start']):
            self.clear_pending()
            return
        self.restore_state(data['before'], data['installed_unit'])
        self.clear_pending()
        print('중단된 작업의 직전 상태를 복구했습니다.')

    def validate_runtime(self, unit):
        print(command('/bin/bash', str(self.project / 'scripts/daemon/01_bring_up.sh'), '--check'))  # 설치된 패키지 확인
        temporary = self.project / '.agent/tmp'
        temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='boot-manager-verify-', dir=temporary) as directory:
            candidate = Path(directory) / SERVICE
            candidate.write_text(unit)
            command('systemd-analyze', '--man=no', 'verify', str(candidate))

    def install(self, unit, start=True, force_restart=False):
        current = self.inspect()
        if self.desired(current, unit, start) and not force_restart:
            # A missing/stale original snapshot is never a reason to restart.
            if self.pending.exists():
                try:
                    data = self.load(self.pending)
                    if data.get('action') == 'install' and data.get('installed_unit') == unit:
                        self.clear_pending()
                    else:
                        print('실제 서비스는 정상입니다. 이전 복구 기록은 보존합니다.')
                except RuntimeError:
                    print('실제 서비스는 정상입니다. 손상된 복구 기록은 보존합니다.')
            self.report()
            return
        self.recover()
        current = self.inspect()
        self.validate_runtime(unit)
        if self.original.exists():
            self.load(self.original)
        else:
            self.save(self.original, {'before': current})
        self.save(self.pending, {'action': 'install', 'before': current, 'installed_unit': unit, 'start': start})
        try:
            changed = self.replace_unit(unit)
            if changed or current['reload']:
                self.ctl('daemon-reload')
            if enabled_targets(current) != wanted_targets(unit):
                if current['enabled']:
                    self.ctl('disable', SERVICE)
                self.ctl('enable', SERVICE)
            if start:
                if current['active'] and (force_restart or changed or not self.desired(current, unit, True)):
                    self.ctl('restart', SERVICE)
                elif not current['active']:
                    # 이전 시작 제한(failed)만 해제합니다. systemd 245는 아직 로드되지 않은 유닛이면 거부하므로 결과를 무시합니다.
                    subprocess.run(['systemctl', 'reset-failed', SERVICE],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.ctl('start', SERVICE)
                # Type=exec: 01_bring_up.sh 실행 직후 시작 완료로 처리하고 준비 신호를 기다리지 않습니다.
                time.sleep(0.25)
            if not self.desired(self.inspect(), unit, start):
                raise RuntimeError('서비스 등록/실행 검증에 실패했습니다.')
        except BaseException:
            self.restore_state(current, unit)
            self.clear_pending()
            print('설치 실패: 직전 서비스 상태로 복구했습니다.', file=sys.stderr)
            raise
        self.clear_pending()
        self.report()

    def restore(self):
        self.recover()
        if not self.original.exists():
            print('복원할 최초 설치 기록이 없습니다. 변경하지 않았습니다.')
            return
        before = self.load(self.original)['before']
        current = self.inspect()
        if not self.same_state(current, before):
            self.save(self.pending, {'action': 'restore', 'before': current, 'installed_unit': before['unit']})
            self.restore_state(before, current['unit'])
            self.clear_pending()
        print('최초 설치 전 서비스 상태로 복원했습니다.')
        self.report()

    def report(self):
        current = self.inspect()
        print('{}: enabled={}, active={}, pid={}'.format(SERVICE, current['enabled'], current['active'], current['pid']))


def interrupted(signum, _):
    raise KeyboardInterrupt('signal ' + str(signum))


def main(argv=None):
    parser = argparse.ArgumentParser(description='R1 부팅 시 설치된 boot_manager 패키지를 런치하는 서비스를 등록합니다.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--print-unit', action='store_true', help='생성할 서비스 파일 출력 (PC에서도 가능)')
    mode.add_argument('--dry-run', action='store_true', help='현재 상태와 적용 계획만 확인')
    mode.add_argument('--restore', action='store_true', help='최초 설치 전 서비스 상태로 복원')
    parser.add_argument('--no-start', action='store_true', help='부팅 자동 실행만 등록; 현재 프로세스는 유지')
    parser.add_argument('--restart', action='store_true', help='런타임 파일 변경 시 현재 서비스도 재시작')
    args = parser.parse_args(argv)
    if args.restart and args.no_start:
        parser.error('--restart와 --no-start는 함께 사용할 수 없습니다.')
    unit = render_unit(PROJECT)
    if args.print_unit:
        print(unit, end='')
        return 0
    require_r1()
    installer = Installer()
    if args.dry_run:
        installer.report()
        print('변경 없음' if installer.desired(installer.inspect(), unit, not args.no_start)
              else '적용 예정: 서비스 파일, 현재 R1 부팅 대상 자동 시작' + ('' if args.no_start else ', 현재 실행'))
        if installer.pending.exists():
            print('이전 작업 복구 기록이 있습니다. 실행 시 먼저 확인합니다.')
        return 0
    if os.geteuid() != 0:
        raise RuntimeError('R1에서 root 또는 sudo로 실행하세요.')
    installer.lock.parent.mkdir(parents=True, exist_ok=True)
    with installer.lock.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('다른 boot_manager 등록/복원 작업이 실행 중입니다.') from error
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, interrupted)
        if args.restore:
            installer.restore()
        else:
            installer.install(unit, start=not args.no_start, force_restart=args.restart)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, KeyboardInterrupt) as error:
        print('오류: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
HUMANOID_SERVICE_PYTHON
