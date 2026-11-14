#!/usr/bin/env bash
# Cross-build on this PC, then deploy only the packaged R1 runtime.
set -euo pipefail
deploy_script="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/03_deploy.sh"
exec python3 -B -c '
import os, sys
script = sys.argv.pop(1)
sys.argv[0] = script
with os.fdopen(3) as source:
    code = compile(source.read(), script, "exec")
exec(code, {"__name__": "__main__", "__file__": script})
' "$deploy_script" "$@" 3<<'R1_DEPLOY_PYTHON'
"""Build all project packages for the R1 on the PC; transfer an immutable runtime.

The existing R1-compatible GCC/sysroot/ROS installation is under
.agent/ros2/lyrical-cross. Build tools and generated sources stay on the PC.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.agent/deploy'
TARGET = 'aarch64-unknown-linux-gnu'
DEFAULT_REMOTE = '/root/workspace/agroid_v1'


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def remote_directory(value):
    path = PurePosixPath(value)
    if (not path.is_absolute() or len(path.parts) < 4 or '..' in path.parts
            or str(path) != value or any(ord(c) < 32 for c in value)
            or path.parts[1] in ('etc', 'usr', 'bin', 'sbin', 'lib', 'lib64', 'proc', 'sys', 'dev', 'run', 'opt')):
        raise argparse.ArgumentTypeError('R1 전용 절대 경로가 필요합니다: ' + DEFAULT_REMOTE)
    return value


def package_manifests():
    result = {}
    for manifest in sorted((ROOT / 'src').rglob('package.xml')):
        if any(p in ('target', '__pycache__', '.git') for p in manifest.relative_to(ROOT / 'src').parts):
            continue
        xml = ET.parse(manifest).getroot()
        name = (xml.findtext('name') or '').strip()
        kind = (xml.findtext('export/build_type') or '').strip()
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*', name):
            raise RuntimeError('잘못된 패키지 이름: ' + str(manifest))
        if name in result:
            raise RuntimeError(f'중복 패키지 {name}: {result[name][0]} / {manifest}')
        if kind not in ('ament_cargo', 'ament_cmake'):
            raise RuntimeError(f'지원하지 않는 패키지 빌드 형식: {name} ({kind})')
        result[name] = (manifest, kind)
    if 'boot_manager' not in result:
        raise RuntimeError('boot_manager 패키지가 없습니다.')
    return result


def check_package_set(actual, expected, location):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise RuntimeError(f'{location} 패키지 목록 불일치: 누락={sorted(missing)}, 예상 외={sorted(extra)}')


def packages(manifests, colcon, env):
    # Compare independent source discovery with colcon: ignore markers and metadata must not hide a package.
    output = run([colcon, '--log-base', WORK / 'log', 'list', '--base-paths', ROOT / 'src',
                  '--ignore-user-meta'], cwd=WORK, env=env, capture_output=True, text=True).stdout
    discovered = {}
    for line in output.splitlines():
        name, path, kind = line.split('\t')
        if name in discovered:
            raise RuntimeError('colcon에서 중복 발견한 패키지: ' + name)
        discovered[name] = (Path(path).resolve(), kind)
    check_package_set(discovered, manifests, 'colcon 탐색')
    result = {}
    for name, (manifest, kind) in manifests.items():
        colcon_kind = '(ament_cargo)' if kind == 'ament_cargo' else '(ros.' + kind + ')'
        if discovered[name] != (manifest.parent.resolve(), colcon_kind):
            raise RuntimeError('colcon 패키지 경로·빌드 형식 불일치: ' + name)
        if kind == 'ament_cargo':
            cargo_manifest = manifest.with_name('Cargo.toml').resolve()
            metadata = json.loads(run(['cargo', 'metadata', '--no-deps', '--format-version', '1',
                                       '--offline', '--manifest-path', cargo_manifest],
                                      cwd=WORK, env=env, capture_output=True, text=True).stdout)
            crate = next((p for p in metadata['packages']
                          if Path(p['manifest_path']).resolve() == cargo_manifest), None)
            if crate is None or crate['name'] != name:
                raise RuntimeError('Cargo와 ROS 패키지 이름 불일치: ' + name)
            # Cargo also discovers src/main.rs and src/bin/* without explicit [[bin]] entries.
            result[name] = sorted(t['name'] for t in crate['targets'] if 'bin' in t['kind'])
        else:
            result[name] = []
    return result


def check_installed_packages(install, expected):
    markers = install / 'share/ament_index/resource_index/packages'
    check_package_set([p.name for p in markers.iterdir() if p.is_file()] if markers.is_dir() else [],
                      expected, '설치 결과')


def check_build_results(expected):
    # These receipts are removed before each build, so previous successes cannot mask a skipped package.
    for name in expected:
        receipt = WORK / 'build' / name / 'colcon_build.rc'
        if not receipt.is_file() or receipt.read_text().strip() != '0':
            raise RuntimeError('이번 빌드에서 완료되지 않은 패키지: ' + name)
    check_installed_packages(WORK / 'install', expected)


def inputs():
    paths = list((ROOT / 'src').rglob('*')) + list((ROOT / 'scripts/daemon').glob('*.sh'))
    paths.append(Path(__file__))
    result = {}
    for p in sorted(paths):
        if p.is_file() and not any(x in ('target', '__pycache__', '.git') for x in p.relative_to(ROOT).parts):
            result[p.relative_to(ROOT).as_posix()] = digest(p)
    return result


def toolchain(cross):
    sysroot = cross / 'sysroot'
    ros = sysroot / 'opt/ros/lyrical'
    host = cross / 'host-tools'
    gcc = cross / 'toolchain/bin/aarch64-linux-gnu-gcc'
    required = [gcc, host / 'bin/python', host / 'bin/colcon', host / 'bin/cargo-ament-build',
                ros / 'lib/librcl.so', sysroot / 'usr/include/stdio.h',
                sysroot / 'opt/ros/lyrical-tools/include/python3.12/Python.h',
                cross / 'py-cross-sysconfig/_sysconfigdata__linux_aarch64-linux-gnu.py']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError('R1 크로스 빌드 환경이 없습니다:\n' + '\n'.join(missing))
    cargo_bin = Path.home() / '.cargo/bin'
    env = {k: v for k, v in os.environ.items() if k in (
        'HOME', 'USER', 'LOGNAME', 'TERM', 'CARGO_HOME', 'RUSTUP_HOME',
        'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')}
    env.update(PATH=f'{host}/bin:{cross}/toolchain/bin:{cargo_bin}:/usr/local/bin:/usr/bin:/bin',
               LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1',
               ROS_VERSION='2', ROS_PYTHON_VERSION='3', ROS_DISTRO='lyrical',
               COLCON_HOME=str(WORK / 'colcon'), COLCON_DEFAULTS_FILE=os.devnull,
               AMENT_PREFIX_PATH=str(ros), CMAKE_PREFIX_PATH=str(ros),
               PYTHONPATH=f'{cross}/py-cross-sysconfig:{ros}/lib/python3.12/site-packages',
               _PYTHON_SYSCONFIGDATA_NAME='_sysconfigdata__linux_aarch64-linux-gnu',
               CARGO_BUILD_TARGET=TARGET, CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=str(gcc),
               CARGO_PROFILE_RELEASE_LTO='true', CARGO_PROFILE_RELEASE_CODEGEN_UNITS='1',
               CARGO_PROFILE_RELEASE_STRIP='debuginfo',
               CC_aarch64_unknown_linux_gnu=str(gcc),
               CXX_aarch64_unknown_linux_gnu=str(gcc.with_name('aarch64-linux-gnu-g++')),
               AR_aarch64_unknown_linux_gnu=str(gcc.with_name('aarch64-linux-gnu-ar')),
               BINDGEN_EXTRA_CLANG_ARGS_aarch64_unknown_linux_gnu=f'--sysroot={sysroot}')
    flags = ['-Lnative=' + str(ros / 'lib'), '-Lnative=' + str(WORK / 'install/lib'),
             '-Clink-arg=-Wl,-rpath-link,' + ':'.join(map(str, [ros / 'lib', WORK / 'install/lib',
                  sysroot / 'opt/ros/lyrical-tools/lib', cross / 'toolchain/aarch64-linux-gnu/lib64'])),
             '-Clink-arg=-Wl,-rpath,$ORIGIN/..:/opt/ros/lyrical/lib',
             '-Clink-arg=-lpthread', '-Clink-arg=-ldl', '-Clink-arg=-lrt']
    env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        env[name] = str(WORK / 'tmp')
    rust = run(['rustc', '--version'], env=env, capture_output=True, text=True).stdout.strip()
    targets = run(['rustup', 'target', 'list', '--installed'], env=env, capture_output=True, text=True).stdout
    if TARGET not in targets.splitlines():
        raise RuntimeError('PC에서 rustup target add ' + TARGET + ' 실행이 필요합니다.')
    return sysroot, ros, host, gcc, env, rust


def build(cross, jobs):
    if platform.machine() not in ('x86_64', 'amd64'):
        raise RuntimeError('03_deploy.sh는 x86_64 PC에서 실행하세요.')
    sysroot, ros, host, gcc, env, rust = toolchain(cross)
    before = inputs()
    for name in ('tmp', 'build', 'log'):
        (WORK / name).mkdir(parents=True, exist_ok=True)
    manifests = package_manifests()
    expected = packages(manifests, host / 'bin/colcon', env)
    cmake = WORK / 'r1-toolchain.cmake'
    cmake.write_text(f'''set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSROOT "{sysroot}")
set(CMAKE_C_COMPILER "{gcc}")
set(CMAKE_CXX_COMPILER "{gcc.with_name('aarch64-linux-gnu-g++')}")
set(CMAKE_FIND_ROOT_PATH "{sysroot}" "{WORK / 'install'}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--as-needed -Wl,-rpath-link,{ros}/lib:{sysroot}/opt/ros/lyrical-tools/lib")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-Wl,--as-needed -Wl,-rpath-link,{ros}/lib:{sysroot}/opt/ros/lyrical-tools/lib")
set(CMAKE_C_STANDARD_LIBRARIES_INIT "-lpthread -ldl -lrt")
set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "-lpthread -ldl -lrt")
''')
    env['CARGO_BUILD_JOBS'] = str(jobs)
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(jobs)
    env['MAKEFLAGS'] = '-j' + str(jobs)
    command = [host / 'bin/colcon', '--log-base', WORK / 'log', 'build',
               '--base-paths', ROOT / 'src', '--build-base', WORK / 'build',
               '--install-base', WORK / 'install', '--merge-install', '--executor', 'sequential',
               '--ignore-user-meta', '--packages-select', *sorted(expected),
               '--event-handlers', 'console_direct+', 'desktop_notification-',
               '--cargo-args', ' --release', ' --target', ' ' + TARGET,
               '--cmake-args', '-DCMAKE_TOOLCHAIN_FILE=' + str(cmake), '-DCMAKE_BUILD_TYPE=Release',
               '-DBUILD_TESTING=OFF', '-DPython3_EXECUTABLE=' + str(host / 'bin/python'),
               '-DPYTHON_EXECUTABLE=' + str(host / 'bin/python'),
               '-DPython3_INCLUDE_DIR=' + str(sysroot / 'opt/ros/lyrical-tools/include/python3.12'),
               '-DPython3_LIBRARY=' + str(sysroot / 'opt/ros/lyrical-tools/lib/libpython3.12.so'),
               '-DPython3_NumPy_INCLUDE_DIR=' + str(sysroot / 'opt/ros/lyrical-tools/lib/python3.12/site-packages/numpy/_core/include'),
               '-DPYTHON_MODULE_EXTENSION=.cpython-312-aarch64-linux-gnu.so',
               '-DCMAKE_INSTALL_RPATH=$ORIGIN;/opt/ros/lyrical/lib;/opt/ros/lyrical-tools/lib']
    print(f'[deploy] 자동 발견한 {len(expected)}개 패키지: ' + ', '.join(sorted(expected)), flush=True)
    # Keep compiler caches, but reinstall everything. Deleted/renamed packages and old binaries must not survive.
    if (WORK / 'install').exists():
        shutil.rmtree(WORK / 'install')
    for receipt in (WORK / 'build').glob('*/colcon_build.rc'):
        receipt.unlink()
    print('[deploy] PC에서 전체 패키지 ARM64 Release 빌드', flush=True)
    run(command, cwd=WORK, env=env)
    check_build_results(expected)
    after = inputs()
    # Cargo creates missing lockfiles on the first build; include them in the release identity.
    added_sources = [p for p in after.keys() - before.keys() if not p.endswith('/Cargo.lock')]
    if added_sources or any(after.get(p) != h for p, h in before.items()):
        raise RuntimeError('빌드 중 소스가 변경되었습니다. 다시 실행하세요.')
    identity = {'inputs': after, 'rustc': rust, 'gcc': digest(gcc), 'ros': digest(ros / 'lib/librcl.so'),
                'cross_root': str(cross)}
    release = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return release, expected, identity


def elf_check(path):
    with path.open('rb') as stream:
        header = stream.read(20)
    if header[:4] != b'\x7fELF':
        return False
    if header[4:6] != b'\x02\x01' or struct.unpack('<H', header[18:20])[0] != 183:
        raise RuntimeError('ARM64 실행 파일이 아닙니다: ' + str(path))
    info = run(['readelf', '--version-info', path], capture_output=True, text=True).stdout
    versions = [tuple(map(int, v.split('.'))) for v in re.findall(r'\bGLIBC_(\d+\.\d+(?:\.\d+)?)', info)]
    if versions and max(versions) > (2, 31):
        raise RuntimeError('R1 glibc 2.31보다 새로운 ABI가 필요합니다: ' + str(path))
    return True


def runtime_files(base, installed_prefix):
    result = {}
    for folder in ('install', 'scripts/daemon'):
        for p in sorted((base / folder).rglob('*')):
            relative = p.relative_to(base).as_posix()
            if p.is_symlink():
                result[relative] = ['link', os.readlink(p)]
            elif p.is_file():
                data = p.read_bytes().replace((installed_prefix + '/install').encode(), b'@AGROID_INSTALL@')
                result[relative] = ['file', hashlib.sha256(data).hexdigest(), p.stat().st_mode & 0o777]
    return result


def package(release, expected, identity, remote):
    stage = WORK / 'releases' / release
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(WORK / 'install', stage / 'install', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    daemon = stage / 'scripts/daemon'
    daemon.mkdir(parents=True)
    for p in (ROOT / 'scripts/daemon').glob('*.sh'):
        shutil.copy2(p, daemon / p.name)
    # Cargo's install tree includes source crates for downstream compilation. R1 needs runtime only.
    for p in [stage / 'install/include', *stage.glob('install/share/*/rust'),
              *stage.glob('install/share/*/cmake'), stage / 'install/lib/pkgconfig']:
        if p.is_dir():
            shutil.rmtree(p)
    # rosidl's Python install also contains generated C sources; only its .py/.so are runtime.
    for p in stage.rglob('*'):
        if p.is_file() and p.suffix in ('.c', '.cc', '.cpp', '.h', '.hpp', '.rs', '.a', '.rlib'):
            p.unlink()
    target = remote + '/.releases/' + release
    local_prefix = str(WORK / 'install')
    for p in stage.rglob('*'):
        if p.is_symlink():
            if not p.resolve().is_relative_to(stage.resolve()):
                raise RuntimeError('배포 폴더 밖을 참조하는 링크: ' + str(p))
        elif p.is_file():
            if elf_check(p):
                continue
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                continue
            text = text.replace(local_prefix, target + '/install')
            text = text.replace(str(Path(identity.get('cross_root', ROOT / '.agent/ros2/lyrical-cross')) / 'sysroot'), '')
            if text.startswith('#!') and '/host-tools/' in text.splitlines()[0]:
                text = '#!/opt/ros/lyrical-tools/bin/python3.12\n' + text.partition('\n')[2]
            text = re.sub(r'"[^"\n]+/host-tools/bin/python(?:3(?:\.12)?)?"',
                          '"/opt/ros/lyrical-tools/bin/python3.12"', text)
            p.write_text(text)
    for name, bins in expected.items():
        marker = stage / 'install/share/ament_index/resource_index/packages' / name
        if not marker.is_file():
            raise RuntimeError('설치되지 않은 패키지: ' + name)
        for binary in bins:
            p = stage / 'install/lib' / name / binary
            if not p.is_file() or not os.access(p, os.X_OK) or not elf_check(p):
                raise RuntimeError('실행 파일이 없거나 잘못되었습니다: ' + str(p))
    check_installed_packages(stage / 'install', expected)
    files, links = {}, {}
    for p in sorted(stage.rglob('*')):
        if p.is_symlink():
            links[p.relative_to(stage).as_posix()] = os.readlink(p)
        elif p.is_file():
            files[p.relative_to(stage).as_posix()] = digest(p)
    manifest = {'release': release, 'target': TARGET, 'packages': expected, 'files': files, 'links': links,
                'runtime': runtime_files(stage, target)}
    (stage / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(f'[deploy] {len(expected)}개 패키지 검증 완료: {stage}', flush=True)
    return stage


def initialize_module():
    script = ROOT / 'scripts/00_initialize.sh'
    source = script.read_text().split("3<<'R1_INITIALIZE_PYTHON'\n", 1)[1].rsplit('\nR1_INITIALIZE_PYTHON', 1)[0]
    module = types.ModuleType('r1_deploy_connection')
    module.__file__ = str(script)
    exec(compile(source, str(script), 'exec'), module.__dict__)
    return module


@contextmanager
def authentication(r1):
    with tempfile.TemporaryDirectory(prefix='r1-deploy-auth-', dir=WORK / 'tmp') as folder:
        helper = Path(folder) / 'askpass'
        helper.write_text('#!/bin/sh\nprintf "%s\\n" "$AGROID_R1_SSH_PASSWORD"\n')
        helper.chmod(0o700)
        env = dict(os.environ, SSH_ASKPASS=str(helper), SSH_ASKPASS_REQUIRE='force',
                   DISPLAY='agroid:0', AGROID_R1_SSH_PASSWORD=r1.ssh_password())
        yield env


REMOTE_COMPARE = r'''
import hashlib, json, os, pathlib, sys

def runtime_files(base, installed_prefix=None):
    result = {}
    prefix = str(installed_prefix if installed_prefix is not None else base)
    for folder in ('install', 'scripts/daemon'):
        for p in sorted((base / folder).rglob('*')):
            relative = p.relative_to(base).as_posix()
            if p.is_symlink():
                result[relative] = ['link', os.readlink(p)]
            elif p.is_file():
                data = p.read_bytes().replace((prefix + '/install').encode(), b'@AGROID_INSTALL@')
                result[relative] = ['file', hashlib.sha256(data).hexdigest(), p.stat().st_mode & 0o777]
    return result

def active_runtime(root):
    current = root / 'current'
    if current.is_symlink():
        active = current.resolve()
        if (root / '.releases').resolve() not in active.parents:
            raise RuntimeError('current points outside releases')
        return active
    if current.exists():
        raise RuntimeError('current must be a symlink')
    return root
'''

REMOTE_APPLY = REMOTE_COMPARE + r'''
import fcntl, subprocess, time
root, release, expected, restart = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == '1'
stage = root / '.incoming' / release
final = root / '.releases' / release
with (root / '.deploy.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = stage / 'manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected, 'manifest checksum mismatch'
    data = json.loads(manifest.read_text())
    assert data['release'] == release and data['target'] == 'aarch64-unknown-linux-gnu'
    for rel, checksum in data['files'].items():
        p = stage / rel
        assert stage.resolve() in p.resolve().parents and not p.is_symlink(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == checksum, rel
    for rel, target in data['links'].items():
        p = stage / rel
        assert p.is_symlink() and os.readlink(p) == target and stage.resolve() in p.resolve().parents, rel
    assert sorted(p.name for p in stage.iterdir()) == ['install', 'manifest.json', 'scripts']
    assert runtime_files(stage, final) == data['runtime'], 'runtime manifest mismatch'
    current = root / 'current'
    if runtime_files(active_runtime(root)) == data['runtime']:
        print('R1 runtime unchanged; boot_manager was not restarted')
        sys.exit(0)
    final.parent.mkdir(exist_ok=True)
    if final.exists():
        if runtime_files(final) != data['runtime']:
            backup = final.with_name(release + '.previous-' + str(time.time_ns()))
            final.rename(backup)
            stage.rename(final)
    else:
        stage.rename(final)
    subprocess.run(['bash', str(final / 'scripts/daemon/01_bring_up.sh'), '--check'], check=True)
    command = ['bash', str(final / 'scripts/daemon/00_add_humanoid_service.sh')]
    command.append('--restart' if restart else '--no-start')
    subprocess.run(command, check=True)
    link = root / ('.current-' + release)
    if link.is_symlink(): link.unlink()
    link.symlink_to('.releases/' + release, target_is_directory=True)
    os.replace(link, current)
    os.sync()
    print('R1 runtime applied: ' + str(final))
'''


def deploy(stage, remote, dry_run=False, restart=True):
    r1 = initialize_module()
    with r1.preview_identity():
        if r1.find_existing_connection() is None:
            raise RuntimeError('R1에 연결되어 있지 않습니다. scripts/00_initialize.sh를 먼저 실행하세요.')
        release = stage.name
        incoming = remote + '/.incoming/' + release
        print(f'[deploy] {stage} → R1 {remote}/.releases/{release}', flush=True)
        # Both preview and actual deployment require the target ROS runtime; neither builds on R1.
        r1.ssh_run(None, r1.AP_IP, 'test -r /opt/ros/lyrical/setup.bash && command -v rsync >/dev/null')
        data = json.loads((stage / 'manifest.json').read_text())
        compare = REMOTE_COMPARE + '\nroot = pathlib.Path(sys.argv[1])\n' + \
            'print(json.dumps(runtime_files(active_runtime(root)) == json.loads(sys.stdin.read())))\n'
        command = shlex.join(['python3', '-c', compare, remote])
        identical = json.loads(r1.ssh_run(None, r1.AP_IP, command, payload=json.dumps(data['runtime'])))
        if identical:
            print('[deploy] R1 파일과 실행 권한이 동일합니다. 전송·서비스 재시작을 생략합니다.')
            return
        if dry_run:
            print('[deploy] 변경 감지: runtime만 전송, 체크섬 검증, current 전환' +
                  (', boot_manager 재시작' if restart else ', 서비스 등록만 갱신'))
            print('[deploy] R1의 파일·서비스를 변경하지 않았습니다.')
            return
        prepare = '''import os, pathlib, sys
p = pathlib.Path(sys.argv[1])
for ancestor in list(reversed(p.parents)) + [p]:
    if ancestor.is_symlink(): raise RuntimeError('symlink in deployment path')
    ancestor.mkdir(exist_ok=True)
'''
        r1.ssh_run(None, r1.AP_IP, shlex.join(['python3', '-', incoming]), payload=prepare)
        ssh = r1.ssh_argv(None, r1.AP_IP)
        with authentication(r1) as env:
            run(['rsync', '-a', '--delete', '--no-owner', '--no-group', '--protect-args',
                 '--partial-dir=.rsync-partial', '--rsh=' + shlex.join(ssh[:-1]),
                 str(stage) + '/', ssh[-1] + ':' + incoming + '/'], env=env)
        command = shlex.join(['python3', '-', remote, release, digest(stage / 'manifest.json'),
                              '1' if restart else '0'])
        print(r1.ssh_run(None, r1.AP_IP, command, payload=REMOTE_APPLY, timeout=120), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='PC에서 ARM64 Release 빌드 후 R1에 실행 결과물만 배포합니다.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--build-only', action='store_true', help='PC 빌드·배포 묶음 검증만 수행 (R1 접속 없음)')
    mode.add_argument('--dry-run', action='store_true', help='PC 빌드 후 R1 연결·배포 계획 점검 (R1 변경 없음)')
    parser.add_argument('--no-restart', action='store_true', help='배포·서비스 등록만 갱신하고 현재 프로세스 유지')
    parser.add_argument('--jobs', type=int, default=8, help='PC 컴파일 병렬 수 (기본 8)')
    parser.add_argument('--cross-root', type=Path, default=ROOT / '.agent/ros2/lyrical-cross')
    parser.add_argument('--remote-dir', type=remote_directory,
                        default=os.environ.get('AGROID_R1_PROJECT_DIR', DEFAULT_REMOTE))
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs는 양수여야 합니다.')
    remote_directory(args.remote_dir)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        release, expected, identity = build(args.cross_root.resolve(), args.jobs)
        stage = package(release, expected, identity, args.remote_dir)
        if not args.build_only:
            deploy(stage, args.remote_dir, args.dry_run, not args.no_restart)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print('배포 중단: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
R1_DEPLOY_PYTHON
