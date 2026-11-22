#!/usr/bin/env bash
# Prepare the GPU server (super-via-ailab:/rl-agent) and keep the trainer folder synchronised with it:
#   trainer/ <-> /rl-agent/
# The server is shared, so nothing else of this project is sent. From the work area agroid_v1/.agent/trainer only
# what a training run itself reads or writes crosses over (WORK_SCOPE: environment files, runs, training inputs).
set -euo pipefail

rl_scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rl_root="$(cd -- "$rl_scripts_dir/.." && pwd)"
if [ -d "$rl_root/.agent" ]; then rl_agent="$rl_root/.agent"; else rl_agent="$(dirname -- "$rl_root")/.agent/trainer"; fi
rl_agent_tmp="$rl_agent/tmp"
mkdir -p -- "$rl_agent_tmp"
export TMPDIR="$rl_agent_tmp" TMP="$rl_agent_tmp" TEMP="$rl_agent_tmp"
export PYTHONDONTWRITEBYTECODE=1

exec python3 -c '
import os, sys
script = sys.argv.pop(1)
sys.argv[0] = script
with os.fdopen(3) as source:
    code = compile(source.read(), script, "exec")
exec(code, {"__name__": "__main__", "__file__": script})
' "$rl_scripts_dir/00_initialize.sh" "$@" 3<<'RL_INITIALIZE_PYTHON'
"""GPU 서버 준비와 PC <-> 서버 trainer 폴더 양방향 동기화.

  scripts/00_initialize.sh                   서버 준비 후 계속 동기화(3초 주기)
  scripts/00_initialize.sh --once            한 번만 동기화
  scripts/00_initialize.sh --check           접속·도구·경로 점검과 서버의 범위 밖 파일 보고, 변경 없음
  scripts/00_initialize.sh --dry-run         동기화 계획만 출력
  scripts/00_initialize.sh --install-service systemd 사용자 서비스로 상시 동기화 등록(실행 중이면 다시 시작)

대응: trainer/ <-> /rl-agent/. 서버는 공유 환경이라 프로젝트의 다른 부분은 보내지 않는다.
범위: 작업 영역 agroid_v1/.agent/trainer/ <-> /rl-agent/.agent/ 는 학습 실행이 읽고 쓰는 것(WORK_SCOPE: 환경 파일,
      runs, 학습 입력)만 오간다. 그 밖의 파일은 양쪽 모두 보지 않는다(보내지도, 지우지도 않는다).
      새 학습 입력은 WORK_SCOPE에 추가한 뒤 --install-service로 반영한다. 경로 표기는 서버 기준 .agent/... 이다.
규칙: 새 파일·수정 파일은 새것(mtime)이 이긴다. 직전 동기화 뒤 한쪽에서 지운 파일은 반대쪽도 지운다.
      한 번에 지울 파일이 많으면(MAX_DELETE) 멈춘다. 경로 설정 실수로 한쪽이 통째로 지워지는 것을 막는다.
제외: 범위 안에서도 .venv, __pycache__, .git, *.pyc. 전송은 rsync(-a)로 한다.
Python 표준 라이브러리만 사용한다.
"""
import argparse
import fcntl
import hashlib
from decimal import Decimal
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]  # trainer/ (code)
# Work area: agroid_v1/.agent/trainer on the workstation (same as /rl-agent/.agent on the server).
AGENT = ROOT / ".agent" if (ROOT / ".agent").is_dir() else ROOT.parent / ".agent" / "trainer"
AP = ".agent/"  # work-area files carry this prefix in the index (server-side relative path)
HOST = os.environ.get("RL_REMOTE", "super-via-ailab")
RDIR = os.environ.get("RL_REMOTE_DIR", "/rl-agent")
STATE = AGENT / ".sync"
MANIFEST = STATE / "manifest.json"
MAX_DELETE = 300
SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
# The trainer folder is synchronised whole. Of the work area only these server-relative path prefixes are: what a
# training run reads or writes. An entry ending in "." names a data file with its sidecars (x.npz, x.manifest.json).
WORK_SCOPE = (
    ".agent/pyproject.toml", ".agent/uv.lock",   # server environment
    ".agent/runs/",                              # run records, logs, checkpoints (resume and teacher sources)
    # inputs loaded by the registered r1lab / r1wb tasks
    ".agent/data/amp/careful_walk.npz", ".agent/data/amp/crouch.npz",
    ".agent/data/amp/human_careful_fullbody_v3.npz",
    ".agent/data/reference_initialization/cautious_rsi_v1.",
    ".agent/data/tracking/cautious_teacher_v1.", ".agent/data/tracking/cautious_teacher_heldout_v1.",
    ".agent/data/tracking/cautious_full_v1.",
    ".agent/data/tracking/careful_library_v2.", ".agent/data/tracking/careful_library_v3.",
    ".agent/data/tracking/careful_library_v4.", ".agent/data/tracking/careful_library_v5.",
    ".agent/data/wb/library_v2_core.",
)
SERVER_LOCAL = (".agent/tmp/",)  # written by runs on the server itself (launch handoff); never synchronised
# Server-only data at the trainer root: datasets (user 2026-11-20: gather all datasets in /rl-agent/dataset) and
# external code/model downloads. Never mirrored to the PC; invisible on both sides, so excluding them deletes nothing.
SERVER_DATA = ("dataset/", "ext/", "models/")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ControlMaster=auto",
       "-o", f"ControlPath={Path.home()}/.ssh/cm-rl-%r@%h:%p", "-o", "ControlPersist=10m"]
UV = "$HOME/.local/bin/uv"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def ssh(cmd, check=True, capture=True):
    r = subprocess.run(SSH + [HOST, cmd], text=True, capture_output=capture)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd}\n{r.stderr}")
    return r.stdout if capture else ""


def skipped(rel):
    # One predicate for both sides: a path outside the scope is invisible locally and on the server alike,
    # so narrowing the scope can never be read as "deleted on one side".
    parts = rel.split("/")
    return (rel.startswith(AP) and not rel.startswith(WORK_SCOPE)
            or rel.startswith(SERVER_DATA)
            or any(p in SKIP_DIRS for p in parts) or rel.endswith(".pyc")
            or (rel.startswith(".agent/runs/") and re.fullmatch(r"\..+\.[A-Za-z0-9]{6}", parts[-1]) is not None))


def may_hold_scope(rel_dir):
    if rel_dir.startswith(SERVER_DATA):
        return False
    return not rel_dir.startswith(AP) or any(s.startswith(rel_dir) or rel_dir.startswith(s) for s in WORK_SCOPE)


def local_path(rel):
    return AGENT / rel[len(AP):] if rel.startswith(AP) else ROOT / rel


def local_index():
    out = {}
    for base, prefix in ((ROOT, ""), (AGENT, AP)):
        for d, dirs, files in os.walk(base):
            here = Path(d).relative_to(base).as_posix()
            here = prefix + ("" if here == "." else here + "/")
            # Do not even walk work-area folders that cannot hold scope files (results, notes, other data).
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not (base == ROOT and x == ".agent")
                       and may_hold_scope(f"{here}{x}/")]
            for f in files:
                p = Path(d) / f
                rel = prefix + p.relative_to(base).as_posix()
                if skipped(rel) or p.is_symlink():
                    continue
                st = p.stat()
                out[rel] = (st.st_size, st.st_mtime_ns)
    return out


def remote_files():
    """Every file below RDIR except environments and caches, in or out of scope."""
    prune = " -o ".join(f"-name {shlex.quote(d)}" for d in sorted(SKIP_DIRS))
    cmd = (f"mkdir -p {RDIR} && cd {RDIR} && find . \\( {prune} \\) -prune -o -type f "
           f"-printf '%P\\t%s\\t%T@\\n'")
    out = {}
    for line in ssh(cmd).splitlines():
        rel, size, mt = line.split("\t")
        out[rel] = (int(size), int(Decimal(mt) * 1_000_000_000))
    return out


def remote_index():
    # Match the local walk's prefix pruning before traversing the remote tree.
    # Keep remote_files() as the explicit complete audit used by --check.
    prune = " -o ".join(f"-name {shlex.quote(d)}" for d in sorted(SKIP_DIRS))
    allowed_parents = {".agent"}
    subtrees = set()
    for prefix in WORK_SCOPE:
        if prefix.endswith("/"):
            subtrees.add(prefix.rstrip("/"))
            parent = Path(prefix.rstrip("/")).parent
        else:
            parent = Path(prefix).parent
            allowed_parents.add(parent.as_posix())
        while parent.as_posix() != ".":
            allowed_parents.add(parent.as_posix())
            parent = parent.parent
    # All trainer-root paths remain visible; only .agent has a restricted scope.
    keep = [f"-path {shlex.quote('./'+p)}" for p in sorted(allowed_parents)]
    for prefix in sorted(subtrees):
        keep.extend((f"-path {shlex.quote('./'+prefix)}", f"-path {shlex.quote('./'+prefix+'/*')}"))
    inside = " -o ".join(keep)
    agent_prune = f"\\( -type d -path './.agent/*' ! \\( {inside} \\) -prune \\)"
    data_prune = " -o ".join(f"-path {shlex.quote('./' + d.rstrip('/'))}" for d in SERVER_DATA)
    cmd = (f"mkdir -p {shlex.quote(RDIR)} && cd {shlex.quote(RDIR)} && find . "
           f"\\( {prune} \\) -prune -o \\( {data_prune} \\) -prune -o {agent_prune} "
           f"-o -type f -printf '%P\\t%s\\t%T@\\n'")
    out = {}
    for line in ssh(cmd).splitlines():
        rel, size, mt = line.split("\t")
        if not skipped(rel):
            out[rel] = (int(size), int(Decimal(mt) * 1_000_000_000))
    return out


def out_of_scope(remote):
    """Server files that are neither the trainer folder nor needed by a run; the shared server should hold none."""
    return sorted(rel for rel in remote if skipped(rel) and not rel.startswith(SERVER_LOCAL + SERVER_DATA))


def timestamp_ns(entry):
    # Old manifests stored Unix seconds. Current indexes/manifests store ns;
    # contemporary Unix seconds and nanoseconds have disjoint magnitudes.
    stamp = entry[1]
    return stamp * 1_000_000_000 if abs(stamp) < 1_000_000_000_000 else stamp


def same(a, b):
    if a is None or b is None or a[0] != b[0]:
        return False
    legacy = any(abs(x[1]) < 1_000_000_000_000 for x in (a, b))
    tolerance = 1_000_000_000 if legacy else 1_000
    return abs(timestamp_ns(a) - timestamp_ns(b)) <= tolerance


def plan(L, R, M):
    push, pull, del_l, del_r = [], [], [], []
    for p in sorted(set(L) | set(R)):
        lo, re_, m = L.get(p), R.get(p), M.get(p)
        if lo and re_:
            if not same(lo, re_):
                (push if timestamp_ns(lo) >= timestamp_ns(re_) else pull).append(p)
        elif lo:
            (del_l if same(lo, m) else push).append(p)
        else:
            (del_r if same(re_, m) else pull).append(p)
    return push, pull, del_l, del_r


def rsync(files, push):
    for base, prefix, rdir in ((ROOT, "", RDIR), (AGENT, AP, f"{RDIR}/.agent")):
        sel = [f[len(prefix):] for f in files if f.startswith(AP) == bool(prefix)]
        if not sel:
            continue
        src, dst = (f"{base}/", f"{HOST}:{rdir}/") if push else (f"{HOST}:{rdir}/", f"{base}/")
        with tempfile.NamedTemporaryFile("w", delete=False, dir=STATE) as f:
            f.write("\n".join(sel) + "\n")
            lst = f.name
        try:
            e = " ".join(shlex.quote(x) for x in SSH)
            # Only the selected list bypasses rsync's whole-second quick check.
            subprocess.run(["rsync", "-az", "--ignore-times", "--files-from", lst, "-e", e, src, dst], check=True)
        finally:
            os.unlink(lst)


def reconcile(dry=False):
    L, R = local_index(), remote_index()
    M = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    M = {k: tuple(v) for k, v in M.items()}
    push, pull, del_l, del_r = plan(L, R, M)
    if dry:
        for name, xs in (("push", push), ("pull", pull), ("delete-local", del_l), ("delete-remote", del_r)):
            log(f"{name}: {len(xs)}" + "".join(f"\n  {x}" for x in xs[:40]))
        return push
    if max(len(del_l), len(del_r)) > MAX_DELETE:
        raise RuntimeError(f"too many deletions planned (local {len(del_l)}, remote {len(del_r)}); "
                           "check paths, inspect with --dry-run, delete by hand if intended")
    rsync(push, push=True)
    rsync(pull, push=False)
    for p in del_l:
        local_path(p).unlink(missing_ok=True)
    if del_r:
        ssh(f"cd {RDIR} && xargs -d '\\n' rm -f -- <<'EOF'\n" + "\n".join(del_r) + "\nEOF")
    if push or pull or del_l or del_r:
        log(f"push {len(push)}, pull {len(pull)}, delete local {len(del_l)}, remote {len(del_r)}")
        L, R = local_index(), remote_index()
    MANIFEST.write_text(json.dumps({k: L[k] for k in L if same(L[k], R.get(k))}))
    return push


# Headless rendering: the container exposes only compute/utility, so the NVIDIA EGL userspace
# libraries matching the host driver are unpacked to /opt/nvidia-egl-<ver> (dpkg state untouched)
# and selected per process via env (see EGL_ENV); Mesa EGL/OSMesa are the fallback.
EGL_SETUP = r"""
set -e
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
dst=/opt/nvidia-egl-$drv
if [ ! -f "$dst/10_nvidia.json" ]; then
  export DEBIAN_FRONTEND=noninteractive
  dpkg -s libegl1 libosmesa6 >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq --no-install-recommends libegl1 libgl1 libosmesa6; }
  tmp=$(mktemp -d); cd "$tmp"
  apt-get download "libnvidia-gl-${drv%%.*}=$drv-0ubuntu1" "libnvidia-gpucomp-${drv%%.*}=$drv-0ubuntu1" >/dev/null
  mkdir -p "$dst"; for d in *.deb; do dpkg-deb -x "$d" "$dst"; done
  printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"}}\n' "$dst" > "$dst/10_nvidia.json"
  cd /; rm -rf "$tmp"
fi
ln -sfn "$dst" /opt/nvidia-egl
"""


def remote_setup():
    ssh(f"mkdir -p {RDIR}")
    if ssh(f"test -x {UV} && echo ok || true").strip() != "ok":
        log("install uv on server")
        ssh("curl -LsSf https://astral.sh/uv/install.sh | sh", capture=False)
    if ssh("test -f /opt/nvidia-egl/10_nvidia.json && echo ok || true").strip() != "ok":
        log("server: install headless EGL (NVIDIA libs matching the host driver)")
        ssh(EGL_SETUP, capture=False)


def env_hash():
    h = hashlib.sha256()
    for f in (".agent/pyproject.toml", ".agent/uv.lock"):
        p = local_path(f)
        h.update(p.read_bytes() if p.exists() else b"")
    return h.hexdigest()


def remote_uv_sync():
    marker = STATE / "remote_env.sha256"
    cur = env_hash()
    if marker.exists() and marker.read_text() == cur:
        return
    log("server: uv sync (.agent/.venv)")
    ssh(f"cd {RDIR} && {UV} sync --project .agent", capture=False)
    marker.write_text(cur)


def check():
    print("local :", ROOT, "+", AGENT)
    print("server:", ssh("hostname; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; "
                         f"ls -d {RDIR} && (test -x {UV} && {UV} --version || echo 'uv 없음')"))
    extra = out_of_scope(remote_files())
    print(f"서버의 범위 밖 파일: {len(extra)}개" + ("" if extra else " (trainer 폴더와 학습 입출력만 있음)"))
    for rel in extra[:40]:
        print(f"  {RDIR}/{rel}")
    if len(extra) > 40:
        print(f"  ... 외 {len(extra) - 40}개")
    return 1 if extra else 0


def install_service():
    unit = Path.home() / ".config/systemd/user/trainer-sync.service"
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(f"""[Unit]
Description=Trainer GPU server sync: agroid_v1 <-> {HOST}:{RDIR}
After=network-online.target

[Service]
ExecStart={ROOT}/scripts/00_initialize.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
""")
    legacy = unit.with_name("rl-agent-sync.service")
    if legacy.exists():
        # Finish the current transfer before replacing the old service.
        with open(STATE / "lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run(["systemctl", "--user", "disable", "--now", legacy.name], check=True)
        legacy.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", unit.name], check=True)
    # A running service keeps the scope it started with; restart between passes so it reads this file again.
    with open(STATE / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(["systemctl", "--user", "restart", unit.name], check=True)
    print(f"등록: {unit}\n상태: systemctl --user status {unit.stem}\n중지: systemctl --user disable --now {unit.stem}")


def sync_once(setup=False):
    # One reconcile at a time (service loop, --once from other scripts); held only for one pass.
    with open(STATE / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if setup:
            remote_setup()
        reconcile()
        remote_uv_sync()  # no-op unless .agent/pyproject.toml or .agent/uv.lock changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--install-service", action="store_true")
    ap.add_argument("--interval", type=float, default=3.0)
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if a.check:
        sys.exit(check())  # non-zero while the server holds files outside the scope
    if a.install_service:
        return install_service()
    if a.dry_run:
        return reconcile(dry=True)
    if a.once:
        return sync_once(setup=True)
    daemon = open(STATE / "daemon.lock", "w")
    try:
        fcntl.flock(daemon, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("상시 동기화가 이미 실행 중입니다(.agent/.sync/daemon.lock).")
        return
    sync_once(setup=True)
    log(f"sync {ROOT} <-> {HOST}:{RDIR} every {a.interval:g}s")
    while True:
        time.sleep(a.interval)
        try:
            sync_once()
        except Exception as e:  # keep running through network hiccups
            log(f"sync error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
RL_INITIALIZE_PYTHON
