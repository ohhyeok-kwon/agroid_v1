"""Training history, server status and read-only artifact catalog."""
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from urllib.parse import quote


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def read_part(path, size, tail=False, offset=0, end=None):
    try:
        with path.open('rb') as stream:
            limit = min(end, path.stat().st_size) if end is not None else path.stat().st_size
            if tail:
                stream.seek(max(offset, limit - size))
            elif offset:
                stream.seek(offset)
            return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', stream.read(min(size, max(0, limit-stream.tell()))).decode(errors='replace'))
    except OSError:
        return ''


class Trainer:
    def __init__(self, project, renderer=None):
        self.work = Path(project) / '.agent/trainer'
        self.result = self.work / 'result'
        self.lock = threading.RLock()
        self.cached = None
        self.cached_at = 0
        self.log_offsets = {}
        self.log_headers = {}
        self.log_sections = {}
        self.server = {'connected': False, 'checked': None, 'processes': [], 'message': '서버 상태 확인 중'}
        self.server_thread = None
        self.server_requested = 0
        self.renderer = renderer
        self.probe_process = None
        self.closed = False

    def probe_server(self):
        code = """import json
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\\0')
  if b'r1lab.train' not in args and not any(a.endswith(b'/r1lab/train.py') for a in args): continue
  rows.append({'pid':int(p.name),'log':(p/'fd/1').resolve().name})
 except OSError: pass
print(json.dumps(rows))
"""
        process = None
        try:
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                     '-o', 'ControlMaster=auto', '-o', 'ControlPersist=10m',
                                     '-o', 'ControlPath='+os.path.expanduser('~/.ssh/cm-rl-%r@%h:%p'),
                                     os.environ.get('RL_REMOTE', 'super-via-ailab'), 'python3 -c '+shlex.quote(code)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with self.lock:
                self.probe_process = process
                if self.closed:
                    process.terminate()
            stdout, _ = process.communicate(timeout=12)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, 'ssh')
            processes = json.loads(stdout)
            if not isinstance(processes, list):
                raise ValueError('잘못된 서버 응답')
            status = {'connected': True, 'checked': time.time(), 'processes': processes, 'message': ''}
        except (OSError, ValueError, subprocess.SubprocessError):
            status = {'connected': False, 'checked': time.time(), 'processes': [], 'message': 'GPU 서버 상태를 확인하지 못했습니다.'}
        finally:
            if process and process.poll() is None:
                process.kill()
                process.communicate()
            with self.lock:
                if self.probe_process is process:
                    self.probe_process = None
        with self.lock:
            if not self.closed:
                self.server = status
                self.cached_at = 0

    def server_status(self):
        if not self.closed and time.monotonic()-self.server_requested > 10 and not (self.server_thread and self.server_thread.is_alive()):
            self.server_requested = time.monotonic()
            self.server_thread = threading.Thread(target=self.probe_server, daemon=True)
            self.server_thread.start()
        return dict(self.server)

    def close(self):
        with self.lock:
            self.closed = True
            process = self.probe_process
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except ProcessLookupError:
                pass
        if self.server_thread:
            self.server_thread.join(timeout=1)

    def logs(self):
        pattern = rb'\[INFO\][^\r\n]*\blog=(?:[^\s]*/)?runs/([^\s]+)'
        paths = set((self.work / 'runs').glob('*.log'))
        for missing in self.log_sections.keys() - paths:
            self.log_sections.pop(missing, None)
            self.log_offsets.pop(missing, None)
        for path in sorted(paths):
            try:
                if path.stat().st_size < self.log_offsets.get(path, 0):
                    self.log_sections.pop(path, None)
                start = max(0, min(self.log_offsets.get(path, 0), path.stat().st_size) - 1024)
                with path.open('rb') as stream:
                    stream.seek(start)
                    while block := stream.read(1024 * 1024):
                        for match in re.finditer(pattern, block):
                            relative = match[1].decode(errors='replace').rstrip('/')
                            self.log_sections.setdefault(path, {})[start + match.start()] = (relative, start + match.end())
                        start = stream.tell()
                        if len(block) == 1024 * 1024:
                            stream.seek(start - 1024)
                            start -= 1024
                    self.log_offsets[path] = stream.tell()
            except OSError:
                continue
        self.log_headers = {}
        for path, sections in self.log_sections.items():
            ordered = sorted(sections.items())
            for i, (_, (relative, offset)) in enumerate(ordered):
                end = ordered[i+1][0] if i+1 < len(ordered) else None
                self.log_headers[relative] = (path, offset, end)
        return self.log_headers

    def run_paths(self):
        for path in (self.work/'runs').glob('*/*/run_info.json'):
            if not path.resolve().is_relative_to((self.work/'runs').resolve()):
                continue
            info = read_json(path, {})
            rid = info.get('id')
            if isinstance(rid, str) and rid not in ('', '.', '..') and '/' not in rid and '\\' not in rid:
                yield path.parent, info

    def artifacts(self, rid):
        with self.lock:
            return self._artifacts(rid)

    def _artifacts(self, rid):
        found = next(((run, info) for run, info in self.run_paths() if info['id'] == rid), None)
        if found is None:
            raise FileNotFoundError('학습 기록을 찾을 수 없습니다.')
        run, info = found
        rows = []
        def add(path, base, source, kind):
            if not path.is_file() or not path.resolve().is_relative_to(base.resolve()):
                return
            match = re.search(r'(?:policy_|model_|(?:checkpoint|best|projectile|gait|flat|hard|safety)_it)(\d+)', path.relative_to(base).as_posix())
            rows.append({'name': path.relative_to(base).as_posix(), 'kind': kind,
                         'iteration': int(match[1]) if match else None,
                         'size': path.stat().st_size, 'modified': path.stat().st_mtime,
                         'url': '/trainer-artifact/'+quote(rid, safe='')+'/'+source+'/'+quote(path.relative_to(base).as_posix(), safe='/')})
        for path in run.rglob('*'):
            if path.suffix in ('.onnx', '.pt'):
                kind = 'policy' if path.suffix == '.onnx' else 'checkpoint'
            elif path.suffix in ('.json', '.yaml', '.yml', '.zip'):
                kind = 'config'
            elif path.name.startswith('events.out.tfevents'):
                kind = 'log'
            elif path.suffix in ('.log', '.jsonl', '.tsv'):
                kind = 'log'
            else:
                continue
            add(path, run, 'run', kind)
        directory = self.result/'render'/rid
        if directory.resolve().is_relative_to((self.result/'render').resolve()):
            for path in directory.rglob('*'):
                kind = 'video' if path.suffix == '.mp4' else 'motion' if path.name.endswith('.motion.json') else 'evaluation'
                if path.suffix in ('.mp4', '.json', '.npz', '.tsv', '.csv', '.txt'):
                    add(path, directory, 'result', kind)
        video_directory = self.result/'video'/rid
        if video_directory.resolve().is_relative_to((self.result/'video').resolve()):
            for path in video_directory.rglob('*'):
                if path.name.startswith('.'):
                    continue
                kinds = {'.mp4': 'video', '.json': 'evaluation', '.npz': 'motion',
                         '.npy': 'motion', '.zip': 'motion', '.mjb': 'config',
                         '.xml': 'config', '.onnx': 'policy', '.jsonl': 'evaluation',
                         '.py': 'config', '.txt': 'config', '.log': 'log'}
                if path.suffix in kinds:
                    add(path, video_directory, 'video', kinds[path.suffix])
        relative = run.relative_to(self.work/'runs').as_posix()
        log = self.logs().get(relative, (None, 0, None))[0]
        if log:
            add(log, log.parent, 'log', 'log')
        rows.sort(key=lambda row: (row['kind'], 0 if row['kind'] == 'video' else -(row['iteration'] or 0), row['name']))
        return {'id': rid, 'files': rows, 'total_size': sum(row['size'] for row in rows)}

    def artifact_path(self, path):
        rid, source, name = path.split('/', 2)
        catalog = self.artifacts(rid)
        url = '/trainer-artifact/'+quote(rid, safe='')+'/'+source+'/'+quote(name, safe='/')
        if not any(item['url'] == url for item in catalog['files']):
            raise FileNotFoundError('산출물을 찾을 수 없습니다.')
        run = next(run for run, info in self.run_paths() if info['id'] == rid)
        base = (run if source == 'run' else self.result/'render'/rid if source == 'result'
                else self.result/'video'/rid if source == 'video' else self.work/'runs')
        return base/name

    def media(self, directory, name):
        if not name:
            return None
        path = (directory / name).resolve()
        base = self.result.resolve()
        if not path.is_relative_to(base) or not path.is_file():
            return None
        return '/trainer/' + quote(path.relative_to(base).as_posix(), safe='/')

    def evaluations(self, rid):
        directory = self.result / 'render' / rid
        if directory.parent.resolve() != (self.result / 'render').resolve():
            return []
        try:
            with (directory / 'scores.tsv').open() as stream:
                raw = list(csv.DictReader(stream, delimiter='\t'))
        except OSError:
            return []
        def number(value):
            try:
                result = float(value)
                return result if math.isfinite(result) else None
            except (TypeError, ValueError):
                return None
        rows = []
        for source in raw:
            try:
                row = {'iteration': int(source['iter']), 'score': number(source.get('score')),
                       'statistics_version': int(source.get('statistics_version') or 1),
                       'evaluation_scope': source.get('evaluation_scope'),
                       'score_kind': source.get('score_kind'), 'purpose_status': source.get('purpose_status'),
                       'evaluated': source.get('evaluated'), 'metrics': {}}
                for key in ('fall', 'vel_err', 'yaw_err', 'touchdown', 'peak_bw', 'ball_fall', 'ball_cases', 'ball_hits',
                            'final_err', 'vmax', 'tilt_deg', 'safety_score', 'safety_cases',
                            'safety_invalid', 'safety_exposure_s'):
                    if source.get(key):
                        row['metrics'][key] = number(source[key])
                row['video'] = self.media(directory, source.get('video'))
                filename = source.get('video', '')
                row['motion'] = self.media(directory, str(Path(filename).with_suffix('.motion.json'))) if filename else None
                rows.append(row)
            except (ValueError, TypeError, KeyError):
                continue
        return sorted(rows, key=lambda row: row['iteration'])

    def snapshot(self):
        with self.lock:
            if self.cached is not None and time.monotonic() - self.cached_at < 3:
                return self.cached
            now = time.time()
            server = self.server_status()
            logs = self.logs()
            runs = []
            for run, info in self.run_paths():
                rid, action = info.get('id'), info.get('action')
                if not isinstance(action, str):
                    continue
                relative = run.relative_to(self.work / 'runs').as_posix()
                log, offset, end = logs.get(relative, (None, 0, None))
                registered = info.get('schema_version') == 1
                if registered:
                    log = run / 'train.log' if (run / 'train.log').is_file() else None
                    offset, end = 0, None
                head = read_part(log, 64000, offset=offset, end=end) if log else ''
                tail = read_part(log, 60000, True, offset=offset, end=end) if log else ''
                steps = list(re.finditer(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', tail))
                first = re.search(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', head)
                current, total = map(int, steps[-1].groups()) if steps else (None, None)
                if registered:
                    if isinstance(info.get('iteration'), int):
                        current = max(current if current is not None else -1, info['iteration'])
                    if total is None and isinstance(info.get('total_iterations'), int):
                        total = info['total_iterations']
                start = info.get('start_iteration', int(first.group(1)) if first else current)
                progress = None if current is None or total is None or start is None else max(0, min(1, (current - start + 1) / max(1, total - start)))
                modified = max(log.stat().st_mtime if log else 0, (run/'run_info.json').stat().st_mtime)
                complete = current is not None and total is not None and current >= total - 1
                running = server['connected'] and any(p.get('pid') == info.get('pid') if registered else end is None and log and p.get('log') == log.name for p in server['processes'])
                failed = 'Traceback (most recent call last)' in tail or bool(re.search(r'^(?:\w*Error|CUDA error):', tail, re.M))
                if registered and info.get('state') in ('completed', 'failed', 'stopped'):
                    state = info['state']
                elif running:
                    state = 'starting' if info.get('state') == 'starting' else 'running'
                else:
                    checked_after_creation = (server.get('checked') or 0) >= (info.get('created_at') or 0)
                    stopped = server['connected'] and (not registered or checked_after_creation)
                    state = 'failed' if failed else 'completed' if complete and not registered else 'stopped' if stopped else 'unknown'
                if state == 'completed':
                    progress = 1
                latest_block = tail[steps[-1].start():] if steps else tail
                def metric(pattern):
                    found = re.search(pattern, latest_block)
                    return float(found.group(1)) if found else None
                eta = re.search(r'ETA:\s*([\d:]+)', latest_block)
                checkpoints = []
                for checkpoint in run.glob('policy_*.onnx'):
                    match = re.fullmatch(r'policy_(\d+)\.onnx', checkpoint.name)
                    if match:
                        checkpoints.append(int(match.group(1)))
                evaluations = self.evaluations(rid)
                rendered = [row for row in evaluations if row['video']]
                version = rendered[-1]['statistics_version'] if rendered else None
                scored = [row for row in rendered if row['score'] is not None
                          and row['statistics_version'] == version]
                runs.append({'id': rid, 'action': action, 'tags': info.get('tags', ''),
                             'evaluation_profile': info.get('evaluation_profile'),
                             'description': info.get('description', ''),
                             'started': info.get('started', ''), 'num_envs': info.get('num_envs'),
                             'state': state, 'iteration': current, 'total': total, 'progress': progress,
                             'reward': metric(r'Mean reward:\s*([-\d.]+)'),
                             'eta': eta.group(1) if eta and state == 'running' else None,
                             'updated': modified, 'checkpoint': max(checkpoints, default=None),
                             'checkpoint_count': len(checkpoints), 'auto_render': bool(info.get('render')),
                             'error': info.get('error'), 'registered': registered,
                             'latest': rendered[-1] if rendered else None,
                             'best': max(scored, key=lambda row: row['score']) if scored else None,
                             'evaluations': evaluations})
            runs.sort(key=lambda run: (run['state'] in ('starting', 'running'), run['started'], run['updated']), reverse=True)
            renderer = read_json(self.result / 'status.json', {})
            renderer.update(self.renderer.snapshot() if self.renderer else
                            {'active': False, 'managed_by': 'operation_center'})
            self.cached = finite({'runs': runs, 'renderer': renderer, 'server': server, 'updated': now})
            self.cached_at = time.monotonic()
            return self.cached
