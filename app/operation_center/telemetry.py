"""Operation Center state, updated exclusively by the ROS 2 receiver process."""
from collections import deque
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from pose import JOINTS, MOTOR_SLOTS, normalize, normalize_pose


class Telemetry:
    def __init__(self, project_root, app_root):
        self.root, self.app = Path(project_root), Path(app_root)
        self.lock = threading.RLock()
        self.operations = threading.Lock()
        self.worker = None
        self.process = None
        self.cancel = threading.Event()
        self.host = self.saved_host()
        self.status = 'disconnected'
        self.message = '로봇을 연결하면 현재 관절 자세를 표시합니다.'
        self.pose = None
        self.received = None
        self.times = deque(maxlen=50)

    def saved_host(self):
        try:
            path = self.root / '.agent/.r1-setup/.r1_wifi.last.local.json'
            return str(ipaddress.IPv4Address(json.loads(path.read_text())['ip']))
        except (OSError, ValueError, KeyError, TypeError):
            return ''

    def snapshot(self):
        with self.lock:
            age = time.monotonic() - self.received if self.received is not None else None
            status, message = self.status, self.message
            if status == 'connected' and age is not None and age > 1.5:
                status, message = 'stale', '관절 상태 수신이 중단되었습니다. 마지막 수신 자세입니다.'
            elapsed = self.times[-1] - self.times[0] if len(self.times) > 1 else 0
            hz = (len(self.times) - 1) / elapsed if elapsed and status == 'connected' else 0
            return {'status': status, 'message': message, 'host': self.host,
                    'age_ms': round(age * 1000) if age is not None else None,
                    'hz': round(hz, 1), 'pose': self.pose, 'transport': 'ros2',
                    'ros_domain_id': os.environ.get('AGROID_ROS_DOMAIN_ID', os.environ.get('ROS_DOMAIN_ID', '0')),
                    'topics': ['/tf', '/tf_static']}

    def update(self, status, message):
        with self.lock:
            self.status, self.message = status, message

    def accept(self, raw):
        self._accept_pose(normalize(raw))

    def _accept_pose(self, pose):
        with self.lock:
            # Repeated frames do not make an old measurement look fresh.
            if self.pose and self.pose['tick'] == pose['tick']:
                return
            self.pose = pose
            self.received = time.monotonic()
            self.times.append(self.received)
            missing = pose.get('missing') or []
            received = f'관절 {len(JOINTS) - len(missing)}/{len(JOINTS)}개를 TF로 수신 중입니다. 미수신: {", ".join(missing)}(마지막 각도 유지).' \
                if missing else 'TF로 관절 자세를 수신 중입니다.'
            self.status, self.message = 'connected', received + ' 골반 기준이며 전신 위치·기울기는 미측정입니다.'

    def _stop(self):
        self.cancel.set()
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.worker:
            self.worker.join(timeout=6)
        self.worker = None

    def disconnect(self):
        with self.operations:
            self._stop()
            self.update('disconnected', '연결을 종료했습니다. 마지막 수신 자세를 유지합니다.' if self.pose else
                        '로봇을 연결하면 현재 관절 자세를 표시합니다.')

    def connect(self, host):
        host = str(ipaddress.IPv4Address(host))
        with self.operations:
            if self.host == host and self.worker and self.worker.is_alive():
                return
            self._stop()
            with self.lock:
                self.host, self.pose, self.received = host, None, None
                self.times.clear()
            self.cancel = threading.Event()
            self.update('connecting', 'R1 ROS 2 수신을 준비하는 중입니다.')
            self.worker = threading.Thread(target=self._read, args=(host,), daemon=True)
            self.worker.start()

    def _read(self, host):
        process = None
        stderr_worker = None
        errors = deque(maxlen=8)
        try:
            folder = self.root / '.agent/ros2/runtime'
            (folder / 'log').mkdir(parents=True, exist_ok=True)
            environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                           'ROS_HOME': str(folder), 'ROS_LOG_DIR': str(folder / 'log'),
                           'ROS_DOMAIN_ID': os.environ.get('AGROID_ROS_DOMAIN_ID', os.environ.get('ROS_DOMAIN_ID', '0')),
                           'RCUTILS_LOGGING_USE_STDOUT': '0'}
            command = [sys.executable, '-u', str(self.app / 'ros_receiver.py'), '--host', host]
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=environment)
            with self.lock:
                self.process = process
            if self.cancel.is_set():
                process.terminate()
                return

            def drain():
                for line in process.stderr:
                    errors.append(line.strip()[-500:])
            stderr_worker = threading.Thread(target=drain, daemon=True)
            stderr_worker.start()
            for line in process.stdout:
                if self.cancel.is_set():
                    break
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError('상태 메시지가 객체가 아닙니다.')
                    if raw.get('event') == 'pose':
                        self._accept_pose(normalize_pose(raw['pose']))
                    elif raw.get('event'):
                        self.update('error' if raw['event'] == 'error' else 'connecting', raw['message'])
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    self.update('error', 'ROS 2 자세 데이터를 해석하지 못했습니다: ' + str(error))
            if not self.cancel.is_set():
                if stderr_worker:
                    stderr_worker.join(timeout=1)
                detail = errors[-1] if errors else 'ROS 2 상태 구독이 종료되었습니다.'
                if 'rclpy' in detail and 'ModuleNotFoundError' in detail:
                    detail = 'ROS 2 환경을 찾지 못했습니다. scripts/01_operation_center.sh로 실행하세요.'
                with self.lock:
                    if self.status != 'error':
                        self.update('error', detail)
        except (OSError, ValueError) as error:
            if not self.cancel.is_set():
                self.update('error', str(error))
        finally:
            if process:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if stderr_worker:
                    stderr_worker.join(timeout=1)
                for stream in (process.stdout, process.stderr):
                    if stream and not stream.closed:
                        stream.close()
            with self.lock:
                self.process = None
