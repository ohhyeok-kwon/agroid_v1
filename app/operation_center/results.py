"""Operation Center-owned result renderer, including lock-owner handover."""
import os
from pathlib import Path
import signal
import subprocess
import threading


class ResultRenderer:
    def __init__(self, root):
        self.root = Path(root)
        self.process = None
        self.worker = None
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.log = None

    def start(self):
        with self.lock:
            if self.worker is not None or self.stopped.is_set():
                return
            path = self.root / '.agent/operation_center/results.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log = path.open('ab', buffering=0)
            self.worker = threading.Thread(target=self._watch, name='result-renderer', daemon=True)
            self.worker.start()

    def _watch(self):
        while not self.stopped.is_set():
            with self.lock:
                if self.stopped.is_set():
                    break
                if self.process is None or self.process.poll() is not None:
                    if self.process is not None:
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    try:
                        self.process = subprocess.Popen(
                            [str(self.root / 'trainer/scripts/05_results.sh'), '--owner-pid', str(os.getpid())],
                            cwd=self.root, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
                            start_new_session=True)
                    except OSError as error:
                        self.log.write((str(error) + '\n').encode())
            # A second app retries after the current daemon releases its lock.
            self.stopped.wait(5)

    def snapshot(self):
        with self.lock:
            active = self.process is not None and self.process.poll() is None
            return {'active': active, 'pid': self.process.pid if active else None,
                    'owner_pid': os.getpid(), 'managed_by': 'operation_center'}

    def close(self):
        self.stopped.set()
        with self.lock:
            process = self.process
            worker = self.worker
            self.process = None
            self.worker = None
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # The leader can exit before an uncooperative grandchild does.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if worker is not None:
            worker.join(timeout=3)
        with self.lock:
            if self.log is not None:
                self.log.close()
                self.log = None
