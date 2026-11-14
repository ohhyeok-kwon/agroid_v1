"""Local Operation Center UI and joint-feedback API. No robot control endpoints."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit

from telemetry import Telemetry
from trainer import Trainer
from simulation import Simulation

APP = Path(__file__).resolve().parent
PROJECT = APP.parents[1]


def create_server(port=0, telemetry=None, ui=None, trainer=None, renderer=None):
    state = telemetry if telemetry is not None else Telemetry(PROJECT, APP)
    ui = Path(ui) if ui is not None else PROJECT / '.agent/operation_center/ui'
    assets = APP / 'assets'
    training = trainer if trainer is not None else Trainer(PROJECT, renderer=renderer)
    simulation = Simulation(PROJECT, APP)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

        def log_message(self, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def respond(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            if self.command != 'HEAD':
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def json(self, status, value):
            self.respond(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

        def stream(self):
            self.connection.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,128*1024)
            self.connection.settimeout(2)
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Connection','close')
            self.end_headers()
            self.close_connection = True
            if self.command=='HEAD':
                return
            # WebKitGTK 2.52 crashes its web process when this image stream ends before any part
            # without the closing boundary. Keep idle streams open and end only with the boundary.
            frame_id = -1
            try:
                while True:
                    update = simulation.wait_frame(frame_id,timeout=1)
                    if update is None:
                        if simulation.closed:
                            break
                        # Heartbeat detects closed clients without completing a part: a discarded preamble
                        # line before the first part, afterwards bytes after the JPEG end marker.
                        self.wfile.write(b'\r\n');self.wfile.flush()
                        continue
                    frame_id,frame = update
                    if frame is None:
                        continue
                    header = f'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\nX-Frame-Id: {frame_id}\r\n\r\n'.encode()
                    self.wfile.write(header+frame+b'\r\n');self.wfile.flush()
                self.wfile.write(b'--frame--\r\n');self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,TimeoutError):
                pass

        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def media(self, target, download=False):
            size = target.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                if match and any(match.groups()):
                    left, right = match.groups()
                    start = int(left) if left else max(0, size - int(right))
                    end = min(int(right), size - 1) if left and right else size - 1
                else:
                    start = size
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(max(0, end - start + 1)))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if download:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''"+quote(target.name, safe=''))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            try:
                with target.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        data = stream.read(min(1024 * 1024, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.valid_host():
                return self.json(403, {'error': '잘못된 호스트입니다.'})
            path = unquote(urlsplit(self.path).path)
            if path == '/api/status':
                return self.json(200, state.snapshot())
            if path == '/api/trainer/runs':
                return self.json(200, training.snapshot())
            if path == '/api/trainer/artifacts':
                rid = parse_qs(urlsplit(self.path).query).get('run', [''])[0]
                try:
                    return self.json(200, training.artifacts(rid))
                except (FileNotFoundError, ValueError):
                    return self.json(404, {'error': '학습 기록을 찾을 수 없습니다.'})
            if path.startswith('/trainer-artifact/'):
                try:
                    target = training.artifact_path(path[len('/trainer-artifact/'):])
                    return self.media(target, download=True)
                except (FileNotFoundError, ValueError):
                    return self.json(404, {'error': '산출물을 찾을 수 없습니다.'})
            if path == '/api/simulation':
                return self.json(200, simulation.snapshot())
            if path == '/api/simulation/frame':
                frame = simulation.frame()
                return self.respond(200 if frame else 204,frame or b'','image/jpeg')
            if path == '/api/simulation/stream':
                return self.stream()
            if path.startswith('/trainer/'):
                base = training.result.resolve()
                target = (base / path[len('/trainer/'):]).resolve()
                if not target.is_relative_to(base):
                    return self.json(403, {'error': '허용되지 않은 경로입니다.'})
                if not target.is_file() or not (target.suffix == '.mp4' or target.name.endswith('.motion.json')):
                    return self.json(404, {'error': '렌더링 결과가 없습니다.'})
                return self.media(target)
            base, relative = (assets, path[8:]) if path.startswith('/assets/') else (ui, path.lstrip('/') or 'index.html')
            target = (base / relative).resolve()
            if not target.is_relative_to(base.resolve()):
                return self.json(403, {'error': '허용되지 않은 경로입니다.'})
            if not target.is_file():
                return self.json(404, {'error': '파일을 찾을 수 없습니다.'})
            mime = mimetypes.guess_type(str(target))[0] or 'application/octet-stream'
            self.respond(200, target.read_bytes(), mime)

        do_HEAD = do_GET

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}')
            if not self.valid_host() or origin not in allowed or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.json(403, {'error': '허용되지 않은 요청입니다.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('요청 크기가 잘못되었습니다.')
                data = json.loads(self.rfile.read(size))
                if self.path == '/api/connect':
                    state.connect(data.get('host', ''))
                elif self.path == '/api/disconnect':
                    state.disconnect()
                elif self.path == '/api/simulation/input':
                    simulation.command(data)
                    return self.json(200, simulation.snapshot())
                else:
                    return self.json(404, {'error': '지원하지 않는 작업입니다.'})
                self.json(200, state.snapshot())
            except (ValueError, TypeError, AttributeError) as error:
                self.json(400, {'error': str(error)})

    class OperationServer(ThreadingHTTPServer):
        owned_closed = False

        def server_close(self):
            if not self.owned_closed:
                self.owned_closed = True
                # Stop all independent children together, before the native
                # window's shutdown deadline expires.
                workers = []
                for callback in (renderer.close if renderer else None,
                                 simulation.close, getattr(training, 'close', None),
                                 getattr(state, 'close', state.disconnect)):
                    if callback:
                        worker = threading.Thread(target=callback, daemon=True)
                        worker.start()
                        workers.append(worker)
                deadline = time.monotonic()+4
                for worker in workers:
                    worker.join(timeout=max(0, deadline-time.monotonic()))
            super().server_close()

    server = OperationServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.telemetry = state
    server.simulation = simulation
    server.trainer = training
    server.renderer = renderer
    return server


def main():
    parser = argparse.ArgumentParser(description='Operation Center 로컬 상태 화면')
    parser.add_argument('--port', type=int, default=8797)
    parser.add_argument('--parent', type=int)
    parser.add_argument('--with-results', action='store_true',
                        help='브라우저 서버가 켜져 있는 동안 자동 평가·렌더링 실행')
    args = parser.parse_args()
    if not (PROJECT / '.agent/operation_center/ui/index.html').is_file():
        parser.error('scripts/01_operation_center.sh로 화면을 먼저 빌드하세요.')
    renderer = None
    if args.parent or args.with_results:
        from results import ResultRenderer
        renderer = ResultRenderer(PROJECT)
    server = create_server(args.port, renderer=renderer)
    if renderer:
        renderer.start()

    def stop(*args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    if args.parent:
        def monitor():
            while os.getppid() == args.parent:
                time.sleep(0.5)
            stop()
        threading.Thread(target=monitor, daemon=True).start()
    print(f'http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
