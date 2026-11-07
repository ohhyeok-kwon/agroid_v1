"""Local Operation Center UI and joint-feedback API. No robot control endpoints."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import signal
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

from telemetry import Telemetry

APP = Path(__file__).resolve().parent
PROJECT = APP.parents[1]


def create_server(port=0, telemetry=None, ui=None):
    state = telemetry if telemetry is not None else Telemetry(PROJECT, APP)
    ui = Path(ui) if ui is not None else PROJECT / '.agent/operation_center/ui'
    assets = APP / 'assets'

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
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

        def valid_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.valid_host():
                return self.json(403, {'error': '잘못된 호스트입니다.'})
            path = unquote(urlsplit(self.path).path)
            if path == '/api/status':
                return self.json(200, state.snapshot())
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
                else:
                    return self.json(404, {'error': '지원하지 않는 작업입니다.'})
                self.json(200, state.snapshot())
            except (ValueError, TypeError, AttributeError) as error:
                self.json(400, {'error': str(error)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.telemetry = state
    return server


def main():
    parser = argparse.ArgumentParser(description='Operation Center 로컬 상태 화면')
    parser.add_argument('--port', type=int, default=8797)
    parser.add_argument('--parent', type=int)
    args = parser.parse_args()
    if not (PROJECT / '.agent/operation_center/ui/index.html').is_file():
        parser.error('scripts/01_operation_center.sh로 화면을 먼저 빌드하세요.')
    server = create_server(args.port)

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
        server.telemetry.disconnect()
        server.server_close()


if __name__ == '__main__':
    main()
