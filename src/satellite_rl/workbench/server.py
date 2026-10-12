"""Launch the local research workbench: python -m satellite_rl.workbench.server."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import traceback
from urllib.parse import parse_qs, unquote, urlparse

from .service import Application
from .storage import SOURCE_ROOT, clean_json, inside, now


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, application):
        self.application = application
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "SatelliteWorkbench/1.0"

    def log_message(self, format, *args):
        # Frequent scalar polling does not drown out startup and exceptions.
        if len(args) > 1 and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

    def _headers(self, status, content_type, size, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'self'; base-uri 'self'")
        if filename:
            self.send_header("Content-Disposition", 'attachment; filename="' + filename.replace('"', '') + '"')
        self.end_headers()

    def _json(self, value, status=200):
        data = json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def _local_request(self):
        allowed = {"127.0.0.1", "localhost", "::1"}
        host = urlparse("http://" + self.headers.get("Host", "")).hostname
        if host not in allowed:
            raise PermissionError("工作台只接受本机访问")
        origin = self.headers.get("Origin")
        if origin and (urlparse(origin).hostname not in allowed or urlparse(origin).port != self.server.server_port):
            raise PermissionError("拒绝来自其他页面的跨源请求")

    def _dispatch(self, method):
        try:
            self._local_request()
            parsed = urlparse(self.path)
            parts = [unquote(piece) for piece in parsed.path.strip("/").split("/") if piece]
            query = parse_qs(parsed.query)
            if parts and parts[0] == "api":
                body = {}
                if method == "POST":
                    if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                        raise ValueError("请求须使用application/json")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4 * 1024 * 1024:
                        raise ValueError("请求内容为空或过大")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("请求须为JSON对象")
                if parts[1:2] == ["ai-search-r2"]:
                    from .coverage_ai_search_r2 import api as r2_api
                    self._json(r2_api(self.server.application, method, parts[2:], body))
                elif parts[1:2] == ["ai-search"]:
                    from .coverage_ai_search import api as ai_search_api
                    self._json(ai_search_api(self.server.application, method, parts[2:], body))
                elif parts[1:2] == ["geoload"]:
                    from .coverage_geoload import api as geoload_api
                    self._json(geoload_api(self.server.application, method, parts[2:], body))
                elif parts[1:2] == ["fair20"]:
                    from .fair20_service import Fair20Application
                    self._json(Fair20Application(self.server.application).api(method, parts[2:], body))
                elif len(parts) == 4 and parts[1] == "runs" and parts[3] == "download" and method == "GET":
                    content_type, filename, data = self.server.application.download(parts[2], query.get("file", [""])[0])
                    self._headers(200, content_type, len(data), filename)
                    self.wfile.write(data)
                elif len(parts) == 4 and parts[1] == "models" and parts[3] == "download" and method == "GET":
                    content_type, filename, data = self.server.application.download_model(parts[2])
                    self._headers(200, content_type, len(data), filename)
                    self.wfile.write(data)
                else:
                    self._json(self.server.application.api(method, parts[1:], query, body))
            elif method == "GET":
                root = Path(__file__).parent / "static"
                page = ("ai_search_r2.html" if parsed.path in ("/ai-search-r2", "/ai-search-r2/") else
                        "ai_search.html" if parsed.path in ("/ai-search", "/ai-search/") else
                        "fair20.html" if parsed.path in ("/fair20", "/fair20/") else (parsed.path.lstrip("/") or "index.html"))
                path = inside(root / page, root)
                if path.suffix not in (".html", ".js", ".css", ".svg", ".ico") or not path.is_file():
                    raise FileNotFoundError("页面不存在")
                data = path.read_bytes()
                self._headers(200, (mimetypes.guess_type(path.name)[0] or "application/octet-stream") + "; charset=utf-8", len(data))
                self.wfile.write(data)
            else:
                raise FileNotFoundError("接口不存在")
        except PermissionError as error:
            self._json({"error": str(error)}, 403)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, 404)
        except (ValueError, KeyError, TypeError) as error:
            self._json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as error:
            traceback.print_exc()
            self._json({"error": str(error), "type": type(error).__name__}, 500)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-scheduler", action="store_true", help="只查看与调试接口，不启动后台作业")
    args = parser.parse_args(argv)
    application = Application(args.archive_root, python_executable=args.python, max_workers=args.max_workers)
    server = WorkbenchServer(("127.0.0.1", args.port), application)
    if not args.no_scheduler:
        application.scheduler.start()
    print(f"卫星资源分配实验工作台 http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"实验与模型保存根目录：{application.store.archive_root}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        application.scheduler.stop_scheduler() if not args.no_scheduler else None
        server.server_close()


if __name__ == "__main__":
    main()
