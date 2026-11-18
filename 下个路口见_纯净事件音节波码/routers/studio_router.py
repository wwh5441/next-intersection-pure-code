"""Static ES-Module & Telemetry HTTP Request Handler for V8 Event-Syllable Studio."""
from __future__ import annotations

from http.server import SimpleHTTPRequestHandler
import json
from pathlib import Path


def create_studio_handler(root_dir: Path):
    class StudioRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root_dir), **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def do_GET(self):
            if self.path == "/api/health":
                sizes_path = root_dir / "data/sizes.json"
                payload = json.loads(sizes_path.read_text(encoding="utf-8")) if sizes_path.is_file() else {"ok": True}
                body = json.dumps({"status": "ok", "telemetry": payload}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, fmt, *args):
            return

    return StudioRequestHandler
