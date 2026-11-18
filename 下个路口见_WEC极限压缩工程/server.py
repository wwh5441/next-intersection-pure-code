#!/usr/bin/env python3
"""Minimal WEC Web Player Server (`server.py`, port 8961)."""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import tempfile
from pathlib import Path

from wec_codec import decode_wec

ROOT = Path(__file__).resolve().parent
DEFAULT_WEC = ROOT / "下个路口见_V9R2.8.wec"


def _build_response(wec_path: Path, name: str) -> bytes:
    objs = decode_wec(wec_path)
    meta = objs["data/studio-metadata.js"]
    payload = {
        "fileName": name,
        "wecKB": round(wec_path.stat().st_size / 1024.0, 2),
        "title": meta["config"].get("title", name),
        "lyrics": meta["lyrics"]["lines"],
        "renderArgs": {
            "config": meta["config"],
            "speakerModel": objs["models/vocal-speaker-model.js"],
            "phonemeAtlas": objs["models/vocal-phoneme-atlas.js"],
            "backingModel": objs["models/backing-model.js"],
            "spaceFdn": objs["models/space-fdn.js"],
            "vocalScore": objs["data/vocal-score.js"],
            "backingScore": objs["data/backing-score.js"],
            "bandModels": {
                "bass": objs["models/bass-model.js"],
                "drums": objs["models/drum-model.js"],
                "poly": objs["models/poly-model.js"],
            },
            "bandScores": {
                "bass": objs["data/bass-score.js"],
                "drums": objs["data/drum-score.js"],
                "poly": objs["data/poly-score.js"],
            },
            "bandGains": objs["data/band-gains.js"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class WecHandler(http.server.SimpleHTTPRequestHandler):
    def _send_json_bytes(self, raw: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/api/decode":
            self._send_json_bytes(_build_response(DEFAULT_WEC, DEFAULT_WEC.name))
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?")[0] == "/api/decode":
            n = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(n)
            with tempfile.NamedTemporaryFile(suffix=".wec") as tf:
                tf.write(data)
                tf.flush()
                self._send_json_bytes(_build_response(Path(tf.name), "uploaded.wec"))
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8961)
    args = ap.parse_args()
    handler = functools.partial(WecHandler, directory=str(ROOT))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"WEC Minimal Player listening at http://127.0.0.1:{args.port}/", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
