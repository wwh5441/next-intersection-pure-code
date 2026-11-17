#!/usr/bin/env python3
"""Minimal Zero-Cache Static HTTP Server for V9 Pure Mathematical Code Studio."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheESModuleHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[V9-MathStudio] {self.address_string()} - {fmt % args}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Serve V9 Pure Mathematical Code Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8945)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    handler = partial(NoCacheESModuleHandler, directory=str(root_dir))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving V9 Pure Mathematical Code Studio at http://{args.host}:{args.port}/ (root={root_dir})", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
