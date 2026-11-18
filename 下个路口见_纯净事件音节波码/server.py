#!/usr/bin/env python3
"""Local Studio HTTP Server for V8.3 Purified Event-Syllable Vocal Genesis Workbench."""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from routers.studio_router import create_studio_handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8939)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    handler = create_studio_handler(PROJECT_ROOT)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"V8.3 Purified Event-Syllable Vocal Studio running on http://{args.host}:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
