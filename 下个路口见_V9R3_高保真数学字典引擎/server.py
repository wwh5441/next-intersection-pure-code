#!/usr/bin/env python3
"""Zero-cache, zero-bytecode HTTP server for V9R3 High-Fidelity Math+Code+Dictionary Studio."""
import argparse
import http.server
import os
import socketserver

ROOT = os.path.dirname(os.path.abspath(__file__))


class StudioHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8949)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    with ReusableTCPServer((args.host, args.port), StudioHandler) as httpd:
        print(f"V9R3 High-Fidelity Math+Code+Dictionary Studio serving at http://{args.host}:{args.port}/", flush=True)
        httpd.serve_forever()
