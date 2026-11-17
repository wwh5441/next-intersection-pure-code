#!/usr/bin/env python3
"""Zero-cache static HTTP server for V9R1 Pure-Code Dual-Dictionary Studio (default port 8947)."""
import argparse
import functools
import http.server
import os
import socketserver


class ZeroCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript; charset=utf-8",
        ".mjs": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".html": "text/html; charset=utf-8",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class ReusableTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="Serve V9R1 Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8947)
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(ZeroCacheHandler, directory=root_dir)
    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        print(f"[V9R1 Studio] Serving {root_dir} at http://{args.host}:{args.port}/", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
