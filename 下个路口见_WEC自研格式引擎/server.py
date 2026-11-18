#!/usr/bin/env python3
"""HTTP Range-Capable Static Server for WEC1 (.wec) Native Audio Studio."""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class WecRangeRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wec": "application/x-wec-audio",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
    }

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isfile(path) and "Range" in self.headers:
            m = re.match(r"bytes=(\d+)-(\d*)", self.headers["Range"])
            if m:
                fsize = os.path.getsize(path)
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else fsize - 1
                end = min(end, fsize - 1)
                if 0 <= start <= end:
                    f = open(path, "rb")
                    f.seek(start)
                    self.range_len = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-type", self.guess_type(path))
                    self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
                    self.send_header("Content-Length", str(self.range_len))
                    self.end_headers()
                    return f
        self.range_len = None
        return super().send_head()

    def copyfile(self, source, outputfile) -> None:
        if getattr(self, "range_len", None) is not None:
            rem = self.range_len
            while rem > 0:
                chunk = source.read(min(65536, rem))
                if not chunk:
                    break
                outputfile.write(chunk)
                rem -= len(chunk)
        else:
            super().copyfile(source, outputfile)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8946)
    args = ap.parse_args()
    handler = partial(WecRangeRequestHandler, directory=str(ROOT))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"WEC1 Native Audio Studio serving at http://127.0.0.1:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
