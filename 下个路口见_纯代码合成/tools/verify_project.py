#!/usr/bin/env python3
"""V4 Integrity, Base122 Roundtrip & Zero-Audio-Binary Auditor."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import struct
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.lzma_packer import decode_base122, sha256_hex, unpack_mdp4_lzma

FORBIDDEN_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".pt", ".pth", ".onnx", ".npz", ".pyc"}


def _read_js_b122(path: Path) -> bytes:
    txt = path.read_bytes().decode("utf-8")
    s = txt.index('"') + 1
    e = txt.rindex('"')
    return decode_base122(txt[s:e])


def main() -> None:
    forbidden_found = [
        str(p.relative_to(PROJECT_ROOT))
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    assert not forbidden_found, f"Forbidden binary files found: {forbidden_found}"

    verified_streams = 0
    for profile in ("compact", "faithful"):
        manifest = json.loads((PROJECT_ROOT / f"data/{profile}/manifest.json").read_text(encoding="utf-8"))
        entries = []
        for track in manifest["tracks"]:
            if "loopBank" in track:
                entries.append(track["loopBank"])
            entries.extend(track["streams"])

        for entry in entries:
            js_name = entry["file"].replace(".lzpar", ".js")
            lzma_bytes = _read_js_b122(PROJECT_ROOT / f"data/{profile}/{js_name}")
            assert len(lzma_bytes) == entry["compressedBytes"], f"Length mismatch on {profile}/{js_name}"
            raw = unpack_mdp4_lzma(lzma_bytes)
            assert len(raw) == entry["parameterBytes"]
            assert sha256_hex(raw) == entry["rawSha256"], f"SHA-256 mismatch on {profile}/{js_name}"
            verified_streams += 1

    single_html = PROJECT_ROOT / "下个路口见_纯代码合成_单文件版.html"
    assert single_html.is_file(), "Missing single-file HTML build"
    html_txt = single_html.read_bytes().decode("utf-8")
    m = re.search(r'<script id="vfs-b122" type="text/plain">([\s\S]*?)</script>', html_txt)
    assert m, "Missing vfs-b122 payload in single-file HTML"
    raw_bundle = decode_base122(m.group(1))
    meta_len = struct.unpack_from("<I", raw_bundle, 0)[0]
    vfs = json.loads(gzip.decompress(raw_bundle[4 : 4 + meta_len]).decode("utf-8"))
    streams_blob = raw_bundle[4 + meta_len :]
    for key, (off, length) in vfs["streams"].items():
        chunk = streams_blob[off : off + length]
        assert len(chunk) == length
        unpack_mdp4_lzma(chunk)

    print(
        json.dumps(
            {
                "version": 4,
                "zeroForbiddenBinaries": True,
                "verifiedStreams": verified_streams,
                "singleHtmlBytes": single_html.stat().st_size,
                "singleHtmlEmbeddedStreams": len(vfs["streams"]),
                "status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
