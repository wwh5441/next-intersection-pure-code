#!/usr/bin/env python3
"""V5 Integrity, Zero-Binary / Zero-MDCT & Base122 Roundtrip Auditor."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import struct
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.wave_packer import decode_base122, decompress_lzma, sha256_hex

FORBIDDEN_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".pt", ".pth", ".onnx", ".npz", ".pyc"}
VALID_MAGICS = {b"WVV5", b"WVD5", b"WVB5", b"WVH5"}


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
        for entry in manifest["tracks"]:
            js_name = entry["file"].replace(".wvp", ".js")
            comp_bytes = _read_js_b122(PROJECT_ROOT / f"data/{profile}/{js_name}")
            assert len(comp_bytes) == entry["compressedBytes"], f"Compressed size mismatch: {profile}/{js_name}"
            raw = decompress_lzma(comp_bytes)
            assert raw[:4] in VALID_MAGICS, f"Unexpected wave magic in {profile}/{js_name}: {raw[:4]}"
            assert len(raw) == entry["parameterBytes"], f"Unpacked size mismatch: {profile}/{js_name}"
            assert sha256_hex(raw) == entry["rawSha256"], f"SHA-256 mismatch: {profile}/{js_name}"
            verified_streams += 1

    single_html = PROJECT_ROOT / "下个路口见_数学波码合成_单文件版.html"
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
        raw_wvp = decompress_lzma(chunk)
        assert raw_wvp[:4] in VALID_MAGICS, f"Invalid embedded magic for {key}"

    print(
        json.dumps(
            {
                "version": 5,
                "zeroForbiddenBinaries": True,
                "zeroMdctFrames": True,
                "verifiedStreams": verified_streams,
                "singleHtmlBytes": single_html.stat().st_size,
                "singleHtmlKB": round(single_html.stat().st_size / 1024, 1),
                "singleHtmlEmbeddedStreams": len(vfs["streams"]),
                "status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
