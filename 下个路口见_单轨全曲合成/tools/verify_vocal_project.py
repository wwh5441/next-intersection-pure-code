#!/usr/bin/env python3
"""V7 Full-Song 1-Track Zero-Binary, Dual-Profile Embed & Bit-Exact Roundtrip Auditor."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import struct
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.wave_packer import decode_base122, sha256_hex, unpack_vwp6_lzma

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
        assert len(manifest["tracks"]) == 1, f"Expected 1 full-song track in {profile}"
        for track in manifest["tracks"]:
            for stream in track["streams"]:
                js_name = stream["file"].replace(".vwp", ".js")
                comp_bytes = _read_js_b122(PROJECT_ROOT / f"data/{profile}/{js_name}")
                assert len(comp_bytes) == stream["compressedBytes"], f"Size mismatch: {profile}/{js_name}"
                raw_pkt = unpack_vwp6_lzma(comp_bytes)
                assert raw_pkt[:4] in (b"VWQ6", b"VWA7", b"VWR7"), f"Invalid wave magic in {profile}/{js_name}"
                assert len(raw_pkt) == stream["parameterBytes"]
                assert sha256_hex(raw_pkt) == stream["rawSha256"], f"SHA-256 mismatch: {profile}/{js_name}"
                verified_streams += 1

    single_html = PROJECT_ROOT / "下个路口见_单轨全曲合成_双档完整单文件版.html"
    compact_html = PROJECT_ROOT / "下个路口见_单轨全曲合成_精粹单文件版.html"
    assert single_html.is_file() and compact_html.is_file(), "Missing single-file HTML builds"
    html_txt = single_html.read_bytes().decode("utf-8")
    m = re.search(r'<script id="vfs-b122" type="text/plain">([\s\S]*?)</script>', html_txt)
    assert m, "Missing vfs-b122 payload in single-file HTML"
    raw_bundle = decode_base122(m.group(1))
    meta_len = struct.unpack_from("<I", raw_bundle, 0)[0]
    vfs = json.loads(gzip.decompress(raw_bundle[4 : 4 + meta_len]).decode("utf-8"))
    embedded_profiles = json.loads(vfs["json"]["data/profiles.json"])["available"]
    assert embedded_profiles == ["compact", "faithful"], f"Both profiles must be clickable: {embedded_profiles}"

    streams_blob = raw_bundle[4 + meta_len :]
    for key, (off, length) in vfs["streams"].items():
        chunk = streams_blob[off : off + length]
        assert len(chunk) == length
        raw_pkt = unpack_vwp6_lzma(chunk)
        assert raw_pkt[:4] in (b"VWQ6", b"VWA7", b"VWR7"), f"Invalid embedded magic for {key}"

    print(
        json.dumps(
            {
                "version": 7,
                "zeroForbiddenBinaries": True,
                "fullSongTracks": 1,
                "verifiedStreams": verified_streams,
                "embeddedProfiles": embedded_profiles,
                "compactSingleHtmlMB": round(compact_html.stat().st_size / 1e6, 2),
                "dualSingleHtmlMB": round(single_html.stat().st_size / 1e6, 2),
                "status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
