#!/usr/bin/env python3
"""Zero-Binary, SHA-256 Roundtrip & Modular Architecture Auditor for V8.3 Purified Engine."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.event_codec import decode_base122, decompress_lzma1_alone, sha256_hex

FORBIDDEN_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".pt", ".pth", ".onnx", ".npz", ".pyc"}


def _decode_js_mod(path: Path) -> bytes:
    txt = path.read_text(encoding="utf-8")
    s = txt.index('"') + 1
    e = txt.rindex('"')
    return decode_base122(txt[s:e])


def main() -> None:
    forbidden = [
        str(p.relative_to(PROJECT_ROOT))
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in FORBIDDEN_EXTS
    ]
    assert not forbidden, f"Forbidden binary files found: {forbidden}"

    idx_txt = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")
    assert "<style" not in idx_txt, "index.html must not contain inline <style> blocks"

    expected_magics = {
        "speakerCore": b"VGS9",
        "syllableAtlas": b"VGA9",
        "spaceFdn": b"VGR9",
        "eventStream": b"VGE9",
    }

    verified = 0
    for prof in ("ultra", "expressive"):
        mf_dir = PROJECT_ROOT / f"data/{prof}"
        manifest = json.loads((mf_dir / "manifest.json").read_text(encoding="utf-8"))
        sum_comp = 0
        for key, pkt in manifest["packets"].items():
            js_path = (mf_dir / pkt["path"]).resolve()
            comp = _decode_js_mod(js_path)
            assert len(comp) == pkt["compressedBytes"], f"Compressed size mismatch for {prof}/{key}"
            raw = decompress_lzma1_alone(comp)
            assert raw[:4] == expected_magics[key], f"Magic mismatch on {key}: {raw[:4]}"
            assert len(raw) == pkt["rawBytes"], f"Raw size mismatch for {prof}/{key}"
            assert sha256_hex(raw) == pkt["rawSha256"], f"SHA-256 mismatch for {prof}/{key}"
            sum_comp += len(comp)
            verified += 1
        assert sum_comp == manifest["totalCompressedBytes"]

    htmls = list(PROJECT_ROOT.glob("下个路口见_纯人声去残纯净_*.html"))
    assert len(htmls) >= 2, f"Expected 2 offline single-file HTMLs, found {[p.name for p in htmls]}"
    for hp in htmls:
        assert hp.stat().st_size > 20000, f"Invalid HTML size for {hp}"

    sizes = json.loads((PROJECT_ROOT / "data/sizes.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "zeroForbiddenBinaries": True,
                "verifiedPackets": verified,
                "ultraKB": sizes["profiles"]["ultra"]["totalKB"],
                "expressiveKB": sizes["profiles"]["expressive"]["totalKB"],
                "ultraHnrDb": sizes["profiles"]["ultra"]["purity"]["hnrDb"],
                "expressiveHnrDb": sizes["profiles"]["expressive"]["purity"]["hnrDb"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
