#!/usr/bin/env python3
"""Zero-Binary, SHA-256 Roundtrip & Modular Architecture Auditor for V8 Event-Syllable Engine."""
from __future__ import annotations

import json
import lzma
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


def _decode_v7_js_mod(path: Path) -> bytes:
    txt = path.read_text(encoding="utf-8")
    s = txt.index('"') + 1
    e = txt.rindex('"')
    payload = txt[s:e]
    colon = payload.index(":")
    out_len = int(payload[:colon])
    illegals = [0, 10, 13, 34, 60, 92]
    out = bytearray(out_len)
    bit_pos = 0

    def push7(v: int) -> None:
        nonlocal bit_pos
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        if byte_idx < out_len:
            shift = 1 - bit_off
            if shift >= 0:
                out[byte_idx] |= (v << shift) & 0xFF
            else:
                out[byte_idx] |= v >> -shift
                if byte_idx + 1 < out_len:
                    out[byte_idx + 1] |= (v << (8 + shift)) & 0xFF
        bit_pos += 7

    for ch in payload[colon + 1 :]:
        cp = ord(ch)
        if cp < 0x80:
            push7(cp)
        else:
            val = cp - 0x100
            idx = val >> 7
            v2 = val & 0x7F
            if idx == 7:
                push7(v2)
            else:
                push7(illegals[idx])
                push7(v2)
    return bytes(out)


def main() -> None:
    # Clean any accidental __pycache__ first check
    forbidden = [
        str(p.relative_to(PROJECT_ROOT))
        for p in PROJECT_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in FORBIDDEN_EXTS
    ]
    assert not forbidden, f"Forbidden binary files found: {forbidden}"

    # Check index.html has zero inline <style> blocks
    idx_txt = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")
    assert "<style" not in idx_txt, "index.html must not contain inline <style> blocks"

    expected_magics = {
        "speakerCore": b"VGS8",
        "syllableAtlas": b"VGA8",
        "spaceFdn": b"VGR8",
        "eventStream": b"VGE8",
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

    accomp_verified = 0
    v8_dict_mf = json.loads((PROJECT_ROOT / "data/accomp_v8_dict/manifest.json").read_text(encoding="utf-8"))
    v8_dict_total = 0
    for t_name, t_info in v8_dict_mf["tracks"].items():
        comp = _decode_js_mod(PROJECT_ROOT / "data/accomp_v8_dict" / f"{t_name}_dict.js")
        assert len(comp) == t_info["compressedBytes"], f"V8 dict size mismatch on {t_name}"
        raw = decompress_lzma1_alone(comp)
        assert raw[:4] == b"VDA8", f"V8 dict magic mismatch on {t_name}: {raw[:4]}"
        assert len(raw) == t_info["rawBytes"]
        assert sha256_hex(raw) == t_info["rawSha256"]
        v8_dict_total += len(comp)
        accomp_verified += 1
    assert v8_dict_total == v8_dict_mf["totalCompressedBytes"]

    v7_streams = {
        "bass.js": b"VWP7",
        "drums_body.js": b"VWP7",
        "drums_air.js": b"VWAP",
        "accomp_mid.js": b"VWP7",
        "accomp_side.js": b"VWP7",
        "accomp_air.js": b"VWAP",
    }
    v7_total = 0
    for fname, exp_magic in v7_streams.items():
        comp = _decode_v7_js_mod(PROJECT_ROOT / "data/accomp_faithful" / fname)
        raw = lzma.decompress(comp, format=lzma.FORMAT_ALONE)
        assert raw[:4] == exp_magic, f"V7 Accomp magic mismatch on {fname}: {raw[:4]}"
        v7_total += len(comp)
        accomp_verified += 1

    for html_name in (
        "下个路口见_全曲事件波码_极简版.html",
        "下个路口见_全曲事件波码_双档完整版.html",
    ):
        hp = PROJECT_ROOT / html_name
        assert hp.is_file() and hp.stat().st_size > 20000, f"Missing or invalid {html_name}"

    ultra_vocal_b = json.loads((PROJECT_ROOT / "data/ultra/manifest.json").read_text())["totalCompressedBytes"]
    expr_vocal_b = json.loads((PROJECT_ROOT / "data/expressive/manifest.json").read_text())["totalCompressedBytes"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "zeroForbiddenBinaries": True,
                "verifiedVocalPackets": verified,
                "verifiedAccompPackets": accomp_verified,
                "ultraVocalKB": round(ultra_vocal_b / 1024, 2),
                "ultraDictAccompKB": round(v8_dict_total / 1024, 2),
                "ultraFullSongKB": round((ultra_vocal_b + v8_dict_total) / 1024, 2),
                "expressiveVocalKB": round(expr_vocal_b / 1024, 2),
                "expressiveFullSongKB": round((expr_vocal_b + v7_total) / 1024, 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
