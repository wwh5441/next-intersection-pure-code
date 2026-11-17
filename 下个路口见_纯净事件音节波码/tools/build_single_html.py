#!/usr/bin/env python3
"""Offline Zero-Dependency Single-File HTML Compiler for V8.3 Purified Vocal Genesis."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.event_codec import encode_base122


def _read_b122_payload(js_path: Path) -> str:
    txt = js_path.read_text(encoding="utf-8")
    s = txt.index('"') + 1
    e = txt.rindex('"')
    return txt[s:e]


def _strip_module_syntax(js_code: str) -> str:
    code = re.sub(r"import\s+(?:[\s\S]*?)\s+from\s+['\"][^'\"]+['\"];?\s*", "", js_code)
    code = re.sub(r"import\s+['\"][^'\"]+['\"];?\s*", "", code)
    code = re.sub(r"^export\s+(?=(?:class|function|const|let|var)\b)", "", code, flags=re.MULTILINE)
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("//")]
    return "\n".join(lines)


def build_offline_html(out_path: Path, include_profiles: list[str], default_profile: str) -> int:
    css_files = [
        "styles/tokens.css",
        "styles/layout.css",
        "styles/components/transport.css",
        "styles/components/event-ribbon.css",
        "styles/components/layer-mixer.css",
        "styles/components/syllable-atlas.css",
    ]
    combined_css = "\n".join((PROJECT_ROOT / p).read_text(encoding="utf-8") for p in css_files)

    js_order = [
        "src/vendor/lzma-d-min.js",
        "src/dsp/fft.js",
        "src/dsp/unpacker.js",
        "src/dsp/reverb-fdn.js",
        "src/dsp/sms-event-synth.js",
        "src/state/store.js",
        "src/modules/loader.js",
        "src/modules/audio-controller.js",
        "src/components/metrics-banner.js",
        "src/components/transport-bar.js",
        "src/components/pitch-event-canvas.js",
        "src/components/layer-mixer-panel.js",
        "src/components/syllable-atlas-panel.js",
        "src/main.js",
    ]
    combined_js = "\n".join(_strip_module_syntax((PROJECT_ROOT / p).read_text(encoding="utf-8")) for p in js_order)

    json_vfs = {
        "schemas/vocal_genesis_config.json": (PROJECT_ROOT / "schemas/vocal_genesis_config.json").read_text("utf-8"),
        "data/profiles.json": json.dumps({"default": default_profile, "available": include_profiles}, ensure_ascii=False),
        "data/sizes.json": (PROJECT_ROOT / "data/sizes.json").read_text("utf-8"),
        "data/lyrics.json": (PROJECT_ROOT / "data/lyrics.json").read_text("utf-8"),
        "data/atlas_catalog.json": (PROJECT_ROOT / "data/atlas_catalog.json").read_text("utf-8"),
    }
    b122_vfs = {
        "data/shared/speaker_core.js": _read_b122_payload(PROJECT_ROOT / "data/shared/speaker_core.js"),
        "data/shared/syllable_atlas.js": _read_b122_payload(PROJECT_ROOT / "data/shared/syllable_atlas.js"),
        "data/shared/space_fdn.js": _read_b122_payload(PROJECT_ROOT / "data/shared/space_fdn.js"),
    }
    for pid in include_profiles:
        json_vfs[f"data/{pid}/manifest.json"] = (PROJECT_ROOT / f"data/{pid}/manifest.json").read_text("utf-8")
        b122_vfs[f"data/{pid}/event_stream.js"] = _read_b122_payload(PROJECT_ROOT / f"data/{pid}/event_stream.js")

    vfs_gz = gzip.compress(json.dumps({"json": json_vfs, "b122": b122_vfs}, ensure_ascii=False).encode("utf-8"), compresslevel=9)
    vfs_b122 = encode_base122(vfs_gz)

    index_body = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")
    body_match = re.search(r"<body>([\s\S]*?)<script type=\"module\"", index_body)
    assert body_match, "Could not locate body markup in index.html"
    body_html = body_match.group(1).strip()

    bootstrap_js = f"""
(async function() {{
  {{
    const b122 = document.getElementById('vfs-gz-b122').textContent;
    const B_ILL = [0, 10, 13, 34, 38, 92];
    const out = [];
    let curByte = 0, bitOfByte = 0;
    function push7(b7) {{
      const s = b7 << 1;
      curByte |= s >>> bitOfByte;
      bitOfByte += 7;
      if (bitOfByte >= 8) {{
        out.push(curByte & 0xff);
        bitOfByte -= 8;
        curByte = (s << (7 - bitOfByte)) & 0xff;
      }}
    }}
    for (let i = 0; i < b122.length; i++) {{
      const c = b122.charCodeAt(i);
      if (c > 127) {{
        const idx = (c >>> 8) & 7;
        if (idx !== 7) push7(B_ILL[idx]);
        push7(c & 127);
      }} else push7(c);
    }}
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    writer.write(Uint8Array.from(out));
    writer.close();
    const vfsTxt = await new Response(ds.readable).text();
    globalThis.__VFS__ = JSON.parse(vfsTxt);
  }}
  {combined_js}
}})();
"""
    full_html = (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"UTF-8\"/>\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1.0\"/>\n"
        "<link rel=\"icon\" href=\"data:,\" />\n"
        "<title>下个，路口，见 · 纯净去残事件音节波码引擎 V8.3 (离线单文件版)</title>\n"
        f"<style>{combined_css}</style>\n</head>\n<body>\n{body_html}\n"
        f"<script id=\"vfs-gz-b122\" type=\"text/plain\">{vfs_b122}</script>\n"
        f"<script>{bootstrap_js}</script>\n</body>\n</html>\n"
    )
    out_path.write_text(full_html, encoding="utf-8")
    return out_path.stat().st_size


def main() -> None:
    sizes = json.loads((PROJECT_ROOT / "data/sizes.json").read_text(encoding="utf-8"))
    kb_ultra = int(round(sizes["profiles"]["ultra"]["totalKB"]))
    s_ultra = build_offline_html(
        PROJECT_ROOT / f"下个路口见_纯人声去残纯净_V8.3_极简{kb_ultra}KB版.html",
        include_profiles=["ultra"],
        default_profile="ultra",
    )
    s_dual = build_offline_html(
        PROJECT_ROOT / "下个路口见_纯人声去残纯净_V8.3_双档完整版.html",
        include_profiles=["ultra", "expressive"],
        default_profile="expressive",
    )
    print(f"Built V8.3 single-file HTMLs: ultra_only={s_ultra:,} B ({s_ultra/1024:.1f} KB), dual={s_dual:,} B ({s_dual/1024:.1f} KB)")


if __name__ == "__main__":
    main()
