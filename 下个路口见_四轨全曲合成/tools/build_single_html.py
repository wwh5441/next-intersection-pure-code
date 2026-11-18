#!/usr/bin/env python3
"""Dual-Profile (`compact` + `faithful`) UTF-8 Base122 Offline Single-File HTML Compiler (Full-Song 4-Track)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import struct
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.wave_packer import decode_base122, encode_base122


def trim_js_lines(code: str) -> str:
    out = []
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        out.append(s)
    return "\n".join(out)


def strip_es_module_syntax(code: str) -> str:
    cleaned = re.sub(r"import\s+(?:[\s\S]*?)\s+from\s+['\"][^'\"]+['\"];?\s*", "", code)
    cleaned = re.sub(r"^export\s+(?=(?:class|function|const|let|var)\b)", "", cleaned, flags=re.MULTILINE)
    return trim_js_lines(cleaned)


def collect_topological_modules(root: Path, entry_rel: str):
    order = []
    visited = set()
    dep_map = {}

    def visit(rel: str):
        if rel in visited:
            return
        visited.add(rel)
        p = root / rel
        txt = p.read_text(encoding="utf-8")
        deps = []
        for m in re.finditer(r"""(?:from|import)\s*\(?\s*['"](\.[^'"]+)['"]""", txt):
            spec = m.group(1)
            nxt = (p.parent / spec).resolve()
            if nxt.is_file():
                nxt_rel = nxt.relative_to(root).as_posix()
                deps.append([spec, nxt_rel])
                visit(nxt_rel)
        dep_map[rel] = deps
        order.append(rel)

    visit(entry_rel)
    return order, dep_map


def extract_b122_from_js_module(js_path: Path) -> bytes:
    raw_txt = js_path.read_bytes().decode("utf-8")
    start = raw_txt.index('"') + 1
    end = raw_txt.rindex('"')
    return decode_base122(raw_txt[start:end])


def render_html(root: Path, profiles_to_embed: list[str], sizes_data: dict) -> tuple[bytes, int, int]:
    css_order = [
        "styles/tokens.css",
        "styles/layout.css",
        "styles/components/player.css",
        "styles/components/presets.css",
        "styles/components/multitrack.css",
        "styles/components/inspector.css",
    ]
    bundled_css = "\n".join((root / rel).read_text(encoding="utf-8") for rel in css_order)

    index_html = (root / "index.html").read_text(encoding="utf-8")
    main_match = re.search(r"(<main class=\"studio\">[\s\S]*?</main>)", index_html)
    if not main_match:
        raise ValueError("Could not locate <main class=\"studio\"> in index.html")
    main_markup = main_match.group(1)

    orig_profiles = json.loads((root / "data/profiles.json").read_text(encoding="utf-8"))
    embedded_profiles_json = {
        "default": profiles_to_embed[0],
        "available": profiles_to_embed,
        "labels": orig_profiles["labels"],
    }

    json_files = {
        "data/profiles.json": json.dumps(embedded_profiles_json, ensure_ascii=False, separators=(",", ":")),
        "data/sizes.json": json.dumps(sizes_data, ensure_ascii=False, separators=(",", ":")),
    }

    stream_slices = {}
    binary_chunks: list[bytes] = []
    bin_cursor = 0

    for prof in profiles_to_embed:
        manifest_path = root / f"data/{prof}/manifest.json"
        json_files[f"data/{prof}/manifest.json"] = manifest_path.read_text(encoding="utf-8")
        for js_file in sorted((root / f"data/{prof}").glob("*.js")):
            vwp_bytes = extract_b122_from_js_module(js_file)
            vwp_name = js_file.name.replace(".js", ".vwp")
            stream_slices[f"{prof}/{vwp_name}"] = [bin_cursor, len(vwp_bytes)]
            binary_chunks.append(vwp_bytes)
            bin_cursor += len(vwp_bytes)

    vendor_lzma = (root / "src/vendor/lzma-d-min.js").read_text(encoding="utf-8")
    storage_code = strip_es_module_syntax((root / "src/dsp/storage.js").read_text(encoding="utf-8"))
    worker_script = (
        vendor_lzma
        + "\n"
        + storage_code
        + """
self.onmessage = ({ data }) => {
  try {
    const decoded = self.LZMA.decompress(new Uint8Array(data.buffer));
    if (typeof decoded === 'string') throw new Error('Unexpected string output in binary VWP6 packet');
    const raw = Uint8Array.from(decoded);
    const buffer = unpackWavePacket(raw);
    self.postMessage({ id: data.id, buffer }, [buffer]);
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message || String(error) });
  }
};
"""
    )

    dsp_order = [
        "src/dsp/fft.js",
        "src/dsp/resampler.js",
        "src/dsp/bark-bands.js",
        "src/dsp/wave-stream.js",
        "src/dsp/air-stream.js",
        "src/dsp/reverb-synth.js",
        "src/dsp/engine.js",
        "src/dsp/worklet.js",
    ]
    worklet_script = "\n".join(
        strip_es_module_syntax((root / rel).read_text(encoding="utf-8")) for rel in dsp_order
    )

    mod_order, dep_map = collect_topological_modules(root, "src/main.js")
    modules_payload = [
        [rel, trim_js_lines((root / rel).read_text(encoding="utf-8")), dep_map[rel]]
        for rel in mod_order
    ]

    meta_obj = {
        "json": json_files,
        "streams": stream_slices,
        "worker": worker_script,
        "worklet": worklet_script,
        "modules": modules_payload,
    }
    meta_gz = gzip.compress(
        json.dumps(meta_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        compresslevel=9,
    )
    raw_streams_blob = b"".join(binary_chunks)
    combined_binary = struct.pack("<I", len(meta_gz)) + meta_gz + raw_streams_blob
    b122_payload = encode_base122(combined_binary)

    title_tag = (
        "下个，路口，见 · 全曲四轨精粹单文件离线版"
        if profiles_to_embed == ["compact"]
        else "下个，路口，见 · 全曲四轨（1轨人声+3轨伴奏）双档单文件离线版"
    )
    html_out = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title_tag}</title>
  <link rel="icon" href="data:,">
  <style>
{bundled_css}
  </style>
</head>
<body>
{main_markup}
<script id="vfs-b122" type="text/plain">{b122_payload}</script>
<script>
(async function bootSingleHtml() {{
  const ILLEGALS = [0, 10, 13, 34, 60, 92];
  function decodeB122(str) {{
    const colon = str.indexOf(':');
    const outLen = Number(str.slice(0, colon));
    const out = new Uint8Array(outLen);
    let bitPos = 0;
    function push7(v) {{
      const byteIdx = bitPos >> 3, bitOff = bitPos & 7;
      if (byteIdx < outLen) {{
        const shift = 1 - bitOff;
        if (shift >= 0) out[byteIdx] |= (v << shift) & 0xff;
        else {{
          out[byteIdx] |= v >> (-shift);
          if (byteIdx + 1 < outLen) out[byteIdx + 1] |= (v << (8 + shift)) & 0xff;
        }}
      }}
      bitPos += 7;
    }}
    for (let i = colon + 1; i < str.length; i++) {{
      const cp = str.charCodeAt(i);
      if (cp < 0x80) push7(cp);
      else {{
        const payload = cp - 0x100;
        const idx = payload >> 7, v2 = payload & 0x7f;
        if (idx === 7) push7(v2);
        else {{ push7(ILLEGALS[idx]); push7(v2); }}
      }}
    }}
    return out;
  }}

  const rawBundle = decodeB122(document.getElementById('vfs-b122').textContent);
  const metaLen = new DataView(rawBundle.buffer, rawBundle.byteOffset, 4).getUint32(0, true);
  const metaGz = rawBundle.subarray(4, 4 + metaLen);
  const streamsBlob = rawBundle.subarray(4 + metaLen);

  const ds = new DecompressionStream('gzip');
  const vfs = JSON.parse(await new Response(new Blob([metaGz]).stream().pipeThrough(ds)).text());

  const origFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {{
    const urlStr = typeof input === 'string' ? input : (input && input.url) || '';
    const norm = urlStr.replace(/^\\.\\//, '').split('?')[0];
    if (Object.prototype.hasOwnProperty.call(vfs.json, norm)) {{
      return new Response(vfs.json[norm], {{
        status: 200,
        headers: {{ 'Content-Type': 'application/json; charset=utf-8' }}
      }});
    }}
    return origFetch(input, init);
  }};

  const binStreams = {{}};
  for (const [key, [off, len]] of Object.entries(vfs.streams)) {{
    binStreams[key] = streamsBlob.subarray(off, off + len);
  }}
  window.__VFS_BINARY_STREAMS = binStreams;
  window.__VFS_WORKER_URL = URL.createObjectURL(
    new Blob([vfs.worker], {{ type: 'application/javascript' }})
  );

  const wkBytes = new TextEncoder().encode(vfs.worklet);
  let wkBin = '';
  for (let i = 0; i < wkBytes.length; i += 0x8000) {{
    wkBin += String.fromCharCode.apply(null, wkBytes.subarray(i, i + 0x8000));
  }}
  window.__VFS_WORKLET_DATA_URL = 'data:text/javascript;base64,' + btoa(wkBin);

  const blobUrls = {{}};
  for (const [rel, rawCode, deps] of vfs.modules) {{
    let code = rawCode;
    for (const [spec, targetRel] of deps) {{
      const targetUrl = blobUrls[targetRel];
      if (!targetUrl) throw new Error('Missing topological module: ' + targetRel);
      code = code.split("'" + spec + "'").join(JSON.stringify(targetUrl));
      code = code.split('"' + spec + '"').join(JSON.stringify(targetUrl));
    }}
    blobUrls[rel] = URL.createObjectURL(new Blob([code], {{ type: 'application/javascript' }}));
  }}

  await import(blobUrls['src/main.js']);
}})().catch((err) => {{
  console.error(err);
  const el = document.getElementById('error');
  if (el) {{ el.textContent = String(err); el.hidden = false; }}
}});
</script>
</body>
</html>
"""
    return html_out.encode("utf-8"), len(combined_binary), len(raw_streams_blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", default=["compact", "faithful"], choices=["compact", "faithful"])
    parser.add_argument("--output", default="下个路口见_四轨全曲合成_双档完整单文件版.html")
    args = parser.parse_args()

    root = PROJECT_ROOT
    sizes_path = root / "data/sizes.json"
    sizes_data = json.loads(sizes_path.read_text(encoding="utf-8"))

    final_bytes, bin_len, stream_len = render_html(root, args.profiles, sizes_data)
    key_name = "compactSingleHtmlBytes" if args.profiles == ["compact"] else "dualSingleHtmlBytes"
    sizes_data[key_name] = len(final_bytes)
    sizes_path.write_text(json.dumps(sizes_data, ensure_ascii=False, indent=2), encoding="utf-8")

    out_path = root / args.output
    out_path.write_bytes(final_bytes)
    print(f"Built {out_path.name} (Full-Song 4-Track Base122 HTML)")
    print(f"  Embedded Profiles: {', '.join(args.profiles)}")
    print(f"  Wave-Code LZMA:    {stream_len:,} B ({stream_len / 1e6:.2f} MB)")
    print(f"  Binary Bundle:     {bin_len:,} B ({bin_len / 1e6:.2f} MB)")
    print(f"  Final UTF-8 HTML:  {len(final_bytes):,} B ({len(final_bytes) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
