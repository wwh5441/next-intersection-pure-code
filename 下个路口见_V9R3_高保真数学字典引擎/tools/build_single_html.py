#!/usr/bin/env python3
"""Bundles standalone zero-dependency single-HTML deliverables for V9R3
while keeping the source studio (`index.html`, `styles/`, `models/`, `data/`, `src/`) 100% decoupled.
"""
import gzip
import os
import re

ROOT = "/Users/weckham/Desktop/MUSIC/下个路口见_V9R3_高保真数学字典引擎"

CSS_FILES = [
    "styles/tokens.css",
    "styles/layout.css",
    "styles/components/header-metrics.css",
    "styles/components/transport.css",
    "styles/components/visualizer.css",
    "styles/components/mixer.css",
    "styles/components/inspector.css",
]

JS_MODULES_ALL = [
    "models/vocal-wavecycle-dictionary.js",
    "models/backing-space-dictionary.js",
    "models/drum-percussion-dictionary.js",
    "models/bass-groove-dictionary.js",
    "models/guitar-rhodes-dictionary.js",
    "data/song-structure.js",
    "data/vocal-lyrics.js",
    "data/vocal-score-master.js",
    "data/vocal-score-compact.js",
    "data/band-score-master.js",
    "data/band-score-compact.js",
    "data/tiers-registry.js",
    "data/benchmarks.js",
    "src/dsp/polyphase-sinc.js",
    "src/dsp/lapped-atom-synth.js",
    "src/dsp/vocal-wavecycle-synth.js",
    "src/dsp/drum-percussion-synth.js",
    "src/dsp/bass-guitar-rhodes-synth.js",
    "src/dsp/backing-fdn-synth.js",
    "src/dsp/engine-orchestrator.js",
    "src/state/studio-store.js",
    "src/modules/audio-mixer.js",
    "src/components/header-metrics.js",
    "src/components/transport-bar.js",
    "src/components/score-motif-canvas.js",
    "src/components/stem-mixer.js",
    "src/components/dictionary-inspector.js",
    "src/main.js",
]


def strip_esm_syntax(code: str) -> str:
    # Strip `import ... from '...';` (including multi-line imports)
    code = re.sub(r"^\s*import\s+[\s\S]*?from\s+['\"][^'\"]+['\"];?\s*$", "", code, flags=re.MULTILINE)
    # Strip `export ` keyword before const/function/class
    code = re.sub(r"^\s*export\s+(?=(?:const|let|var|function|async\s+function|class)\b)", "", code, flags=re.MULTILINE)
    # Guard `import.meta.url` when bundled in single HTML
    code = code.replace("import.meta.url", "window.location.href")
    return code


def build_bundle(out_filename: str, default_tier: str, include_master: bool, include_compact: bool):
    css_bundle = "\n".join(open(os.path.join(ROOT, p), encoding="utf-8").read() for p in CSS_FILES)

    js_parts = ["window.__V9R3_SINGLE_FILE__ = true;"]
    for rel in JS_MODULES_ALL:
        if rel == "data/vocal-score-master.js" and not include_master:
            js_parts.append("const VOCAL_SCORE_MASTER = VOCAL_SCORE_COMPACT;")
            continue
        if rel == "data/band-score-master.js" and not include_master:
            js_parts.append("const BAND_SCORE_MASTER = BAND_SCORE_COMPACT;")
            continue
        if rel == "data/vocal-score-compact.js" and not include_compact:
            js_parts.append("const VOCAL_SCORE_COMPACT = VOCAL_SCORE_MASTER;")
            continue
        if rel == "data/band-score-compact.js" and not include_compact:
            js_parts.append("const BAND_SCORE_COMPACT = BAND_SCORE_MASTER;")
            continue

        raw_js = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if rel == "data/tiers-registry.js":
            raw_js = raw_js.replace('"studio_master"', f'"{default_tier}"', 1)
            if not include_master or not include_compact:
                # Filter single tier in UI dropdown cleanly
                keep_key = "studio_master" if include_master else "compact_pure"
                drop_key = "compact_pure" if include_master else "studio_master"
                raw_js += f'\ndelete V9R3_TIERS["{drop_key}"];\n'
        js_parts.append(f"// --- {rel} ---\n" + strip_esm_syntax(raw_js))

    full_js = "\n".join(js_parts)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>下个，路口，见 · V9R3 高保真数学字典合成引擎 ({default_tier})</title>
  <link rel="icon" href="data:," />
  <style>
{css_bundle}
  </style>
</head>
<body>
  <main class="studio-shell">
    <section class="panel-card" id="panel-header-metrics"></section>
    <section class="panel-card" id="panel-transport"></section>
    <section class="panel-card" id="panel-visualizer"></section>
    <section class="panel-card" id="panel-mixer"></section>
    <section class="panel-card" id="panel-inspector"></section>
    <footer class="studio-footer">
      V9R3 Standalone Pure Math + Code + Dictionary Edition · Exact 98.0000 BPM Grid · 48kHz 10-Stem Independent Synthesis
    </footer>
  </main>
  <script type="module">
{full_js}
  </script>
</body>
</html>
"""
    out_path = os.path.join(ROOT, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    gz = len(gzip.compress(html.encode("utf-8"), compresslevel=9))
    print(f"  -> Built {out_filename:38s} | Raw={len(html)/1024:8.1f} KB | Gzip={gz/1024:7.1f} KB")


if __name__ == "__main__":
    print("=== Building V9R3 Standalone Single-HTML Editions ===")
    build_bundle("下个路口见_V9R3_高保真母带字典版.html", "studio_master", include_master=True, include_compact=False)
    build_bundle("下个路口见_V9R3_纯原型轻量字典版.html", "compact_pure", include_master=False, include_compact=True)
