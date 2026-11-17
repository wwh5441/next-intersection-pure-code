#!/usr/bin/env python3
"""Build standalone offline single-file HTML for pure-code V9R2 (`tools/build_single_html.py`)."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CSS_FILES = [
    "styles/tokens.css",
    "styles/layout.css",
    "styles/components/header-metrics.css",
    "styles/components/transport.css",
    "styles/components/visualizer.css",
    "styles/components/mixer.css",
    "styles/components/syllable-inspector.css",
]

JS_FILES = [
    "models/vocal-speaker-model.js",
    "models/vocal-phoneme-atlas.js",
    "models/backing-model.js",
    "models/space-fdn.js",
    "models/bass-model.js",
    "models/drum-model.js",
    "models/poly-model.js",
    "data/studio-metadata.js",
    "data/vocal-score.js",
    "data/backing-score.js",
    "data/bass-score.js",
    "data/drum-score.js",
    "data/poly-score.js",
    "data/band-gains.js",
    "src/dsp/fft-core.js",
    "src/dsp/v9r21-vocal-synth.js",
    "src/dsp/bass-harmonic-synth.js",
    "src/dsp/drum-event-synth.js",
    "src/dsp/poly-note-synth.js",
    "src/dsp/v21-band-synth.js",
    "src/dsp/v9r2-master-engine.js",
    "src/modules/pure-code-assets.js",
    "src/state/studio-store.js",
    "src/modules/multitrack-audio-engine.js",
    "src/components/header-metrics.js",
    "src/components/transport-bar.js",
    "src/components/pitch-coart-canvas.js",
    "src/components/layer-mixer-panel.js",
    "src/components/syllable-atlas-inspector.js",
    "src/main.js",
]


def strip_esm(src: str) -> str:
    src = re.sub(r"^\s*import\s+[\s\S]*?from\s+['\"][^'\"]+['\"];?\s*$", "", src, flags=re.MULTILINE)
    return re.sub(r"^\s*export\s+(const|let|var|function|class)\s+", r"\1 ", src, flags=re.MULTILINE)


def main() -> None:
    css = "\n\n".join((ROOT / p).read_text(encoding="utf-8") for p in CSS_FILES)
    # Each module wrapped in its own block scope to avoid helper-name collisions; exported names hoisted via `var`.
    parts = ["window.__V9R2_SINGLE_HTML__ = true;"]
    for p in JS_FILES:
        src = (ROOT / p).read_text(encoding="utf-8")
        exports = re.findall(r"^\s*export\s+(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)", src, flags=re.MULTILINE)
        body = strip_esm(src).replace("import.meta.url", "location.href")
        hoist = "".join(f"\n  __X.{n} = {n};" for n in exports)
        parts.append(f"// === {p} ===\n{{\n{body}{hoist}\n}}")
    names = set()
    for p in JS_FILES:
        names.update(re.findall(r"^\s*export\s+(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)", (ROOT / p).read_text(encoding="utf-8"), flags=re.MULTILINE))
    # Make exported names visible to later modules: declare them as getters on a shared object, then rebind before each block.
    preamble = "const __X = {};\n"
    rebinding = "".join(f"var {n};\n" for n in sorted(names))
    blocks = []
    for blk in parts[1:]:
        blocks.append("".join(f"{n} = __X.{n} ?? {n};\n" for n in sorted(names)) + blk)
    js = parts[0] + "\n" + preamble + rebinding + "\n".join(blocks)
    # Inside each block the module declares its own exports with const/function, shadowing the outer var — correct semantics.
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>下个，路口，见 · V9R2.6 真声门共振峰纯代码版</title>
  <style>
{css}
  </style>
</head>
<body>
  <main class="studio-shell">
    <header id="header-panel" class="panel-card studio-header"></header>
    <section id="transport-panel" class="panel-card"></section>
    <section class="studio-grid">
      <div id="visualizer-panel" class="panel-card"></div>
      <div id="mixer-panel" class="panel-card"></div>
    </section>
    <section id="atlas-panel" class="panel-card"></section>
  </main>
  <script>
(() => {{
{js}
}})();
  </script>
</body>
</html>
"""
    out = ROOT / "下个路口见_V9R2.6_松弛开阔人声纯代码版.html"
    raw = html.encode("utf-8")
    out.write_bytes(raw)
    print(f"Built {out.name}: Raw={len(raw)/1024:.1f} KB | Gzip={len(gzip.compress(raw, 6))/1024:.1f} KB")


if __name__ == "__main__":
    main()
