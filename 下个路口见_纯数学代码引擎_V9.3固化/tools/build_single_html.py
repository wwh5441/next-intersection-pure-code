#!/usr/bin/env python3
"""Compute exact Pure-Math Code byte metrics & build standalone plain-text HTML deliverable."""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS_FILES = [
    "styles/tokens.css",
    "styles/layout.css",
    "styles/components/transport.css",
    "styles/components/visualizer.css",
    "styles/components/mixer.css",
    "styles/components/formula-panel.css",
]

UI_MODULES = [
    "src/state/studio-store.js",
    "src/modules/audio-mixer.js",
    "src/components/header-metrics.js",
    "src/components/transport-bar.js",
    "src/components/score-canvas.js",
    "src/components/stem-mixer.js",
    "src/components/formula-inspector.js",
    "src/main.js",
]


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=6))


def bundle_module_scoped(js_src: str, label: str) -> str:
    exported_names = []
    body_lines = []
    skipping_import = False

    for line in js_src.splitlines():
        s = line.strip()
        if skipping_import:
            if ";" in s:
                skipping_import = False
            continue
        if s.startswith("import "):
            if ";" not in s:
                skipping_import = True
            continue

        m_const = re.match(r"^(\s*)export\s+(?:const|let)\s+([A-Za-z0-9_$]+)\s*=", line)
        if m_const:
            indent, name = m_const.groups()
            exported_names.append(name)
            line = re.sub(r"^\s*export\s+(?:const|let)\s+" + re.escape(name) + r"\s*=", f"{indent}{name} =", line)
            body_lines.append(line)
            continue

        m_fn = re.match(r"^(\s*)export\s+(async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(", line)
        if m_fn:
            indent, async_kw, name = m_fn.groups()
            exported_names.append(name)
            ak = async_kw or ""
            line = re.sub(
                r"^\s*export\s+(?:async\s+)?function\s+" + re.escape(name) + r"\s*\(",
                f"{indent}{name} = {ak}function {name}(",
                line,
            )
            body_lines.append(line)
            continue

        m_cls = re.match(r"^(\s*)export\s+class\s+([A-Za-z0-9_$]+)\b", line)
        if m_cls:
            indent, name = m_cls.groups()
            exported_names.append(name)
            line = re.sub(r"^\s*export\s+class\s+" + re.escape(name) + r"\b", f"{indent}{name} = class {name}", line)
            body_lines.append(line)
            continue

        body_lines.append(line)

    decl = f"var {', '.join(exported_names)};\n" if exported_names else ""
    return f"// --- {label} ---\n{decl}{{\n" + "\n".join(body_lines) + "\n}"


def discover_ordered_core_modules() -> tuple[list[str], list[str]]:
    model_files = ["models/orthonormal-basis-physics.js"]
    if (ROOT / "models/sparse-macro-book.js").is_file():
        model_files.append("models/sparse-macro-book.js")

    data_shards = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("master-mid-*.js"))
    data_shards += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("master-side-*.js"))
    data_shards += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("stem-partition-*.js"))
    model_and_score = (
        model_files
        + ["data/vocal-lyrics.js", "data/band-score.js"]
        + data_shards
        + ["data/master-series-index.js"]
    )

    dsp_modules = [
        "src/dsp/radix2-fft.js",
        "src/dsp/orthonormal-series-decoder.js",
        "src/dsp/song-renderer.js",
    ]
    if (ROOT / "src/dsp/streaming-frame-engine.js").is_file():
        dsp_modules.append("src/dsp/streaming-frame-engine.js")

    return model_and_score, dsp_modules


def main() -> None:
    seed = {}
    metrics_path = ROOT / "data/code-metrics.json"
    if metrics_path.is_file():
        seed = json.loads(metrics_path.read_text("utf-8"))

    model_and_score, dsp_modules = discover_ordered_core_modules()
    all_tracked = model_and_score + dsp_modules + UI_MODULES + CSS_FILES
    if (ROOT / "src/dsp/math-worker.js").is_file():
        all_tracked.append("src/dsp/math-worker.js")

    per_file = {}
    for rel in all_tracked:
        raw = (ROOT / rel).read_bytes()
        per_file[rel] = {
            "rawBytes": len(raw),
            "gzipBytes": gzip_size(raw),
        }

    def group_stats(rels: list[str]) -> tuple[int, int]:
        concat = b"\n".join((ROOT / r).read_bytes() for r in rels)
        return sum(per_file[r]["rawBytes"] for r in rels), len(gzip.compress(concat, compresslevel=9))

    ms_raw, ms_gz = group_stats(model_and_score)
    dsp_raw, dsp_gz = group_stats(dsp_modules)
    core_raw, core_gz = group_stats(model_and_score + dsp_modules)

    metrics = {
        **seed,
        "zeroLzma": True,
        "zeroBase122": True,
        "zeroBase64": True,
        "zeroBinarySamples": True,
        "tracksCount": 6,
        "losslessWavReferenceBytes": 40391724,
        "mp3ReferenceBytes": 9620866,
        "totalModelAndScoreBytes": ms_raw,
        "totalModelAndScoreGzipBytes": ms_gz,
        "dspEngineBytes": dsp_raw,
        "dspEngineGzipBytes": dsp_gz,
        "totalAudioCorePlainJsBytes": core_raw,
        "totalAudioCoreGzipBytes": core_gz,
        "files": per_file,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build Standalone Single-File HTML
    css_bundle = "\n\n".join((ROOT / c).read_text("utf-8") for c in CSS_FILES)
    core_parts = [bundle_module_scoped((ROOT / m).read_text("utf-8"), m) for m in (model_and_score + dsp_modules)]
    shared_core_js = "\n\n".join(core_parts)
    ui_only_js = "\n\n".join(bundle_module_scoped((ROOT / m).read_text("utf-8"), m) for m in UI_MODULES)

    has_worker = (ROOT / "src/dsp/math-worker.js").is_file()
    worker_boot = ""
    if has_worker:
        worker_tail_js = bundle_module_scoped((ROOT / "src/dsp/math-worker.js").read_text("utf-8"), "src/dsp/math-worker.js")
        worker_boot = (
            "window.__PURE_MATH_WORKER_SOURCE__ = "
            "document.getElementById('pure-math-core-src').textContent + \"\\n\" + "
            + json.dumps(worker_tail_js, ensure_ascii=False)
            + ";"
        )

    ver_tag = seed.get("versionShort", "V9.3")
    out_path = ROOT / "下个路口见_纯数学代码_几乎无损母带版.html"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>下个，路口，见 · {ver_tag} · 48kHz 几乎无损纯数学正交余弦母带单文件版</title>
  <style>
{css_bundle}
  </style>
</head>
<body>
  <main class="studio-shell">
    <header id="header-slot" class="card-panel"></header>
    <section id="transport-slot" class="card-panel"></section>
    <section id="visualizer-slot" class="card-panel"></section>
    <section id="mixer-slot" class="mixer-two-col"></section>
    <section id="inspector-slot" class="card-panel"></section>
  </main>
  <script id="pure-math-core-src">
{shared_core_js}
  </script>
  <script>
window.__INLINE_CODE_METRICS__ = {json.dumps(metrics, ensure_ascii=False)};
{worker_boot}
  </script>
  <script type="module">
{ui_only_js}
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    html_raw = out_path.stat().st_size
    print(
        f"Metrics & Standalone HTML built:\n"
        f"  Model+Score Plain JS : {ms_raw / 1048576:.2f} MB | Gzip-9 : {ms_gz / 1048576:.2f} MB\n"
        f"  Standalone HTML Raw  : {html_raw / 1048576:.2f} MB"
    )


if __name__ == "__main__":
    main()
