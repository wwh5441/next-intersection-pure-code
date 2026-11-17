#!/usr/bin/env python3
"""Compute exact Pure-Math Code byte metrics & build standalone plain-text HTML."""

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VOCAL_SCORE_AND_MODELS = [
    "models/singer-physics.js",
    "models/mandarin-phonetics.js",
    "data/vocal-score.js",
]

BAND_SCORE_AND_MODELS = [
    "models/instrument-physics.js",
    "data/band-score.js",
]

DSP_ENGINE_MODULES = [
    "src/dsp/fft-minphase.js",
    "src/dsp/math-vocal-synth.js",
    "src/dsp/math-drums-synth.js",
    "src/dsp/math-bass-guitar-synth.js",
    "src/dsp/math-keys-space-synth.js",
    "src/dsp/song-renderer.js",
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

CSS_FILES = [
    "styles/tokens.css",
    "styles/layout.css",
    "styles/components/transport.css",
    "styles/components/visualizer.css",
    "styles/components/mixer.css",
    "styles/components/formula-panel.css",
]


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9))


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


def compute_metrics() -> dict:
    per_file = {}
    all_tracked = VOCAL_SCORE_AND_MODELS + BAND_SCORE_AND_MODELS + DSP_ENGINE_MODULES + UI_MODULES + CSS_FILES
    for rel in all_tracked:
        raw = (ROOT / rel).read_bytes()
        per_file[rel] = {
            "rawBytes": len(raw),
            "gzipBytes": gzip_size(raw),
        }

    def group_stats(rels):
        concat = b"\n".join((ROOT / r).read_bytes() for r in rels)
        raw_sum = sum(per_file[r]["rawBytes"] for r in rels)
        return raw_sum, gzip_size(concat)

    voc_raw, voc_gz = group_stats(VOCAL_SCORE_AND_MODELS)
    band_raw, band_gz = group_stats(BAND_SCORE_AND_MODELS)
    score_model_raw, score_model_gz = group_stats(VOCAL_SCORE_AND_MODELS + BAND_SCORE_AND_MODELS)
    dsp_raw, dsp_gz = group_stats(DSP_ENGINE_MODULES)
    audio_core_raw, audio_core_gz = group_stats(VOCAL_SCORE_AND_MODELS + BAND_SCORE_AND_MODELS + DSP_ENGINE_MODULES)

    return {
        "version": "V9.0 Pure Mathematical Code Engine",
        "zeroLzma": True,
        "zeroBase122": True,
        "zeroBinarySamples": True,
        "tracksCount": 10,
        "vocalModelAndScoreBytes": voc_raw,
        "vocalModelAndScoreGzipBytes": voc_gz,
        "bandModelAndScoreBytes": band_raw,
        "bandModelAndScoreGzipBytes": band_gz,
        "totalModelAndScoreBytes": score_model_raw,
        "totalModelAndScoreGzipBytes": score_model_gz,
        "dspEngineBytes": dsp_raw,
        "dspEngineGzipBytes": dsp_gz,
        "totalAudioCorePlainJsBytes": audio_core_raw,
        "totalAudioCoreGzipBytes": audio_core_gz,
        "comparisonBaselines": {
            "v7FullSongMdctBytes": 1427584,
            "v85VocalOnlyLzmaBytes": 53207,
            "v85VocalOnlyBase122JsPlusDecoderBytes": 77312,
        },
        "files": per_file,
    }


def build_standalone_html(metrics: dict) -> Path:
    css_bundle = "\n\n".join((ROOT / c).read_text("utf-8") for c in CSS_FILES)

    shared_core_modules = VOCAL_SCORE_AND_MODELS + BAND_SCORE_AND_MODELS + DSP_ENGINE_MODULES
    shared_core_js = "\n\n".join(
        bundle_module_scoped((ROOT / m).read_text("utf-8"), m)
        for m in shared_core_modules
    )
    worker_tail_js = bundle_module_scoped((ROOT / "src/dsp/math-worker.js").read_text("utf-8"), "src/dsp/math-worker.js")
    ui_only_js = "\n\n".join(
        bundle_module_scoped((ROOT / m).read_text("utf-8"), m)
        for m in UI_MODULES
    )

    worker_tail_json = json.dumps(worker_tail_js, ensure_ascii=False)
    metrics_json_str = json.dumps(metrics, ensure_ascii=False)

    out_path = ROOT / "下个路口见_纯数学代码全曲版.html"
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>下个，路口，见 · V9 纯数学代码 10 轨全曲合成版 (零二进制独立单页)</title>
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
window.__INLINE_CODE_METRICS__ = {metrics_json_str};
window.__PURE_MATH_WORKER_SOURCE__ = document.getElementById('pure-math-core-src').textContent + "\\n" + {worker_tail_json};
  </script>
  <script type="module">
{ui_only_js}
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    raw_b = out_path.read_bytes()
    metrics["standaloneHtmlBytes"] = len(raw_b)
    metrics["standaloneHtmlGzipBytes"] = gzip_size(raw_b)
    return out_path


def main():
    metrics = compute_metrics()
    out_html = build_standalone_html(metrics)
    metrics_path = ROOT / "data" / "code-metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== V9 Pure Mathematical Code Engine Byte Report ===")
    print(f"  Vocal Physics + Phonetics + 285-Note Score : {metrics['vocalModelAndScoreBytes']:>7,} B ({metrics['vocalModelAndScoreBytes']/1024:.2f} KB) | Gzip: {metrics['vocalModelAndScoreGzipBytes']:>6,} B ({metrics['vocalModelAndScoreGzipBytes']/1024:.2f} KB)")
    print(f"  Band Physics + 83-Bar Full Score           : {metrics['bandModelAndScoreBytes']:>7,} B ({metrics['bandModelAndScoreBytes']/1024:.2f} KB) | Gzip: {metrics['bandModelAndScoreGzipBytes']:>6,} B ({metrics['bandModelAndScoreGzipBytes']/1024:.2f} KB)")
    print(f"  TOTAL Song Score + Acoustic Models (10 trk): {metrics['totalModelAndScoreBytes']:>7,} B ({metrics['totalModelAndScoreBytes']/1024:.2f} KB) | Gzip: {metrics['totalModelAndScoreGzipBytes']:>6,} B ({metrics['totalModelAndScoreGzipBytes']/1024:.2f} KB)")
    print(f"  10-Track Pure-Math DSP Synthesizer Engine  : {metrics['dspEngineBytes']:>7,} B ({metrics['dspEngineBytes']/1024:.2f} KB) | Gzip: {metrics['dspEngineGzipBytes']:>6,} B ({metrics['dspEngineGzipBytes']/1024:.2f} KB)")
    print(f"  TOTAL Audio Core (Models + Score + Synth)  : {metrics['totalAudioCorePlainJsBytes']:>7,} B ({metrics['totalAudioCorePlainJsBytes']/1024:.2f} KB) | Gzip: {metrics['totalAudioCoreGzipBytes']:>6,} B ({metrics['totalAudioCoreGzipBytes']/1024:.2f} KB)")
    print(f"  Standalone Offline Single HTML             : {metrics['standaloneHtmlBytes']:>7,} B ({metrics['standaloneHtmlBytes']/1024:.2f} KB) | Gzip: {metrics['standaloneHtmlGzipBytes']:>6,} B ({metrics['standaloneHtmlGzipBytes']/1024:.2f} KB)")
    print(f"  Wrote metrics -> {metrics_path}")
    print(f"  Wrote standalone HTML -> {out_html}")


if __name__ == "__main__":
    main()
