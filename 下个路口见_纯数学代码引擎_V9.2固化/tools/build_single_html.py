#!/usr/bin/env python3
"""Compute exact V9.2 Dual-Tier Pure-Math Code byte metrics & build standalone plain-text HTMLs."""

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHARED_VOCAL_MODELS = [
    "models/singer-physics.js",
    "models/mandarin-phonetics.js",
    "data/vocal-lyrics.js",
]

SHARED_BAND_MODELS = [
    "models/instrument-physics.js",
    "data/band-score.js",
]

TIER_VOCAL_SCORE_FILES = {
    "ultra": "data/vocal-score-ultra.js",
    "expressive": "data/vocal-score-expressive.js",
}

TIER_BAND_SCORE_FILES = {
    "ultra": "data/band-score-ultra.js",
    "expressive": "data/band-score-expressive.js",
}

DSP_ENGINE_MODULES = [
    "src/dsp/fft-minphase.js",
    "src/dsp/math-vocal-synth.js",
    "src/dsp/math-drums-synth.js",
    "src/dsp/math-keys-space-synth.js",
    "src/dsp/math-bass-guitar-synth.js",
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
    all_tracked = (
        SHARED_VOCAL_MODELS
        + SHARED_BAND_MODELS
        + list(TIER_VOCAL_SCORE_FILES.values())
        + list(TIER_BAND_SCORE_FILES.values())
        + ["data/vocal-tiers.js"]
        + DSP_ENGINE_MODULES
        + UI_MODULES
        + CSS_FILES
    )
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

    dsp_raw, dsp_gz = group_stats(DSP_ENGINE_MODULES)

    tiers_metrics = {}
    for tid in ["ultra", "expressive"]:
        voc_score_rel = TIER_VOCAL_SCORE_FILES[tid]
        band_score_rel = TIER_BAND_SCORE_FILES[tid]
        voc_rels = SHARED_VOCAL_MODELS + [voc_score_rel]
        band_rels = SHARED_BAND_MODELS + [band_score_rel]
        tot_rels = voc_rels + band_rels
        core_rels = tot_rels + DSP_ENGINE_MODULES

        voc_raw, voc_gz = group_stats(voc_rels)
        band_raw, band_gz = group_stats(band_rels)
        tot_raw, tot_gz = group_stats(tot_rels)
        core_raw, core_gz = group_stats(core_rels)

        tiers_metrics[tid] = {
            "vocalScoreOnlyBytes": per_file[voc_score_rel]["rawBytes"],
            "vocalScoreOnlyGzipBytes": per_file[voc_score_rel]["gzipBytes"],
            "vocalModelAndScoreBytes": voc_raw,
            "vocalModelAndScoreGzipBytes": voc_gz,
            "bandScoreOnlyBytes": per_file[band_score_rel]["rawBytes"],
            "bandScoreOnlyGzipBytes": per_file[band_score_rel]["gzipBytes"],
            "bandModelAndScoreBytes": band_raw,
            "bandModelAndScoreGzipBytes": band_gz,
            "totalModelAndScoreBytes": tot_raw,
            "totalModelAndScoreGzipBytes": tot_gz,
            "totalAudioCorePlainJsBytes": core_raw,
            "totalAudioCoreGzipBytes": core_gz,
        }

    dual_tot_rels = (
        SHARED_VOCAL_MODELS
        + SHARED_BAND_MODELS
        + list(TIER_VOCAL_SCORE_FILES.values())
        + list(TIER_BAND_SCORE_FILES.values())
        + ["data/vocal-tiers.js"]
    )
    dual_raw, dual_gz = group_stats(dual_tot_rels)

    return {
        "version": "V9.2 Dual-Tier Pure Mathematical Code Engine (Vocal + Real-Detail 5-Track Band)",
        "zeroLzma": True,
        "zeroBase122": True,
        "zeroBinarySamples": True,
        "tracksCount": 10,
        "bandModelAndScoreBytes": tiers_metrics["expressive"]["bandModelAndScoreBytes"],
        "bandModelAndScoreGzipBytes": tiers_metrics["expressive"]["bandModelAndScoreGzipBytes"],
        "dspEngineBytes": dsp_raw,
        "dspEngineGzipBytes": dsp_gz,
        "tiers": tiers_metrics,
        "dualCombinedModelAndScoreBytes": dual_raw,
        "dualCombinedModelAndScoreGzipBytes": dual_gz,
        "files": per_file,
    }


def make_tiers_registry_js(included_tiers: list[str], default_tier: str) -> str:
    entries = []
    if "expressive" in included_tiers:
        entries.append(
            "  expressive: { profile: VOCAL_PROFILE_EXPRESSIVE, score: VOCAL_SCORE_EXPRESSIVE, bandProfile: BAND_PROFILE_EXPRESSIVE, bandScore: BAND_SCORE_EXPRESSIVE }"
        )
    if "ultra" in included_tiers:
        entries.append(
            "  ultra: { profile: VOCAL_PROFILE_ULTRA, score: VOCAL_SCORE_ULTRA, bandProfile: BAND_PROFILE_ULTRA, bandScore: BAND_SCORE_ULTRA }"
        )
    body = ",\n".join(entries)
    return (
        f"// --- data/vocal-tiers.js (standalone: {','.join(included_tiers)}) ---\n"
        f"var VOCAL_TIERS, DEFAULT_VOCAL_TIER;\n{{\n"
        f"  VOCAL_TIERS = {{\n{body}\n  }};\n"
        f"  DEFAULT_VOCAL_TIER = '{default_tier}';\n}}"
    )


def build_standalone_edition(
    out_name: str,
    title_label: str,
    included_tiers: list[str],
    default_tier: str,
    metrics: dict,
) -> tuple[Path, int, int]:
    css_bundle = "\n\n".join((ROOT / c).read_text("utf-8") for c in CSS_FILES)

    tier_voc_rels = [TIER_VOCAL_SCORE_FILES[t] for t in included_tiers]
    tier_band_rels = [TIER_BAND_SCORE_FILES[t] for t in included_tiers]
    core_before_registry = SHARED_VOCAL_MODELS + SHARED_BAND_MODELS + tier_voc_rels + tier_band_rels

    parts = [bundle_module_scoped((ROOT / m).read_text("utf-8"), m) for m in core_before_registry]
    parts.append(make_tiers_registry_js(included_tiers, default_tier))
    for m in DSP_ENGINE_MODULES:
        parts.append(bundle_module_scoped((ROOT / m).read_text("utf-8"), m))

    shared_core_js = "\n\n".join(parts)
    worker_tail_js = bundle_module_scoped((ROOT / "src/dsp/math-worker.js").read_text("utf-8"), "src/dsp/math-worker.js")
    ui_only_js = "\n\n".join(bundle_module_scoped((ROOT / m).read_text("utf-8"), m) for m in UI_MODULES)

    worker_tail_json = json.dumps(worker_tail_js, ensure_ascii=False)
    metrics_json_str = json.dumps(metrics, ensure_ascii=False)

    out_path = ROOT / out_name
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_label}</title>
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
    return out_path, len(raw_b), gzip_size(raw_b)


def main():
    metrics = compute_metrics()

    editions = [
        (
            "ultra",
            "下个路口见_纯数学代码_极简档全曲版.html",
            "下个，路口，见 · V9.2 纯数学代码 10 轨极简高保真档 (10维人声 + 紧凑真伴奏)",
            ["ultra"],
            "ultra",
        ),
        (
            "expressive",
            "下个路口见_纯数学代码_精修档全曲版.html",
            "下个，路口，见 · V9.2 纯数学代码 10 轨母带精修档 (24维人声 + 8声部真伴奏)",
            ["expressive"],
            "expressive",
        ),
        (
            "dual",
            "下个路口见_纯数学代码_双档完整版.html",
            "下个，路口，见 · V9.2 双档纯数学代码 10 轨对比工作台 (极简档 + 精修档)",
            ["expressive", "ultra"],
            "expressive",
        ),
    ]

    metrics["standaloneHtmlEditions"] = {}
    for key, fname, title, tlist, def_t in editions:
        _, raw_len, gz_len = build_standalone_edition(fname, title, tlist, def_t, metrics)
        metrics["standaloneHtmlEditions"][key] = {
            "file": fname,
            "rawBytes": raw_len,
            "gzipBytes": gz_len,
        }

    metrics_path = ROOT / "data" / "code-metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    u = metrics["tiers"]["ultra"]
    e = metrics["tiers"]["expressive"]
    print("=== V9.2 Dual-Tier Pure Mathematical Code Byte Report (Music Representation Only) ===")
    print(f"  [Ultra]      Vocal (5-Track)       : {u['vocalModelAndScoreBytes']:>9,} B ({u['vocalModelAndScoreBytes']/1024:6.2f} KB) | Gzip: {u['vocalModelAndScoreGzipBytes']:>7,} B ({u['vocalModelAndScoreGzipBytes']/1024:6.2f} KB)")
    print(f"  [Ultra]      Band  (5-Track)       : {u['bandModelAndScoreBytes']:>9,} B ({u['bandModelAndScoreBytes']/1024:6.2f} KB) | Gzip: {u['bandModelAndScoreGzipBytes']:>7,} B ({u['bandModelAndScoreGzipBytes']/1024:6.2f} KB)")
    print(f"  [Ultra]      Full Song (10-Track)  : {u['totalModelAndScoreBytes']:>9,} B ({u['totalModelAndScoreBytes']/1024:6.2f} KB) | Gzip: {u['totalModelAndScoreGzipBytes']:>7,} B ({u['totalModelAndScoreGzipBytes']/1024:6.2f} KB)")
    print(f"  [Expressive] Vocal (5-Track)       : {e['vocalModelAndScoreBytes']:>9,} B ({e['vocalModelAndScoreBytes']/1024:6.2f} KB) | Gzip: {e['vocalModelAndScoreGzipBytes']:>7,} B ({e['vocalModelAndScoreGzipBytes']/1024:6.2f} KB)")
    print(f"  [Expressive] Band  (5-Track)       : {e['bandModelAndScoreBytes']:>9,} B ({e['bandModelAndScoreBytes']/1024:6.2f} KB) | Gzip: {e['bandModelAndScoreGzipBytes']:>7,} B ({e['bandModelAndScoreGzipBytes']/1024:6.2f} KB)")
    print(f"  [Expressive] Full Song (10-Track)  : {e['totalModelAndScoreBytes']:>9,} B ({e['totalModelAndScoreBytes']/1024:6.2f} KB) | Gzip: {e['totalModelAndScoreGzipBytes']:>7,} B ({e['totalModelAndScoreGzipBytes']/1024:6.2f} KB)")
    print(f"  [DSP Synth]  10-Track Equations    : {metrics['dspEngineBytes']:>9,} B ({metrics['dspEngineBytes']/1024:6.2f} KB) | Gzip: {metrics['dspEngineGzipBytes']:>7,} B ({metrics['dspEngineGzipBytes']/1024:6.2f} KB)")


if __name__ == "__main__":
    main()
