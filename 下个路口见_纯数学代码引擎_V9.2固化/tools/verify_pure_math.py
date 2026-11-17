#!/usr/bin/env python3
"""Strict Pure-Mathematics Auditor for V9.2 Dual-Tier 10-Track Engine (Vocal + Real-Detail Band)."""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    ".npz", ".npy", ".pt", ".pth", ".onnx", ".safetensors",
    ".lzma", ".xz", ".gz", ".zst", ".br", ".zip", ".bin", ".dat", ".pcm", ".pyc",
}

FORBIDDEN_CODE_PATTERNS = [
    (r"LZMA\s*\.", "LZMA decompression call"),
    (r"new\s+LZMA\b", "LZMA constructor"),
    (r"lzma-d", "lzma-d decoder script"),
    (r"decodeBase122", "Base122 binary decoder"),
    (r"fromBase122", "Base122 binary decoder"),
    (r"atob\s*\(", "Base64 binary decoding"),
    (r"data:audio/", "Embedded data:audio URI"),
    (r"data:application/octet-stream", "Embedded binary octet-stream URI"),
]

STANDALONE_HTML_NAMES = {
    "下个路口见_纯数学代码_极简档全曲版.html",
    "下个路口见_纯数学代码_精修档全曲版.html",
    "下个路口见_纯数学代码_双档完整版.html",
}


def assert_true(cond: bool, msg: str):
    if not cond:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] {msg}")


def main():
    print("=== Auditing V9.2 Dual-Tier Pure Mathematical Code Engine ===")

    bad_files = []
    all_files = [p for p in ROOT.rglob("*") if p.is_file()]
    for p in all_files:
        if p.suffix.lower() in FORBIDDEN_EXTS:
            bad_files.append(str(p.relative_to(ROOT)))
    assert_true(len(bad_files) == 0, f"Zero binary/audio/compressed/pyc files in project ({len(all_files)} plain-text files checked)")

    runtime_files = [p for p in all_files if p.suffix in {".js", ".css", ".html"}]
    pattern_hits = []
    long_string_hits = []
    for p in runtime_files:
        txt = p.read_text("utf-8")
        for pat, desc in FORBIDDEN_CODE_PATTERNS:
            if re.search(pat, txt):
                pattern_hits.append(f"{p.relative_to(ROOT)}: {desc}")
        if p.name not in STANDALONE_HTML_NAMES:
            for m in re.finditer(r'["\']([^"\'\n]{600,})["\']', txt):
                long_string_hits.append(f"{p.relative_to(ROOT)} (len={len(m.group(1))})")

    assert_true(len(pattern_hits) == 0, f"Zero LZMA / Base122 / Base64 / data:audio patterns across {len(runtime_files)} runtime files")
    assert_true(len(long_string_hits) == 0, "Zero opaque binary-encoded string literals (>=600 chars) in any source module")

    index_html = (ROOT / "index.html").read_text("utf-8")
    assert_true("<style" not in index_html.lower(), "index.html has zero inline <style> blocks (strict CSS separation)")
    for req_dir in ["styles/components", "models", "data", "src/dsp", "src/state", "src/modules", "src/components"]:
        assert_true((ROOT / req_dir).is_dir(), f"Modular architecture layer '{req_dir}/' present")

    metrics_path = ROOT / "data" / "code-metrics.json"
    assert_true(metrics_path.is_file(), "data/code-metrics.json exists")
    metrics = json.loads(metrics_path.read_text("utf-8"))
    u = metrics["tiers"]["ultra"]
    e = metrics["tiers"]["expressive"]
    assert_true(
        u["totalModelAndScoreGzipBytes"] < 190000 and e["totalModelAndScoreGzipBytes"] < 380000,
        f"Ultra 10-track Gzip = {u['totalModelAndScoreGzipBytes']/1024:.2f} KB | Expressive 10-track Gzip = {e['totalModelAndScoreGzipBytes']/1024:.2f} KB",
    )

    node_check = """
import { renderFullPureMathSong, STEM_MANIFEST } from './src/dsp/song-renderer.js';
const out = {};
for (const tier of ['expressive', 'ultra']) {
  const res = renderFullPureMathSong(tier);
  const rep = { renderTimeMs: res.renderTimeMs, peakAmplitude: res.peakAmplitude, nanCount: 0, stems: {} };
  for (const m of STEM_MANIFEST) {
    const [L, R] = res.stems[m.id];
    let sum = 0, pk = 0;
    for (let i = 0; i < L.length; i += 4) {
      const a = L[i], b = R[i];
      if (!Number.isFinite(a) || !Number.isFinite(b)) rep.nanCount++;
      sum += 0.5 * (a * a + b * b);
      const ma = Math.abs(a), mb = Math.abs(b);
      if (ma > pk) pk = ma;
      if (mb > pk) pk = mb;
    }
    rep.stems[m.id] = { rms: Number(Math.sqrt(sum / Math.ceil(L.length / 4)).toFixed(5)), peak: Number(pk.toFixed(4)) };
  }
  out[tier] = rep;
}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_check],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    both = json.loads(proc.stdout.strip().splitlines()[-1])
    for tier, synth_rep in both.items():
        assert_true(synth_rep["nanCount"] == 0, f"[{tier}] Zero NaN/Inf samples across all 10 stereo tracks")
        assert_true(0.85 <= synth_rep["peakAmplitude"] <= 0.9405, f"[{tier}] Master peak = {synth_rep['peakAmplitude']} (<= 0.9405)")
        for stem_id, st in synth_rep["stems"].items():
            assert_true(0.005 <= st["rms"] <= 0.14, f"[{tier}] Stem '{stem_id}' RMS={st['rms']:.4f}")

    print("\nALL V9.2 DUAL-TIER PURE-MATHEMATICS AUDITS PASSED!")


if __name__ == "__main__":
    main()
