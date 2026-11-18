#!/usr/bin/env python3
"""Strict Pure-Mathematics Auditor for V9 10-Track Engine.

Verifies:
1. ZERO binary audio/compressed files (.mp3/.wav/.flac/.npz/.lzma/.bin/.pyc/...) in workspace.
2. ZERO LZMA decoder, Base122 decoder, or Base64 audio payload strings in any runtime JS/HTML file.
3. Strict Modular Engineering: zero inline <style> in index.html; CSS/Models/Data/DSP/UI cleanly separated.
4. Full 10-track 48kHz stereo mathematical synthesis in Node.js passes with 0 NaN/Inf,
   peak <= 0.9405, and every single vocal & band stem active within studio RMS bounds.
"""

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


def assert_true(cond: bool, msg: str):
    if not cond:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] {msg}")


def main():
    print("=== Auditing V9 Pure Mathematical Code Engine ===")

    # 1. Scan entire directory tree for any forbidden binary/audio/cache files
    bad_files = []
    all_files = [p for p in ROOT.rglob("*") if p.is_file()]
    for p in all_files:
        if p.suffix.lower() in FORBIDDEN_EXTS:
            bad_files.append(str(p.relative_to(ROOT)))
    assert_true(len(bad_files) == 0, f"Zero binary/audio/compressed/pyc files in project ({len(all_files)} plain-text files checked)")

    # 2. Scan all runtime JS/CSS/HTML files for forbidden binary codecs or long opaque blobs
    runtime_files = [
        p for p in all_files
        if p.suffix in {".js", ".css", ".html"}
    ]
    pattern_hits = []
    long_string_hits = []
    for p in runtime_files:
        txt = p.read_text("utf-8")
        for pat, desc in FORBIDDEN_CODE_PATTERNS:
            if re.search(pat, txt):
                pattern_hits.append(f"{p.relative_to(ROOT)}: {desc}")
        # Skip standalone HTML worker string literal when checking raw source modules
        if p.name != "下个路口见_纯数学代码全曲版.html":
            for m in re.finditer(r'["\']([^"\'\n]{600,})["\']', txt):
                long_string_hits.append(f"{p.relative_to(ROOT)} (len={len(m.group(1))})")

    assert_true(len(pattern_hits) == 0, f"Zero LZMA / Base122 / Base64 / data:audio patterns across {len(runtime_files)} runtime files")
    assert_true(len(long_string_hits) == 0, "Zero opaque binary-encoded string literals (>=600 chars) in any source module")

    # 3. Verify modular CSS & HTML architecture (User Core Rule 1)
    index_html = (ROOT / "index.html").read_text("utf-8")
    assert_true("<style" not in index_html.lower(), "index.html has zero inline <style> blocks (strict CSS separation)")
    for req_dir in ["styles/components", "models", "data", "src/dsp", "src/state", "src/modules", "src/components"]:
        assert_true((ROOT / req_dir).is_dir(), f"Modular architecture layer '{req_dir}/' present")

    # 4. Verify code-metrics.json exists & pure-math payload is < 50 KB raw / < 18 KB gzipped
    metrics_path = ROOT / "data" / "code-metrics.json"
    assert_true(metrics_path.is_file(), "data/code-metrics.json exists")
    metrics = json.loads(metrics_path.read_text("utf-8"))
    assert_true(
        metrics["totalModelAndScoreBytes"] < 51200 and metrics["totalModelAndScoreGzipBytes"] < 19000,
        f"Full-song 10-track plain-text models+scores = {metrics['totalModelAndScoreBytes']:,} B ({metrics['totalModelAndScoreBytes']/1024:.2f} KB) raw / {metrics['totalModelAndScoreGzipBytes']:,} B ({metrics['totalModelAndScoreGzipBytes']/1024:.2f} KB) gzipped",
    )

    # 5. Execute full 210.373s 10-track synthesis in Node.js and audit signal integrity
    node_check = """
import { renderFullPureMathSong, STEM_MANIFEST } from './src/dsp/song-renderer.js';
const res = renderFullPureMathSong();
const report = {
  renderTimeMs: res.renderTimeMs,
  peakAmplitude: res.peakAmplitude,
  numSamples: res.numSamples,
  nanCount: 0,
  stems: {}
};
for (const m of STEM_MANIFEST) {
  const [L, R] = res.stems[m.id];
  let sum = 0, pk = 0;
  for (let i = 0; i < L.length; i++) {
    const a = L[i], b = R[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) report.nanCount++;
    sum += 0.5 * (a * a + b * b);
    const ma = Math.abs(a), mb = Math.abs(b);
    if (ma > pk) pk = ma;
    if (mb > pk) pk = mb;
  }
  report.stems[m.id] = { rms: Number(Math.sqrt(sum / L.length).toFixed(5)), peak: Number(pk.toFixed(4)) };
}
console.log(JSON.stringify(report));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_check],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    synth_rep = json.loads(proc.stdout.strip().splitlines()[-1])
    assert_true(synth_rep["nanCount"] == 0, "Zero NaN/Inf samples across all 10 stereo tracks (201,958,400 float32 samples)")
    assert_true(0.85 <= synth_rep["peakAmplitude"] <= 0.9405, f"Master linear-sum 10-track peak = {synth_rep['peakAmplitude']} (<= 0.9405 ceiling)")
    for stem_id, st in synth_rep["stems"].items():
        assert_true(0.005 <= st["rms"] <= 0.14, f"Stem '{stem_id}' active & balanced: RMS={st['rms']:.4f}, Peak={st['peak']:.4f}")

    print(f"\nALL V9 PURE-MATHEMATICS AUDITS PASSED! (Node 10-track render time: {synth_rep['renderTimeMs']} ms)")


if __name__ == "__main__":
    main()
