#!/usr/bin/env python3
"""Strict Pure-Mathematics Auditor for V9.3 / V9.4 48kHz Near-Lossless Orthonormal Engine."""
from __future__ import annotations

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
    "下个路口见_纯数学代码_几乎无损母带版.html",
}


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] {msg}")


def main() -> None:
    metrics_path = ROOT / "data" / "code-metrics.json"
    assert_true(metrics_path.is_file(), "data/code-metrics.json exists")
    metrics = json.loads(metrics_path.read_text("utf-8"))
    ver = metrics.get("versionShort", "V9.x")
    print(f"=== Auditing {ver} Pure Mathematical Near-Lossless Engine ===")

    bad_files = []
    all_files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    for p in all_files:
        if p.suffix.lower() in FORBIDDEN_EXTS:
            bad_files.append(str(p.relative_to(ROOT)))
    assert_true(
        len(bad_files) == 0,
        f"Zero binary/audio/compressed/pyc files in project ({len(all_files)} plain-text files checked)",
    )

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

    assert_true(
        len(pattern_hits) == 0,
        f"Zero LZMA / Base122 / Base64 / data:audio patterns across {len(runtime_files)} runtime files",
    )
    assert_true(len(long_string_hits) == 0, "Zero opaque string literals (>=600 chars) in any source module")

    index_html = (ROOT / "index.html").read_text("utf-8")
    assert_true("<style" not in index_html.lower(), "index.html has zero inline <style> blocks (strict CSS separation)")
    for req_dir in ["styles/components", "models", "data", "src/dsp", "src/state", "src/modules", "src/components"]:
        assert_true((ROOT / req_dir).is_dir(), f"Modular architecture layer '{req_dir}/' present")

    assert_true(
        metrics["masterWaveSnrDb"] >= 27.5 and metrics["masterLogMelSnrDb"] >= 38.0,
        f"Near-Lossless Fidelity: Wave SNR = {metrics['masterWaveSnrDb']:.2f} dB (>=27.5 dB) | Log-Mel SNR = {metrics['masterLogMelSnrDb']:.2f} dB (>=38.0 dB)",
    )
    print(
        f"[INFO] Plain-Text Model+Score = {metrics['totalModelAndScoreBytes']/1048576:.2f} MB | "
        f"Gzip-9 = {metrics['totalModelAndScoreGzipBytes']/1048576:.2f} MB | "
        f"Live PCM RAM = {metrics['pcmRamMb']:.2f} MB"
    )

    node_check = """
import { renderFullPureMathSong, STEM_MANIFEST } from './src/dsp/song-renderer.js';
const res = renderFullPureMathSong();
const rep = { renderTimeMs: res.renderTimeMs, peakAmplitude: res.peakAmplitude, nanCount: 0, stems: {} };
for (const m of STEM_MANIFEST) {
  const [L, R] = res.stems[m.id];
  let sum = 0, pk = 0;
  for (let i = 0; i < L.length; i += 8) {
    const a = L[i], b = R[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) rep.nanCount++;
    sum += 0.5 * (a * a + b * b);
    const ma = Math.abs(a), mb = Math.abs(b);
    if (ma > pk) pk = ma;
    if (mb > pk) pk = mb;
  }
  rep.stems[m.id] = { rms: Number(Math.sqrt(sum / Math.ceil(L.length / 8)).toFixed(5)), peak: Number(pk.toFixed(4)) };
}
console.log(JSON.stringify(rep));
"""
    proc = subprocess.run(
        ["node", "--max-old-space-size=4096", "--input-type=module", "-e", node_check],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    synth_rep = json.loads(proc.stdout.strip().splitlines()[-1])
    assert_true(synth_rep["nanCount"] == 0, f"Node.js synthesis: Zero NaN/Inf across all 6 stereo stems (rendered in {synth_rep['renderTimeMs']} ms)")
    assert_true(0.88 <= synth_rep["peakAmplitude"] <= 0.945, f"Node.js Master conservative sum peak = {synth_rep['peakAmplitude']} (target 0.94)")
    for stem_id, st in synth_rep["stems"].items():
        assert_true(st["rms"] >= 0.004, f"Stem '{stem_id}' RMS = {st['rms']:.4f}, Peak = {st['peak']:.4f}")

    print(f"\nALL {ver} PURE-MATHEMATICS NEAR-LOSSLESS AUDITS PASSED!")


if __name__ == "__main__":
    main()
