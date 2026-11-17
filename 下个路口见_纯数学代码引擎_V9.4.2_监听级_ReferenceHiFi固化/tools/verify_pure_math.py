#!/usr/bin/env python3
"""Strict Pure-Mathematics Auditor for V9.4.x Standalone Frozen Edition."""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus",
    ".npz", ".npy", ".pt", ".pth", ".onnx", ".safetensors",
    ".lzma", ".xz", ".gz", ".zst", ".br", ".zip", ".bin", ".dat", ".pcm", ".pyc",
}
FORBIDDEN_CODE_PATTERNS = [
    (r"LZMA\s*\.", "LZMA call"),
    (r"new\s+LZMA\b", "LZMA constructor"),
    (r"decodeBase122", "Base122 decoder"),
    (r"atob\s*\(", "Base64 decoder"),
    (r"data:audio/", "Embedded audio URI"),
    (r"data:application/octet-stream", "Embedded octet-stream URI"),
]

def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] {msg}")

def main() -> None:
    meta = json.loads((ROOT / "data/code-metrics.json").read_text("utf-8"))
    print(f"=== Auditing {meta['versionCode']} {meta['tierName']} Pure-Math Edition ===")
    all_files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    bad = [str(p.relative_to(ROOT)) for p in all_files if p.suffix.lower() in FORBIDDEN_EXTS]
    assert_true(len(bad) == 0, f"Zero binary/audio/compressed/pyc files ({len(all_files)} plain-text files)")

    pat_hits, long_hits = [], []
    for p in all_files:
        if p.suffix not in {".js", ".css", ".html"}:
            continue
        txt = p.read_text("utf-8")
        for pat, desc in FORBIDDEN_CODE_PATTERNS:
            if re.search(pat, txt):
                pat_hits.append(f"{p.name}: {desc}")
        if p.suffix != ".html":
            for m in re.finditer(r'["\']([^"\'\n]{600,})["\']', txt):
                long_hits.append(f"{p.name} ({len(m.group(1))})")
    assert_true(len(pat_hits) == 0, "Zero LZMA / Base122 / Base64 / data:audio patterns")
    assert_true(len(long_hits) == 0, "Zero opaque string literals (>=600 chars) in source modules")
    assert_true("<style" not in (ROOT / "index.html").read_text("utf-8").lower(), "Zero inline <style> in index.html")

    js = """
import { TIER_PHYSICS } from './models/tier-physics.js';
import { HUFFMAN_TABLES } from './models/huffman-algebra.js';
import { TIER_FRAMES } from './data/tier-series-index.js';
import { createStreamingTierEngine } from './src/dsp/streaming-tier-engine.js';
const eng = createStreamingTierEngine(TIER_PHYSICS, HUFFMAN_TABLES, TIER_FRAMES);
const [L, R] = eng.renderFullSongStereo();
let pk = 0, nans = 0;
for (let i = 0; i < L.length; i += 4) {
  if (!Number.isFinite(L[i]) || !Number.isFinite(R[i])) nans++;
  pk = Math.max(pk, Math.abs(L[i]), Math.abs(R[i]));
}
console.log(JSON.stringify({ nans, peak: Number(pk.toFixed(4)) }));
"""
    proc = subprocess.run(["node", "--input-type=module", "-e", js], cwd=str(ROOT), capture_output=True, text=True, check=True)
    rep = json.loads(proc.stdout.strip().splitlines()[-1])
    assert_true(rep["nans"] == 0 and 0.70 <= rep["peak"] <= 1.05, f"Node.js full-song synthesis clean (peak={rep['peak']})")
    print(f"ALL {meta['versionCode']} AUDITS PASSED!")

if __name__ == "__main__":
    main()
