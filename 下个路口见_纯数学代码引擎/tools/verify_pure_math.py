#!/usr/bin/env python3
"""Strict Pure-Mathematics Auditor for Unified 7-Tier Studio (V9.4.1 .. V9.4.7)."""
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
]

def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[FAIL] {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"[PASS] {msg}")

def main() -> None:
    all_files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    bad = [str(p.relative_to(ROOT)) for p in all_files if p.suffix.lower() in FORBIDDEN_EXTS]
    assert_true(len(bad) == 0, f"Zero binary/audio/compressed/pyc files across unified studio ({len(all_files)} plain-text files)")
    assert_true("<style" not in (ROOT / "index.html").read_text("utf-8").lower(), "Zero inline <style> in index.html")

    for p in all_files:
        if p.suffix in {".js", ".css", ".html"}:
            txt = p.read_text("utf-8")
            for pat, desc in FORBIDDEN_CODE_PATTERNS:
                assert_true(not re.search(pat, txt), f"Clean {p.name}: no {desc}")

    js = """
import { TIERS_CATALOG } from './data/tiers-catalog.js';
import { createTierEngineRegistry } from './src/modules/tier-loader.js';
const reg = createTierEngineRegistry();
const out = [];
for (const m of TIERS_CATALOG) {
  const eng = await reg.loadTierEngine(m.tierId);
  const L = new Float32Array(4096), R = new Float32Array(4096);
  eng.synthesizeSampleRange(48000 * 62, 4096, null, L, R);
  let pk = 0;
  for (let i = 0; i < 4096; i++) pk = Math.max(pk, Math.abs(L[i]), Math.abs(R[i]));
  out.push({ tierId: m.tierId, peak: Number(pk.toFixed(4)) });
}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(["node", "--input-type=module", "-e", js], cwd=str(ROOT), capture_output=True, text=True, check=True)
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert_true(len(res) == 7 and all(r["peak"] > 0.05 for r in res), f"All 7 tiers dynamically load & synthesize chorus cleanly: {res}")
    print("ALL UNIFIED 7-TIER PURE-MATH STUDIO AUDITS PASSED!")

if __name__ == "__main__":
    main()
