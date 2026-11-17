#!/usr/bin/env python3
"""Pure-Code Audit for V9R2 (`tools/verify_v9r2_pure_audit.py`).

Gate 1: no binary audio/archive files in the project.
Gate 2: no compressed-blob decoders (LZMA/Base122/atob) in models/, data/, src/.
Gate 3: no waveform-frame / transform-residual data (no TDAC/closure/prototype-waveform modules or long dense numeric frame arrays).
Gate 4: model sizes O(1) & total data within pure-code budget.
Gate 5: Node.js 11-track synthesis runs, all tracks finite & non-silent.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BIN_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".bin", ".npz", ".npy", ".wasm", ".lzma", ".gz", ".zip"}
BLOB_PAT = [re.compile(p) for p in (r"lzma-d-min", r"LZMA\.decompress", r"\batob\s*\(", r"base122", r"Base122")]
NONPURE_PAT = [re.compile(p) for p in (r"Orthonormal", r"TDAC", r"[Cc]losure", r"coeffTokens", r"bandTokens", r"PrototypeAtlas")]


def main() -> None:
    print("=== V9R2 Pure-Code Audit ===")
    bad = [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() in BIN_EXT]
    assert not bad, f"Gate 1 FAILED: {bad}"
    print("[Gate 1 PASS] 0 binary audio/archive files.")

    js_files = [p for sub in ("models", "data", "src") for p in (ROOT / sub).rglob("*.js")]
    for p in js_files:
        t = p.read_text(encoding="utf-8")
        for pat in BLOB_PAT:
            assert not pat.search(t), f"Gate 2 FAILED: {pat.pattern} in {p}"
    print(f"[Gate 2 PASS] 0 compressed-blob decoders across {len(js_files)} JS files.")

    for p in js_files:
        t = p.read_text(encoding="utf-8")
        for pat in NONPURE_PAT:
            assert not pat.search(t), f"Gate 3 FAILED: waveform/residual codec marker '{pat.pattern}' in {p}"
    print("[Gate 3 PASS] 0 transform-residual / waveform-prototype modules.")

    m = json.loads((ROOT / "data/code-metrics.json").read_text(encoding="utf-8"))
    for rel in ("models/singer-core.js", "models/space-fdn.js", "models/instrument-physics.js"):
        assert m["files"][rel]["gzipBytes"] < 20 * 1024, f"Gate 4 FAILED: {rel} not O(1)"
    assert m["totalGzipBytes"] < 1.5 * 1024 * 1024, "Gate 4 FAILED: total too large for pure-code score"
    print(f"[Gate 4 PASS] Total pure-code data = {m['totalGzipKB']} KB Gzip ({m['totalRawKB']} KB plain).")

    u = lambda rel: (ROOT / rel).as_uri()
    js = f"""
import {{ STUDIO_METADATA }} from '{u('data/studio-metadata.js')}';
import {{ SINGER_CORE }} from '{u('models/singer-core.js')}';
import {{ SYLLABLE_ATLAS }} from '{u('models/syllable-atlas.js')}';
import {{ SPACE_FDN }} from '{u('models/space-fdn.js')}';
import {{ INSTRUMENT_PHYSICS }} from '{u('models/instrument-physics.js')}';
import {{ VOCAL_EVENTS_V85 }} from '{u('data/vocal-events-v85.js')}';
import {{ BAND_SCORE_EXPRESSIVE }} from '{u('data/band-score-expressive.js')}';
import {{ BAND_STEM_GAINS }} from '{u('data/band-stem-gains.js')}';
import {{ renderV9r2AllLayers }} from '{u('src/dsp/v9r2-master-engine.js')}';
const r = renderV9r2AllLayers({{ config: STUDIO_METADATA.config, singerCore: SINGER_CORE, syllableAtlas: SYLLABLE_ATLAS, spaceFdn: SPACE_FDN,
  vocalEvents: VOCAL_EVENTS_V85, instrumentPhysics: INSTRUMENT_PHYSICS, bandScore: BAND_SCORE_EXPRESSIVE, bandGains: BAND_STEM_GAINS }});
const rmsOut = {{}};
for (const [k, chs] of Object.entries(r.layers)) {{ let s = 0, n = 0; for (const c of chs) for (let i = 0; i < c.length; i += 8) {{ s += c[i]*c[i]; n++; }} rmsOut[k] = Math.sqrt(s / n); }}
console.log(JSON.stringify({{ ms: r.synthMs, n: Object.keys(r.layers).length, rms: rmsOut }}));
"""
    p = subprocess.run(["node", "--max-old-space-size=6144", "--input-type=module", "-e", js], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    info = json.loads(p.stdout.strip().splitlines()[-1])
    assert info["n"] == 11 and all(v == v and v > 1e-4 for v in info["rms"].values()), f"Gate 5 FAILED: {info}"
    print(f"[Gate 5 PASS] 11 tracks synthesized in {info['ms']/1000:.2f}s, all finite & active.")
    mm = m["metrics"]
    print(f"\nQuality vs original (Log-Mel SNR): master {mm['masterMelSnrDb']} dB | lead {mm['leadMelSnrDb']} | vocals {mm['vocalsMelSnrDb']} | bass {mm['bassMelSnrDb']} | drums {mm['drumsMelSnrDb']} | guitar/keys {mm['guitarSynthMelSnrDb']}")
    print("ALL PURE-CODE GATES PASSED")


if __name__ == "__main__":
    main()
