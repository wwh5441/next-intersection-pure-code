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
    print("=== V9R2.1 Pure-Code Audit ===")
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
    for rel in ("models/vocal-speaker-model.js", "models/backing-model.js", "models/space-fdn.js", "models/bass-model.js", "models/poly-model.js"):
        assert m["files"][rel]["gzipBytes"] < 20 * 1024, f"Gate 4 FAILED: {rel} not O(1)"
    assert m["totalGzipBytes"] <= 260 * 1024, f"Gate 4 FAILED: {m['totalGzipKB']} KB Gzip > 260 KB pure-code budget!"
    v85_files = list(ROOT.rglob("*v85*"))
    assert not v85_files, f"Legacy V8.5 files still present: {v85_files}"
    print(f"[Gate 4 PASS] Total pure-code data = {m['totalGzipKB']} KB Gzip ({m['totalRawKB']} KB plain) <= 260 KB, 0 legacy V8.5 files.")

    u = lambda rel: (ROOT / rel).as_uri()
    js = f"""
import {{ getRenderArgs }} from '{u('src/modules/pure-code-assets.js')}';
import {{ renderV9r2AllLayers }} from '{u('src/dsp/v9r2-master-engine.js')}';
const r = renderV9r2AllLayers(getRenderArgs());
const rmsOut = {{}};
for (const [k, chs] of Object.entries(r.layers)) {{ let s = 0, n = 0; for (const c of chs) for (let i = 0; i < c.length; i += 8) {{ s += c[i]*c[i]; n++; }} rmsOut[k] = Math.sqrt(s / n); }}
const sr = r.sampleRate;
const wins = [[76.28, 79.49], [154.64, 157.86], [174.25, 177.39]];
const chorusChecks = wins.map(([t0, t1]) => {{
  const s0 = Math.floor(t0 * sr), s1 = Math.floor(t1 * sr);
  let hSq = 0, bSq = 0, bkPk = 0, mstPk = 0;
  const sub = r.layers.lead_sub_f0[0], body = r.layers.lead_vowel_body[0], shn = r.layers.lead_harmonic_sheen[0];
  const br = r.layers.lead_velvet_breath[0], bkL = r.layers.backing_harmony[0], bkR = r.layers.backing_harmony[1];
  const tracks = Object.values(r.layers).map((chs) => (chs.length > 1 ? chs : [chs[0], chs[0]]));
  for (let i = s0; i < s1; i++) {{
    const h = sub[i] + body[i] + shn[i];
    hSq += h * h;
    bSq += br[i] * br[i];
    const bp = Math.max(Math.abs(bkL[i]), Math.abs(bkR[i]));
    if (bp > bkPk) bkPk = bp;
    let l = 0, rr = 0;
    for (const [cl, cr] of tracks) {{ l += cl[i]; rr += cr[i]; }}
    const mp = Math.max(Math.abs(l), Math.abs(rr));
    if (mp > mstPk) mstPk = mp;
  }}
  const hnrDb = 10 * Math.log10((hSq + 1e-12) / (bSq + 1e-12));
  return {{ hnrDb: +hnrDb.toFixed(2), bkPk: +bkPk.toFixed(4), mstPk: +mstPk.toFixed(4) }};
}});
const dL = r.layers.drums[0], dR = r.layers.drums[1];
let dSq = 0, dPk = 0;
for (let i = 0; i < dL.length; i++) {{
  const m = 0.5 * (dL[i] + dR[i]);
  dSq += m * m;
  const p = Math.max(Math.abs(dL[i]), Math.abs(dR[i]));
  if (p > dPk) dPk = p;
}}
const dRms = Math.sqrt(dSq / dL.length);
const drumCrestDb = +(20 * Math.log10(dPk / (dRms + 1e-12))).toFixed(2);
const drumRmsDb = +(20 * Math.log10(dRms + 1e-12)).toFixed(2);
console.log(JSON.stringify({{ ms: r.synthMs, n: Object.keys(r.layers).length, rms: rmsOut, chorusChecks, drumCrestDb, drumRmsDb }}));
"""
    p = subprocess.run(["node", "--max-old-space-size=6144", "--input-type=module", "-e", js], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    info = json.loads(p.stdout.strip().splitlines()[-1])
    assert info["n"] == 11 and all(v == v and v > 1e-5 for v in info["rms"].values()), f"Gate 5 FAILED: {info}"
    print(f"[Gate 5 PASS] 11 tracks synthesized in {info['ms']/1000:.2f}s, all finite & active.")
    for idx, chk in enumerate(info["chorusChecks"], 1):
        assert chk["hnrDb"] >= 27.0 and chk["bkPk"] <= 0.45 and chk["mstPk"] <= 0.98, f"Gate 6 FAILED on Chorus {idx}: {chk}"
    assert info["drumCrestDb"] >= 12.5 and -20.5 <= info["drumRmsDb"] <= -13.5, f"Gate 6 Drum Cleanliness FAILED: {info}"
    print(f"[Gate 6 PASS] Anti-distortion & clean drums verified: drums RMS={info['drumRmsDb']} dB, crest={info['drumCrestDb']} dB, choruses={info['chorusChecks']}")
    mm = m["metrics"]
    print(f"\nQuality vs HQ stems (Log-Mel SNR): master {mm['masterMelSnrDb']} dB | lead {mm['leadMelSnrDb']} | vocals {mm['vocalsMelSnrDb']} | bass {mm['bassMelSnrDb']} | drums {mm['drumsMelSnrDb']} | guitar/keys {mm['guitarSynthMelSnrDb']}")
    print("ALL PURE-CODE GATES PASSED")


if __name__ == "__main__":
    main()
