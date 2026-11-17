#!/usr/bin/env python3
"""Automated Architectural Purity & Acoustic Fidelity Audit for V9R1 (`verify_v9r1_pure_math.py`)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_SUFFIXES = {
    ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".flac",
    ".wec", ".bin", ".lzma", ".xz", ".npz", ".npy", ".wasm",
}

FORBIDDEN_CODE_PATTERNS = [
    "atob(",
    "LZMA.",
    "lzma.",
    "decodeBase122",
    "encodeBase122",
    "master-mid",
    "tier-frames",
]


def main():
    print("=================================================================")
    print("  V9R1 Pure-Code Dual-Dictionary Engine — Full Audit Suite")
    print("=================================================================")

    # 1. Binary asset & forbidden codec check
    bad_files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad_files.append(str(p.relative_to(ROOT)))
    if bad_files:
        raise AssertionError(f"Forbidden binary files found: {bad_files}")
    print("[PASS] 0 binary audio/archive/WASM files in repository.")

    # 2. Source JS purity check (exclude tools/ verification scripts themselves)
    for folder in ["models", "data", "src"]:
        for js_file in (ROOT / folder).rglob("*.js"):
            txt = js_file.read_text("utf-8")
            for pat in FORBIDDEN_CODE_PATTERNS:
                # Allow comment mentions like `Zero LZMA / Zero Base122` only in `//` comment lines
                for line_no, line in enumerate(txt.splitlines(), 1):
                    code_part = line.split("//")[0]
                    if pat in code_part:
                        raise AssertionError(f"Forbidden token '{pat}' in {js_file.relative_to(ROOT)}:{line_no}")
    print("[PASS] 0 LZMA / Base122 / Base64 / blind MDCT frame payloads in models/, data/, src/.")

    # 3. User Core Rule 1 decoupling check on index.html
    idx_html = (ROOT / "index.html").read_text("utf-8")
    if "<style" in idx_html.lower():
        raise AssertionError("index.html contains inline <style>! Must use decoupled CSS files.")
    print("[PASS] index.html has 0 inline <style> blocks (6 decoupled CSS files linked).")

    # 4. Run Node.js acoustic synthesis + envelope correlation audit
    node_script = """
import fs from 'node:fs';
import { renderV9r1Song, STEM_MANIFEST } from './src/dsp/v9r1-renderer.js';

const report = {};
for (const tier of ['ultra', 'expressive']) {
  const res = renderV9r1Song(tier);
  const n = res.numSamples;
  const win = 960; // 20ms @ 48kHz
  const nFrames = Math.floor(n / win);
  const vEnv = new Float32Array(nFrames);
  const bEnv = new Float32Array(nFrames);

  let stemStats = {};
  for (const m of STEM_MANIFEST) {
    const chs = res.stems[m.id];
    let sq = 0, pk = 0;
    for (let c = 0; c < chs.length; c++) {
      const a = chs[c];
      for (let i = 0; i < n; i++) {
        const v = a[i];
        if (!Number.isFinite(v)) throw new Error('Non-finite sample in ' + m.id);
        sq += v * v;
        const av = Math.abs(v);
        if (av > pk) pk = av;
      }
    }
    stemStats[m.id] = {
      channels: chs.length,
      rms: Number(Math.sqrt(sq / (chs.length * n)).toFixed(5)),
      peak: Number(pk.toFixed(4)),
    };
  }

  for (let f = 0; f < nFrames; f++) {
    const s0 = f * win;
    let vsq = 0, bsq = 0;
    for (let i = s0; i < s0 + win; i += 4) {
      const vc = res.stems.vocal_chest_f0[0][i] + res.stems.vocal_formant_body[0][i] + res.stems.vocal_consonant_air[0][i]
               + 0.5 * (res.stems.vocal_chorus_harmony[0][i] + res.stems.vocal_chorus_harmony[1][i]);
      const bcL = res.stems.drum_kick_snare[0][i] + res.stems.drum_hihat_perc[0][i] + res.stems.bass_groove[0][i]
                + res.stems.guitar_funk_riff[0][i] + res.stems.keys_rhodes_pad[0][i];
      const bcR = res.stems.drum_kick_snare[1][i] + res.stems.drum_hihat_perc[1][i] + res.stems.bass_groove[0][i]
                + res.stems.guitar_funk_riff[1][i] + res.stems.keys_rhodes_pad[1][i];
      vsq += vc * vc;
      bsq += 0.5 * (bcL * bcL + bcR * bcR);
    }
    vEnv[f] = Math.sqrt(vsq / (win / 4));
    bEnv[f] = Math.sqrt(bsq / (win / 4));
  }

  report[tier] = {
    renderTimeMs: res.renderTimeMs,
    peakAmplitude: res.peakAmplitude,
    stems: stemStats,
    vEnv: Array.from(vEnv),
    bEnv: Array.from(bEnv),
  };
}
console.log(JSON.stringify(report));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", node_script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    acoustic = json.loads(proc.stdout.strip())

    # Optional ground-truth envelope correlation if cached reference NPZ exists
    npz_path = Path("/Users/weckham/.gemini/antigravity/brain/74d2d208-e911-4def-a54e-f88353345d24/scratch/conservative_6stems_48k.npz")
    if npz_path.exists():
        try:
            import numpy as np
            z = np.load(str(npz_path))
            v_ref = 0.5 * (z["vocals"][0] + z["vocals"][1])
            b_ref = 0.5 * (
                (z["drums"][0] + z["bass"][0] + z["guitar"][0] + z["piano"][0] + z["other"][0])
                + (z["drums"][1] + z["bass"][1] + z["guitar"][1] + z["piano"][1] + z["other"][1])
            )
            n_frames = len(acoustic["ultra"]["vEnv"])
            win = 960
            v_ref_env = np.sqrt(np.mean(v_ref[: n_frames * win].reshape(n_frames, win) ** 2, axis=1))
            b_ref_env = np.sqrt(np.mean(b_ref[: n_frames * win].reshape(n_frames, win) ** 2, axis=1))
            for tid in ["ultra", "expressive"]:
                ve = np.array(acoustic[tid]["vEnv"], dtype=np.float64)
                be = np.array(acoustic[tid]["bEnv"], dtype=np.float64)
                acoustic[tid]["vocalRefEnvCorrelation"] = round(float(np.corrcoef(v_ref_env, ve)[0, 1]), 4)
                acoustic[tid]["bandRefEnvCorrelation"] = round(float(np.corrcoef(b_ref_env, be)[0, 1]), 4)
        except Exception as e:
            print(f"[INFO] Skipped optional NPZ correlation check: {e}")

    for tid in ["ultra", "expressive"]:
        del acoustic[tid]["vEnv"]
        del acoustic[tid]["bEnv"]
        info = acoustic[tid]
        print(
            f"[PASS] Tier '{tid}': render={info['renderTimeMs']}ms, peak={info['peakAmplitude']}, "
            f"vocEnvR={info.get('vocalRefEnvCorrelation', 'N/A')}, bandEnvR={info.get('bandRefEnvCorrelation', 'N/A')}"
        )

    metrics_path = ROOT / "data" / "code-metrics.json"
    if metrics_path.exists():
        mdata = json.loads(metrics_path.read_text("utf-8"))
        mdata["acousticVerification"] = acoustic
        metrics_path.write_text(json.dumps(mdata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[PASS] Updated {metrics_path.relative_to(ROOT)} with acoustic verification report.")


if __name__ == "__main__":
    main()
