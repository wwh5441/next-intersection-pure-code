#!/usr/bin/env python3
"""V9R2.1 band experiment harness: bass (more instruments added later)."""
from __future__ import annotations
import json, subprocess, sys, tempfile, time, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import exp_vocal as ev


def write_js(rel, name, obj):
    (ROOT / rel).write_text(f"export const {name} = {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))};\n", encoding="utf-8")


def render_bass(opts: dict, n: int) -> np.ndarray:
    u = lambda r: (ROOT / r).as_uri()
    tmp = Path(tempfile.mkdtemp())
    js = f"""
import fs from 'node:fs';
import {{ BASS_MODEL }} from '{u('models/bass-model.js')}';
import {{ BASS_SCORE }} from '{u('data/bass-score.js')}';
import {{ synthesizeBassHarmonic }} from '{u('src/dsp/bass-harmonic-synth.js')}';
const t0=performance.now();
const y = synthesizeBassHarmonic(BASS_MODEL, BASS_SCORE, {n}, 48000, {json.dumps(opts)});
fs.writeFileSync('{tmp}/b.f32', Buffer.from(y.buffer));
console.log(Math.round(performance.now()-t0));
"""
    p = subprocess.run(["node", "--max-old-space-size=4096", "--input-type=module", "-e", js], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return np.fromfile(tmp / "b.f32", dtype=np.float32), int(p.stdout.strip())


if __name__ == "__main__":
    from services.bass_harmonic_score import analyze_bass
    st = ev.load_prep()
    ref = st["bass_mono"]
    t0 = time.time()
    res = analyze_bass(ref, n_harm=int(sys.argv[1]) if len(sys.argv) > 1 else 40, n_pca=int(sys.argv[2]) if len(sys.argv) > 2 else 12, step=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    write_js("models/bass-model.js", "BASS_MODEL", res["model"])
    write_js("data/bass-score.js", "BASS_SCORE", res["score"])
    print("analysis", round(time.time() - t0, 1), "s", res["stats"])
    for opts in ({"pluckDb": -60}, {"pluckDb": -14}, {"pluckDb": -8}):
        y, ms = render_bass(opts, len(ref))
        print(opts, "bass Mel SNR", ev.mel_snr(ref, y), "rms ref %.4f syn %.4f" % (np.sqrt((ref**2).mean()), np.sqrt((y**2).mean())), ms, "ms")
    import gzip
    for f in ("models/bass-model.js", "data/bass-score.js"):
        b = (ROOT / f).read_bytes(); print(f, len(b) // 1024, "KB raw", len(gzip.compress(b)) // 1024, "KB gz")
