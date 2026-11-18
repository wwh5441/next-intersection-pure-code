#!/usr/bin/env python3
"""V9R2.1 guitar+keys experiment: CQT harmonic NMF -> note runs -> JS synth -> metrics."""
import json, subprocess, sys, tempfile, time, gzip
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import exp_vocal as ev
from exp_band import write_js


def render_poly(gains, n, model="models/poly-model.js", score="data/poly-score.js"):
    u = lambda r: (ROOT / r).as_uri()
    tmp = Path(tempfile.mkdtemp())
    js = f"""
import fs from 'node:fs';
import {{ POLY_MODEL }} from '{u(model)}';
import {{ POLY_SCORE }} from '{u(score)}';
import {{ synthesizePolyNotes }} from '{u('src/dsp/poly-note-synth.js')}';
const t0=performance.now();
const [yl, yr] = synthesizePolyNotes(POLY_MODEL, POLY_SCORE, {n}, 48000, {json.dumps(gains)});
const y = new Float32Array({n}); for (let i=0;i<{n};i++) y[i]=0.5*(yl[i]+yr[i]);
fs.writeFileSync('{tmp}/y.f32', Buffer.from(y.buffer));
console.log(Math.round(performance.now()-t0));
"""
    p = subprocess.run(["node", "--max-old-space-size=6144", "--input-type=module", "-e", js], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return np.fromfile(tmp / "y.f32", np.float32), int(p.stdout.strip())


if __name__ == "__main__":
    from services.poly_note_score import analyze_poly, pack_poly_score
    st = ev.load_prep()
    ref = st["guitar_synth_stereo"]
    x = ref.mean(1).astype(np.float32)
    t0 = time.time()
    res = analyze_poly(x, iters=int(sys.argv[1]) if len(sys.argv) > 1 else 120)
    print("nmf", round(time.time() - t0, 1), "s tuning", round(res["tuning"], 3))
    pk = pack_poly_score(res, floor_db=float(sys.argv[2]) if len(sys.argv) > 2 else -42.0)
    write_js("models/poly-model.js", "POLY_MODEL", pk["model"])
    write_js("data/poly-score.js", "POLY_SCORE", pk["score"])
    print(pk["stats"])
    yt, ms = render_poly({"tonal": 1, "noise": 0}, len(x))
    yn, ms2 = render_poly({"tonal": 0, "noise": 1}, len(x))
    print("render ms", ms, ms2)
    rt = np.sqrt((yt ** 2).mean()); rn = np.sqrt((yn ** 2).mean()); rx = np.sqrt((x ** 2).mean())
    best = (-99, 0, 0)
    for gt in (0.6, 0.8, 1.0, 1.2):
        for gn in (0.0, 0.2, 0.4, 0.6):
            y = (gt * rx / max(rt, 1e-9)) * yt + (gn * rx / max(rn, 1e-9)) * yn
            s = ev.mel_snr(x, y)
            if s > best[0]:
                best = (s, gt * rx / max(rt, 1e-9), gn * rx / max(rn, 1e-9))
    print("guitar+keys Mel SNR (mid)", best[0], "gains", best[1], best[2])
    (ROOT / "data/poly-gains.json").write_text(json.dumps({"tonal": float(best[1]), "noise": float(best[2])}))
    for fn in ("models/poly-model.js", "data/poly-score.js"):
        b = (ROOT / fn).read_bytes(); print(fn, len(b) // 1024, "KB raw", len(gzip.compress(b)) // 1024, "KB gz")
