#!/usr/bin/env python3
"""V9R2.1 drum experiment: analyze -> per-family render -> mel-optimal family gains -> metrics."""
import json, subprocess, sys, tempfile, time, gzip
from pathlib import Path
import numpy as np, librosa
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import exp_vocal as ev
from exp_band import write_js


def render_drums(gains, n, opts=None):
    u = lambda r: (ROOT / r).as_uri()
    tmp = Path(tempfile.mkdtemp())
    js = f"""
import fs from 'node:fs';
import {{ DRUM_MODEL }} from '{u('models/drum-model.js')}';
import {{ DRUM_SCORE }} from '{u('data/drum-score.js')}';
import {{ synthesizeDrumEvents }} from '{u('src/dsp/drum-event-synth.js')}';
const t0=performance.now();
const [L,R] = synthesizeDrumEvents(DRUM_MODEL, DRUM_SCORE, {json.dumps(gains)}, {n}, 48000, {json.dumps(opts or {})});
fs.writeFileSync('{tmp}/L.f32', Buffer.from(L.buffer)); fs.writeFileSync('{tmp}/R.f32', Buffer.from(R.buffer));
console.log(Math.round(performance.now()-t0));
"""
    p = subprocess.run(["node", "--max-old-space-size=6144", "--input-type=module", "-e", js], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return np.column_stack([np.fromfile(tmp / "L.f32", np.float32), np.fromfile(tmp / "R.f32", np.float32)]), int(p.stdout.strip())


if __name__ == "__main__":
    from services.drum_event_score import analyze_drums
    st = ev.load_prep()
    ref = st["drums_stereo"]
    n = len(ref)
    t0 = time.time()
    res = analyze_drums(ref)
    write_js("models/drum-model.js", "DRUM_MODEL", res["model"])
    write_js("data/drum-score.js", "DRUM_SCORE", res["score"])
    print("analysis", round(time.time() - t0, 1), "s", res["stats"])
    fams = list(res["model"]["families"])
    parts = {}
    for f in fams:
        y, ms = render_drums({k: (1.0 if k == f else 0.0) for k in fams}, n)
        parts[f] = y
        print(f, "render", ms, "ms rms", float(np.sqrt((y ** 2).mean())))
    # coordinate search of family gains on mel SNR
    g = {f: float(np.sqrt((ref ** 2).mean()) / max(np.sqrt((parts[f] ** 2).mean()), 1e-9)) / len(fams) for f in fams}
    mix = lambda gg: sum(gg[f] * parts[f] for f in fams)
    best = ev.mel_snr(ref, mix(g))
    for it in range(3):
        for f in fams:
            for mul in (0.5, 0.7, 0.85, 1.18, 1.4, 2.0):
                gg = dict(g); gg[f] *= mul
                s = ev.mel_snr(ref, mix(gg))
                if s > best:
                    best, g = s, gg
        print("iter", it, "drums Mel SNR", best, {k: round(v, 4) for k, v in g.items()})
    (ROOT / "data/drum-gains.json").write_text(json.dumps(g))
    for fn in ("models/drum-model.js", "data/drum-score.js"):
        b = (ROOT / fn).read_bytes(); print(fn, len(b) // 1024, "KB raw", len(gzip.compress(b)) // 1024, "KB gz")
