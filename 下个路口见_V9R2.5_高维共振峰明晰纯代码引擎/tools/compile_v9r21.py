#!/usr/bin/env python3
"""V9R2.1 end-to-end compiler: MP3 -> HQ stems -> pure-code models/scores -> verified render (`tools/compile_v9r21.py`)."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import exp_vocal as ev
from exp_band import write_js
from exp_drums import render_drums
from exp_poly import render_poly
from services.hq_stem_separator import separate_hq
from parsers.lrc_syllable_parser import load_phoneme_lexicon, parse_lrc_with_tokens
from services.v9r21_vocal_engine import analyze_lead_48k, align_syllable_events, compile_v9r21_vocal_bundle
from services.bass_harmonic_score import analyze_bass
from services.drum_event_score import analyze_drums, calibrate_drum_gains
from services.poly_note_score import analyze_poly_dual

MP3 = ROOT.parent / "mp3_output/李宇春 - 下个, 路口, 见.mp3"
PURE_FILES = [
    "models/vocal-speaker-model.js", "models/vocal-phoneme-atlas.js", "models/backing-model.js", "models/space-fdn.js",
    "models/bass-model.js", "models/drum-model.js", "models/poly-model.js",
    "data/vocal-score.js", "data/backing-score.js", "data/bass-score.js", "data/drum-score.js", "data/poly-score.js", "data/band-gains.js",
]


def write_meta(config, lines, catalog, metrics):
    p = ROOT / "data/studio-metadata.js"
    obj = {"config": config, "lyrics": {"durationSec": 210.373, "lines": lines}, "atlasCatalog": catalog, "metrics": metrics}
    p.write_text("export const STUDIO_METADATA = " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")


def render_full(n: int) -> dict:
    u = lambda r: (ROOT / r).as_uri()
    tmp = Path(tempfile.mkdtemp())
    js = f"""
import fs from 'node:fs';
import {{ getRenderArgs }} from '{u('src/modules/pure-code-assets.js')}';
import {{ renderV9r2AllLayers }} from '{u('src/dsp/v9r2-master-engine.js')}';
const r = renderV9r2AllLayers(getRenderArgs());
for (const [k, chs] of Object.entries(r.layers)) chs.forEach((c,i)=>fs.writeFileSync('{tmp}/'+k+'_'+i+'.f32', Buffer.from(c.buffer,c.byteOffset,c.byteLength)));
console.log(r.synthMs);
"""
    p = subprocess.run(["node", "--max-old-space-size=8192", "--input-type=module", "-e", js], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    rd = lambda k, i=0: np.fromfile(tmp / f"{k}_{i}.f32", np.float32)[:n]
    st2 = lambda k: np.column_stack([rd(k, 0), rd(k, 1)])
    lead = sum(rd(k) for k in ("lead_sub_f0", "lead_vowel_body", "lead_harmonic_sheen", "lead_plosive_burst", "lead_sibilant", "lead_velvet_breath"))
    vocals = np.column_stack([lead, lead]) + st2("backing_harmony") + st2("vocal_space")
    bass = rd("bass"); drums = st2("drums"); gtr = st2("guitar_synth")
    return {"lead": lead, "backing": st2("backing_harmony"), "space": st2("vocal_space"), "vocals": vocals, "bass": bass, "drums": drums, "gtr": gtr,
            "master": vocals + np.column_stack([bass, bass]) + drums + gtr, "ms": int(p.stdout.strip())}


def main():
    t0 = time.time()
    config = json.loads((ROOT / "schemas/v9r2_genesis_config.json").read_text(encoding="utf-8"))
    print("[1/6] HQ separation (cached if present)...")
    separate_hq(MP3, ev.HQ)
    st = ev.load_prep()
    n = len(st["master"])
    print("    conservation", st["conservation_snr_db"], "dB")

    print("[2/6] Vocals: direct 64-ERB vocal tract + 97-syllable atlas + true L/R harmony...")
    lex = load_phoneme_lexicon(ROOT / "schemas/phoneme_lexicon.json")
    feats = analyze_lead_48k(st["lead_mono"])
    all_lines, sung = parse_lrc_with_tokens(ev.LRC, lex, n / 48000)
    events = align_syllable_events(sung, feats)
    voc = compile_v9r21_vocal_bundle(events, feats, st, config)
    write_js("models/vocal-speaker-model.js", "VOCAL_SPEAKER_MODEL", voc["speaker_model"])
    write_js("models/vocal-phoneme-atlas.js", "VOCAL_PHONEME_ATLAS", voc["phoneme_atlas"])
    write_js("models/backing-model.js", "BACKING_MODEL", voc["backing_model"])
    write_js("models/space-fdn.js", "SPACE_FDN", voc["space_fdn"])
    write_js("data/vocal-score.js", "VOCAL_SCORE", voc["vocal_score"])
    write_js("data/backing-score.js", "BACKING_SCORE", voc["backing_score"])

    print("[3/6] Bass discrete musical note model...")
    b = analyze_bass(st["bass_mono"], n_harm=36, n_presets=12)
    write_js("models/bass-model.js", "BASS_MODEL", b["model"]); write_js("data/bass-score.js", "BASS_SCORE", b["score"])

    print("[4/6] Drum hit events + physical modal/transient synthesis...")
    d = analyze_drums(st["drums_stereo"])
    write_js("models/drum-model.js", "DRUM_MODEL", d["model"]); write_js("data/drum-score.js", "DRUM_SCORE", d["score"])
    fams = list(d["score"].keys())
    parts = {f: render_drums({k: (1.0 if k == f else 0.0) for k in fams}, n)[0] for f in fams}
    drum_gains = calibrate_drum_gains(st["drums_stereo"], parts)

    print("[5/6] Guitar + Keys separate discrete note models (harmonic NMF + pick transient)...")
    pk = analyze_poly_dual(st["guitar_stereo"], st["keys_stereo"], iters=360)
    write_js("models/poly-model.js", "POLY_MODEL", pk["model"]); write_js("data/poly-score.js", "POLY_SCORE", pk["score"])
    yg, _ = render_poly({"guitar": 1.0, "keys": 0.0, "pick": 0.0}, n)
    yk, _ = render_poly({"guitar": 0.0, "keys": 1.0, "pick": 0.0}, n)
    gg = float(np.sqrt((st["guitar_stereo"].mean(1) ** 2).mean()) / max(np.sqrt((yg ** 2).mean()), 1e-9))
    gk = float(np.sqrt((st["keys_stereo"].mean(1) ** 2).mean()) / max(np.sqrt((yk ** 2).mean()), 1e-9))
    # Optimize guitar/keys balance on combined guitar_synth_stereo Mel-SNR
    best_p, best_mul = ev.mel_snr(st["guitar_synth_stereo"], yg * gg + yk * gk), (1.0, 1.0)
    for mg in (0.7, 0.9, 1.1, 1.3):
        for mk in (0.7, 0.9, 1.1, 1.3):
            s = ev.mel_snr(st["guitar_synth_stereo"], yg * (gg * mg) + yk * (gk * mk))
            if s > best_p:
                best_p, best_mul = s, (mg, mk)
    band_gains = {"bass": 1.0, "drums": drum_gains, "poly": {"guitar": round(gg * best_mul[0], 6), "keys": round(gk * best_mul[1], 6), "pick": 0.28}}
    write_js("data/band-gains.js", "BAND_GAINS", band_gains)

    print("[6/6] Full 11-track render & metrics...")
    write_meta(config, all_lines, voc["atlas_catalog"], {})
    r = render_full(n)
    m = {
        "synthMs": r["ms"],
        "leadMelSnrDb": ev.mel_snr(st["lead_mono"], r["lead"]),
        "backingMelSnrDb": ev.mel_snr(st["backing_stereo"], r["backing"]),
        "vocalsMelSnrDb": ev.mel_snr(st["vocals_stereo"], r["vocals"]),
        "bassMelSnrDb": ev.mel_snr(st["bass_mono"], r["bass"]),
        "drumsMelSnrDb": ev.mel_snr(st["drums_stereo"], r["drums"]),
        "guitarSynthMelSnrDb": ev.mel_snr(st["guitar_synth_stereo"], r["gtr"]),
        "masterMelSnrDb": ev.mel_snr(st["master"], r["master"]),
        "peakAmplitude": round(float(np.abs(r["master"]).max()), 4),
    }
    L = (n // 512) * 512
    er = np.sqrt((st["master"].mean(1)[:L].reshape(-1, 512) ** 2).mean(1)); es = np.sqrt((r["master"].mean(1)[:L].reshape(-1, 512) ** 2).mean(1))
    m["masterEnvCorr"] = round(float(np.corrcoef(er, es)[0, 1]), 4)
    files = {}
    for f in PURE_FILES:
        raw = (ROOT / f).read_bytes()
        files[f] = {"rawBytes": len(raw), "gzipBytes": len(gzip.compress(raw, 6))}
    tr = sum(v["rawBytes"] for v in files.values()); tg = sum(v["gzipBytes"] for v in files.values())
    metrics = {
        "version": "V9R2.1", "pureCode": True, "waveformFrames": 0, "residualCoefficients": 0,
        "sampleRate": 48000, "numSamples": n, "durationSec": round(n / 48000, 3),
        "stemConservationSnrDb": st["conservation_snr_db"],
        "atlasStats": voc["atlas_stats"], "bassStats": b["stats"], "drumStats": d["stats"], "polyStats": pk["stats"],
        "bandGains": band_gains, "metrics": m,
        "totalRawBytes": tr, "totalGzipBytes": tg, "totalRawKB": round(tr / 1024, 1), "totalGzipKB": round(tg / 1024, 1), "files": files,
    }
    (ROOT / "data/code-metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_meta(config, all_lines, voc["atlas_catalog"], metrics)
    print(json.dumps(m, ensure_ascii=False))
    for f, v in files.items():
        print(f"  {f:28s} raw {v['rawBytes']/1024:8.1f} KB  gz {v['gzipBytes']/1024:7.1f} KB")
    print(f"  TOTAL raw {tr/1024:.1f} KB gz {tg/1024:.1f} KB | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
