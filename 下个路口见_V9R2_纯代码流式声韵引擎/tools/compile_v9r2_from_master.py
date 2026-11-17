#!/usr/bin/env python3
"""Pure-Code V9R2 Compiler (`tools/compile_v9r2_from_master.py`).

Original 48kHz master is used ONLY for analysis/calibration. Output contains:
  - Vocals 01–08: speaker core + 97-syllable atlas + FDN params + 286 parametric events (V8.5 route)
  - Band 09–11: V9.2 physical-model score (instrument constants + symbolic parameter curves)
  - 3 scalar band loudness gains
No waveform frames, no transform-coefficient residuals.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
sys.path.insert(0, str(ROOT))

from parsers.lrc_syllable_parser import load_phoneme_lexicon, parse_lrc_with_tokens
from services.master_stem_conservator import extract_conserved_v85_stems
from services.v85_vocal_atlas_pipeline import (
    align_v85_syllable_events,
    analyze_vocal_signal_48k,
    compile_plain_v85_vocal_bundle,
)

PURE_FILES = [
    "models/singer-core.js",
    "models/syllable-atlas.js",
    "models/space-fdn.js",
    "models/instrument-physics.js",
    "data/vocal-events-v85.js",
    "data/band-score-expressive.js",
    "data/band-stem-gains.js",
]


def write_js(rel: str, banner: str, name: str, obj: object) -> None:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"// {banner}\nexport const {name} = {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))};\n", encoding="utf-8")


def size_of(rel: str) -> tuple[int, int]:
    raw = (ROOT / rel).read_bytes()
    return len(raw), len(gzip.compress(raw, compresslevel=6))


def mel_snr(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> float:
    r = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = min(len(r), len(s))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80, fmin=45.0, fmax=19500.0)
    Mr = fb @ np.abs(librosa.stft(r[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s[:L], n_fft=2048, hop_length=512))
    fl = float(np.max(Mr) * 1e-3 + 1e-8)
    a, b = np.log10(np.maximum(Mr, fl)), np.log10(np.maximum(Ms, fl))
    return round(float(10 * np.log10(np.sum(a ** 2) / (np.sum((a - b) ** 2) + 1e-12))), 2)


def env_corr(ref: np.ndarray, syn: np.ndarray, hop: int = 512) -> float:
    r = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = (min(len(r), len(s)) // hop) * hop
    er = np.sqrt(np.mean(r[:L].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    es = np.sqrt(np.mean(s[:L].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    return round(float(np.corrcoef(er, es)[0, 1]), 4)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def best_mel_gain(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> float:
    """Scalar gain maximizing Log-Mel SNR (mel of g*x == g*mel(x), so search in mel domain)."""
    r = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = min(len(r), len(s))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80, fmin=45.0, fmax=19500.0)
    Mr = fb @ np.abs(librosa.stft(r[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s[:L], n_fft=2048, hop_length=512))
    fl = float(np.max(Mr) * 1e-3 + 1e-8)
    a = np.log10(np.maximum(Mr, fl))
    g0 = rms(r) / max(rms(s), 1e-9)
    best_g, best_e = g0, 1e30
    for db in np.arange(-12.0, 12.01, 0.25):
        g = g0 * 10 ** (db / 20)
        e = float(np.sum((a - np.log10(np.maximum(Ms * g, fl))) ** 2))
        if e < best_e:
            best_g, best_e = g, e
    return round(float(best_g), 5)


def node_render(out_dir: Path) -> dict:
    u = lambda rel: (ROOT / rel).as_uri()
    js = f"""
import fs from 'node:fs';
import {{ SINGER_CORE }} from '{u('models/singer-core.js')}';
import {{ SYLLABLE_ATLAS }} from '{u('models/syllable-atlas.js')}';
import {{ SPACE_FDN }} from '{u('models/space-fdn.js')}';
import {{ INSTRUMENT_PHYSICS }} from '{u('models/instrument-physics.js')}';
import {{ VOCAL_EVENTS_V85 }} from '{u('data/vocal-events-v85.js')}';
import {{ BAND_SCORE_EXPRESSIVE }} from '{u('data/band-score-expressive.js')}';
import {{ BAND_STEM_GAINS }} from '{u('data/band-stem-gains.js')}';
import {{ renderV9r2AllLayers }} from '{u('src/dsp/v9r2-master-engine.js')}';
const config = JSON.parse(fs.readFileSync('{ROOT / 'schemas/v9r2_genesis_config.json'}', 'utf8'));
const res = renderV9r2AllLayers({{ config, singerCore: SINGER_CORE, syllableAtlas: SYLLABLE_ATLAS, spaceFdn: SPACE_FDN,
  vocalEvents: VOCAL_EVENTS_V85, instrumentPhysics: INSTRUMENT_PHYSICS, bandScore: BAND_SCORE_EXPRESSIVE, bandGains: BAND_STEM_GAINS }});
for (const [k, chans] of Object.entries(res.layers)) chans.forEach((c, i) => fs.writeFileSync('{out_dir}/' + k + '_ch' + i + '.f32', Buffer.from(c.buffer, c.byteOffset, c.byteLength)));
console.log(JSON.stringify({{ synthMs: res.synthMs }}));
"""
    p = subprocess.run(["node", "--max-old-space-size=6144", "--input-type=module", "-e", js], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def load_layers(d: Path, n: int) -> dict[str, np.ndarray]:
    r1 = lambda k: np.fromfile(d / f"{k}_ch0.f32", dtype=np.float32)[:n]
    r2 = lambda k: np.column_stack([r1(k), np.fromfile(d / f"{k}_ch1.f32", dtype=np.float32)[:n]])
    lead = sum(r1(k) for k in ("lead_sub_f0", "lead_vowel_body", "lead_harmonic_sheen", "lead_plosive_burst", "lead_sibilant", "lead_velvet_breath"))
    vocals = np.column_stack([lead, lead]) + r2("backing_harmony") + r2("vocal_space")
    bass = r1("bass")
    drums = r2("drums")
    gtr = r2("guitar_synth")
    return {
        "lead": lead,
        "vocals": vocals,
        "bass": bass,
        "drums": drums,
        "gtr": gtr,
        "master": vocals + np.column_stack([bass, bass]) + drums + gtr,
    }


def main() -> None:
    t0 = time.time()
    config = json.loads((ROOT / "schemas/v9r2_genesis_config.json").read_text(encoding="utf-8"))
    lexicon = load_phoneme_lexicon(ROOT / "schemas/phoneme_lexicon.json")
    mp3 = WS / "mp3_output/李宇春 - 下个, 路口, 见.mp3"
    lrc = WS / "mp3_output/李宇春 - 下个, 路口, 见.lrc"
    cache = Path("/Users/weckham/.gemini/antigravity/brain/74d2d208-e911-4def-a54e-f88353345d24/scratch/conservative_6stems_48k.npz")
    sr, hop = int(config["sampleRate"]), int(config["analysisHop"])

    print("[1/4] Decoding original master & separating analysis stems...")
    st = extract_conserved_v85_stems(mp3, sr=sr, hop=hop, cached_6stems_npz=cache)
    n = len(st["master"])
    dur = n / sr

    print("[2/4] V8.5 vocal analysis -> 97-syllable atlas + 286 parametric events...")
    feats = analyze_vocal_signal_48k(st["lead_mono"], config, backing_stereo=st["backing_stereo"])
    all_lines, sung = parse_lrc_with_tokens(lrc, lexicon, dur)
    events = align_v85_syllable_events(sung, feats, config)
    voc = compile_plain_v85_vocal_bundle(events, feats, st, config)
    write_js("models/singer-core.js", "Li Yuchun timbre core (48-ERB mean + 36-dim eigen-basis)", "SINGER_CORE", voc["singer_core"])
    write_js("models/syllable-atlas.js", "97 Mandarin syllable onset/nucleus prototypes", "SYLLABLE_ATLAS", voc["syllable_atlas"])
    write_js("models/space-fdn.js", "Early reflections + 8-line prime Householder FDN parameters", "SPACE_FDN", voc["space_fdn"])
    write_js("data/vocal-events-v85.js", "286 LRC-aligned `--` coarticulated parametric vocal events", "VOCAL_EVENTS_V85", voc["vocal_events"])
    write_js("data/band-stem-gains.js", "Band loudness calibration constants", "BAND_STEM_GAINS", {"bass": 1.0, "drums": 1.0, "guitarSynth": 1.0})

    tmp = Path(tempfile.mkdtemp(prefix="v9r2_pure_"))
    try:
        print("[3/4] Calibrating 3 band loudness constants against original stems...")
        node_render(tmp)
        lay = load_layers(tmp, n)
        gains = {
            "bass": best_mel_gain(st["bass_mono"], lay["bass"]),
            "drums": best_mel_gain(st["drums_stereo"], lay["drums"]),
            "guitarSynth": best_mel_gain(st["guitar_synth_stereo"], lay["gtr"]),
        }
        print("    gains:", gains)
        write_js("data/band-stem-gains.js", "Band loudness calibration constants", "BAND_STEM_GAINS", gains)

        print("[4/4] Final pure-code render & metrics vs original...")
        info = node_render(tmp)
        lay = load_layers(tmp, n)
        metrics = {
            "synthMs": info["synthMs"],
            "leadMelSnrDb": mel_snr(st["lead_mono"], lay["lead"]),
            "vocalsMelSnrDb": mel_snr(st["vocals_stereo"], lay["vocals"]),
            "bassMelSnrDb": mel_snr(st["bass_mono"], lay["bass"]),
            "drumsMelSnrDb": mel_snr(st["drums_stereo"], lay["drums"]),
            "guitarSynthMelSnrDb": mel_snr(st["guitar_synth_stereo"], lay["gtr"]),
            "masterMelSnrDb": mel_snr(st["master"], lay["master"]),
            "masterEnvCorr": env_corr(st["master"], lay["master"]),
            "peakAmplitude": round(float(np.max(np.abs(lay["master"]))), 4),
        }
        print("    ", json.dumps(metrics, ensure_ascii=False))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    files = {rel: dict(zip(("rawBytes", "gzipBytes"), size_of(rel))) for rel in PURE_FILES}
    tot_raw = sum(v["rawBytes"] for v in files.values())
    tot_gz = sum(v["gzipBytes"] for v in files.values())
    code_metrics = {
        "version": "V9R2-pure",
        "pureCode": True,
        "waveformFrames": 0,
        "residualCoefficients": 0,
        "sampleRate": sr,
        "numSamples": n,
        "durationSec": round(dur, 3),
        "atlasStats": voc["atlas_stats"],
        "bandStemGains": gains,
        "totalRawBytes": tot_raw,
        "totalGzipBytes": tot_gz,
        "totalRawKB": round(tot_raw / 1024, 1),
        "totalGzipKB": round(tot_gz / 1024, 1),
        "metrics": metrics,
        "files": files,
    }
    (ROOT / "data/code-metrics.json").write_text(json.dumps(code_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_js(
        "data/studio-metadata.js",
        "Studio config, lyrics, syllable catalog & verified metrics",
        "STUDIO_METADATA",
        {"config": config, "lyrics": {"durationSec": round(dur, 3), "lines": all_lines}, "atlasCatalog": voc["atlas_catalog"], "metrics": code_metrics},
    )
    for rel, v in files.items():
        print(f"    {rel:34s} raw={v['rawBytes']/1024:8.1f} KB  gz={v['gzipBytes']/1024:7.1f} KB")
    print(f"    TOTAL raw={tot_raw/1024:.1f} KB  gz={tot_gz/1024:.1f} KB | done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
