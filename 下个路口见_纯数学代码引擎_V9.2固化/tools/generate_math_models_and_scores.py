#!/usr/bin/env python3
"""Generates Dual-Tier (Ultra + Expressive) 100% Human-Readable Plain-Text JS Mathematical Models & Scores.

Zero LZMA, Zero Base122, Zero Waveforms, Zero Binary:
- Shared Vocal Physics (`models/singer-physics.js` + `models/mandarin-phonetics.js` + `data/vocal-lyrics.js`):
  - Full 40-mode Glasberg-Moore ERB vocal-tract baseline + 64-harmonic Rosenberg/Fant LF glottal phase
  - Full 24x40 Li Yuchun orthonormal vocal-tract timbre eigen-basis + 8x8 breathiness HNR eigen-basis
  - All 97 Mandarin syllables `3+5` stage 40-mode formant & 8-band HNR trajectories (0.00 dB loss)
- Shared Instrumental Physics (`models/instrument-physics.js` + `data/band-score.js` timeline metadata):
  - Track 08 Bass: 18-harmonic inharmonic electric string (`B=0.00018`) + `32x18` finger/pluck tone codebook
  - Tracks 06 & 07 Drums: Swept-sine Kick (`58.59Hz`) + Woodblock/Snare modal body (`164.06Hz`) + `64x36` percussive transient spectral shapes (split cleanly at `2.4-4.2kHz`)
  - Tracks 09 & 10 Guitar & Keys: 56-note (`MIDI 40..95`) coupled-rotor harmonic voices (split at `D4 / MIDI 62`) + `80fps` 4-band guitar strum transients + dynamic stereo FDN
- Dual-Tier Musical Scores (Both Vocal + Real-Detail 5-Track Band):
  - Version A (`ultra`): `vocal-score-ultra.js` (10-PCA) + `band-score-ultra.js` (Compact Real-Detail 5-Track Band)
  - Version B (`expressive`): `vocal-score-expressive.js` (24-PCA) + `band-score-expressive.js` (Studio-Precision 5-Track Band)
"""
from __future__ import annotations

import gzip
import json
import lzma
import struct
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = Path("/Users/weckham/Desktop/MUSIC/下个路口见_事件音节波码")
sys.path.insert(0, str(V8_ROOT))

from parsers.lrc_tokenizer import load_phoneme_lexicon, parse_lrc_with_tokens
from services.event_codec import (
    decode_base122,
    decompress_lzma1_alone,
    unpack_event_stream,
    unpack_speaker_core,
    unpack_syllable_atlas,
)

V7_B122_ILLEGALS = [0, 10, 13, 34, 60, 92]


def _load_v8_pkt(rel_path: str) -> bytes:
    txt = (V8_ROOT / rel_path).read_text(encoding="utf-8")
    b122 = txt[txt.index('"') + 1 : txt.rindex('"')]
    return decompress_lzma1_alone(decode_base122(b122))


def _load_v5_accomp_pkt(rel_path: str) -> bytes:
    txt = (V8_ROOT / rel_path).read_text(encoding="utf-8")
    s = txt[txt.index('"') + 1 : txt.rindex('"')]
    colon = s.index(":")
    out_len = int(s[:colon])
    out = bytearray(out_len)
    bit_pos = 0

    def push7(v: int) -> None:
        nonlocal bit_pos
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        if byte_idx < out_len:
            shift = 1 - bit_off
            if shift >= 0:
                out[byte_idx] = (out[byte_idx] | (v << shift)) & 0xFF
            else:
                out[byte_idx] = (out[byte_idx] | (v >> (-shift))) & 0xFF
                if byte_idx + 1 < out_len:
                    out[byte_idx + 1] = (out[byte_idx + 1] | (v << (8 + shift))) & 0xFF
        bit_pos += 7

    for ch in s[colon + 1 :]:
        cp = ord(ch)
        if cp < 0x80:
            push7(cp)
        else:
            payload = cp - 0x100
            idx = payload >> 7
            v2 = payload & 0x7F
            if idx == 7:
                push7(v2)
            else:
                push7(V7_B122_ILLEGALS[idx])
                push7(v2)
    return lzma.decompress(bytes(out), format=lzma.FORMAT_ALONE)


def row_delta_encode_2d(mat_int: np.ndarray) -> list[list[int]]:
    out = []
    for r in range(mat_int.shape[0]):
        if r == 0:
            out.append([int(x) for x in mat_int[0]])
        else:
            out.append([int(x) for x in (mat_int[r] - mat_int[r - 1])])
    return out


def delta_encode_1d(arr_int: np.ndarray) -> list[int]:
    if len(arr_int) == 0:
        return []
    out = [int(arr_int[0])]
    for i in range(1, len(arr_int)):
        out.append(int(arr_int[i]) - int(arr_int[i - 1]))
    return out


def rdp_indices(y: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker 1D knot finder preserving exact peaks & transients."""
    n = len(y)
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]
    xs = np.arange(n, dtype=np.float64)
    yf = y.astype(np.float64)
    while stack:
        i, j = stack.pop()
        if j - i <= 1:
            continue
        slope = (yf[j] - yf[i]) / (j - i)
        interp = yf[i] + slope * (xs[i + 1 : j] - i)
        err = np.abs(yf[i + 1 : j] - interp)
        k_rel = int(np.argmax(err))
        if err[k_rel] > tol:
            k = i + 1 + k_rel
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return np.nonzero(keep)[0]


def encode_pwl_smooth_zero(y: np.ndarray, min_gate: int, div: int, tol_q: float) -> list[list[int]]:
    """Encodes 1D integer trajectory into separated `[dt_list, dv_list]` piecewise-linear knots."""
    ys = np.maximum(0, y.astype(np.int32) - min_gate)
    if np.all(ys == 0):
        return [[], []]
    idx = rdp_indices(ys, tol_q)
    dt = np.diff(idx, prepend=0).astype(np.int32).tolist()
    vq = np.round(ys[idx] / div).astype(np.int32)
    dv = np.diff(vq, prepend=0).astype(np.int32).tolist()
    return [dt, dv]


def encode_rle_pair(arr: np.ndarray, active_mask: np.ndarray) -> list[list[int]]:
    """Encodes discrete codebook index stream into `[runLengths, deltaIndices]`, holding index across silent gaps."""
    clean = arr.copy()
    nz = np.nonzero(active_mask)[0]
    last_v = int(clean[nz[0]]) if len(nz) > 0 else 0
    for i in range(len(clean)):
        if active_mask[i]:
            last_v = int(clean[i])
        else:
            clean[i] = last_v
    lens, dvals = [], []
    i, prev_v, n = 0, 0, len(clean)
    while i < n:
        j = i + 1
        while j < n and clean[j] == clean[i]:
            j += 1
        lens.append(int(j - i))
        dvals.append(int(clean[i]) - prev_v)
        prev_v = int(clean[i])
        i = j
    return [lens, dvals]


def hold_f0_across_silence(f0_q: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """Holds pitch steady across unvoiced gaps and steps cleanly 1 frame before next voiced onset."""
    out = f0_q.copy()
    nz = np.nonzero(active_mask & (f0_q > 0))[0]
    if len(nz) == 0:
        return out
    out[: nz[0]] = int(f0_q[nz[0]])
    last_f = int(f0_q[nz[0]])
    for i in range(nz[0], len(out)):
        if active_mask[i] and f0_q[i] > 0:
            last_f = int(f0_q[i])
            out[i] = last_f
        else:
            out[i] = last_f
    return out


def build_tier_vocal_score_js(
    tier_id: str,
    const_suffix: str,
    short_label: str,
    full_label: str,
    v85_ref_kb: float,
    events: list,
    ev_meta: dict,
    protos: list,
) -> str:
    k_pca = int(ev_meta["pcaDims"])
    k_aper = int(ev_meta["aperExprBands"])
    profile_obj = {
        "id": tier_id,
        "shortLabel": short_label,
        "fullLabel": full_label,
        "v85RefLzmaKb": v85_ref_kb,
        "pcaDims": k_pca,
        "aperExprBands": k_aper,
        "exprKnotStepFrames": int(ev_meta["exprKnotStepFrames"]),
        "f0KnotStepFrames": int(ev_meta["f0KnotStepFrames"]),
        "envKnotStepFrames": int(ev_meta["envKnotStepFrames"]),
        "quantStepDb": round(float(ev_meta["quantStepDb"]), 6),
    }

    note_lines = []
    for ev in events:
        L = int(ev.total_frames)
        cf = int(ev.cons_frames)
        unv_only = ev.voiced_onset_frame >= L
        fl = (
            (1 if ev.legato_prev else 0)
            | (2 if ev.legato_next else 0)
            | (4 if ev.is_hook else 0)
            | (8 if unv_only else 0)
            | (16 if ev.is_plosive else 0)
        )
        p_obj = protos[ev.syllable_id]
        f0_steps = [] if unv_only else [int(x) for x in ev.f0_step_q]
        d_gh = delta_encode_1d(ev.gain_h_db_q.astype(int))
        d_ga = delta_encode_1d(ev.gain_a_db_q.astype(int))
        d_em = row_delta_encode_2d(ev.expr_mcep_q[:, :k_pca].astype(int)) if ev.expr_mcep_q.size else []
        d_ea = row_delta_encode_2d(ev.expr_aper_q[:, :k_aper].astype(int)) if ev.expr_aper_q.size else []
        back = (
            0
            if (ev.back_gain_l_q == 0 and ev.back_gain_r_q == 0)
            else [int(ev.back_gain_l_q), int(ev.back_gain_r_q), int(ev.back_pitch_q)]
        )
        row = [
            int(ev.start_frame),
            L,
            cf,
            ev.display_text,
            p_obj.pinyin,
            fl,
            int(ev.stereo_pan_q),
            back,
            0 if unv_only else int(ev.base_midi_q),
            f0_steps,
            d_gh,
            d_ga,
            d_em,
            d_ea,
        ]
        note_lines.append("  " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    return (
        f"// Pure-Code Vocal Score ({full_label}) — 285 Sung Events, Zero LZMA / Zero Binary\n"
        f"// Tuple: [startFrame, totalFrames, consFrames, hanzi, pinyin, flags, panQ, back[L,R,semiQ]|0, baseMidiQ, f0Steps[], dGainHQ[], dVdegQ[], dExprTimbre{k_pca}[], dExprAper{k_aper}[]]\n\n"
        f"export const VOCAL_PROFILE_{const_suffix} = {json.dumps(profile_obj, ensure_ascii=False, indent=2)};\n\n"
        f"export const VOCAL_SCORE_{const_suffix} = [\n"
        + ",\n".join(note_lines)
        + "\n];\n"
    )


def to_idiomatic_js(val, indent: int = 0) -> str:
    """Formats nested dicts/lists as clean, human-readable ES object literals (unquoted identifier keys)."""
    sp = "  " * indent
    if isinstance(val, dict):
        lines = ["{"]
        items = list(val.items())
        for idx, (k, v) in enumerate(items):
            comma = "," if idx + 1 < len(items) else ""
            lines.append(f"{sp}  {k}: {to_idiomatic_js(v, indent + 1)}{comma}")
        lines.append(f"{sp}}}")
        return "\n".join(lines)
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list) and len(val) >= 4:
        rows = [f"{sp}  {json.dumps(r, ensure_ascii=False, separators=(',', ':'))}" for r in val]
        return "[\n" + ",\n".join(rows) + f"\n{sp}]"
    return json.dumps(val, ensure_ascii=False, separators=(",", ":"))


def unpack_and_build_band_assets() -> tuple[str, str, str, str]:
    accomp_models = json.loads((V8_ROOT / "data/accomp_v5_compact/accomp_models.json").read_text(encoding="utf-8"))
    raw_b = _load_v5_accomp_pkt("data/accomp_v5_compact/bass-wave.js")
    raw_d = _load_v5_accomp_pkt("data/accomp_v5_compact/drums-wave.js")
    raw_h = _load_v5_accomp_pkt("data/accomp_v5_compact/harmony-wave.js")

    # --- 1. Unpack WVB5 (18-Harmonic Inharmonic Bass) ---
    nf_b, max_h_b, n_tcb_b, tone_scale_b = struct.unpack_from("<IBBxxf", raw_b, 4)
    off = 16
    cb_tone_i8 = np.frombuffer(raw_b, dtype=np.int8, count=n_tcb_b * max_h_b, offset=off).reshape(n_tcb_b, max_h_b)
    off += n_tcb_b * max_h_b
    rms_b = (np.cumsum(np.frombuffer(raw_b, dtype=np.uint8, count=nf_b, offset=off), dtype=np.int64) & 0xFF).astype(np.int32)
    off += nf_b
    f0_b = np.cumsum(np.frombuffer(raw_b, dtype="<i2", count=nf_b, offset=off), dtype=np.int64).astype(np.int32)
    off += nf_b * 2
    tone_b = np.frombuffer(raw_b, dtype=np.uint8, count=nf_b, offset=off).astype(np.int32)
    off += nf_b
    nr_b = np.frombuffer(raw_b, dtype=np.uint8, count=nf_b, offset=off).astype(np.int32)

    # --- 2. Unpack WVD5 (Swept-Sine Kick + Woodblock/Snare Modal Body + 36-Band Percussive Transients) ---
    nf_d, n_comp_d, n_bands_d, n_protos_d, ql_d, bar_f_d, n_bars_d, kick_hz, snare_hz = struct.unpack_from(
        "<IBBBBHHff", raw_d, 4
    )
    off = 24
    band_centers = np.frombuffer(raw_d, dtype="<f4", count=n_bands_d, offset=off).astype(np.float64)
    off += n_bands_d * 4
    w_drum = np.frombuffer(raw_d, dtype="<f4", count=n_bands_d * n_comp_d, offset=off).reshape(n_bands_d, n_comp_d).astype(np.float64)
    off += n_bands_d * n_comp_d * 4
    h_max = np.frombuffer(raw_d, dtype="<f4", count=n_comp_d, offset=off).astype(np.float64)
    off += n_comp_d * 4
    proto_grooves = np.frombuffer(raw_d, dtype=np.uint8, count=n_protos_d * n_comp_d * bar_f_d, offset=off).reshape(
        n_protos_d, n_comp_d, bar_f_d
    )
    off += proto_grooves.size
    bar_sched = np.frombuffer(raw_d, dtype=np.uint8, count=n_bars_d, offset=off)
    off += n_bars_d
    d_res_h = np.frombuffer(raw_d, dtype=np.uint8, count=n_comp_d * nf_d, offset=off).reshape(n_comp_d, nf_d)
    off += d_res_h.size
    kick_q = (np.cumsum(np.frombuffer(raw_d, dtype=np.uint8, count=nf_d, offset=off), dtype=np.int64) & 0xFF).astype(np.int32)
    off += nf_d
    body_q = (np.cumsum(np.frombuffer(raw_d, dtype=np.uint8, count=nf_d, offset=off), dtype=np.int64) & 0xFF).astype(np.int32)
    off += nf_d
    high_q = (np.cumsum(np.frombuffer(raw_d, dtype=np.uint8, count=nf_d, offset=off), dtype=np.int64) & 0xFF).astype(np.int32)

    h_rec = np.zeros((n_comp_d, nf_d), dtype=np.float64)
    for c in range(n_comp_d):
        for t in range(nf_d):
            b_idx = t // bar_f_d
            qv = (int(proto_grooves[bar_sched[b_idx], c, t - b_idx * bar_f_d]) + int(d_res_h[c, t])) & 0xFF
            h_rec[c, t] = ((qv / ql_d) ** 2) * h_max[c]

    with np.errstate(all="ignore"):
        mf_all = (w_drum @ h_rec).T
    norms = np.linalg.norm(mf_all, axis=1, keepdims=True)
    active_d = (high_q >= 12) & (norms[:, 0] > 1e-9)
    unit_mf = mf_all[active_d] / norms[active_d]
    high_rms = np.where(high_q > 0, 10.0 ** ((high_q / 3.2 - 75.0) / 20.0), 0.0)
    weights_d = high_rms[active_d]

    rng = np.random.default_rng(42)
    n_shapes = 64
    centers = unit_mf[rng.choice(len(unit_mf), size=n_shapes, replace=False, p=weights_d / weights_d.sum())].copy()
    for _ in range(35):
        with np.errstate(all="ignore"):
            dots = unit_mf @ centers.T
        assign = np.argmax(dots, axis=1)
        for k in range(n_shapes):
            m = assign == k
            if np.any(m):
                v = (unit_mf[m] * weights_d[m, None]).sum(axis=0)
                centers[k] = v / max(1e-12, float(np.linalg.norm(v)))
    with np.errstate(all="ignore"):
        best_idx = np.argmax(unit_mf @ centers.T, axis=1)
    shape_full = np.zeros(nf_d, dtype=np.int32)
    shape_full[active_d] = best_idx

    # --- 3. Unpack WVH5 (56-Note Polyphonic Coupled-Rotor Harmony + 80fps 4-Band Guitar Strum) ---
    nf_h, n_notes_h, top_v_h, ql_h, midi_min_h, max_amp_h = struct.unpack_from("<IBBBBf", raw_h, 4)
    off = 16
    note_amp_q = np.zeros((n_notes_h, nf_h), dtype=np.int32)
    for n in range(n_notes_h):
        note_amp_q[n] = np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off + n * nf_h), dtype=np.int64) & 0xFF
    off += n_notes_h * nf_h
    bright_q = np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off).astype(np.int32)
    off += nf_h
    harm_q = (np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off), dtype=np.int64) & 0xFF).astype(np.int32)
    off += nf_h
    ns_h = nf_h * 2
    strum_q = np.zeros((4, ns_h), dtype=np.int32)
    for b in range(4):
        strum_q[b] = np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=ns_h, offset=off + b * ns_h), dtype=np.int64) & 0xFF
    off += 4 * ns_h
    side_q = (np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off), dtype=np.int64) & 0xFF).astype(np.int32)

    # --- 4. Build `models/instrument-physics.js` ---
    inst_obj = {
        "bass": {
            "sampleRate": int(accomp_models["bass"]["sampleRate"]),
            "hopSamples": int(accomp_models["bass"]["hopSamples"]),
            "maxHarmonics": int(max_h_b),
            "inharmonicityB": float(accomp_models["bass"]["inharmonicityB"]),
            "toneScale": round(float(tone_scale_b), 6),
            "staticHarmonicLogProfile": [round(float(x), 5) for x in accomp_models["bass"]["staticHarmonicLogProfile"]],
            "toneCodebookI8": cb_tone_i8.astype(int).tolist(),
        },
        "drums": {
            "sampleRate": int(accomp_models["drums"]["sampleRate"]),
            "hopSamples": int(accomp_models["drums"]["hopSamples"]),
            "fftSize": 512,
            "kickBaseHz": round(float(kick_hz), 3),
            "snareBaseHz": round(float(snare_hz), 3),
            "crossoverLowHz": 2400.0,
            "crossoverHighHz": 4200.0,
            "shapeScaleDiv": 1000.0,
            "bandCentersHz": [round(float(x), 2) for x in band_centers],
            "percShapeCodebookQ": np.round(centers * 1000.0).astype(int).tolist(),
        },
        "harmony": {
            "sampleRate": int(accomp_models["harmony"]["sampleRate"]),
            "hopSamples": int(accomp_models["harmony"]["hopSamples"]),
            "fftSize": 512,
            "midiMin": int(midi_min_h),
            "midiMax": int(midi_min_h + n_notes_h - 1),
            "nNotes": int(n_notes_h),
            "guitarSplitMidi": 62,
            "maxHarmonics": int(accomp_models["harmony"]["maxHarmonics"]),
            "warmHarmonicWeights": [round(float(x), 5) for x in accomp_models["harmony"]["warmHarmonicWeights"]],
            "brightHarmonicWeights": [round(float(x), 5) for x in accomp_models["harmony"]["brightHarmonicWeights"]],
            "strumCentersHz": [round(float(x), 1) for x in accomp_models["harmony"]["strumCentersHz"]],
            "stereoAcoustics": accomp_models["harmony"]["stereoAcoustics"],
        },
    }
    inst_js = (
        "// Pure Mathematical 5-Track Instrumental Physical & Spectral Prototypes\n"
        "// Zero LZMA / Zero Base122 / Zero Waveforms — Shared across Ultra & Expressive Band Scores\n"
        "export const INSTRUMENT_PHYSICS = "
        + to_idiomatic_js(inst_obj)
        + ";\n"
    )

    # --- 5. Build Structural Timeline `data/band-score.js` (for interactive bar/chord visualizer) ---
    band_struct_js = """// Structural Bar & Harmonic Timeline Metadata for 《李宇春 - 下个, 路口, 见》 (98.0 BPM)
// Used by Interactive Score Canvas & Section Jump Navigation
export const BAND_SCORE = {
  bpm: 98.0,
  bar0Sec: 1.7493,
  totalBars: 83,
  progressions: {
    STD4:  [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Dm9"]],
    ROLL4: [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Am7"]],
  },
  sections: [
    { name: "Intro",        startBar: 0,  bars: 8, prog: "STD4" },
    { name: "Verse 1",      startBar: 8,  bars: 8, prog: "STD4" },
    { name: "Pre-Chorus 1", startBar: 16, bars: 8, prog: "STD4" },
    { name: "Chorus 1",     startBar: 24, bars: 8, prog: "ROLL4", resolveLastDm: true },
    { name: "Interlude",    startBar: 32, bars: 8, prog: "STD4" },
    { name: "Verse 2A",     startBar: 40, bars: 1, prog: "STD4" },
    { name: "Verse 2 Drop", startBar: 41, bars: 3, prog: "STD4", progOffset: 1 },
    { name: "Verse 2B",     startBar: 44, bars: 4, prog: "STD4" },
    { name: "Pre-Chorus 2", startBar: 48, bars: 8, prog: "STD4" },
    { name: "Chorus 2",     startBar: 56, bars: 8, prog: "ROLL4", resolveLastDm: true },
    { name: "Chorus 3",     startBar: 64, bars: 8, prog: "ROLL4", resolveLastDm: true },
    { name: "Outro Groove", startBar: 72, bars: 8, prog: "STD4" },
    { name: "Final Chord",  startBar: 80, bars: 3, prog: "FINAL" },
  ],
};
"""

    def encode_band_tier(
        tier_id: str,
        const_suffix: str,
        label: str,
        rms_gate: int,
        rms_div: int,
        rms_tol: float,
        f0_div: int,
        f0_tol: float,
        drum_gate: int,
        drum_div: int,
        drum_tol: float,
        sh_stride: int,
        note_gate: int,
        note_div: int,
        note_tol: float,
        top_k_notes: int,
        strum_gate: int,
        strum_div: int,
        strum_tol: float,
    ) -> str:
        f0_held = hold_f0_across_silence(f0_b, rms_b >= rms_gate)
        ts = 2 if sh_stride >= 4 else 1
        bass_score = {
            "nFrames": int(nf_b),
            "rmsGate": rms_gate,
            "rmsDiv": rms_div,
            "f0Div": f0_div,
            "nrGate": 6,
            "nrDiv": 8,
            "toneStride": ts,
            "rms": encode_pwl_smooth_zero(rms_b, rms_gate, rms_div, rms_tol),
            "f0": encode_pwl_smooth_zero(f0_held, 0, f0_div, f0_tol),
            "nr": encode_pwl_smooth_zero(nr_b, 6, 8, 10.0),
            "tone": encode_rle_pair(tone_b[::ts], (rms_b >= rms_gate)[::ts]),
        }

        drum_score = {
            "nFrames": int(nf_d),
            "gate": drum_gate,
            "div": drum_div,
            "shapeStride": sh_stride,
            "kick": encode_pwl_smooth_zero(kick_q, drum_gate, drum_div, drum_tol),
            "body": encode_pwl_smooth_zero(body_q, drum_gate, drum_div, drum_tol),
            "high": encode_pwl_smooth_zero(high_q, drum_gate, drum_div, drum_tol),
            "shape": encode_rle_pair(shape_full[::sh_stride], active_d[::sh_stride]),
        }

        na_work = note_amp_q.copy()
        if top_k_notes < 8:
            part = np.argsort(na_work, axis=0)
            for c in range(nf_h):
                na_work[part[:-top_k_notes, c], c] = 0

        harm_score = {
            "nFrames": int(nf_h),
            "qLevels": int(ql_h),
            "maxNoteAmp": round(float(max_amp_h), 6),
            "envGate": 14,
            "envDiv": drum_div,
            "noteGate": note_gate,
            "noteDiv": note_div,
            "strumGate": strum_gate,
            "strumDiv": strum_div,
            "harm": encode_pwl_smooth_zero(harm_q, 14, drum_div, drum_tol),
            "bright": encode_pwl_smooth_zero(bright_q, 0, 4, 8.0),
            "side": encode_pwl_smooth_zero(side_q, 0, 4, 8.0),
            "strum": [encode_pwl_smooth_zero(strum_q[b], strum_gate, strum_div, strum_tol) for b in range(4)],
            "notes": [encode_pwl_smooth_zero(na_work[n], note_gate, note_div, note_tol) for n in range(n_notes_h)],
        }

        band_prof = {
            "id": tier_id,
            "label": label,
            "polyphonicVoicesPerFrame": top_k_notes,
            "drumTimingMs": 10,
            "guitarStrumTimingMs": 12.5,
            "bassPitchPrecisionCents": round(1200.0 * f0_div / 384.0, 2),
        }

        full_obj = {
            "bass": bass_score,
            "drums": drum_score,
            "harmony": harm_score,
        }
        return (
            f"// Pure-Code 5-Track Real-Arrangement Band Score ({label})\n"
            f"// Tracks 06..10: Swept Kick + Woodblock/Snare + 64-Shape Percussive Transients + 18-Harm Inharmonic Bass + 56-Note Coupled-Rotor Guitar/Keys + 80fps Strums\n\n"
            f"export const BAND_PROFILE_{const_suffix} = {to_idiomatic_js(band_prof)};\n\n"
            f"export const BAND_SCORE_{const_suffix} = {to_idiomatic_js(full_obj)};\n"
        )

    band_ultra_js = encode_band_tier(
        tier_id="ultra",
        const_suffix="ULTRA",
        label="紧凑全细节伴奏档 (6声部复调 + 10ms真鼓组加花 + 真贝斯滑音)",
        rms_gate=18,
        rms_div=3,
        rms_tol=4.8,
        f0_div=2,
        f0_tol=3.4,
        drum_gate=22,
        drum_div=3,
        drum_tol=5.2,
        sh_stride=4,
        note_gate=12,
        note_div=3,
        note_tol=3.4,
        top_k_notes=6,
        strum_gate=26,
        strum_div=4,
        strum_tol=5.8,
    )

    band_expr_js = encode_band_tier(
        tier_id="expressive",
        const_suffix="EXPRESSIVE",
        label="母带高精全细节伴奏档 (8声部满配复调 + 10ms木鱼拍手鬼音 + 3.1音分贝斯滑音 + 80fps扫弦瞬态)",
        rms_gate=12,
        rms_div=2,
        rms_tol=2.6,
        f0_div=1,
        f0_tol=2.2,
        drum_gate=16,
        drum_div=2,
        drum_tol=3.0,
        sh_stride=2,
        note_gate=9,
        note_div=2,
        note_tol=2.2,
        top_k_notes=8,
        strum_gate=20,
        strum_div=2,
        strum_tol=3.4,
    )

    return inst_js, band_struct_js, band_ultra_js, band_expr_js


def main() -> None:
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    raw_spk = _load_v8_pkt("data/shared/speaker_core.js")
    raw_atl = _load_v8_pkt("data/shared/syllable_atlas.js")
    raw_fdn = _load_v8_pkt("data/shared/space_fdn.js")
    raw_ultra = _load_v8_pkt("data/ultra/event_stream.js")
    raw_expr = _load_v8_pkt("data/expressive/event_stream.js")

    spk, meta = unpack_speaker_core(raw_spk)
    protos = unpack_syllable_atlas(raw_atl)
    ev_ultra, meta_ultra = unpack_event_stream(raw_ultra)
    ev_expr, meta_expr = unpack_event_stream(raw_expr)

    _, _, n_cep, n_aper, atlas_step_db, _ = struct.unpack_from("<4sHBBfH", raw_atl, 0)

    _, sr_fdn, n_samp_fdn, env_hop_orig, rt60_sec, damp_lp, base_wet, pre_delay = struct.unpack_from(
        "<4sIIIfffI", raw_fdn, 0
    )
    n_env_orig = (n_samp_fdn + env_hop_orig - 1) // env_hop_orig
    gm_raw = np.frombuffer(raw_fdn, dtype=np.uint8, count=n_env_orig, offset=32)
    gs_raw = np.frombuffer(raw_fdn, dtype=np.uint8, count=n_env_orig, offset=32 + n_env_orig)
    stride_fdn = 4
    gm_sub = [int(x) for x in gm_raw[::stride_fdn]]
    gs_sub = [int(x) for x in gs_raw[::stride_fdn]]

    sr = int(meta["sampleRate"])
    n_samples = int(meta["numSamples"])
    hop = int(meta["hop"])
    ref_peak = float(meta["refPeak"])
    dur_sec = round(n_samples / float(sr), 3)

    # 1. `models/singer-physics.js`
    mean_mcep_40 = [round(float(x), 4) for x in spk.mean_mcep[:n_cep]]
    glottal_phase_64 = [round(float(x), 4) for x in spk.glottal_phase]
    aper_mean_8 = [round(float(x), 4) for x in spk.aper_mean[:n_aper]]
    mcep_basis_24 = np.round(spk.mcep_pca_basis[:24, :n_cep] * 2048.0).astype(int).tolist()
    aper_basis_8 = np.round(spk.aper_pca_basis[:8, :n_aper] * 2048.0).astype(int).tolist()

    singer_js = f"""// Pure Mathematical Li Yuchun Timbre Eigen-Model (Full 24-PCA + 40-ERB) & Studio FDN Space Constants
// Zero LZMA / Zero Base122 / Zero Binary — Shared across Ultra (10-PCA) & Expressive (24-PCA) Tiers
export const SINGER_PHYSICS = {{
  sampleRate: {sr},
  hopSamples: {hop},
  fftSize: 512,
  durationSec: {dur_sec},
  totalSamples: {n_samples},
  refPeak: {round(ref_peak, 6)},
  cepstralCoeffs: {n_cep},
  aperiodicBands: {n_aper},
  maxHarmonics: 64,
  erbFreqMinHz: 75.0,
  erbFreqMaxHz: 16000.0,
  aperFreqMinHz: 180.0,
  aperFreqMaxHz: 16500.0,
  atlasStepDb: {round(float(atlas_step_db), 6)},
  basisScaleDiv: 2048.0,
  meanErbModes40: {json.dumps(mean_mcep_40, separators=(",", ":"))},
  glottalPhase64: {json.dumps(glottal_phase_64, separators=(",", ":"))},
  aperMean8: {json.dumps(aper_mean_8, separators=(",", ":"))},
  timbreEigenBasis: {json.dumps(mcep_basis_24, separators=(",", ":"))},
  aperEigenBasis: {json.dumps(aper_basis_8, separators=(",", ":"))},
  fdnReverb: {{
    delaysSamples: [1499, 1777, 2081, 2437, 2801, 3217, 3671, 4177],
    envHopSamples: {env_hop_orig * stride_fdn},
    rt60Sec: {round(float(rt60_sec), 4)},
    dampLp: {round(float(damp_lp), 4)},
    baseWetGain: {round(float(base_wet), 4)},
    preDelaySamples: {int(pre_delay)},
    gmEnvU8: {json.dumps(gm_sub, separators=(",", ":"))},
    gsEnvU8: {json.dumps(gs_sub, separators=(",", ":"))},
  }},
}};
"""
    (PROJECT_ROOT / "models/singer-physics.js").write_text(singer_js, encoding="utf-8")

    # 2. `models/mandarin-phonetics.js`
    syl_lines = []
    for p in protos:
        cm_i = np.round(p.cons_mcep_knots / atlas_step_db).astype(int)
        ca_i = np.round(p.cons_aper_knots / atlas_step_db).astype(int)
        vm_i = np.round(p.vowel_mcep_knots / atlas_step_db).astype(int)
        va_i = np.round(p.vowel_aper_knots / atlas_step_db).astype(int)
        entry = [
            p.initial_class,
            row_delta_encode_2d(cm_i),
            row_delta_encode_2d(ca_i),
            row_delta_encode_2d(vm_i),
            row_delta_encode_2d(va_i),
        ]
        syl_lines.append(f'  "{p.pinyin}":{json.dumps(entry, separators=(",", ":"))}')

    phonetics_js = (
        "// Pure Mathematical 97-Syllable 40-Mode ERB Vocal-Tract & Valley-HNR Trajectory Atlas\n"
        "// Zero LZMA / Zero Base122 / Zero Binary — 100% Plain-Text Row-Delta Integer Matrices\n\n"
        "export const MANDARIN_SYLLABLES = {\n"
        + ",\n".join(syl_lines)
        + "\n};\n"
    )
    (PROJECT_ROOT / "models/mandarin-phonetics.js").write_text(phonetics_js, encoding="utf-8")

    # 3. `data/vocal-lyrics.js`
    lexicon = load_phoneme_lexicon(V8_ROOT / "schemas/phoneme_lexicon.json")
    _, sung_lines = parse_lrc_with_tokens(
        Path("/Users/weckham/Desktop/MUSIC/mp3_output/李宇春 - 下个, 路口, 见.lrc"),
        lexicon,
        dur_sec,
    )
    lrc_entries = [
        f'  [{int(round(ln["startSec"] * 1000.0))}, {json.dumps(ln["text"], ensure_ascii=False)}]'
        for ln in sung_lines
    ]
    lyrics_js = (
        "// Synchronized Plain-Text Lyric Lines for 《李宇春 - 下个, 路口, 见》\n"
        "export const VOCAL_LYRIC_LINES = [\n"
        + ",\n".join(lrc_entries)
        + "\n];\n"
    )
    (PROJECT_ROOT / "data/vocal-lyrics.js").write_text(lyrics_js, encoding="utf-8")

    # 4. Vocal Tier Scores
    ultra_js = build_tier_vocal_score_js(
        tier_id="ultra",
        const_suffix="ULTRA",
        short_label="极简档 · 10维人声 + 紧凑真伴奏",
        full_label="极简高保真档 · 10维PCA人声微表情 + 6声部全细节原曲伴奏",
        v85_ref_kb=51.96,
        events=ev_ultra,
        ev_meta=meta_ultra,
        protos=protos,
    )
    (PROJECT_ROOT / "data/vocal-score-ultra.js").write_text(ultra_js, encoding="utf-8")

    expr_js = build_tier_vocal_score_js(
        tier_id="expressive",
        const_suffix="EXPRESSIVE",
        short_label="精修档 · 24维全人声 + 母带真伴奏",
        full_label="母带精修全维档 · 24维PCA人声微表情 + 8声部80fps扫弦原曲伴奏",
        v85_ref_kb=158.58,
        events=ev_expr,
        ev_meta=meta_expr,
        protos=protos,
    )
    (PROJECT_ROOT / "data/vocal-score-expressive.js").write_text(expr_js, encoding="utf-8")

    # 5. Band Physics & Dual-Tier Real-Detail Band Scores
    inst_js, band_struct_js, band_ultra_js, band_expr_js = unpack_and_build_band_assets()
    (PROJECT_ROOT / "models/instrument-physics.js").write_text(inst_js, encoding="utf-8")
    (PROJECT_ROOT / "data/band-score.js").write_text(band_struct_js, encoding="utf-8")
    (PROJECT_ROOT / "data/band-score-ultra.js").write_text(band_ultra_js, encoding="utf-8")
    (PROJECT_ROOT / "data/band-score-expressive.js").write_text(band_expr_js, encoding="utf-8")

    # 6. `data/vocal-tiers.js` — Unified Dual-Tier Registry (Vocal + Real-Detail 5-Track Band)
    tiers_js = """// Pure-Math Dual-Tier Registry (Matched Vocal + Real-Detail 5-Track Band Editions)
import { VOCAL_PROFILE_EXPRESSIVE, VOCAL_SCORE_EXPRESSIVE } from './vocal-score-expressive.js';
import { VOCAL_PROFILE_ULTRA, VOCAL_SCORE_ULTRA } from './vocal-score-ultra.js';
import { BAND_PROFILE_EXPRESSIVE, BAND_SCORE_EXPRESSIVE } from './band-score-expressive.js';
import { BAND_PROFILE_ULTRA, BAND_SCORE_ULTRA } from './band-score-ultra.js';

export const VOCAL_TIERS = {
  expressive: {
    profile: VOCAL_PROFILE_EXPRESSIVE,
    score: VOCAL_SCORE_EXPRESSIVE,
    bandProfile: BAND_PROFILE_EXPRESSIVE,
    bandScore: BAND_SCORE_EXPRESSIVE,
  },
  ultra: {
    profile: VOCAL_PROFILE_ULTRA,
    score: VOCAL_SCORE_ULTRA,
    bandProfile: BAND_PROFILE_ULTRA,
    bandScore: BAND_SCORE_ULTRA,
  },
};

export const DEFAULT_VOCAL_TIER = 'expressive';
"""
    (PROJECT_ROOT / "data/vocal-tiers.js").write_text(tiers_js, encoding="utf-8")

    def group_stats(rels: list[str]) -> tuple[int, int]:
        raw_sum = sum(len((PROJECT_ROOT / r).read_bytes()) for r in rels)
        concat = b"\n".join((PROJECT_ROOT / r).read_bytes() for r in rels)
        return raw_sum, len(gzip.compress(concat, 9))

    tracked = [
        "models/singer-physics.js",
        "models/mandarin-phonetics.js",
        "data/vocal-lyrics.js",
        "data/vocal-score-ultra.js",
        "data/vocal-score-expressive.js",
        "models/instrument-physics.js",
        "data/band-score.js",
        "data/band-score-ultra.js",
        "data/band-score-expressive.js",
    ]
    for rel in tracked:
        b = (PROJECT_ROOT / rel).read_bytes()
        gz = len(gzip.compress(b, 9))
        print(f"{rel:32s}: {len(b):7,d} B ({len(b)/1024:6.2f} KB) | Gzip: {gz:7,d} B ({gz/1024:6.2f} KB)")

    u_voc = ["models/singer-physics.js", "models/mandarin-phonetics.js", "data/vocal-lyrics.js", "data/vocal-score-ultra.js"]
    e_voc = ["models/singer-physics.js", "models/mandarin-phonetics.js", "data/vocal-lyrics.js", "data/vocal-score-expressive.js"]
    u_band = ["models/instrument-physics.js", "data/band-score.js", "data/band-score-ultra.js"]
    e_band = ["models/instrument-physics.js", "data/band-score.js", "data/band-score-expressive.js"]

    uv_r, uv_g = group_stats(u_voc)
    ub_r, ub_g = group_stats(u_band)
    ut_r, ut_g = group_stats(u_voc + u_band)

    ev_r, ev_g = group_stats(e_voc)
    eb_r, eb_g = group_stats(e_band)
    et_r, et_g = group_stats(e_voc + e_band)

    print("-" * 92)
    print(f"[ULTRA]      Vocal: {uv_r/1024:6.2f} KB ({uv_g/1024:6.2f} KB Gz) | Band: {ub_r/1024:6.2f} KB ({ub_g/1024:6.2f} KB Gz) | 10-Track Total: {ut_r/1024:6.2f} KB ({ut_g/1024:6.2f} KB Gz)")
    print(f"[EXPRESSIVE] Vocal: {ev_r/1024:6.2f} KB ({ev_g/1024:6.2f} KB Gz) | Band: {eb_r/1024:6.2f} KB ({eb_g/1024:6.2f} KB Gz) | 10-Track Total: {et_r/1024:6.2f} KB ({et_g/1024:6.2f} KB Gz)")


if __name__ == "__main__":
    main()
