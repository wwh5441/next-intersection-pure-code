#!/usr/bin/env python3
"""V9R1 Pure-Code Dual-Dictionary Compiler (`compile_v9r1_dictionary_engine.py`).

Rebuilds V9 cleanly from the exact V8.5 -> V9 starting point without ANY:
  - LZMA / Base122 / Base64 / binary blobs
  - Blind MDCT / FFT mix frame dumps (the V9.3/V9.4 regression)

Generates 100% human-readable ES Module JavaScript:
  1. `models/singer-core.js`               — 40-mode ERB vocal tract + 24x40 Li Yuchun PCA + 8-prime FDN
  2. `models/syllable-atlas.js`            — 97 Mandarin syllables (3+5 stage 40-ERB & 8-HNR, 0.00 dB lossless row-trimmed + initial/final phonology)
  3. `models/instrument-physics.js`        — 5-track physical synth models (Kick/Woodblock ODE, 64-shape Mel perc, 18-harm stiff Bass, 56-note Rotor Guitar/Keys + 80fps Strum)
  4. `models/band-motif-atlas.js`          — 1-Bar Medoid Physical Motif Dictionary (18 Drum + 24 Bass + 24 Guitar/Keys motifs @ exact 2.450s bar grid)
  5. `data/song-structure.js`              — 97.9592 BPM 86-bar structural & harmonic timeline
  6. `data/vocal-lyrics.js`                — Time-aligned lyric lines
  7. `data/vocal-score-ultra.js`           — 285-event 10-PCA executable vocal score
  8. `data/band-arrangement-ultra.js`      — 86-bar Motif Arrangement Schedule + top turnaround fills
  9. `data/vocal-score-expressive.js`      — 285-event 24-PCA studio micro-expression vocal score
 10. `data/band-arrangement-expressive.js` — 86-bar Motif Schedule + sparse PWL bar-delta residuals (r > 0.99)
 11. `data/tiers-registry.js`              — Dual-tier (`ultra` & `expressive`) registry
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
MANDARIN_INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
]


def split_mandarin_pinyin(py: str) -> tuple[str, str]:
    for ini in MANDARIN_INITIALS:
        if py.startswith(ini):
            return ini, py[len(ini) :]
    return "Ø", py


def load_v8_pkt(rel_path: str) -> bytes:
    txt = (V8_ROOT / rel_path).read_text(encoding="utf-8")
    b122 = txt[txt.index('"') + 1 : txt.rindex('"')]
    return decompress_lzma1_alone(decode_base122(b122))


def load_v5_accomp_pkt(rel_path: str) -> bytes:
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


def trim_trailing_zeros(arr: list[int]) -> list[int]:
    end = len(arr)
    while end > 0 and arr[end - 1] == 0:
        end -= 1
    return arr[:end]


def delta_1d_trimmed(arr_int: np.ndarray) -> list[int]:
    if len(arr_int) == 0:
        return []
    out = [int(arr_int[0])]
    for i in range(1, len(arr_int)):
        out.append(int(arr_int[i]) - int(arr_int[i - 1]))
    return trim_trailing_zeros(out)


def row_delta_trimmed_2d(mat_int: np.ndarray) -> list[list[int]]:
    out = []
    for r in range(mat_int.shape[0]):
        row = mat_int[0] if r == 0 else (mat_int[r] - mat_int[r - 1])
        out.append(trim_trailing_zeros([int(x) for x in row]))
    while len(out) > 0 and len(out[-1]) == 0:
        out.pop()
    return out


def rdp_indices(y: np.ndarray, tol: float) -> np.ndarray:
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


def encode_pwl(y: np.ndarray, min_gate: int, div: int, tol_q: float) -> list:
    ys = np.maximum(0, y.astype(np.int32) - min_gate)
    if np.all(ys == 0):
        return []
    idx = rdp_indices(ys, tol_q)
    vq = np.round(ys[idx] / div).astype(np.int32)
    if np.all(vq == 0):
        return []
    dt = np.diff(idx, prepend=0).astype(np.int32).tolist()
    dv = np.diff(vq, prepend=0).astype(np.int32).tolist()
    return [dt, dv]


def decode_pwl(pair: list, n_frames: int, div: float, gate: float) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.float64)
    if not pair or len(pair) < 2 or len(pair[0]) == 0:
        return out
    dts, dvs = pair[0], pair[1]
    thresh = 0.5 * div
    idx0 = int(dts[0])
    vq = int(dvs[0])
    val0 = vq * div
    if 0 <= idx0 < n_frames:
        out[idx0] = (val0 + gate if val0 > thresh else 0.0) if gate > 0 else val0
    for m in range(1, len(dts)):
        idx1 = idx0 + int(dts[m])
        vq += int(dvs[m])
        val1 = vq * div
        span = idx1 - idx0
        if span > 0:
            s_end = min(idx1, n_frames - 1)
            for s in range(idx0 + 1, s_end + 1):
                v = val0 + (val1 - val0) * ((s - idx0) / span)
                out[s] = (v + gate if v > thresh else 0.0) if gate > 0 else v
        idx0, val0 = idx1, val1
    return out


def encode_signed_pwl(res: np.ndarray, div: int, tol_q: float, min_peak: float) -> list:
    if np.max(np.abs(res)) < min_peak:
        return []
    idx = rdp_indices(res, tol_q)
    vq = np.round(res[idx] / div).astype(np.int32)
    if np.all(vq == 0):
        return []
    dt = np.diff(idx, prepend=0).astype(np.int32).tolist()
    dv = np.diff(vq, prepend=0).astype(np.int32).tolist()
    return [dt, dv]


def decode_signed_pwl(pair: list, n_frames: int, div: float) -> np.ndarray:
    return decode_pwl(pair, n_frames, div, 0.0)


def encode_rle_pair(arr: np.ndarray, active_mask: np.ndarray) -> list:
    clean = arr.copy()
    nz = np.nonzero(active_mask)[0]
    if len(nz) == 0:
        return []
    last_v = int(clean[nz[0]])
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


def adaptive_expr_knot_positions(total_frames: int, cons_frames: int, expr_step: int) -> np.ndarray:
    if total_frames <= 1:
        return np.array([0], dtype=np.int32)
    last = total_frames - 1
    vot = min(last, max(1, cons_frames))
    step = max(1, expr_step)
    half_s = max(2, step // 2)
    glide = min(last, vot + half_s)
    coda = max(glide, last - half_s)
    pts = {0, vot // 2, vot, glide, coda, last}
    cur = glide + step
    while cur < last:
        pts.add(cur)
        cur += step
    return np.array(sorted(pts), dtype=np.int32)


def select_medoids_pam(
    X: np.ndarray,
    k_active: int,
    allow_gain: bool = True,
    must_include: list[int] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    n_bars = X.shape[0]
    norms = np.linalg.norm(X, axis=1)
    chosen = list(must_include or [])

    def eval_assign(med_list: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not med_list:
            return np.zeros(n_bars, dtype=np.int32), np.zeros(n_bars, dtype=np.int32), norms ** 2
        M = X[med_list]
        m_norm2 = np.sum(M * M, axis=1) + 1e-12
        dots = X @ M.T
        if allow_gain:
            gains = np.clip(dots / m_norm2[None, :], 0.60, 1.45)
            gq = np.round(gains * 64.0).astype(np.int32)
            ge = gq / 64.0
        else:
            gq = np.full_like(dots, 64, dtype=np.int32)
            ge = np.ones_like(dots)
        err_mat = (norms[:, None] ** 2) - 2.0 * ge * dots + (ge ** 2) * m_norm2[None, :]
        best_m = np.argmin(err_mat, axis=1)
        best_err = err_mat[np.arange(n_bars), best_m]
        best_gq = gq[np.arange(n_bars), best_m]
        use_zero = norms ** 2 <= best_err * 1.03
        mid = np.where(use_zero, 0, best_m + 1)
        gqo = np.where(use_zero, 0, best_gq)
        ferr = np.where(use_zero, norms ** 2, best_err)
        return mid, gqo, ferr

    cands = [i for i in range(n_bars) if norms[i] > 1e-3]
    while len(chosen) < k_active:
        _, _, cur_err = eval_assign(chosen)
        best_c, best_tot = -1, float(np.sum(cur_err))
        for c in cands:
            if c in chosen:
                continue
            _, _, ce = eval_assign(chosen + [c])
            tot = float(np.sum(ce))
            if tot < best_tot:
                best_tot, best_c = tot, c
        if best_c < 0:
            break
        chosen.append(best_c)

    locked = set(must_include or [])
    for _ in range(3):
        _, _, cur_err = eval_assign(chosen)
        cur_tot = float(np.sum(cur_err))
        improved = False
        for slot in range(len(chosen)):
            if chosen[slot] in locked:
                continue
            for c in cands:
                if c in chosen:
                    continue
                trial = chosen.copy()
                trial[slot] = c
                _, _, te = eval_assign(trial)
                tt = float(np.sum(te))
                if tt < cur_tot - 1e-6:
                    chosen, cur_tot, improved = trial, tt, True
        if not improved:
            break

    chosen_sorted = sorted(chosen)
    mid, gqo, ferr = eval_assign(chosen_sorted)
    return chosen_sorted, mid, gqo, ferr


def bar_section_tag(bar_idx: int) -> str:
    if bar_idx == 0:
        return "pickup"
    if bar_idx < 8:
        return "intro_hook"
    if bar_idx < 16:
        return "verse_1"
    if bar_idx < 24:
        return "pre_chorus_1"
    if bar_idx < 32:
        return "chorus_1"
    if bar_idx < 40:
        return "interlude"
    if bar_idx == 40:
        return "verse_2a"
    if bar_idx < 44:
        return "verse_2_drop"
    if bar_idx < 48:
        return "verse_2b"
    if bar_idx < 56:
        return "pre_chorus_2"
    if bar_idx < 64:
        return "chorus_2"
    if bar_idx < 72:
        return "chorus_3"
    if bar_idx < 80:
        return "outro_groove"
    return "final_cadence"


def gain_scale_q(q_arr: np.ndarray, gq: int, gate: float) -> np.ndarray:
    """Applies linear amplitude multiplier `g = gq / 64` in `3.2 steps/dB` log-RMS space."""
    if gq <= 0:
        return np.zeros_like(q_arr)
    dq = 64.0 * np.log10(max(0.1, gq / 64.0))
    return np.where(q_arr > gate * 0.5, np.clip(q_arr + dq, 0.0, 255.0), 0.0)


def build_vocal_score_js(
    tier_id: str,
    const_suffix: str,
    short_label: str,
    full_label: str,
    events: list,
    ev_meta: dict,
    protos: list,
    target_expr_step: int | None = None,
    dz_timbre_col: int = 99,
    dz_aper_col: int = 99,
) -> str:
    k_pca = int(ev_meta["pcaDims"])
    k_aper = int(ev_meta["aperExprBands"])
    orig_expr_step = int(ev_meta["exprKnotStepFrames"])
    out_expr_step = target_expr_step if target_expr_step is not None else orig_expr_step

    profile_obj = {
        "id": tier_id,
        "shortLabel": short_label,
        "fullLabel": full_label,
        "pcaDims": k_pca,
        "aperExprBands": k_aper,
        "exprKnotStepFrames": out_expr_step,
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
        f0_steps = [] if unv_only else trim_trailing_zeros([int(x) for x in ev.f0_step_q])
        d_gh = delta_1d_trimmed(ev.gain_h_db_q.astype(int))
        d_ga = delta_1d_trimmed(ev.gain_a_db_q.astype(int))

        if ev.expr_mcep_q.size and L > 0:
            em_raw = ev.expr_mcep_q[:, :k_pca].astype(np.float64)
            if out_expr_step != orig_expr_step and L > 1:
                pos_in = adaptive_expr_knot_positions(L, cf, orig_expr_step)
                pos_out = adaptive_expr_knot_positions(L, cf, out_expr_step)
                xs = np.arange(L, dtype=np.float64)
                full_m = np.column_stack([np.interp(xs, pos_in, em_raw[:, m]) for m in range(k_pca)])
                em_q = np.round(full_m[pos_out]).astype(int)
            else:
                em_q = np.round(em_raw).astype(int)
            if dz_timbre_col < k_pca:
                em_q[:, dz_timbre_col:] = np.where(np.abs(em_q[:, dz_timbre_col:]) <= 1, 0, em_q[:, dz_timbre_col:])
            d_em = row_delta_trimmed_2d(em_q)
        else:
            d_em = []

        if ev.expr_aper_q.size and L > 0:
            ea_raw = ev.expr_aper_q[:, :k_aper].astype(np.float64)
            if out_expr_step != orig_expr_step and L > 1:
                pos_in = adaptive_expr_knot_positions(L, cf, orig_expr_step)
                pos_out = adaptive_expr_knot_positions(L, cf, out_expr_step)
                xs = np.arange(L, dtype=np.float64)
                full_a = np.column_stack([np.interp(xs, pos_in, ea_raw[:, m]) for m in range(k_aper)])
                ea_q = np.round(full_a[pos_out]).astype(int)
            else:
                ea_q = np.round(ea_raw).astype(int)
            if dz_aper_col < k_aper:
                ea_q[:, dz_aper_col:] = np.where(np.abs(ea_q[:, dz_aper_col:]) <= 1, 0, ea_q[:, dz_aper_col:])
            d_ea = row_delta_trimmed_2d(ea_q)
        else:
            d_ea = []

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
        f"// V9R1 Pure-Code Executable Vocal Score ({full_label})\n"
        f"// 285 Coarticulated Mandarin Syllable Events — Zero LZMA / Zero Base122 / Zero Binary\n"
        f"// Tuple: [startFrame, totalFrames, consFrames, hanzi, pinyin, flags, panQ, back[L,R,semiQ]|0, baseMidiQ, f0Steps[], dGainHQ[], dVdegQ[], dExprTimbre{k_pca}[], dExprAper{k_aper}[]]\n\n"
        f"export const VOCAL_PROFILE_{const_suffix} = {json.dumps(profile_obj, ensure_ascii=False, indent=2)};\n\n"
        f"export const VOCAL_SCORE_{const_suffix} = [\n"
        + ",\n".join(note_lines)
        + "\n];\n"
    )


def build_band_dictionary_and_arrangements() -> tuple[str, str, str, str, str, dict]:
    accomp_models = json.loads((V8_ROOT / "data/accomp_v5_compact/accomp_models.json").read_text(encoding="utf-8"))
    raw_b = load_v5_accomp_pkt("data/accomp_v5_compact/bass-wave.js")
    raw_d = load_v5_accomp_pkt("data/accomp_v5_compact/drums-wave.js")
    raw_h = load_v5_accomp_pkt("data/accomp_v5_compact/harmony-wave.js")

    # 1. Unpack Bass (`12 kHz`, `hop=240`, `50 fps`)
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

    # 2. Unpack Drums (`24 kHz`, `hop=240`, `100 fps`)
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
    body_q = (np.cumsum(np.frombuffer(raw_d, dtype=np.uint8, count=nf_d, offset=off + nf_d), dtype=np.int64) & 0xFF).astype(np.int32)
    high_q = (np.cumsum(np.frombuffer(raw_d, dtype=np.uint8, count=nf_d, offset=off + 2 * nf_d), dtype=np.int64) & 0xFF).astype(np.int32)

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

    # 3. Unpack Harmony (`24 kHz`, `hop=600` = `40 fps` chords; `hopS=300` = `80 fps` guitar strums)
    nf_h, n_notes_h, top_v_h, ql_h, midi_min_h, max_amp_h = struct.unpack_from("<IBBBBf", raw_h, 4)
    off = 16
    note_amp_q = np.zeros((n_notes_h, nf_h), dtype=np.int32)
    for n in range(n_notes_h):
        note_amp_q[n] = np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off + n * nf_h), dtype=np.int64) & 0xFF
    off += n_notes_h * nf_h
    bright_q = np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off).astype(np.int32)
    harm_q = (np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off + nf_h), dtype=np.int64) & 0xFF).astype(np.int32)
    ns_h = nf_h * 2
    strum_q = np.zeros((4, ns_h), dtype=np.int32)
    for b in range(4):
        strum_q[b] = np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=ns_h, offset=off + 2 * nf_h + b * ns_h), dtype=np.int64) & 0xFF
    side_q = (np.cumsum(np.frombuffer(raw_h, dtype=np.uint8, count=nf_h, offset=off + 2 * nf_h + 4 * ns_h), dtype=np.int64) & 0xFF).astype(np.int32)

    # 4. Build `models/instrument-physics.js`
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
    def to_compact_js(val, indent: int = 0) -> str:
        sp = "  " * indent
        if isinstance(val, dict):
            lines = ["{"]
            items = list(val.items())
            for idx, (k, v) in enumerate(items):
                comma = "," if idx + 1 < len(items) else ""
                lines.append(f"{sp}  {k}: {to_compact_js(v, indent + 1)}{comma}")
            lines.append(f"{sp}}}")
            return "\n".join(lines)
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
            rows = [f"{sp}  {json.dumps(r, ensure_ascii=False, separators=(',', ':'))}" for r in val]
            return "[\n" + ",\n".join(rows) + f"\n{sp}]"
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))

    inst_js = (
        "// V9R1 Pure Mathematical 5-Track Instrumental Physical & Spectral Equations Constants\n"
        "// Zero LZMA / Zero Base122 / Zero Waveforms — Shared across Ultra & Expressive Tiers\n\n"
        "export const INSTRUMENT_PHYSICS = "
        + to_compact_js(inst_obj)
        + ";\n"
    )

    # 5. Slice exact 86-Bar Commensurable Grid (`1 bar = 2.450000s @ 97.9592 BPM`)
    N_BARS = 86

    def pad_1d(a: np.ndarray, total: int) -> np.ndarray:
        return np.pad(a, (0, max(0, total - len(a))))[:total]

    kick_bars = pad_1d(kick_q, N_BARS * 245).reshape(N_BARS, 245)
    body_bars = pad_1d(body_q, N_BARS * 245).reshape(N_BARS, 245)
    high_bars = pad_1d(high_q, N_BARS * 245).reshape(N_BARS, 245)
    shape_bars = pad_1d(shape_full, N_BARS * 245).reshape(N_BARS, 245)
    act_d_bars = pad_1d(active_d.astype(np.int32), N_BARS * 245).reshape(N_BARS, 245) > 0

    rms_b_pad = pad_1d(rms_b, 43 * 245)
    f0_b_held = hold_f0_across_silence(pad_1d(f0_b, 43 * 245), rms_b_pad >= 14)
    nr_b_pad = pad_1d(nr_b, 43 * 245)
    tone_b_pad = pad_1d(tone_b, 43 * 245)

    bass_rms_bars = np.zeros((N_BARS, 123), dtype=np.int32)
    bass_f0_bars = np.zeros((N_BARS, 123), dtype=np.int32)
    bass_nr_bars = np.zeros((N_BARS, 123), dtype=np.int32)
    bass_tone_bars = np.zeros((N_BARS, 123), dtype=np.int32)
    for b in range(N_BARS):
        s0 = (b // 2) * 245 + (0 if b % 2 == 0 else 122)
        ln = 122 if b % 2 == 0 else 123
        for src_a, dst_a in [
            (rms_b_pad, bass_rms_bars),
            (f0_b_held, bass_f0_bars),
            (nr_b_pad, bass_nr_bars),
            (tone_b_pad, bass_tone_bars),
        ]:
            dst_a[b, :ln] = src_a[s0 : s0 + ln]
            if ln == 122:
                dst_a[b, 122] = dst_a[b, 121]

    na_bars = np.pad(note_amp_q, ((0, 0), (0, N_BARS * 98 - nf_h)))[:, : N_BARS * 98].reshape(n_notes_h, N_BARS, 98).transpose(1, 0, 2)
    harm_bars = pad_1d(harm_q, N_BARS * 98).reshape(N_BARS, 98)
    bright_bars = pad_1d(bright_q, N_BARS * 98).reshape(N_BARS, 98)
    side_bars = pad_1d(side_q, N_BARS * 98).reshape(N_BARS, 98)
    strum_bars = np.pad(strum_q, ((0, 0), (0, N_BARS * 196 - ns_h)))[:, : N_BARS * 196].reshape(4, N_BARS, 196).transpose(1, 0, 2)

    # 6. PAM Perceptual Medoid Motif Clustering (`18D + 22B + 22H`)
    feat_d = np.concatenate(
        [
            np.maximum(0, kick_bars - 18) ** 1.35,
            np.maximum(0, body_bars - 18) ** 1.35,
            0.85 * (np.maximum(0, high_bars - 18) ** 1.35),
        ],
        axis=1,
    ).astype(np.float64)
    med_d, id_d, gq_d, err_d = select_medoids_pam(feat_d, 18, allow_gain=True, must_include=[0, 1, 8, 24, 41])

    bass_amp = np.where(bass_rms_bars > 14, 10.0 ** ((bass_rms_bars / 3.2 - 75.0) / 20.0), 0.0)
    feat_b = np.concatenate([bass_amp ** 0.7, (bass_amp ** 0.7) * (bass_f0_bars / 600.0)], axis=1)
    med_b, id_b, gq_b, err_b = select_medoids_pam(feat_b, 22, allow_gain=True, must_include=[0, 1, 8, 24, 41])

    harm_lin = np.where(harm_bars > 14, 10.0 ** ((harm_bars / 3.2 - 75.0) / 20.0), 0.0)
    note_phys = ((na_bars / float(ql_h)) ** 2) * harm_lin[:, None, :]
    strum_lin = np.where(strum_bars > 20, 10.0 ** ((strum_bars / 3.2 - 78.0) / 20.0), 0.0)
    feat_h = np.concatenate([(note_phys.reshape(N_BARS, -1) ** 0.75), 0.7 * (strum_lin.reshape(N_BARS, -1) ** 0.75)], axis=1)
    med_h, id_h, gq_h, err_h = select_medoids_pam(feat_h, 22, allow_gain=True, must_include=[1, 2, 3, 4, 24, 25, 80, 81])

    drum_motifs = []
    for m_idx, src in enumerate(med_d, start=1):
        drum_motifs.append(
            {
                "id": m_idx,
                "srcBar": int(src),
                "tag": bar_section_tag(int(src)),
                "kick": encode_pwl(kick_bars[src], 18, 3, 4.2),
                "body": encode_pwl(body_bars[src], 18, 3, 4.2),
                "high": encode_pwl(high_bars[src], 18, 3, 4.5),
                "shape": encode_rle_pair(shape_bars[src, ::3], act_d_bars[src, ::3]),
            }
        )

    bass_motifs = []
    for m_idx, src in enumerate(med_b, start=1):
        bass_motifs.append(
            {
                "id": m_idx,
                "srcBar": int(src),
                "tag": bar_section_tag(int(src)),
                "rms": encode_pwl(bass_rms_bars[src], 14, 3, 3.8),
                "f0": encode_pwl(bass_f0_bars[src], 0, 2, 3.0),
                "nr": encode_pwl(bass_nr_bars[src], 6, 8, 9.5),
                "tone": encode_rle_pair(bass_tone_bars[src, ::2], (bass_rms_bars[src] >= 14)[::2]),
            }
        )

    harm_motifs = []
    for m_idx, src in enumerate(med_h, start=1):
        na_m = na_bars[src].copy()
        part = np.argsort(na_m, axis=0)
        for c in range(98):
            na_m[part[:-7, c], c] = 0
        notes_sparse = {}
        for n in range(n_notes_h):
            pwl = encode_pwl(na_m[n], 11, 3, 3.4)
            if pwl:
                notes_sparse[str(n)] = pwl
        harm_motifs.append(
            {
                "id": m_idx,
                "srcBar": int(src),
                "tag": bar_section_tag(int(src)),
                "harm": encode_pwl(harm_bars[src], 14, 3, 4.0),
                "bright": encode_pwl(bright_bars[src], 0, 4, 8.0),
                "side": encode_pwl(side_bars[src], 0, 4, 8.0),
                "strum": [encode_pwl(strum_bars[src, sb], 24, 4, 5.2) for sb in range(4)],
                "notes": notes_sparse,
            }
        )

    atlas_obj = {
        "barDurationSec": 2.45,
        "bpm": 97.9592,
        "totalBars": N_BARS,
        "totalFrames": {"drums": int(nf_d), "bass": int(nf_b), "harmony": int(nf_h), "strum": int(ns_h)},
        "drums": {
            "barFrames": 245,
            "gate": 18,
            "div": 3,
            "resDiv": 3,
            "shapeStride": 3,
            "motifs": drum_motifs,
        },
        "bass": {
            "barFramesEven": 122,
            "barFramesOdd": 123,
            "rmsGate": 14,
            "rmsDiv": 3,
            "f0Div": 2,
            "nrGate": 6,
            "nrDiv": 8,
            "toneStride": 2,
            "resRmsDiv": 3,
            "resF0Div": 2,
            "motifs": bass_motifs,
        },
        "harmony": {
            "barFrames": 98,
            "strumFrames": 196,
            "qLevels": int(ql_h),
            "maxNoteAmp": round(float(max_amp_h), 6),
            "envGate": 14,
            "envDiv": 3,
            "noteGate": 11,
            "noteDiv": 3,
            "strumGate": 24,
            "strumDiv": 4,
            "resDiv": 3,
            "motifs": harm_motifs,
        },
    }

    def format_motif_list(m_list: list[dict]) -> str:
        return "[\n" + ",\n".join("      " + json.dumps(m, ensure_ascii=False, separators=(",", ":")) for m in m_list) + "\n    ]"

    atlas_js = (
        "// V9R1 Pure-Code 1-Bar Medoid Physical Motif Dictionary (`BAND_MOTIF_ATLAS`)\n"
        "// 18 Drum Groove Motifs + 22 Finger-Bass Slide Motifs + 22 Polyphonic Guitar/Keys Motifs\n"
        "// Motif ID 0 = Silent Bar; Motif IDs 1..K index `motifs[id - 1]` scaled by `gainQ / 64.0`\n\n"
        "export const BAND_MOTIF_ATLAS = {\n"
        f"  barDurationSec: 2.45,\n"
        f"  bpm: 97.9592,\n"
        f"  totalBars: {N_BARS},\n"
        f"  totalFrames: {json.dumps(atlas_obj['totalFrames'], separators=(',', ':'))},\n"
        "  drums: {\n"
        "    barFrames: 245, gate: 18, div: 3, resDiv: 3, shapeStride: 3,\n"
        f"    motifs: {format_motif_list(drum_motifs)}\n"
        "  },\n"
        "  bass: {\n"
        "    barFramesEven: 122, barFramesOdd: 123, rmsGate: 14, rmsDiv: 3, f0Div: 2, nrGate: 6, nrDiv: 8, toneStride: 2, resRmsDiv: 3, resF0Div: 2,\n"
        f"    motifs: {format_motif_list(bass_motifs)}\n"
        "  },\n"
        "  harmony: {\n"
        f"    barFrames: 98, strumFrames: 196, qLevels: {int(ql_h)}, maxNoteAmp: {round(float(max_amp_h), 6)},\n"
        "    envGate: 14, envDiv: 3, noteGate: 11, noteDiv: 3, strumGate: 24, strumDiv: 4, resDiv: 3,\n"
        f"    motifs: {format_motif_list(harm_motifs)}\n"
        "  }\n"
        "};\n"
    )

    # Decode motifs exactly as JS `motif-expander.js` will decode them, to compute residuals & correlations
    dec_dm = [
        (
            decode_pwl(m["kick"], 245, 3.0, 18.0),
            decode_pwl(m["body"], 245, 3.0, 18.0),
            decode_pwl(m["high"], 245, 3.0, 18.0),
        )
        for m in drum_motifs
    ]
    dec_bm = [
        (
            decode_pwl(m["rms"], 123, 3.0, 14.0),
            decode_pwl(m["f0"], 123, 2.0, 0.0),
            decode_pwl(m["nr"], 123, 8.0, 6.0),
        )
        for m in bass_motifs
    ]
    dec_hm = []
    for m in harm_motifs:
        na_dec = np.zeros((n_notes_h, 98), dtype=np.float64)
        for k_str, pwl in m["notes"].items():
            na_dec[int(k_str)] = decode_pwl(pwl, 98, 3.0, 11.0)
        st_dec = np.stack([decode_pwl(m["strum"][sb], 196, 4.0, 24.0) for sb in range(4)], axis=0)
        dec_hm.append(
            (
                decode_pwl(m["harm"], 98, 3.0, 14.0),
                decode_pwl(m["bright"], 98, 4.0, 0.0),
                decode_pwl(m["side"], 98, 4.0, 0.0),
                st_dec,
                na_dec,
            )
        )

    # Identify top 6 drum turnaround fill bars for Ultra so major section fills stay punchy even in Ultra!
    top_fill_bars_d = set(np.argsort(err_d)[-6:].tolist())

    sched_rows = []
    ultra_res = {}
    expr_res = {}

    rec_u_k = np.zeros((N_BARS, 245))
    rec_e_k = np.zeros((N_BARS, 245))
    rec_u_bd = np.zeros((N_BARS, 245))
    rec_e_bd = np.zeros((N_BARS, 245))
    rec_u_hi = np.zeros((N_BARS, 245))
    rec_e_hi = np.zeros((N_BARS, 245))
    rec_u_br = np.zeros((N_BARS, 123))
    rec_e_br = np.zeros((N_BARS, 123))
    rec_u_bf = np.zeros((N_BARS, 123))
    rec_e_bf = np.zeros((N_BARS, 123))
    rec_u_hm = np.zeros((N_BARS, 98))
    rec_e_hm = np.zeros((N_BARS, 98))
    rec_u_na = np.zeros((N_BARS, n_notes_h, 98))
    rec_e_na = np.zeros((N_BARS, n_notes_h, 98))

    for b in range(N_BARS):
        md, gd = int(id_d[b]), int(gq_d[b])
        mb, gb = int(id_b[b]), int(gq_b[b])
        mh, gh = int(id_h[b]), int(gq_h[b])
        sched_rows.append([md, gd, mb, gb, mh, gh])

        if md > 0:
            pk, pbd, phi = dec_dm[md - 1]
            pk = gain_scale_q(pk, gd, 18.0)
            pbd = gain_scale_q(pbd, gd, 18.0)
            phi = gain_scale_q(phi, gd, 18.0)
        else:
            pk = np.zeros(245)
            pbd = np.zeros(245)
            phi = np.zeros(245)

        if mb > 0:
            prms, pf0, _ = dec_bm[mb - 1]
            prms = gain_scale_q(prms, gb, 14.0)
        else:
            prms = np.zeros(123)
            pf0 = np.zeros(123)

        if mh > 0:
            pharm, _, _, pstrum, pna = dec_hm[mh - 1]
            pharm = gain_scale_q(pharm, gh, 14.0)
            pstrum_s = np.stack([gain_scale_q(pstrum[sb], gh, 22.0) for sb in range(4)], axis=0)
            pna_s = pna.copy()
        else:
            pharm = np.zeros(98)
            pstrum_s = np.zeros((4, 196))
            pna_s = np.zeros((n_notes_h, 98))

        rec_u_k[b], rec_u_bd[b], rec_u_hi[b] = pk.copy(), pbd.copy(), phi.copy()
        rec_u_br[b], rec_u_bf[b] = prms.copy(), pf0.copy()
        rec_u_hm[b], rec_u_na[b] = pharm.copy(), pna_s.copy()

        # Target gated trajectories
        tk = np.where(kick_bars[b] > 18, kick_bars[b], 0.0)
        tbd = np.where(body_bars[b] > 18, body_bars[b], 0.0)
        thi = np.where(high_bars[b] > 18, high_bars[b], 0.0)

        # Ultra sparse drum turnaround fills (only on top 6 transition bars!)
        if b in top_fill_bars_d:
            uk = encode_signed_pwl(tk - pk, 3, 6.5, 24.0)
            ubd = encode_signed_pwl(tbd - pbd, 3, 6.5, 24.0)
            uhi = encode_signed_pwl(thi - phi, 3, 7.0, 26.0)
            if uk or ubd or uhi:
                ultra_res[str(b)] = {"d": [uk, ubd, uhi]}
                rec_u_k[b] = np.clip(pk + decode_signed_pwl(uk, 245, 3.0), 0, 255)
                rec_u_bd[b] = np.clip(pbd + decode_signed_pwl(ubd, 245, 3.0), 0, 255)
                rec_u_hi[b] = np.clip(phi + decode_signed_pwl(uhi, 245, 3.0), 0, 255)

        # Expressive sparse residuals across Drums, Bass, and Guitar/Keys
        bar_e = {}
        dk = encode_signed_pwl(tk - pk, 3, 5.2, 18.0)
        dbd = encode_signed_pwl(tbd - pbd, 3, 5.2, 18.0)
        dhi = encode_signed_pwl(thi - phi, 3, 5.8, 20.0)
        if dk or dbd or dhi:
            bar_e["d"] = [dk, dbd, dhi]
        rec_e_k[b] = np.clip(pk + decode_signed_pwl(dk, 245, 3.0), 0, 255)
        rec_e_bd[b] = np.clip(pbd + decode_signed_pwl(dbd, 245, 3.0), 0, 255)
        rec_e_hi[b] = np.clip(phi + decode_signed_pwl(dhi, 245, 3.0), 0, 255)

        trms = np.where(bass_rms_bars[b] > 14, bass_rms_bars[b], 0.0)
        db_rms = encode_signed_pwl(trms - prms, 3, 4.8, 16.0)
        act_mask_b = bass_rms_bars[b] > 16
        f0_diff = np.where(act_mask_b, bass_f0_bars[b] - pf0, 0.0)
        db_f0 = encode_signed_pwl(f0_diff, 2, 4.2, 8.0)
        if db_rms or db_f0:
            bar_e["b"] = [db_rms, db_f0]
        rec_e_br[b] = np.clip(prms + decode_signed_pwl(db_rms, 123, 3.0), 0, 255)
        rec_e_bf[b] = np.where(rec_e_br[b] > 10, pf0 + decode_signed_pwl(db_f0, 123, 2.0), pf0)

        tharm = np.where(harm_bars[b] > 14, harm_bars[b], 0.0)
        dh_env = encode_signed_pwl(tharm - pharm, 3, 4.8, 16.0)
        na_target = na_bars[b].copy()
        part = np.argsort(na_target, axis=0)
        for c in range(98):
            na_target[part[:-8, c], c] = 0
        dn_map = {}
        rec_e_na[b] = pna_s.copy()
        for n in range(n_notes_h):
            t_n = np.where(na_target[n] > 11, na_target[n], 0.0)
            dn = encode_signed_pwl(t_n - pna_s[n], 3, 5.2, 18.0)
            if dn:
                dn_map[str(n)] = dn
                rec_e_na[b, n] = np.clip(pna_s[n] + decode_signed_pwl(dn, 98, 3.0), 0, 127)
        ds_list = []
        any_ds = False
        for sb in range(4):
            t_s = np.where(strum_bars[b, sb] > 24, strum_bars[b, sb], 0.0)
            ds = encode_signed_pwl(t_s - pstrum_s[sb], 4, 7.2, 26.0)
            if ds:
                any_ds = True
            ds_list.append(ds)
        rec_e_hm[b] = np.clip(pharm + decode_signed_pwl(dh_env, 98, 3.0), 0, 255)
        if dh_env or dn_map or any_ds:
            bar_e["h"] = [dh_env, dn_map, ds_list if any_ds else []]

        if bar_e:
            expr_res[str(b)] = bar_e

    def format_arrangement_js(const_suffix: str, prof_obj: dict, bars: list, residuals: dict) -> str:
        bar_lines = [f"    {json.dumps(r, separators=(',', ':'))}" for r in bars]
        res_lines = [f'    "{k}":{json.dumps(v, separators=(",", ":"))}' for k, v in residuals.items()]
        res_block = "{\n" + ",\n".join(res_lines) + "\n  }" if res_lines else "{}"
        return (
            f"// V9R1 Pure-Code 86-Bar Executable Band Arrangement ({prof_obj['label']})\n"
            f"// Bar Tuple: [drumMotifId, drumGainQ64, bassMotifId, bassGainQ64, harmMotifId, harmGainQ64]\n\n"
            f"export const BAND_PROFILE_{const_suffix} = {json.dumps(prof_obj, ensure_ascii=False, indent=2)};\n\n"
            f"export const BAND_ARRANGEMENT_{const_suffix} = {{\n"
            f"  bars: [\n" + ",\n".join(bar_lines) + "\n  ],\n"
            f"  residuals: {res_block}\n"
            f"}};\n"
        )

    ultra_prof = {
        "id": "ultra",
        "label": "极简双字典档 · 86小节原型律动调度 + 关键过门加花",
        "drumMotifs": len(drum_motifs),
        "bassMotifs": len(bass_motifs),
        "harmonyMotifs": len(harm_motifs),
        "residualBars": len(ultra_res),
    }
    expr_prof = {
        "id": "expressive",
        "label": "精修双字典+残差档 · 86小节原型字典 + 全曲扫弦/滑音/鬼音稀疏残差",
        "drumMotifs": len(drum_motifs),
        "bassMotifs": len(bass_motifs),
        "harmonyMotifs": len(harm_motifs),
        "residualBars": len(expr_res),
    }

    band_ultra_js = format_arrangement_js("ULTRA", ultra_prof, sched_rows, ultra_res)
    band_expr_js = format_arrangement_js("EXPRESSIVE", expr_prof, sched_rows, expr_res)

    song_struct_js = """// Structural 86-Bar & Harmonic Timeline for 《李宇春 - 下个, 路口, 见》 (97.9592 BPM, 2.450s/bar)
export const SONG_STRUCTURE = {
  bpm: 97.9592,
  barDurationSec: 2.45,
  bar0Sec: 1.7493,
  totalBars: 86,
  progressions: {
    STD4:  [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Dm9"]],
    ROLL4: [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Am7"]],
  },
  sections: [
    { name: "Intro",        zh: "前奏吉他Hook", startBar: 0,  bars: 8, prog: "STD4" },
    { name: "Verse 1",      zh: "主歌一",       startBar: 8,  bars: 8, prog: "STD4" },
    { name: "Pre-Chorus 1", zh: "导歌一",       startBar: 16, bars: 8, prog: "STD4" },
    { name: "Chorus 1",     zh: "副歌一",       startBar: 24, bars: 8, prog: "ROLL4", resolveLastDm: true },
    { name: "Interlude",    zh: "间奏律动",     startBar: 32, bars: 8, prog: "STD4" },
    { name: "Verse 2A",     zh: "主歌二切入",   startBar: 40, bars: 1, prog: "STD4" },
    { name: "Verse 2 Drop", zh: "主歌二抽鼓留白", startBar: 41, bars: 3, prog: "STD4", progOffset: 1 },
    { name: "Verse 2B",     zh: "主歌二推进",   startBar: 44, bars: 4, prog: "STD4" },
    { name: "Pre-Chorus 2", zh: "导歌二",       startBar: 48, bars: 8, prog: "STD4" },
    { name: "Chorus 2",     zh: "副歌二",       startBar: 56, bars: 8, prog: "ROLL4", resolveLastDm: true },
    { name: "Chorus 3",     zh: "副歌三高潮",   startBar: 64, bars: 8, prog: "ROLL4", resolveLastDm: true },
    { name: "Outro Groove", zh: "尾奏律动",     startBar: 72, bars: 8, prog: "STD4" },
    { name: "Final Chord",  zh: "收束余韵",     startBar: 80, bars: 6, prog: "FINAL" },
  ],
};
"""

    act_bu = (bass_rms_bars.ravel() > 20) & (rec_u_br.ravel() > 20)
    act_be = (bass_rms_bars.ravel() > 20) & (rec_e_br.ravel() > 20)
    u_phys = ((rec_u_na / float(ql_h)) ** 2) * np.where(rec_u_hm > 14, 10.0 ** ((rec_u_hm / 3.2 - 75.0) / 20.0), 0.0)[:, None, :]
    e_phys = ((rec_e_na / float(ql_h)) ** 2) * np.where(rec_e_hm > 14, 10.0 ** ((rec_e_hm / 3.2 - 75.0) / 20.0), 0.0)[:, None, :]

    qa_stats = {
        "ultraParameterCorrelations": {
            "drumKickR": round(float(np.corrcoef(np.maximum(0, kick_q - 18), np.maximum(0, rec_u_k.ravel()[:nf_d] - 18))[0, 1]), 4),
            "drumBodyR": round(float(np.corrcoef(np.maximum(0, body_q - 18), np.maximum(0, rec_u_bd.ravel()[:nf_d] - 18))[0, 1]), 4),
            "drumHighR": round(float(np.corrcoef(np.maximum(0, high_q - 18), np.maximum(0, rec_u_hi.ravel()[:nf_d] - 18))[0, 1]), 4),
            "bassEnvR": round(float(np.corrcoef(np.maximum(0, bass_rms_bars.ravel() - 14), np.maximum(0, rec_u_br.ravel() - 14))[0, 1]), 4),
            "bassPitchR": round(float(np.corrcoef(bass_f0_bars.ravel()[act_bu], rec_u_bf.ravel()[act_bu])[0, 1]), 4),
            "harmonyPolyR": round(float(np.corrcoef(note_phys.ravel(), u_phys.ravel())[0, 1]), 4),
        },
        "expressiveParameterCorrelations": {
            "drumKickR": round(float(np.corrcoef(np.maximum(0, kick_q - 18), np.maximum(0, rec_e_k.ravel()[:nf_d] - 18))[0, 1]), 4),
            "drumBodyR": round(float(np.corrcoef(np.maximum(0, body_q - 18), np.maximum(0, rec_e_bd.ravel()[:nf_d] - 18))[0, 1]), 4),
            "drumHighR": round(float(np.corrcoef(np.maximum(0, high_q - 18), np.maximum(0, rec_e_hi.ravel()[:nf_d] - 18))[0, 1]), 4),
            "bassEnvR": round(float(np.corrcoef(np.maximum(0, bass_rms_bars.ravel() - 14), np.maximum(0, rec_e_br.ravel() - 14))[0, 1]), 4),
            "bassPitchR": round(float(np.corrcoef(bass_f0_bars.ravel()[act_be], rec_e_bf.ravel()[act_be])[0, 1]), 4),
            "harmonyPolyR": round(float(np.corrcoef(note_phys.ravel(), e_phys.ravel())[0, 1]), 4),
        },
    }
    return inst_js, atlas_js, song_struct_js, band_ultra_js, band_expr_js, qa_stats


def main() -> None:
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    raw_spk = load_v8_pkt("data/shared/speaker_core.js")
    raw_atl = load_v8_pkt("data/shared/syllable_atlas.js")
    raw_fdn = load_v8_pkt("data/shared/space_fdn.js")
    raw_ultra = load_v8_pkt("data/ultra/event_stream.js")
    raw_expr = load_v8_pkt("data/expressive/event_stream.js")

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

    # 1. `models/singer-core.js`
    mean_mcep_40 = [round(float(x), 4) for x in spk.mean_mcep[:n_cep]]
    glottal_phase_64 = [round(float(x), 4) for x in spk.glottal_phase]
    aper_mean_8 = [round(float(x), 4) for x in spk.aper_mean[:n_aper]]
    mcep_basis_24 = np.round(spk.mcep_pca_basis[:24, :n_cep] * 2048.0).astype(int).tolist()
    aper_basis_8 = np.round(spk.aper_pca_basis[:8, :n_aper] * 2048.0).astype(int).tolist()

    singer_js = f"""// V9R1 Pure Mathematical Li Yuchun Timbre Eigen-Core (40-ERB + 24-PCA) & 8-Prime Householder FDN
// Zero LZMA / Zero Base122 / Zero Binary — Shared across Ultra (10-PCA) & Expressive (24-PCA)
export const SINGER_CORE = {{
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
    (PROJECT_ROOT / "models/singer-core.js").write_text(singer_js, encoding="utf-8")

    # 2. `models/syllable-atlas.js` (0.00 dB lossless row-trimmed + Mandarin initial/final phonology)
    syl_lines = []
    for p in protos:
        ini, fin = split_mandarin_pinyin(p.pinyin)
        cm_i = np.round(p.cons_mcep_knots / atlas_step_db).astype(int)
        ca_i = np.round(p.cons_aper_knots / atlas_step_db).astype(int)
        vm_i = np.round(p.vowel_mcep_knots / atlas_step_db).astype(int)
        va_i = np.round(p.vowel_aper_knots / atlas_step_db).astype(int)
        entry = [
            p.initial_class,
            ini,
            fin,
            row_delta_trimmed_2d(cm_i),
            row_delta_trimmed_2d(ca_i),
            row_delta_trimmed_2d(vm_i),
            row_delta_trimmed_2d(va_i),
        ]
        syl_lines.append(f'  "{p.pinyin}":{json.dumps(entry, ensure_ascii=False, separators=(",", ":"))}')

    syllable_js = (
        "// V9R1 Pure Mathematical 97-Syllable Mandarin Coarticulation Dictionary (`SYLLABLE_ATLAS`)\n"
        "// Tuple: [initialClass, initial, final, consMcep3x40[], consAper3x8[], vowelMcep5x40[], vowelAper5x8[]]\n"
        "// Trailing-zero & empty-row trimmed with 0.000000 dB reconstruction loss\n\n"
        "export const SYLLABLE_ATLAS = {\n"
        + ",\n".join(syl_lines)
        + "\n};\n"
    )
    (PROJECT_ROOT / "models/syllable-atlas.js").write_text(syllable_js, encoding="utf-8")

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

    # 4. Vocal Tier Executable Scores
    vocal_ultra_js = build_vocal_score_js(
        tier_id="ultra",
        const_suffix="ULTRA",
        short_label="极简双字典档 · 10维人声 + 86小节原型律动",
        full_label="V9R1 极简双字典档 · 97音节字典 + 10维PCA微表情 + 86小节伴奏原型字典",
        events=ev_ultra,
        ev_meta=meta_ultra,
        protos=protos,
        target_expr_step=None,
        dz_timbre_col=5,
        dz_aper_col=2,
    )
    (PROJECT_ROOT / "data/vocal-score-ultra.js").write_text(vocal_ultra_js, encoding="utf-8")

    vocal_expr_js = build_vocal_score_js(
        tier_id="expressive",
        const_suffix="EXPRESSIVE",
        short_label="精修双字典+残差档 · 24维全人声 + 全细节伴奏残差",
        full_label="V9R1 母带精修双字典+残差档 · 24维PCA微表情 + 86小节伴奏原型字典与稀疏残差",
        events=ev_expr,
        ev_meta=meta_expr,
        protos=protos,
        target_expr_step=5,
        dz_timbre_col=12,
        dz_aper_col=4,
    )
    (PROJECT_ROOT / "data/vocal-score-expressive.js").write_text(vocal_expr_js, encoding="utf-8")

    # 5. Band Physics, Motif Atlas & Dual-Tier Executable Arrangements
    inst_js, atlas_js, song_struct_js, band_ultra_js, band_expr_js, qa_stats = build_band_dictionary_and_arrangements()
    (PROJECT_ROOT / "models/instrument-physics.js").write_text(inst_js, encoding="utf-8")
    (PROJECT_ROOT / "models/band-motif-atlas.js").write_text(atlas_js, encoding="utf-8")
    (PROJECT_ROOT / "data/song-structure.js").write_text(song_struct_js, encoding="utf-8")
    (PROJECT_ROOT / "data/band-arrangement-ultra.js").write_text(band_ultra_js, encoding="utf-8")
    (PROJECT_ROOT / "data/band-arrangement-expressive.js").write_text(band_expr_js, encoding="utf-8")

    # 6. `data/tiers-registry.js`
    tiers_js = """// V9R1 Pure-Code Dual-Dictionary Tier Registry (`ultra` & `expressive`)
import { VOCAL_PROFILE_ULTRA, VOCAL_SCORE_ULTRA } from './vocal-score-ultra.js';
import { VOCAL_PROFILE_EXPRESSIVE, VOCAL_SCORE_EXPRESSIVE } from './vocal-score-expressive.js';
import { BAND_PROFILE_ULTRA, BAND_ARRANGEMENT_ULTRA } from './band-arrangement-ultra.js';
import { BAND_PROFILE_EXPRESSIVE, BAND_ARRANGEMENT_EXPRESSIVE } from './band-arrangement-expressive.js';

export const V9R1_TIERS = {
  ultra: {
    id: 'ultra',
    vocalProfile: VOCAL_PROFILE_ULTRA,
    vocalScore: VOCAL_SCORE_ULTRA,
    bandProfile: BAND_PROFILE_ULTRA,
    bandArrangement: BAND_ARRANGEMENT_ULTRA,
  },
  expressive: {
    id: 'expressive',
    vocalProfile: VOCAL_PROFILE_EXPRESSIVE,
    vocalScore: VOCAL_SCORE_EXPRESSIVE,
    bandProfile: BAND_PROFILE_EXPRESSIVE,
    bandArrangement: BAND_ARRANGEMENT_EXPRESSIVE,
  },
};

export const DEFAULT_V9R1_TIER = 'ultra';
"""
    (PROJECT_ROOT / "data/tiers-registry.js").write_text(tiers_js, encoding="utf-8")

    def group_stats(rels: list[str]) -> tuple[int, int]:
        raw_sum = sum(len((PROJECT_ROOT / r).read_bytes()) for r in rels)
        concat = b"\n".join((PROJECT_ROOT / r).read_bytes() for r in rels)
        return raw_sum, len(gzip.compress(concat, 9))

    tracked = [
        "models/singer-core.js",
        "models/syllable-atlas.js",
        "models/instrument-physics.js",
        "models/band-motif-atlas.js",
        "data/song-structure.js",
        "data/vocal-lyrics.js",
        "data/vocal-score-ultra.js",
        "data/band-arrangement-ultra.js",
        "data/vocal-score-expressive.js",
        "data/band-arrangement-expressive.js",
    ]
    for rel in tracked:
        b = (PROJECT_ROOT / rel).read_bytes()
        gz = len(gzip.compress(b, 9))
        print(f"{rel:36s}: {len(b):7,d} B ({len(b)/1024:6.2f} KB) | Gzip: {gz:7,d} B ({gz/1024:6.2f} KB)")

    shared_voc = ["models/singer-core.js", "models/syllable-atlas.js", "data/vocal-lyrics.js"]
    shared_band = ["models/instrument-physics.js", "models/band-motif-atlas.js", "data/song-structure.js"]
    u_voc = shared_voc + ["data/vocal-score-ultra.js"]
    u_band = shared_band + ["data/band-arrangement-ultra.js"]
    e_voc = shared_voc + ["data/vocal-score-expressive.js"]
    e_band = shared_band + ["data/band-arrangement-expressive.js"]
    all_dual = shared_voc + shared_band + [
        "data/vocal-score-ultra.js",
        "data/band-arrangement-ultra.js",
        "data/vocal-score-expressive.js",
        "data/band-arrangement-expressive.js",
    ]

    uv_r, uv_g = group_stats(u_voc)
    ub_r, ub_g = group_stats(u_band)
    ut_r, ut_g = group_stats(u_voc + u_band)

    ev_r, ev_g = group_stats(e_voc)
    eb_r, eb_g = group_stats(e_band)
    et_r, et_g = group_stats(e_voc + e_band)
    dt_r, dt_g = group_stats(all_dual)

    print("-" * 100)
    print(f"[V9R1 ULTRA]      Vocal: {uv_r/1024:6.2f} KB ({uv_g/1024:6.2f} KB Gz) | Band: {ub_r/1024:6.2f} KB ({ub_g/1024:6.2f} KB Gz) | 10-Track Total: {ut_r/1024:6.2f} KB ({ut_g/1024:6.2f} KB Gz)")
    print(f"[V9R1 EXPRESSIVE] Vocal: {ev_r/1024:6.2f} KB ({ev_g/1024:6.2f} KB Gz) | Band: {eb_r/1024:6.2f} KB ({eb_g/1024:6.2f} KB Gz) | 10-Track Total: {et_r/1024:6.2f} KB ({et_g/1024:6.2f} KB Gz)")
    print(f"[DUAL COMBINED]   Shared Dual-Dictionary + Both Scores: {dt_r/1024:6.2f} KB ({dt_g/1024:6.2f} KB Gz)")
    print("QA Correlations:", json.dumps(qa_stats, indent=2))


if __name__ == "__main__":
    main()
