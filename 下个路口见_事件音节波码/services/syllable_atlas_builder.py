"""Deduplicated 40-Cepstrum Syllable Atlas Builder & Coarticulated Closed-Loop DPCM Encoder (V8.4)."""
from __future__ import annotations

import math
import numpy as np

from models.vocal_structures import SpeakerTimbreCore, SyllablePrototype, VocalEvent


CONS_ANCHORS = np.linspace(0.0, 1.0, 3, dtype=np.float32)
VOWEL_ANCHORS = np.array([0.0, 0.22, 0.50, 0.78, 1.0], dtype=np.float32)

DB_FLOOR = -78.0
DB_CEIL = 4.0


def _sample_curve_2d(mat: np.ndarray, u_knots: np.ndarray) -> np.ndarray:
    L, _ = mat.shape
    if L <= 1:
        return np.repeat(mat[:1], len(u_knots), axis=0).astype(np.float32)
    idx_f = u_knots * (L - 1)
    i0 = np.clip(np.floor(idx_f).astype(np.int32), 0, L - 1)
    i1 = np.clip(i0 + 1, 0, L - 1)
    frac = (idx_f - i0)[:, None]
    return ((1.0 - frac) * mat[i0] + frac * mat[i1]).astype(np.float32)


def build_envelope_knot_positions(total_frames: int, cons_frames: int, vowel_step: int) -> np.ndarray:
    if total_frames <= 1:
        return np.array([0], dtype=np.int32)
    last = total_frames - 1
    vot = min(last, max(1, cons_frames))
    step = max(1, vowel_step)
    coda = max(vot, last - max(2, step // 2))
    pts = {0, vot // 2, vot, coda, last}
    cur = vot + step
    while cur < last:
        pts.add(int(cur))
        cur += step
    return np.array(sorted(pts), dtype=np.int32)


def build_adaptive_expr_knot_positions(total_frames: int, cons_frames: int, expr_step: int) -> np.ndarray:
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
        pts.add(int(cur))
        cur += step
    return np.array(sorted(pts), dtype=np.int32)


def build_uniform_knot_positions(total_frames: int, step: int) -> np.ndarray:
    if total_frames <= 1:
        return np.array([0], dtype=np.int32)
    last = total_frames - 1
    pts = list(range(0, last, max(1, step)))
    if pts[-1] != last:
        pts.append(last)
    return np.array(pts, dtype=np.int32)


def interp_1d_knots(knot_pos: np.ndarray, knot_vals: np.ndarray, total_frames: int) -> np.ndarray:
    if total_frames <= 1 or len(knot_pos) <= 1:
        return np.full(total_frames, float(knot_vals[0]) if len(knot_vals) else 0.0, dtype=np.float32)
    x = np.arange(total_frames, dtype=np.float32)
    return np.interp(x, knot_pos.astype(np.float32), knot_vals.astype(np.float32)).astype(np.float32)


def interp_2d_knots(knot_pos: np.ndarray, knot_mat: np.ndarray, total_frames: int) -> np.ndarray:
    D = knot_mat.shape[1]
    if total_frames <= 1 or len(knot_pos) <= 1:
        row = knot_mat[:1] if len(knot_mat) else np.zeros((1, D), dtype=np.float32)
        return np.repeat(row, total_frames, axis=0).astype(np.float32)
    x = np.arange(total_frames, dtype=np.float32)
    idx_f = np.interp(x, knot_pos.astype(np.float32), np.arange(len(knot_pos), dtype=np.float32))
    i0 = np.clip(np.floor(idx_f).astype(np.int32), 0, len(knot_pos) - 1)
    i1 = np.clip(i0 + 1, 0, len(knot_pos) - 1)
    frac = (idx_f - i0)[:, None]
    return ((1.0 - frac) * knot_mat[i0] + frac * knot_mat[i1]).astype(np.float32)


def eval_prototype_on_event(
    proto: SyllablePrototype,
    speaker: SpeakerTimbreCore,
    total_frames: int,
    cons_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    L = total_frames
    vot = min(L - 1, cons_frames) if (L > 1 and cons_frames > 0) else 0

    pred_mcep = np.zeros((L, len(speaker.mean_mcep)), dtype=np.float32)
    pred_aper = np.zeros((L, len(speaker.aper_mean)), dtype=np.float32)

    if vot > 0:
        u_c = (
            np.linspace(0.0, 1.0, vot, dtype=np.float32) * (len(CONS_ANCHORS) - 1)
            if vot > 1
            else np.zeros(1, dtype=np.float32)
        )
        ic0 = np.clip(np.floor(u_c).astype(np.int32), 0, len(CONS_ANCHORS) - 2)
        ic1 = ic0 + 1
        fc = (u_c - ic0)[:, None]
        pred_mcep[:vot] = speaker.mean_mcep[None, :] + (1.0 - fc) * proto.cons_mcep_knots[ic0] + fc * proto.cons_mcep_knots[ic1]
        pred_aper[:vot] = speaker.aper_mean[None, :] + (1.0 - fc) * proto.cons_aper_knots[ic0] + fc * proto.cons_aper_knots[ic1]

    v_len = L - vot
    if v_len > 0:
        u_v = (
            np.linspace(0.0, 1.0, v_len, dtype=np.float32) * (len(VOWEL_ANCHORS) - 1)
            if v_len > 1
            else np.zeros(1, dtype=np.float32)
        )
        iv0 = np.clip(np.floor(u_v).astype(np.int32), 0, len(VOWEL_ANCHORS) - 2)
        iv1 = iv0 + 1
        fv = (u_v - iv0)[:, None]
        pred_mcep[vot:] = speaker.mean_mcep[None, :] + (1.0 - fv) * proto.vowel_mcep_knots[iv0] + fv * proto.vowel_mcep_knots[iv1]
        pred_aper[vot:] = speaker.aper_mean[None, :] + (1.0 - fv) * proto.vowel_aper_knots[iv0] + fv * proto.vowel_aper_knots[iv1]

    return pred_mcep.astype(np.float32), pred_aper.astype(np.float32)


def build_coarticulated_prototype_grids(
    events: list,
    prototypes: list[SyllablePrototype],
    speaker: SpeakerTimbreCore,
    total_frames_song: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates syllable prototypes and applies bi-directional '前字启下 + 后字承上' raised-cosine seam coarticulation."""
    n_cep = len(speaker.mean_mcep)
    n_aper = len(speaker.aper_mean)
    base_cep = np.zeros((total_frames_song, n_cep), dtype=np.float32)
    base_aper = np.full((total_frames_song, n_aper), -3.6, dtype=np.float32)

    def _get(ev: object, key: str):
        return ev[key] if isinstance(ev, dict) else getattr(ev, key)

    for ev in events:
        L = int(_get(ev, "total_frames"))
        if L <= 0:
            continue
        s_f = int(_get(ev, "start_frame"))
        cf = int(_get(ev, "cons_frames"))
        sid = int(_get(ev, "syllable_id"))
        f_end_clip = min(total_frames_song, s_f + L)
        span = f_end_clip - s_f
        if span <= 0:
            continue
        pred_m, pred_a = eval_prototype_on_event(prototypes[sid], speaker, L, cf)
        base_cep[s_f:f_end_clip] = pred_m[:span]
        base_aper[s_f:f_end_clip] = pred_a[:span]

    seams = []
    for i in range(1, len(events)):
        prev_ev = events[i - 1]
        curr_ev = events[i]
        prev_s = int(_get(prev_ev, "start_frame"))
        prev_L = int(_get(prev_ev, "total_frames"))
        curr_s = int(_get(curr_ev, "start_frame"))
        curr_L = int(_get(curr_ev, "total_frames"))
        b = curr_s
        if prev_L >= 2 and curr_L >= 2 and (prev_s + prev_L == b) and (1 <= b < total_frames_song):
            curr_plos = bool(_get(curr_ev, "is_plosive"))
            w_pre = min(18, (prev_L * 12) // 25, b)
            w_post = min(4 if curr_plos else 10, (curr_L * 9) // 25, total_frames_song - b)
            if w_pre >= 1 and w_post >= 1:
                seams.append(
                    (
                        b,
                        w_pre,
                        w_post,
                        base_cep[b].copy() - base_cep[b - 1].copy(),
                        base_aper[b].copy() - base_aper[b - 1].copy(),
                    )
                )

    for b, w_pre, w_post, d_cep, d_aper in seams:
        inv_tot = 1.0 / float(w_pre + w_post)
        g_pre = w_pre * inv_tot
        g_post = w_post * inv_tot
        for r in range(w_pre):
            w = 0.5 * g_pre * (1.0 + math.cos(math.pi * (r + 0.5) / float(w_pre)))
            base_cep[b - 1 - r] += w * d_cep
            base_aper[b - 1 - r] += w * d_aper
        for r in range(w_post):
            w = 0.5 * g_post * (1.0 + math.cos(math.pi * (r + 0.5) / float(w_post)))
            base_cep[b + r] -= w * d_cep
            base_aper[b + r] -= w * d_aper

    return base_cep, base_aper


def gain_to_uint8(gain: np.ndarray | float, ref_peak: float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(gain, dtype=np.float32))
    ratio = np.maximum(arr / max(ref_peak, 1e-8), 0.0)
    out = np.zeros(len(arr), dtype=np.uint8)
    nz = ratio > (10.0 ** (DB_FLOOR / 20.0))
    if np.any(nz):
        db = 20.0 * np.log10(ratio[nz])
        q = np.round((np.clip(db, DB_FLOOR, DB_CEIL) - DB_FLOOR) / (DB_CEIL - DB_FLOOR) * 254.0) + 1.0
        out[nz] = np.clip(q, 1, 255).astype(np.uint8)
    return out


def uint8_to_gain(q: np.ndarray | int, ref_peak: float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(q, dtype=np.uint8))
    out = np.zeros(len(arr), dtype=np.float32)
    nz = arr > 0
    if np.any(nz):
        db = DB_FLOOR + (arr[nz].astype(np.float32) - 1.0) / 254.0 * (DB_CEIL - DB_FLOOR)
        out[nz] = (ref_peak * np.power(10.0, db / 20.0)).astype(np.float32)
    return out


def reconstruct_midi_knots(base_midi_q: int, f0_step_q: np.ndarray) -> np.ndarray:
    """Reconstructs exact MIDI pitch knots (`2-cent` anchor + `2.5-cent` closed-loop steps, ZERO clipping!)."""
    n_k = 1 + len(f0_step_q)
    out = np.zeros(n_k, dtype=np.float32)
    cur = base_midi_q * 0.02
    out[0] = cur
    for k in range(len(f0_step_q)):
        cur += float(f0_step_q[k]) * 0.025
        out[k + 1] = cur
    return out


def build_speaker_and_syllable_atlas(
    raw_events: list[dict],
    features: dict,
    anatomy: dict,
    config: dict,
) -> tuple[SpeakerTimbreCore, list[SyllablePrototype], list[dict], float, dict]:
    """Learns SpeakerTimbreCore, Unified 40-Cepstrum Syllable Atlas, and Non-Clipping Scaled SVD Bases."""
    mcep_frames = features["mcep_frames"]
    aper_frames = features["aper_log_shape"]
    tot_gain = features["total_gain"]
    voiced = features["voiced"]
    n_cep = int(config["cepstralCoeffs"])
    n_aper = int(config["aperiodicBands"])

    max_k_mcep = max(int(p["pcaDims"]) for p in config["profiles"].values())
    max_k_aper = max(int(p["aperExprBands"]) for p in config["profiles"].values())
    min_q_step = min(float(p["quantStepDb"]) for p in config["profiles"].values())

    ref_peak = float(max(np.max(tot_gain), 1e-6))

    v_mask = voiced & (tot_gain > ref_peak * 0.015)
    w_v = (tot_gain[v_mask] ** 1.35)[:, None]
    mean_mcep = (np.sum(mcep_frames[v_mask] * w_v, axis=0) / (np.sum(w_v) + 1e-12)).astype(np.float32)

    a_mask = features["active_mask"] & (tot_gain > ref_peak * 0.008)
    w_a = (tot_gain[a_mask] ** 1.15)[:, None]
    aper_mean = (np.sum(aper_frames[a_mask] * w_a, axis=0) / (np.sum(w_a) + 1e-12)).astype(np.float32)

    lyric_groups: dict[str, list[dict]] = {}
    for ev in raw_events:
        if not ev["is_hook"]:
            lyric_groups.setdefault(ev["pinyin"], []).append(ev)

    sorted_pinyins = sorted(lyric_groups.keys())
    prototypes: list[SyllablePrototype] = []
    pinyin_to_id: dict[str, int] = {}

    for sid, py in enumerate(sorted_pinyins):
        evs = lyric_groups[py]
        cons_m_acc = np.zeros((len(CONS_ANCHORS), n_cep), dtype=np.float64)
        cons_a_acc = np.zeros((len(CONS_ANCHORS), n_aper), dtype=np.float64)
        cons_w = 0.0
        v_mcep_acc = np.zeros((len(VOWEL_ANCHORS), n_cep), dtype=np.float64)
        v_aper_acc = np.zeros((len(VOWEL_ANCHORS), n_aper), dtype=np.float64)
        v_w = 0.0

        for ev in evs:
            s_f, e_f = ev["start_frame"], ev["end_frame"]
            cf_raw = int(ev["cons_frames"])
            if cf_raw > 0 and (e_f - s_f) > 1:
                vot = min(e_f - s_f - 1, cf_raw)
                cm_slice = mcep_frames[s_f : s_f + vot] - mean_mcep[None, :]
                ca_slice = aper_frames[s_f : s_f + vot] - aper_mean[None, :]
                cw = float(np.mean(tot_gain[s_f : s_f + vot]) + 1e-5)
                cons_m_acc += cw * _sample_curve_2d(cm_slice, CONS_ANCHORS)
                cons_a_acc += cw * _sample_curve_2d(ca_slice, CONS_ANCHORS)
                cons_w += cw
            else:
                vot = 0

            v_start = s_f + vot if (s_f + vot) < e_f else s_f
            vm_slice = mcep_frames[v_start:e_f] - mean_mcep[None, :]
            va_slice = aper_frames[v_start:e_f] - aper_mean[None, :]
            vw = float(np.mean(tot_gain[v_start:e_f]) + 1e-5)
            v_mcep_acc += vw * _sample_curve_2d(vm_slice, VOWEL_ANCHORS)
            v_aper_acc += vw * _sample_curve_2d(va_slice, VOWEL_ANCHORS)
            v_w += vw

        v_m_knots = (v_mcep_acc / max(v_w, 1e-9)).astype(np.float32)
        v_a_knots = (v_aper_acc / max(v_w, 1e-9)).astype(np.float32)
        if cons_w > 1e-9:
            c_m_knots = (cons_m_acc / cons_w).astype(np.float32)
            c_a_knots = (cons_a_acc / cons_w).astype(np.float32)
        else:
            c_m_knots = np.repeat(v_m_knots[:1], len(CONS_ANCHORS), axis=0).astype(np.float32)
            c_a_knots = np.repeat(v_a_knots[:1], len(CONS_ANCHORS), axis=0).astype(np.float32)

        proto = SyllablePrototype(
            syllable_id=sid,
            pinyin=py,
            initial_class=evs[0]["initial_class"],
            count=len(evs),
            cons_mcep_knots=c_m_knots,
            cons_aper_knots=c_a_knots,
            vowel_mcep_knots=v_m_knots,
            vowel_aper_knots=v_a_knots,
        )
        prototypes.append(proto)
        pinyin_to_id[py] = sid

    proto_sigs = np.array(
        [
            np.concatenate([p.vowel_mcep_knots.mean(axis=0), 0.35 * p.cons_mcep_knots.mean(axis=0)])
            for p in prototypes
        ],
        dtype=np.float32,
    )
    onset_rise_db = features["onset_rise_db"]
    for ev in raw_events:
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        if ev["is_hook"]:
            vot = min(e_f - s_f - 1, max(1, ev["cons_frames"])) if (e_f - s_f) > 1 else 1
            v_start = s_f + vot if (s_f + vot) < e_f else s_f
            sig = np.concatenate(
                [
                    (mcep_frames[v_start:e_f] - mean_mcep[None, :]).mean(axis=0),
                    0.35 * (mcep_frames[s_f : s_f + vot] - mean_mcep[None, :]).mean(axis=0),
                ]
            )
            dists = np.sum((proto_sigs - sig[None, :]) ** 2, axis=1)
            best_sid = int(np.argmin(dists))
            ev["syllable_id"] = best_sid
            ev["pinyin"] = prototypes[best_sid].pinyin
            ev["display_text"] = f"♪{prototypes[best_sid].pinyin}"
            prototypes[best_sid].count += 1
        else:
            ev["syllable_id"] = pinyin_to_id[ev["pinyin"]]

        proto = prototypes[ev["syllable_id"]]
        cons_end = min(e_f, s_f + max(2, ev["cons_frames"]))
        max_rise = float(np.max(onset_rise_db[s_f:cons_end])) if cons_end > s_f else 0.0
        ev["is_plosive"] = bool(proto.initial_class == "plosive" or (ev["is_hook"] and max_rise > 7.5))

    temp_speaker = SpeakerTimbreCore(
        mean_mcep=mean_mcep,
        glottal_phase=features["glottal_phase"],
        mcep_pca_basis=np.eye(n_cep, dtype=np.float32),
        aper_mean=aper_mean,
        aper_pca_basis=np.eye(n_aper, dtype=np.float32),
    )

    total_frames_song = len(mcep_frames) + 6
    base_cep_grid, base_aper_grid = build_coarticulated_prototype_grids(
        raw_events, prototypes, temp_speaker, total_frames_song
    )

    mcep_res_weighted = []
    mcep_res_unweighted = []
    aper_res_unweighted = []
    tot_var_raw = 0.0
    tot_var_after_speaker = 0.0
    tot_var_after_atlas = 0.0

    for ev in raw_events:
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        actual_m = mcep_frames[s_f:e_f]
        actual_a = aper_frames[s_f:e_f]
        res_m = actual_m - base_cep_grid[s_f:e_f]
        res_a = actual_a - base_aper_grid[s_f:e_f]

        w_sub = np.clip(tot_gain[s_f:e_f] / ref_peak, 0.02, 1.0)[:, None]
        mcep_res_weighted.append(res_m * np.sqrt(w_sub))
        mcep_res_unweighted.append(res_m)
        aper_res_unweighted.append(res_a)

        tot_var_raw += float(np.sum((actual_m ** 2) * w_sub))
        tot_var_after_speaker += float(np.sum(((actual_m - mean_mcep[None, :]) ** 2) * w_sub))
        tot_var_after_atlas += float(np.sum((res_m ** 2) * w_sub))

    res_m_w = np.vstack(mcep_res_weighted)
    res_m_all = np.vstack(mcep_res_unweighted)
    res_a_all = np.vstack(aper_res_unweighted)

    _, _, vt_m = np.linalg.svd(res_m_w, full_matrices=False)
    _, _, vt_a = np.linalg.svd(res_a_all, full_matrices=False)

    k_ult = int(config["profiles"]["ultra"]["pcaDims"])
    k_exp = int(config["profiles"]["expressive"]["pcaDims"])
    proj_u = (res_m_w @ vt_m[:k_ult].T) @ vt_m[:k_ult]
    var_after_u = float(np.sum((res_m_w - proj_u) ** 2))
    proj_e = (res_m_w @ vt_m[:k_exp].T) @ vt_m[:k_exp]
    var_after_e = float(np.sum((res_m_w - proj_e) ** 2))

    raw_proj_m = res_m_all @ vt_m[:max_k_mcep].T
    scales_m = np.ones(max_k_mcep, dtype=np.float32)
    for m in range(max_k_mcep):
        p998 = float(np.percentile(np.abs(raw_proj_m[:, m]), 99.8))
        scales_m[m] = max(1.0, p998 / (116.0 * min_q_step))
    scaled_basis_m = (vt_m[:max_k_mcep] * scales_m[:, None]).astype(np.float32)

    raw_proj_a = res_a_all @ vt_a[:max_k_aper].T
    scales_a = np.ones(max_k_aper, dtype=np.float32)
    for m in range(max_k_aper):
        p998 = float(np.percentile(np.abs(raw_proj_a[:, m]), 99.8))
        scales_a[m] = max(0.30, p998 / (116.0 * (min_q_step * 1.25)))
    scaled_basis_a = (vt_a[:max_k_aper] * scales_a[:, None]).astype(np.float32)

    speaker = SpeakerTimbreCore(
        mean_mcep=mean_mcep,
        glottal_phase=features["glottal_phase"],
        mcep_pca_basis=scaled_basis_m,
        aper_mean=aper_mean,
        aper_pca_basis=scaled_basis_a,
    )

    stats = {
        "totalEvents": len(raw_events),
        "lyricEvents": sum(1 for e in raw_events if not e["is_hook"]),
        "hookEvents": sum(1 for e in raw_events if e["is_hook"]),
        "uniqueSyllablePrototypes": len(prototypes),
        "averageEventReuse": round(len(raw_events) / max(1, len(prototypes)), 2),
        "varianceExplainedSpeakerPct": round(100.0 * (1.0 - tot_var_after_speaker / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedSpeakerPlusAtlasPct": round(100.0 * (1.0 - tot_var_after_atlas / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedUltraPct": round(100.0 * (1.0 - var_after_u / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedExpressivePct": round(100.0 * (1.0 - var_after_e / max(tot_var_raw, 1e-12)), 2),
    }

    return speaker, prototypes, raw_events, ref_peak, stats


def encode_events_for_profile(
    raw_events: list[dict],
    speaker: SpeakerTimbreCore,
    prototypes: list[SyllablePrototype],
    features: dict,
    anatomy: dict,
    profile_spec: dict,
    ref_peak: float,
) -> list[VocalEvent]:
    """Quantizes per-event 8-track parameters with closed-loop 2.5-cent DPCM pitch (`0` clipping!) & True Backing."""
    k_pca = int(profile_spec["pcaDims"])
    k_aper = int(profile_spec["aperExprBands"])
    expr_step = int(profile_spec["exprKnotStepFrames"])
    f0_step = int(profile_spec["f0KnotStepFrames"])
    env_step = int(profile_spec["envKnotStepFrames"])
    q_step = float(profile_spec["quantStepDb"])

    f0_arr = features["f0"]
    voiced = features["voiced"]
    voicing_deg = features["voicing_deg"]
    tot_gain = features["total_gain"]
    mcep_frames = features["mcep_frames"]
    aper_frames = features["aper_log_shape"]
    onset_rise_db = features["onset_rise_db"]
    back_pitch_semi = features["back_pitch_semi"]

    pan_frames = anatomy["pan_frames"]
    back_rms_l = anatomy["back_rms_l"]
    back_rms_r = anatomy["back_rms_r"]

    basis_m = speaker.mcep_pca_basis[:k_pca]
    norm_sq_m = np.maximum(np.sum(basis_m ** 2, axis=1), 1e-9)
    basis_a = speaker.aper_pca_basis[:k_aper]
    norm_sq_a = np.maximum(np.sum(basis_a ** 2, axis=1), 1e-9)

    total_frames_song = len(mcep_frames) + 6
    base_cep_grid, base_aper_grid = build_coarticulated_prototype_grids(
        raw_events, prototypes, speaker, total_frames_song
    )

    # Share exact midpoint targets across contiguous non-plosive seams so quantized endpoints meet seamlessly!
    gain_enc = tot_gain.copy()
    vdeg_enc = voicing_deg.copy()
    f0_enc = f0_arr.copy()
    for i in range(1, len(raw_events)):
        prev_ev = raw_events[i - 1]
        curr_ev = raw_events[i]
        b = int(curr_ev["start_frame"])
        if prev_ev["end_frame"] == b and 1 <= b < len(tot_gain):
            if not curr_ev.get("is_plosive", False):
                g_mid = 0.5 * (float(tot_gain[b - 1]) + float(tot_gain[b]))
                v_mid = 0.5 * (float(voicing_deg[b - 1]) + float(voicing_deg[b]))
                gain_enc[b - 1] = g_mid
                gain_enc[b] = g_mid
                vdeg_enc[b - 1] = v_mid
                vdeg_enc[b] = v_mid
            if voiced[b - 1] and voiced[b] and f0_arr[b - 1] > 60.0 and f0_arr[b] > 60.0:
                f_mid = math.sqrt(float(f0_arr[b - 1]) * float(f0_arr[b]))
                f0_enc[b - 1] = f_mid
                f0_enc[b] = f_mid

    encoded_events: list[VocalEvent] = []
    for idx, ev in enumerate(raw_events):
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = ev["total_frames"]
        cons_f = ev["cons_frames"]
        proto = prototypes[ev["syllable_id"]]

        # 1. Closed-Loop 2.5-Cent Pitch Step Encoder (`base_midi_q = round(knot0 * 50)` -> ZERO overflow, ZERO clipping!)
        v_rel = np.flatnonzero(voiced[s_f:e_f] & (f0_enc[s_f:e_f] > 0.0))
        if len(v_rel) > 0:
            voiced_onset = int(v_rel[0])
            midi_v = 69.0 + 12.0 * np.log2(f0_enc[s_f + v_rel] / 440.0)
            midi_full = np.interp(np.arange(L, dtype=np.float32), v_rel.astype(np.float32), midi_v).astype(np.float32)
            f0_pos = build_uniform_knot_positions(L, f0_step)
            knot_targets = midi_full[f0_pos]
            base_midi_q = int(np.clip(round(float(knot_targets[0]) * 50.0), 1000, 6000))
            cur_midi = base_midi_q * 0.02
            steps_list = []
            for k in range(1, len(knot_targets)):
                sq = int(np.clip(round((float(knot_targets[k]) - cur_midi) * 40.0), -127, 127))
                steps_list.append(sq)
                cur_midi += sq * 0.025
            f0_step_q = np.array(steps_list, dtype=np.int8)
        else:
            voiced_onset = L
            base_midi_q = 3000
            f0_step_q = np.zeros(0, dtype=np.int8)

        # 2. Envelope & Voicing Knots + Plosive Burst Detection (keep >=1 breath floor inside contiguous phrases!)
        env_pos = build_envelope_knot_positions(L, cons_f, env_step)
        gain_h_q = np.maximum(gain_to_uint8(gain_enc[s_f:e_f][env_pos], ref_peak), 1).astype(np.uint8)
        if voiced_onset >= L:
            gain_a_q = np.zeros(len(env_pos), dtype=np.uint8)
        else:
            gain_a_q = np.clip(np.round(vdeg_enc[s_f:e_f][env_pos] * 255.0), 0, 255).astype(np.uint8)

        if "is_plosive" in ev:
            is_plosive = bool(ev["is_plosive"])
        else:
            cons_end = min(e_f, s_f + max(2, cons_f))
            max_rise = float(np.max(onset_rise_db[s_f:cons_end])) if cons_end > s_f else 0.0
            is_plosive = bool(proto.initial_class == "plosive" or (ev["is_hook"] and max_rise > 6.5) or max_rise > 9.5)

        # 3. Coarticulated Adaptive Curvature Vocal-Tract & HNR Expression Trajectory
        res_m = mcep_frames[s_f:e_f] - base_cep_grid[s_f:e_f]
        res_a = aper_frames[s_f:e_f] - base_aper_grid[s_f:e_f]

        expr_pos = build_adaptive_expr_knot_positions(L, cons_f, expr_step)
        coef_m = (res_m[expr_pos] @ basis_m.T) / norm_sq_m[None, :]
        coef_a = (res_a[expr_pos] @ basis_a.T) / norm_sq_a[None, :]
        expr_m_q = np.clip(np.round(coef_m / q_step), -127, 127).astype(np.int8)
        expr_a_q = np.clip(np.round(coef_a / (q_step * 1.25)), -127, 127).astype(np.int8)

        # 4. Track 07 True Stereo Chorus Backing & Harmony Descriptors
        bl_val = float(np.percentile(back_rms_l[s_f:e_f], 70.0))
        br_val = float(np.percentile(back_rms_r[s_f:e_f], 70.0))
        if max(bl_val, br_val) < ref_peak * 0.022:
            back_l_q = 0
            back_r_q = 0
            back_p_q = 0
        else:
            back_l_q = int(gain_to_uint8(bl_val, ref_peak)[0])
            back_r_q = int(gain_to_uint8(br_val, ref_peak)[0])
            bp_slice = back_pitch_semi[s_f:e_f][voiced[s_f:e_f]]
            if len(bp_slice) > 0:
                vals, cnts = np.unique(np.round(bp_slice, 1), return_counts=True)
                dom_semi = float(vals[int(np.argmax(cnts))])
                back_p_q = int(np.clip(round(dom_semi * 10.0), -127, 127))
            else:
                back_p_q = 0

        pan_val = float(np.median(pan_frames[s_f:e_f]))

        encoded_events.append(
            VocalEvent(
                event_index=idx,
                start_frame=s_f,
                cons_frames=cons_f,
                total_frames=L,
                syllable_id=ev["syllable_id"],
                display_text=ev["display_text"],
                pinyin=ev["pinyin"],
                is_hook=bool(ev["is_hook"]),
                legato_prev=bool(ev["legato_prev"]),
                legato_next=bool(ev["legato_next"]),
                is_plosive=is_plosive,
                base_midi_q=base_midi_q,
                stereo_pan_q=int(np.clip(round(pan_val * 64.0), -64, 64)),
                back_gain_l_q=back_l_q,
                back_gain_r_q=back_r_q,
                back_pitch_q=back_p_q,
                voiced_onset_frame=voiced_onset,
                f0_step_q=f0_step_q,
                gain_h_db_q=gain_h_q,
                gain_a_db_q=gain_a_q,
                expr_mcep_q=expr_m_q,
                expr_aper_q=expr_a_q,
            )
        )

    return encoded_events
