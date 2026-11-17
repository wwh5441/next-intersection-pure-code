"""Purified Deduplicated Syllable Atlas Builder & Edge-Clamped Zero-Spike Expression Encoder (V9)."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage

from models.vocal_structures import SpeakerTimbreCore, SyllablePrototype, VocalEvent


CONS_ANCHORS = np.linspace(0.0, 1.0, 4, dtype=np.float32)
VOWEL_ANCHORS = np.array([0.0, 0.22, 0.50, 0.78, 1.0], dtype=np.float32)

DB_FLOOR = -58.0
DB_CEIL = 4.0
F0_CENTS_UNIT = 5.0


def _sample_curve_2d(mat: np.ndarray, u_knots: np.ndarray) -> np.ndarray:
    L, _ = mat.shape
    if L <= 1:
        return np.repeat(mat[:1], len(u_knots), axis=0).astype(np.float32)
    idx_f = u_knots * (L - 1)
    i0 = np.clip(np.floor(idx_f).astype(np.int32), 0, L - 1)
    i1 = np.clip(i0 + 1, 0, L - 1)
    frac = (idx_f - i0)[:, None]
    return ((1.0 - frac) * mat[i0] + frac * mat[i1]).astype(np.float32)


def _extract_edge_clamped_voiced_mcep(
    mcep_slice: np.ndarray,
    voiced_slice: np.ndarray,
    gh_slice: np.ndarray,
    fallback_mcep: np.ndarray,
    floor_gh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolates MCEP strictly from clean voiced frames and edge-clamps across unvoiced onset/release!
    Eliminates V8's -60.94dB zero-frame unvoiced MCEP buzzsaw spike completely.
    """
    L, C = mcep_slice.shape
    valid = np.flatnonzero(voiced_slice & (gh_slice > floor_gh))
    if len(valid) == 0:
        valid = np.flatnonzero(voiced_slice)
    if len(valid) == 0:
        return np.repeat(fallback_mcep[None, :], L, axis=0).astype(np.float32), valid
    if len(valid) == 1:
        return np.repeat(mcep_slice[valid[:1]], L, axis=0).astype(np.float32), valid

    x_all = np.arange(L, dtype=np.float32)
    x_val = valid.astype(np.float32)
    src = mcep_slice[valid]
    out = np.empty((L, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x_all, x_val, src[:, c])
    return out, valid


def build_envelope_knot_positions(total_frames: int, cons_frames: int, vowel_step: int) -> np.ndarray:
    if total_frames <= 1:
        return np.array([0], dtype=np.int32)
    last = total_frames - 1
    vot = min(last, max(1, cons_frames))
    pts = {0, vot // 2, vot, last}
    cur = vot + vowel_step
    while cur < last:
        pts.add(int(cur))
        cur += vowel_step
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
    if total_frames <= 1 or len(knot_pos) <= 1 or len(knot_mat) == 0:
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
    vot = min(L - 1, max(1, cons_frames)) if L > 1 else 0
    u_v = np.zeros(L, dtype=np.float32)
    if L - vot > 1:
        u_v[vot:] = np.linspace(0.0, 1.0, L - vot, dtype=np.float32)
    idx_v = u_v * (len(VOWEL_ANCHORS) - 1)
    iv0 = np.clip(np.floor(idx_v).astype(np.int32), 0, len(VOWEL_ANCHORS) - 1)
    iv1 = np.clip(iv0 + 1, 0, len(VOWEL_ANCHORS) - 1)
    fv = (idx_v - iv0)[:, None]
    pred_mcep = speaker.mean_mcep[None, :] + (1.0 - fv) * proto.vowel_mcep_knots[iv0] + fv * proto.vowel_mcep_knots[iv1]

    pred_aper = np.zeros((L, len(speaker.aper_mean)), dtype=np.float32)
    if vot > 0:
        u_c = np.linspace(0.0, 1.0, vot, dtype=np.float32) * (len(CONS_ANCHORS) - 1)
        ic0 = np.clip(np.floor(u_c).astype(np.int32), 0, len(CONS_ANCHORS) - 1)
        ic1 = np.clip(ic0 + 1, 0, len(CONS_ANCHORS) - 1)
        fc = (u_c - ic0)[:, None]
        pred_aper[:vot] = speaker.aper_mean[None, :] + (1.0 - fc) * proto.cons_aper_knots[ic0] + fc * proto.cons_aper_knots[ic1]
    pred_aper[vot:] = (
        speaker.aper_mean[None, :]
        + (1.0 - fv[vot:]) * proto.vowel_aper_knots[iv0[vot:]]
        + fv[vot:] * proto.vowel_aper_knots[iv1[vot:]]
    )
    return pred_mcep.astype(np.float32), pred_aper.astype(np.float32)


def gain_to_uint8(gain: np.ndarray, ref_peak: float) -> np.ndarray:
    ratio = np.maximum(gain / max(ref_peak, 1e-8), 0.0)
    out = np.zeros(len(gain), dtype=np.uint8)
    nz = ratio > (10.0 ** (DB_FLOOR / 20.0))
    if np.any(nz):
        db = 20.0 * np.log10(ratio[nz])
        q = np.round((np.clip(db, DB_FLOOR, DB_CEIL) - DB_FLOOR) / (DB_CEIL - DB_FLOOR) * 254.0) + 1.0
        out[nz] = np.clip(q, 1, 255).astype(np.uint8)
    return out


def uint8_to_gain(q: np.ndarray, ref_peak: float) -> np.ndarray:
    out = np.zeros(len(q), dtype=np.float32)
    nz = q > 0
    if np.any(nz):
        db = DB_FLOOR + (q[nz].astype(np.float32) - 1.0) / 254.0 * (DB_CEIL - DB_FLOOR)
        out[nz] = (ref_peak * np.power(10.0, db / 20.0)).astype(np.float32)
    return out


def build_speaker_and_syllable_atlas(
    raw_events: list[dict],
    features: dict,
    anatomy: dict,
    config: dict,
) -> tuple[SpeakerTimbreCore, list[SyllablePrototype], list[dict], float, dict]:
    """Learns Clean SpeakerTimbreCore, Zero-Unvoiced-Contamination Syllable Atlas, and Orthonormal PCA Bases."""
    mcep_frames = features["mcep_frames"]
    aper_frames = features["aper_log_shape"]
    gh = features["gain_harm"]
    ga = features["gain_aper"]
    voiced = features["voiced"]
    n_cep = int(config["cepstralCoeffs"])
    n_aper = int(config["aperiodicBands"])

    ref_peak = float(max(np.max(gh), np.max(ga), 1e-6))
    floor_gh = ref_peak * 0.015

    # 1. Global Speaker Timbre Mean (strictly across clean high-SNR voiced singing frames)
    v_mask = voiced & (gh > ref_peak * 0.035)
    w_v = (gh[v_mask] ** 1.5)[:, None]
    mean_mcep = (np.sum(mcep_frames[v_mask] * w_v, axis=0) / (np.sum(w_v) + 1e-12)).astype(np.float32)

    a_mask = features["active_mask"] & (ga > ref_peak * 0.010)
    w_a = (ga[a_mask] ** 1.2)[:, None] if np.any(a_mask) else np.ones((1, 1), dtype=np.float32)
    aper_mean = (
        (np.sum(aper_frames[a_mask] * w_a, axis=0) / (np.sum(w_a) + 1e-12)).astype(np.float32)
        if np.any(a_mask)
        else np.zeros(n_aper, dtype=np.float32)
    )

    # 2. Build Clean Deduplicated Lyric Syllable Prototypes (strictly over clean voiced span for vowels!)
    lyric_groups: dict[str, list[dict]] = {}
    for ev in raw_events:
        if not ev["is_hook"]:
            lyric_groups.setdefault(ev["pinyin"], []).append(ev)

    sorted_pinyins = sorted(lyric_groups.keys())
    prototypes: list[SyllablePrototype] = []
    pinyin_to_id: dict[str, int] = {}

    for sid, py in enumerate(sorted_pinyins):
        evs = lyric_groups[py]
        cons_acc = np.zeros((len(CONS_ANCHORS), n_aper), dtype=np.float64)
        cons_w = 0.0
        v_mcep_acc = np.zeros((len(VOWEL_ANCHORS), n_cep), dtype=np.float64)
        v_mcep_w = 0.0
        v_aper_acc = np.zeros((len(VOWEL_ANCHORS), n_aper), dtype=np.float64)
        v_aper_w = 0.0

        for ev in evs:
            s_f, e_f = ev["start_frame"], ev["end_frame"]
            L = e_f - s_f
            vot = min(L - 1, max(1, ev["cons_frames"])) if L > 1 else 1

            c_ga = ga[s_f : s_f + vot]
            if np.any(c_ga > 0):
                c_slice = aper_frames[s_f : s_f + vot] - aper_mean[None, :]
                cw = float(np.mean(c_ga) + 1e-5)
                cons_acc += cw * _sample_curve_2d(c_slice, CONS_ANCHORS)
                cons_w += cw

            clean_m, valid_v = _extract_edge_clamped_voiced_mcep(
                mcep_frames[s_f:e_f], voiced[s_f:e_f], gh[s_f:e_f], mean_mcep, floor_gh
            )
            if len(valid_v) > 0:
                v0 = max(vot, int(valid_v[0])) if int(valid_v[-1]) >= vot else int(valid_v[0])
                v1 = int(valid_v[-1]) + 1
                vm_slice = clean_m[v0:v1] - mean_mcep[None, :]
                va_slice = aper_frames[s_f + v0 : s_f + v1] - aper_mean[None, :]
                vw = float(np.mean(gh[s_f + v0 : s_f + v1]) ** 1.3 + 1e-6)
                v_mcep_acc += vw * _sample_curve_2d(vm_slice, VOWEL_ANCHORS)
                v_mcep_w += vw
                v_aper_acc += vw * _sample_curve_2d(va_slice, VOWEL_ANCHORS)
                v_aper_w += vw

        proto = SyllablePrototype(
            syllable_id=sid,
            pinyin=py,
            initial_class=evs[0]["initial_class"],
            count=len(evs),
            cons_aper_knots=(cons_acc / max(cons_w, 1e-9)).astype(np.float32),
            vowel_mcep_knots=(v_mcep_acc / max(v_mcep_w, 1e-9)).astype(np.float32),
            vowel_aper_knots=(v_aper_acc / max(v_aper_w, 1e-9)).astype(np.float32),
        )
        prototypes.append(proto)
        pinyin_to_id[py] = sid

    for ev in raw_events:
        ev["syllable_id"] = pinyin_to_id[ev["pinyin"]]

    # 3. Learn Orthonormal Expression PCA Basis on Clean Voiced Residuals
    temp_speaker = SpeakerTimbreCore(
        mean_mcep=mean_mcep,
        glottal_phase=features["glottal_phase"],
        mcep_pca_basis=np.eye(n_cep, dtype=np.float32),
        aper_mean=aper_mean,
        aper_pca_basis=np.eye(n_aper, dtype=np.float32),
    )

    mcep_res_rows = []
    aper_res_rows = []
    tot_var_raw = 0.0
    tot_var_after_speaker = 0.0
    tot_var_after_atlas = 0.0

    for ev in raw_events:
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = ev["total_frames"]
        proto = prototypes[ev["syllable_id"]]
        pred_m, pred_a = eval_prototype_on_event(proto, temp_speaker, L, ev["cons_frames"])
        clean_m, valid_v = _extract_edge_clamped_voiced_mcep(
            mcep_frames[s_f:e_f], voiced[s_f:e_f], gh[s_f:e_f], mean_mcep, floor_gh
        )
        res_m = clean_m - pred_m
        res_a = aper_frames[s_f:e_f] - pred_a

        if len(valid_v) > 0:
            w_sub = (gh[s_f + valid_v] / ref_peak)[:, None]
            mcep_res_rows.append(res_m[valid_v] * np.sqrt(w_sub))
            tot_var_raw += float(np.sum((clean_m[valid_v] ** 2) * w_sub))
            tot_var_after_speaker += float(np.sum(((clean_m[valid_v] - mean_mcep[None, :]) ** 2) * w_sub))
            tot_var_after_atlas += float(np.sum((res_m[valid_v] ** 2) * w_sub))

        nz_a = ga[s_f:e_f] > (ref_peak * 0.012)
        if np.any(nz_a):
            w_a_sub = np.sqrt(ga[s_f:e_f][nz_a] / ref_peak)[:, None]
            aper_res_rows.append(res_a[nz_a] * w_a_sub)

    res_m_mat = np.vstack(mcep_res_rows)
    _, _, vt_m = np.linalg.svd(res_m_mat, full_matrices=False)
    res_a_mat = np.vstack(aper_res_rows) if aper_res_rows else np.eye(n_aper, dtype=np.float32)
    _, _, vt_a = np.linalg.svd(res_a_mat, full_matrices=False)

    max_m_pca = max(int(p["pcaDims"]) for p in config["profiles"].values())
    max_a_pca = max(int(p["aperExprBands"]) for p in config["profiles"].values())

    proj_ultra = (res_m_mat @ vt_m[:6].T) @ vt_m[:6]
    var_after_ultra = float(np.sum((res_m_mat - proj_ultra) ** 2))
    proj_expr = (res_m_mat @ vt_m[:max_m_pca].T) @ vt_m[:max_m_pca]
    var_after_expr = float(np.sum((res_m_mat - proj_expr) ** 2))

    speaker = SpeakerTimbreCore(
        mean_mcep=mean_mcep,
        glottal_phase=features["glottal_phase"],
        mcep_pca_basis=vt_m[:max_m_pca].astype(np.float32),
        aper_mean=aper_mean,
        aper_pca_basis=vt_a[:max_a_pca].astype(np.float32),
    )

    stats = {
        "totalEvents": len(raw_events),
        "lyricEvents": sum(1 for e in raw_events if not e["is_hook"]),
        "hookEvents": sum(1 for e in raw_events if e["is_hook"]),
        "uniqueSyllablePrototypes": len(prototypes),
        "averageEventReuse": round(len(raw_events) / max(1, len(prototypes)), 2),
        "varianceExplainedSpeakerPct": round(100.0 * (1.0 - tot_var_after_speaker / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedSpeakerPlusAtlasPct": round(100.0 * (1.0 - tot_var_after_atlas / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedUltra6DPct": round(100.0 * (1.0 - var_after_ultra / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedExpressive16DPct": round(100.0 * (1.0 - var_after_expr / max(tot_var_raw, 1e-12)), 2),
    }

    return speaker, prototypes, raw_events, ref_peak, stats


def _soft_deadzone_quantize(vals: np.ndarray, q_step: float, deadzone_frac: float = 0.34) -> np.ndarray:
    """Purges sub-threshold micro-ripple dirt while keeping real vocal expression trajectories."""
    scaled = vals / q_step
    mask = np.abs(scaled) < deadzone_frac
    q = np.clip(np.round(scaled), -127, 127).astype(np.int8)
    q[mask] = 0
    return q


def encode_events_for_profile(
    raw_events: list[dict],
    speaker: SpeakerTimbreCore,
    prototypes: list[SyllablePrototype],
    features: dict,
    anatomy: dict,
    profile_spec: dict,
    ref_peak: float,
) -> list[VocalEvent]:
    """Quantizes clean edge-clamped trajectories & purges silent residual streams (zero garbage bytes)."""
    k_pca = int(profile_spec["pcaDims"])
    k_aper = int(profile_spec["aperExprBands"])
    expr_step = int(profile_spec["exprKnotStepFrames"])
    f0_step = int(profile_spec["f0KnotStepFrames"])
    env_step = int(profile_spec["envKnotStepFrames"])
    q_step = float(profile_spec["quantStepDb"])

    f0_arr = features["f0"]
    voiced = features["voiced"]
    gh = features["gain_harm"]
    ga = features["gain_aper"]
    mcep_frames = features["mcep_frames"]
    aper_frames = features["aper_log_shape"]
    pan_frames = anatomy["pan_frames"]
    width_frames = anatomy["stereo_width_frames"]
    floor_gh = ref_peak * 0.015

    basis_m = speaker.mcep_pca_basis[:k_pca]
    basis_a = speaker.aper_pca_basis[:k_aper]

    encoded_events: list[VocalEvent] = []
    for idx, ev in enumerate(raw_events):
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = ev["total_frames"]
        cons_f = ev["cons_frames"]
        proto = prototypes[ev["syllable_id"]]

        # 1. Clean Pitch Contour in MIDI + 5-cent spline knots (edge-clamped across consonant onset/release!)
        v_rel = np.flatnonzero(voiced[s_f:e_f] & (f0_arr[s_f:e_f] > 0.0) & (gh[s_f:e_f] > ref_peak * 0.008))
        if len(v_rel) > 0:
            voiced_onset = int(v_rel[0])
            midi_v = 69.0 + 12.0 * np.log2(f0_arr[s_f + v_rel] / 440.0)
            midi_full = np.interp(np.arange(L, dtype=np.float32), v_rel.astype(np.float32), midi_v).astype(np.float32)
            base_midi_q = int(np.clip(round(float(np.median(midi_v)) * 4.0), 120, 420))
            base_midi = base_midi_q * 0.25
            f0_pos = build_uniform_knot_positions(L, f0_step)
            if f0_step >= 3 and L >= 5:
                midi_full = ndimage.gaussian_filter1d(midi_full, sigma=0.35 * f0_step)
            delta_cents = (midi_full[f0_pos] - base_midi) * 100.0
            f0_delta_q = np.clip(np.round(delta_cents / F0_CENTS_UNIT), -127, 127).astype(np.int8)
        else:
            voiced_onset = L
            base_midi_q = 240
            f0_delta_q = np.zeros(1, dtype=np.int8)

        # 2. Loudness Contours (gated cleanly at DB_FLOOR = -58dB; silent air events purged completely!)
        env_pos = build_envelope_knot_positions(L, cons_f, env_step)
        gh_sub = gh[s_f:e_f].copy()
        gh_sub[~voiced[s_f:e_f]] = 0.0
        if voiced_onset >= L:
            gh_sub[:] = 0.0
        gain_h_q = gain_to_uint8(gh_sub[env_pos], ref_peak)
        gain_a_q = gain_to_uint8(ga[s_f:e_f][env_pos], ref_peak)

        silent_aper = int(np.max(gain_a_q)) < 6
        if silent_aper:
            gain_a_q[:] = 0

        # 3. Clean Edge-Clamped Expression Trajectory (ZERO unvoiced -61dB spikes & ZERO silent-air garbage!)
        pred_m, pred_a = eval_prototype_on_event(proto, speaker, L, cons_f)
        expr_pos = build_uniform_knot_positions(L, expr_step)

        if voiced_onset < L:
            clean_m, _ = _extract_edge_clamped_voiced_mcep(
                mcep_frames[s_f:e_f], voiced[s_f:e_f], gh[s_f:e_f], speaker.mean_mcep, floor_gh
            )
            res_m = clean_m - pred_m
            if L >= 4 and expr_step >= 3:
                res_m = ndimage.gaussian_filter1d(res_m, sigma=0.42 * expr_step, axis=0)
            coef_m = res_m[expr_pos] @ basis_m.T
            expr_m_q = _soft_deadzone_quantize(coef_m, q_step, deadzone_frac=0.34)
        else:
            expr_m_q = np.zeros((0, k_pca), dtype=np.int8)

        if not silent_aper:
            actual_a = aper_frames[s_f:e_f].copy()
            silent_frames_a = ga[s_f:e_f] < (ref_peak * 0.008)
            actual_a[silent_frames_a] = pred_a[silent_frames_a]
            res_a = actual_a - pred_a
            if L >= 4 and expr_step >= 3:
                res_a = ndimage.gaussian_filter1d(res_a, sigma=0.40 * expr_step, axis=0)
            coef_a = res_a[expr_pos] @ basis_a.T
            expr_a_q = _soft_deadzone_quantize(coef_a, q_step * 1.35, deadzone_frac=0.38)
        else:
            expr_a_q = np.zeros((0, k_aper), dtype=np.int8)

        pan_val = float(np.median(pan_frames[s_f:e_f]))
        width_val = float(np.percentile(width_frames[s_f:e_f], 65.0))
        if width_val < 0.04:
            width_val = 0.0

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
                base_midi_q=base_midi_q,
                stereo_pan_q=int(np.clip(round(pan_val * 64.0), -64, 64)),
                stereo_width_q=int(np.clip(round(width_val * 255.0), 0, 255)),
                voiced_onset_frame=voiced_onset,
                f0_delta_cents_q=f0_delta_q,
                gain_h_db_q=gain_h_q,
                gain_a_db_q=gain_a_q,
                expr_mcep_q=expr_m_q,
                expr_aper_q=expr_a_q,
            )
        )

    return encoded_events
