"""48kHz 8-Track Physical Vocal Anatomy Synthesizer: DAC Coarticulation + Continuous-Phase Chebyshev + 512-pt WOLA + ER/FDN (V8.5)."""
from __future__ import annotations

import math
import numpy as np
import scipy.signal as signal

from .event_codec import unpack_event_stream, unpack_speaker_core, unpack_syllable_atlas
from .harmonic_aperiodic_analyzer import build_aper_edges, precompute_unvoiced_minphase_tables
from .space_fdn_model import synthesize_space_fdn
from .syllable_atlas_builder import (
    build_adaptive_expr_knot_positions,
    build_coarticulated_prototype_grids,
    build_envelope_knot_positions,
    build_uniform_knot_positions,
    interp_1d_knots,
    interp_2d_knots,
    reconstruct_midi_knots,
    uint8_to_gain,
)


def _smooth_frame_track(track: np.ndarray) -> np.ndarray:
    k = np.array([0.08, 0.17, 0.50, 0.17, 0.08], dtype=np.float32)
    out = np.convolve(track, k, mode="same")
    return np.convolve(out, k, mode="same").astype(np.float32)


def _dc_block_60hz(x: np.ndarray, sr: int) -> np.ndarray:
    r_dc = math.exp(-2.0 * math.pi * 60.0 / float(sr))
    g_dc = 0.5 * (1.0 + r_dc)
    return signal.lfilter([g_dc, -g_dc], [1.0, -r_dc], x.astype(np.float64)).astype(np.float32)


def _build_warped_cosine_lut(n_cep: int, n_lut: int = 4096) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n_lut, dtype=np.float64)[:, None]
    m = np.arange(1, n_cep + 1, dtype=np.float64)[None, :]
    return np.cos(math.pi * u * m).astype(np.float32)


def _bridge_short_f0_gaps(f0_grid: np.ndarray, max_gap_frames: int = 12) -> None:
    voiced_idx = np.flatnonzero(f0_grid > 60.0)
    if len(voiced_idx) < 2:
        return
    for i in range(len(voiced_idx) - 1):
        fa = int(voiced_idx[i])
        fb = int(voiced_idx[i + 1])
        gap = fb - fa - 1
        if 1 <= gap <= max_gap_frames:
            va = float(f0_grid[fa])
            vb = float(f0_grid[fb])
            if abs(math.log2(vb / va)) < 0.75:
                for g in range(1, gap + 1):
                    u = g / float(gap + 1)
                    f0_grid[fa + g] = va + (vb - va) * u


def synthesize_from_packets(
    raw_speaker: bytes,
    raw_atlas: bytes,
    raw_events: bytes,
    raw_fdn: bytes,
    config: dict,
) -> dict:
    """Reconstructs 48kHz 8-track stereo vocal strictly from compact V8.4 packets with zero distortion."""
    speaker, spk_meta = unpack_speaker_core(raw_speaker)
    prototypes = unpack_syllable_atlas(raw_atlas)
    events, ev_meta = unpack_event_stream(raw_events)

    sr = int(spk_meta["sampleRate"])
    n_samples = int(spk_meta["numSamples"])
    hop = int(spk_meta["hop"])
    ref_peak = float(spk_meta["refPeak"])

    n_fft = int(config["fftSize"])          # 512
    half_w = n_fft // 2                     # 256
    n_bins = n_fft // 2 + 1                 # 257
    n_cep = int(config["cepstralCoeffs"])   # 40
    n_aper = int(config["aperiodicBands"])  # 8
    max_harm = int(config["maxHarmonics"])  # 64
    fmin = float(config["harmFreqMin"])
    fmax = float(config["harmFreqMax"])

    e0 = 21.4 * math.log10(1.0 + 0.00437 * fmin)
    e1 = 21.4 * math.log10(1.0 + 0.00437 * fmax)
    inv_e_span = 1.0 / (e1 - e0)
    b_cos_lut = _build_warped_cosine_lut(n_cep, 4096)

    B_ln, Phi_rad, band_taper = precompute_unvoiced_minphase_tables(config)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sib_hp = np.clip((freqs - 900.0) / 1100.0, 0.15, 1.0).astype(np.float32)
    sib_hp[0] = 0.0

    aper_edges = build_aper_edges(n_aper, float(config["aperFreqMin"]), float(config["aperFreqMax"]))
    aper_centers = np.sqrt(aper_edges[:-1] * aper_edges[1:]).astype(np.float64)
    log_freqs = np.log(np.maximum(freqs, 75.0))
    log_aper_c = np.log(aper_centers)

    k_pca = int(ev_meta["pcaDims"])
    k_aper = int(ev_meta["aperExprBands"])
    expr_step = int(ev_meta["exprKnotStepFrames"])
    f0_step = int(ev_meta["f0KnotStepFrames"])
    env_step = int(ev_meta["envKnotStepFrames"])
    pf_beta = float(ev_meta.get("formantPostFilterBeta", 0.0))
    q_step = float(ev_meta["quantStepDb"])

    pf_weights = np.ones(n_cep, dtype=np.float32)
    if pf_beta > 1e-6:
        for m in range(2, n_cep):
            u_m = (m - 2) / max(1.0, float(n_cep - 3))
            pf_weights[m] = 1.0 + pf_beta * math.sin(math.pi * min(1.0, u_m))

    basis_m = speaker.mcep_pca_basis[:k_pca]
    basis_a = speaker.aper_pca_basis[:k_aper]

    total_frames_song = n_samples // hop + 6
    gain_grid = np.zeros(total_frames_song, dtype=np.float32)
    vdeg_grid = np.zeros(total_frames_song, dtype=np.float32)
    f0_grid = np.zeros(total_frames_song, dtype=np.float32)
    plosive_grid = np.zeros(total_frames_song, dtype=np.float32)
    cep_grid = np.zeros((total_frames_song, n_cep), dtype=np.float32)
    aper_grid = np.full((total_frames_song, n_aper), -3.6, dtype=np.float32)
    pan_track = np.zeros(total_frames_song, dtype=np.float32)
    back_l_grid = np.zeros(total_frames_song, dtype=np.float32)
    back_r_grid = np.zeros(total_frames_song, dtype=np.float32)
    back_semi_grid = np.zeros(total_frames_song, dtype=np.float32)

    # Pass 1: Continuous Phrase-Level Coarticulated Prototype Grids (`x-i-a--y-i--g-e--l-u--k-ou--j-ian`)
    base_cep_grid, base_aper_grid = build_coarticulated_prototype_grids(
        events, prototypes, speaker, total_frames_song
    )

    for ev in events:
        L = ev.total_frames
        if L <= 0:
            continue
        s_f = ev.start_frame
        cf = ev.cons_frames

        env_pos = build_envelope_knot_positions(L, cf, env_step)
        gain_curve = interp_1d_knots(env_pos, uint8_to_gain(ev.gain_h_db_q, ref_peak), L)
        vdeg_curve = np.clip(interp_1d_knots(env_pos, ev.gain_a_db_q.astype(np.float32) / 255.0, L), 0.0, 1.0)

        unvoiced_only = ev.voiced_onset_frame >= L
        if not unvoiced_only:
            f0_pos = build_uniform_knot_positions(L, f0_step)
            midi_knots = reconstruct_midi_knots(ev.base_midi_q, ev.f0_step_q)
            midi_curve = interp_1d_knots(f0_pos, midi_knots, L)
            f0_curve = (440.0 * np.power(2.0, (midi_curve - 69.0) / 12.0)).astype(np.float32)
        else:
            f0_curve = np.zeros(L, dtype=np.float32)
            vdeg_curve[:] = 0.0

        expr_pos = build_adaptive_expr_knot_positions(L, cf, expr_step)
        dm_knots = (ev.expr_mcep_q.astype(np.float32) * q_step) @ basis_m
        da_knots = (ev.expr_aper_q.astype(np.float32) * (q_step * 1.25)) @ basis_a
        dm_curve = interp_2d_knots(expr_pos, dm_knots, L)
        da_curve = interp_2d_knots(expr_pos, da_knots, L)

        f_end_clip = min(total_frames_song, s_f + L)
        span = f_end_clip - s_f
        if span <= 0:
            continue

        gain_grid[s_f:f_end_clip] = gain_curve[:span]
        vdeg_grid[s_f:f_end_clip] = vdeg_curve[:span]
        f0_grid[s_f:f_end_clip] = f0_curve[:span]
        cep_grid[s_f:f_end_clip] = (base_cep_grid[s_f:f_end_clip] + dm_curve[:span]) * pf_weights[None, :]
        aper_grid[s_f:f_end_clip] = np.clip(base_aper_grid[s_f:f_end_clip] + da_curve[:span], -4.8, -1.70)
        pan_track[s_f:f_end_clip] = float(ev.stereo_pan_q) / 64.0

        if ev.is_plosive:
            burst_len = min(span, max(2, min(cf, 7)))
            for r in range(burst_len):
                plosive_grid[s_f + r] = 0.5 if (r == burst_len - 1 and burst_len > 2) else 1.0

        bl_g = float(uint8_to_gain(ev.back_gain_l_q, ref_peak)[0]) * 0.68
        br_g = float(uint8_to_gain(ev.back_gain_r_q, ref_peak)[0]) * 0.68
        if (bl_g > 1e-6 or br_g > 1e-6) and not unvoiced_only:
            v_mix_ev = np.clip((vdeg_curve - 0.10) / 0.55, 0.0, 1.0)
            v_env = gain_curve * np.sqrt(v_mix_ev)
            sorted_v = np.sort(v_env)
            p70 = float(sorted_v[min(L - 1, int(0.70 * (L - 1)))])
            if p70 > 1e-6:
                env_shape = np.clip(v_env / p70, 0.0, 1.30)
                back_l_grid[s_f:f_end_clip] = bl_g * env_shape[:span]
                back_r_grid[s_f:f_end_clip] = br_g * env_shape[:span]
                back_semi_grid[s_f:f_end_clip] = float(ev.back_pitch_q) * 0.10

    # Post-decode DAC `--` seam polish across all contiguous syllable bridges (erases any int8/uint8 endpoint step!)
    post_seams = []
    for i in range(1, len(events)):
        prev_ev = events[i - 1]
        curr_ev = events[i]
        b = curr_ev.start_frame
        if prev_ev.total_frames >= 2 and curr_ev.total_frames >= 2 and (prev_ev.start_frame + prev_ev.total_frames == b) and (1 <= b < total_frames_song):
            w_pre = min(6, prev_ev.total_frames // 2, b)
            w_post = min(2 if curr_ev.is_plosive else 5, curr_ev.total_frames // 2, total_frames_song - b)
            if w_pre >= 1 and w_post >= 1:
                post_seams.append(
                    (
                        b,
                        w_pre,
                        w_post,
                        curr_ev.is_plosive,
                        cep_grid[b].copy() - cep_grid[b - 1].copy(),
                        aper_grid[b].copy() - aper_grid[b - 1].copy(),
                        float(gain_grid[b] - gain_grid[b - 1]),
                        float(vdeg_grid[b] - vdeg_grid[b - 1]),
                    )
                )

    for b, w_pre, w_post, is_plos, dc, da, dg, dv in post_seams:
        inv_tot = 1.0 / float(w_pre + w_post)
        g_pre = w_pre * inv_tot
        g_post = w_post * inv_tot
        for r in range(w_pre):
            w = 0.5 * g_pre * (1.0 + math.cos(math.pi * (r + 0.5) / float(w_pre)))
            idx_l = b - 1 - r
            cep_grid[idx_l] += w * dc
            aper_grid[idx_l] = np.clip(aper_grid[idx_l] + w * da, -4.8, -1.70)
            if not is_plos:
                gain_grid[idx_l] = max(0.0, float(gain_grid[idx_l]) + w * dg)
                vdeg_grid[idx_l] = min(1.0, max(0.0, float(vdeg_grid[idx_l]) + w * dv))
        for r in range(w_post):
            w = 0.5 * g_post * (1.0 + math.cos(math.pi * (r + 0.5) / float(w_post)))
            idx_r = b + r
            cep_grid[idx_r] -= w * dc
            aper_grid[idx_r] = np.clip(aper_grid[idx_r] - w * da, -4.8, -1.70)
            if not is_plos:
                gain_grid[idx_r] = max(0.0, float(gain_grid[idx_r]) - w * dg)
                vdeg_grid[idx_r] = min(1.0, max(0.0, float(vdeg_grid[idx_r]) - w * dv))

    # Physiological vocal-tract & airflow inertia across interior active non-plosive frames
    cep_src = cep_grid.copy()
    aper_src = aper_grid.copy()
    gain_src = gain_grid.copy()
    vdeg_src = vdeg_grid.copy()
    for f in range(1, total_frames_song - 1):
        if gain_src[f - 1] > 0.0 and gain_src[f] > 0.0 and gain_src[f + 1] > 0.0 and plosive_grid[f] == 0.0:
            cep_grid[f] = 0.20 * cep_src[f - 1] + 0.60 * cep_src[f] + 0.20 * cep_src[f + 1]
            aper_grid[f] = np.clip(0.20 * aper_src[f - 1] + 0.60 * aper_src[f] + 0.20 * aper_src[f + 1], -4.8, -1.70)
            gain_grid[f] = 0.20 * gain_src[f - 1] + 0.60 * gain_src[f] + 0.20 * gain_src[f + 1]
            vdeg_grid[f] = min(1.0, max(0.0, float(0.20 * vdeg_src[f - 1] + 0.60 * vdeg_src[f] + 0.20 * vdeg_src[f + 1])))

    # Smooth 3-frame micro-taper ONLY at true isolated phrase entrances/exits (adjacent frame silent)
    g_orig = gain_grid.copy()
    fade3 = (0.35, 0.72, 0.93)
    for f in range(1, total_frames_song - 1):
        if g_orig[f] > 0.0 and g_orig[f - 1] == 0.0:
            if plosive_grid[f] > 0.0:
                gain_grid[f] *= 0.65
            else:
                for k_f, fw in enumerate(fade3):
                    if f + k_f < total_frames_song and g_orig[f + k_f] > 0.0:
                        gain_grid[f + k_f] *= fw
        if g_orig[f] > 0.0 and g_orig[f + 1] == 0.0:
            for k_f, fw in enumerate(fade3):
                if f - k_f >= 0 and g_orig[f - k_f] > 0.0:
                    gain_grid[f - k_f] *= fw

    # Continuous breath-phrase glottal phase & anticipatory pitch portamento (`--` pitch ribbon)
    _bridge_short_f0_gaps(f0_grid, max_gap_frames=36)
    f0_seams = []
    for i in range(1, len(events)):
        prev_ev = events[i - 1]
        curr_ev = events[i]
        b = curr_ev.start_frame
        if prev_ev.total_frames >= 2 and curr_ev.total_frames >= 2 and (prev_ev.start_frame + prev_ev.total_frames == b) and (1 <= b < total_frames_song):
            fa = float(f0_grid[b - 1])
            fb = float(f0_grid[b])
            if fa > 60.0 and fb > 60.0:
                dlp = math.log2(fb / fa)
                if abs(dlp) < 0.65:
                    wp = min(6, prev_ev.total_frames // 2, b)
                    wq = min(4, curr_ev.total_frames // 2, total_frames_song - b)
                    if wp >= 1 and wq >= 1:
                        f0_seams.append((b, wp, wq, dlp))

    for b, wp, wq, dlp in f0_seams:
        inv_t = 1.0 / float(wp + wq)
        gp = wp * inv_t
        gq = wq * inv_t
        for r in range(wp):
            idx_l = b - 1 - r
            if f0_grid[idx_l] > 60.0:
                w = 0.5 * gp * (1.0 + math.cos(math.pi * (r + 0.5) / float(wp)))
                f0_grid[idx_l] = float(f0_grid[idx_l] * (2.0 ** (w * dlp)))
        for r in range(wq):
            idx_r = b + r
            if f0_grid[idx_r] > 60.0:
                w = 0.5 * gq * (1.0 + math.cos(math.pi * (r + 0.5) / float(wq)))
                f0_grid[idx_r] = float(f0_grid[idx_r] * (2.0 ** (-w * dlp)))

    f0_src = f0_grid.copy()
    for f in range(1, total_frames_song - 1):
        if f0_src[f - 1] > 60.0 and f0_src[f] > 60.0 and f0_src[f + 1] > 60.0:
            lp_lead = 0.22 * math.log2(f0_src[f - 1]) + 0.56 * math.log2(f0_src[f]) + 0.22 * math.log2(f0_src[f + 1])
            f0_grid[f] = float(2.0 ** lp_lead)

    f0_back_raw = np.zeros(total_frames_song, dtype=np.float32)
    vm_f0 = f0_grid > 60.0
    f0_back_raw[vm_f0] = f0_grid[vm_f0] * np.power(2.0, back_semi_grid[vm_f0] / 12.0)
    f0_back_grid = f0_back_raw.copy()
    for f in range(1, total_frames_song - 1):
        if f0_back_raw[f - 1] > 60.0 and f0_back_raw[f] > 60.0 and f0_back_raw[f + 1] > 60.0:
            lp_sm = 0.25 * math.log2(f0_back_raw[f - 1]) + 0.50 * math.log2(f0_back_raw[f]) + 0.25 * math.log2(f0_back_raw[f + 1])
            f0_back_grid[f] = float(2.0 ** lp_sm)

    back_l_smooth = _smooth_frame_track(back_l_grid)
    back_r_smooth = _smooth_frame_track(back_r_grid)
    pan_smooth = _smooth_frame_track(pan_track)

    # Pass 2: Tracks 01 (`lead_sub_f0`), 02 (`lead_vowel_body`), 03 (`lead_harmonic_sheen`) — Continuous-Phase Harmonics
    buf_len = n_samples + n_fft + hop
    sub_mono = np.zeros(buf_len, dtype=np.float32)
    body_mono = np.zeros(buf_len, dtype=np.float32)
    sheen_mono = np.zeros(buf_len, dtype=np.float32)
    phi_arr = np.zeros(buf_len, dtype=np.float32)
    period_arr = np.zeros(buf_len, dtype=np.int32)

    glott_psi = speaker.glottal_phase[:max_harm].astype(np.float64)
    hs_all = np.arange(1, max_harm + 1, dtype=np.float64)
    hop_s = np.arange(hop, dtype=np.float64)
    hop_frac = (hop_s / float(hop))[None, :]
    two_pi = 2.0 * math.pi
    two_pi_over_sr = two_pi / float(sr)
    n_hops = n_samples // hop + 1

    def eval_lead_frame(f_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        v_mix = min(1.0, max(0.0, (float(vdeg_grid[f_idx]) - 0.10) / 0.55))
        g_h = float(gain_grid[f_idx]) * math.sqrt(v_mix)
        ff = float(f0_grid[f_idx])
        sub_v = np.zeros(max_harm, dtype=np.float64)
        body_v = np.zeros(max_harm, dtype=np.float64)
        sheen_v = np.zeros(max_harm, dtype=np.float64)
        if g_h <= 1e-7 or ff <= 60.0:
            return sub_v, body_v, sheen_v, 0
        n_act = int(min(max_harm, math.floor(15799.0 / ff)))
        if n_act <= 0:
            return sub_v, body_v, sheen_v, 0
        fhs = hs_all[:n_act] * ff
        fc_c = np.clip(fhs, fmin, fmax)
        u_h = (21.4 * np.log10(1.0 + 0.00437 * fc_c) - e0) * inv_e_span
        u_idx = np.clip(np.round(u_h * 4095.0).astype(np.int32), 0, 4095)
        db_h = b_cos_lut[u_idx] @ cep_grid[f_idx]
        low_ref_db = float(np.max(db_h[: min(n_act, 6)]))
        hi_mask = fhs > 2800.0
        if np.any(hi_mask):
            db_h = np.where(hi_mask, np.minimum(db_h, low_ref_db - 3.0), db_h)
        hf_tap = np.clip((15800.0 - fhs) / 1600.0, 0.0, 1.0)
        rel_a = np.power(10.0, db_h / 20.0) * hf_tap
        rms_u = math.sqrt(0.5 * float(np.sum(rel_a ** 2)) + 1e-18)
        amps = rel_a * (g_h / rms_u)

        w_sub = 1.0 / (1.0 + (fhs / 720.0) ** 6)
        w_hi = 1.0 / (1.0 + (3200.0 / fhs) ** 6)
        w_mid = np.maximum(0.0, 1.0 - w_sub - w_hi)
        sub_v[:n_act] = amps * w_sub
        body_v[:n_act] = amps * w_mid
        sheen_v[:n_act] = amps * w_hi
        return sub_v, body_v, sheen_v, n_act

    phi = 0.0
    sub_a, body_a, sheen_a, n_act_a = eval_lead_frame(0)
    for f in range(n_hops):
        sub_b, body_b, sheen_b, n_act_b = eval_lead_frame(f + 1)
        f0_a = float(f0_grid[f])
        f0_b = float(f0_grid[f + 1])

        if f0_a <= 60.0 and f0_b <= 60.0:
            phi = 0.0
            sub_a, body_a, sheen_a, n_act_a = sub_b, body_b, sheen_b, n_act_b
            continue

        if f0_a <= 60.0 and f0_b > 60.0:
            phi = 0.0
            f_s = f_e = f0_b
        elif f0_a > 60.0 and f0_b <= 60.0:
            f_s = f_e = f0_a
        else:
            f_s, f_e = f0_a, f0_b

        base = f * hop
        fc_vec = f_s + ((f_e - f_s) / float(hop)) * hop_s
        dphi_vec = two_pi_over_sr * fc_vec
        phi_hop = np.mod(phi + np.cumsum(dphi_vec), two_pi)
        phi = float(phi_hop[-1])

        phi_arr[base : base + hop] = phi_hop.astype(np.float32)
        period_arr[base : base + hop] = np.round(sr / fc_vec).astype(np.int32)

        n_h = max(n_act_a, n_act_b)
        if n_h > 0:
            waves = np.cos(hs_all[:n_h, None] * phi_hop[None, :] + glott_psi[:n_h, None])
            sa = sub_a[:n_h, None]
            sb = sub_b[:n_h, None]
            ba = body_a[:n_h, None]
            bb = body_b[:n_h, None]
            sha = sheen_a[:n_h, None]
            shb = sheen_b[:n_h, None]

            sub_mono[base : base + hop] = np.sum((sa + (sb - sa) * hop_frac) * waves, axis=0).astype(np.float32)
            body_mono[base : base + hop] = np.sum((ba + (bb - ba) * hop_frac) * waves, axis=0).astype(np.float32)
            sheen_mono[base : base + hop] = np.sum((sha + (shb - sha) * hop_frac) * waves, axis=0).astype(np.float32)

        sub_a, body_a, sheen_a, n_act_a = sub_b, body_b, sheen_b, n_act_b

    # Pass 3: Tracks 04 (`lead_plosive_burst`), 05 (`lead_sibilant`), 06 (`lead_velvet_breath`) — 512-pt WOLA
    rng_unv = np.random.default_rng(20261115)
    e_unv = _dc_block_60hz(rng_unv.standard_normal(buf_len).astype(np.float32), sr)

    rng_br = np.random.default_rng(20260804)
    w_br_raw = rng_br.standard_normal(buf_len).astype(np.float32)
    voiced_s = period_arr > 0
    mod = np.ones_like(w_br_raw)
    mod[voiced_s] = (1.0 + 0.42 * np.cos(phi_arr[voiced_s])) / math.sqrt(1.0 + 0.5 * (0.42 ** 2))
    w_mod = w_br_raw * mod
    e_breath = w_mod.copy()
    v_idx = np.flatnonzero(voiced_s)
    valid_v = v_idx[v_idx >= period_arr[v_idx]]
    e_breath[valid_v] = (w_mod[valid_v] - 0.58 * w_mod[valid_v - period_arr[valid_v]]) / math.sqrt(1.0 + 0.58 ** 2)
    e_breath = _dc_block_60hz(e_breath, sr)

    w_sqrt = np.sin(math.pi * (np.arange(n_fft, dtype=np.float32) + 0.5) / float(n_fft))
    plos_mono = np.zeros(buf_len, dtype=np.float32)
    sib_mono = np.zeros(buf_len, dtype=np.float32)
    breath_mono = np.zeros(buf_len, dtype=np.float32)
    two_over_nfft = 2.0 / float(n_fft)

    for f in np.flatnonzero(gain_grid > 1e-7):
        center = f * hop
        if center >= n_samples:
            break
        start = center - half_w
        if start < 0 or start + n_fft > buf_len:
            continue

        c_vec = cep_grid[f]
        H_env = np.exp(np.clip(c_vec @ B_ln, -14.0, 8.0)) * band_taper
        ph_min = c_vec @ Phi_rad
        phase_unit = np.cos(ph_min) + 1j * np.sin(ph_min)

        v_mix = min(1.0, max(0.0, (float(vdeg_grid[f]) - 0.10) / 0.55))
        g_tot = float(gain_grid[f])
        g_unv = g_tot * math.sqrt(1.0 - v_mix)
        g_harm = g_tot * math.sqrt(v_mix)

        if g_unv > 1e-7:
            p_frac = float(plosive_grid[f])
            tilt_u = p_frac + (1.0 - p_frac) * sib_hp
            H_u = H_env * tilt_u
            pow_u = two_over_nfft * float(np.sum(H_u[1:] ** 2))
            scale_u = g_unv / math.sqrt(pow_u + 1e-18)
            xu = e_unv[start : start + n_fft] * w_sqrt
            yu = np.fft.irfft(np.fft.rfft(xu) * (scale_u * H_u * phase_unit), n=n_fft).astype(np.float32) * w_sqrt
            if p_frac > 0.0:
                plos_mono[start : start + n_fft] += p_frac * yu
            if p_frac < 1.0:
                sib_mono[start : start + n_fft] += (1.0 - p_frac) * yu

        f_val = float(f0_grid[f])
        if g_harm > 1e-7 and f_val > 60.0:
            pow_e = two_over_nfft * float(np.sum(H_env[1:] ** 2))
            H_unit = H_env / math.sqrt(pow_e + 1e-18)
            ap_log = np.interp(
                log_freqs,
                log_aper_c,
                aper_grid[f].astype(np.float64),
                left=float(aper_grid[f, 0]),
                right=float(aper_grid[f, -1]),
            )
            hnr_amp = np.power(10.0, 0.5 * ap_log).astype(np.float32)
            sub_cut = np.clip((freqs - 0.85 * f_val) / max(0.35 * f_val, 35.0), 0.0, 1.0).astype(np.float32)
            H_br = (g_harm * H_unit * hnr_amp * sub_cut) * phase_unit
            xb = e_breath[start : start + n_fft] * w_sqrt
            yb = np.fft.irfft(np.fft.rfft(xb) * H_br, n=n_fft).astype(np.float32) * w_sqrt
            breath_mono[start : start + n_fft] += yb

    # Pass 4: Track 07 (`backing_harmony`) — Independent L/R True Stereo Chorus Voices (Zero Comb Filtering!)
    max_h_back = 28
    hs_back = np.arange(1, max_h_back + 1, dtype=np.float64)
    psi_back = glott_psi[:max_h_back]

    def synth_backing_channel(gain_arr: np.ndarray, frame_lag: int, detune_mult: float, vib_hz: float, vib_cos: bool, init_phi: float) -> np.ndarray:
        out_ch = np.zeros(buf_len, dtype=np.float32)

        def eval_back_frame(f_idx: int) -> tuple[np.ndarray, float, int]:
            g_b = float(gain_arr[f_idx])
            f_src = max(0, f_idx - frame_lag)
            ff_base = float(f0_back_grid[f_src])
            amps_v = np.zeros(max_h_back, dtype=np.float64)
            if g_b <= 1e-6 or ff_base <= 60.0:
                return amps_v, 0.0, 0
            t_sec = (f_idx * hop) / float(sr)
            vib = math.cos(two_pi * vib_hz * t_sec) if vib_cos else math.sin(two_pi * vib_hz * t_sec)
            ff = ff_base * detune_mult * (1.0 + 0.0018 * vib)
            n_act = int(min(max_h_back, math.floor(7600.0 / ff)))
            if n_act <= 0:
                return amps_v, ff, 0
            fhs = hs_back[:n_act] * ff
            fc_c = np.clip(fhs, fmin, fmax)
            u_h = (21.4 * np.log10(1.0 + 0.00437 * fc_c) - e0) * inv_e_span
            u_idx = np.clip(np.round(u_h * 4095.0).astype(np.int32), 0, 4095)
            db_h = b_cos_lut[u_idx] @ cep_grid[f_src]
            warm_roll = (1.0 / (1.0 + (fhs / 3800.0) ** 2)) * np.clip((7600.0 - fhs) / 1200.0, 0.0, 1.0)
            rel_a = np.power(10.0, db_h / 20.0) * warm_roll
            rms_u = math.sqrt(0.5 * float(np.sum(rel_a ** 2)) + 1e-18)
            amps_v[:n_act] = rel_a * (g_b / rms_u)
            return amps_v, ff, n_act

        phi_c = init_phi
        amp_a, ff_a, na = eval_back_frame(0)
        for f in range(n_hops):
            amp_b, ff_b, nb = eval_back_frame(f + 1)
            if ff_a <= 60.0 and ff_b <= 60.0:
                phi_c = init_phi
                amp_a, ff_a, na = amp_b, ff_b, nb
                continue
            if ff_a <= 60.0 and ff_b > 60.0:
                f_s = f_e = ff_b
            elif ff_a > 60.0 and ff_b <= 60.0:
                f_s = f_e = ff_a
            else:
                f_s, f_e = ff_a, ff_b

            base = f * hop
            fc_vec = f_s + ((f_e - f_s) / float(hop)) * hop_s
            phi_hop = np.mod(phi_c + np.cumsum(two_pi_over_sr * fc_vec), two_pi)
            phi_c = float(phi_hop[-1])

            n_h = max(na, nb)
            if n_h > 0:
                waves = np.cos(hs_back[:n_h, None] * phi_hop[None, :] + psi_back[:n_h, None])
                aa = amp_a[:n_h, None]
                ab = amp_b[:n_h, None]
                out_ch[base : base + hop] = np.sum((aa + (ab - aa) * hop_frac) * waves, axis=0).astype(np.float32)
            amp_a, ff_a, na = amp_b, ff_b, nb
        return out_ch[:n_samples]

    back_L = synth_backing_channel(back_l_smooth, frame_lag=3, detune_mult=2.0 ** (6.5 / 1200.0), vib_hz=5.1, vib_cos=False, init_phi=1.7)
    back_R = synth_backing_channel(back_r_smooth, frame_lag=5, detune_mult=2.0 ** (-6.5 / 1200.0), vib_hz=5.4, vib_cos=True, init_phi=4.2)

    # Pan Lead tracks (`01..06`) to Stereo
    idx_f = np.arange(n_samples, dtype=np.float32) / float(hop)
    f0_i = np.clip(np.floor(idx_f).astype(np.int32), 0, len(pan_smooth) - 2)
    frac_i = idx_f - f0_i
    pan_s = (1.0 - frac_i) * pan_smooth[f0_i] + frac_i * pan_smooth[f0_i + 1]
    ang_pan = (pan_s + 1.0) * (math.pi * 0.25)
    gL = (math.sqrt(2.0) * np.cos(ang_pan)).astype(np.float32)
    gR = (math.sqrt(2.0) * np.sin(ang_pan)).astype(np.float32)

    s_m = sub_mono[:n_samples]
    b_m = body_mono[:n_samples]
    sh_m = sheen_mono[:n_samples]
    pl_m = plos_mono[:n_samples]
    sb_m = sib_mono[:n_samples]
    br_m = breath_mono[:n_samples]

    layer_sub = np.column_stack([s_m * gL, s_m * gR]).astype(np.float32)
    layer_body = np.column_stack([b_m * gL, b_m * gR]).astype(np.float32)
    layer_sheen = np.column_stack([sh_m * gL, sh_m * gR]).astype(np.float32)
    layer_plos = np.column_stack([pl_m * gL, pl_m * gR]).astype(np.float32)
    layer_sib = np.column_stack([sb_m * gL, sb_m * gR]).astype(np.float32)
    layer_breath = np.column_stack([br_m * gL, br_m * gR]).astype(np.float32)
    layer_backing = np.column_stack([back_L, back_R]).astype(np.float32)

    # Pass 5: Track 08 (`vocal_space`) — Warm Harmonic Space Send (Zero Sibilant/Plosive Splash!)
    space_send_mono = s_m + b_m + 0.50 * sh_m + 0.20 * (back_L + back_R)
    layer_space = synthesize_space_fdn(raw_fdn, space_send_mono)

    layers = {
        "lead_sub_f0": layer_sub,
        "lead_vowel_body": layer_body,
        "lead_harmonic_sheen": layer_sheen,
        "lead_plosive_burst": layer_plos,
        "lead_sibilant": layer_sib,
        "lead_velvet_breath": layer_breath,
        "backing_harmony": layer_backing,
        "vocal_space": layer_space,
    }
    mix_stereo = sum(layers.values()).astype(np.float32)
    peak_raw = float(np.max(np.abs(mix_stereo)))
    if peak_raw > 0.94:
        n_lh = (n_samples + hop - 1) // hop
        pk_h = np.zeros(n_lh, dtype=np.float32)
        abs_m = np.max(np.abs(mix_stereo), axis=1)
        for h_i in range(n_lh):
            s0 = h_i * hop
            pk_h[h_i] = float(np.max(abs_m[s0 : min(n_samples, s0 + hop)]))
        g_knots = np.ones(n_lh + 1, dtype=np.float32)
        for k_i in range(n_lh + 1):
            p_loc = max(float(pk_h[max(0, k_i - 1)]), float(pk_h[min(n_lh - 1, k_i)]))
            if p_loc > 0.94:
                g_knots[k_i] = 0.94 / p_loc
        idx_lim = np.arange(n_samples, dtype=np.float32) / float(hop)
        h0 = np.clip(np.floor(idx_lim).astype(np.int32), 0, n_lh - 1)
        fr_lim = idx_lim - h0
        g_vec = ((1.0 - fr_lim) * g_knots[h0] + fr_lim * g_knots[h0 + 1])[:, None].astype(np.float32)
        for k in layers:
            layers[k] = (layers[k] * g_vec).astype(np.float32)
        mix_stereo = (mix_stereo * g_vec).astype(np.float32)

    return {
        "mix_stereo": mix_stereo,
        "layers": layers,
        "events_count": len(events),
        "prototypes_count": len(prototypes),
    }
