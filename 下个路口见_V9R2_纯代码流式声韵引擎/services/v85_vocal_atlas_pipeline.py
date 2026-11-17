"""Step 2: Complete V8.5 8-Track Vocal Analysis, LRC Viterbi Alignment, 97-Syllable Atlas & `--` DAC Pipeline
(`services/v85_vocal_atlas_pipeline.py`) — Outputs 100% Plain-Text Mathematical Structures.
"""
from __future__ import annotations

import math
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

from parsers.lrc_syllable_parser import SyllableToken

CONS_ANCHORS = np.linspace(0.0, 1.0, 3, dtype=np.float32)
VOWEL_ANCHORS = np.array([0.0, 0.22, 0.50, 0.78, 1.0], dtype=np.float32)
DB_FLOOR = -78.0
DB_CEIL = 4.0


def hz_to_erb(f: np.ndarray | float) -> np.ndarray | float:
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, dtype=np.float64))


def erb_to_hz(erb: np.ndarray | float) -> np.ndarray | float:
    return (np.power(10.0, np.asarray(erb, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def freq_to_warped_u(f_hz: np.ndarray | float, fmin: float = 75.0, fmax: float = 19200.0) -> np.ndarray:
    f_c = np.clip(np.asarray(f_hz, dtype=np.float64), fmin, fmax)
    e0 = 21.4 * np.log10(1.0 + 0.00437 * fmin)
    e1 = 21.4 * np.log10(1.0 + 0.00437 * fmax)
    return (21.4 * np.log10(1.0 + 0.00437 * f_c) - e0) / (e1 - e0)


def build_aper_edges(n_bands: int = 8, fmin: float = 180.0, fmax: float = 19200.0) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bands + 1).astype(np.float32)


def build_lf_velvet_glottal_phase(max_harm: int = 72) -> np.ndarray:
    n_gl = 512
    t_gl = np.arange(n_gl, dtype=np.float64) / float(n_gl)
    gl_pulse = np.where(
        t_gl < 0.42,
        0.5 * (1.0 - np.cos(math.pi * t_gl / 0.42)),
        np.where(t_gl < 0.58, np.cos(0.5 * math.pi * (t_gl - 0.42) / 0.16), 0.0),
    )
    gl_deriv = np.diff(gl_pulse, prepend=gl_pulse[-1])
    gl_ph = np.angle(np.fft.rfft(gl_deriv)[1 : max_harm + 1])
    h_idx = np.arange(1, max_harm + 1, dtype=np.float64)
    velvet_disp = 0.028 * (np.maximum(0.0, h_idx - 5.0) ** 1.8)
    psi = np.arctan2(np.sin(gl_ph + velvet_disp), np.cos(gl_ph + velvet_disp))
    return psi.astype(np.float32)


def analyze_vocal_signal_48k(
    lead_direct: np.ndarray,
    config: dict,
    backing_stereo: np.ndarray | None = None,
) -> dict:
    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])
    n_fft = int(config.get("analysisFftSize", 4096))
    win_long = 3072
    n_cep = int(config["cepstralCoeffs"])
    n_aper = int(config["aperiodicBands"])
    max_harm = int(config["maxHarmonics"])
    fmin_w = float(config["harmFreqMin"])
    fmax_w = float(config["harmFreqMax"])

    n_samples = len(lead_direct)
    n_frames = n_samples // hop + 1
    bin_hz = sr / float(n_fft)

    w_bh = signal.windows.blackmanharris(win_long, sym=False).astype(np.float32)
    w_bh_sum = float(np.sum(w_bh))
    w_bh_parseval = 2.0 / math.sqrt(float(n_fft) * float(np.sum(w_bh ** 2)))
    y_pad_l = np.pad(lead_direct.astype(np.float32), (win_long // 2, win_long // 2 + hop))
    frames_long = np.lib.stride_tricks.as_strided(
        y_pad_l, shape=(n_frames, win_long), strides=(y_pad_l.strides[0] * hop, y_pad_l.strides[0])
    ) * w_bh[None, :]
    rfft_4k_abs = np.abs(np.fft.rfft(frames_long, n=n_fft, axis=1)).astype(np.float32)
    mag_4k = (rfft_4k_abs * (2.0 / w_bh_sum)).astype(np.float32)

    win_s = 1024
    w_hann = np.hanning(win_s + 1)[:-1].astype(np.float32)
    w_pow_norm = float(np.sum(w_hann ** 2))
    ac_w = np.fft.irfft(np.abs(np.fft.rfft(w_hann, n=2048)) ** 2, n=2048)[:512].astype(np.float32)
    ac_w /= float(ac_w[0])

    y_pad_s = np.pad(lead_direct.astype(np.float32), (win_s // 2, win_s // 2 + hop))
    frames_short = np.lib.stride_tricks.as_strided(
        y_pad_s, shape=(n_frames, win_s), strides=(y_pad_s.strides[0] * hop, y_pad_s.strides[0])
    ) * w_hann[None, :]
    mag_short = np.abs(np.fft.rfft(frames_short, n=2048, axis=1)).astype(np.float32)
    ac_raw = np.fft.irfft(mag_short ** 2, n=2048, axis=1)[:, :512].astype(np.float32)
    rms_short = np.sqrt(ac_raw[:, 0] / w_pow_norm + 1e-16).astype(np.float32)
    rms_peak = float(np.max(rms_short) + 1e-9)

    if backing_stereo is not None and len(backing_stereo) == n_samples:
        side_sig = (0.5 * (backing_stereo[:, 0] - backing_stereo[:, 1])).astype(np.float32)
        y_pad_sd_l = np.pad(side_sig, (win_long // 2, win_long // 2 + hop))
        frames_sd_l = np.lib.stride_tricks.as_strided(
            y_pad_sd_l, shape=(n_frames, win_long), strides=(y_pad_sd_l.strides[0] * hop, y_pad_sd_l.strides[0])
        ) * w_bh[None, :]
        rfft_sd_abs = np.abs(np.fft.rfft(frames_sd_l, n=n_fft, axis=1)).astype(np.float32)
        mag_sd4k = (rfft_sd_abs * (2.0 / w_bh_sum)).astype(np.float32)
        mag_clean = np.maximum(mag_4k * 0.08, mag_4k - 1.35 * mag_sd4k).astype(np.float32)
        rfft_clean_sq = np.maximum((rfft_4k_abs * 0.12) ** 2, (rfft_4k_abs ** 2) - 1.25 * (rfft_sd_abs ** 2))

        y_pad_sd_s = np.pad(side_sig, (win_s // 2, win_s // 2 + hop))
        frames_sd_s = np.lib.stride_tricks.as_strided(
            y_pad_sd_s, shape=(n_frames, win_s), strides=(y_pad_sd_s.strides[0] * hop, y_pad_sd_s.strides[0])
        ) * w_hann[None, :]
        mag_short_sd = np.abs(np.fft.rfft(frames_sd_s, n=2048, axis=1)).astype(np.float32)
        pow_short_clean = np.maximum((mag_short * 0.15) ** 2, (mag_short ** 2) - 1.35 * (mag_short_sd ** 2))
        ac_clean = np.fft.irfft(pow_short_clean, n=2048, axis=1)[:, :512].astype(np.float32)
    else:
        mag_clean = mag_4k
        rfft_clean_sq = rfft_4k_abs ** 2
        ac_clean = ac_raw

    low_fund_pow = np.sum(mag_short[:, 4:60] ** 2, axis=1)
    sib_retro_pow = np.sum(mag_short[:, 94:780] ** 2, axis=1)

    def _build_hysteresis_gate(core_rel: float, soft_rel: float, min_len: int, min_pk_rel: float) -> np.ndarray:
        g_core = rms_short > (rms_peak * core_rel)
        g_soft = rms_short > (rms_peak * soft_rel)
        g_hyst = ndimage.binary_propagation(g_core, mask=g_soft)
        g_cl = ndimage.binary_closing(g_hyst, structure=np.ones(4, dtype=bool))
        g_op = ndimage.binary_opening(g_cl, structure=np.ones(3, dtype=bool))
        g_mask = ndimage.binary_dilation(g_op, structure=np.ones(2, dtype=bool))
        lbl_g, n_g = ndimage.label(g_mask)
        for idx_g in range(1, n_g + 1):
            sl_g = np.flatnonzero(lbl_g == idx_g)
            if len(sl_g) < min_len or float(np.max(rms_short[sl_g])) < rms_peak * min_pk_rel:
                g_mask[sl_g] = False
        return g_mask

    sung_mask = _build_hysteresis_gate(0.055, 0.028, 5, 0.085)
    phrase_mask = _build_hysteresis_gate(0.035, 0.0035, 5, 0.055) | sung_mask
    active_mask = _build_hysteresis_gate(0.022, 0.008, 6, 0.035) | phrase_mask

    log_ms = np.log10(mag_short[:, 4:640] + 1e-4)
    flux = np.sum(np.maximum(0.0, np.diff(log_ms, axis=0, prepend=log_ms[:1])), axis=1).astype(np.float32)
    flux_p98 = float(np.percentile(flux[sung_mask], 98.0)) if np.any(sung_mask) else 1.0
    novelty = np.clip(flux / max(flux_p98, 1e-5), 0.0, 2.0).astype(np.float32)

    rms_db = 20.0 * np.log10(rms_short / rms_peak + 1e-6)
    onset_rise_db = np.maximum(0.0, rms_db - np.roll(rms_db, 2)).astype(np.float32)
    onset_rise_db[:2] = 0.0

    cands_hz = 170.0 * (2.0 ** (np.arange(136, dtype=np.float32) * 14.4 / 1200.0))
    n_c = len(cands_hz)
    score_mat = np.zeros((n_frames, n_c), dtype=np.float32)
    max_r = np.zeros(n_frames, dtype=np.float32)
    act_idx = np.flatnonzero(active_mask)

    if len(act_idx) > 0:
        r_norm = np.clip((ac_clean[act_idx] / (ac_clean[act_idx, :1] + 1e-16)) / np.maximum(ac_w[None, :], 0.22), -1.0, 1.05)
        lags = sr / cands_hz
        l0 = np.floor(lags).astype(np.int32)
        lf = (lags - l0)[None, :]
        ym1, y0, yp1 = r_norm[:, l0 - 1], r_norm[:, l0], r_norm[:, l0 + 1]
        r_cand = y0 + 0.5 * lf * (yp1 - ym1) + 0.5 * (lf ** 2) * (ym1 - 2.0 * y0 + yp1)
        max_r[act_idx] = np.max(r_cand, axis=1)

        m_act = mag_clean[act_idx]

        def _interp_mag_peak(hz_vec: np.ndarray) -> np.ndarray:
            bf = np.clip(hz_vec / bin_hz, 1.0, n_fft // 2 - 2.0)
            b0 = bf.astype(np.int32)
            fr = (bf - b0)[None, :]
            return np.maximum.reduce(
                [m_act[:, np.maximum(1, b0 - 1)], (1.0 - fr) * m_act[:, b0] + fr * m_act[:, b0 + 1], m_act[:, b0 + 1]]
            )

        a_h1 = _interp_mag_peak(cands_hz)
        a_h2 = _interp_mag_peak(2.0 * cands_hz)
        a_h3 = _interp_mag_peak(3.0 * cands_hz)
        a_h4 = _interp_mag_peak(4.0 * cands_hz)
        a_h5 = _interp_mag_peak(5.0 * cands_hz)
        a_sub = _interp_mag_peak(0.5 * cands_hz)
        a_sub3 = _interp_mag_peak(1.5 * cands_hz)

        odd_gate = np.clip(np.maximum(a_h1, 0.72 * a_h3) / (0.35 * a_h2 + 1e-6), 0.0, 1.0)
        raw_shs = a_h1 + odd_gate * (0.78 * a_h2 + 0.62 * a_h3 + 0.48 * a_h4 + 0.36 * a_h5)
        shs_norm = raw_shs / (np.max(raw_shs, axis=1, keepdims=True) + 1e-7)

        odd_to_h2 = np.maximum(a_h1, 0.75 * a_h3) / (a_h2 + 1e-6)
        suboct_pen = np.clip((0.28 - odd_to_h2) / 0.20, 0.0, 1.0) * 0.55

        has_true_sub = (0.5 * cands_hz >= 178.0)[None, :]
        sub_ratio = (0.65 * a_sub + 0.35 * a_sub3) / (a_h1 + 1e-6)
        octup_pen = np.where(has_true_sub, np.clip((sub_ratio - 0.32) / 0.35, 0.0, 1.0) * 0.50, 0.0)

        a_fourth_up1 = _interp_mag_peak((4.0 / 3.0) * cands_hz)
        a_fourth_up2 = _interp_mag_peak((8.0 / 3.0) * cands_hz)
        fourth_sub_ratio = (0.65 * a_fourth_up1 + 0.35 * a_fourth_up2) / (a_h1 + 1e-6)
        fourth_pen = np.clip((fourth_sub_ratio - 1.35) / 1.20, 0.0, 1.0) * 0.45

        oct_bias = 0.055 * np.log2(cands_hz / 170.0)[None, :]
        score_mat[act_idx] = (0.60 * r_cand + 0.45 * shs_norm + oct_bias - suboct_pen - octup_pen - fourth_pen).astype(np.float32)

    voiced = (
        active_mask
        & (max_r > 0.40)
        & (low_fund_pow > 0.42 * sib_retro_pow)
        & (rms_short > rms_peak * 0.010)
    )
    voiced = ndimage.binary_opening(ndimage.binary_closing(voiced, np.ones(5, dtype=bool)), np.ones(2, dtype=bool))

    f0 = np.zeros(n_frames, dtype=np.float32)
    labeled_v, n_isl = ndimage.label(voiced)
    fine_ratios = 2.0 ** (np.linspace(-12.0, 12.0, 25, dtype=np.float32) / 1200.0)
    for r in range(1, n_isl + 1):
        sl = np.flatnonzero(labeled_v == r)
        L_i = len(sl)
        dp = np.zeros((L_i, n_c), dtype=np.float32)
        back = np.zeros((L_i, n_c), dtype=np.int32)
        dp[0] = score_mat[sl[0]]
        for t_i in range(1, L_i):
            best_p = np.full(n_c, -1e9, dtype=np.float32)
            best_idx = np.zeros(n_c, dtype=np.int32)
            prev_row = dp[t_i - 1]
            for d in range(-10, 11):
                pen = 0.016 * (abs(d) ** 1.70)
                if d == 0:
                    m = prev_row > best_p
                    best_p[m] = prev_row[m]
                    best_idx[m] = np.arange(n_c, dtype=np.int32)[m]
                elif d > 0:
                    cand_v = prev_row[:-d] - pen
                    m = cand_v > best_p[d:]
                    best_p[d:][m] = cand_v[m]
                    best_idx[d:][m] = np.arange(n_c - d, dtype=np.int32)[m]
                else:
                    sd = -d
                    cand_v = prev_row[sd:] - pen
                    m = cand_v > best_p[:-sd]
                    best_p[:-sd][m] = cand_v[m]
                    best_idx[:-sd][m] = np.arange(sd, n_c, dtype=np.int32)[m]
            dp[t_i] = best_p + score_mat[sl[t_i]]
            back[t_i] = best_idx

        path = np.zeros(L_i, dtype=np.int32)
        path[-1] = int(np.argmax(dp[-1]))
        for t_i in range(L_i - 1, 0, -1):
            path[t_i - 1] = back[t_i, path[t_i]]

        for k_i, f_abs in enumerate(sl):
            fc_arr = cands_hz[path[k_i]] * fine_ratios
            row = mag_clean[f_abs]
            sc = np.zeros_like(fc_arr)
            for h in range(1, 7):
                bf = np.clip(fc_arr * h / bin_hz, 1.0, n_fft // 2 - 2.0)
                b0 = bf.astype(np.int32)
                fr = bf - b0
                sc += ((1.0 - fr) * row[b0] + fr * row[b0 + 1]) / (h ** 0.4)
            f0[f_abs] = float(fc_arr[int(np.argmax(sc))])
        if L_i >= 4:
            f0[sl] = np.power(2.0, ndimage.gaussian_filter1d(np.log2(f0[sl]), sigma=0.55)).astype(np.float32)

    harm_dom = np.clip(low_fund_pow / (low_fund_pow + 0.65 * sib_retro_pow + 1e-14), 0.0, 1.0)
    raw_vdeg = np.where(voiced, np.clip(0.58 + 0.42 * harm_dom, 0.0, 1.0), 0.0).astype(np.float32)
    voicing_deg = ndimage.gaussian_filter1d(raw_vdeg, sigma=0.95).astype(np.float32)
    voicing_deg[~voiced] = np.minimum(voicing_deg[~voiced], 0.12)
    voicing_deg[~(active_mask | (rms_short > rms_peak * 0.0012))] = 0.0

    m_pos = np.arange(1, n_cep + 1, dtype=np.float64)[None, :]
    u_grid128 = np.linspace(0.0, 1.0, 128, dtype=np.float64)
    B_grid = np.cos(math.pi * u_grid128[:, None] * m_pos)
    reg_grid = 0.012 * (1.0 + np.maximum(0.0, (np.arange(1, n_cep + 1, dtype=np.float64) - 32.0) / 10.0) ** 2)
    P_harm = np.linalg.solve(B_grid.T @ B_grid + np.diag(reg_grid), B_grid.T)

    freqs_s = np.fft.rfftfreq(2048, 1.0 / sr)
    erb_nodes = np.linspace(float(hz_to_erb(fmin_w)), float(hz_to_erb(fmax_w)), 74, dtype=np.float64)
    erb_c = erb_to_hz(erb_nodes[1:-1])
    u_erb = freq_to_warped_u(erb_c, fmin_w, fmax_w)[:, None]
    B_unv = np.cos(math.pi * u_erb * m_pos)
    lhs_unv_inv = np.linalg.solve(B_unv.T @ B_unv + np.diag(reg_grid * 2.0), B_unv.T)
    erb_fs = hz_to_erb(freqs_s)
    fb_unv = np.zeros((72, len(freqs_s)), dtype=np.float64)
    for b in range(72):
        lo, mid_e, hi = erb_nodes[b], erb_nodes[b + 1], erb_nodes[b + 2]
        wb = np.maximum(0.0, np.minimum((erb_fs - lo) / max(mid_e - lo, 1e-9), (hi - erb_fs) / max(hi - mid_e, 1e-9)))
        fb_unv[b] = wb / (np.sum(wb) + 1e-12)

    aper_edges = build_aper_edges(n_aper, float(config["aperFreqMin"]), float(config["aperFreqMax"]))

    total_gain = np.zeros(n_frames, dtype=np.float32)
    gain_harm = np.zeros(n_frames, dtype=np.float32)
    gain_aper = np.zeros(n_frames, dtype=np.float32)
    mcep_frames = np.zeros((n_frames, n_cep), dtype=np.float32)
    aper_log_shape = np.full((n_frames, n_aper), -3.2, dtype=np.float32)
    unv_aper_default = np.array([-4.0, -4.0, -3.8, -3.5, -3.2, -2.8, -2.4, -2.2], dtype=np.float32)

    nonsilent_idx = np.flatnonzero(active_mask | (rms_short > rms_peak * 0.0012))
    for t in nonsilent_idx:
        g_tot = float(rms_short[t])
        if not voiced[t] and g_tot > rms_peak * 0.22:
            knee = rms_peak * 0.22
            g_tot = knee + (g_tot - knee) * 0.55
        total_gain[t] = g_tot

        if voiced[t] and f0[t] > 0.0:
            ff = float(f0[t])
            max_h = int(min(max_harm, math.floor(19000.0 / ff)))
            hs = np.arange(1, max_h + 1, dtype=np.float64)
            h_hz = hs * ff

            raw_row_sq = rfft_clean_sq[t].astype(np.float64)
            cs_sq = np.insert(np.cumsum(raw_row_sq), 0, 0.0)
            half_bw_hz = np.minimum(0.18 * ff, np.maximum(38.0, 0.032 * h_hz))
            k_lo = np.clip(np.floor((h_hz - half_bw_hz) / bin_hz).astype(np.int32), 1, n_fft // 2 - 2)
            k_hi = np.clip(np.ceil((h_hz + half_bw_hz) / bin_hz).astype(np.int32), k_lo + 1, n_fft // 2 - 1)
            vor_sum_sq = np.maximum(0.0, cs_sq[k_hi + 1] - cs_sq[k_lo])
            amps = np.sqrt(vor_sum_sq) * w_bh_parseval

            row = mag_clean[t]
            bv = np.clip((hs - 0.5) * ff / bin_hz, 1.0, n_fft // 2 - 2.0).astype(np.int32)
            vls = np.minimum.reduce([row[bv - 1], row[bv], row[bv + 1]])
            amps_clean = np.sqrt(np.maximum(amps ** 2 - 0.65 * (vls ** 2), (amps * 0.15) ** 2))

            rms_h_long = math.sqrt(0.5 * float(np.sum(amps_clean ** 2)) + 1e-16)
            y_db = np.clip(20.0 * np.log10(np.maximum(amps_clean / max(rms_h_long, 1e-7), 1e-4)), -68.0, 26.0)

            u_h = freq_to_warped_u(h_hz, fmin_w, fmax_w)
            y_g = np.interp(u_grid128, u_h, y_db, left=float(y_db[0]), right=float(y_db[-1]))
            y_zm = y_g - float(np.mean(y_g))
            mcep_frames[t] = (P_harm @ y_zm).astype(np.float32)

            hnr_vec = np.zeros(n_aper, dtype=np.float32)
            for ab in range(n_aper):
                lo_f, hi_f = float(aper_edges[ab]), float(aper_edges[ab + 1])
                fc_a = math.sqrt(lo_f * hi_f)
                mask_b = (h_hz >= lo_f) & (h_hz < hi_f)
                if np.any(mask_b):
                    p_pk = float(np.mean(amps_clean[mask_b] ** 2)) + 1e-16
                    p_vl = float(np.mean(vls[mask_b] ** 2)) + 1e-18
                    db_rel = 10.0 * math.log10(p_vl / p_pk)
                else:
                    db_rel = -36.0 if fc_a < 2000.0 else -24.0
                if fc_a < 1500.0:
                    db_rel = float(np.clip(db_rel, -48.0, -34.0))
                elif fc_a < 4200.0:
                    db_rel = float(np.clip(db_rel, -42.0, -24.0))
                else:
                    db_rel = float(np.clip(db_rel, -36.0, -17.5))
                hnr_vec[ab] = db_rel / 10.0
            aper_log_shape[t] = hnr_vec

            vd = float(voicing_deg[t])
            gain_harm[t] = g_tot * math.sqrt(max(0.15, vd))
            gain_aper[t] = g_tot * math.sqrt(max(0.0, 1.0 - vd))
        else:
            p_short = mag_short[t] ** 2
            erb_p = fb_unv @ p_short
            floor_p = max(float(np.max(erb_p)) * 1e-5, 1e-16)
            erb_db = 10.0 * np.log10(np.maximum(erb_p, floor_p))
            erb_zm = erb_db - float(np.mean(erb_db))
            mcep_frames[t] = (lhs_unv_inv @ erb_zm).astype(np.float32)
            aper_log_shape[t] = unv_aper_default
            gain_harm[t] = 0.0
            gain_aper[t] = g_tot

    return {
        "f0": f0,
        "voiced": voiced,
        "voicing_deg": voicing_deg,
        "active_mask": active_mask,
        "phrase_mask": phrase_mask,
        "sung_mask": sung_mask,
        "total_gain": total_gain,
        "gain_harm": gain_harm,
        "gain_aper": gain_aper,
        "mcep_frames": mcep_frames,
        "aper_log_shape": aper_log_shape,
        "novelty": novelty,
        "onset_rise_db": onset_rise_db,
        "back_pitch_semi": np.zeros(n_frames, dtype=np.float32),
        "glottal_phase": build_lf_velvet_glottal_phase(max_harm),
    }


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    labeled, n = ndimage.label(mask)
    runs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(labeled == i)
        runs.append((int(idx[0]), int(idx[-1]) + 1))
    return runs


def _refine_line_cut(
    target_frame: int,
    sung_mask: np.ndarray,
    total_rms: np.ndarray,
    novelty: np.ndarray,
    lo_back: int = 65,
    hi_fwd: int = 40,
) -> int:
    n_frames = len(sung_mask)
    lo = max(0, target_frame - lo_back)
    hi = min(n_frames - 1, target_frame + hi_fwd)
    if lo >= hi:
        return int(np.clip(target_frame, 0, n_frames - 1))

    sub_inactive = ~sung_mask[lo : hi + 1]
    sil_runs = [(lo + a, lo + b) for a, b in _find_runs(sub_inactive)]
    long_sil = [r for r in sil_runs if (r[1] - r[0]) >= 3]
    if long_sil:
        def _sil_score(r: tuple[int, int]) -> float:
            r_s, r_e = r
            dist = 0 if (r_s <= target_frame <= r_e) else min(abs(target_frame - r_s), abs(target_frame - r_e))
            return min(float(r_e - r_s), 36.0) - 0.28 * float(dist)

        best_s, best_e = max(long_sil, key=_sil_score)
        return int(min(best_e - 1, max(best_s, best_e - 2)))

    if sil_runs:
        inactive = [f for r_s, r_e in sil_runs for f in range(r_s, r_e)]
        return min(inactive, key=lambda f: abs(f - target_frame))

    idx_r = np.arange(lo, hi + 1)
    sc = -np.log10(total_rms[idx_r] + 1e-5) + 1.5 * novelty[idx_r] - 0.022 * np.abs(idx_r - target_frame)
    return int(idx_r[int(np.argmax(sc))])


def _viterbi_split_island(
    seg_s: int,
    seg_e: int,
    tokens: list[SyllableToken],
    features: dict,
) -> list[tuple[int, int, SyllableToken]]:
    m = len(tokens)
    L = seg_e - seg_s
    if m <= 1 or L <= m * 4:
        cuts = [seg_s + int(round(k * L / m)) for k in range(m + 1)]
        return [(cuts[k], max(cuts[k] + 1, cuts[k + 1]), tokens[k]) for k in range(m)]

    nov = features["novelty"][seg_s:seg_e]
    gh = features["gain_harm"][seg_s:seg_e]
    ga = features["gain_aper"][seg_s:seg_e]
    v_sub = features["voiced"][seg_s:seg_e]
    v_prefix = np.concatenate([[0], np.cumsum(v_sub.astype(np.int32))])
    island_has_vowels = int(v_prefix[L]) >= m * 2

    hi_start = max(0, features["aper_log_shape"].shape[1] // 2)
    aper_hi = np.mean(features["aper_log_shape"][seg_s:seg_e, hi_start:], axis=1)
    tot_e = np.sqrt(gh ** 2 + ga ** 2 + 1e-12)
    isl_med_e = float(np.median(tot_e[tot_e > 1e-4])) if np.any(tot_e > 1e-4) else 1e-3
    strong_e = (tot_e > isl_med_e * 0.35).astype(np.int32)
    e_prefix = np.concatenate([[0], np.cumsum(strong_e)])
    log_e = np.log10(tot_e + 1e-5)

    min_len = max(6, min(12, L // (m * 2)))
    avg_len = L / float(m)

    sib_vec = np.array([float(np.mean(ga[p : min(L, p + 6)] / (tot_e[p : min(L, p + 6)] + 1e-7))) for p in range(L)])
    hi_vec = np.array([float(np.mean(aper_hi[p : min(L, p + 6)])) for p in range(L)])
    dip_vec = -0.50 * np.array([float(log_e[max(0, p - 1)]) for p in range(L)])

    vot_trans = np.zeros(L, dtype=np.float64)
    for p in range(1, L):
        if (not v_sub[p - 1]) and v_sub[p]:
            vot_trans[p] = -5.2
        elif v_sub[p - 1] and (not v_sub[p]) and (v_prefix[min(L, p + 18)] - v_prefix[p]) >= 3:
            vot_trans[p] = 2.6

    ob = np.zeros((m, L), dtype=np.float64)
    for k, tok in enumerate(tokens):
        base_s = 2.0 * nov + dip_vec + vot_trans
        if tok.initial_class == "sibilant":
            base_s += 1.65 * sib_vec + 0.08 * hi_vec
        elif tok.initial_class == "plosive":
            base_s += 1.05 * nov + 0.75 * sib_vec
        else:
            base_s += 0.45 * (1.0 - sib_vec)
        ob[k] = base_s

    def _calc_span_pen(p_s_arr: np.ndarray, p_end: int) -> np.ndarray:
        pen = np.zeros_like(p_s_arr, dtype=np.float64)
        if island_has_vowels:
            vc = v_prefix[p_end] - v_prefix[p_s_arr]
            pen += np.where(vc == 0, -14.0, np.where(vc < 3, -7.5, 0.0))
        ec = e_prefix[p_end] - e_prefix[p_s_arr]
        pen += np.where(ec == 0, -14.0, np.where(ec < 3, -5.0, 0.0))
        return pen

    dp = np.full((m, L + 1), -1e18, dtype=np.float64)
    back = np.zeros((m, L + 1), dtype=np.int32)

    p1_arr = np.arange(min_len, L - (m - 1) * min_len + 1)
    d0 = p1_arr.astype(np.float64)
    t0_d = avg_len * (1.45 if m == 1 else 0.92)
    sp0_d = avg_len * (1.35 if m == 1 else 0.62) + 4.0
    span0 = np.array([_calc_span_pen(np.array([0], dtype=np.int32), int(p1))[0] for p1 in p1_arr])
    dp[0, p1_arr] = ob[0, 0] - 0.85 * ((d0 - t0_d) / sp0_d) ** 2 + span0

    for k in range(1, m):
        is_last = k == (m - 1)
        p_range = [L] if is_last else list(range((k + 1) * min_len, L - (m - 1 - k) * min_len + 1))
        prev_min = k * min_len
        target_d = avg_len * (1.40 if is_last else 0.94)
        spread_d = avg_len * (1.25 if is_last else 0.60) + 4.0
        for p_end in p_range:
            ps = np.arange(prev_min, p_end - min_len + 1)
            durs = (p_end - ps).astype(np.float64)
            vals = dp[k - 1, ps] + ob[k, ps] - 0.85 * ((durs - target_d) / spread_d) ** 2 + _calc_span_pen(ps, p_end)
            bi = int(np.argmax(vals))
            dp[k, p_end] = float(vals[bi])
            back[k, p_end] = int(ps[bi])

    cuts = [0] * (m + 1)
    cuts[m] = L
    cur = L
    for k in range(m - 1, 0, -1):
        cur = int(back[k, cur])
        cuts[k] = cur
    cuts[0] = 0
    return [(seg_s + cuts[k], seg_s + cuts[k + 1], tokens[k]) for k in range(m)]


def align_v85_syllable_events(sung_lines: list[dict], features: dict, config: dict) -> list[dict]:
    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])
    active_mask = features["active_mask"]
    phrase_mask = features["phrase_mask"]
    sung_mask = features["sung_mask"]
    n_frames = len(active_mask)
    gh = features["gain_harm"]
    ga = features["gain_aper"]
    tot_rms = np.sqrt(gh ** 2 + ga ** 2 + 1e-14)
    rms_peak = float(np.max(tot_rms) + 1e-9)
    novelty = features["novelty"]
    voiced = features["voiced"]

    claimed = np.zeros(n_frames, dtype=bool)
    raw_events: list[dict] = []

    line_windows: list[tuple[int, int]] = []
    for idx, ln in enumerate(sung_lines):
        raw_s = max(0, int(round(ln["startSec"] * sr / hop)))
        raw_e = min(n_frames - 1, int(round(ln["endSec"] * sr / hop)))
        if idx == 0:
            s_cut = _refine_line_cut(raw_s, sung_mask, tot_rms, novelty, lo_back=120, hi_fwd=35)
        elif abs(ln["startSec"] - sung_lines[idx - 1]["endSec"]) < 0.05:
            s_cut = line_windows[-1][1]
        else:
            s_cut = max(line_windows[-1][1], _refine_line_cut(raw_s, sung_mask, tot_rms, novelty))
        e_cut = _refine_line_cut(raw_e, sung_mask, tot_rms, novelty)
        if e_cut <= s_cut:
            e_cut = min(n_frames, s_cut + len(ln["tokens"]) * 8)
        line_windows.append((s_cut, e_cut))

    for ln, (win_s, win_e) in zip(sung_lines, line_windows):
        sub_mask = sung_mask[win_s:win_e]
        runs = [(win_s + a, win_s + b) for a, b in _find_runs(sub_mask) if (b - a) >= 4]
        tokens: list[SyllableToken] = ln["tokens"]
        pgroups: list[list[SyllableToken]] = ln.get("phraseGroups") or [tokens]
        if not runs:
            continue

        merged_runs = [runs[0]]
        for r_s, r_e in runs[1:]:
            if r_s - merged_runs[-1][1] <= 8:
                merged_runs[-1] = (merged_runs[-1][0], r_e)
            else:
                merged_runs.append((r_s, r_e))
        runs = merged_runs

        cluster_runs = [runs[0]]
        for r_s, r_e in runs[1:]:
            if r_s - cluster_runs[-1][1] > 280:
                break
            cluster_runs.append((r_s, r_e))
        runs = cluster_runs

        while len(runs) > len(pgroups) and len(pgroups) >= 1:
            gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
            min_g_idx = int(np.argmin(gaps))
            if gaps[min_g_idx] <= 15:
                runs = runs[:min_g_idx] + [(runs[min_g_idx][0], runs[min_g_idx + 1][1])] + runs[min_g_idx + 2 :]
            else:
                break

        if len(runs) == 1:
            alloc = [len(tokens)]
        elif len(runs) == len(pgroups):
            alloc = [len(g) for g in pgroups]
        else:
            if len(runs) > len(tokens):
                runs = sorted(
                    sorted(runs, key=lambda r: float(np.sum(tot_rms[r[0] : r[1]] ** 2)), reverse=True)[: len(tokens)],
                    key=lambda r: r[0],
                )
            weights = []
            for r_s, r_e in runs:
                env_sm = ndimage.gaussian_filter1d(tot_rms[r_s:r_e], sigma=2.0)
                pks, _ = signal.find_peaks(env_sm, distance=12, prominence=float(np.max(env_sm)) * 0.12)
                weights.append(0.65 * max(1, len(pks)) + 0.35 * ((r_e - r_s) / 38.0))
            w_arr = np.array(weights, dtype=np.float64)
            raw_counts = np.maximum(1, np.round(w_arr / np.sum(w_arr) * len(tokens)).astype(int))
            while int(np.sum(raw_counts)) > len(tokens):
                idx_dec = int(np.argmax(raw_counts - w_arr / np.sum(w_arr) * len(tokens)))
                if raw_counts[idx_dec] > 1:
                    raw_counts[idx_dec] -= 1
                else:
                    raw_counts[int(np.argmax(raw_counts))] -= 1
            while int(np.sum(raw_counts)) < len(tokens):
                idx_inc = int(np.argmax(w_arr / np.sum(w_arr) * len(tokens) - raw_counts))
                raw_counts[idx_inc] += 1
            alloc = [int(c) for c in raw_counts]

        tok_cursor = 0
        for (r_s, r_e), cnt in zip(runs, alloc):
            sub_toks = tokens[tok_cursor : tok_cursor + cnt]
            tok_cursor += cnt
            for sp_s, sp_e, tok in _viterbi_split_island(r_s, r_e, sub_toks, features):
                raw_events.append(
                    {
                        "start_frame": sp_s,
                        "end_frame": sp_e,
                        "display_text": tok.display,
                        "pinyin": tok.pinyin,
                        "initial_class": tok.initial_class,
                        "is_hook": False,
                    }
                )

    raw_events.sort(key=lambda e: e["start_frame"])

    for idx, ev in enumerate(raw_events):
        bound_lo = raw_events[idx - 1]["end_frame"] if idx > 0 else 0
        if ev["start_frame"] > bound_lo:
            max_back = max(bound_lo, ev["start_frame"] - 18)
            cur_s = ev["start_frame"]
            for f in range(cur_s - 1, max_back - 1, -1):
                if phrase_mask[f]:
                    cur_s = f
                else:
                    break
            ev["start_frame"] = cur_s

        bound_hi = raw_events[idx + 1]["start_frame"] if (idx + 1 < len(raw_events)) else n_frames
        if ev["end_frame"] < bound_hi:
            orig_e = ev["end_frame"]
            max_fwd = min(bound_hi, orig_e + 110)
            min_r = float(tot_rms[max(0, orig_e - 1)])
            cur_e = orig_e
            for f in range(orig_e, max_fwd):
                if not phrase_mask[f]:
                    break
                cur_r = float(tot_rms[f])
                if cur_r < min_r:
                    min_r = cur_r
                elif (f - orig_e) >= 10 and cur_r > max(min_r * 3.5, rms_peak * 0.085):
                    break
                cur_e = f + 1
            ev["end_frame"] = cur_e

        claimed[ev["start_frame"] : ev["end_frame"]] = True

    claimed_padded = ndimage.binary_dilation(claimed, structure=np.ones(41, dtype=bool))
    unclaimed_active = active_mask & (~claimed_padded)
    hook_runs = _find_runs(unclaimed_active)
    for h_idx, (r_s, r_e) in enumerate(hook_runs):
        L = r_e - r_s
        if L < 14 or float(np.max(tot_rms[r_s:r_e])) < rms_peak * 0.038:
            continue
        next_lim = hook_runs[h_idx + 1][0] if (h_idx + 1 < len(hook_runs)) else n_frames
        max_h_fwd = min(next_lim, r_e + 45)
        min_hr = float(tot_rms[max(0, r_e - 1)])
        for f in range(r_e, max_h_fwd):
            if claimed[f] or not phrase_mask[f]:
                break
            cr = float(tot_rms[f])
            if cr < min_hr:
                min_hr = cr
            elif (f - r_e) >= 6 and cr > max(min_hr * 2.1, rms_peak * 0.030):
                break
            r_e = f + 1
        L = r_e - r_s
        seg_nov = novelty[r_s:r_e]
        peaks, _ = signal.find_peaks(seg_nov, distance=19, prominence=0.14)
        valid_cuts = [0] + [int(p) for p in peaks if 9 <= p <= L - 9] + [L]
        for k in range(len(valid_cuts) - 1):
            raw_events.append(
                {
                    "start_frame": r_s + valid_cuts[k],
                    "end_frame": r_s + valid_cuts[k + 1],
                    "display_text": "♪",
                    "pinyin": "_hook",
                    "initial_class": "plosive",
                    "is_hook": True,
                }
            )

    raw_events.sort(key=lambda e: e["start_frame"])

    merged_events: list[dict] = []
    for ev in raw_events:
        if (
            merged_events
            and ev["is_hook"]
            and (not merged_events[-1]["is_hook"])
            and (ev["start_frame"] - merged_events[-1]["end_frame"]) <= 34
        ):
            merged_events[-1]["end_frame"] = max(merged_events[-1]["end_frame"], ev["end_frame"])
        else:
            merged_events.append(ev)
    raw_events = merged_events

    for idx in range(len(raw_events) - 1):
        e_prev = raw_events[idx]["end_frame"]
        s_next = raw_events[idx + 1]["start_frame"]
        gap = s_next - e_prev
        if gap <= 0:
            if s_next < e_prev:
                mid_f = (e_prev + s_next) // 2
                raw_events[idx]["end_frame"] = max(raw_events[idx]["start_frame"] + 1, mid_f)
                raw_events[idx + 1]["start_frame"] = raw_events[idx]["end_frame"]
            continue
        if gap <= 34 or (gap <= 44 and bool(np.all(phrase_mask[e_prev:s_next]))):
            cut_f = e_prev + int(np.argmin(tot_rms[e_prev : s_next + 1]))
            raw_events[idx]["end_frame"] = cut_f
            raw_events[idx + 1]["start_frame"] = cut_f

    for idx, ev in enumerate(raw_events):
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = max(1, e_f - s_f)
        ev["end_frame"] = s_f + L
        leg_prev = bool(idx > 0 and raw_events[idx - 1]["end_frame"] == s_f)
        leg_next = bool(idx + 1 < len(raw_events) and raw_events[idx + 1]["start_frame"] == (s_f + L))
        max_cons = min(18, max(1, L - 2))
        vot = 2 if L >= 6 else 1
        found_v = False
        for rel in range(max_cons):
            f_abs = s_f + rel
            if voiced[f_abs] and gh[f_abs] >= 0.65 * ga[f_abs]:
                if rel == 0 and leg_prev and ev["initial_class"] not in ("plosive", "sibilant"):
                    vot = 0
                else:
                    vot = max(1, rel)
                found_v = True
                break
        if not found_v:
            vot = min(max_cons, max(2, L // 3))
        ev["cons_frames"] = int(vot)
        ev["total_frames"] = int(L)
        ev["legato_prev"] = leg_prev
        ev["legato_next"] = leg_next

    return raw_events


def _sample_curve_2d(mat: np.ndarray, u_knots: np.ndarray) -> np.ndarray:
    L, _ = mat.shape
    if L <= 1:
        return np.repeat(mat[:1], len(u_knots), axis=0).astype(np.float32)
    idx_f = u_knots * (L - 1)
    i0 = np.clip(np.floor(idx_f).astype(np.int32), 0, L - 1)
    i1 = np.clip(i0 + 1, 0, L - 1)
    frac = (idx_f - i0)[:, None]
    return ((1.0 - frac) * mat[i0] + frac * mat[i1]).astype(np.float32)


def _build_envelope_knot_positions(total_frames: int, cons_frames: int, vowel_step: int) -> np.ndarray:
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


def _build_adaptive_expr_knot_positions(total_frames: int, cons_frames: int, expr_step: int) -> np.ndarray:
    if total_frames <= 1:
        return np.array([0], dtype=np.int32)
    if expr_step <= 1:
        return np.arange(total_frames, dtype=np.int32)
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


def _build_uniform_knot_positions(total_frames: int, step: int) -> np.ndarray:
    if total_frames <= 1:
        return np.array([0], dtype=np.int32)
    if step <= 1:
        return np.arange(total_frames, dtype=np.int32)
    last = total_frames - 1
    pts = list(range(0, last, max(1, step)))
    if pts[-1] != last:
        pts.append(last)
    return np.array(pts, dtype=np.int32)


def _gain_to_uint8(gain: np.ndarray | float, ref_peak: float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(gain, dtype=np.float32))
    ratio = np.maximum(arr / max(ref_peak, 1e-8), 0.0)
    out = np.zeros(len(arr), dtype=np.uint8)
    nz = ratio > (10.0 ** (DB_FLOOR / 20.0))
    if np.any(nz):
        db = 20.0 * np.log10(ratio[nz])
        q = np.round((np.clip(db, DB_FLOOR, DB_CEIL) - DB_FLOOR) / (DB_CEIL - DB_FLOOR) * 254.0) + 1.0
        out[nz] = np.clip(q, 1, 255).astype(np.uint8)
    return out


def _eval_proto_on_event(proto: dict, mean_mcep: np.ndarray, aper_mean: np.ndarray, L: int, cf: int, q_scale: float):
    vot = min(L - 1, cf) if (L > 1 and cf > 0) else 0
    pred_m = np.zeros((L, len(mean_mcep)), dtype=np.float32)
    pred_a = np.zeros((L, len(aper_mean)), dtype=np.float32)
    cm = np.asarray(proto["consMcep"], dtype=np.float32) * q_scale
    ca = np.asarray(proto["consAper"], dtype=np.float32) * q_scale
    vm = np.asarray(proto["vowelMcep"], dtype=np.float32) * q_scale
    va = np.asarray(proto["vowelAper"], dtype=np.float32) * q_scale

    if vot > 0:
        u_c = np.linspace(0.0, 1.0, vot, dtype=np.float32) * (len(CONS_ANCHORS) - 1) if vot > 1 else np.zeros(1, dtype=np.float32)
        ic0 = np.clip(np.floor(u_c).astype(np.int32), 0, len(CONS_ANCHORS) - 2)
        fc = (u_c - ic0)[:, None]
        pred_m[:vot] = mean_mcep[None, :] + (1.0 - fc) * cm[ic0] + fc * cm[ic0 + 1]
        pred_a[:vot] = aper_mean[None, :] + (1.0 - fc) * ca[ic0] + fc * ca[ic0 + 1]

    v_len = L - vot
    if v_len > 0:
        u_v = np.linspace(0.0, 1.0, v_len, dtype=np.float32) * (len(VOWEL_ANCHORS) - 1) if v_len > 1 else np.zeros(1, dtype=np.float32)
        iv0 = np.clip(np.floor(u_v).astype(np.int32), 0, len(VOWEL_ANCHORS) - 2)
        fv = (u_v - iv0)[:, None]
        pred_m[vot:] = mean_mcep[None, :] + (1.0 - fv) * vm[iv0] + fv * vm[iv0 + 1]
        pred_a[vot:] = aper_mean[None, :] + (1.0 - fv) * va[iv0] + fv * va[iv0 + 1]
    return pred_m, pred_a


def _build_coarticulated_grids(events: list[dict], prototypes: list[dict], atlas_step_db: float, mean_mcep: np.ndarray, aper_mean: np.ndarray, total_frames_song: int):
    n_cep = len(mean_mcep)
    n_aper = len(aper_mean)
    base_cep = np.zeros((total_frames_song, n_cep), dtype=np.float32)
    base_aper = np.full((total_frames_song, n_aper), -3.6, dtype=np.float32)

    for ev in events:
        L = int(ev["total_frames"])
        if L <= 0:
            continue
        s_f = int(ev["start_frame"])
        f_end = min(total_frames_song, s_f + L)
        span = f_end - s_f
        if span <= 0:
            continue
        pm, pa = _eval_proto_on_event(prototypes[ev["syllable_id"]], mean_mcep, aper_mean, L, int(ev["cons_frames"]), atlas_step_db)
        base_cep[s_f:f_end] = pm[:span]
        base_aper[s_f:f_end] = pa[:span]

    seams = []
    for i in range(1, len(events)):
        prev_ev = events[i - 1]
        curr_ev = events[i]
        b = int(curr_ev["start_frame"])
        if prev_ev["total_frames"] >= 2 and curr_ev["total_frames"] >= 2 and (prev_ev["start_frame"] + prev_ev["total_frames"] == b) and (1 <= b < total_frames_song):
            w_pre = min(18, (prev_ev["total_frames"] * 12) // 25, b)
            w_post = min(4 if curr_ev["is_plosive"] else 10, (curr_ev["total_frames"] * 9) // 25, total_frames_song - b)
            if w_pre >= 1 and w_post >= 1:
                seams.append((b, w_pre, w_post, base_cep[b].copy() - base_cep[b - 1].copy(), base_aper[b].copy() - base_aper[b - 1].copy()))

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


def compile_plain_v85_vocal_bundle(
    raw_events: list[dict],
    features: dict,
    stems: dict,
    config: dict,
) -> dict:
    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])
    n_samples = len(stems["lead_mono"])
    mcep_frames = features["mcep_frames"]
    aper_frames = features["aper_log_shape"]
    tot_gain = features["total_gain"]
    voiced = features["voiced"]
    n_cep = int(config["cepstralCoeffs"])
    n_aper = int(config["aperiodicBands"])

    prof_spec = config["profiles"]["pure"]
    k_pca = int(prof_spec["pcaDims"])
    k_aper = int(prof_spec["aperExprBands"])
    q_step = float(prof_spec["quantStepDb"])
    atlas_step_db = 0.10

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
    prototypes: list[dict] = []
    pinyin_to_id: dict[str, int] = {}

    for sid, py in enumerate(sorted_pinyins):
        evs = lyric_groups[py]
        cons_m_acc = np.zeros((len(CONS_ANCHORS), n_cep), dtype=np.float64)
        cons_a_acc = np.zeros((len(CONS_ANCHORS), n_aper), dtype=np.float64)
        cons_w = 0.0
        v_m_acc = np.zeros((len(VOWEL_ANCHORS), n_cep), dtype=np.float64)
        v_a_acc = np.zeros((len(VOWEL_ANCHORS), n_aper), dtype=np.float64)
        v_w = 0.0

        for ev in evs:
            s_f, e_f = ev["start_frame"], ev["end_frame"]
            cf_raw = int(ev["cons_frames"])
            if cf_raw > 0 and (e_f - s_f) > 1:
                vot = min(e_f - s_f - 1, cf_raw)
                cw = float(np.mean(tot_gain[s_f : s_f + vot]) + 1e-5)
                cons_m_acc += cw * _sample_curve_2d(mcep_frames[s_f : s_f + vot] - mean_mcep[None, :], CONS_ANCHORS)
                cons_a_acc += cw * _sample_curve_2d(aper_frames[s_f : s_f + vot] - aper_mean[None, :], CONS_ANCHORS)
                cons_w += cw
            else:
                vot = 0
            v_start = s_f + vot if (s_f + vot) < e_f else s_f
            vw = float(np.mean(tot_gain[v_start:e_f]) + 1e-5)
            v_m_acc += vw * _sample_curve_2d(mcep_frames[v_start:e_f] - mean_mcep[None, :], VOWEL_ANCHORS)
            v_a_acc += vw * _sample_curve_2d(aper_frames[v_start:e_f] - aper_mean[None, :], VOWEL_ANCHORS)
            v_w += vw

        v_m = (v_m_acc / max(v_w, 1e-9)).astype(np.float32)
        v_a = (v_a_acc / max(v_w, 1e-9)).astype(np.float32)
        c_m = (cons_m_acc / cons_w).astype(np.float32) if cons_w > 1e-9 else np.repeat(v_m[:1], len(CONS_ANCHORS), axis=0)
        c_a = (cons_a_acc / cons_w).astype(np.float32) if cons_w > 1e-9 else np.repeat(v_a[:1], len(CONS_ANCHORS), axis=0)

        prototypes.append(
            {
                "id": sid,
                "pinyin": py,
                "initialClass": evs[0]["initial_class"],
                "count": len(evs),
                "consMcep": np.round(c_m / atlas_step_db).astype(int).tolist(),
                "consAper": np.round(c_a / atlas_step_db).astype(int).tolist(),
                "vowelMcep": np.round(v_m / atlas_step_db).astype(int).tolist(),
                "vowelAper": np.round(v_a / atlas_step_db).astype(int).tolist(),
            }
        )
        pinyin_to_id[py] = sid

    proto_sigs = np.array(
        [
            np.concatenate(
                [
                    np.asarray(p["vowelMcep"], dtype=np.float32).mean(axis=0) * atlas_step_db,
                    0.35 * np.asarray(p["consMcep"], dtype=np.float32).mean(axis=0) * atlas_step_db,
                ]
            )
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
            best_sid = int(np.argmin(np.sum((proto_sigs - sig[None, :]) ** 2, axis=1)))
            ev["syllable_id"] = best_sid
            ev["pinyin"] = prototypes[best_sid]["pinyin"]
            ev["display_text"] = f"♪{prototypes[best_sid]['pinyin']}"
            prototypes[best_sid]["count"] += 1
        else:
            ev["syllable_id"] = pinyin_to_id[ev["pinyin"]]

        proto = prototypes[ev["syllable_id"]]
        cons_end = min(e_f, s_f + max(2, ev["cons_frames"]))
        max_rise = float(np.max(onset_rise_db[s_f:cons_end])) if cons_end > s_f else 0.0
        ev["is_plosive"] = bool(proto["initialClass"] == "plosive" or (ev["is_hook"] and max_rise > 7.5))

    total_frames_song = len(mcep_frames) + 6
    base_cep_grid, base_aper_grid = _build_coarticulated_grids(
        raw_events, prototypes, atlas_step_db, mean_mcep, aper_mean, total_frames_song
    )

    mcep_res_w, mcep_res_all, aper_res_all = [], [], []
    tot_var_raw, tot_var_after_atlas = 0.0, 0.0
    for ev in raw_events:
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        actual_m = mcep_frames[s_f:e_f]
        res_m = actual_m - base_cep_grid[s_f:e_f]
        res_a = aper_frames[s_f:e_f] - base_aper_grid[s_f:e_f]
        w_sub = np.clip(tot_gain[s_f:e_f] / ref_peak, 0.02, 1.0)[:, None]
        mcep_res_w.append(res_m * np.sqrt(w_sub))
        mcep_res_all.append(res_m)
        aper_res_all.append(res_a)
        tot_var_raw += float(np.sum((actual_m ** 2) * w_sub))
        tot_var_after_atlas += float(np.sum((res_m ** 2) * w_sub))

    res_m_w_mat = np.vstack(mcep_res_w)
    res_m_mat = np.vstack(mcep_res_all)
    res_a_mat = np.vstack(aper_res_all)

    _, _, vt_m = np.linalg.svd(res_m_w_mat, full_matrices=False)
    _, _, vt_a = np.linalg.svd(res_a_mat, full_matrices=False)

    proj_m = (res_m_w_mat @ vt_m[:k_pca].T) @ vt_m[:k_pca]
    var_explained_pca = round(100.0 * (1.0 - float(np.sum((res_m_w_mat - proj_m) ** 2)) / max(tot_var_raw, 1e-12)), 2)

    basis_quant_div = 4096
    basis_m_q = np.round(vt_m[:k_pca] * basis_quant_div).astype(np.int32)
    basis_a_q = np.round(vt_a[:k_aper] * basis_quant_div).astype(np.int32)
    basis_m = basis_m_q.astype(np.float32) / basis_quant_div
    basis_a = basis_a_q.astype(np.float32) / basis_quant_div
    norm_sq_m = np.maximum(np.sum(basis_m ** 2, axis=1), 1e-9)
    norm_sq_a = np.maximum(np.sum(basis_a ** 2, axis=1), 1e-9)

    # Encode 286 events cleanly with unclipped integers!
    expr_step = int(prof_spec["exprKnotStepFrames"])
    f0_step = int(prof_spec["f0KnotStepFrames"])
    env_step = int(prof_spec["envKnotStepFrames"])

    gain_enc = tot_gain.copy()
    vdeg_enc = features["voicing_deg"].copy()
    f0_enc = features["f0"].copy()
    for i in range(1, len(raw_events)):
        prev_ev = raw_events[i - 1]
        curr_ev = raw_events[i]
        b = int(curr_ev["start_frame"])
        if prev_ev["end_frame"] == b and 1 <= b < len(tot_gain):
            if not curr_ev["is_plosive"]:
                gm = 0.5 * (float(tot_gain[b - 1]) + float(tot_gain[b]))
                vm = 0.5 * (float(vdeg_enc[b - 1]) + float(vdeg_enc[b]))
                gain_enc[b - 1] = gain_enc[b] = gm
                vdeg_enc[b - 1] = vdeg_enc[b] = vm
            if voiced[b - 1] and voiced[b] and f0_enc[b - 1] > 60.0 and f0_enc[b] > 60.0:
                fm = math.sqrt(float(f0_enc[b - 1]) * float(f0_enc[b]))
                f0_enc[b - 1] = f0_enc[b] = fm

    back_rms_l = stems["back_rms_l"]
    back_rms_r = stems["back_rms_r"]
    events_js: list[dict] = []

    for idx, ev in enumerate(raw_events):
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = ev["total_frames"]
        cf = ev["cons_frames"]

        v_rel = np.flatnonzero(voiced[s_f:e_f] & (f0_enc[s_f:e_f] > 0.0))
        if len(v_rel) > 0:
            voiced_onset = int(v_rel[0])
            midi_v = 69.0 + 12.0 * np.log2(f0_enc[s_f + v_rel] / 440.0)
            midi_full = np.interp(np.arange(L, dtype=np.float32), v_rel.astype(np.float32), midi_v).astype(np.float32)
            f0_pos = _build_uniform_knot_positions(L, f0_step)
            knot_targets = midi_full[f0_pos]
            base_midi_q = int(np.clip(round(float(knot_targets[0]) * 50.0), 1000, 6000))
            cur_midi = base_midi_q * 0.02
            steps_list = []
            for k in range(1, len(knot_targets)):
                sq = int(round((float(knot_targets[k]) - cur_midi) * 40.0))
                steps_list.append(sq)
                cur_midi += sq * 0.025
        else:
            voiced_onset = L
            base_midi_q = 3000
            steps_list = []

        env_pos = _build_envelope_knot_positions(L, cf, env_step)
        gain_h_q = np.maximum(_gain_to_uint8(gain_enc[s_f:e_f][env_pos], ref_peak), 1).astype(int).tolist()
        if voiced_onset >= L:
            gain_a_q = [0] * len(env_pos)
        else:
            gain_a_q = np.clip(np.round(vdeg_enc[s_f:e_f][env_pos] * 255.0), 0, 255).astype(int).tolist()

        res_m = mcep_frames[s_f:e_f] - base_cep_grid[s_f:e_f]
        res_a = aper_frames[s_f:e_f] - base_aper_grid[s_f:e_f]
        expr_pos = _build_adaptive_expr_knot_positions(L, cf, expr_step)
        coef_m = (res_m[expr_pos] @ basis_m.T) / norm_sq_m[None, :]
        coef_a = (res_a[expr_pos] @ basis_a.T) / norm_sq_a[None, :]
        expr_m_q_mat = np.round(coef_m / q_step).astype(int)
        expr_a_q_mat = np.round(coef_a / (q_step * 1.25)).astype(int)

        def _trim_rows(mat: np.ndarray) -> list[list[int]]:
            rows = []
            for row in mat.tolist():
                while row and row[-1] == 0:
                    row.pop()
                rows.append(row)
            return rows

        bl_val = float(np.percentile(back_rms_l[s_f:e_f], 70.0))
        br_val = float(np.percentile(back_rms_r[s_f:e_f], 70.0))
        if max(bl_val, br_val) < ref_peak * 0.022:
            back_l_q = back_r_q = 0
        else:
            back_l_q = int(_gain_to_uint8(bl_val, ref_peak)[0])
            back_r_q = int(_gain_to_uint8(br_val, ref_peak)[0])

        events_js.append(
            {
                "startFrame": int(s_f),
                "consFrames": int(cf),
                "totalFrames": int(L),
                "syllableId": int(ev["syllable_id"]),
                "text": ev["display_text"],
                "pinyin": ev["pinyin"],
                "isHook": bool(ev["is_hook"]),
                "legatoPrev": bool(ev["legato_prev"]),
                "legatoNext": bool(ev["legato_next"]),
                "isPlosive": bool(ev["is_plosive"]),
                "baseMidiQ": base_midi_q,
                "backGainLQ": back_l_q,
                "backGainRQ": back_r_q,
                "backPitchQ": 0,
                "voicedOnsetFrame": voiced_onset,
                "f0StepQ": steps_list,
                "gainHQ": gain_h_q,
                "gainAQ": gain_a_q,
                "exprMcepQ": _trim_rows(expr_m_q_mat),
                "exprAperQ": _trim_rows(expr_a_q_mat),
            }
        )

    # Fit 4-Tap ER + 48kHz 8-Line Prime Householder FDN (`models/space-fdn.js`)
    env_hop = 8192
    n_env = math.ceil(n_samples / env_hop)
    dry_pad = np.pad(stems["lead_mono"], (0, n_env * env_hop - n_samples))
    rev_side = 0.5 * (stems["space_stereo"][:, 0] - stems["space_stereo"][:, 1])
    rev_s_pad = np.pad(rev_side, (0, n_env * env_hop - n_samples))
    dry_e = np.sqrt(np.mean(dry_pad.reshape(n_env, env_hop) ** 2, axis=1) + 1e-12)
    rs_e = np.sqrt(np.mean(rev_s_pad.reshape(n_env, env_hop) ** 2, axis=1) + 1e-12)
    dry_macro = ndimage.gaussian_filter1d(dry_e, sigma=4.5)
    dry_ref = float(np.percentile(dry_macro[dry_macro > np.max(dry_macro) * 0.03], 85.0)) if np.max(dry_macro) > 1e-7 else 1.0
    norm_phrase = np.clip(dry_macro / max(dry_ref, 1e-6), 0.0, 1.0)
    side_ratio = ndimage.gaussian_filter1d(np.clip(rs_e / np.maximum(dry_e, dry_ref * 0.08), 0.0, 1.0), sigma=4.5)
    q_mid = np.clip(np.round((0.52 + 0.34 * norm_phrase) * 255.0), 0, 255).astype(int).tolist()
    q_side = np.clip(np.round((0.58 + 0.36 * np.clip(0.65 * norm_phrase + 0.35 * side_ratio, 0.0, 1.0)) * 255.0), 0, 255).astype(int).tolist()

    singer_core_js = {
        "sampleRate": sr,
        "numSamples": int(n_samples),
        "hop": hop,
        "refPeak": round(ref_peak, 6),
        "basisQuantDiv": basis_quant_div,
        "meanMcep": [round(float(x), 4) for x in mean_mcep],
        "aperMean": [round(float(x), 4) for x in aper_mean],
        "glottalPhase": [round(float(x), 5) for x in features["glottal_phase"]],
        "mcepPcaBasisQ": basis_m_q.tolist(),
        "aperPcaBasisQ": basis_a_q.tolist(),
    }
    syllable_atlas_js = {
        "atlasStepDb": atlas_step_db,
        "prototypes": prototypes,
    }
    space_fdn_js = {
        "sampleRate": sr,
        "numSamples": int(n_samples),
        "envHop": env_hop,
        "rt60Sec": 0.94,
        "dampLp": 0.48,
        "baseWetGain": 0.38,
        "preDelaySamples": int(round(0.022 * sr)),
        "qMid": q_mid,
        "qSide": q_side,
    }
    vocal_events_obj_js = {
        "pcaDims": k_pca,
        "aperExprBands": k_aper,
        "exprKnotStepFrames": expr_step,
        "f0KnotStepFrames": f0_step,
        "envKnotStepFrames": env_step,
        "quantStepDb": q_step,
        "events": events_js,
    }
    atlas_catalog = [
        {"id": p["id"], "pinyin": p["pinyin"], "initialClass": p["initialClass"], "count": p["count"]}
        for p in prototypes
    ]
    atlas_stats = {
        "totalEvents": len(raw_events),
        "lyricEvents": sum(1 for e in raw_events if not e["is_hook"]),
        "hookEvents": sum(1 for e in raw_events if e["is_hook"]),
        "uniqueSyllablePrototypes": len(prototypes),
        "averageEventReuse": round(len(raw_events) / max(1, len(prototypes)), 2),
        "varianceExplainedSpeakerPlusAtlasPct": round(100.0 * (1.0 - tot_var_after_atlas / max(tot_var_raw, 1e-12)), 2),
        "varianceExplainedFullPcaPct": var_explained_pca,
    }
    return {
        "singer_core": singer_core_js,
        "syllable_atlas": syllable_atlas_js,
        "space_fdn": space_fdn_js,
        "vocal_events": vocal_events_obj_js,
        "atlas_catalog": atlas_catalog,
        "atlas_stats": atlas_stats,
    }
