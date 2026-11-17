"""Purified Blackman-Harris (-92dB) Harmonic + Liftered Formant + Zero-Hiss Consonant/Sheen Analyzer (V9)."""
from __future__ import annotations

import math
import librosa
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

from .audio_io import resample_audio


def hz_to_erb(f: np.ndarray | float) -> np.ndarray | float:
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, dtype=np.float64))


def erb_to_hz(erb: np.ndarray | float) -> np.ndarray | float:
    return (np.power(10.0, np.asarray(erb, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def build_erb_grid(n_bands: int = 40, fmin: float = 75.0, fmax: float = 9600.0) -> np.ndarray:
    e0 = float(hz_to_erb(fmin))
    e1 = float(hz_to_erb(fmax))
    return erb_to_hz(np.linspace(e0, e1, n_bands)).astype(np.float32)


def build_aper_edges(n_bands: int = 16, fmin: float = 550.0, fmax: float = 16000.0) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bands + 1).astype(np.float32)


def build_mcep_basis(n_cep: int = 18, n_bands: int = 40) -> np.ndarray:
    """Orthonormal zero-mean DCT-II basis (m = 1..n_cep) of shape (n_cep, n_bands)."""
    b = np.arange(n_bands, dtype=np.float64)[None, :]
    m = np.arange(1, n_cep + 1, dtype=np.float64)[:, None]
    return (math.sqrt(2.0 / n_bands) * np.cos(math.pi * m * (b + 0.5) / n_bands)).astype(np.float32)


def zero_phase_bh_stft(
    y: np.ndarray,
    n_fft: int = 2048,
    win_len: int = 1536,
    hop: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase 4-term Blackman-Harris (-92dB sidelobe) STFT eliminating harmonic skirt leakage."""
    n_samples = len(y)
    n_frames = n_samples // hop + 1
    pad = win_len // 2
    y_pad = np.pad(y.astype(np.float32), (pad, pad + hop), mode="constant")
    w = signal.windows.blackmanharris(win_len, sym=False).astype(np.float32)

    shape = (n_frames, win_len)
    strides = (y_pad.strides[0] * hop, y_pad.strides[0])
    frames = np.lib.stride_tricks.as_strided(y_pad, shape=shape, strides=strides) * w[None, :]

    zp = np.zeros((n_frames, n_fft), dtype=np.float32)
    half = win_len // 2
    zp[:, :half] = frames[:, half:]
    zp[:, n_fft - half :] = frames[:, :half]
    return np.fft.rfft(zp, axis=1), w


def analyze_vocal_signal(lead_direct: np.ndarray, config: dict) -> dict:
    """Extracts pure F0, GCI glottal phase, liftered 18-MCEP formants, and zero-hiss consonant/sheen bands."""
    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])
    n_fft = int(config["fftSize"])
    n_erb = int(config["harmonicBands"])
    n_cep = int(config["cepstralCoeffs"])
    n_aper = int(config["aperiodicBands"])
    max_harm = int(config["maxHarmonics"])
    f_h_min = float(config["harmFreqMin"])
    f_h_max = float(config["harmFreqMax"])

    erb_freqs = build_erb_grid(n_erb, f_h_min, f_h_max)
    aper_edges = build_aper_edges(n_aper, float(config["aperFreqMin"]), float(config["aperFreqMax"]))
    mcep_basis = build_mcep_basis(n_cep, n_erb)

    # Cepstral liftering weights: preserves F1-F5 formants (m=1..10) 100%, gently damps Demucs comb ripples (m=11..18)
    m_orders = np.arange(1, n_cep + 1, dtype=np.float32)
    cep_lifter = np.exp(-0.5 * (np.maximum(0.0, m_orders - 10.0) / 5.5) ** 2).astype(np.float32)

    # 1. -92dB Blackman-Harris long STFT (32ms) + 8ms short Blackman-Harris STFT for crisp consonants
    spec_bh, w_bh = zero_phase_bh_stft(lead_direct, n_fft=n_fft, win_len=1536, hop=hop)
    spec_short, w_short = zero_phase_bh_stft(lead_direct, n_fft=512, win_len=384, hop=hop)
    n_frames = len(spec_bh)

    w_bh_half_sum = 0.5 * float(np.sum(w_bh))
    w_bh_pow_norm = float(np.sum(w_bh.astype(np.float64) ** 2)) * n_fft
    w_short_pow_norm = float(np.sum(w_short.astype(np.float64) ** 2)) * 512

    mag_amp = (np.abs(spec_bh) / w_bh_half_sum).astype(np.float32)
    pow_long = (np.abs(spec_bh) ** 2) * (2.0 / w_bh_pow_norm)
    pow_short = (np.abs(spec_short) ** 2) * (2.0 / w_short_pow_norm)

    # 2. High-Purity F0 Tracking with Sibilant-Dominance Rejection (prevents consonant pitch chirps!)
    lead_24k = resample_audio(lead_direct, sr, 24000)
    f0_raw, _, voiced_prob = librosa.pyin(
        lead_24k, fmin=142.0, fmax=650.0, sr=24000, frame_length=2048, hop_length=hop // 2, fill_na=0.0
    )
    if len(f0_raw) < n_frames:
        f0_raw = np.pad(f0_raw, (0, n_frames - len(f0_raw)))
        voiced_prob = np.pad(voiced_prob, (0, n_frames - len(voiced_prob)))
    else:
        f0_raw = f0_raw[:n_frames]
        voiced_prob = voiced_prob[:n_frames]

    bin_hz = sr / float(n_fft)
    low_harm_pow = np.sum(pow_long[:, 3:180], axis=1)   # 70Hz - 4.2kHz voiced harmonic zone
    hi_cons_pow = np.sum(pow_short[:, 48:205], axis=1)  # 4.5kHz - 19.2kHz sibilant burst zone
    total_rms = np.sqrt(np.sum(pow_long[:, 3:740], axis=1) + 1e-16).astype(np.float32)
    rms_peak = float(np.max(total_rms) + 1e-9)

    f0 = f0_raw.astype(np.float32).copy()
    voiced = (
        (f0 > 140.0)
        & (voiced_prob >= 0.25)
        & (total_rms > rms_peak * 0.0085)
        & (low_harm_pow > 0.40 * hi_cons_pow)
    )
    f0[~voiced] = 0.0

    voiced_idx = np.flatnonzero(voiced)
    if len(voiced_idx) > 0:
        cand_ratios = np.linspace(0.975, 1.025, 31, dtype=np.float32)
        for idx in voiced_idx:
            f_est = f0[idx]
            cands = f_est * cand_ratios
            scores = np.zeros_like(cands)
            row = mag_amp[idx]
            for h in range(1, 6):
                bf = np.clip(cands * h / bin_hz, 1.0, (n_fft // 2) - 2.0)
                b0 = bf.astype(np.int32)
                frac = bf - b0
                scores += ((1.0 - frac) * row[b0] + frac * row[b0 + 1]) / (h ** 0.42)
            f0[idx] = float(cands[int(np.argmax(scores))])

    # Despike boundary pitch swoops & smooth micro-jitter in log-Hz inside contiguous voiced runs
    labeled_v, n_v = ndimage.label(voiced)
    for r in range(1, n_v + 1):
        sl = np.flatnonzero(labeled_v == r)
        if len(sl) < 3:
            voiced[sl] = False
            f0[sl] = 0.0
            continue
        med = ndimage.median_filter(f0[sl], size=min(7, len(sl)))
        core_med = float(np.median(f0[sl[1:-1]])) if len(sl) >= 4 else float(np.median(f0[sl]))
        ratio = f0[sl] / np.maximum(med, 1.0)
        bad = (ratio > 1.18) | (ratio < 0.84)
        f0[sl[bad]] = med[bad]
        for edge_k in (0, 1, -1, -2):
            if abs(edge_k) <= len(sl):
                if abs(12.0 * math.log2(max(f0[sl[edge_k]], 1.0) / max(core_med, 1.0))) > 2.0:
                    f0[sl[edge_k]] = core_med
        if len(sl) >= 5:
            lf = np.log2(f0[sl])
            f0[sl] = np.power(2.0, ndimage.gaussian_filter1d(lf, sigma=0.88)).astype(np.float32)

    # 3. Clean Active Vocal Mask
    active_raw = total_rms > (rms_peak * 0.009)
    active_closed = ndimage.binary_closing(active_raw, structure=np.ones(7, dtype=bool))
    active_open = ndimage.binary_opening(active_closed, structure=np.ones(4, dtype=bool))
    active_mask = ndimage.binary_dilation(active_open, structure=np.ones(4, dtype=bool))

    # 4. Pure Harmonic (<9.6kHz) vs Zero-Hiss Consonant/Sheen Decomposition
    gain_harm = np.zeros(n_frames, dtype=np.float32)
    mcep_frames = np.zeros((n_frames, n_cep), dtype=np.float32)
    gain_aper = np.zeros(n_frames, dtype=np.float32)
    aper_log_shape = np.zeros((n_frames, n_aper), dtype=np.float32)

    glottal_cos = np.zeros(max_harm, dtype=np.float64)
    glottal_sin = np.zeros(max_harm, dtype=np.float64)
    glottal_w = np.zeros(max_harm, dtype=np.float64)

    freqs_short = np.linspace(0.0, sr * 0.5, 257)
    aper_short_bins = []
    aper_fc = np.zeros(n_aper, dtype=np.float64)
    for b in range(n_aper):
        lo, hi = float(aper_edges[b]), float(aper_edges[b + 1])
        aper_fc[b] = math.sqrt(lo * hi)
        bs = np.flatnonzero((freqs_short >= lo) & (freqs_short < hi))
        if len(bs) == 0:
            bs = np.array([min(256, int(round(0.5 * (lo + hi) / (sr / 512.0))))], dtype=np.int32)
        aper_short_bins.append(bs)

    log_erb = np.log(erb_freqs.astype(np.float64))
    erb_min_v = float(hz_to_erb(f_h_min))
    erb_span_v = float(hz_to_erb(f_h_max)) - erb_min_v
    m_idx = np.arange(1, n_cep + 1, dtype=np.float64)[:, None]

    for t in np.flatnonzero(active_mask):
        if voiced[t] and f0[t] > 0.0:
            f_fund = float(f0[t])
            n_h = min(max_harm, int(f_h_max // f_fund))
            ks = np.arange(1, n_h + 1, dtype=np.float64)
            f_ks = ks * f_fund
            b_ks = f_ks / bin_hz
            b0 = np.clip(b_ks.astype(np.int32), 1, n_fft // 2 - 2)
            frac = b_ks - b0

            row_amp = mag_amp[t]
            peak_amp = np.maximum(
                (1.0 - frac) * row_amp[b0] + frac * row_amp[b0 + 1],
                np.maximum(row_amp[b0], row_amp[b0 + 1]),
            )
            v_mid = np.clip(((ks + 0.5) * f_fund / bin_hz).astype(np.int32), 1, n_fft // 2 - 1)
            valley_amp = row_amp[v_mid]

            pure_h_amp = np.sqrt(np.maximum(0.0, peak_amp ** 2 - 1.25 * (valley_amp ** 2)))
            max_partial = float(np.max(pure_h_amp) + 1e-9)
            # Purge sub-threshold upper harmonic dirt (< -52dB below dominant harmonic)
            pure_h_amp[pure_h_amp < max_partial * 0.0025] *= 0.05

            h_rms = float(np.sqrt(0.5 * np.sum(pure_h_amp ** 2) + 1e-16))
            gain_harm[t] = h_rms

            h_db = 20.0 * np.log10(np.maximum(pure_h_amp, h_rms * 1.6e-3 + 1e-8))
            erb_db = np.interp(log_erb, np.log(f_ks), h_db, left=h_db[0] - 8.0, right=h_db[-1] - 18.0)
            erb_db_zm = (erb_db - np.mean(erb_db)).astype(np.float32)
            # Apply cepstral liftering right at extraction so Demucs fine-comb notches never enter MCEP!
            c_vec = (mcep_basis @ erb_db_zm) * cep_lifter
            mcep_frames[t] = c_vec

            if h_rms > rms_peak * 0.10 and voiced_prob[t] > 0.78:
                z_row = spec_bh[t]
                z_k = (1.0 - frac) * z_row[b0] + frac * z_row[b0 + 1]
                raw_ph = np.angle(z_k)
                theta_erb = np.clip((hz_to_erb(f_ks) - erb_min_v) / erb_span_v * math.pi, 0.0, math.pi)
                tract_ph = -(math.log(10.0) / 20.0) * math.sqrt(2.0 / n_erb) * np.sum(
                    c_vec[:, None].astype(np.float64) * np.sin(m_idx * theta_erb[None, :]), axis=0
                )
                glott_ph = raw_ph - tract_ph
                rel_ph = glott_ph - ks * glott_ph[0]
                w_h = (pure_h_amp[0] ** 2) * np.clip(pure_h_amp / max_partial, 0.0, 1.0)
                glottal_cos[:n_h] += w_h * np.cos(rel_ph)
                glottal_sin[:n_h] += w_h * np.sin(rel_ph)
                glottal_w[:n_h] += w_h

            # VOICED VOWEL ZERO-HISS LAW:
            # Strictly ZERO noise below 5.0 kHz during voiced singing! Only gentle velvet sheen >5.0 kHz (<= -27.5 dB)!
            band_pows = np.full(n_aper, 1e-18, dtype=np.float64)
            for b in range(n_aper):
                fc = aper_fc[b]
                if fc >= 5000.0:
                    w_hi = min(1.0, (fc - 5000.0) / 3000.0)
                    if fc > 12500.0:
                        w_hi *= max(0.35, 1.0 - (fc - 12500.0) / 6000.0)
                    bs = aper_short_bins[b]
                    h_in_band = pure_h_amp[(f_ks >= aper_edges[b]) & (f_ks < aper_edges[b + 1])]
                    h_pow_b = 0.5 * float(np.sum(h_in_band ** 2))
                    raw_p = max(0.0, float(np.sum(pow_short[t, bs])) - 1.25 * h_pow_b)
                    band_pows[b] = max(1e-18, w_hi * raw_p * 0.36)

            a_rms = min(float(np.sqrt(np.sum(band_pows))), h_rms * 0.042)  # <= -27.5 dB ceiling (>31 dB Voiced HNR!)
            if a_rms < rms_peak * 0.0040:
                a_rms = 0.0
            gain_aper[t] = a_rms
            max_bp = max(float(np.max(band_pows)), 1e-16)
            log_bp = 10.0 * np.log10(np.maximum(band_pows, max_bp * 1e-3))
            aper_log_shape[t] = (log_bp - np.mean(log_bp)).astype(np.float32)

        else:
            # Unvoiced Consonant Frame (s, sh, x, q, ch, t, d, p, k, f, h):
            # Clean de-essed transient band [550Hz .. 16kHz], gated so dead gaps stay 100% black!
            band_pows = np.zeros(n_aper, dtype=np.float64)
            for b in range(n_aper):
                bs = aper_short_bins[b]
                fc = aper_fc[b]
                if fc < 950.0:
                    w_cons = 0.18
                elif fc < 1650.0:
                    w_cons = 0.58
                elif fc > 10500.0:
                    w_cons = max(0.35, 1.0 - (fc - 10500.0) / 8500.0)
                else:
                    w_cons = 1.0
                band_pows[b] = max(1e-18, w_cons * float(np.sum(pow_short[t, bs])) * 0.34)

            a_rms = float(np.sqrt(np.sum(band_pows)))
            # Hard-black noise gate on weak background floor + smooth Pro-DS sibilant ceiling
            if a_rms < rms_peak * 0.013:
                a_rms *= max(0.0, (a_rms - rms_peak * 0.007) / (rms_peak * 0.006))
            knee = rms_peak * 0.135
            if a_rms > knee:
                a_rms = knee + (a_rms - knee) / (1.0 + (a_rms - knee) / (rms_peak * 0.075))
            gain_aper[t] = a_rms
            max_bp = max(float(np.max(band_pows)), 1e-16)
            log_bp = 10.0 * np.log10(np.maximum(band_pows, max_bp * 1e-3))
            aper_log_shape[t] = (log_bp - np.mean(log_bp)).astype(np.float32)

    # Temporal formant de-gargling inside each contiguous voiced island:
    # 3-frame median despike + sigma=0.90 Gaussian wipes 2-frame Demucs musical-noise notches ("咕噜水声残留")!
    for r in range(1, n_v + 1):
        sl = np.flatnonzero(labeled_v == r)
        if len(sl) >= 4:
            sub_m = mcep_frames[sl]
            if len(sl) >= 5:
                sub_m = ndimage.median_filter(sub_m, size=(3, 1))
            mcep_frames[sl] = ndimage.gaussian_filter1d(sub_m, sigma=0.90, axis=0)

    glottal_phase = np.zeros(max_harm, dtype=np.float32)
    valid_g = glottal_w > 1e-9
    glottal_phase[valid_g] = np.arctan2(glottal_sin[valid_g], glottal_cos[valid_g]).astype(np.float32)
    glottal_phase[0] = 0.0

    # 5. Clean Syllable Onset Novelty Curve
    log_h = np.log10(gain_harm + 1e-5)
    log_a = np.log10(gain_aper + 1e-5)
    dh = np.maximum(0.0, np.diff(log_h, prepend=log_h[0]))
    da = np.maximum(0.0, np.diff(log_a, prepend=log_a[0]))
    dmcep = np.sqrt(np.mean(np.diff(mcep_frames, axis=0, prepend=mcep_frames[:1]) ** 2, axis=1))
    novelty = ndimage.gaussian_filter1d(2.1 * dh + 2.2 * da + 0.32 * dmcep, sigma=1.4).astype(np.float32)

    return {
        "n_samples": len(lead_direct),
        "n_frames": n_frames,
        "rms_peak": rms_peak,
        "f0": f0,
        "voiced": voiced,
        "voiced_prob": voiced_prob,
        "active_mask": active_mask,
        "gain_harm": gain_harm,
        "mcep_frames": mcep_frames,
        "gain_aper": gain_aper,
        "aper_log_shape": aper_log_shape,
        "glottal_phase": glottal_phase,
        "novelty": novelty,
        "erb_freqs": erb_freqs,
        "aper_edges": aper_edges,
        "mcep_basis": mcep_basis,
    }
