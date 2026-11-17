"""Native 48kHz Studio Vocal Booth Early Reflections + De-Essed 8-Line Householder FDN (`VGR8`, V8.4)."""
from __future__ import annotations

import math
import struct
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

FDN_DELAYS_48K = [1499, 1777, 2081, 2437, 2801, 3217, 3671, 4177]
ER_TAPS_L = ((542, 0.110), (1186, -0.075))
ER_TAPS_R = ((859, 0.105), (1589, -0.070))


def fit_and_pack_space_fdn(
    lead_direct: np.ndarray,
    reverb_residual: np.ndarray,
    sr: int = 48000,
    env_hop: int = 8192,
) -> tuple[bytes, dict]:
    """Fits smooth section-level Mid/Side studio vocal space send curves (`VGR8`) with zero bleed pumping."""
    n_samples = len(lead_direct)
    n_env = math.ceil(n_samples / env_hop)

    dry_pad = np.pad(lead_direct, (0, n_env * env_hop - n_samples))
    rev_side = 0.5 * (reverb_residual[:, 0] - reverb_residual[:, 1])
    rev_s_pad = np.pad(rev_side, (0, n_env * env_hop - n_samples))

    dry_e = np.sqrt(np.mean(dry_pad.reshape(n_env, env_hop) ** 2, axis=1) + 1e-12)
    rs_e = np.sqrt(np.mean(rev_s_pad.reshape(n_env, env_hop) ** 2, axis=1) + 1e-12)

    dry_macro = ndimage.gaussian_filter1d(dry_e, sigma=4.5)
    dry_ref = float(np.percentile(dry_macro[dry_macro > np.max(dry_macro) * 0.03], 85.0)) if np.max(dry_macro) > 1e-7 else 1.0
    norm_phrase = np.clip(dry_macro / max(dry_ref, 1e-6), 0.0, 1.0)

    side_ratio = ndimage.gaussian_filter1d(np.clip(rs_e / np.maximum(dry_e, dry_ref * 0.08), 0.0, 1.0), sigma=4.5)

    rt60_sec = 0.94
    damp_lp = 0.48
    base_wet_gain = 0.38
    pre_delay_samples = int(round(0.022 * sr))  # 22.0ms studio vocal pre-delay

    send_mid = 0.52 + 0.34 * norm_phrase
    send_side = 0.58 + 0.36 * np.clip(0.65 * norm_phrase + 0.35 * side_ratio, 0.0, 1.0)

    q_mid = np.clip(np.round(send_mid * 255.0), 0, 255).astype(np.uint8)
    q_side = np.clip(np.round(send_side * 255.0), 0, 255).astype(np.uint8)

    header = struct.pack(
        "<4sIIIfff",
        b"VGR8",
        sr,
        n_samples,
        env_hop,
        float(rt60_sec),
        float(damp_lp),
        float(base_wet_gain),
    )
    raw_pkt = header + struct.pack("<I", pre_delay_samples) + q_mid.tobytes() + q_side.tobytes()
    meta = {
        "rt60Sec": rt60_sec,
        "damping": damp_lp,
        "preDelayMs": round(1000.0 * pre_delay_samples / sr, 1),
        "envFrames": n_env,
    }
    return raw_pkt, meta


def synthesize_space_fdn(raw_pkt: bytes, dry_mono: np.ndarray) -> np.ndarray:
    """Synthesizes 4-tap Studio Vocal Booth Early Reflections + 48kHz 8-Line Householder FDN from `VGR8`."""
    magic, sr, n_samples, env_hop, rt60_sec, damp_lp, base_wet_gain = struct.unpack_from("<4sIIIfff", raw_pkt, 0)
    assert magic == b"VGR8"
    (pre_delay,) = struct.unpack_from("<I", raw_pkt, 28)
    n_env = math.ceil(n_samples / env_hop)
    off = 32
    q_mid = np.frombuffer(raw_pkt, dtype=np.uint8, count=n_env, offset=off).astype(np.float32) / 255.0
    q_side = np.frombuffer(raw_pkt, dtype=np.uint8, count=n_env, offset=off + n_env).astype(np.float32) / 255.0

    # Filter input first with 210Hz HPF (mud cut) + 4.8kHz LPF (de-esser) for both ER taps & FDN tail
    x_in = np.zeros(n_samples, dtype=np.float64)
    copy_len = min(len(dry_mono), n_samples)
    x_in[:copy_len] = dry_mono[:copy_len]

    hp_alpha = 0.973
    x_hp = signal.lfilter([hp_alpha, -hp_alpha], [1.0, -hp_alpha], x_in)
    in_lp = 0.54
    y_warm = signal.lfilter([1.0 - in_lp], [1.0, -in_lp], x_hp)

    exc = np.zeros(n_samples, dtype=np.float64)
    if pre_delay < n_samples:
        exc[pre_delay:] = y_warm[: n_samples - pre_delay]

    er_l = np.zeros(n_samples, dtype=np.float32)
    er_r = np.zeros(n_samples, dtype=np.float32)
    for d_tap, g_tap in ER_TAPS_L:
        if d_tap < n_samples:
            er_l[d_tap:] += (g_tap * y_warm[: n_samples - d_tap]).astype(np.float32)
    for d_tap, g_tap in ER_TAPS_R:
        if d_tap < n_samples:
            er_r[d_tap:] += (g_tap * y_warm[: n_samples - d_tap]).astype(np.float32)

    delays = list(FDN_DELAYS_48K)
    gains = [float(10.0 ** (-3.0 * d / (rt60_sec * sr))) for d in delays]
    min_d = min(delays)

    buf_mat = [np.zeros(d, dtype=np.float64) for d in delays]
    ptrs = [0] * 8
    zi_lp = [np.zeros(1, dtype=np.float64) for _ in range(8)]
    one_m = 1.0 - damp_lp
    a_lp = np.array([1.0, -damp_lp], dtype=np.float64)
    b_lps = [np.array([one_m * g], dtype=np.float64) for g in gains]
    in_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64) * 0.25

    out_m = np.zeros(n_samples, dtype=np.float32)
    out_s = np.zeros(n_samples, dtype=np.float32)

    cursor = 0
    while cursor < n_samples:
        blk = min(min_d, n_samples - cursor)
        reads = np.empty((8, blk), dtype=np.float64)
        damped = np.empty((8, blk), dtype=np.float64)

        for i in range(8):
            d = delays[i]
            p = ptrs[i]
            if p + blk <= d:
                reads[i] = buf_mat[i][p : p + blk]
            else:
                first = d - p
                reads[i, :first] = buf_mat[i][p:]
                reads[i, first:] = buf_mat[i][: blk - first]
            damped[i], zi_lp[i] = signal.lfilter(b_lps[i], a_lp, reads[i], zi=zi_lp[i])

        s_sum = 0.25 * np.sum(damped, axis=0)
        fb = damped - s_sum[None, :] + in_signs[:, None] * exc[cursor : cursor + blk][None, :]

        for i in range(8):
            d = delays[i]
            p = ptrs[i]
            if p + blk <= d:
                buf_mat[i][p : p + blk] = fb[i]
            else:
                first = d - p
                buf_mat[i][p:] = fb[i, :first]
                buf_mat[i][: blk - first] = fb[i, first:]
            ptrs[i] = (p + blk) % d

        out_m[cursor : cursor + blk] = 0.5 * (reads[0] + reads[2] + reads[4] + reads[6])
        out_s[cursor : cursor + blk] = 0.5 * (reads[1] - reads[3] + reads[5] - reads[7])
        cursor += blk

    t_centers = (np.arange(n_env, dtype=np.float32) + 0.5) * env_hop
    idx_all = np.arange(n_samples, dtype=np.float32)
    gm_full = np.interp(idx_all, t_centers, q_mid * base_wet_gain).astype(np.float32)
    gs_full = np.interp(idx_all, t_centers, q_side * base_wet_gain).astype(np.float32)

    m_final = out_m * gm_full
    s_final = out_s * gs_full
    left = m_final + s_final + er_l * gm_full
    right = m_final - s_final + er_r * gm_full
    return np.column_stack([left, right]).astype(np.float32)
