"""Native 48kHz Zero-Aliasing 8-Line Householder Studio Plate Reverb (`VGR9`).

Eliminates V8's two major reverb noise defects:
1. Runs natively at 48kHz (0% >12kHz sample-and-hold mirror aliasing!).
2. Replaces the 2.5KB accompaniment-bleed-contaminated `qMid/qSide` dynamic envelope with a crystal-pure
   40-byte LTI studio vocal plate model + sibilance damping + smooth vocal-clarity ducking!
"""
from __future__ import annotations

import math
import struct
import numpy as np
import scipy.ndimage as ndimage

FDN_DELAYS_48K = [1499, 1777, 2081, 2437, 2801, 3217, 3671, 4177]


def fit_and_pack_space_fdn(
    lead_direct: np.ndarray,
    reverb_residual: np.ndarray,
    sr: int = 48000,
) -> tuple[bytes, dict]:
    """Calibrates clean studio vocal plate ratios strictly on active singing windows and packs `VGR9` (40 B)."""
    n_samples = len(lead_direct)
    hop_blk = 4096
    n_blk = n_samples // hop_blk
    dry_b = lead_direct[: n_blk * hop_blk].reshape(n_blk, hop_blk)
    rev_m = 0.5 * (reverb_residual[: n_blk * hop_blk, 0] + reverb_residual[: n_blk * hop_blk, 1])
    rev_s = 0.5 * (reverb_residual[: n_blk * hop_blk, 0] - reverb_residual[: n_blk * hop_blk, 1])

    dry_rms = np.sqrt(np.mean(dry_b ** 2, axis=1) + 1e-16)
    rm_rms = np.sqrt(np.mean(rev_m.reshape(n_blk, hop_blk) ** 2, axis=1) + 1e-16)
    rs_rms = np.sqrt(np.mean(rev_s.reshape(n_blk, hop_blk) ** 2, axis=1) + 1e-16)

    # Measure natural wet-to-dry ratio strictly on strong vocal phrases (ignoring accompaniment bleed in silence!)
    strong = dry_rms > (np.max(dry_rms) * 0.18)
    if np.any(strong):
        ratio_m = float(np.clip(np.median(rm_rms[strong] / (dry_rms[strong] + 1e-9)), 0.12, 0.28))
        ratio_s = float(np.clip(np.median(rs_rms[strong] / (dry_rms[strong] + 1e-9)), 0.14, 0.32))
    else:
        ratio_m, ratio_s = 0.18, 0.22

    rt60_sec = 0.98
    damp_lp = 0.42          # Smooth HF absorption per reflection loop at 48kHz
    hp_hz = 195.0           # Clean sub/chest cut before plate
    lp_hz = 6800.0          # De-ess sibilance cut before plate so s/sh never splash metallically!
    pre_delay_samples = int(round(0.021 * sr))  # 21ms studio vocal pre-delay
    duck_amount = 0.28      # Gentle clarity ducking during dense syllables, blooming into phrase tails

    raw_pkt = struct.pack(
        "<4sIIffffffIf",
        b"VGR9",
        sr,
        n_samples,
        float(rt60_sec),
        float(damp_lp),
        float(hp_hz),
        float(lp_hz),
        float(ratio_m),
        float(ratio_s),
        pre_delay_samples,
        float(duck_amount),
    )
    meta = {
        "rt60Sec": rt60_sec,
        "damping": damp_lp,
        "preDelayMs": round(1000.0 * pre_delay_samples / sr, 1),
        "wetMidRatio": round(ratio_m, 3),
        "wetSideRatio": round(ratio_s, 3),
    }
    return raw_pkt, meta


def synthesize_space_fdn(raw_pkt: bytes, dry_mono: np.ndarray) -> np.ndarray:
    """Synthesizes native 48kHz 8-line Householder studio plate reverb with 0% imaging aliasing."""
    (
        magic,
        sr,
        n_samples,
        rt60_sec,
        damp_lp,
        hp_hz,
        lp_hz,
        ratio_m,
        ratio_s,
        pre_delay,
        duck_amount,
    ) = struct.unpack_from("<4sIIffffffIf", raw_pkt, 0)
    assert magic == b"VGR9"

    # 1. Pre-delay + 195Hz HPF + 6.8kHz Sibilance-Damping LPF on excitation
    hp_a = math.exp(-2.0 * math.pi * hp_hz / sr)
    lp_a = math.exp(-2.0 * math.pi * lp_hz / sr)
    one_m_lp = 1.0 - lp_a

    exc = np.zeros(n_samples, dtype=np.float64)
    if pre_delay < n_samples:
        exc[pre_delay:] = dry_mono[: n_samples - pre_delay].astype(np.float64)

    # Vectorized or fast IIR pre-filtering via scipy.signal.lfilter
    import scipy.signal as signal

    # HPF: y[n] = hp_a * (y[n-1] + x[n] - x[n-1])
    exc = signal.lfilter([hp_a, -hp_a], [1.0, -hp_a], exc)
    # LPF: y[n] = one_m_lp * x[n] + lp_a * y[n-1]
    exc = signal.lfilter([one_m_lp], [1.0, -lp_a], exc)

    # 2. Native 48kHz 8-line Householder FDN (block-vectorized over min_delay = 1499 samples!)
    delays = FDN_DELAYS_48K
    min_d = min(delays)
    gains = np.array([10.0 ** (-3.0 * d / (rt60_sec * sr)) for d in delays], dtype=np.float64)
    buf_mat = [np.zeros(d, dtype=np.float64) for d in delays]
    ptrs = [0] * 8
    lp_v = np.zeros(8, dtype=np.float64)
    in_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float64) * 0.25
    b_damp = [1.0 - damp_lp]
    a_damp = [1.0, -damp_lp]

    out_m = np.zeros(n_samples, dtype=np.float32)
    out_s = np.zeros(n_samples, dtype=np.float32)

    cursor = 0
    while cursor < n_samples:
        blk = min(min_d, n_samples - cursor)
        reads = np.empty((8, blk), dtype=np.float64)
        for i in range(8):
            d = delays[i]
            p = ptrs[i]
            if p + blk <= d:
                reads[i] = buf_mat[i][p : p + blk]
            else:
                first = d - p
                reads[i, :first] = buf_mat[i][p:]
                reads[i, first:] = buf_mat[i][: blk - first]

        damped = np.empty_like(reads)
        for i in range(8):
            row_scaled = reads[i] * gains[i]
            y_row, zf = signal.lfilter(b_damp, a_damp, row_scaled, zi=[lp_v[i] * damp_lp])
            lp_v[i] = float(y_row[-1])
            damped[i] = y_row

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

        out_m[cursor : cursor + blk] = 0.35 * (reads[0] + reads[2] + reads[4] + reads[6])
        out_s[cursor : cursor + blk] = 0.35 * (reads[1] - reads[3] + reads[5] - reads[7])
        cursor += blk

    # Calibrate global LTI wet gain against dry RMS + gentle clarity ducking
    dry_act = np.abs(dry_mono) > 1e-4
    if np.any(dry_act):
        dry_rms_act = float(np.sqrt(np.mean(dry_mono[dry_act] ** 2) + 1e-16))
        m_rms_act = float(np.sqrt(np.mean(out_m[dry_act] ** 2) + 1e-16))
        s_rms_act = float(np.sqrt(np.mean(out_s[dry_act] ** 2) + 1e-16))
        gain_m = (dry_rms_act * ratio_m) / max(m_rms_act, 1e-6)
        gain_s = (dry_rms_act * ratio_s) / max(s_rms_act, 1e-6)
    else:
        gain_m, gain_s = ratio_m, ratio_s

    # Smooth vocal envelope ducking (256-sample decimated envelope -> identical to JS!)
    hop_d = 256
    n_d = n_samples // hop_d + 1
    dry_pad = np.pad(dry_mono, (0, n_d * hop_d - n_samples)).reshape(n_d, hop_d)
    env_d = np.sqrt(np.mean(dry_pad ** 2, axis=1) + 1e-16)
    pk_d = float(np.max(env_d) + 1e-9)
    # 1-pole attack/release follower on 256-sample blocks
    sm_d = np.zeros(n_d, dtype=np.float32)
    st = 0.0
    for k in range(n_d):
        target = min(1.0, float(env_d[k]) / (pk_d * 0.45))
        coef = 0.35 if target > st else 0.06
        st = st + coef * (target - st)
        sm_d[k] = 1.0 - duck_amount * st

    idx_s = np.arange(n_samples, dtype=np.float32) / float(hop_d)
    i0 = np.clip(np.floor(idx_s).astype(np.int32), 0, n_d - 2)
    fr = idx_s - i0
    duck_s = (1.0 - fr) * sm_d[i0] + fr * sm_d[i0 + 1]

    m_final = out_m * (gain_m * duck_s)
    s_final = out_s * (gain_s * duck_s)
    return np.column_stack([m_final + s_final, m_final - s_final]).astype(np.float32)
