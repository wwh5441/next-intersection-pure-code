"""Fitted 8-Line Householder Stereo FDN Studio Reverb Model (`VWR7`, < 1.2 KB for full 3m30s song)."""
from __future__ import annotations

import lzma
import math
import struct
import numpy as np
from scipy.signal import lfilter

FDN_DELAYS = (1423, 1657, 1901, 2179, 2477, 2803, 3163, 3559)
ENV_HOP = 8192  # 170.67 ms per dynamic reverb automation frame at 48kHz
GAIN_MIN_DB = -44.0
GAIN_STEP_DB = 0.25  # codes 1..255 => -44.0 dB .. +19.5 dB


def _one_pole_lp(x: np.ndarray, alpha: float) -> np.ndarray:
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x.astype(np.float64)).astype(np.float32)


def run_raw_fdn_stereo(
    dry_mid: np.ndarray,
    dry_side: np.ndarray,
    sr: int = 48000,
    rt60_low: float = 1.48,
    rt60_high: float = 0.58,
    pre_delay_samples: int = 960,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample-exact vectorized 8-line Householder FDN matching `src/dsp/reverb-synth.js`."""
    n = len(dry_mid)
    a_hp = 1.0 - math.exp(-2.0 * math.pi * 220.0 / sr)
    a_lp = 1.0 - math.exp(-2.0 * math.pi * 6400.0 / sr)

    m_hp = dry_mid - _one_pole_lp(dry_mid, a_hp)
    m_in = _one_pole_lp(m_hp, a_lp)
    s_hp = dry_side - _one_pole_lp(dry_side, a_hp)
    s_in = _one_pole_lp(s_hp, a_lp)

    # Pre-delay & early stereo reflections
    m_pad = np.pad(m_in, (pre_delay_samples, 0))
    er_mid = (0.22 * np.pad(m_in, (576, 0))[:n] + 0.14 * np.pad(m_in, (888, 0))[:n]).astype(np.float32)
    er_side = (0.20 * np.pad(m_in, (672, 0))[:n] - 0.15 * np.pad(m_in, (936, 0))[:n]).astype(np.float32)
    x_late = m_pad[:n]

    g_dc = np.array([10.0 ** (-3.0 * d / (sr * rt60_low)) for d in FDN_DELAYS], dtype=np.float64)
    g_hf = np.array([10.0 ** (-3.0 * d / (sr * rt60_high)) for d in FDN_DELAYS], dtype=np.float64)
    damp = np.clip(1.0 - g_hf / g_dc, 0.08, 0.72)

    bufs = [np.zeros(d, dtype=np.float64) for d in FDN_DELAYS]
    ptrs = [0] * 8
    d_state = np.zeros(8, dtype=np.float64)

    out_mid = np.zeros(n, dtype=np.float32)
    out_side = np.zeros(n, dtype=np.float32)

    # Block size <= 1024 < min(FDN_DELAYS)=1423 => zero intra-block feedback dependency!
    BLK = 1024
    z_blk = np.zeros((8, BLK), dtype=np.float64)
    side_signs = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)[:, None]
    mid_weights = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64) * 0.35
    side_weights = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64) * 0.35

    for pos in range(0, n, BLK):
        L = min(BLK, n - pos)
        xm = x_late[pos : pos + L].astype(np.float64)[None, :]
        xs = s_in[pos : pos + L].astype(np.float64)[None, :]

        for i in range(8):
            d = FDN_DELAYS[i]
            p = ptrs[i]
            if p + L <= d:
                y_seg = bufs[i][p : p + L].copy()
            else:
                first = d - p
                y_seg = np.concatenate([bufs[i][p:], bufs[i][: L - first]])
            # 1-pole damping filter across segment
            dm = float(damp[i])
            st = float(d_state[i])
            filt, zf = lfilter([1.0 - dm], [1.0, -dm], y_seg, zi=[st * dm])
            d_state[i] = filt[-1]
            z_blk[i, :L] = filt * g_dc[i]

        z_cur = z_blk[:, :L]
        fb_sum = 0.25 * np.sum(z_cur, axis=0, keepdims=True)
        write_blk = xm * 0.35 + xs * (0.18 * side_signs) + (z_cur - fb_sum)

        for i in range(8):
            d = FDN_DELAYS[i]
            p = ptrs[i]
            if p + L <= d:
                bufs[i][p : p + L] = write_blk[i]
                ptrs[i] = 0 if (p + L == d) else (p + L)
            else:
                first = d - p
                bufs[i][p:] = write_blk[i, :first]
                bufs[i][: L - first] = write_blk[i, first:]
                ptrs[i] = L - first

        out_mid[pos : pos + L] = (mid_weights @ z_cur).astype(np.float32) + er_mid[pos : pos + L]
        out_side[pos : pos + L] = (side_weights @ z_cur).astype(np.float32) + er_side[pos : pos + L]

    return out_mid, out_side


def _quantize_gain_codes(gains: np.ndarray) -> np.ndarray:
    codes = np.zeros(len(gains), dtype=np.uint8)
    act = gains > 10.0 ** (GAIN_MIN_DB / 20.0)
    if np.any(act):
        db = 20.0 * np.log10(np.maximum(gains[act], 1e-6))
        c = np.rint((db - GAIN_MIN_DB) / GAIN_STEP_DB).astype(np.int32) + 1
        codes[act] = np.clip(c, 1, 255).astype(np.uint8)
    return codes


def _dequantize_gain_codes(codes: np.ndarray) -> np.ndarray:
    out = np.zeros(len(codes), dtype=np.float32)
    act = codes > 0
    if np.any(act):
        out[act] = (10.0 ** ((GAIN_MIN_DB + (codes[act].astype(np.float32) - 1.0) * GAIN_STEP_DB) / 20.0)).astype(
            np.float32
        )
    return out


def encode_fitted_reverb_stream(
    dry_mid: np.ndarray,
    dry_side: np.ndarray,
    target_space_mid: np.ndarray,
    target_space_side: np.ndarray,
    sr: int = 48000,
) -> tuple[bytes, bytes, dict, np.ndarray]:
    """Fits dynamic Mid/Side wet envelopes onto 8-line FDN reverb and compresses into <1.2 KB `VWR7`."""
    rt60_low = 1.48
    rt60_high = 0.58
    pre_delay = 960

    raw_m, raw_s = run_raw_fdn_stereo(dry_mid, dry_side, sr=sr, rt60_low=rt60_low, rt60_high=rt60_high, pre_delay_samples=pre_delay)
    n = len(dry_mid)
    n_hops = int(np.ceil(n / ENV_HOP)) + 1

    gains_m = np.zeros(n_hops, dtype=np.float32)
    gains_s = np.zeros(n_hops, dtype=np.float32)

    peak_target = float(np.sqrt(np.max(target_space_mid ** 2 + target_space_side ** 2)) + 1e-9)
    floor_rms = peak_target * 0.004

    for h in range(n_hops - 1):
        s0 = h * ENV_HOP
        s1 = min(n, s0 + ENV_HOP)
        tm_rms = float(np.sqrt(np.mean(target_space_mid[s0:s1].astype(np.float64) ** 2)))
        ts_rms = float(np.sqrt(np.mean(target_space_side[s0:s1].astype(np.float64) ** 2)))
        rm_rms = float(np.sqrt(np.mean(raw_m[s0:s1].astype(np.float64) ** 2)) + 1e-7)
        rs_rms = float(np.sqrt(np.mean(raw_s[s0:s1].astype(np.float64) ** 2)) + 1e-7)

        if max(tm_rms, ts_rms) < floor_rms:
            gains_m[h] = 0.0
            gains_s[h] = 0.0
        else:
            gains_m[h] = min(3.5, tm_rms / max(rm_rms, 3e-4))
            gains_s[h] = min(3.5, ts_rms / max(rs_rms, 3e-4))

    # Smooth 3-point temporal filter so gain automation never clicks
    gains_m = np.convolve(gains_m, [0.22, 0.56, 0.22], mode="same").astype(np.float32)
    gains_s = np.convolve(gains_s, [0.22, 0.56, 0.22], mode="same").astype(np.float32)

    codes_m = _quantize_gain_codes(gains_m)
    codes_s = _quantize_gain_codes(gains_s)

    dec_m = _dequantize_gain_codes(codes_m)
    dec_s = _dequantize_gain_codes(codes_s)

    # Reconstruct sample-accurate stereo reverb output via linear hop interpolation
    idx = np.arange(n, dtype=np.float64) / ENV_HOP
    h0 = np.floor(idx).astype(np.int32)
    frac = (idx - h0).astype(np.float32)
    env_m = dec_m[h0] * (1.0 - frac) + dec_m[np.minimum(h0 + 1, n_hops - 1)] * frac
    env_s = dec_s[h0] * (1.0 - frac) + dec_s[np.minimum(h0 + 1, n_hops - 1)] * frac

    syn_m = raw_m * env_m
    syn_s = raw_s * env_s
    syn_stereo = np.stack([syn_m + syn_s, syn_m - syn_s], axis=1).astype(np.float32)

    header = struct.pack(
        "<4sIIIIHHHHI",
        b"VWR7",
        sr,
        n,
        ENV_HOP,
        n_hops,
        round(rt60_low * 1000),
        round(rt60_high * 1000),
        pre_delay,
        0,
        0,
    )
    raw_vwr7 = header + codes_m.tobytes() + codes_s.tobytes()

    # Delta-encode codes_m and codes_s before LZMA for ultra-high compression (< 1 KB!)
    dm = np.empty_like(codes_m)
    ds = np.empty_like(codes_s)
    dm[0] = codes_m[0]
    ds[0] = codes_s[0]
    dm[1:] = (codes_m[1:].astype(np.int16) - codes_m[:-1].astype(np.int16)) & 0xFF
    ds[1:] = (codes_s[1:].astype(np.int16) - codes_s[:-1].astype(np.int16)) & 0xFF
    packed_hdr = struct.pack(
        "<4sIIIIHHHHI",
        b"VWRP",
        sr,
        n,
        ENV_HOP,
        n_hops,
        round(rt60_low * 1000),
        round(rt60_high * 1000),
        pre_delay,
        0,
        0,
    )
    packed_payload = packed_hdr + dm.tobytes() + ds.tobytes()
    comp_bytes = lzma.compress(packed_payload, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)

    meta = {
        "sampleRate": sr,
        "samples": n,
        "parameterBytes": len(raw_vwr7),
        "compressedBytes": len(comp_bytes),
    }
    return raw_vwr7, comp_bytes, meta, syn_stereo
