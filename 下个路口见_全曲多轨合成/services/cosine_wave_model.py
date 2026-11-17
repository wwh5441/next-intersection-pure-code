"""True-Phase 5-Mode TDAC Adaptive Local Cosine Wavelet (`2048` Long / `4x512` Short) & 24-Bark Quantizer (`VWQ6`)."""
from __future__ import annotations

import struct
import numpy as np
from scipy.signal import butter, sosfiltfilt

LONG_BINS = 2048
SHORT_BINS = 512
SHORT_COUNT = 4
PAD_ZEROS = (LONG_BINS - SHORT_BINS) // 2  # 768

MODE_ONLY_LONG = 0
MODE_LONG_START = 1
MODE_FOUR_SHORT = 2
MODE_LONG_STOP = 3
MODE_STOP_START = 4

LONG_BAND_WIDTHS = [
    16, 16, 16, 16, 20, 20, 24, 24, 28, 32, 36, 40,
    48, 56, 64, 76, 88, 104, 124, 148, 180, 220, 288, 364,
]
SHORT_BAND_WIDTHS = [8, 8, 12, 12, 16, 20, 28, 36, 48, 68, 104, 152]

SCALE_MIN_DB = -58.0
SCALE_STEP_DB = 0.45

SCALE_LUT = np.array(
    [0.0 if c == 0 else 10.0 ** ((SCALE_MIN_DB + (c - 1) * SCALE_STEP_DB) / 20.0) for c in range(256)],
    dtype=np.float32,
)


def _build_windows() -> tuple[dict[int, np.ndarray], np.ndarray]:
    sin_l = np.sin(np.pi * (np.arange(2 * LONG_BINS) + 0.5) / (2 * LONG_BINS)).astype(np.float64)
    sin_s = np.sin(np.pi * (np.arange(2 * SHORT_BINS) + 0.5) / (2 * SHORT_BINS)).astype(np.float64)
    l_left, l_right = sin_l[:LONG_BINS], sin_l[LONG_BINS:]
    s_left = np.concatenate([np.zeros(PAD_ZEROS), sin_s[:SHORT_BINS], np.ones(PAD_ZEROS)])
    s_right = np.concatenate([np.ones(PAD_ZEROS), sin_s[SHORT_BINS:], np.zeros(PAD_ZEROS)])
    modes = {
        MODE_ONLY_LONG: np.concatenate([l_left, l_right]),
        MODE_LONG_START: np.concatenate([l_left, s_right]),
        MODE_LONG_STOP: np.concatenate([s_left, l_right]),
        MODE_STOP_START: np.concatenate([s_left, s_right]),
    }
    return modes, sin_s


LONG_WINDOWS, SHORT_WINDOW = _build_windows()


def get_band_edges(n: int) -> np.ndarray:
    widths = LONG_BAND_WIDTHS if n == LONG_BINS else SHORT_BAND_WIDTHS
    return np.concatenate(([0], np.cumsum(widths))).astype(np.int32)


def get_band_qmax_table(base_qmax: int, num_bands: int) -> np.ndarray:
    out = np.zeros(num_bands, dtype=np.int32)
    for g in range(num_bands):
        u = g / max(1, num_bands - 1)
        if u <= 0.65:
            q = base_qmax
        elif u <= 0.85:
            q = max(9, round(base_qmax * 0.68))
        else:
            q = max(7, round(base_qmax * 0.45))
        out[g] = min(127, q)
    return out


def encode_scale_db(scales: np.ndarray) -> np.ndarray:
    codes = np.zeros(len(scales), dtype=np.uint8)
    pos = scales > 0.0
    if np.any(pos):
        db = 20.0 * np.log10(np.maximum(scales[pos], 1e-12))
        c = np.ceil((db - SCALE_MIN_DB) / SCALE_STEP_DB).astype(np.int32) + 1
        codes[pos] = np.clip(c, 1, 255).astype(np.uint8)
    return codes


def mdct_fast_batch(frames: np.ndarray, n: int) -> np.ndarray:
    h = n // 2
    u_lo = -frames[:, n : n + h][:, ::-1] - frames[:, n + h :]
    u_hi = frames[:, :h] - frames[:, h:n][:, ::-1]
    u = np.concatenate([u_lo, u_hi], axis=1)

    r = np.arange(h)
    pre = np.exp(-1j * np.pi * r / n)[None, :]
    post = np.exp(-1j * np.pi * (r + 0.25) / n)[None, :]
    z_in = (u[:, 0::2] + 1j * u[:, n - 1 :: -2]) * pre
    z_out = np.fft.fft(z_in, axis=1) * post
    out = np.empty((len(frames), n), dtype=np.float32)
    out[:, 0::2] = z_out.real
    out[:, 1::2] = -z_out.imag[:, ::-1]
    return out


def imdct_fast_batch(coeffs: np.ndarray, n: int) -> np.ndarray:
    h = n // 2
    r = np.arange(h)
    pre = np.exp(-1j * np.pi * r / n)[None, :]
    post = np.exp(-1j * np.pi * (r + 0.25) / n)[None, :]
    z_in = (coeffs[:, 0::2] + 1j * coeffs[:, n - 1 :: -2]) * pre
    z_out = np.fft.fft(z_in, axis=1) * post
    y_dct = np.empty((len(coeffs), n), dtype=np.float64)
    y_dct[:, 0::2] = z_out.real
    y_dct[:, 1::2] = -z_out.imag[:, ::-1]
    scale = 2.0 / n
    return (
        np.concatenate(
            [y_dct[:, h:], -y_dct[:, h:][:, ::-1], -y_dct[:, :h][:, ::-1], -y_dct[:, :h]],
            axis=1,
        )
        * scale
    )


def detect_window_modes(padded: np.ndarray, super_frames: int, sr: int) -> np.ndarray:
    cutoff = min(900.0, sr * 0.25)
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    hp = sosfiltfilt(sos, padded).astype(np.float32)

    is_short = np.zeros(super_frames, dtype=bool)
    seg_len = 256
    for t in range(1, super_frames - 1):
        core = hp[t * LONG_BINS + 512 : t * LONG_BINS + 3584]
        energies = np.mean(core.reshape(-1, seg_len) ** 2, axis=1) + 1e-11
        if float(np.max(energies)) < 1e-7:
            continue
        ratios = energies[2:] / (0.5 * (energies[1:-1] + energies[:-2]))
        if float(np.max(ratios)) > 6.5:
            is_short[t] = True

    modes = np.zeros(super_frames, dtype=np.uint8)
    for t in range(super_frames):
        if is_short[t]:
            modes[t] = MODE_FOUR_SHORT
        else:
            prev_s = t > 0 and is_short[t - 1]
            next_s = t + 1 < super_frames and is_short[t + 1]
            if prev_s and next_s:
                modes[t] = MODE_STOP_START
            elif prev_s:
                modes[t] = MODE_LONG_STOP
            elif next_s:
                modes[t] = MODE_LONG_START
            else:
                modes[t] = MODE_ONLY_LONG
    return modes


def _quantize_cosine_block(
    block: np.ndarray,
    edges: np.ndarray,
    qmax_table: np.ndarray,
    power: float,
    retain_energy: float,
    gate_db: float,
    mask_low_db: float,
    mask_high_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(block)
    num_bands = len(edges) - 1
    work = block.copy()

    if retain_energy < 1.0:
        sq = np.sort(work.astype(np.float64) ** 2)
        cum = np.cumsum(sq)
        if cum[-1] > 0.0:
            drop = int(np.sum(cum <= cum[-1] * (1.0 - retain_energy)))
            thresh = sq[min(drop, n - 1)]
            work[work.astype(np.float64) ** 2 < thresh] = 0.0

    raw_scales = np.zeros(num_bands, dtype=np.float32)
    masked_bands: list[np.ndarray] = []

    for g in range(num_bands):
        s_k, e_k = int(edges[g]), int(edges[g + 1])
        sub = work[s_k:e_k].copy()
        u = g / max(1, num_bands - 1)
        ath_lift = 7.0 if u > 0.85 else (3.0 if u > 0.70 else 0.0)
        eff_gate_lin = 10.0 ** ((gate_db + ath_lift) / 20.0)
        band_norm = float(np.sqrt(2.0 * np.sum(sub.astype(np.float64) ** 2)) / n)

        if band_norm < eff_gate_lin:
            raw_scales[g] = 0.0
            masked_bands.append(np.zeros_like(sub))
            continue

        peak = float(np.max(np.abs(sub)))
        mask_db = mask_low_db * (1.0 - u) + mask_high_db * u
        mask_thresh = peak * (10.0 ** (-mask_db / 20.0))
        sub[np.abs(sub) < mask_thresh] = 0.0
        raw_scales[g] = peak
        masked_bands.append(sub)

    scale_codes = encode_scale_db(raw_scales)
    decoded_scales = SCALE_LUT[scale_codes]

    q_out = np.zeros(n, dtype=np.int8)
    restored = np.zeros(n, dtype=np.float32)
    inv_power = 1.0 / power

    for g in range(num_bands):
        sc_val = float(decoded_scales[g])
        if sc_val <= 0.0:
            continue
        s_k, e_k = int(edges[g]), int(edges[g + 1])
        sub = masked_bands[g]
        qm = int(qmax_table[g])
        normed = np.clip(np.abs(sub) / sc_val, 0.0, 1.0)
        q_sub = np.rint(np.sign(sub) * (normed ** power) * qm).astype(np.int8)
        q_out[s_k:e_k] = q_sub
        restored[s_k:e_k] = np.sign(q_sub) * ((np.abs(q_sub).astype(np.float32) / qm) ** inv_power) * sc_val

    return scale_codes, q_out, restored


def encode_cosine_wave_stream(y: np.ndarray, spec: dict, power: float) -> tuple[bytes, dict, np.ndarray]:
    """Encodes 1D vocal sub-stem `y` into true-phase `VWQ6` local cosine wave stream."""
    sr = int(spec["sampleRate"])
    base_qmax = int(spec["baseQmax"])
    retain_energy = float(spec.get("retainEnergy", 0.998))
    gate_db = float(spec["gateDb"])
    mask_low_db = float(spec.get("maskLowDb", 28.0))
    mask_high_db = float(spec.get("maskHighDb", 20.0))

    super_frames = int(np.ceil(len(y) / LONG_BINS)) + 1
    total_len = (super_frames + 1) * LONG_BINS
    padded = np.pad(y.astype(np.float64), (LONG_BINS, total_len - len(y) - LONG_BINS))
    modes = detect_window_modes(padded, super_frames, sr)

    coeffs = np.zeros((super_frames, LONG_BINS), dtype=np.float32)
    for m_val in (MODE_ONLY_LONG, MODE_LONG_START, MODE_LONG_STOP, MODE_STOP_START):
        idx = np.flatnonzero(modes == m_val)
        if len(idx) == 0:
            continue
        w = LONG_WINDOWS[m_val][None, :]
        batch = np.stack([padded[t * LONG_BINS : t * LONG_BINS + 2 * LONG_BINS] for t in idx], axis=0) * w
        coeffs[idx] = mdct_fast_batch(batch, LONG_BINS)

    short_idx = np.flatnonzero(modes == MODE_FOUR_SHORT)
    if len(short_idx) > 0:
        w_s = SHORT_WINDOW[None, :]
        for k in range(SHORT_COUNT):
            off = PAD_ZEROS + k * SHORT_BINS
            batch = (
                np.stack([padded[t * LONG_BINS + off : t * LONG_BINS + off + 2 * SHORT_BINS] for t in short_idx], axis=0)
                * w_s
            )
            coeffs[short_idx, k * SHORT_BINS : (k + 1) * SHORT_BINS] = mdct_fast_batch(batch, SHORT_BINS)

    edges_l = get_band_edges(LONG_BINS)
    edges_s = get_band_edges(SHORT_BINS)
    qmax_l = get_band_qmax_table(base_qmax, len(edges_l) - 1)
    qmax_s = get_band_qmax_table(base_qmax, len(edges_s) - 1)

    all_scales = np.zeros((super_frames, 48), dtype=np.uint8)
    all_q = np.zeros((super_frames, LONG_BINS), dtype=np.int8)
    restored_coeffs = np.zeros((super_frames, LONG_BINS), dtype=np.float32)

    for t in range(super_frames):
        if modes[t] == MODE_FOUR_SHORT:
            for k in range(SHORT_COUNT):
                s_off = k * SHORT_BINS
                sc_k, q_k, rec_k = _quantize_cosine_block(
                    coeffs[t, s_off : s_off + SHORT_BINS],
                    edges_s,
                    qmax_s,
                    power,
                    retain_energy,
                    gate_db,
                    mask_low_db,
                    mask_high_db,
                )
                all_scales[t, k * 12 : (k + 1) * 12] = sc_k
                all_q[t, s_off : s_off + SHORT_BINS] = q_k
                restored_coeffs[t, s_off : s_off + SHORT_BINS] = rec_k
        else:
            sc_l, q_l, rec_l = _quantize_cosine_block(
                coeffs[t],
                edges_l,
                qmax_l,
                power,
                retain_energy,
                gate_db,
                mask_low_db,
                mask_high_db,
            )
            all_scales[t, :24] = sc_l
            all_q[t] = q_l
            restored_coeffs[t] = rec_l

    # Canonicalize silent frames (zero coefficients => zero IMDCT regardless of window mode) to mode 0
    modes[np.all(all_scales == 0, axis=1)] = MODE_ONLY_LONG

    header = struct.pack(
        "<4sIHHIIBBBBBBHI",
        b"VWQ6",
        sr,
        LONG_BINS,
        SHORT_BINS,
        super_frames,
        len(y),
        base_qmax,
        round(power * 100),
        24,
        12,
        24,
        12,
        0,
        0,
    )
    frame_bytes = np.concatenate([modes[:, None], all_scales, all_q.view(np.uint8)], axis=1)
    raw = header + frame_bytes.tobytes()

    # Exact TDAC wave reconstruction
    out = np.zeros((super_frames + 1) * LONG_BINS, dtype=np.float64)
    for m_val in (MODE_ONLY_LONG, MODE_LONG_START, MODE_LONG_STOP, MODE_STOP_START):
        idx = np.flatnonzero(modes == m_val)
        if len(idx) == 0:
            continue
        w = LONG_WINDOWS[m_val][None, :]
        rec = imdct_fast_batch(restored_coeffs[idx], LONG_BINS) * w
        for row, t in enumerate(idx):
            pos = int(t) * LONG_BINS
            out[pos : pos + 2 * LONG_BINS] += rec[row]

    if len(short_idx) > 0:
        w_s = SHORT_WINDOW[None, :]
        for k in range(SHORT_COUNT):
            off = PAD_ZEROS + k * SHORT_BINS
            sub_c = restored_coeffs[short_idx, k * SHORT_BINS : (k + 1) * SHORT_BINS]
            rec = imdct_fast_batch(sub_c, SHORT_BINS) * w_s
            for row, t in enumerate(short_idx):
                pos = int(t) * LONG_BINS + off
                out[pos : pos + 2 * SHORT_BINS] += rec[row]

    reconstructed = out[LONG_BINS : LONG_BINS + len(y)].astype(np.float32)
    snr = 10.0 * np.log10(
        (np.sum(y.astype(np.float64) ** 2) + 1e-15)
        / (np.sum((y - reconstructed).astype(np.float64) ** 2) + 1e-15)
    )
    meta = {
        "sampleRate": sr,
        "samples": int(len(y)),
        "superFrames": super_frames,
        "parameterBytes": len(raw),
        "waveSnrDb": round(float(snr), 2),
        "shortFramePercent": round(float(np.mean(modes == MODE_FOUR_SHORT)) * 100.0, 1),
    }
    return raw, meta, reconstructed
