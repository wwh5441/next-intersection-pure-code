"""5-Mode TDAC Adaptive Window MDCT (`2048` Long / `4x512` Short), Fast N/2-FFT IMDCT & CELT Spectral Folding."""
from __future__ import annotations

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


def mdct_fast_batch(frames: np.ndarray, n: int) -> np.ndarray:
    """Computes N-bin MDCT for batch `(B, 2*n)` via N/2-point FFT."""
    h = n // 2
    # Fold 2n -> n for DCT-IV:
    # u[0..h-1] = -frames[:, 3h-1 : n-1 : -1] - frames[:, 3h : 2n]
    # u[h..n-1] = frames[:, 0 : h] - frames[:, n-1 : h-1 : -1]
    a = frames[:, :h]
    b = frames[:, h:n][:, ::-1]
    c = frames[:, n : n + h][:, ::-1]
    d = frames[:, n + h :]
    u = np.concatenate([a - b, -c - d], axis=1)  # wait: check index alignment with DCT-IV!
    # Let us use exact DCT-IV on u where u[:h] corresponds to m=n..2n-1 and u[h:] to m=0..n-1:
    # From verified identity: X = M.T @ x where (M @ X)*(2/n) has signs [+Y[h:], -Y[h:][::-1], -Y[:h][::-1], -Y[:h]]
    # Transpose of [ +Y[h:], -rev(Y[h:]), -rev(Y[:h]), -Y[:h] ]:
    u_lo = -frames[:, n : n + h][:, ::-1] - frames[:, n + h :]
    u_hi = frames[:, :h] - frames[:, h:n][:, ::-1]
    u = np.concatenate([u_lo, u_hi], axis=1)

    r = np.arange(h)
    p = np.arange(h)
    pre = np.exp(-1j * np.pi * r / n)[None, :]
    post = np.exp(-1j * np.pi * (p + 0.25) / n)[None, :]
    z_in = (u[:, 0::2] + 1j * u[:, n - 1 :: -2]) * pre
    z_out = np.fft.fft(z_in, axis=1) * post
    out = np.empty((len(frames), n), dtype=np.float32)
    out[:, 0::2] = z_out.real
    out[:, 1::2] = -z_out.imag[:, ::-1]
    return out


def imdct_fast_batch(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Computes 2N-sample IMDCT for batch `(B, n)` via N/2-point FFT."""
    h = n // 2
    r = np.arange(h)
    p = np.arange(h)
    pre = np.exp(-1j * np.pi * r / n)[None, :]
    post = np.exp(-1j * np.pi * (p + 0.25) / n)[None, :]
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
    """Detects transients and assigns TDAC-consistent window modes (0..4) per 2048-hop superframe."""
    cutoff = min(900.0, sr * 0.25)
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    hp = sosfiltfilt(sos, padded).astype(np.float32)

    is_short = np.zeros(super_frames, dtype=bool)
    seg_len = 256
    for t in range(1, super_frames - 1):
        core = hp[t * LONG_BINS + 512 : t * LONG_BINS + 3584]
        energies = np.mean(core.reshape(-1, seg_len) ** 2, axis=1) + 1e-11
        max_e = float(np.max(energies))
        if max_e < 1e-7:
            continue
        ratios = energies[2:] / (0.5 * (energies[1:-1] + energies[:-2]))
        if float(np.max(ratios)) > 7.5:
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


def analyze_adaptive_mdct(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Analyzes 1D signal `y` into `(modes, coeffs)` of shape `(T,)` and `(T, 2048)`."""
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

    return modes, coeffs


def reconstruct_adaptive_mdct(modes: np.ndarray, coeffs: np.ndarray, sample_count: int) -> np.ndarray:
    """Reconstructs 1D waveform of `sample_count` samples from `(modes, coeffs)`."""
    super_frames = len(modes)
    out = np.zeros((super_frames + 1) * LONG_BINS, dtype=np.float64)

    for m_val in (MODE_ONLY_LONG, MODE_LONG_START, MODE_LONG_STOP, MODE_STOP_START):
        idx = np.flatnonzero(modes == m_val)
        if len(idx) == 0:
            continue
        w = LONG_WINDOWS[m_val][None, :]
        rec = imdct_fast_batch(coeffs[idx], LONG_BINS) * w
        for row, t in enumerate(idx):
            pos = int(t) * LONG_BINS
            out[pos : pos + 2 * LONG_BINS] += rec[row]

    short_idx = np.flatnonzero(modes == MODE_FOUR_SHORT)
    if len(short_idx) > 0:
        w_s = SHORT_WINDOW[None, :]
        for k in range(SHORT_COUNT):
            off = PAD_ZEROS + k * SHORT_BINS
            sub_c = coeffs[short_idx, k * SHORT_BINS : (k + 1) * SHORT_BINS]
            rec = imdct_fast_batch(sub_c, SHORT_BINS) * w_s
            for row, t in enumerate(short_idx):
                pos = int(t) * LONG_BINS + off
                out[pos : pos + 2 * SHORT_BINS] += rec[row]

    return out[LONG_BINS : LONG_BINS + sample_count].astype(np.float32)


def apply_celt_folding_1d(
    spectrum: np.ndarray,
    edges: np.ndarray,
    scales: np.ndarray,
    fold_start_band: int,
    seed_base: int,
) -> None:
    """In-place deterministic CELT-style high-band spectral folding matching JS bit-for-bit."""
    num_bands = len(edges) - 1
    if fold_start_band >= num_bands:
        return
    n = len(spectrum)
    src_lo = n >> 3
    src_len = n >> 2  # [n/8 .. 3n/8)
    for g in range(fold_start_band, num_bands):
        target_rms = float(scales[g])
        start_k = int(edges[g])
        end_k = int(edges[g + 1])
        if target_rms <= 0.0:
            spectrum[start_k:end_k] = 0.0
            continue
        width = end_k - start_k
        folded = np.empty(width, dtype=np.float64)
        for i in range(width):
            k = start_k + i
            src_val = float(spectrum[src_lo + (k % src_len)])
            # Exact 32-bit unsigned hash matching JS Math.imul
            h = ((seed_base * 2654435761) + (k * 2246822519)) & 0xFFFFFFFF
            h = ((h ^ (h >> 15)) * 2246822519) & 0xFFFFFFFF
            sign = -1.0 if (h & 0x80000000) else 1.0
            # Blend 85% folded harmonic/transient structure + 15% deterministic excitation floor
            noise_unit = (((h & 0xFFFF) / 32768.0) - 1.0) * 0.15
            folded[i] = sign * src_val + noise_unit * target_rms
        cur_rms = np.sqrt(np.mean(folded * folded))
        if cur_rms > 1e-12:
            spectrum[start_k:end_k] = (folded * (target_rms / cur_rms)).astype(np.float32)
        else:
            spectrum[start_k:end_k] = 0.0
