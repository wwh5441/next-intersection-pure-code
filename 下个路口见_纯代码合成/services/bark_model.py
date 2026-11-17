"""24-Band (Long) / 12-Band (Short) Bark Critical Band Psychoacoustic Model & dB Log-Scale Codec."""
from __future__ import annotations

import numpy as np

# Exact integer widths summing to 2048 (24 Bark-spaced bands) and 512 (12 Bark-spaced bands)
LONG_BAND_WIDTHS = (
    16, 16, 16, 16, 20, 20, 24, 24, 28, 32, 36, 40,
    48, 56, 64, 76, 88, 104, 124, 148, 180, 220, 288, 364,
)
SHORT_BAND_WIDTHS = (8, 8, 12, 12, 16, 20, 28, 36, 48, 68, 104, 152)

assert sum(LONG_BAND_WIDTHS) == 2048
assert sum(SHORT_BAND_WIDTHS) == 512

SCALE_MIN_DB = -52.0
SCALE_STEP_DB = 0.50


def get_band_edges(n: int) -> np.ndarray:
    widths = LONG_BAND_WIDTHS if n == 2048 else SHORT_BAND_WIDTHS
    return np.r_[0, np.cumsum(widths)].astype(np.int32)


def get_band_qmax_table(base_qmax: int, num_bands: int) -> np.ndarray:
    """Deterministic psychoacoustic quantization precision curve shared bit-for-bit with JS."""
    out = np.empty(num_bands, dtype=np.int32)
    for g in range(num_bands):
        u = g / max(1, num_bands - 1)
        if u <= 0.60:
            q = base_qmax
        elif u <= 0.80:
            q = max(7, int(round(base_qmax * 0.58)))
        else:
            q = max(5, int(round(base_qmax * 0.34)))
        out[g] = min(127, q)
    return out


def encode_scale_db(values: np.ndarray, ceil_mode: bool = False) -> np.ndarray:
    """Maps non-negative linear scale amplitudes to uint8 codes (0 = zero, 1..255 = -52.0..+75.0 dB)."""
    codes = np.zeros(values.shape, dtype=np.uint8)
    pos = values > 10.0 ** ((SCALE_MIN_DB - 0.25) / 20.0)
    if np.any(pos):
        db = 20.0 * np.log10(np.maximum(values[pos], 1e-12))
        raw_q = (db - SCALE_MIN_DB) / SCALE_STEP_DB
        q = (np.ceil(raw_q) if ceil_mode else np.rint(raw_q)).astype(np.int32) + 1
        codes[pos] = np.clip(q, 1, 255).astype(np.uint8)
    return codes


def decode_scale_db(codes: np.ndarray) -> np.ndarray:
    """Maps uint8 dB scale codes back to float32 linear amplitudes (exact match to JS LUT)."""
    lut = np.zeros(256, dtype=np.float32)
    idx = np.arange(1, 256, dtype=np.float64)
    lut[1:] = (10.0 ** ((SCALE_MIN_DB + (idx - 1.0) * SCALE_STEP_DB) / 20.0)).astype(np.float32)
    return lut[codes]
