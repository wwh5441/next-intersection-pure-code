"""4-Band 8ms High-Resolution Sibilant & 20kHz Breath Air Synthesizer (`VWA7` / `VWAP`, ~12 KB)."""
from __future__ import annotations

import lzma
import math
import struct
import numpy as np
from scipy.signal import lfilter

AIR_HOP = 384  # 8.0 ms envelope frame at 48,000 Hz (125 Hz temporal resolution)
AIR_MIN_DB = -64.0
AIR_STEP_DB = 0.45  # codes 1..127 => -64.0 dB .. -7.3 dB


def design_2stage_bandpass(f_lo: float, f_hi: float, sr: int = 48000) -> tuple[float, float, float, float]:
    """Returns `(b0, a1, a2, inv_l2_norm)` matching `src/dsp/air-stream.js` bit-for-bit."""
    f0 = math.sqrt(f_lo * f_hi)
    q = max(0.55, f0 / max(200.0, f_hi - f_lo))
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    inv_a0 = 1.0 / (1.0 + alpha)
    b0 = alpha * inv_a0
    a1 = (-2.0 * math.cos(w0)) * inv_a0
    a2 = (1.0 - alpha) * inv_a0

    imp = np.zeros(1024, dtype=np.float64)
    imp[0] = 1.0
    b = [b0, 0.0, -b0]
    a = [1.0, a1, a2]
    h1 = lfilter(b, a, imp)
    h2 = lfilter(b, a, h1)
    l2_norm = math.sqrt(float(np.sum(h2 * h2)) + 1e-18)
    return b0, a1, a2, 1.0 / l2_norm


def filter_band_2stage(x: np.ndarray, f_lo: float, f_hi: float, sr: int = 48000) -> np.ndarray:
    b0, a1, a2, _ = design_2stage_bandpass(f_lo, f_hi, sr)
    b = [b0, 0.0, -b0]
    a = [1.0, a1, a2]
    y1 = lfilter(b, a, x.astype(np.float64))
    y2 = lfilter(b, a, y1)
    return y2.astype(np.float32)


def encode_high_air_stream(
    y_48k: np.ndarray,
    bands_hz: list[tuple[float, float]],
    sr: int = 48000,
    gate_db: float = -62.0,
) -> tuple[bytes, bytes, dict, np.ndarray]:
    """Encodes 4-band high-frequency sibilant & breath air (`f_cut .. 20 kHz`) into compact `VWA7`."""
    assert len(bands_hz) == 4
    n = len(y_48k)
    n_hops = int(np.ceil(n / AIR_HOP)) + 1
    gate_lin = 10.0 ** (gate_db / 20.0)

    codes = np.zeros((n_hops, 4), dtype=np.uint8)
    decoded_rms = np.zeros((n_hops, 4), dtype=np.float32)

    for b_idx, (f_lo, f_hi) in enumerate(bands_hz):
        sub = filter_band_2stage(y_48k, f_lo, f_hi, sr=sr)
        pad_len = (n_hops - 1) * AIR_HOP - n
        sub_pad = np.pad(sub, (0, max(0, pad_len)))
        frames = sub_pad[: (n_hops - 1) * AIR_HOP].reshape(n_hops - 1, AIR_HOP)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

        act = rms >= gate_lin
        if np.any(act):
            db = 20.0 * np.log10(np.maximum(rms[act], 1e-9))
            q = np.clip(np.rint((db - AIR_MIN_DB) / AIR_STEP_DB).astype(np.int32) + 1, 1, 127).astype(np.uint8)
            codes[: n_hops - 1, b_idx][act] = q

        nz = codes[:, b_idx] > 0
        decoded_rms[nz, b_idx] = (
            10.0 ** ((AIR_MIN_DB + (codes[nz, b_idx].astype(np.float32) - 1.0) * AIR_STEP_DB) / 20.0)
        ).astype(np.float32)

    # Synthesize Python reference reconstruction using unit-variance noise + exact calibrated 2-stage bandpass
    rng = np.random.default_rng(20261115)
    idx = np.arange(n, dtype=np.float64) / AIR_HOP
    h0 = np.floor(idx).astype(np.int32)
    h1 = np.minimum(h0 + 1, n_hops - 1)
    frac = (idx - h0).astype(np.float32)

    syn = np.zeros(n, dtype=np.float32)
    for b_idx, (f_lo, f_hi) in enumerate(bands_hz):
        b0, a1, a2, inv_norm = design_2stage_bandpass(f_lo, f_hi, sr=sr)
        b = [b0, 0.0, -b0]
        a = [1.0, a1, a2]
        white = rng.standard_normal(n)
        shaped = lfilter(b, a, lfilter(b, a, white)) * inv_norm
        env = decoded_rms[h0, b_idx] * (1.0 - frac) + decoded_rms[h1, b_idx] * frac
        syn += shaped.astype(np.float32) * env

    band_u16 = [val for pair in bands_hz for val in (round(pair[0]), round(pair[1]))]
    header = struct.pack(
        "<4sIIII8H",
        b"VWA7",
        sr,
        n,
        AIR_HOP,
        n_hops,
        *band_u16,
    )
    raw_vwa7 = header + codes.tobytes()

    # Sparse active-hop + delta packing (`VWAP`)
    active_hop = np.any(codes > 0, axis=1)
    act_codes = codes[active_hop].astype(np.int16)
    if len(act_codes) > 0:
        deltas = np.empty_like(act_codes, dtype=np.uint8)
        deltas[0] = act_codes[0].astype(np.uint8)
        deltas[1:] = ((act_codes[1:] - act_codes[:-1]) & 0xFF).astype(np.uint8)
        delta_bytes = deltas.T.tobytes()  # Band-major layout for stronger LZMA locality
    else:
        delta_bytes = b""

    packed_hdr = struct.pack(
        "<4sIIII8HI",
        b"VWAP",
        sr,
        n,
        AIR_HOP,
        n_hops,
        *band_u16,
        int(len(act_codes)),
    )
    packed_payload = packed_hdr + np.packbits(active_hop).tobytes() + delta_bytes
    comp_bytes = lzma.compress(packed_payload, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)

    meta = {
        "sampleRate": sr,
        "samples": n,
        "parameterBytes": len(raw_vwa7),
        "compressedBytes": len(comp_bytes),
        "activeHopPercent": round(float(np.mean(active_hop)) * 100.0, 1),
    }
    return raw_vwa7, comp_bytes, meta, syn
