"""V4 Psychoacoustic Quantizer: Adaptive 2048/512 MDCT + 24-Bark Masking + CELT Spectral Folding (MDQ4)."""
from __future__ import annotations

import struct
import numpy as np

from .adaptive_mdct import (
    LONG_BINS,
    MODE_FOUR_SHORT,
    SHORT_BINS,
    SHORT_COUNT,
    analyze_adaptive_mdct,
    apply_celt_folding_1d,
    reconstruct_adaptive_mdct,
)
from .bark_model import (
    decode_scale_db,
    encode_scale_db,
    get_band_edges,
    get_band_qmax_table,
)


def _quantize_1d_block(
    block: np.ndarray,
    edges: np.ndarray,
    qmax_table: np.ndarray,
    power: float,
    retain_energy: float,
    gate_db: float,
    mask_low_db: float,
    mask_high_db: float,
    fold_start_band: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantizes a single Long (2048) or Short (512) spectral block into `(scale_codes, q_int8, restored_f32)`."""
    n = len(block)
    num_bands = len(edges) - 1
    work = block.copy()

    coded_limit = int(edges[min(fold_start_band, num_bands)])
    if coded_limit > 0 and retain_energy < 1.0:
        coded_slice = work[:coded_limit]
        sq = np.sort(coded_slice.astype(np.float64) ** 2)
        cum = np.cumsum(sq)
        if cum[-1] > 0.0:
            drop = int(np.sum(cum <= cum[-1] * (1.0 - retain_energy)))
            thresh = sq[min(drop, coded_limit - 1)]
            coded_slice[coded_slice.astype(np.float64) ** 2 < thresh] = 0.0

    raw_scales = np.zeros(num_bands, dtype=np.float32)
    masked_bands: list[np.ndarray] = []

    for g in range(num_bands):
        s_k, e_k = int(edges[g]), int(edges[g + 1])
        sub = work[s_k:e_k].copy()
        u = g / max(1, num_bands - 1)
        ath_lift = 8.5 if u > 0.82 else (3.5 if u > 0.68 else 0.0)
        eff_gate_lin = 10.0 ** ((gate_db + ath_lift) / 20.0)
        band_norm = float(np.sqrt(2.0 * np.sum(sub.astype(np.float64) ** 2)) / n)

        if band_norm < eff_gate_lin:
            raw_scales[g] = 0.0
            masked_bands.append(np.zeros_like(sub))
            continue

        if g >= fold_start_band:
            # High-band PS/PNS: store true subband RMS amplitude, zero coded pulses
            raw_scales[g] = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2)))
            masked_bands.append(np.zeros_like(sub))
        else:
            peak = float(np.max(np.abs(sub)))
            mask_db = mask_low_db * (1.0 - u) + mask_high_db * u
            mask_thresh = peak * (10.0 ** (-mask_db / 20.0))
            sub[np.abs(sub) < mask_thresh] = 0.0
            raw_scales[g] = peak
            masked_bands.append(sub)

    scale_codes = encode_scale_db(raw_scales, ceil_mode=True)
    if fold_start_band < num_bands:
        scale_codes[fold_start_band:] = encode_scale_db(raw_scales[fold_start_band:], ceil_mode=False)
    decoded_scales = decode_scale_db(scale_codes)

    q_out = np.zeros(n, dtype=np.int8)
    restored = np.zeros(n, dtype=np.float32)
    inv_power = 1.0 / power

    for g in range(min(fold_start_band, num_bands)):
        sc_val = float(decoded_scales[g])
        if sc_val <= 0.0:
            continue
        s_k, e_k = int(edges[g]), int(edges[g + 1])
        sub = masked_bands[g]
        qm = int(qmax_table[g])
        normed = np.clip(np.abs(sub) / sc_val, 0.0, 1.0)
        q_sub = np.rint(np.sign(sub) * (normed ** power) * qm).astype(np.int8)
        q_out[s_k:e_k] = q_sub
        restored[s_k:e_k] = (
            np.sign(q_sub) * ((np.abs(q_sub).astype(np.float32) / qm) ** inv_power) * sc_val
        )

    apply_celt_folding_1d(restored, edges, decoded_scales, fold_start_band, seed_base)
    return scale_codes, q_out, restored


def quantize_spectrum(y: np.ndarray, spec: dict, power: float) -> tuple[bytes, dict, np.ndarray]:
    """Encodes 1D waveform `y` into V4 `MDQ4` container bytes, returning `(raw_mdq4, stats, reconstructed)`."""
    sr = int(spec["sampleRate"])
    base_qmax = int(spec["baseQmax"])
    retain_energy = float(spec.get("retainEnergy", 0.995))
    gate_db = float(spec["gateDb"])
    mask_low_db = float(spec.get("maskLowDb", 26.0))
    mask_high_db = float(spec.get("maskHighDb", 15.0))
    fold_long = int(spec.get("foldStartLong", 24))
    fold_short = int(spec.get("foldStartShort", 12))

    modes, coeffs = analyze_adaptive_mdct(y, sr)
    t_count = len(modes)

    edges_l = get_band_edges(LONG_BINS)
    edges_s = get_band_edges(SHORT_BINS)
    qmax_l = get_band_qmax_table(base_qmax, len(edges_l) - 1)
    qmax_s = get_band_qmax_table(base_qmax, len(edges_s) - 1)

    all_scales = np.zeros((t_count, 48), dtype=np.uint8)
    all_q = np.zeros((t_count, LONG_BINS), dtype=np.int8)
    restored_coeffs = np.zeros((t_count, LONG_BINS), dtype=np.float32)

    for t in range(t_count):
        if modes[t] == MODE_FOUR_SHORT:
            for k in range(SHORT_COUNT):
                s_off = k * SHORT_BINS
                sc_k, q_k, rec_k = _quantize_1d_block(
                    coeffs[t, s_off : s_off + SHORT_BINS],
                    edges_s,
                    qmax_s,
                    power,
                    retain_energy,
                    gate_db,
                    mask_low_db,
                    mask_high_db,
                    fold_short,
                    t * 4 + k,
                )
                all_scales[t, k * 12 : (k + 1) * 12] = sc_k
                all_q[t, s_off : s_off + SHORT_BINS] = q_k
                restored_coeffs[t, s_off : s_off + SHORT_BINS] = rec_k
        else:
            sc_l, q_l, rec_l = _quantize_1d_block(
                coeffs[t],
                edges_l,
                qmax_l,
                power,
                retain_energy,
                gate_db,
                mask_low_db,
                mask_high_db,
                fold_long,
                t * 4,
            )
            all_scales[t, :24] = sc_l
            all_q[t] = q_l
            restored_coeffs[t] = rec_l

    header = struct.pack(
        "<4sIHHIIBBBBBBHI",
        b"MDQ4",
        sr,
        LONG_BINS,
        SHORT_BINS,
        t_count,
        len(y),
        base_qmax,
        round(power * 100),
        fold_long,
        fold_short,
        24,
        12,
        0,
        0,
    )
    frame_bytes = np.concatenate(
        [modes[:, None], all_scales, all_q.view(np.uint8)],
        axis=1,
    )
    raw = header + frame_bytes.tobytes()

    reconstructed = reconstruct_adaptive_mdct(modes, restored_coeffs, len(y))
    snr = 10.0 * np.log10(
        (np.sum(y.astype(np.float64) ** 2) + 1e-15)
        / (np.sum((y - reconstructed).astype(np.float64) ** 2) + 1e-15)
    )
    short_ratio = round(float(np.mean(modes == MODE_FOUR_SHORT)) * 100.0, 1)
    meta = {
        "sampleRate": sr,
        "samples": int(len(y)),
        "superFrames": t_count,
        "blockSize": LONG_BINS,
        "shortBlockSize": SHORT_BINS,
        "shortFramePercent": short_ratio,
        "parameterBytes": len(raw),
        "snrDb": round(float(snr), 2),
        "meanActiveCoefficients": round(float(np.mean(np.count_nonzero(all_q, axis=1))), 1),
    }
    return raw, meta, reconstructed
