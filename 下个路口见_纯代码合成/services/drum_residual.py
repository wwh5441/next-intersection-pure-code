"""Closed-Loop Sample-Exact Drum Bar-Loop Prototype Dictionary & Residual Decomposition."""
from __future__ import annotations

import numpy as np


def detect_bar_period_samples(y: np.ndarray, sr: int) -> float:
    """Finds exact studio bar period in fractional samples using FFT autocorrelation + parabolic peak fit."""
    t_lo = min(len(y) // 6, sr * 20)
    t_hi = min(len(y) - sr * 5, sr * 150)
    seg = y[t_lo:t_hi].astype(np.float64)
    n_fft = 1 << int(np.ceil(np.log2(len(seg) * 2)))
    spec = np.fft.rfft(seg, n=n_fft)
    ac = np.fft.irfft(np.abs(spec) ** 2)[: int(sr * 6.5)]
    ac /= max(float(ac[0]), 1e-12)

    # Search 2-bar period in [3.6s .. 6.0s] (80..133 BPM in 4/4) for highest stability
    lo = int(3.6 * sr)
    hi = min(len(ac) - 2, int(6.0 * sr))
    pk = lo + int(np.argmax(ac[lo:hi]))
    # Parabolic subsample refinement
    y0, y1, y2 = float(ac[pk - 1]), float(ac[pk]), float(ac[pk + 1])
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    two_bar_samples = pk + float(np.clip(delta, -0.5, 0.5))
    return two_bar_samples * 0.5


def build_drum_prototypes(y: np.ndarray, sr: int, n_protos: int = 4) -> tuple[np.ndarray, float, int]:
    """Clusters 1-bar drum segments into `n_protos` clean prototype bars of length `bar_samples`."""
    bar_period = detect_bar_period_samples(y, sr)
    bar_samples = int(round(bar_period))
    num_bars = int((len(y) - bar_samples) // bar_period)

    bars = []
    energies = []
    for m in range(num_bars):
        s = int(round(m * bar_period))
        seg = y[s : s + bar_samples].astype(np.float64)
        e = float(np.mean(seg * seg))
        bars.append(seg)
        energies.append(e)

    bars_arr = np.stack(bars, axis=0)
    energies_arr = np.array(energies)
    med_e = float(np.median(energies_arr[energies_arr > 1e-7])) if np.any(energies_arr > 1e-7) else 1.0
    active_idx = np.flatnonzero(energies_arr > med_e * 0.12)
    if len(active_idx) < n_protos:
        protos = np.zeros((n_protos, bar_samples), dtype=np.float32)
        for i, idx in enumerate(active_idx):
            protos[i] = bars_arr[idx].astype(np.float32)
        return protos, bar_period, bar_samples

    active_bars = bars_arr[active_idx]
    norms = np.linalg.norm(active_bars, axis=1, keepdims=True) + 1e-12
    unit_bars = active_bars / norms

    # Deterministic K-means++ style initialization sorted by energy quantiles
    quantiles = np.linspace(0.2, 0.9, n_protos)
    order = np.argsort(energies_arr[active_idx])
    init_sel = [order[int(q * (len(order) - 1))] for q in quantiles]
    centers = unit_bars[init_sel].copy()

    for _ in range(14):
        sim = unit_bars @ centers.T
        labels = np.argmax(sim, axis=1)
        for k in range(n_protos):
            members = active_bars[labels == k]
            if len(members) > 0:
                mean_vec = np.median(members, axis=0)
                nrm = np.linalg.norm(mean_vec)
                if nrm > 1e-9:
                    centers[k] = mean_vec / nrm

    # Final prototype = mean of assigned unnormalized bars in each cluster
    sim = unit_bars @ centers.T
    labels = np.argmax(sim, axis=1)
    protos = np.zeros((n_protos, bar_samples), dtype=np.float32)
    for k in range(n_protos):
        members = active_bars[labels == k]
        if len(members) > 0:
            # Trim subtle outlier fills via median-weighted average for ultra-clean loop prototype
            protos[k] = np.median(members, axis=0).astype(np.float32)
    return protos, bar_period, bar_samples


def synthesize_drum_prediction(
    total_samples: int,
    decoded_protos: np.ndarray,
    bar_samples: int,
    schedule: list[list[int]],
    fade_samples: int = 64,
) -> np.ndarray:
    """Synthesizes predicted drum waveform matching JS `DrumLoopSynth` sample-for-sample."""
    pred = np.zeros(total_samples, dtype=np.float32)
    fade_in = (0.5 - 0.5 * np.cos(np.pi * np.arange(fade_samples) / fade_samples)).astype(np.float32)
    fade_out = (1.0 - fade_in).astype(np.float32)

    for start, proto_idx, gain_byte in schedule:
        if gain_byte == 0:
            continue
        gain = gain_byte / 128.0
        end = min(total_samples, start + bar_samples)
        length = end - start
        if length <= 0:
            continue
        seg = decoded_protos[proto_idx, :length] * gain
        if length > 2 * fade_samples:
            seg = seg.copy()
            seg[:fade_samples] *= fade_in
            seg[-fade_samples:] *= fade_out
        pred[start:end] += seg
    return pred


def fit_closed_loop_schedule(
    y: np.ndarray,
    decoded_protos: np.ndarray,
    bar_period: float,
    bar_samples: int,
    fade_samples: int = 64,
) -> tuple[list[list[int]], np.ndarray]:
    """Fits per-bar `(startSample, protoIdx, gainByte)` against ALREADY-DECODED prototypes (Closed-Loop)."""
    proto_energies = np.sum(decoded_protos.astype(np.float64) ** 2, axis=1) + 1e-12
    num_bars = int(np.ceil(len(y) / bar_period))
    schedule: list[list[int]] = []

    for m in range(num_bars):
        start = int(round(m * bar_period))
        if start >= len(y):
            break
        end = min(len(y), start + bar_samples)
        length = end - start
        if length < bar_samples // 2:
            break
        seg = y[start:end].astype(np.float64)
        seg_e = float(np.sum(seg * seg))
        if seg_e < 1e-6:
            continue

        best_k = 0
        best_g_byte = 0
        best_reduction = 0.0
        for k in range(len(decoded_protos)):
            p = decoded_protos[k, :length].astype(np.float64)
            p_e = float(np.sum(p * p)) + 1e-12
            g_raw = float(np.sum(seg * p)) / p_e
            g_byte = int(np.clip(round(g_raw * 128.0), 0, 192))
            if g_byte == 0:
                continue
            g = g_byte / 128.0
            res_e = float(np.sum((seg - g * p) ** 2))
            reduction = 1.0 - res_e / seg_e
            if reduction > best_reduction:
                best_reduction = reduction
                best_k = k
                best_g_byte = g_byte

        # Only activate prototype prediction if it reduces bar residual energy by >= 12%
        if best_reduction >= 0.12 and best_g_byte > 0:
            schedule.append([start, best_k, best_g_byte])

    pred = synthesize_drum_prediction(len(y), decoded_protos, bar_samples, schedule, fade_samples=fade_samples)
    return schedule, pred
