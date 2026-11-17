"""AI NMF Percussive KitTimbre x 98-BPM Groove Residual + Modal/Transient Wave Modeler (`WVD5`)."""
from __future__ import annotations

import struct
import librosa
import numpy as np
from scipy.cluster.vq import kmeans2
import torch

from .audio_io import resample_audio


def _detect_bar_frames(onset_env: np.ndarray, fps: float) -> int:
    """Detects exact 4/4 bar length in frames around 88..112 BPM via autocorrelation."""
    x = onset_env - np.mean(onset_env)
    n = 1 << (len(x) * 2 - 1).bit_length()
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, n=n)) ** 2, n=n)[: len(x)]
    # 1 bar at 88..112 BPM = 2.14s .. 2.73s
    lo = int(round(2.15 * fps))
    hi = min(len(ac) - 2, int(round(2.72 * fps)))
    best = lo + int(np.argmax(ac[lo:hi]))
    return best


def encode_drums_wave(drums_48k: np.ndarray, orig_sr: int, spec: dict) -> tuple[bytes, dict, np.ndarray]:
    """Decomposes drums into Swept-Sine Kick + Snare Body Modal + NMF Transient Kit + 98 BPM Groove (`WVD5`)."""
    dsr = int(spec["sampleRate"])
    hop = int(spec["hopSamples"])
    n_comp = int(spec["nComponents"])
    n_bands = int(spec["nBands"])
    n_protos = int(spec["groovePrototypes"])
    q_levels = int(spec["actQuantLevels"])

    drums = resample_audio(drums_48k, orig_sr, dsr)
    n_fft = 512  # Crisp 21.3ms window (10.6ms mainlobe) -> pinpoint percussion attacks!
    pad = n_fft // 2
    stft = librosa.stft(np.pad(drums, (pad, pad)), n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=False)
    mag = np.abs(stft).astype(np.float32)
    n_bins, n_frames = mag.shape
    freqs = np.linspace(0.0, dsr / 2.0, n_bins, dtype=np.float32)
    rms_scale = float(np.sqrt(2.0 / (n_fft * n_fft * 0.375)))

    # 1. Modal Low-End Analysis: Kick (38..135 Hz) & Snare/Tom Tonal Body (135..255 Hz)
    kick_mask = (freqs >= 35.0) & (freqs < 135.0)
    body_mask = (freqs >= 135.0) & (freqs < 255.0)
    high_mask = freqs >= 230.0

    # Use high-res 2048 FFT once on whole drum stem to pinpoint exact Kick & Snare resonant frequencies
    mag_hi_res = np.mean(np.abs(librosa.stft(drums, n_fft=2048, hop_length=512)), axis=1)
    hr_freqs = np.linspace(0.0, dsr / 2.0, len(mag_hi_res))
    km_hr = (hr_freqs >= 45.0) & (hr_freqs <= 95.0)
    bm_hr = (hr_freqs >= 155.0) & (hr_freqs <= 235.0)
    kick_base_hz = float(hr_freqs[km_hr][int(np.argmax(mag_hi_res[km_hr]))])
    snare_base_hz = float(hr_freqs[bm_hr][int(np.argmax(mag_hi_res[bm_hr]))])

    kick_rms = np.sqrt(np.sum(mag[kick_mask] ** 2, axis=0)) * rms_scale
    body_rms = np.sqrt(np.sum(mag[body_mask] ** 2, axis=0)) * rms_scale
    high_rms = np.sqrt(np.sum(mag[high_mask] ** 2, axis=0)) * rms_scale

    # 2. Mel-Band Percussive NMF on MPS (230 Hz .. 11700 Hz)
    mel_fb = librosa.filters.mel(sr=dsr, n_fft=n_fft, n_mels=n_bands, fmin=220.0, fmax=11700.0).astype(np.float32)
    mel_fb_norm = mel_fb / (np.sum(mel_fb, axis=1, keepdims=True) + 1e-8)
    band_centers = librosa.mel_frequencies(n_mels=n_bands, fmin=220.0, fmax=11700.0).astype(np.float32)
    M_mel = mel_fb_norm @ mag  # (n_bands, n_frames)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(108)
    M_t = torch.tensor(M_mel, dtype=torch.float32, device=device)
    W_t = torch.rand((n_bands, n_comp), dtype=torch.float32, device=device) + 0.02
    H_t = torch.rand((n_comp, n_frames), dtype=torch.float32, device=device) + 0.02

    for _ in range(120):
        WH = W_t @ H_t + 1e-8
        H_t = H_t * ((W_t.T @ M_t) / (W_t.T @ WH + 0.0015))
        WH = W_t @ H_t + 1e-8
        W_t = W_t * ((M_t @ H_t.T) / (WH @ H_t.T + 1e-8))
        norms = torch.amax(W_t, dim=0, keepdim=True) + 1e-8
        W_t = W_t / norms
        H_t = H_t * norms.T

    W_np = W_t.cpu().numpy().astype(np.float32)  # (n_bands, n_comp)
    H_np = H_t.cpu().numpy().astype(np.float32)  # (n_comp, n_frames)

    h_max = np.max(H_np, axis=1).astype(np.float32) + 1e-8
    H_q = np.clip(np.round(np.sqrt(H_np / h_max[:, None]) * q_levels), 0, q_levels).astype(np.uint8)

    # 3. 98.00 BPM Bar-Loop Groove Clustering + Residual Coding
    fps = dsr / hop
    onset_curve = np.sum(np.maximum(0.0, np.diff(M_mel, axis=1, prepend=0.0)), axis=0)
    bar_frames = _detect_bar_frames(onset_curve, fps)
    n_bars = int(np.ceil(n_frames / bar_frames))
    padded_len = n_bars * bar_frames
    H_pad = np.zeros((n_comp, padded_len), dtype=np.uint8)
    H_pad[:, :n_frames] = H_q

    bars_flat = H_pad.reshape(n_comp, n_bars, bar_frames).transpose(1, 0, 2).reshape(n_bars, -1).astype(np.float32)
    rng = np.random.default_rng(303)
    proto_c, bar_sched = kmeans2(bars_flat, n_protos, minit="++", iter=35, seed=rng)
    proto_grooves = np.clip(np.round(proto_c), 0, q_levels).astype(np.uint8).reshape(n_protos, n_comp, bar_frames)
    bar_schedule = bar_sched.astype(np.uint8)

    H_pred_pad = proto_grooves[bar_schedule].transpose(1, 0, 2).reshape(n_comp, padded_len)
    H_pred = H_pred_pad[:, :n_frames]
    d_res_h = ((H_q.astype(np.int16) - H_pred.astype(np.int16)) & 0xFF).astype(np.uint8)

    # Quantize modal & high-band RMS curves
    def _quant_rms(arr: np.ndarray) -> np.ndarray:
        q = np.zeros(len(arr), dtype=np.uint8)
        act = arr > (float(np.max(arr) + 1e-9) * 0.005)
        q[act] = np.clip(np.round((20.0 * np.log10(arr[act] + 1e-7) + 75.0) * 3.2), 1, 255).astype(np.uint8)
        return q

    kick_q = _quant_rms(kick_rms)
    body_q = _quant_rms(body_rms)
    high_q = _quant_rms(high_rms)

    d_kick = np.diff(kick_q.astype(np.int16), prepend=0).astype(np.uint8)
    d_body = np.diff(body_q.astype(np.int16), prepend=0).astype(np.uint8)
    d_high = np.diff(high_q.astype(np.int16), prepend=0).astype(np.uint8)

    header = struct.pack(
        "<4sIBBBBHHff",
        b"WVD5",
        n_frames,
        n_comp,
        n_bands,
        n_protos,
        q_levels,
        bar_frames,
        n_bars,
        kick_base_hz,
        snare_base_hz,
    )
    raw_wvd5 = b"".join(
        [
            header,
            band_centers.astype("<f4").tobytes(),
            W_np.astype("<f4").tobytes(),
            h_max.astype("<f4").tobytes(),
            proto_grooves.tobytes(),
            bar_schedule.tobytes(),
            d_res_h.tobytes(),
            d_kick.tobytes(),
            d_body.tobytes(),
            d_high.tobytes(),
        ]
    )

    drum_model = {
        "sampleRate": dsr,
        "hopSamples": hop,
        "nComponents": n_comp,
        "nBands": n_bands,
        "barFrames": bar_frames,
        "estimatedBpm": round(240.0 * fps / bar_frames, 2),
        "kickBaseHz": round(kick_base_hz, 2),
        "snareBaseHz": round(snare_base_hz, 2),
    }

    synth_24k = synthesize_drums_from_wvd5(raw_wvd5, drum_model, len(drums))
    synth_48k = resample_audio(synth_24k, dsr, orig_sr)[: len(drums_48k)]
    if len(synth_48k) < len(drums_48k):
        synth_48k = np.pad(synth_48k, (0, len(drums_48k) - len(synth_48k)))
    return raw_wvd5, drum_model, synth_48k


def synthesize_drums_from_wvd5(raw: bytes, model: dict, out_samples: int) -> np.ndarray:
    """Reference modal + crisp percussive grain wave synthesizer matching `src/dsp/drum-synth.js`."""
    magic, n_frames, n_comp, n_bands, n_protos, q_levels, bar_frames, n_bars, kick_hz, snare_hz = struct.unpack_from(
        "<4sIBBBBHHff", raw, 0
    )
    assert magic == b"WVD5"
    off = 24
    band_centers = np.frombuffer(raw, dtype="<f4", count=n_bands, offset=off); off += n_bands * 4
    W_np = np.frombuffer(raw, dtype="<f4", count=n_bands * n_comp, offset=off).reshape(n_bands, n_comp); off += n_bands * n_comp * 4
    h_max = np.frombuffer(raw, dtype="<f4", count=n_comp, offset=off); off += n_comp * 4
    proto_grooves = np.frombuffer(raw, dtype=np.uint8, count=n_protos * n_comp * bar_frames, offset=off).reshape(
        n_protos, n_comp, bar_frames
    )
    off += n_protos * n_comp * bar_frames
    bar_schedule = np.frombuffer(raw, dtype=np.uint8, count=n_bars, offset=off); off += n_bars
    d_res_h = np.frombuffer(raw, dtype=np.uint8, count=n_comp * n_frames, offset=off).reshape(n_comp, n_frames)
    off += n_comp * n_frames

    d_kick = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    d_body = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    d_high = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off)

    kick_q = (np.cumsum(d_kick.astype(np.int32)) & 0xFF).astype(np.uint8)
    body_q = (np.cumsum(d_body.astype(np.int32)) & 0xFF).astype(np.uint8)
    high_q = (np.cumsum(d_high.astype(np.int32)) & 0xFF).astype(np.uint8)

    H_pred_pad = proto_grooves[bar_schedule].transpose(1, 0, 2).reshape(n_comp, n_bars * bar_frames)
    H_q = ((H_pred_pad[:, :n_frames].astype(np.int32) + d_res_h.astype(np.int32)) & 0xFF).astype(np.uint8)
    H_rec = ((H_q.astype(np.float32) / float(q_levels)) ** 2) * h_max[:, None]
    mel_hat = W_np @ H_rec  # (n_bands, n_frames)

    def _deq_rms(q: np.ndarray) -> np.ndarray:
        r = np.zeros(len(q), dtype=np.float32)
        act = q > 0
        r[act] = 10.0 ** (((q[act].astype(np.float32) / 3.2) - 75.0) / 20.0)
        return r

    kick_rms = _deq_rms(kick_q)
    body_rms = _deq_rms(body_q)
    high_rms = _deq_rms(high_q)

    sr = int(model["sampleRate"])
    hop = int(model["hopSamples"])
    n_ola = 512
    win_ola = np.hanning(n_ola).astype(np.float32)
    ola_freqs = np.linspace(0.0, sr / 2.0, n_ola // 2 + 1, dtype=np.float32)
    ola_noise_norm = float(1.0 / np.sqrt((0.375 * n_ola) / hop))

    out = np.zeros(out_samples, dtype=np.float32)
    kick_phase = 0.0
    body_phase = 0.0
    rng = np.random.default_rng(404)

    for t in range(n_frames - 1):
        s0_i = t * hop
        s1_i = min(out_samples, s0_i + hop)
        L = s1_i - s0_i
        if L <= 0:
            break
        alpha = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float32)

        # 1. Swept-Sine Kick Modal Wave (pitch boost on transient onset!)
        k0, k1 = kick_rms[t] * 1.4142, kick_rms[t + 1] * 1.4142
        if k0 > 1e-5 or k1 > 1e-5:
            onset_boost = max(0.0, (k1 - k0) / (max(k1, 0.02)))
            f_k = kick_hz + 72.0 * onset_boost * (1.0 - alpha)
            ph_k = kick_phase + np.cumsum(2.0 * np.pi * f_k / sr)
            kick_phase = float(ph_k[-1] % (2.0 * np.pi))
            out[s0_i:s1_i] += (k0 + (k1 - k0) * alpha) * np.sin(ph_k)

        # 2. Snare / Woodblock Body Modal Wave
        b0, b1 = body_rms[t] * 1.4142, body_rms[t + 1] * 1.4142
        if b0 > 1e-5 or b1 > 1e-5:
            f_b = snare_hz + 25.0 * max(0.0, (b1 - b0) / max(b1, 0.02)) * (1.0 - alpha)
            ph_b = body_phase + np.cumsum(2.0 * np.pi * f_b / sr)
            body_phase = float(ph_b[-1] % (2.0 * np.pi))
            out[s0_i:s1_i] += (b0 + (b1 - b0) * alpha) * np.sin(ph_b)

        # 3. Crisp 512-pt Percussive Transient Wave (> 220 Hz)
        c_start = s0_i + hop // 2 - n_ola // 2
        if 0 <= c_start and c_start + n_ola <= out_samples and high_rms[t] > 1e-5:
            spec_mag = np.interp(ola_freqs, band_centers, mel_hat[:, t], left=0.0, right=mel_hat[-1, t] * 0.5)
            spec_mag[ola_freqs < 210.0] = 0.0
            irfft_rms = np.sqrt((2.0 / (n_ola * n_ola)) * float(np.sum(spec_mag[1:-1] ** 2)) + 1e-12)
            cal_mag = spec_mag * (high_rms[t] / irfft_rms)
            rph = rng.uniform(0.0, 2.0 * np.pi, size=len(ola_freqs)).astype(np.float32)
            rph[0] = rph[-1] = 0.0
            grain = np.fft.irfft(cal_mag * np.exp(1j * rph), n=n_ola).astype(np.float32) * win_ola * ola_noise_norm
            out[c_start : c_start + n_ola] += grain

    return out
