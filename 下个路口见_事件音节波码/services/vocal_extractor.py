"""Demucs v4 MPS 3-Stream Pure Lead / True Stereo Backing / Diffuse Space Vocal Extractor (V8.4)."""
from __future__ import annotations

import librosa
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import torch
from demucs.api import Separator

from .audio_io import resample_audio


def extract_clean_vocal_anatomy(master_stereo: np.ndarray, sr: int = 48000, hop: int = 256) -> dict:
    """Extracts 48kHz stereo vocal and cleanly separates Monophonic Lead, Stereo Backing Harmony, and Studio Space."""
    n_samples = len(master_stereo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sep = Separator(model="htdemucs", device=device, shifts=1, overlap=0.25, split=True)
    model_sr = int(getattr(sep, "samplerate", 44100))
    with torch.inference_mode():
        _, raw_stems = sep.separate_tensor(torch.from_numpy(master_stereo.T.copy()), sr=sr)

    vocals_raw = raw_stems["vocals"].cpu().numpy().T
    vocals = resample_audio(vocals_raw, model_sr, sr)
    if len(vocals) >= n_samples:
        vocals = vocals[:n_samples].astype(np.float32)
    else:
        vocals = np.pad(vocals, ((0, n_samples - len(vocals)), (0, 0))).astype(np.float32)

    del sep, raw_stems
    if device == "mps":
        torch.mps.empty_cache()

    # 1. Zero-phase 80Hz 4th-order Butterworth high-pass removes sub-bass rumble with zero phase distortion
    sos_hp = signal.butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    vocals = signal.sosfiltfilt(sos_hp, vocals, axis=0).astype(np.float32)

    # 2. Pure Time-Domain Studio Vocal Phrase Gate (-240 dBFS in instrumental gaps, 1.000 on active phrases)
    mono_sq = 0.5 * (vocals[:, 0] ** 2 + vocals[:, 1] ** 2)
    env_rms = np.sqrt(ndimage.uniform_filter1d(mono_sq, size=2400) + 1e-14)
    peak_v = float(np.max(env_rms) + 1e-9)
    gate_raw = np.clip((env_rms - peak_v * 0.0040) / (peak_v * 0.0062), 0.0, 1.0).astype(np.float32)
    gate_env = ndimage.gaussian_filter1d(gate_raw, sigma=840).astype(np.float32)
    gate_env[gate_env < 1e-4] = 0.0
    vocals = (vocals * gate_env[:, None]).astype(np.float32)

    L = vocals[:, 0]
    R = vocals[:, 1]
    mid = (0.5 * (L + R)).astype(np.float32)
    side = (0.5 * (L - R)).astype(np.float32)

    M_stft = librosa.stft(mid, n_fft=2048, hop_length=hop, window="hann")
    S_stft = librosa.stft(side, n_fft=2048, hop_length=hop, window="hann")
    mag_M = np.abs(M_stft)
    mag_S = np.abs(S_stft)

    # 3. Energy-Conserving Wiener Center-Lead Separator (preserves full Lead vocal power while rejecting Side backing!)
    pow_M = ndimage.gaussian_filter(mag_M ** 2, sigma=(3.5, 2.5))
    pow_S = ndimage.gaussian_filter(mag_S ** 2, sigma=(3.5, 2.5))
    w_lead_pow = np.clip((pow_M - 0.85 * pow_S) / (pow_M + 1e-14), 0.12, 1.0)
    lead_mask = np.sqrt(w_lead_pow).astype(np.float32)

    lead_direct = librosa.istft(M_stft * lead_mask, hop_length=hop, length=n_samples).astype(np.float32)
    lead_direct *= gate_env

    # 4. Separate True Stereo Chorus Backing / Double-Tracking (`backing_stereo`) vs Diffuse Reverb (`reverb_residual`)
    rem_L = (L - lead_direct).astype(np.float32)
    rem_R = (R - lead_direct).astype(np.float32)
    rem_L_stft = librosa.stft(rem_L, n_fft=2048, hop_length=512, window="hann")
    rem_R_stft = librosa.stft(rem_R, n_fft=2048, hop_length=512, window="hann")
    rem_mag = np.sqrt(0.5 * (np.abs(rem_L_stft) ** 2 + np.abs(rem_R_stft) ** 2))

    # Fast separable 1D minimum + Gaussian filter estimates stationary diffuse reverb tail in <0.05s
    rev_min = ndimage.minimum_filter1d(rem_mag, size=15, axis=1)
    rev_floor = ndimage.gaussian_filter(rev_min * 1.45, sigma=(1.6, 3.0))
    rev_mask = np.clip(rev_floor / (rem_mag + 1e-9), 0.0, 0.90).astype(np.float32)

    raw_back_L = librosa.istft(rem_L_stft * (1.0 - rev_mask), hop_length=512, length=n_samples).astype(np.float32)
    raw_back_R = librosa.istft(rem_R_stft * (1.0 - rev_mask), hop_length=512, length=n_samples).astype(np.float32)

    b_rms_sample = np.sqrt(ndimage.uniform_filter1d(0.5 * (raw_back_L ** 2 + raw_back_R ** 2), size=4800) + 1e-14)
    b_peak = float(np.max(b_rms_sample) + 1e-9)
    b_gate = ndimage.gaussian_filter1d(
        np.clip((b_rms_sample - b_peak * 0.045) / (b_peak * 0.055), 0.0, 1.0).astype(np.float32),
        sigma=1200,
    ).astype(np.float32)
    b_gate[b_gate < 1e-3] = 0.0

    back_L = (raw_back_L * b_gate).astype(np.float32)
    back_R = (raw_back_R * b_gate).astype(np.float32)
    space_L = (rem_L - back_L).astype(np.float32)
    space_R = (rem_R - back_R).astype(np.float32)

    # Frame-aligned (`hop=256`) backing L/R RMS & subtle lead pan
    n_frames = n_samples // hop + 1
    pad_b = 512
    bl_pad = np.pad(back_L, (pad_b, pad_b + hop))
    br_pad = np.pad(back_R, (pad_b, pad_b + hop))
    bl_fr = np.lib.stride_tricks.as_strided(bl_pad, shape=(n_frames, 1024), strides=(bl_pad.strides[0] * hop, bl_pad.strides[0]))
    br_fr = np.lib.stride_tricks.as_strided(br_pad, shape=(n_frames, 1024), strides=(br_pad.strides[0] * hop, br_pad.strides[0]))
    back_rms_l = np.sqrt(np.mean(bl_fr ** 2, axis=1) + 1e-16).astype(np.float32)
    back_rms_r = np.sqrt(np.mean(br_fr ** 2, axis=1) + 1e-16).astype(np.float32)

    core_bins = slice(8, 220)
    m_core_rms = np.sqrt(np.mean(mag_M[core_bins] ** 2, axis=0) + 1e-12)
    m_ref = float(np.percentile(m_core_rms[m_core_rms > 1e-4], 90.0)) if np.any(m_core_rms > 1e-4) else 1.0
    presence_weight = np.clip((m_core_rms - m_ref * 0.18) / (m_ref * 0.30), 0.0, 1.0)

    L_stft_mag = np.abs(M_stft[core_bins] + S_stft[core_bins])
    R_stft_mag = np.abs(M_stft[core_bins] - S_stft[core_bins])
    l_e = np.sqrt(np.mean(L_stft_mag ** 2, axis=0) + 1e-12)
    r_e = np.sqrt(np.mean(R_stft_mag ** 2, axis=0) + 1e-12)
    # Subtle center lead pan capped to [-0.18, +0.18] because true stereo width now lives cleanly in Track 07!
    pan_raw = np.clip((r_e - l_e) / (r_e + l_e + 1e-6), -0.18, 0.18) * presence_weight
    pan_frames = ndimage.gaussian_filter1d(pan_raw.astype(np.float32), sigma=4.0).astype(np.float32)

    return {
        "vocals_stereo": vocals,
        "lead_direct": lead_direct,
        "backing_stereo": np.column_stack([back_L, back_R]).astype(np.float32),
        "reverb_residual": np.column_stack([space_L, space_R]).astype(np.float32),
        "back_rms_l": back_rms_l,
        "back_rms_r": back_rms_r,
        "pan_frames": pan_frames,
    }
