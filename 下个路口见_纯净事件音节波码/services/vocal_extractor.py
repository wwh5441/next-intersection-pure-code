"""Demucs v4 MPS Cross-Stem De-Bleed Vocal Purifier & Direct/Stereo/Space Decomposer."""
from __future__ import annotations

import librosa
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import torch
from demucs.api import Separator

from .audio_io import resample_audio


def extract_clean_vocal_anatomy(master_stereo: np.ndarray, sr: int = 48000, hop: int = 256) -> dict:
    """Extracts 48kHz vocal and purges accompaniment cross-bleed (drums/bass/guitar) using reference stems."""
    n_samples = len(master_stereo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sep = Separator(model="htdemucs", device=device, shifts=1, overlap=0.25, split=True)
    model_sr = int(getattr(sep, "samplerate", 44100))
    with torch.inference_mode():
        _, raw_stems = sep.separate_tensor(torch.from_numpy(master_stereo.T.copy()), sr=sr)

    vocals_raw = raw_stems["vocals"].cpu().numpy().T
    acc_raw = (raw_stems["drums"] + raw_stems["bass"] + raw_stems["other"]).cpu().numpy().T
    del sep, raw_stems
    if device == "mps":
        torch.mps.empty_cache()

    vocals_unfiltered = resample_audio(vocals_raw, model_sr, sr)
    acc_stereo = resample_audio(acc_raw, model_sr, sr)
    del vocals_raw, acc_raw

    def _fit_len(arr: np.ndarray) -> np.ndarray:
        if len(arr) >= n_samples:
            return arr[:n_samples].astype(np.float32)
        return np.pad(arr, ((0, n_samples - len(arr)), (0, 0))).astype(np.float32)

    vocals_unfiltered = _fit_len(vocals_unfiltered)
    acc_stereo = _fit_len(acc_stereo)

    # 1. Cross-Stem Accompaniment Bleed Suppression in 2048-pt STFT domain
    n_fft = 2048
    VL = librosa.stft(vocals_unfiltered[:, 0], n_fft=n_fft, hop_length=hop, window="hann")
    VR = librosa.stft(vocals_unfiltered[:, 1], n_fft=n_fft, hop_length=hop, window="hann")
    AL = librosa.stft(acc_stereo[:, 0], n_fft=n_fft, hop_length=hop, window="hann")
    AR = librosa.stft(acc_stereo[:, 1], n_fft=n_fft, hop_length=hop, window="hann")
    del acc_stereo

    pv = 0.5 * (np.abs(VL) ** 2 + np.abs(VR) ** 2)
    pa = 0.5 * (np.abs(AL) ** 2 + np.abs(AR) ** 2)
    del AL, AR

    freqs = np.linspace(0.0, sr * 0.5, n_fft // 2 + 1, dtype=np.float32)[:, None]
    # Frequency-dependent accompaniment bleed coupling alpha(f):
    # Strong sub-145Hz bass/kick rejection, gentle vocal-core protection, stronger >4.8kHz hi-hat/cymbal rejection
    alpha_f = np.full_like(freqs, 0.0024)
    alpha_f[freqs < 155.0] = 0.09
    hi_mask = freqs > 4800.0
    alpha_f[hi_mask] = 0.0024 + 0.0048 * np.clip((freqs[hi_mask] - 4800.0) / 8000.0, 0.0, 1.0)

    # Stationary background noise floor estimated from quietest 22% frames per bin
    noise_floor = np.percentile(pv, 22.0, axis=1, keepdims=True).astype(np.float32) * 1.45
    clean_num = np.maximum(0.0, pv - noise_floor)
    wiener_bleed = (clean_num / (pv + alpha_f * pa + noise_floor + 1e-14)).astype(np.float32) ** 0.88
    # Steep sub-105Hz high-pass cut (lowest sung note in song is ~165Hz; below 105Hz is 100% kick/bass rumble)
    hp_sub = np.clip((freqs - 90.0) / 45.0, 0.0, 1.0).astype(np.float32)
    debleed_gain = signal.medfilt2d(wiener_bleed, kernel_size=(3, 3)) * hp_sub

    VL_clean = VL * debleed_gain
    VR_clean = VR * debleed_gain
    vocals_clean = np.column_stack(
        [
            librosa.istft(VL_clean, hop_length=hop, length=n_samples),
            librosa.istft(VR_clean, hop_length=hop, length=n_samples),
        ]
    ).astype(np.float32)

    # Smooth time-domain presence gate to guarantee 0.0 digital silence in pure instrumental bars
    mono_sq = 0.5 * (vocals_clean[:, 0] ** 2 + vocals_clean[:, 1] ** 2)
    env_rms = np.sqrt(ndimage.uniform_filter1d(mono_sq, size=2800) + 1e-14)
    peak_v = float(np.max(env_rms) + 1e-9)
    gate_env = np.clip((env_rms - peak_v * 0.0065) / (peak_v * 0.0095), 0.0, 1.0).astype(np.float32)
    gate_env = ndimage.gaussian_filter1d(gate_env, sigma=720).astype(np.float32)
    vocals_clean = (vocals_clean * gate_env[:, None]).astype(np.float32)

    # 2. Separate Dry Direct Center Voice from Diffuse Stereo Plate/Room Reverb
    M_stft = 0.5 * (VL_clean + VR_clean)
    S_stft = 0.5 * (VL_clean - VR_clean)
    mag_M = np.abs(M_stft)
    mag_S = np.abs(S_stft)

    coh = (mag_M ** 2) / (mag_M ** 2 + 1.90 * (mag_S ** 2) + 1e-14)
    # Floor lowered from V8's 0.12 (-18dB bleed!) to 0.015 (-36.5dB) so stereo reverb tail doesn't smear dry formants
    direct_mask = np.clip(signal.medfilt2d(coh.astype(np.float32), kernel_size=(3, 5)) ** 1.08, 0.015, 1.0)
    lead_direct = librosa.istft(M_stft * direct_mask, hop_length=hop, length=n_samples).astype(np.float32)
    lead_direct = (lead_direct * gate_env).astype(np.float32)

    # 3. High-Confidence Stereo Double-Tracking Width (gated strictly on strong sung vocal core!)
    core_bins = slice(8, 215)  # 187Hz .. 5.0kHz
    m_core_rms = np.sqrt(np.mean(mag_M[core_bins] ** 2, axis=0) + 1e-14)
    s_core_rms = np.sqrt(np.mean(mag_S[core_bins] ** 2, axis=0) + 1e-14)
    peak_m_core = float(np.max(m_core_rms) + 1e-9)
    strong_voice_gate = np.clip((m_core_rms / peak_m_core - 0.08) / 0.12, 0.0, 1.0).astype(np.float32)
    raw_width = np.clip((s_core_rms / (m_core_rms + 1e-7) - 0.18) / 0.56, 0.0, 1.0) * strong_voice_gate
    stereo_width_frames = ndimage.gaussian_filter1d(raw_width.astype(np.float32), sigma=3.0).astype(np.float32)

    # Pan balance per frame (clamped gently around center; gated to 0 on weak frames)
    L_mag = np.abs(M_stft[core_bins] + S_stft[core_bins])
    R_mag = np.abs(M_stft[core_bins] - S_stft[core_bins])
    l_e = np.sqrt(np.mean(L_mag ** 2, axis=0) + 1e-14)
    r_e = np.sqrt(np.mean(R_mag ** 2, axis=0) + 1e-14)
    raw_pan = np.clip((r_e - l_e) / (r_e + l_e + 1e-7), -0.65, 0.65) * strong_voice_gate
    pan_frames = ndimage.gaussian_filter1d(raw_pan.astype(np.float32), sigma=3.0).astype(np.float32)

    direct_stereo = np.column_stack([lead_direct, lead_direct])
    reverb_residual = (vocals_clean - direct_stereo).astype(np.float32)

    return {
        "vocals_raw_stereo": vocals_unfiltered,
        "vocals_stereo": vocals_clean,
        "lead_direct": lead_direct,
        "reverb_residual": reverb_residual,
        "stereo_width_frames": stereo_width_frames,
        "pan_frames": pan_frames,
    }
