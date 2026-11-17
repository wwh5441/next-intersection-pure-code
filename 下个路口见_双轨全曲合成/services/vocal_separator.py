"""Demucs v4 (`htdemucs`, shifts=1 on MPS) Full-Song 2-Track Splitter (1 Merged Stereo Vocal + 1 Full-Band Stereo Accompaniment)."""
from __future__ import annotations

import librosa
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import torch
from demucs.api import Separator

from .audio_io import resample_audio


def _fit_len(arr: np.ndarray, n_samples: int) -> np.ndarray:
    if len(arr) >= n_samples:
        return arr[:n_samples].astype(np.float32)
    pad_spec = ((0, n_samples - len(arr)), (0, 0)) if arr.ndim == 2 else (0, n_samples - len(arr))
    return np.pad(arr, pad_spec).astype(np.float32)


def extract_full_song_2tracks(
    master_stereo: np.ndarray,
    sr: int = 48000,
    precomputed: dict | None = None,
) -> dict:
    """Separates 48kHz stereo master into 1 merged stereo vocal (dry + FDN space) + 1 full-band stereo accompaniment."""
    n_samples = len(master_stereo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if precomputed is not None:
        vocals = _fit_len(precomputed["vocals_gated"], n_samples)
        accompaniment_st = _fit_len(precomputed["acc_full"], n_samples)
    else:
        wav_tensor = torch.from_numpy(master_stereo.T.copy())
        separator = Separator(model="htdemucs", device=device, shifts=1, overlap=0.25, split=True)
        model_sr = int(getattr(separator, "samplerate", 44100))
        with torch.inference_mode():
            _, raw_stems = separator.separate_tensor(wav_tensor, sr=sr)

        vocals_raw = _fit_len(resample_audio(raw_stems["vocals"].cpu().numpy().T, model_sr, sr), n_samples)

        del separator, raw_stems
        if device == "mps":
            torch.mps.empty_cache()

        mono_sq = 0.5 * (vocals_raw[:, 0] ** 2 + vocals_raw[:, 1] ** 2)
        env_rms = np.sqrt(ndimage.uniform_filter1d(mono_sq, size=4800) + 1e-12)
        peak_v = float(np.max(env_rms) + 1e-9)
        gate_env = np.clip((env_rms - peak_v * 0.0032) / (peak_v * 0.0058), 0.0, 1.0).astype(np.float32)
        gate_env = ndimage.gaussian_filter1d(gate_env, sigma=1400).astype(np.float32)
        vocals = (vocals_raw * gate_env[:, None]).astype(np.float32)
        accompaniment_st = (master_stereo - vocals).astype(np.float32)

    # Split merged stereo vocal cleanly into Dry (Lead + Stereo Backing) vs Diffuse Stereo Reverb Space
    n_fft, hop = 2048, 512
    L, R = vocals[:, 0], vocals[:, 1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)

    M_stft = librosa.stft(mid, n_fft=n_fft, hop_length=hop, window="hann")
    S_stft = librosa.stft(side, n_fft=n_fft, hop_length=hop, window="hann")
    mag_M = np.abs(M_stft)
    mag_S = np.abs(S_stft)

    center_coh = (mag_M ** 2) / (mag_M ** 2 + 2.2 * (mag_S ** 2) + 1e-12)
    lead_mask = signal.medfilt2d(center_coh.astype(np.float32), kernel_size=(3, 5)) ** 1.15
    lead_dry = librosa.istft(M_stft * lead_mask, hop_length=hop, length=n_samples).astype(np.float32)

    rem_L = (L - lead_dry).astype(np.float32)
    rem_R = (R - lead_dry).astype(np.float32)
    rem_L_stft = librosa.stft(rem_L, n_fft=n_fft, hop_length=hop, window="hann")
    rem_R_stft = librosa.stft(rem_R, n_fft=n_fft, hop_length=hop, window="hann")
    rem_mag = np.sqrt(0.5 * (np.abs(rem_L_stft) ** 2 + np.abs(rem_R_stft) ** 2))

    reverb_floor = ndimage.gaussian_filter(
        ndimage.percentile_filter(rem_mag, percentile=25, size=(3, 19)),
        sigma=(1.0, 2.2),
    )
    rev_mask = np.clip(reverb_floor / (rem_mag + 1e-9), 0.0, 0.90)
    raw_back_L = librosa.istft(rem_L_stft * (1.0 - rev_mask), hop_length=hop, length=n_samples).astype(np.float32)
    raw_back_R = librosa.istft(rem_R_stft * (1.0 - rev_mask), hop_length=hop, length=n_samples).astype(np.float32)

    b_rms = np.sqrt(ndimage.uniform_filter1d(0.5 * (raw_back_L ** 2 + raw_back_R ** 2), size=4800) + 1e-12)
    b_peak = float(np.max(b_rms) + 1e-9)
    b_gate = np.clip((b_rms - b_peak * 0.035) / (b_peak * 0.045), 0.0, 1.0).astype(np.float32)
    b_gate = ndimage.gaussian_filter1d(b_gate, sigma=1200).astype(np.float32)

    back_L = (raw_back_L * b_gate).astype(np.float32)
    back_R = (raw_back_R * b_gate).astype(np.float32)
    space_L = (rem_L - back_L).astype(np.float32)
    space_R = (rem_R - back_R).astype(np.float32)

    dry_mid = (lead_dry + 0.5 * (back_L + back_R)).astype(np.float32)
    dry_side = (0.5 * (back_L - back_R)).astype(np.float32)
    space_mid = (0.5 * (space_L + space_R)).astype(np.float32)
    space_side = (0.5 * (space_L - space_R)).astype(np.float32)

    max_err = float(np.max(np.abs((vocals + accompaniment_st) - master_stereo)))

    # 64-Mel Speaker Timbre x Pronunciation Disentanglement Curves (70 Hz - 18.5 kHz) for Inspector
    fb64 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=70.0, fmax=18500.0).astype(np.float32)
    mel_centers = librosa.mel_frequencies(n_mels=64, fmin=70.0, fmax=18500.0).astype(np.float32)
    lead_stft = librosa.stft(lead_dry, n_fft=2048, hop_length=512, window="hann")
    lead_H_stft, lead_P_stft = librosa.decompose.hpss(lead_stft, kernel_size=(17, 7), power=2.0, margin=1.0)
    mel_harm = fb64 @ np.abs(lead_H_stft).astype(np.float32)
    mel_cons = fb64 @ np.abs(lead_P_stft).astype(np.float32)

    f_rms = np.sqrt(np.mean(mel_harm ** 2, axis=0))
    act = f_rms > (float(np.max(f_rms)) * 0.025)
    log_mh = np.log(np.maximum(mel_harm[:, act], 1e-6))
    centered_h = log_mh - np.mean(log_mh, axis=0, keepdims=True)
    static_timbre = np.mean(centered_h, axis=1)
    pron_variance = np.std(centered_h - static_timbre[:, None], axis=1)
    cons_profile = np.mean(np.log(np.maximum(mel_cons[:, act], 1e-6)), axis=1) - float(
        np.mean(np.log(np.maximum(mel_cons[:, act], 1e-6)))
    )

    return {
        "device": device,
        "sumMaxError": max_err,
        "tracks": {
            "vocal": vocals,
            "accompaniment": accompaniment_st,
        },
        "streams": {
            "vocal_mid": dry_mid,
            "vocal_side": dry_side,
            "space_mid": space_mid,
            "space_side": space_side,
            "accomp_mid": (0.5 * (accompaniment_st[:, 0] + accompaniment_st[:, 1])).astype(np.float32),
            "accomp_side": (0.5 * (accompaniment_st[:, 0] - accompaniment_st[:, 1])).astype(np.float32),
        },
        "timbreProfile": {
            "melCentersHz": [round(float(x), 1) for x in mel_centers],
            "staticTimbreLogMel": [round(float(x), 4) for x in static_timbre],
            "pronunciationStdLogMel": [round(float(x), 4) for x in pron_variance],
            "consonantAirLogMel": [round(float(x), 4) for x in cons_profile],
        },
    }
