"""Demucs v4 (`htdemucs`, shifts=1 on MPS) Full-Song 8-Track Splitter (5 Vocal Sub-Tracks + 3 Accompaniment Stems)."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import torch
from demucs.api import Separator

from .audio_io import resample_audio


def _fit_len(arr: np.ndarray, n_samples: int) -> np.ndarray:
    if len(arr) >= n_samples:
        return arr[:n_samples].astype(np.float32)
    pad_spec = ((0, n_samples - len(arr)), (0, 0)) if arr.ndim == 2 else (0, n_samples - len(arr))
    return np.pad(arr, pad_spec).astype(np.float32)


def extract_accompaniment_3tracks(master_stereo: np.ndarray, sr: int = 48000) -> dict:
    """Extracts gated vocal + Bass + Drums + exact-conservation Stereo Accompaniment (`master - vocal - bass - drums`)."""
    n_samples = len(master_stereo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wav_tensor = torch.from_numpy(master_stereo.T.copy())

    separator = Separator(model="htdemucs", device=device, shifts=1, overlap=0.25, split=True)
    model_sr = int(getattr(separator, "samplerate", 44100))
    with torch.inference_mode():
        _, raw_stems = separator.separate_tensor(wav_tensor, sr=sr)

    vocals_raw = _fit_len(resample_audio(raw_stems["vocals"].cpu().numpy().T, model_sr, sr), n_samples)
    bass_st = _fit_len(resample_audio(raw_stems["bass"].cpu().numpy().T, model_sr, sr), n_samples)
    drums_st = _fit_len(resample_audio(raw_stems["drums"].cpu().numpy().T, model_sr, sr), n_samples)

    del separator, raw_stems
    if device == "mps":
        torch.mps.empty_cache()

    # Exact same vocal silence gate as 5-track vocal extractor so any gated intro/interlude residual returns to `accompaniment`
    mono_sq = 0.5 * (vocals_raw[:, 0] ** 2 + vocals_raw[:, 1] ** 2)
    env_rms = np.sqrt(ndimage.uniform_filter1d(mono_sq, size=4800) + 1e-12)
    peak_v = float(np.max(env_rms) + 1e-9)
    gate_env = np.clip((env_rms - peak_v * 0.0032) / (peak_v * 0.0058), 0.0, 1.0).astype(np.float32)
    gate_env = ndimage.gaussian_filter1d(gate_env, sigma=1400).astype(np.float32)
    vocals_gated = (vocals_raw * gate_env[:, None]).astype(np.float32)

    bass_mono = (0.5 * (bass_st[:, 0] + bass_st[:, 1])).astype(np.float32)
    drums_mono = (0.5 * (drums_st[:, 0] + drums_st[:, 1])).astype(np.float32)
    bass_center = np.stack([bass_mono, bass_mono], axis=1)
    drums_center = np.stack([drums_mono, drums_mono], axis=1)

    # Exact full-mix conservation: any stereo side component of drums/bass + guitar/synth/pads stays in `accompaniment`
    accompaniment_st = (master_stereo - vocals_gated - bass_center - drums_center).astype(np.float32)

    max_err = float(
        np.max(np.abs((vocals_gated + bass_center + drums_center + accompaniment_st) - master_stereo))
    )

    return {
        "device": device,
        "vocalsGated": vocals_gated,
        "sumMaxError": max_err,
        "tracks": {
            "bass": bass_mono,
            "drums": drums_mono,
            "accompaniment": accompaniment_st,
        },
        "streams": {
            "bass": bass_mono,
            "drums_body": drums_mono,
            "accomp_mid": (0.5 * (accompaniment_st[:, 0] + accompaniment_st[:, 1])).astype(np.float32),
            "accomp_side": (0.5 * (accompaniment_st[:, 0] - accompaniment_st[:, 1])).astype(np.float32),
        },
    }
