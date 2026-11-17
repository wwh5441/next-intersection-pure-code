"""In-memory Demucs v4 (`htdemucs`) MPS separation + phase-coherent 4-stem Mid/Side decomposition."""
from __future__ import annotations

import librosa
import numpy as np
import torch
from demucs.api import Separator

from .audio_io import resample_audio


def _fit_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len].astype(np.float32)
    return np.pad(arr, ((0, target_len - len(arr)), (0, 0))).astype(np.float32)


def separate_coherent_stems(original: np.ndarray, sr: int = 48000) -> dict:
    """Separates stereo audio into 4 phase-coherent stems (`vocal`, `drums`, `bass`, `harmony`) in memory."""
    n_samples = len(original)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wav_tensor = torch.from_numpy(original.T.copy())

    separator = Separator(model="htdemucs", device=device, shifts=0, overlap=0.25, split=True)
    model_sr = int(getattr(separator, "samplerate", 44100))
    with torch.inference_mode():
        _, raw_stems = separator.separate_tensor(wav_tensor, sr=sr)

    demucs_stems = {}
    for key in ("vocals", "drums", "bass", "other"):
        arr = raw_stems[key].cpu().numpy().T
        arr = resample_audio(arr, model_sr, sr)
        demucs_stems[key] = _fit_length(arr, n_samples)

    del separator, raw_stems
    if device == "mps":
        torch.mps.empty_cache()

    mid = 0.5 * (original[:, 0] + original[:, 1])
    side = 0.5 * (original[:, 0] - original[:, 1])

    vocal_mono = np.mean(demucs_stems["vocals"], axis=1)
    drums_mono = np.mean(demucs_stems["drums"], axis=1)
    bass_mono = np.mean(demucs_stems["bass"], axis=1)
    other_mono = np.mean(demucs_stems["other"], axis=1)

    orig_spec = librosa.stft(mid, n_fft=2048, hop_length=512)
    s_vocal = librosa.stft(vocal_mono, n_fft=2048, hop_length=512)
    s_drums = librosa.stft(drums_mono, n_fft=2048, hop_length=512)
    s_bass = librosa.stft(bass_mono, n_fft=2048, hop_length=512)
    s_other = librosa.stft(other_mono, n_fft=2048, hop_length=512)
    s_resid = orig_spec - (s_vocal + s_drums + s_bass + s_other)

    weights = [
        np.abs(s_vocal) ** 2,
        np.abs(s_drums) ** 2,
        np.abs(s_bass) ** 2,
        np.abs(s_other) ** 2 + np.abs(s_resid) ** 2,
    ]
    denom = np.maximum(sum(weights), 1e-20)
    coherent = [
        librosa.istft(orig_spec * (w / denom), hop_length=512, length=n_samples).astype(np.float32)
        for w in weights
    ]
    vocal_c, drums_c, bass_c, harmony_mid = coherent
    exact_mid_diff = mid - (vocal_c + drums_c + bass_c + harmony_mid)
    harmony_mid = (harmony_mid + exact_mid_diff).astype(np.float32)
    harmony_stereo = np.stack([harmony_mid + side, harmony_mid - side], axis=1).astype(np.float32)

    stems = {
        "vocal": vocal_c,
        "drums": drums_c,
        "bass": bass_c,
        "harmony": harmony_stereo,
    }
    reconstructed = vocal_c[:, None] + drums_c[:, None] + bass_c[:, None] + harmony_stereo
    max_err = float(np.max(np.abs(reconstructed - original)))
    return {
        "stems": stems,
        "device": device,
        "sumMaxError": max_err,
    }
