"""Full-Song 1-Track Unified Stereo Master Extractor (Zero Demucs Separation Residual)."""
from __future__ import annotations

import librosa
import numpy as np
import torch


def extract_full_song_1track(
    master_stereo: np.ndarray,
    sr: int = 48000,
    precomputed: dict | None = None,
) -> dict:
    """Packs full-band stereo master directly into Mid/Side streams without separation artifacts."""
    n_samples = len(master_stereo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    master_mid = (0.5 * (master_stereo[:, 0] + master_stereo[:, 1])).astype(np.float32)
    master_side = (0.5 * (master_stereo[:, 0] - master_stereo[:, 1])).astype(np.float32)

    # 64-Mel Spectral Curve for Inspector UI
    ref_mono = (
        0.5 * (precomputed["vocals_gated"][:n_samples, 0] + precomputed["vocals_gated"][:n_samples, 1])
        if (precomputed is not None and "vocals_gated" in precomputed)
        else master_mid
    )
    fb64 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=70.0, fmax=18500.0).astype(np.float32)
    mel_centers = librosa.mel_frequencies(n_mels=64, fmin=70.0, fmax=18500.0).astype(np.float32)
    stft_mat = librosa.stft(ref_mono.astype(np.float32), n_fft=2048, hop_length=512, window="hann")
    H_stft, P_stft = librosa.decompose.hpss(stft_mat, kernel_size=(17, 7), power=2.0, margin=1.0)
    mel_harm = fb64 @ np.abs(H_stft).astype(np.float32)
    mel_cons = fb64 @ np.abs(P_stft).astype(np.float32)

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
        "sumMaxError": 0.0,
        "tracks": {
            "master": master_stereo.astype(np.float32),
        },
        "streams": {
            "master_mid": master_mid,
            "master_side": master_side,
        },
        "timbreProfile": {
            "melCentersHz": [round(float(x), 1) for x in mel_centers],
            "staticTimbreLogMel": [round(float(x), 4) for x in static_timbre],
            "pronunciationStdLogMel": [round(float(x), 4) for x in pron_variance],
            "consonantAirLogMel": [round(float(x), 4) for x in cons_profile],
        },
    }
