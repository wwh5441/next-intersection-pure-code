"""Demucs v4 (`htdemucs`, shifts=1 on MPS) Pure-Vocal Extractor + 5-Track Anatomy Splitter + Speaker Timbre Analyzer."""
from __future__ import annotations

import librosa
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import torch
from demucs.api import Separator

from .audio_io import resample_audio


def extract_and_split_vocal_5tracks(master_stereo: np.ndarray, sr: int = 48000) -> dict:
    """Extracts pure 48kHz stereo human vocal and decomposes it into 5 exact-sum vocal sub-tracks."""
    n_samples = len(master_stereo)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    wav_tensor = torch.from_numpy(master_stereo.T.copy())

    separator = Separator(model="htdemucs", device=device, shifts=1, overlap=0.25, split=True)
    model_sr = int(getattr(separator, "samplerate", 44100))
    with torch.inference_mode():
        _, raw_stems = separator.separate_tensor(wav_tensor, sr=sr)

    vocals_raw = raw_stems["vocals"].cpu().numpy().T
    vocals = resample_audio(vocals_raw, model_sr, sr)
    if len(vocals) >= n_samples:
        vocals = vocals[:n_samples].astype(np.float32)
    else:
        vocals = np.pad(vocals, ((0, n_samples - len(vocals)), (0, 0))).astype(np.float32)

    del separator, raw_stems
    if device == "mps":
        torch.mps.empty_cache()

    # Smooth studio vocal gate so instrumental-only sections (intro/interlude/outro) are 100% silent black
    mono_sq = 0.5 * (vocals[:, 0] ** 2 + vocals[:, 1] ** 2)
    env_rms = np.sqrt(ndimage.uniform_filter1d(mono_sq, size=4800) + 1e-12)
    peak_v = float(np.max(env_rms) + 1e-9)
    gate_env = np.clip((env_rms - peak_v * 0.0032) / (peak_v * 0.0058), 0.0, 1.0).astype(np.float32)
    gate_env = ndimage.gaussian_filter1d(gate_env, sigma=1400).astype(np.float32)
    vocals = (vocals * gate_env[:, None]).astype(np.float32)

    # 1. Center Coherence Wiener Mask -> Dry Center Lead Vocal vs Stereo Residual
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

    # 2. Lead Dry Anatomy Split:
    #    - 01 lead_core: Harmonic Chest & Fundamental (80 - 1,350 Hz)
    #    - 02 lead_formant: Harmonic Vowel Articulation & Formants (1,350 - 11,500 Hz)
    #    - 03 lead_consonant: Mandarin Consonants, Plosives, Breath & >11.5kHz Studio Air (exact conservation!)
    lead_stft = librosa.stft(lead_dry, n_fft=2048, hop_length=256, window="hann")
    lead_H_stft, _ = librosa.decompose.hpss(lead_stft, kernel_size=(21, 9), power=2.0, margin=1.0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    w_low = 1.0 / (1.0 + (np.maximum(freqs, 1.0) / 1350.0) ** 6)
    w_air = 1.0 / (1.0 + (7800.0 / np.maximum(freqs, 1.0)) ** 8)
    w_mid = np.maximum(0.0, (1.0 - w_low) * (1.0 - w_air))

    lead_core = librosa.istft(lead_H_stft * w_low[:, None], hop_length=256, length=n_samples).astype(np.float32)
    lead_formant = librosa.istft(lead_H_stft * w_mid[:, None], hop_length=256, length=n_samples).astype(np.float32)
    lead_consonant = (lead_dry - lead_core - lead_formant).astype(np.float32)

    # 3. Stereo Residual Split:
    #    - 04 backing_harmony: Stereo Double-Tracking & Chorus Harmonies (phrase-gated to 99.8% energy)
    #    - 05 vocal_space: Diffuse Studio Reverb Tail & Stereo Ambience (exact conservation!)
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

    # Smooth phrase gate on backing_harmony so low-level solo-verse ambience flows cleanly into vocal_space
    b_rms = np.sqrt(ndimage.uniform_filter1d(0.5 * (raw_back_L ** 2 + raw_back_R ** 2), size=4800) + 1e-12)
    b_peak = float(np.max(b_rms) + 1e-9)
    b_gate = np.clip((b_rms - b_peak * 0.035) / (b_peak * 0.045), 0.0, 1.0).astype(np.float32)
    b_gate = ndimage.gaussian_filter1d(b_gate, sigma=1200).astype(np.float32)

    back_L = (raw_back_L * b_gate).astype(np.float32)
    back_R = (raw_back_R * b_gate).astype(np.float32)
    space_L = (rem_L - back_L).astype(np.float32)
    space_R = (rem_R - back_R).astype(np.float32)

    backing_stereo = np.stack([back_L, back_R], axis=1)
    space_stereo = np.stack([space_L, space_R], axis=1)

    recon_L = lead_core + lead_formant + lead_consonant + back_L + space_L
    recon_R = lead_core + lead_formant + lead_consonant + back_R + space_R
    max_err = float(np.max(np.abs(np.stack([recon_L, recon_R], axis=1) - vocals)))

    # 4. Extract 64-Mel Speaker Timbre x Pronunciation Disentanglement Curves (70 Hz - 18.5 kHz) for Inspector
    fb64 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=70.0, fmax=18500.0).astype(np.float32)
    mel_centers = librosa.mel_frequencies(n_mels=64, fmin=70.0, fmax=18500.0).astype(np.float32)
    harm_mag = np.abs(lead_H_stft).astype(np.float32)
    cons_mag = np.abs(librosa.stft(lead_consonant, n_fft=2048, hop_length=256, window="hann")).astype(np.float32)

    mel_harm = fb64 @ harm_mag
    mel_cons = fb64 @ cons_mag
    f_rms = np.sqrt(np.mean(mel_harm ** 2, axis=0))
    act = f_rms > (float(np.max(f_rms)) * 0.025)

    log_mh = np.log(np.maximum(mel_harm[:, act], 1e-6))
    centered_h = log_mh - np.mean(log_mh, axis=0, keepdims=True)
    static_timbre = np.mean(centered_h, axis=1)
    pron_variance = np.std(centered_h - static_timbre[:, None], axis=1)
    cons_profile = np.mean(np.log(np.maximum(mel_cons[:, act], 1e-6)), axis=1) - float(
        np.mean(np.log(np.maximum(mel_cons[:, act], 1e-6)))
    )

    # Active vocal timing ratio
    first_active_sec = round(float(np.argmax(gate_env > 0.05)) / sr, 2)

    return {
        "device": device,
        "vocalsStereo": vocals,
        "sumMaxError": max_err,
        "firstVocalSec": first_active_sec,
        "tracks": {
            "lead_core": lead_core,
            "lead_formant": lead_formant,
            "lead_consonant": lead_consonant,
            "backing_harmony": backing_stereo,
            "vocal_space": space_stereo,
        },
        "streams": {
            "lead_core": lead_core,
            "lead_formant": lead_formant,
            "lead_consonant": lead_consonant,
            "backing_mid": (0.5 * (back_L + back_R)).astype(np.float32),
            "backing_side": (0.5 * (back_L - back_R)).astype(np.float32),
            "space_mid": (0.5 * (space_L + space_R)).astype(np.float32),
            "space_side": (0.5 * (space_L - space_R)).astype(np.float32),
        },
        "timbreProfile": {
            "melCentersHz": [round(float(x), 1) for x in mel_centers],
            "staticTimbreLogMel": [round(float(x), 4) for x in static_timbre],
            "pronunciationStdLogMel": [round(float(x), 4) for x in pron_variance],
            "consonantAirLogMel": [round(float(x), 4) for x in cons_profile],
        },
    }
