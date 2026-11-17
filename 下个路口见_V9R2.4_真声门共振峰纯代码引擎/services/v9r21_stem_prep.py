"""V9R2.1 analysis-stem preparation from the HQ separation (`services/v9r21_stem_prep.py`).

Splits the HQ vocal stem into: dry lead (mono, center) / stereo backing / vocal space,
and exposes band stems (bass mono, drums stereo, guitar stereo, keys+rest stereo).
Every split is a complement, so all 11 analysis targets sum exactly to the master.
"""
from __future__ import annotations

from pathlib import Path
import librosa
import numpy as np
import scipy.ndimage as ndimage


def _lead_from_vocals(voc: np.ndarray) -> np.ndarray:
    n = len(voc)
    mid = (0.5 * (voc[:, 0] + voc[:, 1])).astype(np.float32)
    side = (0.5 * (voc[:, 0] - voc[:, 1])).astype(np.float32)
    Vm = librosa.stft(mid, n_fft=2048, hop_length=256, window="hann")
    Vs = librosa.stft(side, n_fft=2048, hop_length=256, window="hann")
    pm = ndimage.gaussian_filter(np.abs(Vm) ** 2, sigma=(3.0, 2.0))
    ps = ndimage.gaussian_filter(np.abs(Vs) ** 2, sigma=(3.0, 2.0))
    mag = np.abs(Vm)
    rev_floor = ndimage.gaussian_filter(ndimage.minimum_filter1d(mag, size=14, axis=1) * 1.25, sigma=(1.5, 2.5))
    dry = np.clip(1.0 - 0.65 * (rev_floor / (mag + 1e-9)), 0.25, 1.0)
    center = np.sqrt(np.clip((pm - 0.80 * ps) / (pm + 1e-14), 0.10, 1.0))
    lead = librosa.istft(Vm * (center * dry), hop_length=256, length=n).astype(np.float32)
    v_rms = np.sqrt(ndimage.uniform_filter1d(lead ** 2, size=2400) + 1e-14)
    pk = float(np.max(v_rms) + 1e-9)
    gate = ndimage.gaussian_filter1d(np.clip((v_rms - pk * 0.004) / (pk * 0.006), 0.0, 1.0).astype(np.float32), sigma=600)
    gate[gate < 1e-4] = 0.0
    return (lead * gate).astype(np.float32)


def prepare_v9r21_stems(hq_npz: Path, hop: int = 256) -> dict:
    z = np.load(hq_npz)
    sr = 48000
    master = z["master"].astype(np.float32)
    voc = z["vocals"].astype(np.float32)
    n = len(master)

    lead = _lead_from_vocals(voc)
    rest = voc - np.column_stack([lead, lead])

    # gate vocal remainder to vocal phrases (+1.25 s tail); bleed outside goes to keys/rest
    lead_env = np.sqrt(ndimage.uniform_filter1d(lead ** 2, size=2400) + 1e-16)
    act = (lead_env > float(np.max(lead_env)) * 0.004)[::hop]
    m = ndimage.binary_dilation(act, structure=np.ones(int(1.25 * sr / hop), dtype=bool)) | ndimage.binary_dilation(
        act, structure=np.ones(int(0.25 * sr / hop), dtype=bool)
    )
    g = ndimage.gaussian_filter1d(m.astype(np.float32), sigma=12.0)
    gate = np.interp(np.arange(n, dtype=np.float32) / hop, np.arange(len(g), dtype=np.float32), g).astype(np.float32)
    gate[gate < 1e-4] = 0.0
    gate[gate > 0.999] = 1.0
    voc_rest = rest * gate[:, None]
    bleed = rest - voc_rest

    L = librosa.stft(voc_rest[:, 0], n_fft=2048, hop_length=512)
    R = librosa.stft(voc_rest[:, 1], n_fft=2048, hop_length=512)
    mag = np.sqrt(0.5 * (np.abs(L) ** 2 + np.abs(R) ** 2))
    rev = ndimage.gaussian_filter(ndimage.minimum_filter1d(mag, size=15, axis=1) * 1.45, sigma=(1.6, 3.0))
    rmask = np.clip(rev / (mag + 1e-9), 0.0, 0.90)
    bL = librosa.istft(L * (1 - rmask), hop_length=512, length=n).astype(np.float32)
    bR = librosa.istft(R * (1 - rmask), hop_length=512, length=n).astype(np.float32)
    b_rms = np.sqrt(ndimage.uniform_filter1d(0.5 * (bL ** 2 + bR ** 2), size=4800) + 1e-14)
    bpk = float(np.max(b_rms) + 1e-9)
    bg = ndimage.gaussian_filter1d(np.clip((b_rms - bpk * 0.045) / (bpk * 0.055), 0.0, 1.0).astype(np.float32), sigma=1200)
    bg[bg < 1e-3] = 0.0
    backing = np.column_stack([bL * bg, bR * bg]).astype(np.float32)
    space = (voc_rest - backing).astype(np.float32)

    bass_mono = (0.5 * (z["bass"][:, 0] + z["bass"][:, 1])).astype(np.float32)
    bass_side = z["bass"] - np.column_stack([bass_mono, bass_mono])
    drums = z["drums"].astype(np.float32)
    guitar = z["guitar"].astype(np.float32)
    keys = (z["piano"] + z["other_rest"] + bleed + bass_side).astype(np.float32)

    vocals_st = np.column_stack([lead, lead]) + backing + space
    tot = vocals_st + np.column_stack([bass_mono, bass_mono]) + drums + guitar + keys
    cons = float(10 * np.log10(np.sum(master.astype(np.float64) ** 2) / (np.sum((master - tot).astype(np.float64) ** 2) + 1e-30)))

    nf = n // hop + 1
    pad = 512
    def frame_rms(x):
        xp = np.pad(x, (pad, pad + hop))
        fr = np.lib.stride_tricks.as_strided(xp, shape=(nf, 1024), strides=(xp.strides[0] * hop, xp.strides[0]))
        return np.sqrt(np.mean(fr ** 2, axis=1) + 1e-16).astype(np.float32)

    return {
        "master": master,
        "lead_mono": lead,
        "backing_stereo": backing,
        "space_stereo": space,
        "vocals_stereo": vocals_st,
        "bass_mono": bass_mono,
        "drums_stereo": drums,
        "guitar_stereo": guitar,
        "keys_stereo": keys,
        "guitar_synth_stereo": guitar + keys,
        # Track 07 gains are Mid / Side RMS (backing is a wide M/S double, L/R anti-correlated)
        "back_rms_l": frame_rms(0.5 * (backing[:, 0] + backing[:, 1])),
        "back_rms_r": frame_rms(0.5 * (backing[:, 0] - backing[:, 1])),
        "pan_frames": np.zeros(nf, dtype=np.float32),
        "conservation_snr_db": round(cons, 2),
    }
