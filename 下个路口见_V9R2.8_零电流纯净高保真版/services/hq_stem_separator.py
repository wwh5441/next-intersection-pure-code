"""V9R2.1 Step 1: Highest-quality 48kHz stem separation from the original MP3
(`services/hq_stem_separator.py`).

- htdemucs (4 stems) with shift-ensemble (shifts=8, overlap=0.5) for vocals / drums / bass / other
- htdemucs_6s with shift-ensemble to split `other` into guitar / piano(keys) / remainder
- Polyphase 44.1k<->48k resampling, float32 throughout
- Wiener redistribution + exact closure so every stem set sums to the 48kHz master
Output is ANALYSIS MATERIAL ONLY (never shipped in the pure-code product).
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
import subprocess
import time
import librosa
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

FFMPEG = "/Users/weckham/.hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg"
SR = 48000


def decode_master(path: Path, sr: int = SR) -> np.ndarray:
    exe = FFMPEG if Path(FFMPEG).exists() else "ffmpeg"
    raw = subprocess.check_output([exe, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sr), "pipe:1"])
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def _run_model(master: np.ndarray, model: str, shifts: int, overlap: float) -> dict[str, np.ndarray]:
    import torch
    from demucs.api import Separator

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sep = Separator(model=model, device=device, shifts=shifts, overlap=overlap, split=True)
    msr = int(sep.samplerate)
    g = gcd(SR, msr)
    x = signal.resample_poly(master, msr // g, SR // g, axis=0).astype(np.float32)
    with torch.inference_mode():
        _, stems = sep.separate_tensor(torch.from_numpy(x.T.copy()), sr=msr)
    n = len(master)
    out = {}
    for k, v in stems.items():
        y = signal.resample_poly(v.cpu().numpy().T, SR // g, msr // g, axis=0).astype(np.float32)
        out[k] = y[:n] if len(y) >= n else np.pad(y, ((0, n - len(y)), (0, 0)))
    del sep, stems
    if device == "mps":
        torch.mps.empty_cache()
    return out


def _wiener_close(target: np.ndarray, stems: dict[str, np.ndarray], close_key: str) -> dict[str, np.ndarray]:
    """Redistribute (target - sum) by smoothed power ratios, then close exactly on `close_key`."""
    names = list(stems)
    n = len(target)
    out = {k: v.copy() for k, v in stems.items()}
    for ch in range(2):
        T = librosa.stft(target[:, ch], n_fft=4096, hop_length=1024, window="hann")
        S = [librosa.stft(stems[k][:, ch], n_fft=4096, hop_length=1024, window="hann") for k in names]
        R = T - sum(S)
        P = [ndimage.gaussian_filter(np.abs(s) ** 2, sigma=(1.5, 1.0)) + 1e-12 for s in S]
        Pt = sum(P)
        for i, k in enumerate(names):
            out[k][:, ch] = librosa.istft(S[i] + (P[i] / Pt) * R, hop_length=1024, length=n).astype(np.float32)
        out[close_key][:, ch] = target[:, ch] - sum(out[k][:, ch] for k in names if k != close_key)
    return out


def separate_hq(mp3: Path, cache_npz: Path, shifts: int = 8) -> dict[str, np.ndarray]:
    if cache_npz.exists():
        z = np.load(cache_npz)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    master = decode_master(mp3)
    print(f"  decoded {len(master):,} samples @48k")

    s4 = _run_model(master, "htdemucs", shifts=shifts, overlap=0.5)
    print(f"  htdemucs x{shifts} done ({time.time()-t0:.0f}s)")
    s4 = _wiener_close(master, {k: s4[k] for k in ("vocals", "drums", "bass", "other")}, "other")

    s6 = _run_model(master, "htdemucs_6s", shifts=shifts, overlap=0.5)
    print(f"  htdemucs_6s x{shifts} done ({time.time()-t0:.0f}s)")
    # Split the 4-stem `other` into guitar / piano / remainder using 6s estimates as guides
    split = _wiener_close(s4["other"], {"guitar": s6["guitar"], "piano": s6["piano"], "other_rest": s6["other"]}, "other_rest")

    res = {
        "master": master,
        "vocals": s4["vocals"],
        "drums": s4["drums"],
        "bass": s4["bass"],
        "guitar": split["guitar"],
        "piano": split["piano"],
        "other_rest": split["other_rest"],
    }
    tot = res["vocals"] + res["drums"] + res["bass"] + res["guitar"] + res["piano"] + res["other_rest"]
    snr = 10 * np.log10(np.sum(master.astype(np.float64) ** 2) / (np.sum((master - tot).astype(np.float64) ** 2) + 1e-30))
    print(f"  conservation SNR = {snr:.2f} dB ({time.time()-t0:.0f}s)")
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_npz, **res)
    return res
