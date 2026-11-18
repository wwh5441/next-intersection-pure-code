"""FFmpeg 48kHz PCM decoding and polyphase rational resampling."""
from __future__ import annotations

from math import gcd
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy.signal import resample_poly

DEFAULT_FFMPEG = "/Users/weckham/.cache/music-live/bin/ffmpeg"


def resolve_ffmpeg() -> str:
    if Path(DEFAULT_FFMPEG).is_file():
        return DEFAULT_FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg executable not found.")


def decode_stereo(path: Path, sr: int = 48000) -> np.ndarray:
    """Decodes an audio file into float32 stereo PCM of shape (N, 2) at `sr` Hz."""
    ffmpeg = resolve_ffmpeg()
    raw = subprocess.check_output(
        [ffmpeg, "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"]
    )
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def resample_audio(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resamples 1D or 2D audio from `source_sr` to `target_sr`."""
    if source_sr == target_sr:
        return y.astype(np.float32, copy=True)
    g = gcd(source_sr, target_sr)
    return resample_poly(y, target_sr // g, source_sr // g, axis=0).astype(np.float32)
