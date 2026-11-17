"""48kHz Stereo Audio Decoder & Polyphase Resampler."""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy.signal import resample_poly

FALLBACK_FFMPEG = "/Users/weckham/.cache/music-live/bin/ffmpeg"


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if Path(FALLBACK_FFMPEG).is_file():
        return FALLBACK_FFMPEG
    raise FileNotFoundError("ffmpeg binary not found")


def decode_stereo(path: str | Path, sr: int = 48000) -> np.ndarray:
    cmd = [
        _find_ffmpeg(),
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "2",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True)
    raw = np.frombuffer(proc.stdout, dtype="<f4")
    return raw.reshape(-1, 2).copy()


def resample_audio(arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return arr.astype(np.float32, copy=False)
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(arr, up, down, axis=0).astype(np.float32)
