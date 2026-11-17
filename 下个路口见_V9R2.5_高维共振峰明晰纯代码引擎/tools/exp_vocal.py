#!/usr/bin/env python3
"""V9R2.1 vocal utility & metrics helper (`tools/exp_vocal.py`)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCR = Path("/Users/weckham/.gemini/antigravity/brain/983ec71b-8826-49ac-8121-db9639eade27/scratch")
HQ = SCR / "v9r21_hq_stems_48k.npz"
PREP = SCR / "v9r21_prep_stems.pkl"
LRC = ROOT.parent / "mp3_output/李宇春 - 下个, 路口, 见.mp3".replace(".mp3", ".lrc")


def mel_snr(ref, syn, sr=48000):
    r = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = min(len(r), len(s))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80, fmin=45.0, fmax=19500.0)
    Mr = fb @ np.abs(librosa.stft(r[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s[:L], n_fft=2048, hop_length=512))
    fl = float(np.max(Mr) * 1e-3 + 1e-8)
    a, b = np.log10(np.maximum(Mr, fl)), np.log10(np.maximum(Ms, fl))
    return round(float(10 * np.log10(np.sum(a ** 2) / (np.sum((a - b) ** 2) + 1e-12))), 2)


def load_prep():
    if PREP.exists():
        return pickle.load(open(PREP, "rb"))
    from services.v9r21_stem_prep import prepare_v9r21_stems
    st = prepare_v9r21_stems(HQ)
    pickle.dump(st, open(PREP, "wb"), protocol=4)
    return st


def write_js(rel, name, obj):
    p = ROOT / rel
    p.write_text(f"export const {name} = {json.dumps(obj, ensure_ascii=False, separators=(',', ':'))};\n", encoding="utf-8")
