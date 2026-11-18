#!/usr/bin/env python3
"""Diagnose vocal error distribution (per mel band, voiced/unvoiced/silent)."""
import sys, json, pickle
from pathlib import Path
import numpy as np, librosa
ROOT = Path("/Users/weckham/Desktop/MUSIC/下个路口见_V9R2.1_纯代码最高音质引擎")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import exp_vocal as ev

cfg = json.loads((ROOT / "schemas/v9r2_genesis_config.json").read_text(encoding="utf-8"))
st = ev.load_prep()
r = ev.render_vocals(cfg)
ref, syn = st["lead_mono"], r["lead"]
fb = librosa.filters.mel(sr=48000, n_fft=2048, n_mels=40, fmin=45.0, fmax=19500.0)
Mr = fb @ np.abs(librosa.stft(ref, n_fft=2048, hop_length=512))
Ms = fb @ np.abs(librosa.stft(syn, n_fft=2048, hop_length=512))
fl = float(np.max(Mr) * 1e-3)
A, B = 20 * np.log10(np.maximum(Mr, fl)), 20 * np.log10(np.maximum(Ms, fl))
act = A.max(0) > 20 * np.log10(fl) + 30
cf = librosa.mel_frequencies(42, fmin=45, fmax=19500)[1:-1]
print("band  Hz     bias(dB) rmse(dB)")
for i in range(0, 40, 3):
    d = (B[i] - A[i])[act]
    print(f"{i:3d} {cf[i]:7.0f} {d.mean():+7.2f} {np.sqrt((d**2).mean()):7.2f}")
# silent-region leakage
sil = ~act
print("silent frames: ref energy dB %.1f syn energy dB %.1f" % (A[:, sil].mean(), B[:, sil].mean()))
# per-frame rmse vs time quantiles
fr = np.sqrt(((B - A) ** 2).mean(0))
print("active frame rmse median %.2f p90 %.2f; silent frame rmse median %.2f" % (np.median(fr[act]), np.percentile(fr[act], 90), np.median(fr[sil])))
# overall level
print("overall rms ref %.4f syn %.4f" % (np.sqrt((ref**2).mean()), np.sqrt((syn**2).mean())))
