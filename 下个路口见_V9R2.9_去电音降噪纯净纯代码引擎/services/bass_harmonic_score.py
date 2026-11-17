"""V9R2.1 Step 7: Bass -> Discrete Pure-Code Musical Note Events (`services/bass_harmonic_score.py`).

Instead of storing 19,232 continuous 10ms frames, segments the bass line into ~350-550 discrete
plucked/slid note events with sparse pitch/gain anchors + 12 learned fingerstyle timbre presets.
"""
from __future__ import annotations

import numpy as np
import librosa
import scipy.ndimage as ndimage
import scipy.signal as signal

SR = 48000
HOP = 256


def _rdp_knots(y: np.ndarray, tol: float, min_step: int = 2) -> list[int]:
    """Iterative Ramer-Douglas-Peucker simplification returning key frame indices."""
    L = len(y)
    if L <= 2:
        return list(range(L))
    keep = {0, L - 1}
    stack = [(0, L - 1)]
    while stack:
        a, b = stack.pop()
        if b - a <= min_step:
            continue
        ya, yb = float(y[a]), float(y[b])
        dx = float(b - a)
        best_i, best_d = -1, -1.0
        for i in range(a + 1, b):
            pred = ya + (yb - ya) * ((i - a) / dx)
            d = abs(float(y[i]) - pred)
            if d > best_d:
                best_d, best_i = d, i
        if best_d > tol and best_i > 0:
            keep.add(best_i)
            stack.append((a, best_i))
            stack.append((best_i, b))
    return sorted(keep)


def analyze_bass(bass: np.ndarray, n_harm: int = 36, n_presets: int = 12, seed: int = 0) -> dict:
    n = len(bass)
    nf = n // HOP + 1

    # 1) F0 via pYIN on 12 kHz (hop 64 == 256 @ 48 kHz)
    y12 = signal.resample_poly(bass, 1, 4).astype(np.float32)
    f0, _, _ = librosa.pyin(y12, fmin=28.0, fmax=420.0, sr=12000, frame_length=2048, hop_length=64, center=True)
    f0 = np.nan_to_num(f0)[:nf]
    f0 = np.pad(f0, (0, max(0, nf - len(f0))))

    yp = np.pad(bass, (1024, 1024 + HOP))
    fr = np.lib.stride_tricks.as_strided(yp, shape=(nf, 2048), strides=(yp.strides[0] * HOP, yp.strides[0]))
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-14).astype(np.float32)
    act = rms > (float(rms.max()) * 10.0 ** (-44.0 / 20.0))

    lbl, k = ndimage.label(act)
    f0b = f0.copy()
    for i in range(1, k + 1):
        idx = np.flatnonzero(lbl == i)
        vi = idx[f0[idx] > 0]
        if len(vi) == 0:
            continue
        f0b[idx] = np.interp(idx, vi, f0[vi])
    f0 = np.where(act, f0b, 0.0)
    f0 = np.where(f0 > 0, 2.0 ** ndimage.median_filter(np.log2(np.maximum(f0, 1e-3)), size=3), 0.0).astype(np.float32)

    # 2) Extract harmonic profiles on active frames
    W = 8192
    win = signal.windows.blackmanharris(W, sym=False).astype(np.float32)
    norm = 2.0 / float(win.sum())
    xp = np.pad(bass, (W // 2, W // 2 + HOP))
    H = np.full((nf, n_harm), -120.0, dtype=np.float32)
    hidx = np.arange(1, n_harm + 1)
    act_idx = np.flatnonzero(f0 > 0)
    for c0 in range(0, len(act_idx), 512):
        ids = act_idx[c0 : c0 + 512]
        frames = np.stack([xp[i * HOP : i * HOP + W] for i in ids]) * win
        mag = np.abs(np.fft.rfft(frames, axis=1)) * norm
        for j, i in enumerate(ids):
            fh = hidx * f0[i]
            ok = fh < 14000.0
            b0 = np.floor(fh[ok] * W / SR).astype(int)
            cand = np.stack([mag[j, np.clip(b0 + d, 1, W // 2 - 1)] for d in (-2, -1, 0, 1, 2)])
            H[i, : int(ok.sum())] = 20.0 * np.log10(cand.max(0) + 1e-9)

    lin = 10.0 ** (H / 20.0)
    tot = np.sqrt(0.5 * np.sum(lin ** 2, axis=1) + 1e-18)
    rel = np.where(H > -119.0, H - 20.0 * np.log10(tot)[:, None], -60.0)
    rel = np.clip(rel, -60.0, 6.0).astype(np.float32)

    # Cluster active frames into `n_presets` fingerstyle/pluck timbre presets
    rng = np.random.default_rng(seed)
    A = rel[act_idx]
    K_eff = min(n_presets, max(1, len(A)))
    C = A[rng.choice(len(A), K_eff, replace=False)].copy()
    for _ in range(25):
        d = ((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for p in range(K_eff):
            if np.any(lab == p):
                C[p] = A[lab == p].mean(0)
    frame_preset = np.zeros(nf, dtype=int)
    frame_preset[act_idx] = lab

    # 3) Segment into discrete musical bass notes using onset novelty + pitch jumps
    onset_env = librosa.onset.onset_strength(y=bass, sr=SR, hop_length=HOP, n_fft=2048)
    on_pks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=float(np.percentile(onset_env, 82) * 0.35), wait=6
    )
    onset_set = set(int(p) for p in on_pks if act[min(int(p), nf - 1)])

    notes = []
    prev_s = 0
    lbl, k = ndimage.label(f0 > 0)
    for i in range(1, k + 1):
        idx = np.flatnonzero(lbl == i)
        if len(idx) < 3:
            continue
        s0, s1 = int(idx[0]), int(idx[-1]) + 1
        cuts = [s0]
        for f in range(s0 + 4, s1 - 3):
            df_cents = abs(1200.0 * np.log2(max(f0[f], 1.0) / max(f0[f - 1], 1.0)))
            if (f in onset_set or df_cents > 80.0) and (f - cuts[-1]) >= 6:
                cuts.append(f)
        cuts.append(s1)

        for a, b in zip(cuts[:-1], cuts[1:]):
            L = b - a
            if L < 3:
                continue
            cents_seg = 1200.0 * np.log2(f0[a:b] / 55.0)
            gdb_seg = 20.0 * np.log10(rms[a:b] + 1e-9)
            kp = _rdp_knots(cents_seg, tol=22.0, min_step=2)
            kg = _rdp_knots(gdb_seg, tol=1.1, min_step=2)
            knots = sorted(set(kp) | set(kg))
            cq = np.round(cents_seg[knots]).astype(int)
            gq = np.round(gdb_seg[knots] * 2.0).astype(int)
            tid = int(np.bincount(frame_preset[a : min(b, a + 12)]).argmax())
            ds = int(a) - prev_s
            prev_s = int(a)
            notes.append([
                ds,
                tid,
                np.diff(knots, prepend=0).tolist(),
                np.diff(cq, prepend=0).tolist(),
                np.diff(gq, prepend=0).tolist(),
            ])

    model = {
        "hop": HOP,
        "nHarm": n_harm,
        "f0RefHz": 55.0,
        "gainStepDb": 0.5,
        "timbresDbQ": np.round(C * 2.0).astype(int).tolist(),  # 0.5 dB steps
    }
    score = {"notes": notes}
    stats = {"activeFrames": int(len(act_idx)), "notes": len(notes), "presets": int(K_eff)}
    return {"model": model, "score": score, "stats": stats}
