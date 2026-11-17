"""V9R2.1 Step 8: Clean Physical + Tight-Transient Drum Score (`services/drum_event_score.py`).

Eliminates random-phase STFT wash/hiss by modeling drums according to physical acoustics:
1. `low` (Kick, 30-160 Hz): Pure time-domain swept-sine + 2nd-harmonic membrane equations (0 FFT noise, 100% mono).
2. `midModal` (Snare shell / Woodblock `DI DI DA` / Rim, 165-1350 Hz): Dual damped sinusoidal resonators (0 noise).
3. `midSnap` (Snare wires / Clap, 1200-6800 Hz) & `high` (Hi-hat / Shaker / Cymbal, 4800-18500 Hz):
   512-pt tight-windowed transient bursts with strict -22 dB noise-floor gating and zero bleed extrapolation.
"""
from __future__ import annotations

import math
import numpy as np
import librosa
import scipy.signal as signal

SR = 48000
HOP = 256
NFFT = 512


def _mel_fb(lo: float, hi: float, n_mel: int) -> np.ndarray:
    return librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=n_mel, fmin=lo, fmax=hi, norm=None)


def _analyze_kicks(mono: np.ndarray) -> tuple[dict, dict]:
    sos_k = signal.butter(4, [28.0, 158.0], btype="bandpass", fs=SR, output="sos")
    yk = signal.sosfiltfilt(sos_k, mono).astype(np.float32)
    n = len(yk)

    env = librosa.onset.onset_strength(y=yk, sr=SR, hop_length=HOP, n_fft=1024)
    thr = float(np.percentile(env, 82) * 0.42)
    pks = librosa.util.peak_pick(env, pre_max=4, post_max=4, pre_avg=12, post_avg=12, delta=max(thr, 1e-4), wait=10)

    # Filter out bleed ripples (< -25 dB of max kick peak)
    raw_hits = []
    for p in pks:
        s0 = max(0, int(p) * HOP - 128)
        s1 = min(n, s0 + 3584)
        if s1 - s0 < 1024:
            continue
        pk_amp = float(np.max(np.abs(yk[s0:s1])))
        raw_hits.append((int(p), s0, pk_amp))

    max_pk = max((h[2] for h in raw_hits), default=1.0)
    min_pk = max_pk * (10.0 ** (-25.0 / 20.0))
    valid_hits = [h for h in raw_hits if h[2] >= min_pk]

    frames_out, amp_q, f_start_list, f_body_list, tau_ms_list = [], [], [], [], []
    prev_f = 0
    win_body = np.hanning(4096).astype(np.float32)
    win_atk = np.hanning(1024).astype(np.float32)

    for p, s0, pk_amp in valid_hits:
        # Fine-align s0 to local onset rise
        seg_full = yk[s0 : min(n, s0 + 6144)]
        pk_rel = int(np.argmax(np.abs(seg_full[:2048])))
        # Walk back to 12% threshold for crisp onset
        onset_rel = pk_rel
        thresh = 0.12 * abs(float(seg_full[pk_rel]))
        while onset_rel > 0 and abs(float(seg_full[onset_rel])) > thresh:
            onset_rel -= 1
        s_hit = s0 + onset_rel
        f_hit = int(round(s_hit / HOP))
        if frames_out and f_hit <= prev_f + 6:
            continue

        # Attack frequency (first 21 ms)
        seg_a = yk[s_hit : min(n, s_hit + 1024)]
        if len(seg_a) < 1024:
            seg_a = np.pad(seg_a, (0, 1024 - len(seg_a)))
        sp_a = np.abs(np.fft.rfft(seg_a * win_atk))
        ba = int(np.argmax(sp_a[1:5])) + 1
        f_st = float(np.clip(ba * SR / 1024.0, 82.0, 155.0))

        # Body frequency (15 ms .. 100 ms)
        s_b = min(n, s_hit + 512)
        seg_b = yk[s_b : min(n, s_b + 4096)]
        if len(seg_b) < 4096:
            seg_b = np.pad(seg_b, (0, 4096 - len(seg_b)))
        sp_b = np.abs(np.fft.rfft(seg_b * win_body))
        # Parabolic peak around bins 3..9 (35..105 Hz)
        bb = int(np.argmax(sp_b[3:9])) + 3
        y0, y1, y2 = float(sp_b[bb - 1]), float(sp_b[bb]), float(sp_b[bb + 1])
        denom = y0 - 2.0 * y1 + y2
        delta_b = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
        f_bd = float(np.clip((bb + np.clip(delta_b, -0.5, 0.5)) * SR / 4096.0, 48.0, 88.0))
        if f_st < f_bd + 14.0:
            f_st = f_bd + 24.0

        # Effective decay tau from energy ratio (0..45ms vs 45..110ms)
        seg_e = yk[s_hit : min(n, s_hit + 5760)]
        e1 = float(np.mean(seg_e[:2160] ** 2) + 1e-12)
        e2 = float(np.mean(seg_e[2160:5280] ** 2) + 1e-12) if len(seg_e) > 3000 else e1 * 0.15
        ratio = np.clip(e2 / e1, 0.02, 0.72)
        tau_s = -0.055 / math.log(ratio)
        tau_ms = int(np.clip(round(tau_s * 1000.0), 32, 105))

        frames_out.append(f_hit - prev_f)
        prev_f = f_hit
        amp_q.append(int(round(20.0 * math.log10(max(pk_amp, 1e-6)) * 4.0)))
        f_start_list.append(int(round(f_st)))
        f_body_list.append(int(round(f_bd)))
        tau_ms_list.append(tau_ms)

    score_low = {
        "frame": frames_out,
        "ampQ": amp_q,
        "fStart": f_start_list,
        "fBody": f_body_list,
        "tauMs": tau_ms_list,
    }
    return score_low, {"hits": len(frames_out)}


def _analyze_mid_modal(drums_st: np.ndarray) -> tuple[dict, dict]:
    mono = (0.5 * (drums_st[:, 0] + drums_st[:, 1])).astype(np.float32)
    sos_m = signal.butter(4, [168.0, 1350.0], btype="bandpass", fs=SR, output="sos")
    ym = signal.sosfiltfilt(sos_m, mono).astype(np.float32)
    yL = signal.sosfiltfilt(sos_m, drums_st[:, 0]).astype(np.float32)
    yR = signal.sosfiltfilt(sos_m, drums_st[:, 1]).astype(np.float32)
    n = len(ym)

    env = librosa.onset.onset_strength(y=ym, sr=SR, hop_length=HOP, n_fft=1024)
    thr = float(np.percentile(env, 84) * 0.45)
    pks = librosa.util.peak_pick(env, pre_max=4, post_max=4, pre_avg=12, post_avg=12, delta=max(thr, 1e-4), wait=8)

    cands = []
    for p in pks:
        s0 = max(0, int(p) * HOP - 64)
        s1 = min(n, s0 + 2048)
        if s1 - s0 < 1024:
            continue
        pk_a = float(np.max(np.abs(ym[s0:s1])))
        cands.append((int(p), s0, pk_a))

    max_a = max((c[2] for c in cands), default=1.0)
    min_a = max_a * (10.0 ** (-24.0 / 20.0))
    valid = [c for c in cands if c[2] >= min_a]

    win = np.hanning(2048).astype(np.float32)
    frames_out, amp_q, f1_l, f2_l, r2_q, tau_l, pan_q = [], [], [], [], [], [], []
    prev_f = 0

    for p, s0, pk_a in valid:
        if frames_out and p <= prev_f + 5:
            continue
        seg = ym[s0 : min(n, s0 + 2048)]
        if len(seg) < 2048:
            seg = np.pad(seg, (0, 2048 - len(seg)))
        sp = np.abs(np.fft.rfft(seg * win))
        b_lo, b_hi = int(round(170 * 2048 / SR)), int(round(1350 * 2048 / SR))
        sub = sp[b_lo:b_hi].copy()
        idx1 = int(np.argmax(sub))
        a1 = float(sub[idx1] + 1e-9)
        f1 = (b_lo + idx1) * SR / 2048.0
        # Mask +-3 bins around idx1 to find 2nd mode
        sub[max(0, idx1 - 3) : min(len(sub), idx1 + 4)] = 0.0
        idx2 = int(np.argmax(sub))
        a2 = float(sub[idx2])
        f2 = (b_lo + idx2) * SR / 2048.0

        # Decay time constant
        e1 = float(np.mean(seg[:768] ** 2) + 1e-12)
        e2 = float(np.mean(seg[768:1920] ** 2) + 1e-12)
        ratio = np.clip(e2 / e1, 0.015, 0.60)
        tau_ms = int(np.clip(round((-0.020 / math.log(ratio)) * 1000.0), 12, 58))

        el = float(np.sqrt(np.mean(yL[s0 : min(n, s0 + 1024)] ** 2) + 1e-12))
        er = float(np.sqrt(np.mean(yR[s0 : min(n, s0 + 1024)] ** 2) + 1e-12))
        pan = int(np.clip(round(64.0 * math.atan2(er, el) / (0.5 * math.pi)), 8, 56))

        frames_out.append(p - prev_f)
        prev_f = p
        amp_q.append(int(round(20.0 * math.log10(max(pk_a, 1e-6)) * 4.0)))
        f1_l.append(int(round(f1)))
        f2_l.append(int(round(f2)))
        r2_q.append(int(np.clip(round((a2 / a1) * 100.0), 0, 95)))
        tau_l.append(tau_ms)
        pan_q.append(pan)

    score_modal = {
        "frame": frames_out,
        "ampQ": amp_q,
        "f1": f1_l,
        "f2": f2_l,
        "r2Pct": r2_q,
        "tauMs": tau_l,
        "pan": pan_q,
    }
    return score_modal, {"hits": len(frames_out)}


def _analyze_transient_band(
    drums_st: np.ndarray,
    lo: float,
    hi: float,
    n_mel: int,
    T: int,
    K: int,
    wait: int,
    floor_rel_db: float,
    gate_db: float,
    seed: int,
) -> tuple[dict, dict, dict]:
    mono = (0.5 * (drums_st[:, 0] + drums_st[:, 1])).astype(np.float32)
    side = (0.5 * (drums_st[:, 0] - drums_st[:, 1])).astype(np.float32)
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    yb = signal.sosfiltfilt(sos, mono).astype(np.float32)
    ys = signal.sosfiltfilt(sos, side).astype(np.float32)

    win = np.sin(np.pi * (np.arange(NFFT, dtype=np.float32) + 0.5) / NFFT)
    S = np.abs(librosa.stft(yb, n_fft=NFFT, hop_length=HOP, window=win)) ** 2 / float(NFFT)
    Ss = np.abs(librosa.stft(ys, n_fft=NFFT, hop_length=HOP, window=win)) ** 2 / float(NFFT)
    nf = S.shape[1]

    fb = _mel_fb(lo, hi, n_mel)
    E = fb @ S
    Es = fb @ Ss
    Edb = 10.0 * np.log10(E + 1e-14)
    tot = 10.0 * np.log10(E.sum(0) + 1e-14)

    flux = np.maximum(0.0, np.diff(tot, prepend=tot[0]))
    thr = float(np.percentile(flux, 82) + 0.55 * np.std(flux))
    pk = librosa.util.peak_pick(flux, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=thr * 0.55, wait=wait)
    floor_db = float(tot.max()) + floor_rel_db
    pk = pk[tot[np.minimum(pk + 1, nf - 1)] > floor_db]

    rng = np.random.default_rng(seed)
    P = np.full((len(pk), n_mel, T), -90.0, dtype=np.float32)
    G = np.zeros(len(pk), dtype=np.float32)

    t_decay_mask = np.arange(T, dtype=np.float32)[None, :]
    for i, p in enumerate(pk):
        q = min(nf, p + T)
        nxt = pk[i + 1] if i + 1 < len(pk) else nf
        L = min(q - p, max(2, nxt - p))
        seg = Edb[:, p : p + L]
        g = float(np.max(seg[:, : min(L, 3)]))
        G[i] = g
        rel = seg - g
        # Beyond next hit, decay steeply (-3.5 dB/frame) into silence instead of extrapolating bleed!
        pat = np.full((n_mel, T), -90.0, dtype=np.float32)
        pat[:, :L] = rel
        if L < T:
            tail_start = rel[:, L - 1 : L]
            steps = np.arange(1, T - L + 1, dtype=np.float32)[None, :]
            pat[:, L:] = np.maximum(-90.0, tail_start - 3.8 * steps)
        # Natural exponential damper + strict noise-floor gate
        pat -= 0.45 * t_decay_mask
        pat = np.where(pat < gate_db, -90.0, pat)
        P[i] = np.clip(pat, -90.0, 3.0)

    K_eff = min(K, max(1, len(pk)))
    X = P.reshape(len(pk), -1)
    w = np.repeat(np.exp(-np.arange(T) / (T / 3.0))[None, :], n_mel, 0).reshape(-1)
    Xw = X * w
    C = Xw[rng.choice(len(pk), K_eff, replace=False)]
    for _ in range(25):
        d = ((Xw[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(K_eff):
            if np.any(lab == k):
                C[k] = Xw[lab == k].mean(0)

    tmpl = np.stack([X[lab == k].mean(0) if np.any(lab == k) else X.mean(0) for k in range(K_eff)]).reshape(K_eff, n_mel, T)
    # Re-apply clean gate on cluster centroids so averaging doesn't create a lingering noise shelf
    tmpl = np.where(tmpl < gate_db, -90.0, np.clip(tmpl, -90.0, 3.0))

    width = np.full((K_eff, n_mel), -18.0, dtype=np.float32)
    for k in range(K_eff):
        idx = pk[lab == k]
        if len(idx):
            ws = np.mean([Es[:, p : p + 3].sum(1) / (E[:, p : p + 3].sum(1) + 1e-14) for p in idx], axis=0)
            # Cap side/mid ratio at -5 dB so stereo noise never dominates
            width[k] = np.clip(10.0 * np.log10(ws + 1e-6), -36.0, -5.0)

    for i in range(len(pk)):
        k = int(lab[i])
        G[i] += float(np.mean(P[i, :, :2] - tmpl[k, :, :2]))

    fam_model = {
        "band": [lo, hi],
        "nMel": n_mel,
        "T": T,
        "templatesDb": np.round(tmpl * 2.0).astype(int).tolist(),
        "sideDb": np.round(width).astype(int).tolist(),
    }
    frames = pk.astype(int)
    fam_score = {
        "frame": np.diff(frames, prepend=0).tolist(),
        "tmpl": lab.astype(int).tolist(),
        "gainQ": np.round(G * 4.0).astype(int).tolist(),
    }
    return fam_model, fam_score, {"hits": int(len(pk)), "templates": int(K_eff)}


def analyze_drums(drums_st: np.ndarray, seed: int = 0) -> dict:
    mono = (0.5 * (drums_st[:, 0] + drums_st[:, 1])).astype(np.float32)
    low_score, low_stats = _analyze_kicks(mono)
    modal_score, modal_stats = _analyze_mid_modal(drums_st)
    snap_model, snap_score, snap_stats = _analyze_transient_band(
        drums_st, lo=1150.0, hi=6800.0, n_mel=12, T=18, K=12, wait=8, floor_rel_db=-27.0, gate_db=-20.0, seed=seed + 1
    )
    high_model, high_score, high_stats = _analyze_transient_band(
        drums_st, lo=4800.0, hi=18800.0, n_mel=12, T=22, K=12, wait=6, floor_rel_db=-28.0, gate_db=-21.0, seed=seed + 2
    )

    model = {
        "hop": HOP,
        "nfft": NFFT,
        "families": {
            "midSnap": snap_model,
            "high": high_model,
        },
    }
    score = {
        "low": low_score,
        "midModal": modal_score,
        "midSnap": snap_score,
        "high": high_score,
    }
    stats = {
        "low": low_stats,
        "midModal": modal_stats,
        "midSnap": snap_stats,
        "high": high_stats,
    }
    return {"model": model, "score": score, "stats": stats}


def calibrate_drum_gains(ref_st: np.ndarray, parts: dict[str, np.ndarray]) -> dict[str, float]:
    """Sub-band physical RMS matching so mid/high transients never get boosted to match sub-bass kick energy."""
    bands = {
        "low": (30.0, 158.0, 0.96),
        "midModal": (168.0, 1350.0, 0.92),
        "midSnap": (1250.0, 4800.0, 0.78),
        "high": (4800.0, 18500.0, 0.75),
    }
    gains: dict[str, float] = {}
    for k, (lo, hi, scale) in bands.items():
        if k not in parts:
            continue
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        r_b = signal.sosfilt(sos, ref_st, axis=0)
        p_b = signal.sosfilt(sos, parts[k], axis=0)
        r_rms = float(np.sqrt(np.mean(r_b.astype(np.float64) ** 2)) + 1e-12)
        p_rms = float(np.sqrt(np.mean(p_b.astype(np.float64) ** 2)) + 1e-12)
        gains[k] = round(scale * (r_rms / p_rms), 6)
    return gains

