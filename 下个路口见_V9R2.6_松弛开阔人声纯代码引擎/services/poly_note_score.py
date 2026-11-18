"""V9R2.1 Step 9: Guitar + Keys -> Discrete Pure-Code Musical Note Events (`services/poly_note_score.py`).

Separately analyzes `guitar_stereo` (acoustic guitar picking/strumming) and `keys_stereo` (Rhodes/pad)
into compact discrete note events `[inst, midiIdx, startFrame, durFrames, peakDbQ, atkFrames, decayQ, sustainQ, panQ]`
so guitar and keys have distinct physical timbres, real guitar pick transients, and ~20-35 KB Gzip total.
"""
from __future__ import annotations

import numpy as np
import librosa

SR = 48000
HOP = 512
FMIN = librosa.midi_to_hz(28)
BPO = 36
N_BINS = BPO * 9
MIDI_LO, MIDI_HI = 36, 96
N_HARM = 24
N_NOISE = 8


def _cqt_freqs(tuning: float) -> np.ndarray:
    return librosa.cqt_frequencies(N_BINS, fmin=FMIN * 2.0 ** (tuning / 12.0), bins_per_octave=BPO)


def _nmf_instrument(mono: np.ndarray, stereo: np.ndarray | None, tuning: float, iters: int = 360, seed: int = 0) -> dict:
    n_bins = int(min(N_BINS, np.floor(BPO * np.log2((SR / 2.0 * 0.95) / FMIN))))
    f0_ref = FMIN * 2.0 ** (tuning / 12.0)
    C = np.abs(librosa.cqt(mono, sr=SR, hop_length=HOP, fmin=f0_ref, n_bins=n_bins, bins_per_octave=BPO))
    freqs = _cqt_freqs(tuning)[:n_bins]
    notes = np.arange(MIDI_LO, MIDI_HI + 1)
    n_notes = len(notes)
    rng = np.random.default_rng(seed)

    W = np.zeros((n_bins, n_notes + N_NOISE))
    for j, m in enumerate(notes):
        f0 = librosa.midi_to_hz(m) * 2.0 ** (tuning / 12.0)
        for h in range(1, N_HARM + 1):
            fh = f0 * h
            if fh >= freqs[-1]:
                break
            b = int(np.argmin(np.abs(freqs - fh)))
            for d, w in ((-1, 0.4), (0, 1.0), (1, 0.4)):
                if 0 <= b + d < n_bins:
                    W[b + d, j] += w / h
    W[:, n_notes:] = rng.random((n_bins, N_NOISE)) + 0.1
    H = rng.random((W.shape[1], C.shape[1])) * max(float(C.mean()), 1e-5) + 1e-6
    eps = 1e-12
    for it in range(iters):
        WH = W @ H + eps
        H *= (W.T @ (C / WH)) / (W.sum(0)[:, None] + eps)
        if it % 2 == 0:
            WH = W @ H + eps
            W *= ((C / WH) @ H.T) / (H.sum(1)[None, :] + eps)
            s = W.sum(0) + eps
            W /= s
            H *= s[:, None]

    harm = np.zeros((n_notes, N_HARM))
    for j, m in enumerate(notes):
        f0 = librosa.midi_to_hz(m) * 2.0 ** (tuning / 12.0)
        for h in range(1, N_HARM + 1):
            fh = f0 * h
            if fh >= freqs[-1]:
                break
            b = int(np.argmin(np.abs(freqs - fh)))
            harm[j, h - 1] = W[max(0, b - 1) : b + 2, j].sum()
    harm_db = 20.0 * np.log10(harm / (harm.max(1, keepdims=True) + eps) + 1e-6)

    Hl = Hr = None
    if stereo is not None:
        def _fit(ch: np.ndarray) -> np.ndarray:
            Cc = np.abs(librosa.cqt(ch, sr=SR, hop_length=HOP, fmin=f0_ref, n_bins=n_bins, bins_per_octave=BPO))
            Hc = H.copy() * 0.5 + 1e-6
            for _ in range(45):
                Hc *= (W.T @ (Cc / (W @ Hc + eps))) / (W.sum(0)[:, None] + eps)
            return Hc[:n_notes]
        Hl = _fit(np.ascontiguousarray(stereo[:, 0], dtype=np.float32))
        Hr = _fit(np.ascontiguousarray(stereo[:, 1], dtype=np.float32))

    return {"harmDb": harm_db, "Hnotes": H[:n_notes], "Hl": Hl, "Hr": Hr}


def _extract_discrete_notes(res: dict, inst_id: int, ref_global: float, floor_db: float, min_dur: int = 3) -> list[list[int]]:
    Hn = res["Hnotes"]
    D = 20.0 * np.log10(Hn / (ref_global + 1e-12) + 1e-9)
    notes_out: list[list[int]] = []

    for j in range(Hn.shape[0]):
        act = D[j] > floor_db
        if not act.any():
            continue
        idx = np.flatnonzero(act)
        brk = np.flatnonzero(np.diff(idx) > 2)
        segs = np.split(idx, brk + 1)

        for sgm in segs:
            if len(sgm) < min_dur:
                continue
            s0, s1 = int(sgm[0]), int(sgm[-1]) + 1
            # Split re-articulated notes on local dips followed by >4.5 dB re-attack
            env = D[j, s0:s1]
            cuts = [0]
            for t in range(3, len(env) - 3):
                if env[t] < env[t - 1] and env[t] <= env[t + 1]:
                    look_pk = float(np.max(env[t : min(len(env), t + 5)]))
                    if (look_pk - float(env[t])) >= 5.0 and (t - cuts[-1]) >= 4:
                        cuts.append(t)
            cuts.append(len(env))

            for a, b in zip(cuts[:-1], cuts[1:]):
                L = b - a
                if L < min_dur:
                    continue
                seg_db = env[a:b]
                pk_i = int(np.argmax(seg_db))
                pk_db = float(seg_db[pk_i])
                if pk_db < floor_db + 3.0:
                    continue
                pk_q = int(round(pk_db))  # 1 dB step
                atk = int(max(1, min(L - 1, pk_i + 1)))
                # Decay & sustain parameters from post-peak curve
                if L - pk_i > 2:
                    tail_db = float(np.mean(seg_db[max(pk_i + 1, b - a - 3) :]))
                    drop_db = max(0.0, pk_db - tail_db)
                    sustain_lin = float(np.clip(10.0 ** (-drop_db / 20.0), 0.05, 1.0))
                    half_idx = next((t - pk_i for t in range(pk_i + 1, L) if seg_db[t] <= pk_db - 6.0), max(2, (L - pk_i) // 2))
                    decay_q = int(np.clip(half_idx, 1, 64))
                else:
                    sustain_lin = 0.75
                    decay_q = 4
                sust_q = int(round(sustain_lin * 10.0))

                pan = 8
                if res.get("Hl") is not None:
                    hl = float(res["Hl"][j, s0 + a : s0 + b].sum())
                    hr = float(res["Hr"][j, s0 + a : s0 + b].sum())
                    pan = int(np.clip(round(16.0 * np.arctan2(hr, hl) / (0.5 * np.pi)), 1, 15))

                # [inst, midiIdx, startFrame, durFrames, peakDbQ, atkFrames, decayQ, sustainQ, pan16]
                notes_out.append([inst_id, int(j), s0 + a, L, pk_q, atk, decay_q, sust_q, pan])

    return notes_out


def analyze_poly_dual(guitar_st: np.ndarray, keys_st: np.ndarray, iters: int = 360) -> dict:
    mix_mono = (0.5 * (guitar_st.mean(1) + keys_st.mean(1))).astype(np.float32)
    tuning = float(librosa.estimate_tuning(y=mix_mono[::4], sr=SR // 4))

    g_mono = guitar_st.mean(1).astype(np.float32)
    k_mono = keys_st.mean(1).astype(np.float32)
    g_res = _nmf_instrument(g_mono, guitar_st, tuning, iters=iters, seed=1)
    k_res = _nmf_instrument(k_mono, keys_st, tuning, iters=iters, seed=2)

    ref_g = float(max(g_res["Hnotes"].max(), k_res["Hnotes"].max(), 1e-9))
    g_notes = _extract_discrete_notes(g_res, inst_id=0, ref_global=ref_g, floor_db=-31.0, min_dur=3)
    k_notes = _extract_discrete_notes(k_res, inst_id=1, ref_global=ref_g, floor_db=-31.0, min_dur=4)

    all_notes = g_notes + k_notes
    all_notes.sort(key=lambda r: (r[2], r[0], r[1]))
    # Delta-encode startFrame so gzip compresses musical timing cleanly
    prev_s = 0
    for r in all_notes:
        s = r[2]
        r[2] = s - prev_s
        prev_s = s

    model = {
        "hop": HOP,
        "tuning": round(tuning, 4),
        "midiLo": MIDI_LO,
        "nHarm": N_HARM,
        "qDb": 1.0,
        "guitarHarmDbQ": np.round(np.maximum(g_res["harmDb"], -60.0) * 2.0).astype(int).tolist(),
        "keysHarmDbQ": np.round(np.maximum(k_res["harmDb"], -60.0) * 2.0).astype(int).tolist(),
    }
    score = {"notes": all_notes}
    stats = {"guitarNotes": len(g_notes), "keysNotes": len(k_notes), "totalNotes": len(all_notes)}
    return {"model": model, "score": score, "stats": stats}
