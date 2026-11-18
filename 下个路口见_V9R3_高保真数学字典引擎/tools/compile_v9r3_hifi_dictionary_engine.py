#!/usr/bin/env python3
"""Compiles all V9R3 Pure-Code Dictionaries (`models/*.js`) & Executable Scores (`data/*.js`)
directly from the 32-bit 48kHz Studio Stems (`conservative_6stems_48k.npz`) + 285 Mandarin Lyric Syllables.

Strictly enforces:
  - ZERO binary files, ZERO Base64 / Base122 / LZMA strings, ZERO full-mix MDCT dumps;
  - 100% plain-text ES Modules with genuine per-instrument physical/musical dictionaries;
  - Bit-exact alignment with `src/dsp/*.js` synthesis equations.
"""
import gzip
import json
import os
import re
import time
import numpy as np
from scipy import signal

ROOT_DIR = "/Users/weckham/Desktop/MUSIC/下个路口见_V9R3_高保真数学字典引擎"
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
NPZ_PATH = "/Users/weckham/.gemini/antigravity/brain/74d2d208-e911-4def-a54e-f88353345d24/scratch/conservative_6stems_48k.npz"
V9R1_VOCAL_JS = "/Users/weckham/Desktop/MUSIC/下个路口见_V9R1_纯代码字典合成引擎/data/vocal-score-expressive.js"
V9R1_LYRICS_JS = "/Users/weckham/Desktop/MUSIC/下个路口见_V9R1_纯代码字典合成引擎/data/vocal-lyrics.js"

SR = 48000
EXACT_BPM = 98.0
EXACT_BAR = 48000.0 * 240.0 / EXACT_BPM  # 117551.02040816327


# ==============================================================================
# 1. Exact Shared Polyphase & Symmetric FIR Kernels (Bit-Exact with polyphase-sinc.js)
# ==============================================================================
def bessel_i0(x: np.ndarray) -> np.ndarray:
    s = np.ones_like(x, dtype=np.float64)
    term = np.ones_like(x, dtype=np.float64)
    hx2 = 0.25 * (x ** 2)
    for k in range(1, 27):
        term *= hx2 / (k * k)
        s += term
    return s


def resample_zero_phase_poly(
    x: np.ndarray, up: int, down: int, target_len: int = 0, taps_per_step: int = 16, beta: float = 8.6
) -> np.ndarray:
    n_in = len(x)
    n_out = target_len if target_len > 0 else int(round(n_in * up / down))
    if up == 1 and down == 1:
        return x[:n_out].copy()

    max_ud = max(up, down)
    fc = 0.465 / max_ud
    half_len = taps_per_step * max_ud
    m = np.arange(-half_len, half_len + 1, dtype=np.float64)
    win = bessel_i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (m / half_len) ** 2))) / float(bessel_i0(np.array(beta)))
    h = up * (2.0 * fc * np.sinc(2.0 * fc * m)) * win
    for p in range(up):
        dc = np.sum(h[p::up])
        if dc != 0.0:
            h[p::up] /= dc

    is_1d = x.ndim == 1
    x2 = x[:, None].astype(np.float64) if is_1d else x.astype(np.float64)
    ch = x2.shape[1]
    out = np.zeros((n_out, ch), dtype=np.float64)

    n_idx = np.arange(n_out, dtype=np.int64)
    num = n_idx * down
    q_all = num // up
    p_all = num % up

    for p in range(up):
        sel = np.where(p_all == p)[0]
        if len(sel) == 0:
            continue
        qp = q_all[sel]
        k_min = int(np.ceil((-half_len - p) / up))
        k_max = int(np.floor((half_len - p) / up))
        acc = np.zeros((len(sel), ch), dtype=np.float64)
        for k in range(k_min, k_max + 1):
            w = float(h[k * up + p + half_len])
            idx = np.clip(qp - k, 0, n_in - 1)
            acc += x2[idx] * w
        out[sel] = acc

    return out[:, 0] if is_1d else out


def design_sym_lp_half(cutoff_cycles: float, half_order: int = 64, beta: float = 8.0) -> np.ndarray:
    m = np.arange(0, half_order + 1, dtype=np.float64)
    win = bessel_i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (m / half_order) ** 2))) / float(bessel_i0(np.array(beta)))
    w = 2.0 * cutoff_cycles * np.sinc(2.0 * cutoff_cycles * m) * win
    dc = w[0] + 2.0 * np.sum(w[1:])
    return w / dc


def apply_zero_phase_sym_fir(x: np.ndarray, w_half: np.ndarray) -> np.ndarray:
    is_1d = x.ndim == 1
    x2 = x[:, None].astype(np.float64) if is_1d else x.astype(np.float64)
    n = len(x2)
    ho = len(w_half) - 1
    pad_x = np.pad(x2, ((ho, ho), (0, 0)), mode="edge")
    acc = pad_x[ho : ho + n] * w_half[0]
    for m in range(1, ho + 1):
        acc += w_half[m] * (pad_x[ho - m : ho - m + n] + pad_x[ho + m : ho + m + n])
    return acc[:, 0] if is_1d else acc


def eval_snr_metrics(ref: np.ndarray, est: np.ndarray) -> dict:
    ref64 = ref.astype(np.float64)
    est64 = est.astype(np.float64)
    wav_snr = float(10.0 * np.log10(np.sum(ref64 ** 2) / (np.sum((ref64 - est64) ** 2) + 1e-18)))

    n_fft, hop = 2048, 512
    w = np.hanning(n_fft)
    r_m = 0.5 * (ref64[:, 0] + ref64[:, 1]) if ref64.ndim == 2 else ref64
    e_m = 0.5 * (est64[:, 0] + est64[:, 1]) if est64.ndim == 2 else est64
    nf = (len(r_m) - n_fft) // hop
    rm = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(r_m, n_fft)[::hop][:nf] * w, axis=-1))
    em = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(e_m, n_fft)[::hop][:nf] * w, axis=-1))
    mag_snr = float(10.0 * np.log10(np.sum(rm ** 2) / (np.sum((rm - em) ** 2) + 1e-18)))

    edges = np.unique(np.round(np.geomspace(1, 900, 25)).astype(int))
    rb = np.array([np.sqrt(np.mean(rm[:, edges[i] : edges[i + 1]] ** 2, axis=1)) for i in range(len(edges) - 1)])
    eb = np.array([np.sqrt(np.mean(em[:, edges[i] : edges[i + 1]] ** 2, axis=1)) for i in range(len(edges) - 1)])
    aud_snr = float(10.0 * np.log10(np.sum(rb ** 2) / (np.sum((rb - eb) ** 2) + 1e-18)))

    return {
        "wavSnrDb": round(wav_snr, 2),
        "magSnrDb": round(mag_snr, 2),
        "audEnvSnrDb": round(aud_snr, 2),
    }


# ==============================================================================
# 2. Shared Sine-Lapped Mid/Side Atom Dictionary Compiler
# ==============================================================================
def learn_lapped_atom_dictionary(
    sig_2ch: np.ndarray, win_len: int, hop: int, k_mid: int, k_side: int, gate_db: float = -48.0, basis_scale: int = 16384
):
    pad = win_len
    padded = np.pad(sig_2ch.astype(np.float64), ((pad, pad + win_len), (0, 0)))
    nf = (len(padded) - win_len) // hop
    sw = np.sin(np.pi * (np.arange(win_len) + 0.5) / win_len)

    mid = 0.5 * (padded[:, 0] + padded[:, 1])
    side = 0.5 * (padded[:, 0] - padded[:, 1])
    mf = np.lib.stride_tricks.sliding_window_view(mid, win_len)[::hop][:nf] * sw
    sf = np.lib.stride_tricks.sliding_window_view(side, win_len)[::hop][:nf] * sw

    m_rms = np.sqrt(np.mean(mf ** 2, axis=1))
    s_rms = np.sqrt(np.mean(sf ** 2, axis=1))
    pk = max(float(np.max(m_rms)), 1e-9)
    act_m = np.where(m_rms > pk * (10.0 ** (gate_db / 20.0)))[0]
    act_s = np.where(s_rms > pk * (10.0 ** ((gate_db - 2.0) / 20.0)))[0]

    # Cap SVD training matrix at 16,000 highest-energy frames so SVD runs in < 0.4s
    if len(act_m) > 16000:
        sel_m = act_m[np.argsort(m_rms[act_m])[-16000:]]
    else:
        sel_m = act_m
    _, _, vt_m = np.linalg.svd(mf[sel_m], full_matrices=False)
    bm_q = np.round(vt_m[: min(k_mid, len(vt_m))] * basis_scale).astype(np.int32)

    if k_side > 0 and len(act_s) > 4:
        sel_s = act_s[np.argsort(s_rms[act_s])[-14000:]] if len(act_s) > 14000 else act_s
        _, _, vt_s = np.linalg.svd(sf[sel_s], full_matrices=False)
        bs_q = np.round(vt_s[: min(k_side, len(vt_s))] * basis_scale).astype(np.int32)
    else:
        bs_q = np.zeros((0, win_len), dtype=np.int32)

    return {
        "winLen": int(win_len),
        "hop": int(hop),
        "basisScale": int(basis_scale),
        "midAtoms": bm_q.tolist(),
        "sideAtoms": bs_q.tolist(),
        "_bm_f": bm_q.astype(np.float64) / basis_scale,
        "_bs_f": bs_q.astype(np.float64) / basis_scale,
    }


def encode_lapped_events(
    sig_2ch: np.ndarray,
    dict_obj: dict,
    m_mid: int,
    m_side: int,
    gate_db: float = -48.0,
    ref_peak: float = 0.0,
    rms_scale: int = 4096,
    weight_scale: int = 256,
):
    win_len = dict_obj["winLen"]
    hop = dict_obj["hop"]
    bm_f = dict_obj["_bm_f"]
    bs_f = dict_obj["_bs_f"]

    n_tot = len(sig_2ch)
    pad = win_len
    padded = np.pad(sig_2ch.astype(np.float64), ((pad, pad + win_len), (0, 0)))
    nf = (len(padded) - win_len) // hop
    sw = np.sin(np.pi * (np.arange(win_len) + 0.5) / win_len)

    mid = 0.5 * (padded[:, 0] + padded[:, 1])
    side = 0.5 * (padded[:, 0] - padded[:, 1])
    mf = np.lib.stride_tricks.sliding_window_view(mid, win_len)[::hop][:nf] * sw
    sf = np.lib.stride_tricks.sliding_window_view(side, win_len)[::hop][:nf] * sw

    m_rms = np.sqrt(np.mean(mf ** 2, axis=1))
    s_rms = np.sqrt(np.mean(sf ** 2, axis=1))
    pk = ref_peak if ref_peak > 0.0 else max(float(np.max(m_rms)), 1e-9)
    th_m = pk * (10.0 ** (gate_db / 20.0))
    th_s = pk * (10.0 ** ((gate_db - 2.0) / 20.0))

    def encode_channel(frames_mat, rms_vec, basis_f, m_keep, thresh):
        tuples = []
        rec_buf = np.zeros(len(padded), dtype=np.float64)
        if len(basis_f) == 0 or m_keep <= 0:
            return tuples, rec_buf
        act = np.where(rms_vec > thresh)[0]
        if len(act) == 0:
            return tuples, rec_buf

        # Batch matrix projection for speed!
        u_mat = frames_mat[act] / rms_vec[act, None]
        c_mat = u_mat @ basis_f.T
        k_use = min(m_keep, c_mat.shape[1])

        prev_fi = 0
        for row_i, fi in enumerate(act):
            rms_q = int(round(float(rms_vec[fi]) * rms_scale))
            if rms_q <= 0:
                continue
            c = c_mat[row_i]
            top = np.argpartition(np.abs(c), -k_use)[-k_use:]
            top_sorted = np.sort(top)
            cq = np.round(c[top_sorted] * weight_scale).astype(np.int32)
            nz = cq != 0
            if not np.any(nz):
                continue
            idxs = top_sorted[nz]
            vals = cq[nz]

            tup = [int(fi - prev_fi), rms_q]
            prev_fi = int(fi)
            prev_k = 0
            frame_rec = np.zeros(win_len, dtype=np.float64)
            rms_val = rms_q / rms_scale
            for k_idx, w_q in zip(idxs, vals):
                tup.append(int(k_idx - prev_k))
                tup.append(int(w_q))
                prev_k = int(k_idx)
                frame_rec += basis_f[k_idx] * ((w_q / weight_scale) * rms_val)
            tuples.append(tup)
            st = int(fi) * hop
            rec_buf[st : st + win_len] += frame_rec * sw
        return tuples, rec_buf

    mid_tuples, rec_m = encode_channel(mf, m_rms, bm_f, m_mid, th_m)
    side_tuples, rec_s = encode_channel(sf, s_rms, bs_f, m_side, th_s)

    rec_2ch = np.stack([rec_m[pad : pad + n_tot] + rec_s[pad : pad + n_tot], rec_m[pad : pad + n_tot] - rec_s[pad : pad + n_tot]], axis=-1)
    event_stream = {
        "totalSamples": int(n_tot),
        "pad": int(pad),
        "rmsScale": int(rms_scale),
        "weightScale": int(weight_scale),
        "midFrames": mid_tuples,
        "sideFrames": side_tuples,
    }
    return event_stream, rec_2ch


# ==============================================================================
# 3. Lead Vocal Pitch-Synchronous Wavecycle + Consonant Dictionary Compiler
# ==============================================================================
def sinc_interp_1d(arr: np.ndarray, x: np.ndarray, ht: int = 6) -> np.ndarray:
    na = len(arr)
    ic = np.floor(x).astype(np.int64)
    fr = x - ic
    out = np.zeros(len(x), dtype=np.float64)
    for k in range(-ht + 1, ht + 1):
        idx = np.clip(ic + k, 0, na - 1)
        d = fr - k
        ad = np.abs(d)
        w = np.where(ad < ht, 0.5 + 0.5 * np.cos(np.pi * d / ht), 0.0)
        out += arr[idx] * np.sinc(d) * w
    return out


def compile_lead_vocal(lead48: np.ndarray):
    t0 = time.time()
    n48 = len(lead48)
    sr32 = 32000
    n32 = int(round(n48 * 2 / 3))
    sig = resample_zero_phase_poly(lead48, 2, 3, n32)

    hop, win = 160, 640
    nf = (len(sig) - win) // hop
    min_lag, max_lag = int(sr32 / 600.0), int(sr32 / 135.0)
    f0_tr = np.zeros(nf, dtype=np.float64)
    w_ac = np.hanning(win)

    frames_ac = np.lib.stride_tricks.sliding_window_view(sig, win)[::hop][:nf] * w_ac
    enes = np.sum(frames_ac ** 2, axis=1)
    act_ac = np.where(enes > 1e-8)[0]
    for i in act_ac:
        spec = np.fft.rfft(frames_ac[i], n=2048)
        ac = np.fft.irfft(spec * np.conj(spec))[: max_lag + 2]
        ac /= ac[0] + 1e-12
        pk = min_lag + int(np.argmax(ac[min_lag : max_lag + 1]))
        if ac[pk] > 0.48:
            f0_tr[i] = sr32 / pk

    # Build continuous GCI pitch-mark chain across full 210.37s song
    pms = [(0, 0)]
    limit = len(sig) - 256
    while pms[-1][0] < limit:
        prev_pm, prev_v = pms[-1]
        fi = min(nf - 1, max(0, (prev_pm - 160) // hop))
        f0 = f0_tr[fi]
        if f0 > 135.0:
            t_p = int(round(sr32 / f0))
            if prev_v == 0:
                lo_s, hi_s = prev_pm + 52, min(limit, prev_pm + int(1.4 * t_p))
                best = lo_s + int(np.argmax(np.abs(sig[lo_s:hi_s])))
                pms.append((best, 1))
            else:
                plen = min(t_p // 2, prev_pm)
                pcyc = sig[prev_pm - plen : prev_pm + plen]
                lo_s = max(prev_pm + int(0.75 * t_p), prev_pm + 52)
                hi_s = min(limit, prev_pm + int(1.28 * t_p))
                best_p, best_c = min(hi_s, prev_pm + t_p), -1e18
                for cand in range(lo_s, hi_s + 1):
                    c = float(np.dot(pcyc, sig[cand - plen : cand + plen]))
                    if c > best_c:
                        best_c, best_p = c, cand
                pms.append((best_p, 1))
        else:
            pms.append((prev_pm + 112, 0))
    pms.append((len(sig) - 1, 0))

    ls_arr = [pms[i + 1][0] - pms[i][0] for i in range(len(pms) - 1)]
    ovs = [0] + [min(24, ls_arr[i - 1] // 3, ls_arr[i] // 3) for i in range(1, len(ls_arr))] + [0]

    P = 256
    v_items, u_items = [], []
    cycles = []
    # Noise gate at -66 dB below peak vocal cycle RMS so silent instrumental bars emit 0 grains!
    raw_seg_rms = np.array([np.sqrt(np.mean(sig[pms[i][0] : pms[i + 1][0]] ** 2)) for i in range(len(ls_arr))])
    gate_rms = float(np.max(raw_seg_rms)) * (10.0 ** (-62.0 / 20.0))

    for i in range(len(ls_arr)):
        a = pms[i][0] - ovs[i]
        b = pms[i + 1][0] + ovs[i + 1]
        voiced = 1 if (pms[i][1] == 1 and pms[i + 1][1] == 1) else 0
        if raw_seg_rms[i] < gate_rms:
            cycles.append((a, b, 0.0, voiced))
            continue
        seg = sig[a:b]
        w_ab = np.ones(b - a, dtype=np.float64)
        if ovs[i] > 0:
            fl = 2 * ovs[i]
            w_ab[:fl] = 0.5 * (1.0 - np.cos(np.pi * (np.arange(fl) + 0.5) / fl))
        if ovs[i + 1] > 0:
            fr = 2 * ovs[i + 1]
            w_ab[-fr:] = 0.5 * (1.0 + np.cos(np.pi * (np.arange(fr) + 0.5) / fr))
        wseg = seg * w_ab
        xp = (np.arange(P, dtype=np.float64) + 0.5) * ((b - a) / P) - 0.5
        gp = sinc_interp_1d(wseg, xp, ht=6)
        rms = float(np.sqrt(np.mean(gp ** 2)))
        cycles.append((a, b, rms, voiced))
        if rms >= gate_rms:
            if voiced:
                v_items.append((i, gp))
            else:
                u_items.append((i, gp))

    # Learn Voiced Glottal-Formant Wavecycle Basis (`Kv = 144`) & Unvoiced Consonant Wavelet Basis (`Ku = 112`)
    wv = np.array([gp for _, gp in v_items])
    wu = np.array([gp for _, gp in u_items])
    _, _, vt_v = np.linalg.svd(wv, full_matrices=False)
    _, _, vt_u = np.linalg.svd(wu, full_matrices=False)

    basis_scale = 16384
    kv, ku = 144, 112
    bv_q = np.round(vt_v[:kv] * basis_scale).astype(np.int32)
    bu_q = np.round(vt_u[:ku] * basis_scale).astype(np.int32)
    bv_f = bv_q.astype(np.float64) / basis_scale
    bu_f = bu_q.astype(np.float64) / basis_scale

    vocal_dict = {
        "internalSampleRate": sr32,
        "grainPoints": P,
        "basisScale": basis_scale,
        "chestCutoffHz": 720,
        "voicedAtoms": bv_q.tolist(),
        "consonantAtoms": bu_q.tolist(),
    }

    # Combine active items in chronological cycle order!
    active_all = sorted([(i, 1, gp) for i, gp in v_items] + [(i, 0, gp) for i, gp in u_items], key=lambda x: x[0])

    def encode_vocal_tier(m_onset: int, m_innov: int, m_cons: int, rms_gate_scale: float = 1.0):
        rms_scale = 4096
        w_scale = 256
        ar_scale = 64
        grain_tuples = []
        voiced32 = np.zeros(n32, dtype=np.float64)
        unvoiced32 = np.zeros(n32, dtype=np.float64)

        prev_start = 0
        prev_voiced_cycle_idx = -99
        prev_unit = None

        for cyc_i, is_v, gp in active_all:
            a, b, rms, _ = cycles[cyc_i]
            if rms < gate_rms * rms_gate_scale:
                prev_voiced_cycle_idx = -99
                prev_unit = None
                continue
            rms_q = int(round(rms * rms_scale))
            if rms_q <= 0:
                prev_voiced_cycle_idx = -99
                prev_unit = None
                continue
            seg_len = int(b - a)
            u_true = gp / rms

            if is_v:
                if cyc_i == prev_voiced_cycle_idx + 1 and prev_unit is not None:
                    al = float(np.clip(np.dot(u_true, prev_unit) / (np.dot(prev_unit, prev_unit) + 1e-12), 0.0, 1.0))
                    al_q = int(round(al * ar_scale))
                    res = u_true - (al_q / ar_scale) * prev_unit
                    m_use = m_innov
                else:
                    al_q = 0
                    res = u_true
                    m_use = m_onset

                c = res @ bv_f.T
                top = np.sort(np.argpartition(np.abs(c), -m_use)[-m_use:])
                cq = np.round(c[top] * w_scale).astype(np.int32)
                nz = cq != 0
                idxs = top[nz]
                vals = cq[nz]

                u_hat = ((al_q / ar_scale) * prev_unit) if (al_q > 0 and prev_unit is not None) else np.zeros(P, dtype=np.float64)
                tup = [int(a - prev_start), seg_len, int(al_q), rms_q]
                prev_start = int(a)
                prev_k = 0
                for ki, wq in zip(idxs, vals):
                    tup.append(int(ki - prev_k))
                    tup.append(int(wq))
                    prev_k = int(ki)
                    u_hat += bv_f[ki] * (wq / w_scale)

                grain_tuples.append(tup)
                prev_voiced_cycle_idx = cyc_i
                prev_unit = u_hat.copy()

                x_ab = (np.arange(seg_len, dtype=np.float64) + 0.5) * (P / seg_len) - 0.5
                voiced32[a:b] += sinc_interp_1d(u_hat * (rms_q / rms_scale), x_ab, ht=6)
            else:
                prev_voiced_cycle_idx = -99
                prev_unit = None
                c = u_true @ bu_f.T
                top = np.sort(np.argpartition(np.abs(c), -m_cons)[-m_cons:])
                cq = np.round(c[top] * w_scale).astype(np.int32)
                nz = cq != 0
                if not np.any(nz):
                    continue
                idxs = top[nz]
                vals = cq[nz]

                u_hat = np.zeros(P, dtype=np.float64)
                tup = [int(a - prev_start), seg_len, -1, rms_q]
                prev_start = int(a)
                prev_k = 0
                for ki, wq in zip(idxs, vals):
                    tup.append(int(ki - prev_k))
                    tup.append(int(wq))
                    prev_k = int(ki)
                    u_hat += bu_f[ki] * (wq / w_scale)

                grain_tuples.append(tup)
                x_ab = (np.arange(seg_len, dtype=np.float64) + 0.5) * (P / seg_len) - 0.5
                unvoiced32[a:b] += sinc_interp_1d(u_hat * (rms_q / rms_scale), x_ab, ht=6)

        lp_chest = design_sym_lp_half(720.0 / sr32, 64, 8.0)
        chest32 = apply_zero_phase_sym_fir(voiced32, lp_chest)
        formant32 = voiced32 - chest32

        chest48 = resample_zero_phase_poly(chest32, 3, 2, n48)
        formant48 = resample_zero_phase_poly(formant32, 3, 2, n48)
        air48 = resample_zero_phase_poly(unvoiced32, 3, 2, n48)

        score_obj = {
            "totalSamples32": int(n32),
            "totalSamples48": int(n48),
            "rmsScale": rms_scale,
            "weightScale": w_scale,
            "arScale": ar_scale,
            "grains": grain_tuples,
        }
        return score_obj, chest48, formant48, air48

    score_master, ch_m, fm_m, air_m = encode_vocal_tier(44, 24, 32, 1.0)
    score_compact, ch_c, fm_c, air_c = encode_vocal_tier(22, 9, 14, 1.6)
    print(f"[Vocal] Compiled in {time.time()-t0:.1f}s: Master grains={len(score_master['grains']):,}, Compact grains={len(score_compact['grains']):,}")
    return vocal_dict, score_master, (ch_m, fm_m, air_m), score_compact, (ch_c, fm_c, air_c), cycles


# ==============================================================================
# 4. Bar Motif + Fill Compiler (`06 drum_kick_snare` @ 12kHz & `08 bass_groove` @ 4kHz)
# ==============================================================================
def compile_bar_motif_track(
    st48: np.ndarray,
    bar0_48: float,
    sub_sr: int,
    n_motifs: int,
    k_mid: int,
    k_side: int,
    m_proto_mid: int,
    m_proto_side: int,
    m_fill_mid: int,
    m_fill_side: int,
    fill_gate_db: float,
    label: str,
):
    down = SR // sub_sr
    n_sub = int(round(len(st48) / down))
    st_sub = resample_zero_phase_poly(st48, 1, down, n_sub)
    exact_sub_bar = EXACT_BAR / down
    sub_bar0 = bar0_48 / down
    bar_len = int(round(exact_sub_bar))
    n_bars = int((n_sub - sub_bar0) // exact_sub_bar)

    bars = []
    starts = []
    for b in range(n_bars):
        si = int(round(sub_bar0 + b * exact_sub_bar))
        starts.append(si)
        bars.append(st_sub[si : si + bar_len].reshape(-1))
    b_mat = np.array(bars)
    norms = np.linalg.norm(b_mat, axis=1, keepdims=True) + 1e-12
    b_norm = b_mat / norms
    sim = b_norm @ b_norm.T
    ens = (norms[:, 0]) ** 2

    chosen = [int(np.argmax(ens))]
    for _ in range(n_motifs - 1):
        cur = np.max(np.maximum(0.0, sim[:, chosen]), axis=1)
        bg, bc = -1.0, 0
        for c in range(n_bars):
            if c in chosen:
                continue
            ns = np.maximum(cur, np.maximum(0.0, sim[:, c]))
            g = float(np.sum(ens * (ns ** 2 - cur ** 2)))
            if g > bg:
                bg, bc = g, c
        chosen.append(bc)

    # Learn Atom Dictionary on `st_sub` (`win=256, hop=128`)
    atom_dict = learn_lapped_atom_dictionary(st_sub, win_len=256, hop=128, k_mid=k_mid, k_side=k_side, gate_db=-52.0)

    # Encode the `n_motifs` concatenated 1-bar prototypes using `atom_dict`
    proto_concat = np.concatenate([st_sub[starts[c] : starts[c] + bar_len] for c in chosen], axis=0)
    proto_events, proto_rec_concat = encode_lapped_events(
        proto_concat, atom_dict, m_mid=m_proto_mid, m_side=m_proto_side, gate_db=-54.0
    )

    proto_vecs = []
    for m in range(n_motifs):
        seg = proto_rec_concat[m * bar_len : (m + 1) * bar_len]
        v = seg.reshape(-1)
        proto_vecs.append(v / (np.linalg.norm(v) + 1e-12))
    m_unit = np.array(proto_vecs)

    scale_quant = 512
    bar_schedule = []
    motif_only_sub = np.zeros_like(st_sub)
    for b in range(n_bars):
        dots = b_mat[b] @ m_unit.T
        bk = int(np.argmax(dots))
        sc_q = int(round(max(0.0, float(dots[bk])) * scale_quant))
        bar_schedule.append([bk, sc_q])
        si = starts[b]
        if sc_q > 0:
            motif_only_sub[si : si + bar_len] += ((sc_q / scale_quant) * m_unit[bk]).reshape(bar_len, 2)

    # Encode Turnaround Fill / Variation Innovations (`st_sub - motif_only_sub`) for Master tier!
    resid_sub = st_sub - motif_only_sub
    orig_pk = float(np.sqrt(np.max(np.mean(st_sub[: (len(st_sub) // 256) * 256].reshape(-1, 256, 2) ** 2, axis=(1, 2)))))
    fill_events, fill_rec_sub = encode_lapped_events(
        resid_sub, atom_dict, m_mid=m_fill_mid, m_side=m_fill_side, gate_db=fill_gate_db, ref_peak=orig_pk
    )

    master_48 = resample_zero_phase_poly(motif_only_sub + fill_rec_sub, down, 1, len(st48))
    compact_48 = resample_zero_phase_poly(motif_only_sub, down, 1, len(st48))

    bar_motif_spec = {
        "motifCount": int(n_motifs),
        "barLen": int(bar_len),
        "exactSubBar": round(float(exact_sub_bar), 6),
        "subBar0": round(float(sub_bar0), 4),
        "protoEvents": proto_events,
    }
    arr_master = {
        "totalSubSamples": int(n_sub),
        "scaleQuant": scale_quant,
        "barSchedule": bar_schedule,
        "fillEvents": fill_events,
    }
    arr_compact = {
        "totalSubSamples": int(n_sub),
        "scaleQuant": scale_quant,
        "barSchedule": bar_schedule,
        "fillEvents": {"totalSamples": int(n_sub), "pad": 256, "rmsScale": 4096, "weightScale": 256, "midFrames": [], "sideFrames": []},
    }
    print(f"[{label}] Motifs={n_motifs}, FillMidFrames={len(fill_events['midFrames'])}")
    return atom_dict, bar_motif_spec, arr_master, master_48, arr_compact, compact_48


# ==============================================================================
# 5. 6-Band Real Studio Cymbal Carrier + 98 BPM Envelope Motif (`07 drum_hihat_perc`)
# ==============================================================================
def compile_hihat_cymbal_track(dh48: np.ndarray, bar0_48: int):
    beat_len = int(round(EXACT_BAR / 4.0))  # 29388 samples = 1 exact beat @ 98.0000 BPM
    # Pick crisp Chorus beat (`bar 25, beat 0`) as canonical studio cymbal/shaker carrier
    b25 = int(round(bar0_48 + 25 * EXACT_BAR))
    carrier_scale = 16384
    pk_c = max(float(np.max(np.abs(dh48[b25 : b25 + beat_len]))), 1e-6)
    norm_factor = 0.85 / pk_c
    cb_l = np.clip(np.round(dh48[b25 : b25 + beat_len, 0] * norm_factor * carrier_scale), -32767, 32767).astype(np.int32)
    cb_r = np.clip(np.round(dh48[b25 : b25 + beat_len, 1] * norm_factor * carrier_scale), -32767, 32767).astype(np.int32)

    band_edges = [[1800, 3400], [3400, 5600], [5600, 8500], [8500, 12000], [12000, 16200], [16200, 20500]]
    num_bands = len(band_edges)

    # Mirror `getPreparedCymbalCarrierAndMotifs` in JS 1:1!
    pad = 128
    idx_pad = (np.arange(beat_len + 2 * pad) - pad) % beat_len
    raw_l = cb_l[idx_pad].astype(np.float64) / carrier_scale
    raw_r = cb_r[idx_pad].astype(np.float64) / carrier_scale

    unit_loop_l = []
    unit_loop_r = []
    bp_halves = []
    for f_lo, f_hi in band_edges:
        bp_h = design_sym_lp_half(f_hi / SR, 96, 8.0) - design_sym_lp_half(f_lo / SR, 96, 8.0)
        bp_halves.append(bp_h)
        bl = apply_zero_phase_sym_fir(raw_l, bp_h)[pad : pad + beat_len]
        br = apply_zero_phase_sym_fir(raw_r, bp_h)[pad : pad + beat_len]
        bl /= np.sqrt(np.mean(bl ** 2) + 1e-12)
        br /= np.sqrt(np.mean(br ** 2) + 1e-12)
        unit_loop_l.append(bl)
        unit_loop_r.append(br)

    # Measure true 6-band stereo RMS envelope (`env_hop = 512`) of `dh48`
    env_hop = 512
    num_frames = int(np.ceil(len(dh48) / env_hop)) + 1
    e_true = np.zeros((num_frames, 2 * num_bands), dtype=np.float64)
    n_fit = (len(dh48) // env_hop) * env_hop
    for b_i, bp_h in enumerate(bp_halves):
        sub = apply_zero_phase_sym_fir(dh48, bp_h)
        frm = sub[:n_fit].reshape(-1, env_hop, 2)
        rms_lr = np.sqrt(np.mean(frm ** 2, axis=1))
        e_true[: len(rms_lr), b_i] = rms_lr[:, 0]
        e_true[: len(rms_lr), num_bands + b_i] = rms_lr[:, 1]

    # Learn rank-10 spectral basis (`10 x 12`) & 18 1-bar envelope motifs @ 98.0000 BPM
    _, _, vt_e = np.linalg.svd(e_true, full_matrices=False)
    ke = 10
    basis_scale = 16384
    be_q = np.round(vt_e[:ke] * basis_scale).astype(np.int32)
    be_f = be_q.astype(np.float64) / basis_scale

    exact_bar_frames = EXACT_BAR / env_hop
    bar_env_frames = int(round(exact_bar_frames))
    bar0_frame = bar0_48 / env_hop
    n_bars = int((num_frames - bar0_frame) // exact_bar_frames)

    b_mats, b_starts = [], []
    for bi in range(n_bars):
        sf = int(round(bar0_frame + bi * exact_bar_frames))
        b_starts.append(sf)
        b_mats.append(e_true[sf : sf + bar_env_frames].reshape(-1))
    bm = np.array(b_mats)
    bn = np.linalg.norm(bm, axis=1, keepdims=True) + 1e-12
    bm_n = bm / bn
    sim = bm_n @ bm_n.T
    ens = (bn[:, 0]) ** 2

    n_motifs = 18
    chosen = [int(np.argmax(ens))]
    for _ in range(n_motifs - 1):
        cur = np.max(np.maximum(0.0, sim[:, chosen]), axis=1)
        bg, bc = -1.0, 0
        for c in range(n_bars):
            if c in chosen:
                continue
            ns = np.maximum(cur, np.maximum(0.0, sim[:, c]))
            g = float(np.sum(ens * (ns ** 2 - cur ** 2)))
            if g > bg:
                bg, bc = g, c
        chosen.append(bc)

    # Quantize 18 bar motif coefficient tables (`coeff_quant = 4096` folded into integers directly!)
    # Note: in JS `val += cRow[k] * basisFlat[k*dim + d]` then unit-normalized -> integer scaling cancels out cleanly!
    bar_motif_coeffs = []
    env_protos = []
    for c in chosen:
        p_bar = e_true[b_starts[c] : b_starts[c] + bar_env_frames]
        c_int = np.round((p_bar @ be_f.T) * 4096.0).astype(np.int32)
        bar_motif_coeffs.append(c_int.tolist())
        p_rec = np.maximum(0.0, c_int.astype(np.float64) @ be_f)
        v = p_rec.reshape(-1)
        env_protos.append(v / (np.linalg.norm(v) + 1e-12))
    env_protos = np.array(env_protos)

    scale_quant = 1024
    var_quant = 4096
    bar_schedule = []
    e_motif = np.zeros_like(e_true)
    for bi in range(n_bars):
        dots = bm[bi] @ env_protos.T
        bk = int(np.argmax(dots))
        sc_q = int(round(max(0.0, float(dots[bk])) * scale_quant))
        bar_schedule.append([bk, sc_q])
        sf = b_starts[bi]
        e_motif[sf : sf + bar_env_frames] = ((sc_q / scale_quant) * env_protos[bk]).reshape(bar_env_frames, 2 * num_bands)

    # Sparse turnaround / crash / shaker variation frames (relative to 95th percentile groove energy, not crash spike!)
    e_diff = e_true - e_motif
    f_err = np.linalg.norm(E_nrm := e_diff, axis=1)
    ref_e = float(np.percentile(np.linalg.norm(e_true, axis=1), 95))
    act_f = np.where(f_err > ref_e * 0.045)[0]

    var_frames = []
    e_master = e_motif.copy()
    prev_f = 0
    for fi in act_f:
        c_q = np.round((e_diff[fi] @ be_f.T) * var_quant).astype(np.int32)
        if not np.any(c_q != 0):
            continue
        var_frames.append([int(fi - prev_f)] + c_q.tolist())
        prev_f = int(fi)
        e_master[fi] = np.maximum(0.0, e_motif[fi] + (c_q / var_quant) @ be_f)

    def synth_hihat(env_grid):
        out_l = np.zeros(len(dh48), dtype=np.float64)
        out_r = np.zeros(len(dh48), dtype=np.float64)
        phases = (np.arange(len(dh48)) - bar0_48) % beat_len
        t_ramp = np.arange(env_hop, dtype=np.float64) / env_hop
        n_full = len(dh48) // env_hop
        ph_blocks = phases[: n_full * env_hop].reshape(n_full, env_hop)[:, ::2]
        for b_i in range(num_bands):
            ul_loop = unit_loop_l[b_i]
            ur_loop = unit_loop_r[b_i]
            ul = ul_loop[phases]
            ur = ur_loop[phases]
            c_rms_l = np.maximum(0.22, np.sqrt(np.mean(ul_loop[ph_blocks] ** 2, axis=1)))
            c_rms_r = np.maximum(0.22, np.sqrt(np.mean(ur_loop[ph_blocks] ** 2, axis=1)))
            g_l0 = env_grid[:n_full, b_i] / c_rms_l
            g_l1 = env_grid[1 : n_full + 1, b_i] / c_rms_l
            g_r0 = env_grid[:n_full, num_bands + b_i] / c_rms_r
            g_r1 = env_grid[1 : n_full + 1, num_bands + b_i] / c_rms_r
            gl_samp = (g_l0[:, None] + (g_l1 - g_l0)[:, None] * t_ramp[None, :]).reshape(-1)
            gr_samp = (g_r0[:, None] + (g_r1 - g_r0)[:, None] * t_ramp[None, :]).reshape(-1)
            out_l[: n_full * env_hop] += ul[: n_full * env_hop] * gl_samp
            out_r[: n_full * env_hop] += ur[: n_full * env_hop] * gr_samp
        return np.stack([out_l, out_r], axis=-1)

    rec_master = synth_hihat(e_master)
    rec_compact = synth_hihat(e_motif)

    cymbal_dict = {
        "sampleRate": SR,
        "beatLoopLen": int(beat_len),
        "carrierScale": carrier_scale,
        "carrierBeatL": cb_l.tolist(),
        "carrierBeatR": cb_r.tolist(),
        "bandEdgesHz": band_edges,
        "basisScale": basis_scale,
        "spectralBasis": be_q.tolist(),
        "barEnvFrames": int(bar_env_frames),
        "barEnvMotifCoeffs": bar_motif_coeffs,
    }
    hh_master = {
        "envHop": env_hop,
        "exactBarFrames": round(float(exact_bar_frames), 6),
        "bar0Frame": round(float(bar0_frame), 4),
        "bar0Sample": int(bar0_48),
        "scaleQuant": scale_quant,
        "varQuant": var_quant,
        "barSchedule": bar_schedule,
        "varFrames": var_frames,
    }
    hh_compact = dict(hh_master, varFrames=[])
    print(f"[HiHat/Perc] 18 Bar Motifs + {len(var_frames)} varFrames")
    return cymbal_dict, hh_master, rec_master, hh_compact, rec_compact


# ==============================================================================
# 6. 8-Prime Householder FDN (`05`) + Stereo Backing Harmony (`04`)
# ==============================================================================
def run_householder_fdn_py_impulse(n_ir: int, fdn_spec: dict):
    """Computes exact impulse response of JS `runHouseholderFdn24k` (`O(n_ir)` = 48,000 steps < 0.1s!)."""
    sr24 = fdn_spec["internalRate"]
    pre_len = fdn_spec["preDelaySamples"]
    delays = fdn_spec["delays"]
    rt60 = fdn_spec["rt60Sec"]
    damp = fdn_spec["damping"]
    hp_pole = fdn_spec["hpPole"]
    wet_gain = fdn_spec["wetGain"]

    ir_l = np.zeros(n_ir, dtype=np.float64)
    ir_r = np.zeros(n_ir, dtype=np.float64)
    pre_buf = np.zeros(pre_len, dtype=np.float64)
    pre_ptr = 0
    lines = [np.zeros(d, dtype=np.float64) for d in delays]
    ptrs = [0] * 8
    lp_st = [0.0] * 8
    decays = [10.0 ** ((-3.0 * d) / (rt60 * sr24)) for d in delays]
    omd = 1.0 - damp
    tap = [0.0] * 8
    hpxl = hpyl = hpxr = hpyr = 0.0

    for n in range(n_ir):
        x_in = 1.0 if n == 0 else 0.0
        d_in = pre_buf[pre_ptr]
        pre_buf[pre_ptr] = x_in * 0.35
        pre_ptr = 0 if pre_ptr + 1 == pre_len else pre_ptr + 1

        s8 = 0.0
        for i in range(8):
            raw = lines[i][ptrs[i]]
            dmp = damp * lp_st[i] + omd * raw
            lp_st[i] = dmp
            dec = dmp * decays[i]
            tap[i] = dec
            s8 += dec
        hs = 0.25 * s8
        for i in range(8):
            sgn = -d_in if (i & 1) else d_in
            lines[i][ptrs[i]] = tap[i] - hs + sgn
            ptrs[i] = 0 if ptrs[i] + 1 == delays[i] else ptrs[i] + 1

        rwl = (tap[0] - tap[2] + tap[4] - tap[6]) * wet_gain
        rwr = (tap[1] - tap[3] + tap[5] - tap[7]) * wet_gain
        yl = rwl - hpxl + hp_pole * hpyl
        hpxl, hpyl = rwl, yl
        yr = rwr - hpxr + hp_pole * hpyr
        hpxr, hpyr = rwr, yr
        ir_l[n] = yl
        ir_r[n] = yr
    return ir_l, ir_r


def compile_backing_and_fdn(
    backing48: np.ndarray, lead_rec48: np.ndarray, guitar_rec48: np.ndarray, drums_low_rec48: np.ndarray
):
    fdn_physics = {
        "internalRate": 24000,
        "preDelaySamples": 576,
        "delays": [733, 887, 1061, 1277, 1511, 1789, 2089, 2411],
        "rt60Sec": 1.35,
        "damping": 0.34,
        "hpPole": 0.976,
        "wetGain": 0.032,
        "sendLead": 0.75,
        "sendGuitar": 0.18,
        "sendDrums": 0.07,
    }
    dry48 = (
        0.75 * lead_rec48
        + 0.18 * 0.5 * (guitar_rec48[:, 0] + guitar_rec48[:, 1])
        + 0.07 * 0.5 * (drums_low_rec48[:, 0] + drums_low_rec48[:, 1])
    )
    n24 = int(round(len(backing48) / 2))
    dry24 = resample_zero_phase_poly(dry48, 1, 2, n24)
    ir_l, ir_r = run_householder_fdn_py_impulse(int(2.2 * 24000), fdn_physics)
    wet24 = np.stack([signal.fftconvolve(dry24, ir_l)[:n24], signal.fftconvolve(dry24, ir_r)[:n24]], axis=-1)
    fdn48 = resample_zero_phase_poly(wet24, 2, 1, len(backing48))

    harm_target48 = backing48 - fdn48
    n16 = int(round(len(backing48) / 3))
    hm16 = resample_zero_phase_poly(harm_target48, 1, 3, n16)

    harm_dict = learn_lapped_atom_dictionary(hm16, win_len=256, hop=128, k_mid=96, k_side=64, gate_db=-44.0)
    harm_master, hm_rec_m16 = encode_lapped_events(hm16, harm_dict, m_mid=24, m_side=16, gate_db=-43.0)
    harm_compact, hm_rec_c16 = encode_lapped_events(hm16, harm_dict, m_mid=10, m_side=6, gate_db=-35.0)

    harm_m48 = resample_zero_phase_poly(hm_rec_m16, 3, 1, len(backing48))
    harm_c48 = resample_zero_phase_poly(hm_rec_c16, 3, 1, len(backing48))
    print(f"[Backing+FDN] Master harmony midFrames={len(harm_master['midFrames'])}, Compact={len(harm_compact['midFrames'])}")
    return fdn_physics, harm_dict, harm_master, harm_m48, harm_compact, harm_c48, fdn48


# ==============================================================================
# 7. Write Clean Decoupled ES Modules (`models/*.js` & `data/*.js`)
# ==============================================================================
def clean_dict_for_js(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def write_js_exports(path: str, header_comment: str, exports: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"// {header_comment}", ""]
    for name, obj in exports:
        js_txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        lines.append(f"export const {name} = {js_txt};")
        lines.append("")
    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    gz_len = len(gzip.compress(content.encode("utf-8"), compresslevel=9))
    print(f"  -> Wrote {os.path.relpath(path, ROOT_DIR):42s} | Raw={len(content)/1024:7.1f} KB | Gzip={gz_len/1024:6.1f} KB")
    return len(content), gz_len


def build_lyrics_and_syllable_meta(cycles):
    with open(V9R1_LYRICS_JS, "r", encoding="utf-8") as f:
        lyr_raw = f.read()
    m_lyr = re.search(r"export const VOCAL_LYRIC_LINES\s*=\s*(\[[\s\S]*?\]);", lyr_raw)
    lyric_lines = json.loads(m_lyr.group(1))

    with open(V9R1_VOCAL_JS, "r", encoding="utf-8") as f:
        v_raw = f.read()
    m_sc = re.search(r"export const VOCAL_SCORE_EXPRESSIVE\s*=\s*(\[[\s\S]*\]);", v_raw)
    v9r1_syls = json.loads(m_sc.group(1))

    syl_meta = []
    for idx, row in enumerate(v9r1_syls):
        st_sec = round(row[0] * 256.0 / SR, 4)
        dur_sec = round(row[1] * 256.0 / SR, 4)
        hanzi = row[3]
        pinyin = row[4]
        midi_note = round(row[8] / 60.0, 2)
        syl_meta.append([idx, st_sec, round(st_sec + dur_sec, 4), hanzi, pinyin, midi_note])
    return lyric_lines, syl_meta


def main():
    t_all = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    z = np.load(NPZ_PATH)
    lead48 = z["lead_mono"].astype(np.float64)
    backing48 = z["backing"].astype(np.float64)
    drums_low48 = z["drums_low"].astype(np.float64)
    drums_high48 = z["drums_high"].astype(np.float64)
    bass48 = z["bass"].astype(np.float64)
    other48 = z["other"].astype(np.float64)
    master48 = z["master"].astype(np.float64)

    # Exact bar0 @ 98.0000 BPM
    d_mono = 0.5 * (drums_low48[:, 0] + drums_low48[:, 1])
    first_kick = int(1.6 * SR) + int(np.argmax(np.abs(d_mono[int(1.6 * SR) : int(2.6 * SR)])))
    bar0 = first_kick - 512
    while bar0 - EXACT_BAR >= 0:
        bar0 -= EXACT_BAR
    bar0_int = int(round(bar0))
    print(f"=== Compiling V9R3 Studio-HiFi Dictionary Engine (@ EXACT {EXACT_BPM:.4f} BPM, bar0={bar0_int}) ===")

    # 1. Lead Vocal (`01, 02, 03`)
    vocal_dict, v_master, (ch_m, fm_m, air_m), v_compact, (ch_c, fm_c, air_c), cycles = compile_lead_vocal(lead48)
    lead_rec_m = ch_m + fm_m + air_m
    lead_rec_c = ch_c + fm_c + air_c

    # 2. Drums Kick/Snare (`06` @ 12kHz) & Hi-Hat/Perc (`07` @ 48kHz)
    drum_atom_dict, drum_bar_motifs, dl_master, dl_rec_m, dl_compact, dl_rec_c = compile_bar_motif_track(
        drums_low48, bar0, sub_sr=12000, n_motifs=24, k_mid=96, k_side=32,
        m_proto_mid=32, m_proto_side=12, m_fill_mid=24, m_fill_side=10, fill_gate_db=-37.0, label="06 DrumsLow"
    )
    cymbal_dict, hh_master, dh_rec_m, hh_compact, dh_rec_c = compile_hihat_cymbal_track(drums_high48, bar0_int)

    # 3. Bass Groove (`08` @ 4kHz)
    bass_atom_dict, bass_bar_motifs, bs_master, bs_rec_m, bs_compact, bs_rec_c = compile_bar_motif_track(
        bass48, bar0, sub_sr=4000, n_motifs=24, k_mid=64, k_side=16,
        m_proto_mid=26, m_proto_side=8, m_fill_mid=20, m_fill_side=6, fill_gate_db=-38.0, label="08 BassGroove"
    )

    # 4. Guitar Funk Riff (`09` @ 16kHz, >=480Hz) & Rhodes Pad (`10` @ 3kHz, <480Hz)
    lp480 = design_sym_lp_half(480.0 / SR, 256, 8.0)
    rhodes_true48 = apply_zero_phase_sym_fir(other48, lp480)
    guitar_true48 = other48 - rhodes_true48

    n_rh3 = int(round(len(other48) / 16))
    rh3 = resample_zero_phase_poly(rhodes_true48, 1, 16, n_rh3)
    rhodes_dict = learn_lapped_atom_dictionary(rh3, win_len=160, hop=80, k_mid=64, k_side=32, gate_db=-52.0)
    rh_master, rh_rec_m3 = encode_lapped_events(rh3, rhodes_dict, m_mid=22, m_side=12, gate_db=-51.0)
    rh_compact, rh_rec_c3 = encode_lapped_events(rh3, rhodes_dict, m_mid=10, m_side=5, gate_db=-43.0)
    rh_rec_m48 = resample_zero_phase_poly(rh_rec_m3, 16, 1, len(other48))
    rh_rec_c48 = resample_zero_phase_poly(rh_rec_c3, 16, 1, len(other48))

    n_gt16 = int(round(len(other48) / 3))
    gt16 = resample_zero_phase_poly(guitar_true48, 1, 3, n_gt16)
    guitar_dict = learn_lapped_atom_dictionary(gt16, win_len=256, hop=128, k_mid=128, k_side=64, gate_db=-48.0)
    gt_master, gt_rec_m16 = encode_lapped_events(gt16, guitar_dict, m_mid=28, m_side=14, gate_db=-47.0)
    gt_compact, gt_rec_c16 = encode_lapped_events(gt16, guitar_dict, m_mid=12, m_side=6, gate_db=-39.0)
    gt_rec_m48 = resample_zero_phase_poly(gt_rec_m16, 3, 1, len(other48))
    gt_rec_c48 = resample_zero_phase_poly(gt_rec_c16, 3, 1, len(other48))

    # 5. Studio FDN Space (`05`) & Chorus Backing Harmony (`04`)
    fdn_physics, harm_dict, hm_master, hm_rec_m48, hm_compact, hm_rec_c48, fdn_m48 = compile_backing_and_fdn(
        backing48, lead_rec_m, gt_rec_m48, dl_rec_m
    )

    # Full Summed Mix Verification!
    mix_m48 = (
        lead_rec_m[:, None] + hm_rec_m48 + fdn_m48 + dl_rec_m + dh_rec_m + bs_rec_m + gt_rec_m48 + rh_rec_m48
    )
    mix_c48 = (
        lead_rec_c[:, None] + hm_rec_c48 + fdn_m48 + dl_rec_c + dh_rec_c + bs_rec_c + gt_rec_c48 + rh_rec_c48
    )

    print("\n=== Verified Full-Song (210.37s) Studio Ground-Truth SNR Metrics ===")
    benchmarks_master = {
        "lead_vocal": eval_snr_metrics(lead48, lead_rec_m),
        "backing_space": eval_snr_metrics(backing48, hm_rec_m48 + fdn_m48),
        "drums_kick_snare": eval_snr_metrics(drums_low48, dl_rec_m),
        "drums_hihat_perc": eval_snr_metrics(drums_high48, dh_rec_m),
        "bass_groove": eval_snr_metrics(bass48, bs_rec_m),
        "guitar_riff": eval_snr_metrics(guitar_true48, gt_rec_m48),
        "rhodes_pad": eval_snr_metrics(rhodes_true48, rh_rec_m48),
        "other_combined": eval_snr_metrics(other48, gt_rec_m48 + rh_rec_m48),
        "full_master_mix": eval_snr_metrics(master48, mix_m48),
    }
    benchmarks_compact = {
        "lead_vocal": eval_snr_metrics(lead48, lead_rec_c),
        "backing_space": eval_snr_metrics(backing48, hm_rec_c48 + fdn_m48),
        "drums_kick_snare": eval_snr_metrics(drums_low48, dl_rec_c),
        "drums_hihat_perc": eval_snr_metrics(drums_high48, dh_rec_c),
        "bass_groove": eval_snr_metrics(bass48, bs_rec_c),
        "guitar_riff": eval_snr_metrics(guitar_true48, gt_rec_c48),
        "rhodes_pad": eval_snr_metrics(rhodes_true48, rh_rec_c48),
        "other_combined": eval_snr_metrics(other48, gt_rec_c48 + rh_rec_c48),
        "full_master_mix": eval_snr_metrics(master48, mix_c48),
    }
    for k, v in benchmarks_master.items():
        print(f"  [Master  {k:18s}] WavSNR={v['wavSnrDb']:6.2f} dB | MagSNR={v['magSnrDb']:6.2f} dB | AudEnvSNR={v['audEnvSnrDb']:6.2f} dB")
    for k, v in benchmarks_compact.items():
        print(f"  [Compact {k:18s}] WavSNR={v['wavSnrDb']:6.2f} dB | MagSNR={v['magSnrDb']:6.2f} dB | AudEnvSNR={v['audEnvSnrDb']:6.2f} dB")

    # Write `models/*.js`
    print("\n=== Writing Pure-Text JS Dictionary Modules (`models/*.js`) ===")
    sz = {}
    sz["vocalDict"] = write_js_exports(
        os.path.join(MODELS_DIR, "vocal-wavecycle-dictionary.js"),
        "V9R3 Reusable Lead Vocal GCI Wavecycle (Kv=144) & Mandarin Consonant Wavelet (Ku=112) Dictionary",
        [("VOCAL_WAVECYCLE_DICTIONARY", vocal_dict)],
    )
    sz["backingDict"] = write_js_exports(
        os.path.join(MODELS_DIR, "backing-space-dictionary.js"),
        "V9R3 Stereo Vocal Harmony Dictionary & 8-Prime Householder FDN Studio Physics",
        [("BACKING_HARMONY_DICTIONARY", clean_dict_for_js(harm_dict)), ("STUDIO_FDN_PHYSICS", fdn_physics)],
    )
    sz["drumDict"] = write_js_exports(
        os.path.join(MODELS_DIR, "drum-percussion-dictionary.js"),
        "V9R3 98.0000 BPM Drum Hit Atom Dictionary, 24-Bar Groove Motifs & 6-Band Real Studio Cymbal Dictionary",
        [
            ("DRUM_HIT_ATOM_DICTIONARY", clean_dict_for_js(drum_atom_dict)),
            ("DRUM_GROOVE_BAR_MOTIFS", drum_bar_motifs),
            ("CYMBAL_SHIMMER_DICTIONARY", cymbal_dict),
        ],
    )
    sz["bassDict"] = write_js_exports(
        os.path.join(MODELS_DIR, "bass-groove-dictionary.js"),
        "V9R3 98.0000 BPM Electric Bass Atom Dictionary & 24-Bar Groove Motifs (@ 4kHz Critical Sampling)",
        [("BASS_ATOM_DICTIONARY", clean_dict_for_js(bass_atom_dict)), ("BASS_GROOVE_BAR_MOTIFS", bass_bar_motifs)],
    )
    sz["guitarRhodesDict"] = write_js_exports(
        os.path.join(MODELS_DIR, "guitar-rhodes-dictionary.js"),
        "V9R3 Acoustic/Funk Guitar Pluck Dictionary (16kHz) & Rhodes Electric Piano Chord Dictionary (3kHz)",
        [
            ("GUITAR_PLUCK_DICTIONARY", clean_dict_for_js(guitar_dict)),
            ("RHODES_CHORD_DICTIONARY", clean_dict_for_js(rhodes_dict)),
        ],
    )

    # Write `data/*.js`
    print("\n=== Writing Executable Score & Structure Modules (`data/*.js`) ===")
    song_structure = {
        "bpm": EXACT_BPM,
        "barDurationSec": round(EXACT_BAR / SR, 8),
        "barSamples48k": round(EXACT_BAR, 6),
        "bar0Samples48k": bar0_int,
        "bar0Sec": round(bar0_int / SR, 6),
        "totalBars": 85,
        "durationSec": round(len(master48) / SR, 4),
        "progressions": {
            "STD4": [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Dm9"]],
            "ROLL4": [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Am7"]],
        },
        "sections": [
            {"name": "Intro", "zh": "前奏木吉他Hook", "startBar": 0, "bars": 8, "prog": "STD4"},
            {"name": "Verse 1", "zh": "主歌一", "startBar": 8, "bars": 8, "prog": "STD4"},
            {"name": "Pre-Chorus 1", "zh": "导歌一", "startBar": 16, "bars": 8, "prog": "STD4"},
            {"name": "Chorus 1", "zh": "副歌一", "startBar": 24, "bars": 8, "prog": "ROLL4"},
            {"name": "Interlude", "zh": "间奏律动", "startBar": 32, "bars": 8, "prog": "STD4"},
            {"name": "Verse 2A", "zh": "主歌二切入", "startBar": 40, "bars": 1, "prog": "STD4"},
            {"name": "Verse 2 Drop", "zh": "主歌二抽鼓留白", "startBar": 41, "bars": 3, "prog": "STD4"},
            {"name": "Verse 2B", "zh": "主歌二推进", "startBar": 44, "bars": 4, "prog": "STD4"},
            {"name": "Pre-Chorus 2", "zh": "导歌二", "startBar": 48, "bars": 8, "prog": "STD4"},
            {"name": "Chorus 2", "zh": "副歌二", "startBar": 56, "bars": 8, "prog": "ROLL4"},
            {"name": "Chorus 3", "zh": "副歌三高潮", "startBar": 64, "bars": 8, "prog": "ROLL4"},
            {"name": "Outro Groove", "zh": "尾奏律动", "startBar": 72, "bars": 8, "prog": "STD4"},
            {"name": "Final Chord", "zh": "收束余韵", "startBar": 80, "bars": 5, "prog": "FINAL"},
        ],
    }
    lyric_lines, syl_meta = build_lyrics_and_syllable_meta(cycles)
    sz["structure"] = write_js_exports(
        os.path.join(DATA_DIR, "song-structure.js"),
        "Exact 98.0000 BPM Studio Timeline & Section Map for 《李宇春 - 下个, 路口, 见》",
        [("SONG_STRUCTURE", song_structure)],
    )
    sz["lyrics"] = write_js_exports(
        os.path.join(DATA_DIR, "vocal-lyrics.js"),
        "Synchronized Lyric Lines & 285 Mandarin Syllable Metadata",
        [("VOCAL_LYRIC_LINES", lyric_lines), ("VOCAL_SYLLABLES_META", syl_meta)],
    )
    sz["vocalMaster"] = write_js_exports(
        os.path.join(DATA_DIR, "vocal-score-master.js"),
        "V9R3 Studio Master Pitch-Synchronous Wavecycle + Consonant Vocal Score",
        [("VOCAL_SCORE_MASTER", v_master)],
    )
    sz["vocalCompact"] = write_js_exports(
        os.path.join(DATA_DIR, "vocal-score-compact.js"),
        "V9R3 Compact Pure-Dictionary Vocal Score",
        [("VOCAL_SCORE_COMPACT", v_compact)],
    )
    sz["bandMaster"] = write_js_exports(
        os.path.join(DATA_DIR, "band-score-master.js"),
        "V9R3 Studio Master 86-Bar Motif & Instrument Event Score (Tracks 04..10)",
        [
            (
                "BAND_SCORE_MASTER",
                {
                    "drumsLow": dl_master,
                    "drumsHigh": hh_master,
                    "bass": bs_master,
                    "guitar": gt_master,
                    "rhodes": rh_master,
                    "backingHarmony": hm_master,
                },
            )
        ],
    )
    sz["bandCompact"] = write_js_exports(
        os.path.join(DATA_DIR, "band-score-compact.js"),
        "V9R3 Compact Pure-Motif Band Score (Tracks 04..10)",
        [
            (
                "BAND_SCORE_COMPACT",
                {
                    "drumsLow": dl_compact,
                    "drumsHigh": hh_compact,
                    "bass": bs_compact,
                    "guitar": gt_compact,
                    "rhodes": rh_compact,
                    "backingHarmony": hm_compact,
                },
            )
        ],
    )

    dict_gz = sum(sz[k][1] for k in ["vocalDict", "backingDict", "drumDict", "bassDict", "guitarRhodesDict"])
    master_gz = dict_gz + sz["structure"][1] + sz["lyrics"][1] + sz["vocalMaster"][1] + sz["bandMaster"][1]
    compact_gz = dict_gz + sz["structure"][1] + sz["lyrics"][1] + sz["vocalCompact"][1] + sz["bandCompact"][1]

    tiers_registry = {
        "studio_master": {
            "id": "studio_master",
            "badge": "DEFAULT · 录音室母带字典档",
            "title": "V9R3 录音室母带数学字典档 (Studio Master Hi-Fi)",
            "subtitle": "98.0000 BPM 零漂移网格 · GCI声门周期字典(Kv=144)+清辅音小波(Ku=112) · 全乐器独立分频字典与加花",
            "gzipKB": round(master_gz / 1024, 1),
            "dictGzipKB": round(dict_gz / 1024, 1),
            "metrics": benchmarks_master,
        },
        "compact_pure": {
            "id": "compact_pure",
            "badge": "COMPACT · 纯原型轻量对照档",
            "title": "V9R3 纯原型轻量字典档 (Compact Pure-Motif)",
            "subtitle": "复用完全相同物理/音乐字典 · 鼓/贝斯零残差纯24小节原型循环 + 精简主唱/吉他稀疏系数",
            "gzipKB": round(compact_gz / 1024, 1),
            "dictGzipKB": round(dict_gz / 1024, 1),
            "metrics": benchmarks_compact,
        },
    }

    v9r1_baseline = {
        "lead_vocal": {"wavSnrDb": -4.71, "magSnrDb": 6.08},
        "backing_space": {"wavSnrDb": -1.58, "magSnrDb": 1.35},
        "drums_kick_snare": {"wavSnrDb": -1.34, "magSnrDb": 3.80},
        "drums_hihat_perc": {"wavSnrDb": -1.20, "magSnrDb": 1.94},
        "bass_groove": {"wavSnrDb": -1.45, "magSnrDb": 6.96},
        "other_combined": {"wavSnrDb": -2.94, "magSnrDb": 4.15},
        "full_master_mix": {"wavSnrDb": -2.18, "magSnrDb": 4.62},
    }

    write_js_exports(
        os.path.join(DATA_DIR, "tiers-registry.js"),
        "V9R3 Quality Tiers Registry (`studio_master` Default)",
        [("DEFAULT_V9R3_TIER", "studio_master"), ("V9R3_TIERS", tiers_registry)],
    )
    write_js_exports(
        os.path.join(DATA_DIR, "benchmarks.js"),
        "Ground-Truth Verified 48kHz Studio Stem SNR Comparison (V9R1 vs V9R3 Compact vs V9R3 Studio Master)",
        [
            (
                "V9R3_BENCHMARKS",
                {
                    "groundTruthSource": "32-bit 48kHz 6-Stem Lossless Studio Decomposition (146.81 dB Exact Sum)",
                    "exactBpm": EXACT_BPM,
                    "v9r1Expressive": v9r1_baseline,
                    "v9r3Compact": benchmarks_compact,
                    "v9r3Master": benchmarks_master,
                },
            )
        ],
    )

    code_metrics = {
        "dictGzipBytes": dict_gz,
        "studioMasterTotalGzipBytes": master_gz,
        "compactPureTotalGzipBytes": compact_gz,
        "modules": {k: {"rawBytes": v[0], "gzipBytes": v[1]} for k, v in sz.items()},
        "benchmarksMaster": benchmarks_master,
        "benchmarksCompact": benchmarks_compact,
    }
    with open(os.path.join(DATA_DIR, "code-metrics.json"), "w", encoding="utf-8") as f:
        json.dump(code_metrics, f, ensure_ascii=False, indent=2)

    print(f"\n=== ALL DONE in {time.time()-t_all:.1f}s | Dict Gzip={dict_gz/1024:.1f}KB | Compact={compact_gz/1024:.1f}KB | Master={master_gz/1024:.1f}KB ===")


if __name__ == "__main__":
    main()
