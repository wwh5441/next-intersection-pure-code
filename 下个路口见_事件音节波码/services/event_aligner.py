"""Acoustic Breath-Island Detector & Viterbi Lyric-to-Syllable Forced Aligner."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal

from models.vocal_structures import SyllableToken


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    labeled, n = ndimage.label(mask)
    runs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(labeled == i)
        runs.append((int(idx[0]), int(idx[-1]) + 1))
    return runs


def _refine_line_cut(
    target_frame: int,
    sung_mask: np.ndarray,
    total_rms: np.ndarray,
    novelty: np.ndarray,
    lo_back: int = 65,
    hi_fwd: int = 40,
) -> int:
    n_frames = len(sung_mask)
    lo = max(0, target_frame - lo_back)
    hi = min(n_frames - 1, target_frame + hi_fwd)
    if lo >= hi:
        return int(np.clip(target_frame, 0, n_frames - 1))

    # Prefer clean inter-phrase breath/silence gap (handles early vocal pickups before LRC timestamp)
    sub_inactive = ~sung_mask[lo : hi + 1]
    sil_runs = [(lo + a, lo + b) for a, b in _find_runs(sub_inactive)]
    long_sil = [r for r in sil_runs if (r[1] - r[0]) >= 3]
    if long_sil:
        def _sil_score(r: tuple[int, int]) -> float:
            r_s, r_e = r
            dist = 0 if (r_s <= target_frame <= r_e) else min(abs(target_frame - r_s), abs(target_frame - r_e))
            return min(float(r_e - r_s), 36.0) - 0.28 * float(dist)

        best_s, best_e = max(long_sil, key=_sil_score)
        return int(min(best_e - 1, max(best_s, best_e - 2)))

    if sil_runs:
        inactive = [f for r_s, r_e in sil_runs for f in range(r_s, r_e)]
        return min(inactive, key=lambda f: abs(f - target_frame))

    # Vectorized legato fallback: minimum RMS + onset novelty valley/boundary
    idx_r = np.arange(lo, hi + 1)
    sc = -np.log10(total_rms[idx_r] + 1e-5) + 1.5 * novelty[idx_r] - 0.022 * np.abs(idx_r - target_frame)
    return int(idx_r[int(np.argmax(sc))])


def _viterbi_split_island(
    seg_s: int,
    seg_e: int,
    tokens: list[SyllableToken],
    features: dict,
) -> list[tuple[int, int, SyllableToken]]:
    """Vectorized DP splitter (`O(m * L^2)` NumPy array ops) with vowel/energy & release-tail guards."""
    m = len(tokens)
    L = seg_e - seg_s
    if m <= 1 or L <= m * 4:
        cuts = [seg_s + int(round(k * L / m)) for k in range(m + 1)]
        return [(cuts[k], max(cuts[k] + 1, cuts[k + 1]), tokens[k]) for k in range(m)]

    nov = features["novelty"][seg_s:seg_e]
    gh = features["gain_harm"][seg_s:seg_e]
    ga = features["gain_aper"][seg_s:seg_e]
    v_sub = features["voiced"][seg_s:seg_e]
    v_prefix = np.concatenate([[0], np.cumsum(v_sub.astype(np.int32))])
    island_has_vowels = int(v_prefix[L]) >= m * 2

    hi_start = max(0, features["aper_log_shape"].shape[1] // 2)
    aper_hi = np.mean(features["aper_log_shape"][seg_s:seg_e, hi_start:], axis=1)
    tot_e = np.sqrt(gh ** 2 + ga ** 2 + 1e-12)
    isl_med_e = float(np.median(tot_e[tot_e > 1e-4])) if np.any(tot_e > 1e-4) else 1e-3
    strong_e = (tot_e > isl_med_e * 0.35).astype(np.int32)
    e_prefix = np.concatenate([[0], np.cumsum(strong_e)])
    log_e = np.log10(tot_e + 1e-5)

    min_len = max(6, min(12, L // (m * 2)))
    avg_len = L / float(m)

    sib_vec = np.array([float(np.mean(ga[p : min(L, p + 6)] / (tot_e[p : min(L, p + 6)] + 1e-7))) for p in range(L)])
    hi_vec = np.array([float(np.mean(aper_hi[p : min(L, p + 6)])) for p in range(L)])
    dip_vec = -0.50 * np.array([float(log_e[max(0, p - 1)]) for p in range(L)])

    vot_trans = np.zeros(L, dtype=np.float64)
    for p in range(1, L):
        if (not v_sub[p - 1]) and v_sub[p]:
            vot_trans[p] = -5.2
        elif v_sub[p - 1] and (not v_sub[p]) and (v_prefix[min(L, p + 18)] - v_prefix[p]) >= 3:
            vot_trans[p] = 2.6

    ob = np.zeros((m, L), dtype=np.float64)
    for k, tok in enumerate(tokens):
        base_s = 2.0 * nov + dip_vec + vot_trans
        if tok.initial_class == "sibilant":
            base_s += 1.65 * sib_vec + 0.08 * hi_vec
        elif tok.initial_class == "plosive":
            base_s += 1.05 * nov + 0.75 * sib_vec
        else:
            base_s += 0.45 * (1.0 - sib_vec)
        ob[k] = base_s

    def _calc_span_pen(p_s_arr: np.ndarray, p_end: int) -> np.ndarray:
        pen = np.zeros_like(p_s_arr, dtype=np.float64)
        if island_has_vowels:
            vc = v_prefix[p_end] - v_prefix[p_s_arr]
            pen += np.where(vc == 0, -14.0, np.where(vc < 3, -7.5, 0.0))
        ec = e_prefix[p_end] - e_prefix[p_s_arr]
        pen += np.where(ec == 0, -14.0, np.where(ec < 3, -5.0, 0.0))
        return pen

    neg_inf = -1e18
    dp = np.full((m, L + 1), neg_inf, dtype=np.float64)
    back = np.zeros((m, L + 1), dtype=np.int32)

    p1_arr = np.arange(min_len, L - (m - 1) * min_len + 1)
    d0 = p1_arr.astype(np.float64)
    t0_d = avg_len * (1.45 if m == 1 else 0.92)
    sp0_d = avg_len * (1.35 if m == 1 else 0.62) + 4.0
    span0 = np.array([_calc_span_pen(np.array([0], dtype=np.int32), int(p1))[0] for p1 in p1_arr])
    dp[0, p1_arr] = ob[0, 0] - 0.85 * ((d0 - t0_d) / sp0_d) ** 2 + span0

    for k in range(1, m):
        is_last = k == (m - 1)
        p_range = [L] if is_last else list(range((k + 1) * min_len, L - (m - 1 - k) * min_len + 1))
        prev_min = k * min_len
        target_d = avg_len * (1.40 if is_last else 0.94)
        spread_d = avg_len * (1.25 if is_last else 0.60) + 4.0
        for p_end in p_range:
            ps = np.arange(prev_min, p_end - min_len + 1)
            durs = (p_end - ps).astype(np.float64)
            vals = dp[k - 1, ps] + ob[k, ps] - 0.85 * ((durs - target_d) / spread_d) ** 2 + _calc_span_pen(ps, p_end)
            bi = int(np.argmax(vals))
            dp[k, p_end] = float(vals[bi])
            back[k, p_end] = int(ps[bi])

    cuts = [0] * (m + 1)
    cuts[m] = L
    cur = L
    for k in range(m - 1, 0, -1):
        cur = int(back[k, cur])
        cuts[k] = cur
    cuts[0] = 0

    return [(seg_s + cuts[k], seg_s + cuts[k + 1], tokens[k]) for k in range(m)]


def segment_song_into_events(sung_lines: list[dict], features: dict, config: dict) -> list[dict]:
    """Aligns all LRC syllables via sub-phrase/onset islands and captures all non-LRC hooks & standalone breaths."""
    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])
    active_mask = features["active_mask"]
    phrase_mask = features.get("phrase_mask", active_mask)
    sung_mask = features.get("sung_mask", active_mask)
    n_frames = len(active_mask)
    gh = features["gain_harm"]
    ga = features["gain_aper"]
    tot_rms = np.sqrt(gh ** 2 + ga ** 2 + 1e-14)
    rms_peak = float(np.max(tot_rms) + 1e-9)
    novelty = features["novelty"]
    voiced = features["voiced"]

    claimed = np.zeros(n_frames, dtype=bool)
    raw_events: list[dict] = []

    # 1. Align each sung LRC line strictly on clean `sung_mask` (immune to inhale breaths & room tails!)
    line_windows: list[tuple[int, int]] = []
    for idx, ln in enumerate(sung_lines):
        raw_s = max(0, int(round(ln["startSec"] * sr / hop)))
        raw_e = min(n_frames - 1, int(round(ln["endSec"] * sr / hop)))
        if idx == 0:
            s_cut = _refine_line_cut(raw_s, sung_mask, tot_rms, novelty, lo_back=120, hi_fwd=35)
        elif abs(ln["startSec"] - sung_lines[idx - 1]["endSec"]) < 0.05:
            s_cut = line_windows[-1][1]
        else:
            s_cut = max(line_windows[-1][1], _refine_line_cut(raw_s, sung_mask, tot_rms, novelty))
        e_cut = _refine_line_cut(raw_e, sung_mask, tot_rms, novelty)
        if e_cut <= s_cut:
            e_cut = min(n_frames, s_cut + len(ln["tokens"]) * 8)
        line_windows.append((s_cut, e_cut))

    for ln, (win_s, win_e) in zip(sung_lines, line_windows):
        sub_mask = sung_mask[win_s:win_e]
        runs = [(win_s + a, win_s + b) for a, b in _find_runs(sub_mask) if (b - a) >= 4]
        tokens: list[SyllableToken] = ln["tokens"]
        pgroups: list[list[SyllableToken]] = ln.get("phraseGroups") or [tokens]
        if not runs:
            continue

        # Merge runs separated by brief consonant closures (<= 8 frames = 43ms)
        merged_runs = [runs[0]]
        for r_s, r_e in runs[1:]:
            if r_s - merged_runs[-1][1] <= 8:
                merged_runs[-1] = (merged_runs[-1][0], r_e)
            else:
                merged_runs.append((r_s, r_e))
        runs = merged_runs

        # Drop trailing interlude/outro hook islands separated by > 1.5s (280 frames) silence after lyric line ends
        cluster_runs = [runs[0]]
        for r_s, r_e in runs[1:]:
            if r_s - cluster_runs[-1][1] > 280:
                break
            cluster_runs.append((r_s, r_e))
        runs = cluster_runs

        # Merge intra-word unvoiced consonant closures (<= 15 frames = 80ms) when aligning to LRC sub-phrase groups
        while len(runs) > len(pgroups) and len(pgroups) >= 1:
            gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
            min_g_idx = int(np.argmin(gaps))
            if gaps[min_g_idx] <= 15:
                runs = runs[:min_g_idx] + [(runs[min_g_idx][0], runs[min_g_idx + 1][1])] + runs[min_g_idx + 2 :]
            else:
                break

        # Allocate tokens via exact LRC sub-phrase structure (`7-11` [5] + `暖杯巧克力` [5]) or onset-peak count!
        if len(runs) == 1:
            alloc = [len(tokens)]
        elif len(runs) == len(pgroups):
            alloc = [len(g) for g in pgroups]
        else:
            if len(runs) > len(tokens):
                runs_by_energy = sorted(
                    runs, key=lambda r: float(np.sum(tot_rms[r[0] : r[1]] ** 2)), reverse=True
                )[: len(tokens)]
                runs = sorted(runs_by_energy, key=lambda r: r[0])

            weights = []
            for r_s, r_e in runs:
                env_sm = ndimage.gaussian_filter1d(tot_rms[r_s:r_e], sigma=2.0)
                pks, _ = signal.find_peaks(env_sm, distance=12, prominence=float(np.max(env_sm)) * 0.12)
                weights.append(0.65 * max(1, len(pks)) + 0.35 * ((r_e - r_s) / 38.0))
            w_arr = np.array(weights, dtype=np.float64)
            raw_counts = np.maximum(1, np.round(w_arr / np.sum(w_arr) * len(tokens)).astype(int))
            while int(np.sum(raw_counts)) > len(tokens):
                idx_dec = int(np.argmax(raw_counts - w_arr / np.sum(w_arr) * len(tokens)))
                if raw_counts[idx_dec] > 1:
                    raw_counts[idx_dec] -= 1
                else:
                    raw_counts[int(np.argmax(raw_counts))] -= 1
            while int(np.sum(raw_counts)) < len(tokens):
                idx_inc = int(np.argmax(w_arr / np.sum(w_arr) * len(tokens) - raw_counts))
                raw_counts[idx_inc] += 1
            alloc = [int(c) for c in raw_counts]

        tok_cursor = 0
        for (r_s, r_e), cnt in zip(runs, alloc):
            sub_toks = tokens[tok_cursor : tok_cursor + cnt]
            tok_cursor += cnt
            spans = _viterbi_split_island(r_s, r_e, sub_toks, features)
            for sp_s, sp_e, tok in spans:
                raw_events.append(
                    {
                        "start_frame": sp_s,
                        "end_frame": sp_e,
                        "display_text": tok.display,
                        "pinyin": tok.pinyin,
                        "initial_class": tok.initial_class,
                        "is_hook": False,
                    }
                )

    raw_events.sort(key=lambda e: e["start_frame"])

    # 1b. Attach natural consonant pickups & unbroken vibrato fade-out tails along `phrase_mask`
    for idx, ev in enumerate(raw_events):
        bound_lo = raw_events[idx - 1]["end_frame"] if idx > 0 else 0
        if ev["start_frame"] > bound_lo:
            max_back = max(bound_lo, ev["start_frame"] - 18)
            cur_s = ev["start_frame"]
            for f in range(cur_s - 1, max_back - 1, -1):
                if phrase_mask[f]:
                    cur_s = f
                else:
                    break
            ev["start_frame"] = cur_s

        bound_hi = raw_events[idx + 1]["start_frame"] if (idx + 1 < len(raw_events)) else n_frames
        if ev["end_frame"] < bound_hi:
            orig_e = ev["end_frame"]
            max_fwd = min(bound_hi, orig_e + 110)
            min_r = float(tot_rms[max(0, orig_e - 1)])
            cur_e = orig_e
            for f in range(orig_e, max_fwd):
                if not phrase_mask[f]:
                    break
                cur_r = float(tot_rms[f])
                if cur_r < min_r:
                    min_r = cur_r
                elif (f - orig_e) >= 10 and cur_r > max(min_r * 3.5, rms_peak * 0.085):
                    break
                cur_e = f + 1
            ev["end_frame"] = cur_e

        claimed[ev["start_frame"] : ev["end_frame"]] = True

    # 2. Capture standalone non-LRC hooks (Intro/Interlude/Outro DI DI DA) while guarding lyric release tails
    claimed_padded = ndimage.binary_dilation(claimed, structure=np.ones(41, dtype=bool))
    unclaimed_active = active_mask & (~claimed_padded)
    hook_runs = _find_runs(unclaimed_active)
    for h_idx, (r_s, r_e) in enumerate(hook_runs):
        L = r_e - r_s
        if L < 14 or float(np.max(tot_rms[r_s:r_e])) < rms_peak * 0.038:
            continue
        next_lim = hook_runs[h_idx + 1][0] if (h_idx + 1 < len(hook_runs)) else n_frames
        max_h_fwd = min(next_lim, r_e + 45)
        min_hr = float(tot_rms[max(0, r_e - 1)])
        for f in range(r_e, max_h_fwd):
            if claimed[f] or not phrase_mask[f]:
                break
            cr = float(tot_rms[f])
            if cr < min_hr:
                min_hr = cr
            elif (f - r_e) >= 6 and cr > max(min_hr * 2.1, rms_peak * 0.030):
                break
            r_e = f + 1
        L = r_e - r_s
        seg_nov = novelty[r_s:r_e]
        peaks, _ = signal.find_peaks(seg_nov, distance=19, prominence=0.14)
        valid_cuts = [0] + [int(p) for p in peaks if 9 <= p <= L - 9] + [L]
        for k in range(len(valid_cuts) - 1):
            sp_s = r_s + valid_cuts[k]
            sp_e = r_s + valid_cuts[k + 1]
            raw_events.append(
                {
                    "start_frame": sp_s,
                    "end_frame": sp_e,
                    "display_text": "♪",
                    "pinyin": "_hook",
                    "initial_class": "plosive",
                    "is_hook": True,
                }
            )

    raw_events.sort(key=lambda e: e["start_frame"])

    # Absorb any trailing `is_hook` release-tail crumbs contiguous with a preceding lyric syllable (e.g. "情" vibrato tail!)
    merged_events: list[dict] = []
    for ev in raw_events:
        if (
            merged_events
            and ev["is_hook"]
            and (not merged_events[-1]["is_hook"])
            and (ev["start_frame"] - merged_events[-1]["end_frame"]) <= 34
        ):
            merged_events[-1]["end_frame"] = max(merged_events[-1]["end_frame"], ev["end_frame"])
        else:
            merged_events.append(ev)
    raw_events = merged_events

    # 2b. Seamless Intra-Phrase Bridging: eliminate all choppy <181ms (or <=235ms active-phrase) dead-zero holes!
    for idx in range(len(raw_events) - 1):
        e_prev = raw_events[idx]["end_frame"]
        s_next = raw_events[idx + 1]["start_frame"]
        gap = s_next - e_prev
        if gap <= 0:
            if s_next < e_prev:
                mid_f = (e_prev + s_next) // 2
                raw_events[idx]["end_frame"] = max(raw_events[idx]["start_frame"] + 1, mid_f)
                raw_events[idx + 1]["start_frame"] = raw_events[idx]["end_frame"]
            continue
        if gap <= 34 or (gap <= 44 and bool(np.all(phrase_mask[e_prev:s_next]))):
            cut_f = e_prev + int(np.argmin(tot_rms[e_prev : s_next + 1]))
            raw_events[idx]["end_frame"] = cut_f
            raw_events[idx + 1]["start_frame"] = cut_f

    # 3. Compute Voice Onset Time (cons_frames, 0 for pure voiced legato!) & Legato continuity flags
    for idx, ev in enumerate(raw_events):
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = max(1, e_f - s_f)
        ev["end_frame"] = s_f + L
        leg_prev = bool(idx > 0 and raw_events[idx - 1]["end_frame"] == s_f)
        leg_next = bool(idx + 1 < len(raw_events) and raw_events[idx + 1]["start_frame"] == (s_f + L))
        max_cons = min(18, max(1, L - 2))
        vot = 2 if L >= 6 else 1
        found_v = False
        for rel in range(max_cons):
            f_abs = s_f + rel
            if voiced[f_abs] and gh[f_abs] >= 0.65 * ga[f_abs]:
                if rel == 0 and leg_prev and ev["initial_class"] not in ("plosive", "sibilant"):
                    vot = 0
                else:
                    vot = max(1, rel)
                found_v = True
                break
        if not found_v:
            vot = min(max_cons, max(2, L // 3))
        ev["cons_frames"] = int(vot)
        ev["total_frames"] = int(L)
        ev["legato_prev"] = leg_prev
        ev["legato_next"] = leg_next

    return raw_events
