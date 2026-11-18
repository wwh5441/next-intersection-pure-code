"""LRC Lag-Compensated Phrase Aligner, Phantom-Hook Purger & Viterbi Syllable Aligner (V9)."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage

from models.vocal_structures import SyllableToken


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    labeled, n = ndimage.label(mask)
    runs: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(labeled == i)
        runs.append((int(idx[0]), int(idx[-1]) + 1))
    return runs


def _refine_line_cut(
    target_frame: int, active_mask: np.ndarray, total_rms: np.ndarray, novelty: np.ndarray, radius: int = 34
) -> int:
    n_frames = len(active_mask)
    lo = max(0, target_frame - radius)
    hi = min(n_frames - 1, target_frame + radius)
    if lo >= hi:
        return int(np.clip(target_frame, 0, n_frames - 1))
    inactive = [f for f in range(lo, hi + 1) if not active_mask[f]]
    if inactive:
        return min(inactive, key=lambda f: abs(f - target_frame))
    scores = []
    for f in range(lo, hi + 1):
        dist_pen = 0.015 * abs(f - target_frame)
        rms_score = -np.log10(float(total_rms[f]) + 1e-6)
        nov_score = 1.55 * float(novelty[f])
        scores.append((rms_score + nov_score - dist_pen, f))
    return max(scores, key=lambda x: x[0])[1]


def _trim_event_tails(
    sp_s: int, sp_e: int, tot_rms: np.ndarray, floor_rms: float, min_len: int = 4
) -> tuple[int, int]:
    """Trims low-energy reverb/breath residue off leading and trailing syllable edges."""
    s, e = sp_s, sp_e
    while e - s > min_len and tot_rms[s] < floor_rms:
        s += 1
    while e - s > min_len and tot_rms[e - 1] < floor_rms:
        e -= 1
    return s, e


def _viterbi_split_island(
    seg_s: int,
    seg_e: int,
    tokens: list[SyllableToken],
    features: dict,
) -> list[tuple[int, int, SyllableToken]]:
    m = len(tokens)
    L = seg_e - seg_s
    if m <= 1 or L <= m * 4:
        cuts = [seg_s + int(round(k * L / m)) for k in range(m + 1)]
        return [(cuts[k], max(cuts[k] + 1, cuts[k + 1]), tokens[k]) for k in range(m)]

    nov = features["novelty"][seg_s:seg_e]
    gh = features["gain_harm"][seg_s:seg_e]
    ga = features["gain_aper"][seg_s:seg_e]
    aper_hi = np.mean(features["aper_log_shape"][seg_s:seg_e, 9:], axis=1)
    tot_e = np.sqrt(gh ** 2 + ga ** 2 + 1e-12)
    log_e = np.log10(tot_e + 1e-5)

    min_len = max(5, min(11, L // (m * 2)))
    avg_len = L / float(m)

    def onset_bonus(local_pos: int, tok: SyllableToken) -> float:
        w_end = min(L, local_pos + 6)
        sib_ratio = float(np.mean(ga[local_pos:w_end] / (tot_e[local_pos:w_end] + 1e-7)))
        hi_tilt = float(np.mean(aper_hi[local_pos:w_end]))
        b_nov = float(nov[local_pos])
        dip = -0.45 * float(log_e[max(0, local_pos - 1)])
        score = 2.2 * b_nov + dip
        if tok.initial_class == "sibilant":
            score += 1.45 * sib_ratio + 0.08 * hi_tilt
        elif tok.initial_class == "plosive":
            score += 0.95 * b_nov + 0.65 * sib_ratio
        else:
            score += 0.45 * (1.0 - sib_ratio)
        return score

    def dur_score(d: int, is_last: bool) -> float:
        target = avg_len * (1.42 if is_last else 0.93)
        spread = avg_len * (1.32 if is_last else 0.60) + 4.0
        z = (d - target) / spread
        return -0.85 * (z * z)

    neg_inf = -1e18
    dp = np.full((m, L + 1), neg_inf, dtype=np.float64)
    back = np.zeros((m, L + 1), dtype=np.int32)

    for p1 in range(min_len, L - (m - 1) * min_len + 1):
        dp[0, p1] = onset_bonus(0, tokens[0]) + dur_score(p1, m == 1)

    for k in range(1, m):
        is_last = k == (m - 1)
        p_min = (k + 1) * min_len
        p_max = L if is_last else (L - (m - 1 - k) * min_len)
        p_range = [L] if is_last else list(range(p_min, p_max + 1))
        prev_min = k * min_len
        for p_end in p_range:
            best_val = neg_inf
            best_prev = prev_min
            prev_hi = p_end - min_len
            for p_start in range(prev_min, prev_hi + 1):
                prev_v = dp[k - 1, p_start]
                if prev_v <= neg_inf * 0.5:
                    continue
                val = prev_v + onset_bonus(p_start, tokens[k]) + dur_score(p_end - p_start, is_last)
                if val > best_val:
                    best_val = val
                    best_prev = p_start
            dp[k, p_end] = best_val
            back[k, p_end] = best_prev

    cuts = [0] * (m + 1)
    cuts[m] = L
    cur = L
    for k in range(m - 1, 0, -1):
        cur = int(back[k, cur])
        cuts[k] = cur
    cuts[0] = 0

    return [(seg_s + cuts[k], seg_s + cuts[k + 1], tokens[k]) for k in range(m)]


def segment_song_into_events(sung_lines: list[dict], features: dict, config: dict) -> tuple[list[dict], int]:
    """Compensates LRC timestamp lag, captures complete phrase onsets/releases, and purges 100% of phantom hooks!"""
    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])
    active_mask = features["active_mask"]
    n_frames = len(active_mask)
    gh = features["gain_harm"]
    ga = features["gain_aper"]
    tot_rms = np.sqrt(gh ** 2 + ga ** 2 + 1e-16)
    peak_tot = float(np.max(tot_rms) + 1e-9)
    trim_floor = peak_tot * 0.011
    novelty = features["novelty"]
    voiced = features["voiced"]

    claimed = np.zeros(n_frames, dtype=bool)
    raw_events: list[dict] = []

    # 1. Lag-compensated phrase boundary alignment:
    #    - Standard .lrc timestamps lag real vocal pickups by ~0.18s..0.42s (`刚` @22.70 vs 23.03, `秒` @61.58 vs 61.90, `平` @71.17 vs 71.47, `秒` @139.96 vs 140.35).
    #    - Phrase-final vowels sustain past .lrc endSec (`起` @118.34 -> 118.65).
    line_windows: list[tuple[int, int]] = []
    n_lines = len(sung_lines)
    for idx, ln in enumerate(sung_lines):
        lrc_s = int(round(ln["startSec"] * sr / hop))
        lrc_e = int(round(ln["endSec"] * sr / hop))
        prev_e_sec = sung_lines[idx - 1]["endSec"] if idx > 0 else 0.0
        next_s_sec = sung_lines[idx + 1]["startSec"] if idx + 1 < n_lines else (n_frames * hop / sr)
        gap_before = ln["startSec"] - prev_e_sec
        gap_after = next_s_sec - ln["endSec"]

        # Start boundary:
        if idx == 0 or gap_before > 0.25:
            # Phrase starts after an instrumental/breath gap: look back up to 0.46s (86 frames) to capture real vocal pickup!
            prev_bound = line_windows[-1][1] if idx > 0 else 0
            lookback_lo = max(prev_bound, lrc_s - 86)
            sub_runs = _find_runs(active_mask[lookback_lo : min(n_frames, lrc_s + 16)])
            if sub_runs:
                s_cut = max(prev_bound, lookback_lo + sub_runs[0][0] - 1)
            else:
                s_cut = max(prev_bound, lrc_s - 12)
        else:
            # Back-to-back sung lines: center search 0.12s earlier to absorb LRC button-press lag
            s_cut = _refine_line_cut(max(0, lrc_s - 22), active_mask, tot_rms, novelty, radius=34)
            if s_cut < line_windows[-1][1]:
                s_cut = line_windows[-1][1]

        # End boundary:
        if idx == n_lines - 1 or gap_after > 0.25:
            # Phrase precedes an empty gap: if vowel is still unbroken across lrc_e, follow that contiguous island to its natural end!
            next_lrc_s = int(round(next_s_sec * sr / hop))
            hard_hi = min(n_frames - 1, next_lrc_s - 6)
            probe = min(hard_hi, max(s_cut + 2, lrc_e - 4))
            if active_mask[probe]:
                cur = probe
                while cur < hard_hi and active_mask[cur] and cur < lrc_e + 75:
                    cur += 1
                e_cut = min(hard_hi, cur + 1)
            else:
                e_cut = min(hard_hi, _refine_line_cut(lrc_e, active_mask, tot_rms, novelty, radius=26))
        else:
            e_cut = _refine_line_cut(max(s_cut + 4, lrc_e - 22), active_mask, tot_rms, novelty, radius=34)

        if e_cut <= s_cut:
            e_cut = min(n_frames, s_cut + len(ln["tokens"]) * 8)
        line_windows.append((s_cut, e_cut))

    # 2. Split each aligned phrase window into exact lyric syllables via Viterbi
    for ln, (win_s, win_e) in zip(sung_lines, line_windows):
        claimed[win_s:win_e] = True
        sub_mask = active_mask[win_s:win_e]
        runs = [(win_s + a, win_s + b) for a, b in _find_runs(sub_mask) if (b - a) >= 4]
        tokens: list[SyllableToken] = ln["tokens"]
        if not runs:
            continue

        if len(runs) == 1:
            alloc = [len(tokens)]
        else:
            merged_runs = [runs[0]]
            for r_s, r_e in runs[1:]:
                if r_s - merged_runs[-1][1] <= 6:
                    merged_runs[-1] = (merged_runs[-1][0], r_e)
                else:
                    merged_runs.append((r_s, r_e))
            runs = merged_runs
            if len(runs) > len(tokens):
                runs_by_energy = sorted(
                    runs, key=lambda r: float(np.sum(tot_rms[r[0] : r[1]] ** 2)), reverse=True
                )[: len(tokens)]
                runs = sorted(runs_by_energy, key=lambda r: r[0])

            lengths = np.array([r[1] - r[0] for r in runs], dtype=np.float64)
            raw_counts = np.maximum(1, np.round(lengths / np.sum(lengths) * len(tokens)).astype(int))
            while int(np.sum(raw_counts)) > len(tokens):
                idx_dec = int(np.argmax(raw_counts - lengths / np.sum(lengths) * len(tokens)))
                if raw_counts[idx_dec] > 1:
                    raw_counts[idx_dec] -= 1
                else:
                    raw_counts[int(np.argmax(raw_counts))] -= 1
            while int(np.sum(raw_counts)) < len(tokens):
                idx_inc = int(np.argmax(lengths / np.sum(lengths) * len(tokens) - raw_counts))
                raw_counts[idx_inc] += 1
            alloc = [int(c) for c in raw_counts]

        tok_cursor = 0
        for (r_s, r_e), cnt in zip(runs, alloc):
            sub_toks = tokens[tok_cursor : tok_cursor + cnt]
            tok_cursor += cnt
            spans = _viterbi_split_island(r_s, r_e, sub_toks, features)
            for idx_sp, (sp_s, sp_e, tok) in enumerate(spans):
                t_s, t_e = sp_s, sp_e
                if idx_sp == 0:
                    t_s, _ = _trim_event_tails(t_s, t_e, tot_rms, trim_floor)
                if idx_sp == len(spans) - 1:
                    _, t_e = _trim_event_tails(t_s, t_e, tot_rms, trim_floor)
                raw_events.append(
                    {
                        "start_frame": t_s,
                        "end_frame": max(t_s + 2, t_e),
                        "display_text": tok.display,
                        "pinyin": tok.pinyin,
                        "initial_class": tok.initial_class,
                        "is_hook": False,
                    }
                )

    # 3. Purge 100% of unclaimed islands outside sung lyrics (zero phantom gibberish hooks or accompaniment bleed!)
    unclaimed_runs = _find_runs(active_mask & (~claimed))
    rejected_ghosts = len(unclaimed_runs)

    raw_events.sort(key=lambda e: e["start_frame"])

    # 4. Voice Onset Time (cons_frames) & Seamless Legato Continuity Flags
    for idx, ev in enumerate(raw_events):
        s_f, e_f = ev["start_frame"], ev["end_frame"]
        L = e_f - s_f
        max_cons = min(13, max(1, L - 2))
        vot = 2 if L >= 6 else 1
        found_v = False
        for rel in range(max_cons):
            f_abs = s_f + rel
            if voiced[f_abs] and gh[f_abs] >= 0.85 * ga[f_abs]:
                vot = max(1, rel)
                found_v = True
                break
        if not found_v:
            vot = min(max_cons, max(1, L // 3))
        ev["cons_frames"] = int(vot)
        ev["total_frames"] = int(L)
        ev["legato_prev"] = bool(
            idx > 0
            and raw_events[idx - 1]["end_frame"] >= s_f - 1
            and voiced[s_f]
            and voiced[raw_events[idx - 1]["end_frame"] - 1]
        )
        ev["legato_next"] = bool(
            idx + 1 < len(raw_events)
            and raw_events[idx + 1]["start_frame"] <= e_f + 1
            and voiced[max(s_f, e_f - 1)]
            and voiced[raw_events[idx + 1]["start_frame"]]
        )

    return raw_events, rejected_ghosts
