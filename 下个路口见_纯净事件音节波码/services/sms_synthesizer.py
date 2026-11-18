"""48kHz Global Quadrature Phasor Harmonic + BH92 Zero-Hiss Consonant/Sheen + Velvet Doubler + Native FDN (V9)."""
from __future__ import annotations

import math
import numpy as np
import scipy.signal as signal

from .event_codec import unpack_event_stream, unpack_speaker_core, unpack_syllable_atlas
from .harmonic_aperiodic_analyzer import build_aper_edges, build_erb_grid, build_mcep_basis, hz_to_erb
from .space_fdn_model import synthesize_space_fdn
from .syllable_atlas_builder import (
    F0_CENTS_UNIT,
    build_envelope_knot_positions,
    build_uniform_knot_positions,
    eval_prototype_on_event,
    interp_1d_knots,
    interp_2d_knots,
    uint8_to_gain,
)


def _build_bh92_cola_window(win_len: int = 1024, hop: int = 256) -> np.ndarray:
    u = np.arange(win_len, dtype=np.float64)
    ang = (2.0 * math.pi * u) / float(win_len)
    w_bh = 0.35875 - 0.48829 * np.cos(ang) + 0.14128 * np.cos(2.0 * ang) - 0.01168 * np.cos(3.0 * ang)
    n_ov = win_len // hop
    denom = np.zeros(win_len, dtype=np.float64)
    for r in range(n_ov):
        denom += np.roll(w_bh, r * hop) ** 2
    return (w_bh / denom).astype(np.float32)


def _smooth_frame_track(track: np.ndarray) -> np.ndarray:
    k = np.array([0.08, 0.17, 0.50, 0.17, 0.08], dtype=np.float32)
    out = np.convolve(track, k, mode="same")
    return np.convolve(out, k, mode="same").astype(np.float32)


def synthesize_from_packets(
    raw_speaker: bytes,
    raw_atlas: bytes,
    raw_events: bytes,
    raw_fdn: bytes,
    config: dict,
) -> dict:
    """Reconstructs crystal-pure 48kHz 4-layer stereo vocal strictly from purified V9 packets."""
    speaker, spk_meta = unpack_speaker_core(raw_speaker)
    prototypes = unpack_syllable_atlas(raw_atlas)
    events, ev_meta = unpack_event_stream(raw_events)

    sr = int(spk_meta["sampleRate"])
    n_samples = int(spk_meta["numSamples"])
    hop = int(spk_meta["hop"])
    ref_peak = float(spk_meta["refPeak"])

    n_fft = int(config["fftSize"])       # 2048
    win_len = int(config["windowSize"])  # 1024
    half_w = win_len // 2
    n_erb = int(config["harmonicBands"])
    n_cep = int(config["cepstralCoeffs"])
    n_aper = int(config["aperiodicBands"])
    max_harm = int(config["maxHarmonics"])
    f_h_min = float(config["harmFreqMin"])
    f_h_max = float(config["harmFreqMax"])

    erb_freqs = build_erb_grid(n_erb, f_h_min, f_h_max)
    log_erb = np.log(erb_freqs.astype(np.float64))
    erb_min_v = float(hz_to_erb(f_h_min))
    erb_span_v = float(hz_to_erb(f_h_max)) - erb_min_v
    mcep_basis = build_mcep_basis(n_cep, n_erb)
    aper_edges = build_aper_edges(n_aper, float(config["aperFreqMin"]), float(config["aperFreqMax"]))

    # 4-term Blackman-Harris (-92dB sidelobes) exact COLA synthesis window
    w_ola = _build_bh92_cola_window(win_len, hop)
    e_ola_over_hop = float(np.sum(w_ola.astype(np.float64) ** 2)) / float(hop)

    bin_hz = sr / float(n_fft)
    aper_bin_ranges: list[tuple[int, int, float]] = []
    vowel_sheen_mask = np.zeros(n_aper, dtype=np.float64)
    for b in range(n_aper):
        lo, hi = float(aper_edges[b]), float(aper_edges[b + 1])
        fc = math.sqrt(lo * hi)
        vowel_sheen_mask[b] = min(1.0, max(0.0, (fc - 5000.0) / 2800.0))
        b0 = max(3, int(math.floor(lo / bin_hz)))
        b1 = min(n_fft // 2 - 1, max(b0 + 1, int(math.ceil(hi / bin_hz))))
        kb = b1 - b0
        scale_b = float(n_fft) / math.sqrt(2.0 * float(kb) * e_ola_over_hop)
        aper_bin_ranges.append((b0, b1, scale_b))

    k_pca = int(ev_meta["pcaDims"])
    k_aper = int(ev_meta["aperExprBands"])
    expr_step = int(ev_meta["exprKnotStepFrames"])
    f0_step = int(ev_meta["f0KnotStepFrames"])
    env_step = int(ev_meta["envKnotStepFrames"])
    q_step = float(ev_meta["quantStepDb"])

    basis_m = speaker.mcep_pca_basis[:k_pca]
    basis_a = speaker.aper_pca_basis[:k_aper]

    total_frames_song = n_samples // hop + 4
    u_grid = np.zeros((total_frames_song, max_harm), dtype=np.float32)
    v_grid = np.zeros((total_frames_song, max_harm), dtype=np.float32)
    f0_grid = np.zeros(total_frames_song, dtype=np.float32)
    nh_grid = np.zeros(total_frames_song, dtype=np.uint8)
    v_dom_grid = np.zeros(total_frames_song, dtype=np.float32)

    aper_mono = np.zeros(n_samples + n_fft, dtype=np.float32)
    glottal_mod = np.ones(n_samples + hop, dtype=np.float32)
    pan_track = np.zeros(total_frames_song, dtype=np.float32)
    width_track = np.zeros(total_frames_song, dtype=np.float32)

    rng = np.random.default_rng(20261115)
    m_idx = np.arange(1, n_cep + 1, dtype=np.float64)[:, None]
    min_phase_scale = -(math.log(10.0) / 20.0) * math.sqrt(2.0 / n_erb)
    rolloff_start = 5400.0
    rolloff_span = max(200.0, f_h_max - rolloff_start)

    # Pass 1: Decode purified events onto global quadrature grid + synthesize BH92 COLA consonants/sheen
    for ev in events:
        L = ev.total_frames
        if L <= 0:
            continue
        s_f = ev.start_frame
        cf = ev.cons_frames
        vot = min(L - 1, max(1, cf)) if L > 1 else 0
        proto = prototypes[ev.syllable_id]

        env_pos = build_envelope_knot_positions(L, cf, env_step)
        gh_curve = interp_1d_knots(env_pos, uint8_to_gain(ev.gain_h_db_q, ref_peak), L)
        ga_curve = interp_1d_knots(env_pos, uint8_to_gain(ev.gain_a_db_q, ref_peak), L)

        if not ev.legato_prev and L >= 4:
            gh_curve[0] *= 0.42
            ga_curve[0] *= 0.55
        if not ev.legato_next and L >= 4:
            gh_curve[-1] *= 0.42
            ga_curve[-1] *= 0.55

        unvoiced_only = ev.voiced_onset_frame >= L
        if not unvoiced_only:
            f0_pos = build_uniform_knot_positions(L, f0_step)
            delta_semi = (ev.f0_delta_cents_q.astype(np.float32) * F0_CENTS_UNIT) / 100.0
            midi_curve = (ev.base_midi_q * 0.25) + interp_1d_knots(f0_pos, delta_semi, L)
            f0_curve = 440.0 * np.power(2.0, (midi_curve - 69.0) / 12.0)
        else:
            f0_curve = np.zeros(L, dtype=np.float32)

        pred_m, pred_a = eval_prototype_on_event(proto, speaker, L, cf)
        expr_pos = build_uniform_knot_positions(L, expr_step)
        if not unvoiced_only and len(ev.expr_mcep_q) > 0:
            dm_knots = (ev.expr_mcep_q.astype(np.float32) * q_step) @ basis_m
            mcep_curve = pred_m + interp_2d_knots(expr_pos, dm_knots, L)
        else:
            mcep_curve = pred_m

        if len(ev.expr_aper_q) > 0:
            da_knots = (ev.expr_aper_q.astype(np.float32) * (q_step * 1.35)) @ basis_a
            aper_curve = pred_a + interp_2d_knots(expr_pos, da_knots, L)
        else:
            aper_curve = pred_a

        f_end_clip = min(total_frames_song, s_f + L)
        pan_track[s_f:f_end_clip] = float(ev.stereo_pan_q) / 64.0
        width_track[s_f:f_end_clip] = float(ev.stereo_width_q) / 255.0

        erb_shapes = mcep_curve @ mcep_basis

        for r in range(L):
            f_abs = s_f + r
            center = f_abs * hop
            if center >= n_samples or f_abs >= total_frames_song:
                break

            gh_val = float(gh_curve[r])
            ga_raw = float(ga_curve[r])
            f0_val = float(f0_curve[r])
            has_harm = gh_val > 1e-6 and f0_val > 60.0 and r >= max(0, ev.voiced_onset_frame - 1)

            if has_harm:
                n_h = min(max_harm, int(f_h_max // f0_val))
                if n_h >= 1:
                    ks = np.arange(1, n_h + 1, dtype=np.float64)
                    f_ks = ks * f0_val
                    h_db = np.interp(np.log(f_ks), log_erb, erb_shapes[r].astype(np.float64))
                    u_roll = np.clip((f_ks - rolloff_start) / rolloff_span, 0.0, 1.0)
                    velvet = 0.5 * (1.0 + np.cos(math.pi * u_roll))
                    rel_a = np.power(10.0, h_db / 20.0) * (0.06 + 0.94 * velvet)
                    norm_h = math.sqrt(0.5 * float(np.sum(rel_a ** 2)) + 1e-18)
                    amps = (gh_val / norm_h) * rel_a

                    theta_erb = np.clip((hz_to_erb(f_ks) - erb_min_v) / erb_span_v * math.pi, 0.0, math.pi)
                    tract_ph = min_phase_scale * np.sum(
                        mcep_curve[r][:, None].astype(np.float64) * np.sin(m_idx * theta_erb[None, :]),
                        axis=0,
                    )
                    psi = speaker.glottal_phase[:n_h].astype(np.float64) + tract_ph
                    u_grid[f_abs, :n_h] = (amps * np.cos(psi)).astype(np.float32)
                    v_grid[f_abs, :n_h] = (-amps * np.sin(psi)).astype(np.float32)
                    f0_grid[f_abs] = f0_val
                    nh_grid[f_abs] = n_h
                    v_dom_grid[f_abs] = gh_val / (gh_val + 1.5 * ga_raw + 1e-9)
                else:
                    has_harm = False

            # Voiced Vowel Zero-Hiss Law: <5kHz noise = 0.0, >5kHz velvet sheen <= 0.042 * gh_val
            ga_val = min(ga_raw, gh_val * 0.042) if (has_harm and r >= vot) else ga_raw
            if ga_val > 1e-6:
                bp_rel = np.power(10.0, aper_curve[r].astype(np.float64) / 10.0)
                if has_harm and r >= vot:
                    bp_rel *= vowel_sheen_mask
                bp_sum = float(np.sum(bp_rel))
                if bp_sum > 1e-16:
                    band_sigmas = ga_val * np.sqrt(bp_rel / bp_sum)
                    spec_a = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
                    rand_ph = rng.uniform(0.0, 2.0 * math.pi, size=n_fft // 2 + 1)
                    unit_z = np.cos(rand_ph) + 1j * np.sin(rand_ph)
                    for b in range(n_aper):
                        if band_sigmas[b] > 1e-8:
                            b0, b1, sc_b = aper_bin_ranges[b]
                            spec_a[b0:b1] = (band_sigmas[b] * sc_b) * unit_z[b0:b1]

                    ga_zp = np.fft.irfft(spec_a, n=n_fft)
                    grain_a = np.empty(win_len, dtype=np.float32)
                    grain_a[:half_w] = ga_zp[n_fft - half_w :]
                    grain_a[half_w:] = ga_zp[:half_w]
                    start = center - half_w
                    s0 = max(0, start)
                    g0 = s0 - start
                    e0 = min(len(aper_mono), start + win_len)
                    g1 = g0 + (e0 - s0)
                    aper_mono[s0:e0] += grain_a[g0:g1] * w_ola[g0:g1]

    # Pass 2: Global Sample-Continuous Quadrature Synthesis + Glottal Open-Phase Air Modulation
    harm_mono = np.zeros(n_samples + hop, dtype=np.float32)
    s_norm = np.arange(hop, dtype=np.float64) / float(hop)
    two_pi_over_sr = (2.0 * math.pi) / float(sr)
    ks_full = np.arange(1, max_harm + 1, dtype=np.float64)[:, None]
    phi0 = 0.0
    n_hops = n_samples // hop + 1

    for f in range(n_hops):
        nh0 = int(nh_grid[f])
        nh1 = int(nh_grid[f + 1])
        nh = max(nh0, nh1)
        if nh == 0:
            continue
        if nh0 == 0 and nh1 > 0:
            phi0 = 0.0
            f_start = float(f0_grid[f + 1])
            f_end = f_start
        elif nh0 > 0 and nh1 == 0:
            f_start = float(f0_grid[f])
            f_end = f_start
        else:
            f_start = float(f0_grid[f])
            f_end = float(f0_grid[f + 1])

        f_vec = f_start + (f_end - f_start) * s_norm
        phi_vec = (phi0 + np.cumsum(two_pi_over_sr * f_vec)) % (2.0 * math.pi)
        phi0 = float(phi_vec[-1])

        u0 = u_grid[f, :nh].astype(np.float64)[:, None]
        u1 = u_grid[f + 1, :nh].astype(np.float64)[:, None]
        v0 = v_grid[f, :nh].astype(np.float64)[:, None]
        v1 = v_grid[f + 1, :nh].astype(np.float64)[:, None]
        u_mat = u0 + (u1 - u0) * s_norm[None, :]
        v_mat = v0 + (v1 - v0) * s_norm[None, :]

        ang = ks_full[:nh] * phi_vec[None, :]
        base = f * hop
        harm_mono[base : base + hop] = np.sum(u_mat * np.cos(ang) + v_mat * np.sin(ang), axis=0).astype(np.float32)

        if nh0 > 0 and nh1 > 0:
            m_g = 0.45 * float(0.5 * (v_dom_grid[f] + v_dom_grid[f + 1]))
            g_norm = 1.0 / math.sqrt(1.0 + 0.5 * m_g * m_g)
            glottal_mod[base : base + hop] = ((1.0 + m_g * np.cos(phi_vec)) * g_norm).astype(np.float32)

    harm_mono = harm_mono[:n_samples]
    aper_mono = (aper_mono[:n_samples] * glottal_mod[:n_samples]).astype(np.float32)
    dry_sum = harm_mono + aper_mono

    pan_smooth = _smooth_frame_track(pan_track)
    width_smooth = _smooth_frame_track(width_track)

    idx_f = np.arange(n_samples, dtype=np.float32) / float(hop)
    f0_i = np.clip(np.floor(idx_f).astype(np.int32), 0, len(pan_smooth) - 2)
    frac_i = idx_f - f0_i
    pan_s = (1.0 - frac_i) * pan_smooth[f0_i] + frac_i * pan_smooth[f0_i + 1]
    width_s = (1.0 - frac_i) * width_smooth[f0_i] + frac_i * width_smooth[f0_i + 1]

    ang_pan = (pan_s + 1.0) * (math.pi * 0.25)
    gL = (math.sqrt(2.0) * np.cos(ang_pan)).astype(np.float32)
    gR = (math.sqrt(2.0) * np.sin(ang_pan)).astype(np.float32)

    layer_harm = np.column_stack([harm_mono * gL, harm_mono * gR]).astype(np.float32)
    layer_aper = np.column_stack([aper_mono * gL, aper_mono * gR]).astype(np.float32)

    # Layer 3: Band-Limited (240Hz - 5.4kHz) Silky Stereo Chorus Doubler (zero sibilance splatter!)
    hp_a = math.exp(-2.0 * math.pi * 240.0 / sr)
    lp_a = math.exp(-2.0 * math.pi * 5400.0 / sr)
    bp_dry = signal.lfilter([hp_a, -hp_a], [1.0, -hp_a], dry_sum.astype(np.float64))
    bp_dry = signal.lfilter([1.0 - lp_a], [1.0, -lp_a], bp_dry).astype(np.float32)

    t_sec = np.arange(n_samples, dtype=np.float32) / float(sr)
    dL = 595.0 + 34.0 * np.sin(2.0 * math.pi * 0.63 * t_sec)
    dR = 868.0 + 38.0 * np.cos(2.0 * math.pi * 0.74 * t_sec)
    n_idx = np.arange(n_samples, dtype=np.float32)

    posL = np.clip(n_idx - dL, 0.0, n_samples - 2.0)
    iL = posL.astype(np.int32)
    fL = posL - iL
    dbl_L = ((1.0 - fL) * bp_dry[iL] + fL * bp_dry[iL + 1]) * width_s * 0.28

    posR = np.clip(n_idx - dR, 0.0, n_samples - 2.0)
    iR = posR.astype(np.int32)
    fR = posR - iR
    dbl_R = -((1.0 - fR) * bp_dry[iR] + fR * bp_dry[iR + 1]) * width_s * 0.28
    layer_double = np.column_stack([dbl_L, dbl_R]).astype(np.float32)

    # Layer 4: Native 48kHz Zero-Aliasing 8-Line Householder Studio Vocal Plate
    layer_fdn = synthesize_space_fdn(raw_fdn, dry_sum)

    mix_stereo = (layer_harm + layer_aper + layer_double + layer_fdn).astype(np.float32)
    return {
        "mix_stereo": mix_stereo,
        "layers": {
            "harmonic_core": layer_harm,
            "consonant_air": layer_aper,
            "stereo_double": layer_double,
            "space_fdn": layer_fdn,
        },
        "events_count": len(events),
        "prototypes_count": len(prototypes),
    }
