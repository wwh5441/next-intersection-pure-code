"""Polyphonic Chromatic Guitar/Keys Harmonic Synth + Strum Noise + 4x4 Householder FDN Stereo Reverb (`WVH5`)."""
from __future__ import annotations

import struct
import librosa
import numpy as np
import torch

from .audio_io import resample_audio

STRUM_EDGES_HZ = (2000.0, 3200.0, 5000.0, 7800.0, 11400.0)
STRUM_CENTERS_HZ = [0.5 * (STRUM_EDGES_HZ[i] + STRUM_EDGES_HZ[i + 1]) for i in range(4)]
FDN_DELAYS_48K = (1193, 1511, 1877, 2293)


def apply_stereo_fdn_reverb(
    mono_48k: np.ndarray,
    side_ratio_curve_48k: np.ndarray,
    reverb_wet: float,
    rt60_sec: float,
    damping: float,
) -> np.ndarray:
    """Block-vectorized 4x4 Householder FDN Stereo Reverb + Haas Doubler matching sample-by-sample `fdn-reverb.js`."""
    from scipy.signal import lfilter

    n_samples = len(mono_48k)
    out = np.zeros((n_samples, 2), dtype=np.float32)

    d0, d1, d2, d3 = FDN_DELAYS_48K
    haas_len = 629
    # Block size <= min(delays, haas_len) == 629 ensures zero intra-block feedback dependency!
    blk = 512

    buf0 = np.zeros(d0, dtype=np.float32)
    buf1 = np.zeros(d1, dtype=np.float32)
    buf2 = np.zeros(d2, dtype=np.float32)
    buf3 = np.zeros(d3, dtype=np.float32)
    haas_buf = np.zeros(haas_len, dtype=np.float32)

    g0 = float(10.0 ** (-3.0 * (d0 / 48000.0) / max(rt60_sec, 0.25)))
    g1 = float(10.0 ** (-3.0 * (d1 / 48000.0) / max(rt60_sec, 0.25)))
    g2 = float(10.0 ** (-3.0 * (d2 / 48000.0) / max(rt60_sec, 0.25)))
    g3 = float(10.0 ** (-3.0 * (d3 / 48000.0) / max(rt60_sec, 0.25)))

    b_lp = np.array([1.0 - damping], dtype=np.float32)
    a_lp = np.array([1.0, -damping], dtype=np.float32)
    z0 = np.zeros(1, dtype=np.float32)
    z1 = np.zeros(1, dtype=np.float32)
    z2 = np.zeros(1, dtype=np.float32)
    z3 = np.zeros(1, dtype=np.float32)
    i0 = i1 = i2 = i3 = ih = 0

    def _read_ring(buf: np.ndarray, idx: int, L: int) -> np.ndarray:
        end = idx + L
        if end <= len(buf):
            return buf[idx:end].copy()
        first = len(buf) - idx
        return np.concatenate((buf[idx:], buf[: L - first]))

    def _write_ring(buf: np.ndarray, idx: int, vals: np.ndarray) -> int:
        L = len(vals)
        end = idx + L
        if end <= len(buf):
            buf[idx:end] = vals
            return 0 if end == len(buf) else end
        first = len(buf) - idx
        buf[idx:] = vals[:first]
        rem = L - first
        buf[:rem] = vals[first:]
        return rem

    for pos in range(0, n_samples, blk):
        x = mono_48k[pos : min(n_samples, pos + blk)]
        sr_w = side_ratio_curve_48k[pos : pos + len(x)]
        L = len(x)

        y0 = _read_ring(buf0, i0, L)
        y1 = _read_ring(buf1, i1, L)
        y2 = _read_ring(buf2, i2, L)
        y3 = _read_ring(buf3, i3, L)
        hx = _read_ring(haas_buf, ih, L)

        s = 0.5 * (y0 + y1 + y2 + y3)
        lp0, z0 = lfilter(b_lp, a_lp, y0 - s, zi=z0)
        lp1, z1 = lfilter(b_lp, a_lp, y1 - s, zi=z1)
        lp2, z2 = lfilter(b_lp, a_lp, y2 - s, zi=z2)
        lp3, z3 = lfilter(b_lp, a_lp, y3 - s, zi=z3)

        inp = x * 0.35
        i0 = _write_ring(buf0, i0, (inp + lp0 * g0).astype(np.float32))
        i1 = _write_ring(buf1, i1, (inp - lp1 * g1).astype(np.float32))
        i2 = _write_ring(buf2, i2, (inp + lp2 * g2).astype(np.float32))
        i3 = _write_ring(buf3, i3, (inp - lp3 * g3).astype(np.float32))
        ih = _write_ring(haas_buf, ih, x)

        rev_mid = 0.5 * (y0 + y2) * reverb_wet
        rev_side = (0.55 * (y0 - y1 + y2 - y3) + 0.45 * (x - hx)) * sr_w
        mid_val = x + rev_mid
        out[pos : pos + L, 0] = mid_val + rev_side
        out[pos : pos + L, 1] = mid_val - rev_side

    return out


def encode_harmony_wave(harmony_stereo_48k: np.ndarray, orig_sr: int, spec: dict) -> tuple[bytes, dict, np.ndarray]:
    """Transcribes polyphonic chords + guitar strum noise + stereo FDN parameters (`WVH5`)."""
    hsr = int(spec["sampleRate"])
    hop = int(spec["hopSamples"])
    midi_min = int(spec["midiMin"])
    midi_max = int(spec["midiMax"])
    max_harm = int(spec["maxHarmonics"])
    top_voices = int(spec["topVoices"])
    q_levels = int(spec["ampQuantLevels"])

    mid_48k = 0.5 * (harmony_stereo_48k[:, 0] + harmony_stereo_48k[:, 1])
    side_48k = 0.5 * (harmony_stereo_48k[:, 0] - harmony_stereo_48k[:, 1])

    mid = resample_audio(mid_48k, orig_sr, hsr)
    side = resample_audio(side_48k, orig_sr, hsr)

    # 1. High-Res Tonal Chord Analysis (< 4200 Hz)
    n_fft = 4096
    pad = n_fft // 2
    stft_mid = librosa.stft(np.pad(mid, (pad, pad)), n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=False)
    stft_side = librosa.stft(np.pad(side, (pad, pad)), n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=False)
    mag = np.abs(stft_mid).astype(np.float32)
    mag_side = np.abs(stft_side).astype(np.float32)
    n_bins, n_frames = mag.shape
    bin_hz = hsr / n_fft
    freqs = np.arange(n_bins, dtype=np.float32) * bin_hz
    rms_scale = float(np.sqrt(2.0 / (n_fft * n_fft * 0.375)))

    tonal_mask = (freqs >= 70.0) & (freqs <= 4200.0)
    harm_rms = np.sqrt(np.sum(mag[tonal_mask] ** 2, axis=0)) * rms_scale
    mid_full_rms = np.sqrt(np.sum(mag ** 2, axis=0)) * rms_scale
    side_full_rms = np.sqrt(np.sum(mag_side ** 2, axis=0)) * rms_scale

    # Dynamic Side/Mid width curve
    side_ratio = np.clip(side_full_rms / (mid_full_rms + 1e-4), 0.0, 1.35).astype(np.float32)
    base_side_ratio = float(np.median(side_ratio[mid_full_rms > np.max(mid_full_rms) * 0.05]))

    # Spectral centroid brightness (0 = warm Rhodes keys, 1 = bright steel-string acoustic guitar)
    cent_num = np.sum(mag[tonal_mask] * freqs[tonal_mask, None], axis=0)
    cent_den = np.sum(mag[tonal_mask], axis=0) + 1e-8
    centroid = cent_num / cent_den
    bright_norm = np.clip((centroid - 520.0) / 1450.0, 0.0, 1.0).astype(np.float32)

    # 2. PyTorch MPS Chromatic Polyphonic Chord Transcription (MIDI 40..95)
    midi_notes = np.arange(midi_min, midi_max + 1, dtype=np.float32)
    n_notes = len(midi_notes)
    f0s = 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0))

    warm_weights = np.array([1.0 / (k ** 1.35) for k in range(1, max_harm + 1)], dtype=np.float32)
    bright_weights = np.array([1.0 / (k ** 0.68) for k in range(1, max_harm + 1)], dtype=np.float32)
    avg_weights = 0.5 * (warm_weights + bright_weights)

    tonal_bins = int(np.searchsorted(freqs, 4400.0))
    D = np.zeros((tonal_bins, n_notes), dtype=np.float32)
    sigma_bins = 1.15
    for i, f0 in enumerate(f0s):
        for k in range(1, max_harm + 1):
            fk = k * f0
            if fk >= 4350.0:
                break
            bc = fk / bin_hz
            lo = max(0, int(bc - 4))
            hi = min(tonal_bins, int(bc + 5))
            bs = np.arange(lo, hi, dtype=np.float32)
            D[lo:hi, i] += avg_weights[k - 1] * np.exp(-0.5 * ((bs - bc) / sigma_bins) ** 2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    inv_peak = 2.0 / float(np.sum(np.hanning(n_fft)))
    D_t = torch.tensor(D, dtype=torch.float32, device=device)
    M_t = torch.tensor(mag[:tonal_bins] * inv_peak, dtype=torch.float32, device=device)
    A_t = torch.full((n_notes, n_frames), 0.0005, dtype=torch.float32, device=device)
    Dt = D_t.T
    DtM = Dt @ M_t
    DtD = Dt @ D_t
    for _ in range(90):
        A_t = A_t * (DtM / (DtD @ A_t + 0.00035))

    A_np = A_t.cpu().numpy().astype(np.float32)
    A_sparse = np.zeros_like(A_np)
    for t in range(n_frames):
        idx = np.argpartition(A_np[:, t], -top_voices)[-top_voices:]
        A_sparse[idx, t] = A_np[idx, t]

    max_note_amp = float(np.max(A_sparse) + 1e-8)
    A_q = np.clip(np.round(np.sqrt(A_sparse / max_note_amp) * q_levels), 0, q_levels).astype(np.uint8)
    d_note_q = np.diff(A_q.astype(np.int16), axis=1, prepend=0).astype(np.uint8)

    # 3. 2x-Rate Crisp Acoustic Guitar Strum / Pick High-Band Envelope (2.0..11.4 kHz)
    hop_strum = hop // 2
    n_fft_s = 512
    pad_s = n_fft_s // 2
    stft_s = librosa.stft(np.pad(mid, (pad_s, pad_s)), n_fft=n_fft_s, hop_length=hop_strum, win_length=n_fft_s, window="hann", center=False)
    mag_s = np.abs(stft_s).astype(np.float32)
    n_s_frames_target = n_frames * 2
    if mag_s.shape[1] < n_s_frames_target:
        mag_s = np.pad(mag_s, ((0, 0), (0, n_s_frames_target - mag_s.shape[1])))
    else:
        mag_s = mag_s[:, :n_s_frames_target]

    freqs_s = np.linspace(0.0, hsr / 2.0, mag_s.shape[0], dtype=np.float32)
    rms_scale_s = float(np.sqrt(2.0 / (n_fft_s * n_fft_s * 0.375)))
    strum_bands_q = np.zeros((4, n_s_frames_target), dtype=np.uint8)
    for b in range(4):
        m = (freqs_s >= STRUM_EDGES_HZ[b]) & (freqs_s < STRUM_EDGES_HZ[b + 1])
        b_rms = np.sqrt(np.sum(mag_s[m] ** 2, axis=0)) * rms_scale_s
        act = b_rms > (float(np.max(b_rms) + 1e-9) * 0.008)
        strum_bands_q[b, act] = np.clip(np.round((20.0 * np.log10(b_rms[act] + 1e-7) + 78.0) * 3.2), 1, 255).astype(np.uint8)

    d_strum = np.diff(strum_bands_q.astype(np.int16), axis=1, prepend=0).astype(np.uint8)

    harm_q = np.zeros(n_frames, dtype=np.uint8)
    act_h = harm_rms > (float(np.max(harm_rms) + 1e-9) * 0.005)
    harm_q[act_h] = np.clip(np.round((20.0 * np.log10(harm_rms[act_h] + 1e-7) + 75.0) * 3.2), 1, 255).astype(np.uint8)
    d_harm_q = np.diff(harm_q.astype(np.int16), prepend=0).astype(np.uint8)

    bright_q = np.clip(np.round(bright_norm * 255.0), 0, 255).astype(np.uint8)
    side_q = np.clip(np.round((side_ratio / 1.4) * 255.0), 0, 255).astype(np.uint8)
    d_side_q = np.diff(side_q.astype(np.int16), prepend=0).astype(np.uint8)

    header = struct.pack("<4sIBBBBf", b"WVH5", n_frames, n_notes, top_voices, q_levels, midi_min, max_note_amp)
    raw_wvh5 = b"".join(
        [
            header,
            d_note_q.tobytes(),
            bright_q.tobytes(),
            d_harm_q.tobytes(),
            d_strum.tobytes(),
            d_side_q.tobytes(),
        ]
    )

    harmony_model = {
        "sampleRate": hsr,
        "hopSamples": hop,
        "midiMin": midi_min,
        "midiMax": midi_max,
        "maxHarmonics": max_harm,
        "warmHarmonicWeights": [round(float(x), 5) for x in warm_weights],
        "brightHarmonicWeights": [round(float(x), 5) for x in bright_weights],
        "strumCentersHz": [round(float(x), 1) for x in STRUM_CENTERS_HZ],
        "stereoAcoustics": {
            "reverbWet": 0.16,
            "rt60Sec": 0.95,
            "damping": 0.34,
            "baseSideRatio": round(base_side_ratio, 4),
        },
    }

    synth_stereo_48k = synthesize_harmony_from_wvh5(raw_wvh5, harmony_model, len(harmony_stereo_48k), orig_sr)
    return raw_wvh5, harmony_model, synth_stereo_48k


def synthesize_harmony_from_wvh5(raw: bytes, model: dict, out_samples_48k: int, out_sr: int = 48000) -> np.ndarray:
    """Reference polyphonic chord oscillator bank + guitar strum + 4x4 FDN stereo reverb matching JS."""
    magic, n_frames, n_notes, _, q_levels, midi_min, max_note_amp = struct.unpack_from("<4sIBBBBf", raw, 0)
    assert magic == b"WVH5"
    off = 16
    d_note_q = np.frombuffer(raw, dtype=np.uint8, count=n_notes * n_frames, offset=off).reshape(n_notes, n_frames)
    off += n_notes * n_frames
    bright_q = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    d_harm_q = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    n_s_frames = n_frames * 2
    d_strum = np.frombuffer(raw, dtype=np.uint8, count=4 * n_s_frames, offset=off).reshape(4, n_s_frames)
    off += 4 * n_s_frames
    d_side_q = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off)

    A_q = (np.cumsum(d_note_q.astype(np.int32), axis=1) & 0xFF).astype(np.uint8)
    harm_q = (np.cumsum(d_harm_q.astype(np.int32)) & 0xFF).astype(np.uint8)
    strum_q = (np.cumsum(d_strum.astype(np.int32), axis=1) & 0xFF).astype(np.uint8)
    side_q = (np.cumsum(d_side_q.astype(np.int32)) & 0xFF).astype(np.uint8)

    hsr = int(model["sampleRate"])
    hop = int(model["hopSamples"])
    max_harm = int(model["maxHarmonics"])
    warm_w = np.array(model["warmHarmonicWeights"], dtype=np.float32)
    bright_w = np.array(model["brightHarmonicWeights"], dtype=np.float32)
    strum_centers = np.array(model["strumCentersHz"], dtype=np.float32)

    A_rec = ((A_q.astype(np.float32) / float(q_levels)) ** 2) * max_note_amp
    bright = bright_q.astype(np.float32) / 255.0

    harm_rms = np.zeros(n_frames, dtype=np.float32)
    act_h = harm_q > 0
    harm_rms[act_h] = 10.0 ** (((harm_q[act_h].astype(np.float32) / 3.2) - 75.0) / 20.0)

    midi_notes = np.arange(midi_min, midi_min + n_notes, dtype=np.float32)
    f0s = 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0))

    # Calibrate per-frame chord amplitude to match exact tonal RMS `harm_rms[t]`
    w_frame = warm_w[:, None] * (1.0 - bright[None, :]) + bright_w[:, None] * bright[None, :]  # (max_harm, n_frames)
    for t in range(n_frames):
        if harm_rms[t] <= 1e-6:
            A_rec[:, t] = 0.0
            continue
        act_n = np.where(A_rec[:, t] > 1e-5)[0]
        if len(act_n) == 0:
            continue
        var_sum = 0.0
        for ni in act_n:
            ks_valid = (np.arange(1, max_harm + 1) * f0s[ni]) < 4350.0
            var_sum += 0.5 * float(np.sum((A_rec[ni, t] * w_frame[ks_valid, t]) ** 2))
        if var_sum > 1e-12:
            A_rec[act_n, t] *= float(np.sqrt((harm_rms[t] ** 2) / var_sum))

    n_samples_24k = int(np.ceil(out_samples_48k * (hsr / out_sr)))
    mid_24k = np.zeros(n_samples_24k, dtype=np.float32)

    rng = np.random.default_rng(505)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_notes, max_harm)).astype(np.float32)
    alpha = np.linspace(0.0, 1.0, hop, endpoint=False, dtype=np.float32)
    t_idx = np.arange(hop, dtype=np.float32)

    for t in range(n_frames - 1):
        s0_i = t * hop
        s1_i = min(n_samples_24k, s0_i + hop)
        L = s1_i - s0_i
        if L <= 0:
            break
        act_notes = np.where((A_rec[:, t] > 1e-5) | (A_rec[:, t + 1] > 1e-5))[0]
        if len(act_notes) == 0:
            continue
        frame_buf = np.zeros(L, dtype=np.float32)
        w0 = w_frame[:, t]
        w1 = w_frame[:, t + 1]
        for ni in act_notes:
            f0 = f0s[ni]
            a0, a1 = A_rec[ni, t], A_rec[ni, t + 1]
            for k in range(max_harm):
                fk = (k + 1) * f0
                if fk >= 4350.0:
                    break
                amp0, amp1 = a0 * w0[k], a1 * w1[k]
                env = amp0 + (amp1 - amp0) * alpha[:L]
                omega = 2.0 * np.pi * fk / hsr
                ph = phases[ni, k] + omega * t_idx[:L]
                phases[ni, k] = float((phases[ni, k] + omega * L) % (2.0 * np.pi))
                frame_buf += env * np.sin(ph)
        mid_24k[s0_i:s1_i] = frame_buf

    # Add 2x-rate crisp acoustic guitar pick/strum bands (> 2.0 kHz)
    hop_s = hop // 2
    n_ola = 512
    win_ola = np.hanning(n_ola).astype(np.float32)
    ola_freqs = np.linspace(0.0, hsr / 2.0, n_ola // 2 + 1, dtype=np.float32)
    ola_norm = float(1.0 / np.sqrt((0.375 * n_ola) / hop_s))

    strum_rms = np.zeros_like(strum_q, dtype=np.float32)
    act_s = strum_q > 0
    strum_rms[act_s] = 10.0 ** (((strum_q[act_s].astype(np.float32) / 3.2) - 78.0) / 20.0)

    for ts in range(n_s_frames):
        tot_s_var = float(np.sum(strum_rms[:, ts] ** 2))
        if tot_s_var <= 1e-10:
            continue
        c_start = ts * hop_s + hop_s // 2 - n_ola // 2
        if 0 <= c_start and c_start + n_ola <= n_samples_24k:
            spec_s = np.interp(ola_freqs, strum_centers, strum_rms[:, ts], left=0.0, right=strum_rms[-1, ts] * 0.4)
            spec_s[ola_freqs < 1950.0] = 0.0
            irfft_rms = np.sqrt((2.0 / (n_ola * n_ola)) * float(np.sum(spec_s[1:-1] ** 2)) + 1e-12)
            cal_s = spec_s * (np.sqrt(tot_s_var) / irfft_rms)
            rph = rng.uniform(0.0, 2.0 * np.pi, size=len(ola_freqs)).astype(np.float32)
            rph[0] = rph[-1] = 0.0
            grain = np.fft.irfft(cal_s * np.exp(1j * rph), n=n_ola).astype(np.float32) * win_ola * ola_norm
            mid_24k[c_start : c_start + n_ola] += grain

    mid_48k_syn = resample_audio(mid_24k, hsr, out_sr)[:out_samples_48k]
    if len(mid_48k_syn) < out_samples_48k:
        mid_48k_syn = np.pad(mid_48k_syn, (0, out_samples_48k - len(mid_48k_syn)))

    side_ratio_frames = (side_q.astype(np.float32) / 255.0) * 1.4
    frame_times_48k = np.arange(n_frames, dtype=np.float32) * (hop * (out_sr / hsr))
    side_curve_48k = np.interp(np.arange(out_samples_48k, dtype=np.float32), frame_times_48k, side_ratio_frames).astype(np.float32)

    ac = model["stereoAcoustics"]
    return apply_stereo_fdn_reverb(
        mid_48k_syn,
        side_curve_48k,
        reverb_wet=float(ac["reverbWet"]),
        rt60_sec=float(ac["rt60Sec"]),
        damping=float(ac["damping"]),
    )
