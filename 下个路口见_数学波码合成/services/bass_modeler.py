"""Monophonic Electric Bass String Timbre x Pluck Articulation Wave Modeler (`WVB5`)."""
from __future__ import annotations

import struct
import librosa
import numpy as np
from scipy.cluster.vq import kmeans2

from .audio_io import resample_audio


def encode_bass_wave(bass_48k: np.ndarray, orig_sr: int, spec: dict) -> tuple[bytes, dict, np.ndarray]:
    """Extracts static Electric Bass harmonic timbre + dynamic pitch & pluck VQ trajectory (`WVB5`)."""
    bsr = int(spec["sampleRate"])
    hop = int(spec["hopSamples"])
    max_harm = int(spec["maxHarmonics"])
    n_tone_cb = int(spec["toneCodebookSize"])
    inharm_b = float(spec["inharmonicityB"])

    bass = resample_audio(bass_48k, orig_sr, bsr)
    n_fft = 2048
    pad = n_fft // 2
    stft = librosa.stft(np.pad(bass, (pad, pad)), n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=False)
    mag = np.abs(stft).astype(np.float32)
    n_bins, n_frames = mag.shape
    bin_hz = bsr / n_fft
    inv_peak = 2.0 / float(np.sum(np.hanning(n_fft)))
    rms_scale = float(np.sqrt(2.0 / (n_fft * n_fft * 0.375)))
    frame_rms = np.sqrt(np.sum(mag ** 2, axis=0)) * rms_scale

    f0_raw, _, _ = librosa.pyin(
        bass,
        fmin=36.0,
        fmax=340.0,
        sr=bsr,
        frame_length=n_fft,
        hop_length=hop,
        center=True,
        fill_na=0.0,
    )
    f0 = np.nan_to_num(f0_raw[:n_frames], nan=0.0).astype(np.float32)

    # Refine sub-bass fundamental from first 5 harmonics
    for t in range(n_frames):
        if f0[t] > 32.0:
            col = mag[:, t]
            num, den = 0.0, 0.0
            for k in range(1, 6):
                fk_nom = k * np.sqrt(1.0 + inharm_b * k * k) * f0[t]
                b0 = int(round(fk_nom / bin_hz))
                if 2 <= b0 < n_bins - 2:
                    loc = b0 - 1 + int(np.argmax(col[b0 - 1 : b0 + 2]))
                    y1, y2, y3 = float(col[loc - 1]), float(col[loc]), float(col[loc + 1])
                    denom = y1 - 2.0 * y2 + y3
                    delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-9 and y2 >= y1 and y2 >= y3 else 0.0
                    fk_meas = (loc + float(np.clip(delta, -0.5, 0.5))) * bin_hz
                    k_eff = k * float(np.sqrt(1.0 + inharm_b * k * k))
                    w = y2 * y2
                    num += k_eff * w * fk_meas
                    den += (k_eff * k_eff) * w
            if den > 1e-9:
                f0[t] = float(num / den)

    active = frame_rms > (float(np.max(frame_rms)) * 0.008)
    voiced = (f0 > 32.0) & active

    # Fill short pitch dropouts inside active bass notes by holding last pitch
    last_f0 = 82.41
    for t in range(n_frames):
        if voiced[t]:
            last_f0 = float(f0[t])
        elif active[t]:
            f0[t] = last_f0
            voiced[t] = True

    harm_amps = np.full((max_harm, n_frames), 1e-5, dtype=np.float32)
    noise_ratio = np.zeros(n_frames, dtype=np.float32)

    for t in range(n_frames):
        if voiced[t]:
            col = mag[:, t]
            ks = np.arange(1, max_harm + 1, dtype=np.float32)
            hf = ks * np.sqrt(1.0 + inharm_b * ks * ks) * f0[t]
            valid = hf < (bsr * 0.46)
            pb = np.clip(np.round(hf[valid] / bin_hz).astype(int), 1, n_bins - 2)
            pa = np.maximum(col[pb], np.maximum(col[pb - 1], col[pb + 1])) * inv_peak
            harm_amps[: len(pa), t] = np.maximum(pa, 1e-6)
            harm_e = 0.5 * float(np.sum(pa ** 2))
            tot_e = float(frame_rms[t] ** 2) + 1e-12
            noise_ratio[t] = float(np.clip(np.sqrt(max(0.0, 1.0 - harm_e / tot_e)) * 0.5, 0.0, 0.45))

    log_h = np.log(harm_amps)
    log_h_centered = log_h - np.mean(log_h, axis=0, keepdims=True)
    static_profile = np.mean(log_h_centered[:, voiced], axis=1).astype(np.float32)
    tone_dev = log_h_centered - static_profile[:, None]

    rng = np.random.default_rng(2027)
    cb_f, idx_a = kmeans2(tone_dev[:, voiced].T.astype(np.float32), n_tone_cb, minit="++", iter=30, seed=rng)
    tone_scale = float(np.max(np.abs(cb_f)) / 127.0 + 1e-6)
    cb_q = np.clip(np.round(cb_f / tone_scale), -127, 127).astype(np.int8)

    tone_idx = np.zeros(n_frames, dtype=np.uint8)
    tone_idx[voiced] = idx_a.astype(np.uint8)

    rms_q = np.zeros(n_frames, dtype=np.uint8)
    rms_q[active] = np.clip(np.round((20.0 * np.log10(frame_rms[active] + 1e-7) + 75.0) * 3.2), 1, 255).astype(np.uint8)

    f0_q = np.zeros(n_frames, dtype=np.uint16)
    f0_q[voiced] = np.clip(np.round(12.0 * np.log2(f0[voiced] / 32.7032) * 32.0) + 1, 1, 2047).astype(np.uint16)

    nr_q = np.clip(np.round(noise_ratio * 255.0), 0, 255).astype(np.uint8)

    header = struct.pack("<4sIBBHf", b"WVB5", n_frames, max_harm, n_tone_cb, 0, tone_scale)
    d_rms = np.diff(rms_q.astype(np.int16), prepend=0).astype(np.uint8)
    d_f0 = np.diff(f0_q.astype(np.int16), prepend=0).astype("<i2")
    raw_wvb5 = b"".join([header, cb_q.tobytes(), d_rms.tobytes(), d_f0.tobytes(), tone_idx.tobytes(), nr_q.tobytes()])

    bass_model = {
        "sampleRate": bsr,
        "hopSamples": hop,
        "maxHarmonics": max_harm,
        "inharmonicityB": inharm_b,
        "staticHarmonicLogProfile": [round(float(x), 5) for x in static_profile],
    }

    synth_12k = synthesize_bass_from_wvb5(raw_wvb5, bass_model, len(bass))
    synth_48k = resample_audio(synth_12k, bsr, orig_sr)[: len(bass_48k)]
    if len(synth_48k) < len(bass_48k):
        synth_48k = np.pad(synth_48k, (0, len(bass_48k) - len(synth_48k)))
    return raw_wvb5, bass_model, synth_48k


def synthesize_bass_from_wvb5(raw: bytes, model: dict, out_samples: int) -> np.ndarray:
    """Reference electric bass string wave synthesizer matching `src/dsp/bass-synth.js`."""
    magic, n_frames, max_harm, n_tone_cb, _, tone_scale = struct.unpack_from("<4sIBBHf", raw, 0)
    assert magic == b"WVB5"
    off = 16
    cb = np.frombuffer(raw, dtype=np.int8, count=n_tone_cb * max_harm, offset=off).reshape(n_tone_cb, max_harm).astype(np.float32) * tone_scale
    off += n_tone_cb * max_harm

    d_rms = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    d_f0 = np.frombuffer(raw, dtype="<i2", count=n_frames, offset=off); off += n_frames * 2
    tone_idx = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    nr_q = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off)

    rms_q = (np.cumsum(d_rms.astype(np.int32)) & 0xFF).astype(np.uint8)
    f0_q = np.cumsum(d_f0.astype(np.int32)).astype(np.int32)

    sr = int(model["sampleRate"])
    hop = int(model["hopSamples"])
    inharm_b = float(model["inharmonicityB"])
    static_prof = np.array(model["staticHarmonicLogProfile"], dtype=np.float32)

    active = rms_q > 0
    rms_target = np.zeros(n_frames, dtype=np.float32)
    rms_target[active] = 10.0 ** (((rms_q[active].astype(np.float32) / 3.2) - 75.0) / 20.0)

    f0 = np.zeros(n_frames, dtype=np.float32)
    v_mask = (f0_q > 0) & active
    f0[v_mask] = 32.7032 * (2.0 ** ((f0_q[v_mask].astype(np.float32) - 1.0) / 384.0))
    noise_ratio = nr_q.astype(np.float32) / 255.0

    ks = np.arange(1, max_harm + 1, dtype=np.float32)
    k_stretch = ks * np.sqrt(1.0 + inharm_b * ks * ks)

    h_amps = np.zeros((max_harm, n_frames), dtype=np.float32)
    for t in range(n_frames):
        if v_mask[t]:
            rel = np.exp(np.clip(static_prof + cb[tone_idx[t]], -12.0, 8.0))
            rel[k_stretch * f0[t] >= sr * 0.46] = 0.0
            h_var = 0.5 * float(np.sum(rel ** 2)) + 1e-12
            nr = noise_ratio[t]
            t_h_var = (rms_target[t] ** 2) * (1.0 - nr * nr)
            h_amps[:, t] = rel * np.sqrt(t_h_var / h_var)

    out = np.zeros(out_samples, dtype=np.float32)
    harm_phases = -0.22 * ks.copy()
    rng = np.random.default_rng(202)
    lp_state = 0.0

    for t in range(n_frames - 1):
        s0_i = t * hop
        s1_i = min(out_samples, s0_i + hop)
        L = s1_i - s0_i
        if L <= 0:
            break
        fc, fn = f0[t], f0[t + 1]
        if fc > 30.0 or fn > 30.0:
            fs = fc if fc > 30.0 else fn
            fe = fn if fn > 30.0 else fc
            alpha = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float32)
            f_inst = fs + (fe - fs) * alpha
            dphi0 = 2.0 * np.pi * f_inst / sr
            phi_delta = np.cumsum(dphi0)
            ph = harm_phases[:, None] + k_stretch[:, None] * phi_delta[None, :]
            harm_phases = (ph[:, -1] % (2.0 * np.pi)).astype(np.float32)

            a0, a1 = h_amps[:, t], h_amps[:, t + 1]
            amps = a0[:, None] + (a1 - a0)[:, None] * alpha[None, :]
            out[s0_i:s1_i] += np.sum(amps * np.cos(ph), axis=0)

        # Subtle lowpass pluck transient noise
        n_amp = 0.5 * (rms_target[t] * noise_ratio[t] + rms_target[t + 1] * noise_ratio[t + 1])
        if n_amp > 1e-5:
            wn = rng.standard_normal(L).astype(np.float32)
            for i in range(L):
                lp_state = 0.65 * lp_state + 0.35 * wn[i]
                out[s0_i + i] += lp_state * n_amp * 1.6

    return out
