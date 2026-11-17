"""AI Mel-Spectrum Pronunciation x 3D Speaker Timbre Disentangler & Pure Wave Synthesizer (`WVV5`)."""
from __future__ import annotations

import struct
import librosa
import numpy as np
from scipy.cluster.vq import kmeans2

from .audio_io import resample_audio

AP_EDGES_HZ = (0.0, 350.0, 800.0, 1600.0, 2800.0, 4500.0, 7000.0, 9500.0, 12000.0)
AP_CENTERS_HZ = [0.5 * (AP_EDGES_HZ[i] + AP_EDGES_HZ[i + 1]) for i in range(8)]


def min_phase_64(log_mag_33: np.ndarray) -> np.ndarray:
    """Exact 64-pt real-cepstrum minimum phase from 33 linearly spaced log-magnitude bins (0..Nyquist)."""
    sym = np.concatenate([log_mag_33, log_mag_33[-2:0:-1]])  # length 64
    cep = np.fft.ifft(sym).real
    win = np.zeros(64, dtype=np.float32)
    win[0] = 1.0
    win[1:32] = 2.0
    win[32] = 1.0
    analytic = np.fft.fft(cep * win)
    return (-analytic[:33].imag).astype(np.float32)


def _refine_f0_track(vocal: np.ndarray, sr: int, hop: int, n_fft: int, mag: np.ndarray) -> np.ndarray:
    f0_raw, _, _ = librosa.pyin(
        vocal,
        fmin=118.0,
        fmax=680.0,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop,
        center=True,
        fill_na=0.0,
    )
    n_bins, n_frames = mag.shape
    f0 = np.nan_to_num(f0_raw[:n_frames], nan=0.0).astype(np.float32)
    bin_hz = sr / n_fft

    # Sub-Hz weighted harmonic peak refinement
    for t in range(n_frames):
        if f0[t] > 80.0:
            col = mag[:, t]
            num, den = 0.0, 0.0
            for k in range(1, 7):
                b0 = int(round(k * f0[t] / bin_hz))
                if 2 <= b0 < n_bins - 2:
                    loc = b0 - 1 + int(np.argmax(col[b0 - 1 : b0 + 2]))
                    y1, y2, y3 = float(col[loc - 1]), float(col[loc]), float(col[loc + 1])
                    denom = y1 - 2.0 * y2 + y3
                    delta = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-9 and y2 >= y1 and y2 >= y3 else 0.0
                    fk = (loc + float(np.clip(delta, -0.5, 0.5))) * bin_hz
                    w = y2 * y2
                    num += k * w * fk
                    den += (k * k) * w
            if den > 1e-9:
                f0[t] = float(num / den)

    # Bridge single-frame pitch dropouts inside voiced vowels
    for t in range(1, n_frames - 1):
        if f0[t] <= 80.0 and f0[t - 1] > 80.0 and f0[t + 1] > 80.0:
            if abs(np.log2(f0[t + 1] / f0[t - 1])) < 0.18:
                f0[t] = 0.5 * (f0[t - 1] + f0[t + 1])
    return f0


def encode_vocal_wave(vocal_48k: np.ndarray, orig_sr: int, spec: dict) -> tuple[bytes, dict, np.ndarray]:
    """Disentangles Vocal into 3D Speaker Timbre x RVQ Pronunciation Codebook, packs `WVV5`, and synthesizes wave."""
    vsr = int(spec["sampleRate"])
    hop = int(spec["hopSamples"])
    n_mels = int(spec["nMels"])
    rank = int(spec["pronBasisRank"])
    rvq_stages = list(spec["rvqStages"])
    n_ap_cb = int(spec["apCodebookSize"])
    max_harm = int(spec["maxHarmonics"])

    vocal = resample_audio(vocal_48k, orig_sr, vsr)
    n_fft = 2048
    pad = n_fft // 2
    stft = librosa.stft(np.pad(vocal, (pad, pad)), n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=False)
    mag = np.abs(stft).astype(np.float32)
    n_bins, n_frames = mag.shape
    bin_hz = vsr / n_fft
    inv_peak = 2.0 / float(np.sum(np.hanning(n_fft)))

    f0 = _refine_f0_track(vocal, vsr, hop, n_fft, mag)

    mel_centers = librosa.mel_frequencies(n_mels=n_mels, fmin=70.0, fmax=11600.0).astype(np.float32)
    mel_fb = librosa.filters.mel(sr=vsr, n_fft=n_fft, n_mels=n_mels, fmin=70.0, fmax=11600.0).astype(np.float32)
    mel_fb_norm = mel_fb / (np.sum(mel_fb, axis=1, keepdims=True) + 1e-8)

    rms_scale = float(np.sqrt(2.0 / (n_fft * n_fft * 0.375)))
    frame_rms = np.sqrt(np.sum(mag ** 2, axis=0)) * rms_scale

    ap_ratio = np.ones((8, n_frames), dtype=np.float32)
    mel_env = np.zeros((n_mels, n_frames), dtype=np.float32)

    for t in range(n_frames):
        col = mag[:, t]
        if f0[t] > 80.0:
            max_k = int(min(11600.0, vsr * 0.48) // f0[t])
            ks = np.arange(1, max_k + 1, dtype=np.float32)
            hf = ks * f0[t]
            pb = np.clip(np.round(hf / bin_hz).astype(int), 1, n_bins - 2)
            pa = np.maximum(col[pb], np.maximum(col[pb - 1], col[pb + 1])) * inv_peak
            vb = np.clip(np.round((ks - 0.5) * f0[t] / bin_hz).astype(int), 1, n_bins - 2)
            va = np.minimum(col[vb], np.minimum(col[vb - 1], col[vb + 1])) * inv_peak
            for b in range(8):
                m = (hf >= AP_EDGES_HZ[b]) & (hf < AP_EDGES_HZ[b + 1])
                if np.any(m):
                    ap_ratio[b, t] = float(np.clip(np.sqrt(np.sum(va[m] ** 2) / (np.sum(pa[m] ** 2) + 1e-12)), 0.02, 0.96))
                else:
                    ap_ratio[b, t] = 0.15 if b < 4 else 0.65
            mel_env[:, t] = np.interp(
                mel_centers,
                np.concatenate(([0.0], hf, [12000.0])),
                np.concatenate(([pa[0] * 0.25], pa, [pa[-1] * 0.2])),
            )
        else:
            ap_ratio[:, t] = 1.0
            mel_env[:, t] = (mel_fb_norm @ col) * inv_peak

    peak_rms = float(np.max(frame_rms) + 1e-9)
    active = frame_rms > (peak_rms * 0.0045)
    voiced = (f0 > 80.0) & active

    # 1. AI Bilinear Disentanglement: 3D Speaker Timbre Field (Static + Pitch-Register + Vocal-Effort)
    log_mel = np.log(np.maximum(mel_env, 1e-6))
    gain = np.mean(log_mel, axis=0, keepdims=True)
    centered = log_mel - gain

    timbre_static = np.mean(centered[:, voiced], axis=1).astype(np.float32)  # (48,)
    mean_voiced_gain = float(np.mean(gain[0, voiced]))

    f0_oct = np.zeros(n_frames, dtype=np.float32)
    f0_oct[voiced] = np.log2(f0[voiced] / 261.63)
    effort = np.zeros(n_frames, dtype=np.float32)
    effort[active] = gain[0, active] - mean_voiced_gain

    X_reg = np.stack([f0_oct[voiced], effort[voiced]], axis=1)
    Y_reg = (centered[:, voiced] - timbre_static[:, None]).T
    beta = np.linalg.lstsq(X_reg, Y_reg, rcond=None)[0].astype(np.float32)  # (2, 48)
    timbre_pitch = beta[0]
    timbre_effort = beta[1]

    pron_res = (
        centered
        - timbre_static[:, None]
        - timbre_pitch[:, None] * f0_oct[None, :]
        - timbre_effort[:, None] * effort[None, :]
    )

    # 2. Orthonormal Pronunciation Basis + Multi-Stage Residual VQ Codebooks
    U_full, _, _ = np.linalg.svd(pron_res[:, active], full_matrices=False)
    U_pron = U_full[:, :rank].astype(np.float32)  # (48, rank)
    Z_act = (U_pron.T @ pron_res[:, active]).T.astype(np.float32)  # (N_act, rank)

    rng = np.random.default_rng(2026)
    rem = Z_act.copy()
    stage_codebooks_q: list[np.ndarray] = []
    stage_scales = [0.0, 0.0, 0.0]
    stage_indices_full: list[np.ndarray] = []

    for s_idx, k_size in enumerate(rvq_stages):
        cb_f, codes_a = kmeans2(rem, k_size, minit="++", iter=35, seed=rng)
        sc = float(np.max(np.abs(cb_f)) / 127.0 + 1e-6)
        cb_q = np.clip(np.round(cb_f / sc), -127, 127).astype(np.int8)
        cb_deq = cb_q.astype(np.float32) * sc
        rem = rem - cb_deq[codes_a]

        stage_scales[s_idx] = sc
        stage_codebooks_q.append(cb_q)
        idx_arr = np.zeros(n_frames, dtype=np.uint8)
        idx_arr[active] = codes_a.astype(np.uint8)
        stage_indices_full.append(idx_arr)

    cb_ap_f, idx_ap_a = kmeans2(ap_ratio[:, active].T.astype(np.float32), n_ap_cb, minit="++", iter=25, seed=rng)
    cb_ap_q = np.clip(np.round(cb_ap_f * 255.0), 0, 255).astype(np.uint8)
    idx_ap = np.zeros(n_frames, dtype=np.uint8)
    idx_ap[active] = idx_ap_a.astype(np.uint8)

    # Quantize continuous trajectories: RMS, F0 (4-cent precision), Gain (effort driver)
    rms_q = np.zeros(n_frames, dtype=np.uint8)
    rms_q[active] = np.clip(np.round((20.0 * np.log10(frame_rms[active] + 1e-7) + 75.0) * 3.2), 1, 255).astype(np.uint8)

    f0_q = np.zeros(n_frames, dtype=np.uint16)
    f0_q[voiced] = np.clip(np.round(12.0 * np.log2(f0[voiced] / 110.0) * 24.0) + 1, 1, 1600).astype(np.uint16)

    gain_q = np.zeros(n_frames, dtype=np.uint8)
    gain_q[active] = np.clip(np.round((gain[0, active] + 14.0) * 14.0), 0, 255).astype(np.uint8)

    # Pack WVV5 binary payload
    st_sizes = [rvq_stages[i] if i < len(rvq_stages) else 0 for i in range(3)]
    header = struct.pack(
        "<4sIBBBB3H3H",
        b"WVV5",
        n_frames,
        n_mels,
        rank,
        len(rvq_stages),
        n_ap_cb,
        st_sizes[0],
        st_sizes[1],
        st_sizes[2],
        0,
        0,
        0,
    )
    parts = [
        header,
        np.array(stage_scales, dtype="<f4").tobytes(),
        U_pron.astype("<f4").tobytes(),
    ]
    for cb_q in stage_codebooks_q:
        parts.append(cb_q.tobytes())
    parts.append(cb_ap_q.tobytes())

    d_rms = np.diff(rms_q.astype(np.int16), prepend=0).astype(np.uint8)
    d_f0 = np.diff(f0_q.astype(np.int16), prepend=0).astype("<i2")
    d_gain = np.diff(gain_q.astype(np.int16), prepend=0).astype(np.uint8)
    parts.extend([d_rms.tobytes(), d_f0.tobytes(), d_gain.tobytes()])
    for idx_arr in stage_indices_full:
        parts.append(idx_arr.tobytes())
    parts.append(idx_ap.tobytes())
    raw_wvv5 = b"".join(parts)

    speaker_model = {
        "sampleRate": vsr,
        "hopSamples": hop,
        "maxHarmonics": max_harm,
        "meanVoicedGainLog": round(mean_voiced_gain, 5),
        "melCentersHz": [round(float(x), 2) for x in mel_centers],
        "apCentersHz": [round(float(x), 2) for x in AP_CENTERS_HZ],
        "staticTimbreLogMel": [round(float(x), 5) for x in timbre_static],
        "pitchSlopeLogMel": [round(float(x), 5) for x in timbre_pitch],
        "effortSlopeLogMel": [round(float(x), 5) for x in timbre_effort],
    }

    synth_24k = synthesize_vocal_from_wvv5(raw_wvv5, speaker_model, len(vocal))
    synth_48k = resample_audio(synth_24k, vsr, orig_sr)[: len(vocal_48k)]
    if len(synth_48k) < len(vocal_48k):
        synth_48k = np.pad(synth_48k, (0, len(vocal_48k) - len(synth_48k)))
    return raw_wvv5, speaker_model, synth_48k


def synthesize_vocal_from_wvv5(raw: bytes, model: dict, out_samples: int) -> np.ndarray:
    """Reference bit-exact wave synthesizer matching `src/dsp/vocal-synth.js`."""
    magic, n_frames, n_mels, rank, n_stages, n_ap_cb, s0, s1, s2, _, _, _ = struct.unpack_from("<4sIBBBB3H3H", raw, 0)
    assert magic == b"WVV5"
    stage_sizes = [s0, s1, s2][:n_stages]
    off = 24
    stage_scales = np.frombuffer(raw, dtype="<f4", count=3, offset=off); off += 12
    U_pron = np.frombuffer(raw, dtype="<f4", count=n_mels * rank, offset=off).reshape(n_mels, rank); off += n_mels * rank * 4

    cbs = []
    for s_idx, sz in enumerate(stage_sizes):
        c = np.frombuffer(raw, dtype=np.int8, count=sz * rank, offset=off).reshape(sz, rank).astype(np.float32) * stage_scales[s_idx]
        off += sz * rank
        cbs.append(c)

    cb_ap = np.frombuffer(raw, dtype=np.uint8, count=n_ap_cb * 8, offset=off).reshape(n_ap_cb, 8).astype(np.float32) / 255.0
    off += n_ap_cb * 8

    d_rms = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames
    d_f0 = np.frombuffer(raw, dtype="<i2", count=n_frames, offset=off); off += n_frames * 2
    d_gain = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off); off += n_frames

    rms_q = (np.cumsum(d_rms.astype(np.int32)) & 0xFF).astype(np.uint8)
    f0_q = np.cumsum(d_f0.astype(np.int32)).astype(np.int32)
    gain_q = (np.cumsum(d_gain.astype(np.int32)) & 0xFF).astype(np.uint8)

    p_indices = []
    for _ in range(n_stages):
        p_indices.append(np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off))
        off += n_frames
    ap_idx = np.frombuffer(raw, dtype=np.uint8, count=n_frames, offset=off)

    sr = int(model["sampleRate"])
    hop = int(model["hopSamples"])
    max_harm = int(model["maxHarmonics"])
    mean_vg = float(model["meanVoicedGainLog"])
    mel_centers = np.array(model["melCentersHz"], dtype=np.float32)
    ap_centers = np.array(model["apCentersHz"], dtype=np.float32)
    t_static = np.array(model["staticTimbreLogMel"], dtype=np.float32)
    t_pitch = np.array(model["pitchSlopeLogMel"], dtype=np.float32)
    t_effort = np.array(model["effortSlopeLogMel"], dtype=np.float32)

    active = rms_q > 0
    rms_target = np.zeros(n_frames, dtype=np.float32)
    rms_target[active] = 10.0 ** (((rms_q[active].astype(np.float32) / 3.2) - 75.0) / 20.0)

    f0 = np.zeros(n_frames, dtype=np.float32)
    v_mask = (f0_q > 0) & active
    f0[v_mask] = 110.0 * (2.0 ** ((f0_q[v_mask].astype(np.float32) - 1.0) / 288.0))

    gain_log = (gain_q.astype(np.float32) / 14.0) - 14.0
    Z = np.zeros((n_frames, rank), dtype=np.float32)
    for s_idx, cb in enumerate(cbs):
        Z += cb[p_indices[s_idx]]
    pron_mel = (U_pron @ Z.T)  # (48, n_frames)

    f0_oct = np.zeros(n_frames, dtype=np.float32)
    f0_oct[v_mask] = np.log2(f0[v_mask] / 261.63)
    eff = gain_log - mean_vg

    log_env = t_static[:, None] + t_pitch[:, None] * f0_oct[None, :] + t_effort[:, None] * eff[None, :] + pron_mel
    env_hat = np.exp(np.clip(log_env, -16.0, 12.0))
    env_hat[:, ~active] = 0.0
    ap_hat = cb_ap[ap_idx].T  # (8, n_frames)

    n_ola = 512
    win_ola = np.hanning(n_ola).astype(np.float32)
    ola_freqs = np.linspace(0, sr / 2, n_ola // 2 + 1, dtype=np.float32)
    grid_33 = np.linspace(0, sr / 2, 33, dtype=np.float32)
    ola_noise_norm = float(1.0 / np.sqrt((0.375 * n_ola) / hop))

    h_env_cal = np.zeros_like(env_hat)
    n_spec_cal = np.zeros((len(ola_freqs), n_frames), dtype=np.float32)

    for t in range(n_frames):
        if not active[t]:
            continue
        ef_ola = np.interp(ola_freqs, mel_centers, env_hat[:, t])
        ap_ola = np.interp(ola_freqs, ap_centers, ap_hat[:, t])
        if f0[t] > 80.0:
            K = min(max_harm, int((sr * 0.47) // f0[t]))
            ks = np.arange(1, K + 1, dtype=np.float32)
            ak = np.interp(ks * f0[t], mel_centers, env_hat[:, t])
            apk = np.interp(ks * f0[t], ap_centers, ap_hat[:, t])
            hk = ak * np.sqrt(np.maximum(0.0, 1.0 - apk ** 2))
            harm_var = 0.5 * float(np.sum(hk ** 2))
            ap_frac = float(np.sum((ef_ola * ap_ola) ** 2) / (np.sum(ef_ola ** 2) + 1e-12))
            t_var = float(rms_target[t] ** 2)
            h_env_cal[:, t] = env_hat[:, t] * np.sqrt((t_var * (1.0 - ap_frac)) / (harm_var + 1e-12))
            raw_n = ef_ola * ap_ola
            irfft_rms = np.sqrt((2.0 / (n_ola * n_ola)) * float(np.sum(raw_n[1:-1] ** 2)) + 1e-12)
            n_spec_cal[:, t] = raw_n * (np.sqrt(t_var * ap_frac) / irfft_rms)
        else:
            raw_n = ef_ola
            irfft_rms = np.sqrt((2.0 / (n_ola * n_ola)) * float(np.sum(raw_n[1:-1] ** 2)) + 1e-12)
            n_spec_cal[:, t] = raw_n * (rms_target[t] / irfft_rms)

    out = np.zeros(out_samples, dtype=np.float32)
    phase0 = 0.0
    rng = np.random.default_rng(101)

    for t in range(n_frames - 1):
        s0_i = t * hop
        s1_i = min(out_samples, s0_i + hop)
        L = s1_i - s0_i
        if L <= 0:
            break
        fc, fn = f0[t], f0[t + 1]
        if fc > 80.0 or fn > 80.0:
            f_start = fc if fc > 80.0 else fn
            f_end = fn if fn > 80.0 else fc
            vw0 = 1.0 if fc > 80.0 else 0.0
            vw1 = 1.0 if fn > 80.0 else 0.0
            alpha = np.linspace(0.0, 1.0, L, endpoint=False, dtype=np.float32)
            f_inst = f_start + (f_end - f_start) * alpha
            phi_vec = phase0 + np.cumsum(2.0 * np.pi * f_inst / sr)
            phase0 = float(phi_vec[-1] % (2.0 * np.pi))

            K = min(max_harm, int((sr * 0.47) // max(0.5 * (f_start + f_end), 90.0)))
            ks = np.arange(1, K + 1, dtype=np.float32)
            a0 = np.interp(ks * f_start, mel_centers, h_env_cal[:, t]) * vw0
            a1 = np.interp(ks * f_end, mel_centers, h_env_cal[:, t + 1]) * vw1
            ap0 = np.interp(ks * f_start, ap_centers, ap_hat[:, t])
            ap1 = np.interp(ks * f_end, ap_centers, ap_hat[:, t + 1])
            hg0 = a0 * np.sqrt(np.maximum(0.0, 1.0 - ap0 ** 2))
            hg1 = a1 * np.sqrt(np.maximum(0.0, 1.0 - ap1 ** 2))

            lm0 = np.log(np.maximum(np.interp(grid_33, mel_centers, env_hat[:, t] + 1e-6), 1e-6))
            lm1 = np.log(np.maximum(np.interp(grid_33, mel_centers, env_hat[:, t + 1] + 1e-6), 1e-6))
            mp0 = np.interp(ks * f_start, grid_33, min_phase_64(lm0))
            mp1 = np.interp(ks * f_end, grid_33, min_phase_64(lm1))

            u0, v0 = hg0 * np.cos(mp0), hg0 * np.sin(mp0)
            u1, v1 = hg1 * np.cos(mp1), hg1 * np.sin(mp1)
            u = u0[:, None] + (u1 - u0)[:, None] * alpha[None, :]
            v = v0[:, None] + (v1 - v0)[:, None] * alpha[None, :]
            ph = ks[:, None] * phi_vec[None, :]
            out[s0_i:s1_i] += np.sum(u * np.cos(ph) - v * np.sin(ph), axis=0)

        c_start = s0_i + hop // 2 - n_ola // 2
        if 0 <= c_start and c_start + n_ola <= out_samples and active[t]:
            rph = rng.uniform(0, 2.0 * np.pi, size=len(ola_freqs)).astype(np.float32)
            rph[0] = rph[-1] = 0.0
            grain = np.fft.irfft(n_spec_cal[:, t] * np.exp(1j * rph), n=n_ola).astype(np.float32) * win_ola * ola_noise_norm
            out[c_start : c_start + n_ola] += grain

    return out
