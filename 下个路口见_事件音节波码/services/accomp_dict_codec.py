"""V8.5 Dictionary + Event-Scheduled Accompaniment Codec (`VDA8`).

Encodes 48kHz separated accompaniment stems (`bass`, `drums`, `guitar_synth`) using:
- Log-Mel perceptual clustering + Crisp Medoid prototype selection (zero comb-filtering)
- V7 TDAC MDCT (`VWQ6`) + 4-band 8ms High-Frequency Air (`VWA7`) prototype bank compression
- Beat/Half-Beat/Bar event scheduling (`startSample`, `protoId`, `gain`) with smooth crossfade
"""
from __future__ import annotations

import lzma
import struct
import sys
from pathlib import Path

import librosa
import numpy as np

V7_SERVICES = Path("/Users/weckham/Desktop/MUSIC/下个路口见_全曲多轨合成/services")
if str(V7_SERVICES) not in sys.path:
    sys.path.insert(0, str(V7_SERVICES))

from air_envelope_model import encode_high_air_stream
from cosine_wave_model import encode_cosine_wave_stream
from wave_packer import pack_vwp6_lzma
from services.audio_io import resample_audio

VDA8_MAGIC = b"VDA8"


def compute_seg_mel_features(
    X: np.ndarray,
    sr: int = 48000,
    n_mels: int = 32,
    hop: int = 512,
    fmin: float = 50.0,
    fmax: float = 16000.0,
) -> np.ndarray:
    n_seg, d_len = X.shape
    fb = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=n_mels, fmin=fmin, fmax=fmax)
    win = np.hanning(1024).astype(np.float32)
    n_frames = max(1, (d_len - 1024) // hop + 1)
    feats = np.zeros((n_seg, n_mels * n_frames), dtype=np.float32)
    for i in range(n_seg):
        seg = X[i]
        frames = np.lib.stride_tricks.sliding_window_view(seg, 1024)[::hop][:n_frames]
        spec = np.abs(np.fft.rfft(frames * win[None, :], axis=1))
        mel = fb @ spec.T
        feats[i] = np.log1p(mel * 10.0).reshape(-1)
    return feats


def medoid_cluster_by_mel(
    X_wave: np.ndarray,
    F_mel: np.ndarray,
    k: int,
    iters: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_seg = len(X_wave)
    k = min(k, n_seg)
    norms = np.linalg.norm(F_mel, axis=1, keepdims=True) + 1e-9
    fn = F_mel / norms
    idx0 = int(np.argmax(norms[:, 0]))
    centers = [fn[idx0]]
    for _ in range(1, k):
        sims = np.max(fn @ np.stack(centers, axis=0).T, axis=1)
        probs = np.maximum(1.0 - sims, 1e-6) ** 2
        probs /= np.sum(probs)
        centers.append(fn[int(rng.choice(n_seg, p=probs))])
    c_f = np.stack(centers, axis=0)
    labels = np.zeros(n_seg, dtype=np.int32)
    for _ in range(iters):
        sims = fn @ c_f.T
        labels = np.argmax(sims, axis=1)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                v = np.mean(fn[mask], axis=0)
                n = np.linalg.norm(v)
                if n > 1e-9:
                    c_f[j] = v / n
    c_wave = np.zeros((k, X_wave.shape[1]), dtype=np.float32)
    for j in range(k):
        idxs = np.where(labels == j)[0]
        if len(idxs) > 0:
            best_i = idxs[int(np.argmax(fn[idxs] @ c_f[j]))]
            c_wave[j] = X_wave[best_i]
    rms_proto = np.sqrt(np.mean(c_wave ** 2, axis=1) + 1e-12)
    rms_seg = np.sqrt(np.mean(X_wave ** 2, axis=1) + 1e-12)
    gains = np.where(rms_seg < 1e-5, 0.0, rms_seg / np.maximum(rms_proto[labels], 1e-6)).astype(np.float32)
    return c_wave, labels, gains


def synth_segments_simple(
    seg_starts: np.ndarray,
    seg_len: int,
    c_protos: np.ndarray,
    labels: np.ndarray,
    gains: np.ndarray,
    n_samples: int,
    xfade: int = 384,
) -> np.ndarray:
    syn = np.zeros(n_samples, dtype=np.float32)
    t_xf = np.linspace(0.0, np.pi, xfade, dtype=np.float32)
    w_in = 0.5 * (1.0 - np.cos(t_xf))
    w_out = 1.0 - w_in
    for i, pos in enumerate(seg_starts):
        p = int(pos)
        proto = gains[i] * c_protos[labels[i]]
        next_p = int(seg_starts[i + 1]) if i + 1 < len(seg_starts) else n_samples
        avail = min(seg_len, n_samples - p, max(0, next_p - p + xfade))
        if avail <= 0:
            continue
        chunk = proto[:avail].copy()
        if i > 0 and avail >= xfade:
            chunk[:xfade] *= w_in
        if i + 1 < len(seg_starts) and avail >= xfade:
            chunk[-xfade:] *= w_out
        syn[p : p + avail] += chunk
    return syn


def encode_vwq6_bank(proto_1d: np.ndarray, spec: dict, power: float, sr: int = 48000) -> tuple[bytes, np.ndarray]:
    target_sr = int(spec["sampleRate"])
    y_sub = resample_audio(proto_1d, sr, target_sr) if target_sr != sr else proto_1d
    raw_vwq6, _, rec_sub = encode_cosine_wave_stream(y_sub, spec, power)
    raw_vwp7 = lzma.decompress(pack_vwp6_lzma(raw_vwq6))
    rec_48k = resample_audio(rec_sub, target_sr, sr)[: len(proto_1d)]
    if len(rec_48k) < len(proto_1d):
        rec_48k = np.pad(rec_48k, (0, len(proto_1d) - len(rec_48k)))
    return raw_vwp7, rec_48k


def pack_vda8_packet(
    num_protos: int,
    seg_len: int,
    stride: int,
    xfade: int,
    n_samples: int,
    streams: list[bytes],
    starts: np.ndarray,
    labels: np.ndarray,
    gains: np.ndarray,
) -> bytes:
    num_events = len(starts)
    header = struct.pack(
        "<4sHHIIHIII",
        VDA8_MAGIC,
        1,
        num_protos,
        seg_len,
        stride,
        xfade,
        num_events,
        n_samples,
        len(streams),
    )
    parts = [header]
    for st in streams:
        parts.append(struct.pack("<I", len(st)))
        parts.append(st)
    for i in range(num_events):
        parts.append(struct.pack("<IHf", int(starts[i]), int(labels[i]), float(gains[i])))
    return b"".join(parts)


def build_beat_grids(drums: np.ndarray, n_samples: int, sr: int = 48000) -> dict[str, np.ndarray | int]:
    _, beats = librosa.beat.beat_track(y=drums, sr=sr, hop_length=512, units="samples")
    med_b = int(round(np.median(np.diff(beats))))
    half_beats = []
    t = int(beats[0])
    pre = []
    while t - med_b // 2 >= 0:
        t -= med_b // 2
        pre.append(t)
    half_beats.extend(reversed(pre))
    for i in range(len(beats) - 1):
        half_beats.append(int(beats[i]))
        half_beats.append(int((beats[i] + beats[i + 1]) // 2))
    t = int(beats[-1])
    while t < n_samples - med_b // 4:
        half_beats.append(t)
        t += med_b // 2
    hb = np.array(half_beats, dtype=np.int32)
    return {
        "med_beat": med_b,
        "half_beats": hb,
        "two_beats": hb[::4],
        "four_beats": hb[::8],
    }


def encode_dict_track_mono(
    signal_1d: np.ndarray,
    starts: np.ndarray,
    active_len: int,
    k_protos: int,
    xfade: int,
    body_spec: dict,
    power: float,
    air_bands_hz: list | None = None,
    air_gate_db: float = -60.0,
    mel_fmin: float = 50.0,
    mel_fmax: float = 16000.0,
    mel_bins: int = 32,
    mel_hop: int = 512,
    sr: int = 48000,
) -> tuple[bytes, np.ndarray]:
    n_samples = len(signal_1d)
    seg_len = active_len + 512
    X = np.zeros((len(starts), seg_len), dtype=np.float32)
    for i, pos in enumerate(starts):
        end = min(n_samples, int(pos) + seg_len)
        X[i, : end - int(pos)] = signal_1d[int(pos):end]
    feats = compute_seg_mel_features(X, sr=sr, n_mels=mel_bins, hop=mel_hop, fmin=mel_fmin, fmax=mel_fmax)
    c_wave, labels, gains = medoid_cluster_by_mel(X, feats, k_protos)
    p_scales = np.where(np.max(np.abs(c_wave), axis=1) / 0.5 < 1e-6, 1.0, np.max(np.abs(c_wave), axis=1) / 0.5)
    c_norm = c_wave / p_scales[:, None]
    gains_scaled = gains * p_scales[labels]

    pad = 512
    stride = seg_len + pad
    bank = np.zeros(len(c_wave) * stride, dtype=np.float32)
    for j in range(len(c_wave)):
        bank[j * stride : j * stride + seg_len] = c_norm[j]

    raw_body, rec_body = encode_vwq6_bank(bank, body_spec, power, sr=sr)
    streams = [raw_body]
    rec_total = rec_body
    if air_bands_hz is not None:
        _, comp_air, _, rec_air = encode_high_air_stream(bank - rec_body, air_bands_hz, sr=sr, gate_db=air_gate_db)
        streams.append(lzma.decompress(comp_air))
        rec_total = rec_body + rec_air

    c_rec = np.stack([rec_total[j * stride : j * stride + seg_len] for j in range(len(c_wave))], axis=0)
    syn = synth_segments_simple(starts, active_len, c_rec, labels, gains_scaled, n_samples, xfade=xfade)
    pkt = pack_vda8_packet(len(c_wave), active_len, stride, xfade, n_samples, streams, starts, labels, gains_scaled)
    return pkt, syn


def encode_dict_track_stereo_ms(
    mid_1d: np.ndarray,
    side_1d: np.ndarray,
    starts: np.ndarray,
    active_len: int,
    k_protos: int,
    xfade: int,
    mid_spec: dict,
    side_spec: dict,
    power: float,
    air_bands_hz: list,
    air_gate_db: float,
    sr: int = 48000,
) -> tuple[bytes, np.ndarray]:
    n_samples = len(mid_1d)
    seg_len = active_len + 512
    Xm = np.zeros((len(starts), seg_len), dtype=np.float32)
    Xs = np.zeros((len(starts), seg_len), dtype=np.float32)
    for i, pos in enumerate(starts):
        end = min(n_samples, int(pos) + seg_len)
        w = end - int(pos)
        Xm[i, :w] = mid_1d[int(pos):end]
        Xs[i, :w] = side_1d[int(pos):end]
    feats = compute_seg_mel_features(Xm, sr=sr, n_mels=32, hop=1024, fmin=50.0, fmax=16000.0)
    Xj = np.concatenate([Xm, Xs], axis=1)
    c_wave, labels, gains = medoid_cluster_by_mel(Xj, feats, k_protos)
    p_scales = np.where(np.max(np.abs(c_wave), axis=1) / 0.5 < 1e-6, 1.0, np.max(np.abs(c_wave), axis=1) / 0.5)
    c_norm = c_wave / p_scales[:, None]
    gains_scaled = gains * p_scales[labels]

    pad = 512
    stride = seg_len + pad
    bank_m = np.zeros(len(c_wave) * stride, dtype=np.float32)
    bank_s = np.zeros(len(c_wave) * stride, dtype=np.float32)
    for j in range(len(c_wave)):
        bank_m[j * stride : j * stride + seg_len] = c_norm[j, :seg_len]
        bank_s[j * stride : j * stride + seg_len] = c_norm[j, seg_len:]

    raw_mid, rec_mid = encode_vwq6_bank(bank_m, mid_spec, power, sr=sr)
    raw_side, rec_side = encode_vwq6_bank(bank_s, side_spec, power, sr=sr)
    _, comp_air, _, rec_air = encode_high_air_stream(bank_m - rec_mid, air_bands_hz, sr=sr, gate_db=air_gate_db)
    raw_air = lzma.decompress(comp_air)

    cm_rec = np.stack([(rec_mid + rec_air)[j * stride : j * stride + seg_len] for j in range(len(c_wave))], axis=0)
    cs_rec = np.stack([rec_side[j * stride : j * stride + seg_len] for j in range(len(c_wave))], axis=0)
    syn_m = synth_segments_simple(starts, active_len, cm_rec, labels, gains_scaled, n_samples, xfade=xfade)
    syn_s = synth_segments_simple(starts, active_len, cs_rec, labels, gains_scaled, n_samples, xfade=xfade)
    syn_st = np.stack([syn_m + syn_s, syn_m - syn_s], axis=1).astype(np.float32)
    pkt = pack_vda8_packet(len(c_wave), active_len, stride, xfade, n_samples, [raw_mid, raw_side, raw_air], starts, labels, gains_scaled)
    return pkt, syn_st
