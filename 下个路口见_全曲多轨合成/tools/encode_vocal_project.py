#!/usr/bin/env python3
"""Full-Song 8-Track (`5 Vocal Sub-Tracks + 3 Accompaniment Stems`) Pure-Code Encoder."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import sys
import time
import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VOCAL_5TRACK_ROOT = WORKSPACE_ROOT / "下个路口见_人声多轨波码"
sys.path.insert(0, str(PROJECT_ROOT))

from services.air_envelope_model import AIR_MIN_DB, AIR_STEP_DB, design_2stage_bandpass, encode_high_air_stream
from services.audio_io import decode_stereo, resample_audio
from services.cosine_wave_model import (
    LONG_BINS,
    LONG_WINDOWS,
    MODE_FOUR_SHORT,
    MODE_LONG_START,
    MODE_LONG_STOP,
    MODE_ONLY_LONG,
    MODE_STOP_START,
    PAD_ZEROS,
    SCALE_LUT,
    SHORT_BINS,
    SHORT_COUNT,
    SHORT_WINDOW,
    encode_cosine_wave_stream,
    get_band_edges,
    get_band_qmax_table,
    imdct_fast_batch,
)
from services.lrc_parser import parse_lrc
from services.reverb_model import GAIN_MIN_DB, GAIN_STEP_DB, run_raw_fdn_stereo
from services.vocal_separator import extract_accompaniment_3tracks
from services.wave_packer import decode_base122, pack_vwp6_lzma, sha256_hex, unpack_vwp6_lzma, write_b122_js_module
from scipy.signal import lfilter


def compute_wave_snr_db(ref: np.ndarray, syn: np.ndarray) -> float:
    L = min(len(ref), len(syn))
    r = ref[:L].astype(np.float64)
    s = syn[:L].astype(np.float64)
    return round(float(10.0 * np.log10((np.sum(r ** 2) + 1e-15) / (np.sum((r - s) ** 2) + 1e-15))), 2)


def compute_log_mel_snr_db(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> float:
    if ref.ndim == 2:
        r_m = 0.5 * (ref[:, 0] + ref[:, 1])
        s_m = 0.5 * (syn[:, 0] + syn[:, 1])
    else:
        r_m, s_m = ref, syn
    L = min(len(r_m), len(s_m))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=45.0, fmax=19000.0)
    Mr = fb @ np.abs(librosa.stft(r_m[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s_m[:L], n_fft=2048, hop_length=512))
    floor = float(np.max(Mr) * 1e-3 + 1e-8)
    lr = np.log10(np.maximum(Mr, floor))
    ls = np.log10(np.maximum(Ms, floor))
    return round(float(10.0 * np.log10((np.sum(lr ** 2) + 1e-12) / (np.sum((lr - ls) ** 2) + 1e-12))), 2)


def _decode_runtime_stream_48k(raw_pkt: bytes, n_48k: int, dry_mid: np.ndarray | None = None, dry_side: np.ndarray | None = None):
    magic = raw_pkt[:4]
    if magic == b"VWQ6":
        _, sr, long_n, short_n, t_count, samples, base_qmax, p100, *_ = struct.unpack("<4sIHHIIBBBBBBHI", raw_pkt[:32])
        inv_power = 100.0 / p100
        stride = 1 + 48 + long_n
        body = np.frombuffer(raw_pkt, np.uint8, offset=32).reshape(t_count, stride)
        modes = body[:, 0]
        scales = body[:, 1:49]
        q_mat = body[:, 49:].view(np.int8)

        edges_l = get_band_edges(long_n)
        edges_s = get_band_edges(short_n)
        qm_l = get_band_qmax_table(base_qmax, len(edges_l) - 1)
        qm_s = get_band_qmax_table(base_qmax, len(edges_s) - 1)

        rec = np.zeros((t_count, long_n), dtype=np.float32)
        long_mask = modes != MODE_FOUR_SHORT
        if np.any(long_mask):
            for g in range(24):
                s_k, e_k = int(edges_l[g]), int(edges_l[g + 1])
                sc = SCALE_LUT[scales[long_mask, g]][:, None]
                q = q_mat[long_mask, s_k:e_k]
                rec[long_mask, s_k:e_k] = np.sign(q) * ((np.abs(q).astype(np.float32) / qm_l[g]) ** inv_power) * sc

        short_mask = modes == MODE_FOUR_SHORT
        if np.any(short_mask):
            for sub_i in range(4):
                for g in range(12):
                    s_k = sub_i * short_n + int(edges_s[g])
                    e_k = sub_i * short_n + int(edges_s[g + 1])
                    sc = SCALE_LUT[scales[short_mask, sub_i * 12 + g]][:, None]
                    q = q_mat[short_mask, s_k:e_k]
                    rec[short_mask, s_k:e_k] = np.sign(q) * ((np.abs(q).astype(np.float32) / qm_s[g]) ** inv_power) * sc

        out = np.zeros((t_count + 1) * long_n, dtype=np.float64)
        for m_val in (MODE_ONLY_LONG, MODE_LONG_START, MODE_LONG_STOP, MODE_STOP_START):
            idx = np.flatnonzero(modes == m_val)
            if len(idx) == 0:
                continue
            y_blk = imdct_fast_batch(rec[idx], long_n) * LONG_WINDOWS[m_val][None, :]
            for row, t in enumerate(idx):
                pos = int(t) * long_n
                out[pos : pos + 2 * long_n] += y_blk[row]

        s_idx = np.flatnonzero(short_mask)
        if len(s_idx) > 0:
            for k in range(SHORT_COUNT):
                off = PAD_ZEROS + k * short_n
                y_blk = imdct_fast_batch(rec[s_idx, k * short_n : (k + 1) * short_n], short_n) * SHORT_WINDOW[None, :]
                for row, t in enumerate(s_idx):
                    pos = int(t) * long_n + off
                    out[pos : pos + 2 * short_n] += y_blk[row]

        y_sub = out[long_n : long_n + samples].astype(np.float32)
        y_48 = resample_audio(y_sub, sr, 48000)[:n_48k]
        return np.pad(y_48, (0, max(0, n_48k - len(y_48))))

    if magic == b"VWA7":
        _, sr, n, hop, n_hops, *b_u16 = struct.unpack("<4sIIII8H", raw_pkt[:36])
        codes = np.frombuffer(raw_pkt, np.uint8, offset=36).reshape(n_hops, 4)
        rng = np.random.default_rng(20261115)
        idx = np.arange(n_48k, dtype=np.float64) / hop
        h0 = np.minimum(np.floor(idx).astype(np.int32), n_hops - 1)
        h1 = np.minimum(h0 + 1, n_hops - 1)
        frac = (idx - h0).astype(np.float32)
        syn = np.zeros(n_48k, dtype=np.float32)
        for b_i in range(4):
            f_lo, f_hi = b_u16[2 * b_i], b_u16[2 * b_i + 1]
            b0, a1, a2, inv_norm = design_2stage_bandpass(f_lo, f_hi, sr=48000)
            b = [b0, 0.0, -b0]
            a = [1.0, a1, a2]
            dec = np.where(
                codes[:, b_i] > 0,
                10.0 ** ((AIR_MIN_DB + (codes[:, b_i].astype(np.float32) - 1.0) * AIR_STEP_DB) / 20.0),
                0.0,
            ).astype(np.float32)
            env = dec[h0] * (1.0 - frac) + dec[h1] * frac
            syn += (lfilter(b, a, lfilter(b, a, rng.standard_normal(n_48k))) * inv_norm).astype(np.float32) * env
        return syn

    if magic == b"VWR7":
        _, sr, n, env_hop, n_hops, rt_lo_ms, rt_hi_ms, pre_d, *_ = struct.unpack("<4sIIIIHHHHI", raw_pkt[:32])
        cm = np.frombuffer(raw_pkt, np.uint8, offset=32, count=n_hops)
        cs = np.frombuffer(raw_pkt, np.uint8, offset=32 + n_hops, count=n_hops)
        dm = np.where(cm > 0, 10.0 ** ((GAIN_MIN_DB + (cm.astype(np.float32) - 1.0) * GAIN_STEP_DB) / 20.0), 0.0)
        ds = np.where(cs > 0, 10.0 ** ((GAIN_MIN_DB + (cs.astype(np.float32) - 1.0) * GAIN_STEP_DB) / 20.0), 0.0)
        raw_m, raw_s = run_raw_fdn_stereo(dry_mid, dry_side, sr=48000, rt60_low=rt_lo_ms / 1000.0, rt60_high=rt_hi_ms / 1000.0, pre_delay_samples=pre_d)
        idx = np.arange(n_48k, dtype=np.float64) / env_hop
        h0 = np.minimum(np.floor(idx).astype(np.int32), n_hops - 1)
        h1 = np.minimum(h0 + 1, n_hops - 1)
        frac = (idx - h0).astype(np.float32)
        em = dm[h0] * (1.0 - frac) + dm[h1] * frac
        es = ds[h0] * (1.0 - frac) + ds[h1] * frac
        return np.stack([raw_m * em + raw_s * es, raw_m * em - raw_s * es], axis=1).astype(np.float32)

    raise ValueError(f"Unknown magic {magic}")


def _encode_cosine_substream(y_48k: np.ndarray, orig_sr: int, spec: dict, power: float):
    target_sr = int(spec["sampleRate"])
    y_sub = resample_audio(y_48k, orig_sr, target_sr) if target_sr != orig_sr else y_48k
    raw_vwq6, meta, rec_sub = encode_cosine_wave_stream(y_sub, spec, power)
    comp_bytes = pack_vwp6_lzma(raw_vwq6)
    rec_48k = resample_audio(rec_sub, target_sr, orig_sr)[: len(y_48k)]
    if len(rec_48k) < len(y_48k):
        rec_48k = np.pad(rec_48k, (0, len(y_48k) - len(rec_48k)))
    return raw_vwq6, comp_bytes, meta, rec_48k


def main() -> None:
    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/full_song_multitrack_config.json").read_text(encoding="utf-8"))
    source_audio = WORKSPACE_ROOT / config["sourceAudio"]
    source_lrc = WORKSPACE_ROOT / config["sourceLrc"]
    sr = int(config["outputSampleRate"])

    print(f"[1/4] Decoding {source_audio.name} at {sr} Hz...", flush=True)
    master = decode_stereo(source_audio, sr=sr)
    n_samples = len(master)
    duration = n_samples / sr
    lossless_wav_bytes = 44 + n_samples * 2 * 2

    print(f"[2/4] Parsing aligned lyrics from {source_lrc.name}...", flush=True)
    lyrics = parse_lrc(source_lrc, duration)

    print("[3/4] Extracting Accompaniment Stems (`bass`, `drums`, `accompaniment`) via `htdemucs` on MPS...", flush=True)
    acc_res = extract_accompaniment_3tracks(master, sr=sr)
    print(f"      Device: {acc_res['device']} | Full-Song Conservation Error: {acc_res['sumMaxError']:.2e}", flush=True)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sizes_summary = {
        "version": 7,
        "architecture": config["architecture"],
        "title": config["title"],
        "artist": config["artist"],
        "duration": round(duration, 6),
        "referenceMp3Bytes": source_audio.stat().st_size,
        "losslessStereoWavBytes": lossless_wav_bytes,
        "standardAudioFilesIncluded": False,
        "accompanimentIncluded": True,
    }

    profile_labels = {}
    acc_tracks_ref = acc_res["tracks"]
    acc_streams_raw = acc_res["streams"]

    for profile_id, p_cfg in config["profiles"].items():
        print(f"[4/4] Building Full-Song 8-Track Profile `{profile_id}` ({p_cfg['name']})...", flush=True)
        profile_labels[profile_id] = p_cfg["name"]
        profile_dir = data_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        for stale in profile_dir.glob("*.js"):
            stale.unlink()

        vocal_src_dir = VOCAL_5TRACK_ROOT / "data" / profile_id
        vocal_manifest = json.loads((vocal_src_dir / "manifest.json").read_text(encoding="utf-8"))
        vocal_by_id = {t["id"]: t for t in vocal_manifest["tracks"]}

        power = float(p_cfg["power"])
        combined_mix = np.zeros_like(master, dtype=np.float32)
        vocal_dry_L = np.zeros(n_samples, dtype=np.float32)
        vocal_dry_R = np.zeros(n_samples, dtype=np.float32)

        manifest_tracks = []
        total_comp = 0
        total_js = 0
        total_raw = 0

        for t_meta in config["tracks"]:
            tid = t_meta["id"]
            group = t_meta["group"]

            if group == "vocal":
                v_entry = vocal_by_id[tid]
                for s_item in v_entry["streams"]:
                    js_name = s_item["file"].replace(".vwp", ".js")
                    shutil.copy2(vocal_src_dir / js_name, profile_dir / js_name)
                    txt = (profile_dir / js_name).read_bytes().decode("utf-8")
                    comp_b = decode_base122(txt[txt.index('"') + 1 : txt.rindex('"')])
                    raw_pkt = unpack_vwp6_lzma(comp_b)
                    if s_item.get("kind") == "reverb":
                        rev_st = _decode_runtime_stream_48k(
                            raw_pkt,
                            n_samples,
                            (0.5 * (vocal_dry_L + vocal_dry_R)).astype(np.float32),
                            (0.5 * (vocal_dry_L - vocal_dry_R)).astype(np.float32),
                        )
                        combined_mix += rev_st
                    else:
                        sig = _decode_runtime_stream_48k(raw_pkt, n_samples)
                        if s_item["channel"] in ("mono", "mid"):
                            vocal_dry_L += sig
                            vocal_dry_R += sig
                            combined_mix[:, 0] += sig
                            combined_mix[:, 1] += sig
                        else:
                            vocal_dry_L += sig
                            vocal_dry_R -= sig
                            combined_mix[:, 0] += sig
                            combined_mix[:, 1] -= sig

                total_comp += v_entry["compressedBytes"]
                total_js += v_entry["jsModuleBytes"]
                total_raw += v_entry["parameterBytes"]

                manifest_tracks.append(
                    {
                        "id": tid,
                        "group": "vocal",
                        "name": t_meta["name"],
                        "subtitle": t_meta["subtitle"],
                        "color": t_meta["color"],
                        "stereo": t_meta["stereo"],
                        "parametric": v_entry.get("parametric", False),
                        "formula": t_meta["formula"],
                        "compressedBytes": v_entry["compressedBytes"],
                        "jsModuleBytes": v_entry["jsModuleBytes"],
                        "parameterBytes": v_entry["parameterBytes"],
                        "waveSnrDb": v_entry["waveSnrDb"],
                        "logMelSnrDb": v_entry["logMelSnrDb"],
                        "streams": v_entry["streams"],
                    }
                )
                print(
                    f"      - {tid:16s} [vocal]: {v_entry['compressedBytes'] / 1024:6.1f} KB LZMA | Mel SNR {v_entry['logMelSnrDb']:5.2f} dB",
                    flush=True,
                )
                continue

            # Accompaniment tracks (`bass`, `drums`, `accompaniment`)
            stream_entries = []
            track_comp = 0
            track_js = 0
            track_raw = 0
            track_syn_L = np.zeros(n_samples, dtype=np.float32)
            track_syn_R = np.zeros(n_samples, dtype=np.float32)

            if tid == "bass":
                raw_vwq6, comp_b, meta, syn_48k = _encode_cosine_substream(
                    acc_streams_raw["bass"], sr, p_cfg["streams"]["bass"], power
                )
                js_b = write_b122_js_module(profile_dir / "bass.js", comp_b)
                track_syn_L += syn_48k
                track_syn_R += syn_48k
                track_comp += len(comp_b)
                track_js += js_b
                track_raw += len(raw_vwq6)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "wave",
                        "file": "bass.vwp",
                        "sampleRate": meta["sampleRate"],
                        "compressedBytes": len(comp_b),
                        "parameterBytes": len(raw_vwq6),
                        "rawSha256": sha256_hex(raw_vwq6),
                        "waveSnrDb": meta["waveSnrDb"],
                    }
                )

            elif tid == "drums":
                raw_vwq6, comp_b, meta, syn_body = _encode_cosine_substream(
                    acc_streams_raw["drums_body"], sr, p_cfg["streams"]["drums_body"], power
                )
                js_b = write_b122_js_module(profile_dir / "drums_body.js", comp_b)
                track_syn_L += syn_body
                track_syn_R += syn_body
                track_comp += len(comp_b)
                track_js += js_b
                track_raw += len(raw_vwq6)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "wave",
                        "file": "drums_body.vwp",
                        "sampleRate": meta["sampleRate"],
                        "compressedBytes": len(comp_b),
                        "parameterBytes": len(raw_vwq6),
                        "rawSha256": sha256_hex(raw_vwq6),
                        "waveSnrDb": meta["waveSnrDb"],
                    }
                )
                drum_bands = [tuple(b) for b in p_cfg["drumAirBandsHz"]]
                raw_vwa7, comp_air, _, syn_air = encode_high_air_stream(
                    acc_streams_raw["drums_body"], drum_bands, sr=sr, gate_db=float(p_cfg["drumAirGateDb"])
                )
                assert unpack_vwp6_lzma(comp_air) == raw_vwa7
                js_air = write_b122_js_module(profile_dir / "drums_air.js", comp_air)
                track_syn_L += syn_air
                track_syn_R += syn_air
                track_comp += len(comp_air)
                track_js += js_air
                track_raw += len(raw_vwa7)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "air",
                        "file": "drums_air.vwp",
                        "sampleRate": sr,
                        "compressedBytes": len(comp_air),
                        "parameterBytes": len(raw_vwa7),
                        "rawSha256": sha256_hex(raw_vwa7),
                    }
                )

            elif tid == "accompaniment":
                for ch_role, s_key in [("mid", "accomp_mid"), ("side", "accomp_side")]:
                    raw_vwq6, comp_b, meta, syn_48k = _encode_cosine_substream(
                        acc_streams_raw[s_key], sr, p_cfg["streams"][s_key], power
                    )
                    js_b = write_b122_js_module(profile_dir / f"{s_key}.js", comp_b)
                    if ch_role == "mid":
                        track_syn_L += syn_48k
                        track_syn_R += syn_48k
                    else:
                        track_syn_L += syn_48k
                        track_syn_R -= syn_48k
                    track_comp += len(comp_b)
                    track_js += js_b
                    track_raw += len(raw_vwq6)
                    stream_entries.append(
                        {
                            "channel": ch_role,
                            "kind": "wave",
                            "file": f"{s_key}.vwp",
                            "sampleRate": meta["sampleRate"],
                            "compressedBytes": len(comp_b),
                            "parameterBytes": len(raw_vwq6),
                            "rawSha256": sha256_hex(raw_vwq6),
                            "waveSnrDb": meta["waveSnrDb"],
                        }
                    )
                acc_bands = [tuple(b) for b in p_cfg["accompAirBandsHz"]]
                raw_vwa7, comp_air, _, syn_air = encode_high_air_stream(
                    acc_streams_raw["accomp_mid"], acc_bands, sr=sr, gate_db=float(p_cfg["accompAirGateDb"])
                )
                assert unpack_vwp6_lzma(comp_air) == raw_vwa7
                js_air = write_b122_js_module(profile_dir / "accomp_air.js", comp_air)
                track_syn_L += syn_air
                track_syn_R += syn_air
                track_comp += len(comp_air)
                track_js += js_air
                track_raw += len(raw_vwa7)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "air",
                        "file": "accomp_air.vwp",
                        "sampleRate": sr,
                        "compressedBytes": len(comp_air),
                        "parameterBytes": len(raw_vwa7),
                        "rawSha256": sha256_hex(raw_vwa7),
                    }
                )

            combined_mix[:, 0] += track_syn_L
            combined_mix[:, 1] += track_syn_R

            track_syn_eval = np.stack([track_syn_L, track_syn_R], axis=1) if t_meta["stereo"] else track_syn_L
            t_wave_snr = compute_wave_snr_db(acc_tracks_ref[tid], track_syn_eval)
            t_mel_snr = compute_log_mel_snr_db(acc_tracks_ref[tid], track_syn_eval, sr=sr)

            total_comp += track_comp
            total_js += track_js
            total_raw += track_raw

            manifest_tracks.append(
                {
                    "id": tid,
                    "group": "accompaniment",
                    "name": t_meta["name"],
                    "subtitle": t_meta["subtitle"],
                    "color": t_meta["color"],
                    "stereo": t_meta["stereo"],
                    "parametric": tid in ("drums", "accompaniment"),
                    "formula": t_meta["formula"],
                    "compressedBytes": track_comp,
                    "jsModuleBytes": track_js,
                    "parameterBytes": track_raw,
                    "waveSnrDb": t_wave_snr,
                    "logMelSnrDb": t_mel_snr,
                    "streams": stream_entries,
                }
            )
            print(
                f"      - {tid:16s} [band ]: {track_comp / 1024:6.1f} KB LZMA | Wave SNR {t_wave_snr:5.2f} dB | Mel SNR {t_mel_snr:5.2f} dB",
                flush=True,
            )

        mix_wave_snr = compute_wave_snr_db(master, combined_mix)
        mix_mel_snr = compute_log_mel_snr_db(master, combined_mix, sr=sr)
        kbps = round((total_comp * 8.0) / (duration * 1000.0), 1)
        smaller_vs_wav = round(100.0 * (1.0 - total_comp / lossless_wav_bytes), 2)
        smaller_vs_mp3 = round(100.0 * (1.0 - total_comp / source_audio.stat().st_size), 2)

        manifest = {
            "title": f"{config['title']} · {p_cfg['name']}",
            "artist": config["artist"],
            "arranger": config["arranger"],
            "duration": round(duration, 6),
            "outputSampleRate": sr,
            "kbps": kbps,
            "mixWaveSnrDb": mix_wave_snr,
            "mixLogMelSnrDb": mix_mel_snr,
            "presets": config["presets"],
            "tracks": manifest_tracks,
            "timbreProfile": vocal_manifest["timbreProfile"],
            "sections": config["sections"],
            "lyrics": lyrics,
            "losslessStereoWavBytes": lossless_wav_bytes,
            "format": "FullSong-8Track-VWP7+VWA7+VWR7+LZMA+Base122",
            "version": 7,
            "profile": profile_id,
            "profileName": p_cfg["name"],
            "description": p_cfg["description"],
        }
        (profile_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        sizes_summary[profile_id] = {
            "profileName": p_cfg["name"],
            "parameterBytes": total_comp,
            "jsModuleBytes": total_js,
            "unpackedParameterBytes": total_raw,
            "kbps": kbps,
            "mixWaveSnrDb": mix_wave_snr,
            "mixLogMelSnrDb": mix_mel_snr,
            "smallerThanWavPercent": smaller_vs_wav,
            "smallerThanMp3Percent": smaller_vs_mp3,
        }
        print(
            f"      => `{profile_id}` Full 8-Track Song: {total_comp:,} B ({total_comp / 1e6:.2f} MB, {kbps} kbps, -{smaller_vs_mp3}% vs MP3, -{smaller_vs_wav}% vs WAV) | Mix Mel SNR: {mix_mel_snr:.2f} dB",
            flush=True,
        )

    (data_dir / "profiles.json").write_text(
        json.dumps(
            {
                "default": "compact",
                "available": list(config["profiles"].keys()),
                "labels": profile_labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (data_dir / "sizes.json").write_text(
        json.dumps(sizes_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
