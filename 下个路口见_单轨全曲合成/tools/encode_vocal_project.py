#!/usr/bin/env python3
"""Full-Song 1-Track (`Unified Stereo Master`) Pure-Code Encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.air_envelope_model import encode_high_air_stream
from services.audio_io import decode_stereo, resample_audio
from services.cosine_wave_model import encode_cosine_wave_stream
from services.lrc_parser import parse_lrc
from services.vocal_separator import extract_full_song_1track
from services.wave_packer import pack_vwp6_lzma, sha256_hex, unpack_vwp6_lzma, write_b122_js_module


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=None, help="Optional .npz cache for inspector vocal curve")
    args = parser.parse_args()

    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/full_song_1track_config.json").read_text(encoding="utf-8"))
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

    precomputed = np.load(args.cache) if (args.cache and args.cache.is_file()) else None
    print("[3/4] Preparing Unified 1-Track Full-Song Stereo Master (Mid + Side + Air)...", flush=True)
    sep_res = extract_full_song_1track(master, sr=sr, precomputed=precomputed)

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
    tracks_ref = sep_res["tracks"]
    streams_raw = sep_res["streams"]
    t_meta = config["tracks"][0]

    for profile_id, p_cfg in config["profiles"].items():
        print(f"[4/4] Building Full-Song 1-Track Profile `{profile_id}` ({p_cfg['name']})...", flush=True)
        profile_labels[profile_id] = p_cfg["name"]
        profile_dir = data_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        for stale in profile_dir.glob("*.js"):
            stale.unlink()

        power = float(p_cfg["power"])
        stream_entries = []
        track_comp = 0
        track_js = 0
        track_raw = 0
        track_syn_L = np.zeros(n_samples, dtype=np.float32)
        track_syn_R = np.zeros(n_samples, dtype=np.float32)

        for ch_role, s_key in [("mid", "master_mid"), ("side", "master_side")]:
            raw_vwq6, comp_b, meta, syn_48k = _encode_cosine_substream(
                streams_raw[s_key], sr, p_cfg["streams"][s_key], power
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

        air_bands = [tuple(b) for b in p_cfg["masterAirBandsHz"]]
        raw_vwa7, comp_air, _, syn_air = encode_high_air_stream(
            streams_raw["master_mid"], air_bands, sr=sr, gate_db=float(p_cfg["masterAirGateDb"])
        )
        assert unpack_vwp6_lzma(comp_air) == raw_vwa7
        js_air = write_b122_js_module(profile_dir / "master_air.js", comp_air)
        track_syn_L += syn_air
        track_syn_R += syn_air
        track_comp += len(comp_air)
        track_js += js_air
        track_raw += len(raw_vwa7)
        stream_entries.append(
            {
                "channel": "mono",
                "kind": "air",
                "file": "master_air.vwp",
                "sampleRate": sr,
                "compressedBytes": len(comp_air),
                "parameterBytes": len(raw_vwa7),
                "rawSha256": sha256_hex(raw_vwa7),
            }
        )

        combined_mix = np.stack([track_syn_L, track_syn_R], axis=1)
        mix_wave_snr = compute_wave_snr_db(tracks_ref["master"], combined_mix)
        mix_mel_snr = compute_log_mel_snr_db(tracks_ref["master"], combined_mix, sr=sr)
        kbps = round((track_comp * 8.0) / (duration * 1000.0), 1)
        smaller_vs_wav = round(100.0 * (1.0 - track_comp / lossless_wav_bytes), 2)
        smaller_vs_mp3 = round(100.0 * (1.0 - track_comp / source_audio.stat().st_size), 2)

        print(
            f"        [master streams] mid={stream_entries[0]['compressedBytes']/1024:.1f} KB + side={stream_entries[1]['compressedBytes']/1024:.1f} KB + air={len(comp_air)/1024:.1f} KB",
            flush=True,
        )

        manifest_tracks = [
            {
                "id": t_meta["id"],
                "group": t_meta["group"],
                "name": t_meta["name"],
                "subtitle": t_meta["subtitle"],
                "color": t_meta["color"],
                "stereo": t_meta["stereo"],
                "parametric": True,
                "formula": t_meta["formula"],
                "compressedBytes": track_comp,
                "jsModuleBytes": track_js,
                "parameterBytes": track_raw,
                "waveSnrDb": mix_wave_snr,
                "logMelSnrDb": mix_mel_snr,
                "streams": stream_entries,
            }
        ]

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
            "timbreProfile": sep_res["timbreProfile"],
            "sections": config["sections"],
            "lyrics": lyrics,
            "losslessStereoWavBytes": lossless_wav_bytes,
            "format": "FullSong-1Track-VWP7+VWA7+LZMA+Base122",
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
            "parameterBytes": track_comp,
            "jsModuleBytes": track_js,
            "unpackedParameterBytes": track_raw,
            "kbps": kbps,
            "mixWaveSnrDb": mix_wave_snr,
            "mixLogMelSnrDb": mix_mel_snr,
            "smallerThanWavPercent": smaller_vs_wav,
            "smallerThanMp3Percent": smaller_vs_mp3,
        }
        print(
            f"      => `{profile_id}` Full 1-Track Song: {track_comp:,} B ({track_comp / 1024:.1f} KB = {track_comp / 1e6:.2f} MB, {kbps} kbps, -{smaller_vs_mp3}% vs MP3, -{smaller_vs_wav}% vs WAV) | Mix Wave SNR: {mix_wave_snr:.2f} dB | Mix Mel SNR: {mix_mel_snr:.2f} dB",
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
