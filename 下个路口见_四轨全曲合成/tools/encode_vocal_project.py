#!/usr/bin/env python3
"""Full-Song 4-Track (`1 Merged Stereo Vocal + 3 Accompaniment Stems`) Pure-Code Encoder."""
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
from services.reverb_model import encode_fitted_reverb_stream
from services.vocal_separator import extract_full_song_4tracks
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
    parser.add_argument("--cache", type=Path, default=None, help="Optional .npz cache with precomputed Demucs stems")
    args = parser.parse_args()

    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/full_song_4track_config.json").read_text(encoding="utf-8"))
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

    precomputed = None
    if args.cache and args.cache.is_file():
        print(f"[3/4] Loading precomputed Demucs stems from {args.cache} & splitting 4 tracks...", flush=True)
        precomputed = np.load(args.cache)
    else:
        print("[3/4] Extracting 1 Stereo Vocal + 3 Accompaniment Stems via `htdemucs` on MPS...", flush=True)
    sep_res = extract_full_song_4tracks(master, sr=sr, precomputed=precomputed)
    print(f"      Device: {sep_res['device']} | Full-Song Conservation Error: {sep_res['sumMaxError']:.2e}", flush=True)

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

    for profile_id, p_cfg in config["profiles"].items():
        print(f"[4/4] Building Full-Song 4-Track Profile `{profile_id}` ({p_cfg['name']})...", flush=True)
        profile_labels[profile_id] = p_cfg["name"]
        profile_dir = data_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        for stale in profile_dir.glob("*.js"):
            stale.unlink()

        power = float(p_cfg["power"])
        combined_mix = np.zeros_like(master, dtype=np.float32)

        manifest_tracks = []
        total_comp = 0
        total_js = 0
        total_raw = 0
        vocal_comp_bytes = 0
        acc_comp_bytes = 0

        for t_meta in config["tracks"]:
            tid = t_meta["id"]
            group = t_meta["group"]
            stream_entries = []
            track_comp = 0
            track_js = 0
            track_raw = 0
            track_syn_L = np.zeros(n_samples, dtype=np.float32)
            track_syn_R = np.zeros(n_samples, dtype=np.float32)

            if tid == "vocal":
                raw_mid, comp_mid, meta_mid, syn_mid = _encode_cosine_substream(
                    streams_raw["vocal_mid"], sr, p_cfg["streams"]["vocal_mid"], power
                )
                js_mid = write_b122_js_module(profile_dir / "vocal_mid.js", comp_mid)
                track_syn_L += syn_mid
                track_syn_R += syn_mid
                track_comp += len(comp_mid)
                track_js += js_mid
                track_raw += len(raw_mid)
                stream_entries.append(
                    {
                        "channel": "mid",
                        "kind": "wave",
                        "file": "vocal_mid.vwp",
                        "sampleRate": meta_mid["sampleRate"],
                        "compressedBytes": len(comp_mid),
                        "parameterBytes": len(raw_mid),
                        "rawSha256": sha256_hex(raw_mid),
                        "waveSnrDb": meta_mid["waveSnrDb"],
                    }
                )

                raw_side, comp_side, meta_side, syn_side = _encode_cosine_substream(
                    streams_raw["vocal_side"], sr, p_cfg["streams"]["vocal_side"], power
                )
                js_side = write_b122_js_module(profile_dir / "vocal_side.js", comp_side)
                track_syn_L += syn_side
                track_syn_R -= syn_side
                track_comp += len(comp_side)
                track_js += js_side
                track_raw += len(raw_side)
                stream_entries.append(
                    {
                        "channel": "side",
                        "kind": "wave",
                        "file": "vocal_side.vwp",
                        "sampleRate": meta_side["sampleRate"],
                        "compressedBytes": len(comp_side),
                        "parameterBytes": len(raw_side),
                        "rawSha256": sha256_hex(raw_side),
                        "waveSnrDb": meta_side["waveSnrDb"],
                    }
                )

                vocal_air_bands = [tuple(b) for b in p_cfg["vocalAirBandsHz"]]
                raw_air, comp_air, _, syn_air = encode_high_air_stream(
                    streams_raw["vocal_mid"], vocal_air_bands, sr=sr, gate_db=float(p_cfg["vocalAirGateDb"])
                )
                assert unpack_vwp6_lzma(comp_air) == raw_air
                js_air = write_b122_js_module(profile_dir / "vocal_air.js", comp_air)
                track_syn_L += syn_air
                track_syn_R += syn_air
                track_comp += len(comp_air)
                track_js += js_air
                track_raw += len(raw_air)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "air",
                        "file": "vocal_air.vwp",
                        "sampleRate": sr,
                        "compressedBytes": len(comp_air),
                        "parameterBytes": len(raw_air),
                        "rawSha256": sha256_hex(raw_air),
                    }
                )

                dry_mid_for_fdn = (syn_mid + syn_air).astype(np.float32)
                dry_side_for_fdn = syn_side.astype(np.float32)
                raw_rev, comp_rev, _, syn_rev = encode_fitted_reverb_stream(
                    dry_mid_for_fdn,
                    dry_side_for_fdn,
                    streams_raw["space_mid"],
                    streams_raw["space_side"],
                    sr=sr,
                )
                assert unpack_vwp6_lzma(comp_rev) == raw_rev
                js_rev = write_b122_js_module(profile_dir / "space_fdn.js", comp_rev)
                track_syn_L += syn_rev[:, 0]
                track_syn_R += syn_rev[:, 1]
                track_comp += len(comp_rev)
                track_js += js_rev
                track_raw += len(raw_rev)
                stream_entries.append(
                    {
                        "channel": "stereo",
                        "kind": "reverb",
                        "file": "space_fdn.vwp",
                        "sampleRate": sr,
                        "compressedBytes": len(comp_rev),
                        "parameterBytes": len(raw_rev),
                        "rawSha256": sha256_hex(raw_rev),
                    }
                )
                vocal_comp_bytes = track_comp
                print(
                    f"        [vocal streams] mid={len(comp_mid)/1024:.1f} KB + side={len(comp_side)/1024:.1f} KB + air={len(comp_air)/1024:.1f} KB + fdn={len(comp_rev)/1024:.1f} KB",
                    flush=True,
                )

            elif tid == "bass":
                raw_vwq6, comp_b, meta, syn_48k = _encode_cosine_substream(
                    streams_raw["bass"], sr, p_cfg["streams"]["bass"], power
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
                    streams_raw["drums_body"], sr, p_cfg["streams"]["drums_body"], power
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
                    streams_raw["drums_body"], drum_bands, sr=sr, gate_db=float(p_cfg["drumAirGateDb"])
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
                acc_bands = [tuple(b) for b in p_cfg["accompAirBandsHz"]]
                raw_vwa7, comp_air, _, syn_air = encode_high_air_stream(
                    streams_raw["accomp_mid"], acc_bands, sr=sr, gate_db=float(p_cfg["accompAirGateDb"])
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

            if group == "accompaniment":
                acc_comp_bytes += track_comp

            combined_mix[:, 0] += track_syn_L
            combined_mix[:, 1] += track_syn_R

            track_syn_eval = np.stack([track_syn_L, track_syn_R], axis=1) if t_meta["stereo"] else track_syn_L
            t_wave_snr = compute_wave_snr_db(tracks_ref[tid], track_syn_eval)
            t_mel_snr = compute_log_mel_snr_db(tracks_ref[tid], track_syn_eval, sr=sr)

            total_comp += track_comp
            total_js += track_js
            total_raw += track_raw

            manifest_tracks.append(
                {
                    "id": tid,
                    "group": group,
                    "name": t_meta["name"],
                    "subtitle": t_meta["subtitle"],
                    "color": t_meta["color"],
                    "stereo": t_meta["stereo"],
                    "parametric": tid in ("vocal", "drums", "accompaniment"),
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
                f"      - {tid:16s} [{group[:5]}]: {track_comp / 1024:6.1f} KB LZMA | Wave SNR {t_wave_snr:5.2f} dB | Mel SNR {t_mel_snr:5.2f} dB",
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
            "timbreProfile": sep_res["timbreProfile"],
            "sections": config["sections"],
            "lyrics": lyrics,
            "losslessStereoWavBytes": lossless_wav_bytes,
            "format": "FullSong-4Track-VWP7+VWA7+VWR7+LZMA+Base122",
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
            "vocalBytes": vocal_comp_bytes,
            "accompanimentBytes": acc_comp_bytes,
            "jsModuleBytes": total_js,
            "unpackedParameterBytes": total_raw,
            "kbps": kbps,
            "mixWaveSnrDb": mix_wave_snr,
            "mixLogMelSnrDb": mix_mel_snr,
            "smallerThanWavPercent": smaller_vs_wav,
            "smallerThanMp3Percent": smaller_vs_mp3,
        }
        print(
            f"      => `{profile_id}` Full 4-Track Song: {total_comp:,} B ({total_comp / 1024:.1f} KB = {total_comp / 1e6:.2f} MB, {kbps} kbps, -{smaller_vs_mp3}% vs MP3, -{smaller_vs_wav}% vs WAV) | Mix Wave SNR: {mix_wave_snr:.2f} dB | Mix Mel SNR: {mix_mel_snr:.2f} dB",
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
