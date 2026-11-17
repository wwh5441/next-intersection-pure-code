#!/usr/bin/env python3
"""End-to-End V7 Ultra-Lean 5-Track High-Fidelity Pure-Vocal Wave-Code Encoder."""
from __future__ import annotations

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
from services.vocal_separator import extract_and_split_vocal_5tracks
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
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=55.0, fmax=19000.0)
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
    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/vocal_multitrack_config.json").read_text(encoding="utf-8"))
    source_audio = WORKSPACE_ROOT / config["sourceAudio"]
    source_lrc = WORKSPACE_ROOT / config["sourceLrc"]
    sr = int(config["outputSampleRate"])

    print(f"[1/4] Decoding {source_audio.name} at {sr} Hz...", flush=True)
    master = decode_stereo(source_audio, sr=sr)
    duration = len(master) / sr
    lossless_wav_bytes = 44 + len(master) * 2 * 2

    print(f"[2/4] Parsing aligned lyrics from {source_lrc.name}...", flush=True)
    lyrics = parse_lrc(source_lrc, duration)

    print("[3/4] Extracting 48kHz Stereo Vocal (`htdemucs` shifts=1 on MPS) & Splitting 5 Exact-Sum Vocal Sub-Tracks...", flush=True)
    sep_res = extract_and_split_vocal_5tracks(master, sr=sr)
    print(
        f"      Device: {sep_res['device']} | 5-Track Sum Max Error: {sep_res['sumMaxError']:.2e} | Vocal Entry: {sep_res['firstVocalSec']}s",
        flush=True,
    )

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
        "v6CompactBytes": 2037451,
        "standardAudioFilesIncluded": False,
        "accompanimentIncluded": False,
    }

    profile_labels = {}
    ref_tracks = sep_res["tracks"]
    raw_streams = sep_res["streams"]
    vocals_ref = sep_res["vocalsStereo"]
    n_samples = len(vocals_ref)

    for profile_id, p_cfg in config["profiles"].items():
        print(f"[4/4] Encoding V7 Ultra-Lean 5-Track Vocal `{profile_id}` ({p_cfg['name']})...", flush=True)
        profile_labels[profile_id] = p_cfg["name"]
        profile_dir = data_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        for stale in profile_dir.glob("*.js"):
            stale.unlink()

        power = float(p_cfg["power"])
        air_bands = [tuple(b) for b in p_cfg["airBandsHz"]]
        air_gate = float(p_cfg["airGateDb"])

        combined_vocal = np.zeros_like(vocals_ref, dtype=np.float32)
        dry_syn_L = np.zeros(n_samples, dtype=np.float32)
        dry_syn_R = np.zeros(n_samples, dtype=np.float32)
        dry_ref_L = (
            ref_tracks["lead_core"]
            + ref_tracks["lead_formant"]
            + ref_tracks["lead_consonant"]
            + ref_tracks["backing_harmony"][:, 0]
        )
        dry_ref_R = (
            ref_tracks["lead_core"]
            + ref_tracks["lead_formant"]
            + ref_tracks["lead_consonant"]
            + ref_tracks["backing_harmony"][:, 1]
        )

        manifest_tracks = []
        total_comp = 0
        total_js = 0
        total_raw = 0

        for t_meta in config["tracks"]:
            tid = t_meta["id"]
            stream_entries = []
            track_comp = 0
            track_js = 0
            track_raw = 0
            track_syn_L = np.zeros(n_samples, dtype=np.float32)
            track_syn_R = np.zeros(n_samples, dtype=np.float32)

            if tid in ("lead_core", "lead_formant"):
                raw_vwq6, comp_b, meta, syn_48k = _encode_cosine_substream(
                    raw_streams[tid], sr, p_cfg["streams"][tid], power
                )
                js_b = write_b122_js_module(profile_dir / f"{tid}.js", comp_b)
                track_syn_L += syn_48k
                track_syn_R += syn_48k
                track_comp += len(comp_b)
                track_js += js_b
                track_raw += len(raw_vwq6)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "wave",
                        "file": f"{tid}.vwp",
                        "sampleRate": meta["sampleRate"],
                        "compressedBytes": len(comp_b),
                        "parameterBytes": len(raw_vwq6),
                        "rawSha256": sha256_hex(raw_vwq6),
                        "waveSnrDb": meta["waveSnrDb"],
                    }
                )

            elif tid == "lead_consonant":
                # 1) True-phase low/mid burst body
                raw_vwq6, comp_b, meta, syn_body = _encode_cosine_substream(
                    raw_streams["lead_consonant"], sr, p_cfg["streams"]["lead_consonant"], power
                )
                js_b = write_b122_js_module(profile_dir / "lead_consonant.js", comp_b)
                track_syn_L += syn_body
                track_syn_R += syn_body
                track_comp += len(comp_b)
                track_js += js_b
                track_raw += len(raw_vwq6)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "wave",
                        "file": "lead_consonant.vwp",
                        "sampleRate": meta["sampleRate"],
                        "compressedBytes": len(comp_b),
                        "parameterBytes": len(raw_vwq6),
                        "rawSha256": sha256_hex(raw_vwq6),
                        "waveSnrDb": meta["waveSnrDb"],
                    }
                )
                # 2) 4-Band 8ms High-Air Sibilance & Breath Stream (`VWA7`)
                raw_vwa7, comp_air, meta_air, syn_air = encode_high_air_stream(
                    raw_streams["lead_consonant"], air_bands, sr=sr, gate_db=air_gate
                )
                assert unpack_vwp6_lzma(comp_air) == raw_vwa7
                js_air = write_b122_js_module(profile_dir / "lead_air.js", comp_air)
                track_syn_L += syn_air
                track_syn_R += syn_air
                track_comp += len(comp_air)
                track_js += js_air
                track_raw += len(raw_vwa7)
                stream_entries.append(
                    {
                        "channel": "mono",
                        "kind": "air",
                        "file": "lead_air.vwp",
                        "sampleRate": sr,
                        "compressedBytes": len(comp_air),
                        "parameterBytes": len(raw_vwa7),
                        "rawSha256": sha256_hex(raw_vwa7),
                    }
                )

            elif tid == "backing_harmony":
                for ch_role, s_key in [("mid", "backing_mid"), ("side", "backing_side")]:
                    raw_vwq6, comp_b, meta, syn_48k = _encode_cosine_substream(
                        raw_streams[s_key], sr, p_cfg["streams"][s_key], power
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

            elif tid == "vocal_space":
                # Pure mathematical 8-line Householder FDN stereo reverb driven by synthesized dry vocal (`01..04`)
                dry_mid_in = (0.5 * (dry_syn_L + dry_syn_R)).astype(np.float32)
                dry_side_in = (0.5 * (dry_syn_L - dry_syn_R)).astype(np.float32)
                raw_vwr7, comp_rev, meta_rev, syn_rev_stereo = encode_fitted_reverb_stream(
                    dry_mid_in,
                    dry_side_in,
                    raw_streams["space_mid"],
                    raw_streams["space_side"],
                    sr=sr,
                )
                assert unpack_vwp6_lzma(comp_rev) == raw_vwr7
                js_rev = write_b122_js_module(profile_dir / "space_fdn.js", comp_rev)
                track_syn_L += syn_rev_stereo[:, 0]
                track_syn_R += syn_rev_stereo[:, 1]
                track_comp += len(comp_rev)
                track_js += js_rev
                track_raw += len(raw_vwr7)
                stream_entries.append(
                    {
                        "channel": "stereo",
                        "kind": "reverb",
                        "file": "space_fdn.vwp",
                        "sampleRate": sr,
                        "compressedBytes": len(comp_rev),
                        "parameterBytes": len(raw_vwr7),
                        "rawSha256": sha256_hex(raw_vwr7),
                    }
                )

            if tid != "vocal_space":
                dry_syn_L += track_syn_L
                dry_syn_R += track_syn_R

            combined_vocal[:, 0] += track_syn_L
            combined_vocal[:, 1] += track_syn_R

            track_syn_eval = np.stack([track_syn_L, track_syn_R], axis=1) if t_meta["stereo"] else track_syn_L
            t_wave_snr = compute_wave_snr_db(ref_tracks[tid], track_syn_eval)
            t_mel_snr = compute_log_mel_snr_db(ref_tracks[tid], track_syn_eval, sr=sr)

            total_comp += track_comp
            total_js += track_js
            total_raw += track_raw

            manifest_tracks.append(
                {
                    "id": tid,
                    "name": t_meta["name"],
                    "subtitle": t_meta["subtitle"],
                    "color": t_meta["color"],
                    "stereo": t_meta["stereo"],
                    "parametric": tid in ("lead_consonant", "vocal_space"),
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
                f"      - {tid:16s}: {track_comp / 1024:6.1f} KB LZMA | Wave SNR {t_wave_snr:5.2f} dB | Log-Mel SNR {t_mel_snr:5.2f} dB",
                flush=True,
            )

        dry_wave_snr = compute_wave_snr_db(
            np.stack([dry_ref_L, dry_ref_R], axis=1),
            np.stack([dry_syn_L, dry_syn_R], axis=1),
        )
        full_wave_snr = compute_wave_snr_db(vocals_ref, combined_vocal)
        full_mel_snr = compute_log_mel_snr_db(vocals_ref, combined_vocal, sr=sr)
        smaller_vs_wav = round(100.0 * (1.0 - total_comp / lossless_wav_bytes), 2)

        manifest = {
            "title": f"{config['title']} · {p_cfg['name']}",
            "artist": config["artist"],
            "arranger": config["arranger"],
            "duration": round(duration, 6),
            "outputSampleRate": sr,
            "dryWaveSnrDb": dry_wave_snr,
            "vocalWaveSnrDb": full_wave_snr,
            "vocalLogMelSnrDb": full_mel_snr,
            "presets": config["presets"],
            "tracks": manifest_tracks,
            "timbreProfile": sep_res["timbreProfile"],
            "sections": config["sections"],
            "lyrics": lyrics,
            "losslessStereoWavBytes": lossless_wav_bytes,
            "format": "VWP7+VWA7+VWR7/HybridTruePhase+Air+FDN+LZMA+Base122",
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
            "dryWaveSnrDb": dry_wave_snr,
            "vocalWaveSnrDb": full_wave_snr,
            "vocalLogMelSnrDb": full_mel_snr,
            "smallerThanWavPercent": smaller_vs_wav,
        }
        print(
            f"      => `{profile_id}` Full 5-Track Vocal: {total_comp:,} B ({total_comp / 1024:.1f} KB = {total_comp / 1e6:.2f} MB, -{smaller_vs_wav}% vs WAV) | Dry Wave SNR: {dry_wave_snr:.2f} dB | Full Mel SNR: {full_mel_snr:.2f} dB",
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
