#!/usr/bin/env python3
"""End-to-end V5 AI-Disentangled Pure Mathematical Wave-Code Encoder for 《下个，路口，见》."""
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

from services.audio_io import decode_stereo
from services.bass_modeler import encode_bass_wave
from services.drum_modeler import encode_drums_wave
from services.harmony_modeler import encode_harmony_wave
from services.lrc_parser import parse_lrc
from services.stem_separator import separate_coherent_stems
from services.vocal_disentangler import encode_vocal_wave
from services.wave_packer import compress_lzma, sha256_hex, write_b122_js_module

V4_COMPACT_BYTES = 1131172
V4_FAITHFUL_BYTES = 2586755
V4_SINGLE_HTML_BYTES = 1330565


def compute_log_mel_snr_db(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> float:
    """Computes perceptual Log-Mel spectrogram SNR (dB) across 48 critical Mel bands."""
    if ref.ndim == 2:
        r_m = 0.5 * (ref[:, 0] + ref[:, 1])
        s_m = 0.5 * (syn[:, 0] + syn[:, 1])
    else:
        r_m, s_m = ref, syn
    L = min(len(r_m), len(s_m))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=48, fmin=55.0, fmax=16000.0)
    Mr = fb @ np.abs(librosa.stft(r_m[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s_m[:L], n_fft=2048, hop_length=512))
    floor = float(np.max(Mr) * 1e-3 + 1e-8)
    lr = np.log10(np.maximum(Mr, floor))
    ls = np.log10(np.maximum(Ms, floor))
    return round(float(10.0 * np.log10((np.sum(lr ** 2) + 1e-12) / (np.sum((lr - ls) ** 2) + 1e-12))), 2)


def main() -> None:
    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/wave_config.json").read_text(encoding="utf-8"))
    source_audio = WORKSPACE_ROOT / config["sourceAudio"]
    source_lrc = WORKSPACE_ROOT / config["sourceLrc"]
    sr = int(config["outputSampleRate"])

    print(f"[1/4] Decoding {source_audio.name} at {sr} Hz...", flush=True)
    original = decode_stereo(source_audio, sr=sr)
    duration = len(original) / sr
    ref_bytes = source_audio.stat().st_size
    print(f"      Duration: {duration:.3f}s ({len(original):,} stereo samples), Reference MP3: {ref_bytes:,} B", flush=True)

    print(f"[2/4] Parsing aligned lyrics from {source_lrc.name}...", flush=True)
    lyrics = parse_lrc(source_lrc, duration)

    print("[3/4] Running in-memory Demucs v4 (`htdemucs`) MPS + phase-coherent 4-stem split...", flush=True)
    separated = separate_coherent_stems(original, sr=sr)
    print(f"      Device: {separated['device']} | 4-Stem Sum Max Error: {separated['sumMaxError']:.3e}", flush=True)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sizes_summary = {
        "version": 5,
        "architecture": config["architecture"],
        "title": config["title"],
        "artist": config["artist"],
        "duration": round(duration, 6),
        "referenceBytes": ref_bytes,
        "v4CompactBytes": V4_COMPACT_BYTES,
        "v4FaithfulBytes": V4_FAITHFUL_BYTES,
        "v4SingleHtmlBytes": V4_SINGLE_HTML_BYTES,
        "standardAudioFilesIncluded": False,
        "mdctSpectralFramesIncluded": False,
        "aiModelWeightsIncluded": False,
        "externalMediaAccess": False,
    }

    profile_labels = {}
    stems = separated["stems"]

    for profile_id, p_cfg in config["profiles"].items():
        print(f"[4/4] AI Disentangling & Wave-Modeling `{profile_id}` ({p_cfg['name']})...", flush=True)
        profile_labels[profile_id] = p_cfg["name"]
        profile_dir = data_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        for stale in profile_dir.glob("*.js"):
            stale.unlink()

        combined = np.zeros_like(original, dtype=np.float32)
        manifest_tracks = []
        total_lzma_bytes = 0
        total_js_bytes = 0
        total_raw_bytes = 0

        encoders = {
            "vocal": encode_vocal_wave,
            "drums": encode_drums_wave,
            "bass": encode_bass_wave,
            "harmony": encode_harmony_wave,
        }

        for t_meta in config["tracks"]:
            tid = t_meta["id"]
            raw_wvp, math_model, synth_48k = encoders[tid](stems[tid], sr, p_cfg[tid])
            comp_bytes = compress_lzma(raw_wvp)
            wvp_name = f"{tid}-wave.wvp"
            js_bytes = write_b122_js_module(profile_dir / f"{tid}-wave.js", comp_bytes)

            if synth_48k.ndim == 1:
                combined[:, 0] += synth_48k
                combined[:, 1] += synth_48k
            else:
                combined += synth_48k

            log_mel_snr = compute_log_mel_snr_db(stems[tid], synth_48k, sr=sr)
            total_lzma_bytes += len(comp_bytes)
            total_js_bytes += js_bytes
            total_raw_bytes += len(raw_wvp)

            manifest_tracks.append(
                {
                    "id": tid,
                    "name": t_meta["name"],
                    "color": t_meta["color"],
                    "formula": t_meta["formula"],
                    "file": wvp_name,
                    "compressedBytes": len(comp_bytes),
                    "jsModuleBytes": js_bytes,
                    "parameterBytes": len(raw_wvp),
                    "rawSha256": sha256_hex(raw_wvp),
                    "logMelSnrDb": log_mel_snr,
                    "model": math_model,
                }
            )
            print(
                f"      - {tid}-wave: {len(comp_bytes):,} B LZMA ({len(comp_bytes)/1024:.1f} KB) | Log-Mel SNR {log_mel_snr:.2f} dB",
                flush=True,
            )

        mix_log_mel_snr = compute_log_mel_snr_db(original, combined, sr=sr)
        v4_ref = V4_COMPACT_BYTES if profile_id == "compact" else V4_FAITHFUL_BYTES
        saved_vs_v4 = round(100.0 * (1.0 - total_lzma_bytes / v4_ref), 1)
        saved_vs_mp3 = round(100.0 * (1.0 - total_lzma_bytes / ref_bytes), 2)

        manifest = {
            "title": f"{config['title']} · {p_cfg['name']}",
            "artist": config["artist"],
            "arranger": config["arranger"],
            "duration": round(duration, 6),
            "outputSampleRate": sr,
            "mixLogMelSnrDb": mix_log_mel_snr,
            "tracks": manifest_tracks,
            "sections": config["sections"],
            "lyrics": lyrics,
            "referenceBytes": ref_bytes,
            "format": "WVP5/Disentangled-Wave+LZMA+Base122",
            "version": 5,
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
            "parameterBytes": total_lzma_bytes,
            "jsModuleBytes": total_js_bytes,
            "unpackedParameterBytes": total_raw_bytes,
            "mixLogMelSnrDb": mix_log_mel_snr,
            "savedVsV4Percent": saved_vs_v4,
            "smallerThanOriginalPercent": saved_vs_mp3,
        }
        print(
            f"      => `{profile_id}` V5 Total Wave-Code: {total_lzma_bytes:,} B ({total_lzma_bytes / 1024:.1f} KB, -{saved_vs_v4}% vs V4, -{saved_vs_mp3}% vs MP3!) | Mix Log-Mel SNR: {mix_log_mel_snr:.2f} dB",
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
