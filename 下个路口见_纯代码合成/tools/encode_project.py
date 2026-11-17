#!/usr/bin/env python3
"""End-to-end V4 Psychoacoustic Pure-Code Encoder for 《下个，路口，见》."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.audio_io import decode_stereo, resample_audio
from services.drum_residual import build_drum_prototypes, fit_closed_loop_schedule
from services.lrc_parser import parse_lrc
from services.lzma_packer import pack_mdp4_lzma, sha256_hex, write_js_parameter_module
from services.spectral_quantizer import quantize_spectrum
from services.stem_separator import separate_coherent_stems

V3_FAITHFUL_BYTES = 5413298
V3_COMPACT_BYTES = 2185632
V3_SINGLE_HTML_BYTES = 2979817


def _core_band_snr_db(ref: np.ndarray, test: np.ndarray, sr: int, cutoff_hz: float = 4200.0) -> float:
    """Computes tonal/formant core-band (< 4.2 kHz) waveform SNR where human ear is phase-sensitive."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, min(cutoff_hz, sr * 0.45), btype="lowpass", fs=sr, output="sos")
    r_lp = sosfiltfilt(sos, ref, axis=0).astype(np.float64)
    t_lp = sosfiltfilt(sos, test, axis=0).astype(np.float64)
    return float(10.0 * np.log10((np.sum(r_lp**2) + 1e-15) / (np.sum((r_lp - t_lp) ** 2) + 1e-15)))


def main() -> None:
    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/codec_config.json").read_text(encoding="utf-8"))
    source_audio = WORKSPACE_ROOT / config["sourceAudio"]
    source_lrc = WORKSPACE_ROOT / config["sourceLrc"]
    sr = int(config["outputSampleRate"])
    power = float(config["companding"])

    print(f"[1/4] Decoding {source_audio.name} at {sr} Hz...", flush=True)
    original = decode_stereo(source_audio, sr=sr)
    duration = len(original) / sr
    ref_bytes = source_audio.stat().st_size
    print(f"      Duration: {duration:.3f}s ({len(original):,} stereo samples), Reference MP3: {ref_bytes:,} B", flush=True)

    print(f"[2/4] Parsing aligned lyrics from {source_lrc.name}...", flush=True)
    lyrics = parse_lrc(source_lrc, duration)

    print("[3/4] Running in-memory Demucs v4 (`htdemucs`) MPS + phase-coherent 4-stem split...", flush=True)
    separated = separate_coherent_stems(original, sr=sr)
    print(
        f"      Device: {separated['device']} | 4-Stem Exact Sum Max Error: {separated['sumMaxError']:.3e}",
        flush=True,
    )

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sizes_summary = {
        "version": 4,
        "title": config["title"],
        "artist": config["artist"],
        "duration": round(duration, 6),
        "referenceBytes": ref_bytes,
        "v3CompactBytes": V3_COMPACT_BYTES,
        "v3FaithfulBytes": V3_FAITHFUL_BYTES,
        "v3SingleHtmlBytes": V3_SINGLE_HTML_BYTES,
        "stemsSumMaxError": round(separated["sumMaxError"], 9),
        "standardAudioFilesIncluded": False,
        "externalMediaAccess": False,
        "aiModelWeightsIncluded": False,
    }

    profile_labels = {}
    for profile_id, p_cfg in config["profiles"].items():
        print(f"[4/4] Encoding V4 profile `{profile_id}` ({p_cfg['name']})...", flush=True)
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

        for t_meta in config["tracks"]:
            tid = t_meta["id"]
            base_spec = {**t_meta, **p_cfg["tracks"][tid]}
            stem_wave = separated["stems"][tid]
            if stem_wave.ndim == 1:
                components = [("mid", stem_wave)]
            else:
                components = [
                    ("mid", 0.5 * (stem_wave[:, 0] + stem_wave[:, 1])),
                    ("side", 0.5 * (stem_wave[:, 0] - stem_wave[:, 1])),
                ]

            streams_out = []
            loop_bank_entry = None

            for channel, comp_wave in components:
                ch_spec = dict(base_spec)
                if channel == "side" and "sideOverrides" in ch_spec:
                    ch_spec.update(ch_spec["sideOverrides"])
                ch_spec.pop("sideOverrides", None)

                target_sr = int(ch_spec["sampleRate"])
                down = resample_audio(comp_wave, sr, target_sr)

                # Closed-Loop Drum Prototype Dictionary + Residual Coding
                n_protos = int(ch_spec.pop("drumLoopPrototypes", 0)) if channel == "mid" else 0
                pred_down = np.zeros_like(down)

                if n_protos > 0:
                    protos, bar_period, bar_samples = build_drum_prototypes(down, target_sr, n_protos=n_protos)
                    proto_concat = protos.reshape(-1)
                    proto_spec = {
                        "sampleRate": target_sr,
                        "baseQmax": 63,
                        "gateDb": -74,
                        "retainEnergy": 0.998,
                        "maskLowDb": 32.0,
                        "maskHighDb": 20.0,
                        "foldStartLong": 24,
                        "foldStartShort": 12,
                    }
                    proto_raw, proto_meta, proto_decoded_flat = quantize_spectrum(proto_concat, proto_spec, power)
                    proto_lzma = pack_mdp4_lzma(proto_raw)
                    proto_js_bytes = write_js_parameter_module(profile_dir / "drums-proto.js", proto_lzma)
                    decoded_protos = proto_decoded_flat.reshape(n_protos, bar_samples)

                    schedule, pred_down = fit_closed_loop_schedule(
                        down, decoded_protos, bar_period, bar_samples, fade_samples=64
                    )
                    energy_saved_pct = (
                        1.0 - float(np.sum((down - pred_down) ** 2)) / (float(np.sum(down**2)) + 1e-12)
                    ) * 100.0
                    loop_bank_entry = {
                        "file": "drums-proto.lzpar",
                        "sampleRate": target_sr,
                        "barSamples": bar_samples,
                        "fadeSamples": 64,
                        "totalSamples": int(len(down)),
                        "compressedBytes": len(proto_lzma),
                        "parameterBytes": len(proto_raw),
                        "rawSha256": sha256_hex(proto_raw),
                        "bars": schedule,
                    }
                    total_lzma_bytes += len(proto_lzma)
                    total_js_bytes += proto_js_bytes
                    total_raw_bytes += len(proto_raw)
                    print(
                        f"      - drums-proto ({n_protos} bars @ {240*target_sr/bar_period:.2f} BPM): {len(proto_lzma):,} B LZMA | loop captures {energy_saved_pct:.1f}% drum energy",
                        flush=True,
                    )

                residual_in = (down - pred_down).astype(np.float32)
                raw_mdq4, meta, restored_res = quantize_spectrum(residual_in, ch_spec, power)
                lzma_bytes = pack_mdp4_lzma(raw_mdq4)
                lzpar_name = f"{tid}-{channel}.lzpar"
                js_bytes = write_js_parameter_module(profile_dir / f"{tid}-{channel}.js", lzma_bytes)

                restored_full = pred_down + restored_res
                core_snr = _core_band_snr_db(down, restored_full, target_sr)

                up = resample_audio(restored_full, target_sr, sr)[: len(combined)]
                if len(up) < len(combined):
                    up = np.pad(up, (0, len(combined) - len(up)))
                if channel == "mid":
                    combined += up[:, None]
                else:
                    combined[:, 0] += up
                    combined[:, 1] -= up

                stream_entry = {
                    "file": lzpar_name,
                    **meta,
                    "coreSnrDb": round(core_snr, 2),
                    "compressedBytes": len(lzma_bytes),
                    "jsModuleBytes": js_bytes,
                    "channel": channel,
                    "rawSha256": sha256_hex(raw_mdq4),
                    "storage": "LZMA delta-scale sparse MDP4",
                }
                streams_out.append(stream_entry)
                total_lzma_bytes += len(lzma_bytes)
                total_js_bytes += js_bytes
                total_raw_bytes += len(raw_mdq4)
                print(
                    f"      - {tid}-{channel}: {len(lzma_bytes):,} B LZMA | Core(<4.2k) SNR {core_snr:.2f} dB | active {meta['meanActiveCoefficients']}/2048 | short {meta['shortFramePercent']}%",
                    flush=True,
                )

            track_manifest = {
                "id": tid,
                "name": t_meta["name"],
                "color": t_meta["color"],
                "streams": streams_out,
            }
            if loop_bank_entry:
                track_manifest["loopBank"] = loop_bank_entry
            manifest_tracks.append(track_manifest)

        core_mix_snr = round(_core_band_snr_db(original, combined, sr), 2)
        full_mix_snr = round(
            float(
                10.0
                * np.log10(
                    (np.sum(original.astype(np.float64) ** 2) + 1e-15)
                    / (np.sum((original - combined).astype(np.float64) ** 2) + 1e-15)
                )
            ),
            2,
        )

        v3_ref = V3_COMPACT_BYTES if profile_id == "compact" else V3_FAITHFUL_BYTES
        saved_vs_v3 = round(100.0 * (1.0 - total_lzma_bytes / v3_ref), 1)

        manifest = {
            "title": f"{config['title']} · {p_cfg['name']}",
            "artist": config["artist"],
            "arranger": config["arranger"],
            "duration": round(duration, 6),
            "outputSampleRate": sr,
            "mixSnrDb": core_mix_snr,
            "fullWaveformSnrDb": full_mix_snr,
            "tracks": manifest_tracks,
            "sections": config["sections"],
            "lyrics": lyrics,
            "referenceBytes": ref_bytes,
            "format": "MDP4/LZMA+Base122",
            "version": 4,
            "profile": profile_id,
            "profileName": p_cfg["name"],
            "notice": "V4 心理声学纯代码参数：长短窗切换 + 24-Bark 掩蔽 + 高频参数立体声 + 鼓闭环残差。",
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
            "coreBandSnrDb": core_mix_snr,
            "fullWaveformSnrDb": full_mix_snr,
            "savedVsV3Percent": saved_vs_v3,
            "smallerThanOriginalPercent": round(100.0 * (1.0 - total_lzma_bytes / ref_bytes), 2),
        }
        print(
            f"      => `{profile_id}` V4 Total LZMA: {total_lzma_bytes:,} B ({total_lzma_bytes / 1e6:.2f} MB, -{saved_vs_v3}% vs V3!) | Core(<4.2k) Mix SNR: {core_mix_snr:.2f} dB",
            flush=True,
        )

    profiles_index = {
        "default": "compact",
        "available": list(config["profiles"].keys()),
        "labels": profile_labels,
    }
    (data_dir / "profiles.json").write_text(
        json.dumps(profiles_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (data_dir / "sizes.json").write_text(
        json.dumps(sizes_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
