#!/usr/bin/env python3
"""Compile V8 Dictionary Accompaniment (`data/accomp_v8_dict/*.js`) using VDA8 + LZMA1 + Base122."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
V7_ROOT = WORKSPACE_ROOT / "下个路口见_全曲多轨合成"
sys.path.insert(0, str(V7_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from services.accomp_dict_codec import (
    build_beat_grids,
    encode_dict_track_mono,
    encode_dict_track_stereo_ms,
)
from services.audio_io import decode_stereo
from services.event_codec import (
    compress_lzma1_alone,
    decompress_lzma1_alone,
    sha256_hex,
    write_b122_js_module,
)


def compute_log_mel_snr_db(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> float:
    r_m = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s_m = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = min(len(r_m), len(s_m))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=45.0, fmax=19000.0)
    Mr = fb @ np.abs(librosa.stft(r_m[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s_m[:L], n_fft=2048, hop_length=512))
    floor = float(np.max(Mr) * 1e-3 + 1e-8)
    lr = np.log10(np.maximum(Mr, floor))
    ls = np.log10(np.maximum(Ms, floor))
    return round(float(10.0 * np.log10((np.sum(lr ** 2) + 1e-12) / (np.sum((lr - ls) ** 2) + 1e-12))), 2)


def main() -> None:
    t0 = time.time()
    v7_cfg = json.loads((V7_ROOT / "schemas/full_song_multitrack_config.json").read_text(encoding="utf-8"))
    p_cfg = v7_cfg["profiles"]["compact"]
    power = float(p_cfg["power"])
    sr = 48000

    cache_path = Path("/tmp/accomp_stems_48k.npz")
    if cache_path.exists():
        print(f"[1/4] Loading cached 48kHz stems from {cache_path}...")
        stems = np.load(cache_path)
        bass = stems["bass"].astype(np.float32)
        drums = stems["drums"].astype(np.float32)
        accomp_mid = stems["accomp_mid"].astype(np.float32)
        accomp_side = stems["accomp_side"].astype(np.float32)
    else:
        print("[1/4] Extracting 48kHz accompaniment stems via htdemucs on MPS...")
        master = decode_stereo(WORKSPACE_ROOT / "mp3_output/李宇春 - 下个, 路口, 见.mp3", sr=sr)
        res = extract_accompaniment_3tracks(master, sr=sr)
        bass = res["tracks"]["bass"].astype(np.float32)
        drums = res["tracks"]["drums"].astype(np.float32)
        accomp_mid = res["streams"]["accomp_mid"].astype(np.float32)
        accomp_side = res["streams"]["accomp_side"].astype(np.float32)

    n_samples = len(bass)
    ref_acc = np.stack([accomp_mid + accomp_side, accomp_mid - accomp_side], axis=1)
    ref_inst = np.stack([bass + drums, bass + drums], axis=1) + ref_acc

    print("[2/4] Building Beat/Half-Beat/Bar grids...")
    grids = build_beat_grids(drums, n_samples, sr=sr)
    med_b = int(grids["med_beat"])

    print("[3/4] Encoding Bass (2-beat K=24), Drums (0.5-beat K=36), Guitar/Synth (4-beat K=32)...")
    bass_len = int(np.max(np.diff(grids["two_beats"]))) + 512
    raw_bass, syn_bass = encode_dict_track_mono(
        bass,
        grids["two_beats"],
        bass_len,
        k_protos=24,
        xfade=512,
        body_spec=p_cfg["streams"]["bass"],
        power=power,
        mel_fmin=35.0,
        mel_fmax=2500.0,
        mel_bins=24,
        mel_hop=512,
        sr=sr,
    )
    drums_len = int(np.max(np.diff(grids["half_beats"]))) + 384
    raw_drums, syn_drums = encode_dict_track_mono(
        drums,
        grids["half_beats"],
        drums_len,
        k_protos=36,
        xfade=384,
        body_spec=p_cfg["streams"]["drums_body"],
        power=power,
        air_bands_hz=p_cfg["drumAirBandsHz"],
        air_gate_db=float(p_cfg["drumAirGateDb"]),
        mel_fmin=50.0,
        mel_fmax=16000.0,
        mel_bins=32,
        mel_hop=512,
        sr=sr,
    )
    side_spec_compact = {
        "sampleRate": 10000,
        "baseQmax": 18,
        "retainEnergy": 0.980,
        "gateDb": -54.0,
        "maskLowDb": 18.0,
        "maskHighDb": 12.0,
    }
    gtr_len = int(np.max(np.diff(grids["four_beats"]))) + 768
    raw_gtr, syn_gtr = encode_dict_track_stereo_ms(
        accomp_mid,
        accomp_side,
        grids["four_beats"],
        gtr_len,
        k_protos=32,
        xfade=768,
        mid_spec=p_cfg["streams"]["accomp_mid"],
        side_spec=side_spec_compact,
        power=power,
        air_bands_hz=p_cfg["accompAirBandsHz"],
        air_gate_db=float(p_cfg["accompAirGateDb"]),
        sr=sr,
    )

    out_dir = PROJECT_ROOT / "data/accomp_v8_dict"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_meta = {}
    total_comp = 0
    total_raw = 0
    for name, raw_pkt, ref_t, syn_t in [
        ("bass", raw_bass, bass, syn_bass),
        ("drums", raw_drums, drums, syn_drums),
        ("guitar_synth", raw_gtr, ref_acc, syn_gtr),
    ]:
        comp_pkt = compress_lzma1_alone(raw_pkt)
        assert decompress_lzma1_alone(comp_pkt) == raw_pkt
        js_path = out_dir / f"{name}_dict.js"
        write_b122_js_module(js_path, comp_pkt)
        snr = compute_log_mel_snr_db(ref_t, syn_t, sr=sr)
        total_comp += len(comp_pkt)
        total_raw += len(raw_pkt)
        tracks_meta[name] = {
            "path": f"../accomp_v8_dict/{name}_dict.js",
            "compressedBytes": len(comp_pkt),
            "rawBytes": len(raw_pkt),
            "rawSha256": sha256_hex(raw_pkt),
            "logMelSnrDb": snr,
        }
        print(f"  - {name:12s}: {len(comp_pkt)/1024:6.2f} KB ({len(comp_pkt):6d} B) | logMelSNR={snr:.2f} dB")

    syn_inst = np.stack([syn_bass + syn_drums, syn_bass + syn_drums], axis=1) + syn_gtr
    inst_snr = compute_log_mel_snr_db(ref_inst, syn_inst, sr=sr)

    manifest = {
        "version": "v8_dict",
        "sampleRate": sr,
        "numSamples": n_samples,
        "totalCompressedBytes": total_comp,
        "totalRawBytes": total_raw,
        "instrumentalLogMelSnrDb": inst_snr,
        "tracks": tracks_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[4/4] Done in {time.time() - t0:.2f}s | Total Dict Accomp: {total_comp/1024:.2f} KB ({total_comp} B) | Inst Mix logMelSNR={inst_snr:.2f} dB")


if __name__ == "__main__":
    main()
