#!/usr/bin/env python3
"""End-to-End V8.3 Purified Event-Syllable + Global Quadrature Phasor Genesis Compiler."""
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

from parsers.lrc_tokenizer import load_phoneme_lexicon, parse_lrc_with_tokens
from services.audio_io import decode_stereo
from services.event_aligner import segment_song_into_events
from services.event_codec import (
    compress_lzma1_alone,
    decompress_lzma1_alone,
    pack_event_stream,
    pack_speaker_core,
    pack_syllable_atlas,
    sha256_hex,
    write_b122_js_module,
)
from services.harmonic_aperiodic_analyzer import analyze_vocal_signal
from services.sms_synthesizer import synthesize_from_packets
from services.space_fdn_model import fit_and_pack_space_fdn
from services.syllable_atlas_builder import build_speaker_and_syllable_atlas, encode_events_for_profile
from services.vocal_extractor import extract_clean_vocal_anatomy


def compute_log_mel_snr_db(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> float:
    r_m = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s_m = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = min(len(r_m), len(s_m))
    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=64, fmin=65.0, fmax=17500.0)
    Mr = fb @ np.abs(librosa.stft(r_m[:L], n_fft=2048, hop_length=512))
    Ms = fb @ np.abs(librosa.stft(s_m[:L], n_fft=2048, hop_length=512))
    floor = float(np.max(Mr) * 1.5e-3 + 1e-8)
    lr = np.log10(np.maximum(Mr, floor))
    ls = np.log10(np.maximum(Ms, floor))
    return round(float(10.0 * np.log10((np.sum(lr ** 2) + 1e-12) / (np.sum((lr - ls) ** 2) + 1e-12))), 2)


def compute_envelope_corr(ref: np.ndarray, syn: np.ndarray, hop: int = 512) -> float:
    r_m = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s_m = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = (min(len(r_m), len(s_m)) // hop) * hop
    er = np.sqrt(np.mean(r_m[:L].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    es = np.sqrt(np.mean(s_m[:L].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    c = np.corrcoef(er, es)[0, 1]
    return round(float(c), 4)


def compute_purity_telemetry(layers: dict, events_enc: list, q_step: float) -> dict:
    h_l = layers["harmonic_core"][:, 0]
    a_l = layers["consonant_air"][:, 0]
    whole_hnr_db = round(10.0 * np.log10((float(np.sum(h_l ** 2)) + 1e-12) / (float(np.sum(a_l ** 2)) + 1e-12)), 2)

    hop = 256
    nb = len(h_l) // hop
    hb = np.sqrt(np.mean(h_l[: nb * hop].reshape(nb, hop) ** 2, axis=1))
    ab = np.sqrt(np.mean(a_l[: nb * hop].reshape(nb, hop) ** 2, axis=1))
    vm = hb > (np.max(hb) * 0.08)
    voiced_hnr_db = round(
        10.0 * np.log10((float(np.sum(hb[vm] ** 2)) + 1e-12) / (float(np.sum(ab[vm] ** 2)) + 1e-12)), 2
    )

    F_fdn = np.abs(np.fft.rfft(layers["space_fdn"][:, 0]))
    freqs = np.linspace(0.0, 24000.0, len(F_fdn))
    fdn_hf_pct = round(float(100.0 * np.sum(F_fdn[freqs > 12000.0] ** 2) / (np.sum(F_fdn ** 2) + 1e-12)), 4)

    k0_norms = [
        float(np.linalg.norm(e.expr_mcep_q[0] * q_step))
        for e in events_enc
        if len(e.expr_mcep_q) > 0
    ]
    knot0_db = round(float(np.mean(k0_norms)) if k0_norms else 0.0, 2)
    return {
        "hnrDb": voiced_hnr_db,
        "wholeSongHnrDb": whole_hnr_db,
        "v8HnrDb": 1.90,
        "v8LatestVoicedHnrDb": 28.38,
        "fdnAliasingAbove12kPct": fdn_hf_pct,
        "v8FdnAliasingAbove12kPct": 1.26,
        "knot0ResidualDb": knot0_db,
        "v8Knot0ResidualDb": 60.94,
    }


def main() -> None:
    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/vocal_genesis_config.json").read_text(encoding="utf-8"))
    lexicon = load_phoneme_lexicon(PROJECT_ROOT / "schemas/phoneme_lexicon.json")

    mp3_path = WORKSPACE_ROOT / "mp3_output/李宇春 - 下个, 路口, 见.mp3"
    lrc_path = WORKSPACE_ROOT / "mp3_output/李宇春 - 下个, 路口, 见.lrc"

    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])

    print("[1/6] Decoding master & running Cross-Stem Accompaniment De-Bleed Vocal Purifier...", flush=True)
    master = decode_stereo(mp3_path, sr=sr)
    n_samples = len(master)
    duration_sec = n_samples / float(sr)
    anatomy = extract_clean_vocal_anatomy(master, sr=sr, hop=hop)

    print("[2/6] Running -92dB Blackman-Harris harmonic, liftered formant & zero-hiss consonant analysis...", flush=True)
    features = analyze_vocal_signal(anatomy["lead_direct"], config)

    print("[3/6] Lag-compensated LRC alignment & 100% phantom-hook purge...", flush=True)
    all_lines, sung_lines = parse_lrc_with_tokens(lrc_path, lexicon, duration_sec)
    raw_events, rejected_ghosts = segment_song_into_events(sung_lines, features, config)

    print("[4/6] Learning Clean Speaker Core, Zero-Spike Syllable Atlas & PCA Expression Basis...", flush=True)
    speaker, prototypes, raw_events, ref_peak, atlas_stats = build_speaker_and_syllable_atlas(
        raw_events, features, anatomy, config
    )
    atlas_stats["purgedGhostBleedIslands"] = rejected_ghosts

    raw_speaker = pack_speaker_core(speaker, sr, n_samples, hop, ref_peak)
    comp_speaker = compress_lzma1_alone(raw_speaker)
    assert decompress_lzma1_alone(comp_speaker) == raw_speaker

    raw_atlas = pack_syllable_atlas(prototypes, atlas_q_step=0.25)
    comp_atlas = compress_lzma1_alone(raw_atlas)
    assert decompress_lzma1_alone(comp_atlas) == raw_atlas

    raw_fdn, fdn_meta = fit_and_pack_space_fdn(anatomy["lead_direct"], anatomy["reverb_residual"], sr=sr)
    comp_fdn = compress_lzma1_alone(raw_fdn)
    assert decompress_lzma1_alone(comp_fdn) == raw_fdn

    shared_dir = PROJECT_ROOT / "data/shared"
    write_b122_js_module(shared_dir / "speaker_core.js", comp_speaker)
    write_b122_js_module(shared_dir / "syllable_atlas.js", comp_atlas)
    write_b122_js_module(shared_dir / "space_fdn.js", comp_fdn)

    shared_bytes = len(comp_speaker) + len(comp_atlas) + len(comp_fdn)
    print(
        f"    Shared Core: speaker={len(comp_speaker)} B + syllable_atlas({len(prototypes)} protos)={len(comp_atlas)} B + space_fdn={len(comp_fdn)} B -> {shared_bytes/1024:.2f} KB",
        flush=True,
    )
    print(f"    Atlas Stats: {json.dumps(atlas_stats, ensure_ascii=False)}", flush=True)

    wav_equiv_bytes = n_samples * 2 * 2
    orig_mp3_bytes = mp3_path.stat().st_size

    v8_ultra_bytes = 44281
    v8_expr_bytes = 112045

    profiles_summary: dict[str, dict] = {}
    for prof_id, prof_spec in config["profiles"].items():
        print(f"[5/6] Encoding & synthesizing purified profile `{prof_id}` ({prof_spec['label']})...", flush=True)
        t_p = time.time()
        events_enc = encode_events_for_profile(
            raw_events, speaker, prototypes, features, anatomy, prof_spec, ref_peak
        )
        raw_ev = pack_event_stream(events_enc, prof_spec)
        comp_ev = compress_lzma1_alone(raw_ev)
        assert decompress_lzma1_alone(comp_ev) == raw_ev

        prof_dir = PROJECT_ROOT / f"data/{prof_id}"
        write_b122_js_module(prof_dir / "event_stream.js", comp_ev)

        syn_res = synthesize_from_packets(raw_speaker, raw_atlas, raw_ev, raw_fdn, config)
        mix_syn = syn_res["mix_stereo"]
        mel_snr_full = compute_log_mel_snr_db(anatomy["vocals_stereo"], mix_syn, sr=sr)
        mel_snr_dry = compute_log_mel_snr_db(
            anatomy["lead_direct"],
            0.5 * (syn_res["layers"]["harmonic_core"][:, 0] + syn_res["layers"]["harmonic_core"][:, 1])
            + 0.5 * (syn_res["layers"]["consonant_air"][:, 0] + syn_res["layers"]["consonant_air"][:, 1]),
            sr=sr,
        )
        env_corr = compute_envelope_corr(anatomy["vocals_stereo"], mix_syn)
        purity = compute_purity_telemetry(syn_res["layers"], events_enc, float(prof_spec["quantStepDb"]))

        tot_comp = shared_bytes + len(comp_ev)
        tot_raw = len(raw_speaker) + len(raw_atlas) + len(raw_fdn) + len(raw_ev)
        kbps = round((tot_comp * 8.0) / (duration_sec * 1000.0), 2)

        manifest = {
            "profile": prof_id,
            "label": prof_spec["label"],
            "tagline": prof_spec["tagline"],
            "sampleRate": sr,
            "numSamples": n_samples,
            "durationSec": round(duration_sec, 3),
            "totalCompressedBytes": tot_comp,
            "totalRawBytes": tot_raw,
            "equivalentKbps": kbps,
            "logMelSnrFullDb": mel_snr_full,
            "logMelSnrDryDb": mel_snr_dry,
            "envelopeCorrelation": env_corr,
            "purity": purity,
            "bytesPerEventAvg": round(len(comp_ev) / max(1, len(events_enc)), 1),
            "packets": {
                "speakerCore": {
                    "path": "../shared/speaker_core.js",
                    "compressedBytes": len(comp_speaker),
                    "rawBytes": len(raw_speaker),
                    "rawSha256": sha256_hex(raw_speaker),
                },
                "syllableAtlas": {
                    "path": "../shared/syllable_atlas.js",
                    "compressedBytes": len(comp_atlas),
                    "rawBytes": len(raw_atlas),
                    "rawSha256": sha256_hex(raw_atlas),
                },
                "spaceFdn": {
                    "path": "../shared/space_fdn.js",
                    "compressedBytes": len(comp_fdn),
                    "rawBytes": len(raw_fdn),
                    "rawSha256": sha256_hex(raw_fdn),
                },
                "eventStream": {
                    "path": "./event_stream.js",
                    "compressedBytes": len(comp_ev),
                    "rawBytes": len(raw_ev),
                    "rawSha256": sha256_hex(raw_ev),
                },
            },
            "layers": config["layers"],
        }
        (prof_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        profiles_summary[prof_id] = manifest
        print(
            f"    -> `{prof_id}`: total={tot_comp:,} B ({tot_comp/1024:.2f} KB | {kbps} kbps | {manifest['bytesPerEventAvg']} B/ev), "
            f"Dry Mel-SNR={mel_snr_dry} dB, Full={mel_snr_full} dB, Voiced-HNR={purity['hnrDb']} dB (Whole={purity['wholeSongHnrDb']} dB), "
            f"Knot0={purity['knot0ResidualDb']} dB, FDN>12k={purity['fdnAliasingAbove12kPct']}% ({time.time() - t_p:.1f}s)",
            flush=True,
        )

    atlas_catalog = [
        {
            "id": p.syllable_id,
            "pinyin": p.pinyin,
            "initialClass": p.initial_class,
            "count": p.count,
        }
        for p in prototypes
    ]
    (PROJECT_ROOT / "data/atlas_catalog.json").write_text(
        json.dumps({"stats": atlas_stats, "prototypes": atlas_catalog}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "data/lyrics.json").write_text(
        json.dumps({"durationSec": round(duration_sec, 3), "lines": all_lines}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "data/profiles.json").write_text(
        json.dumps({"default": "expressive", "available": list(config["profiles"].keys())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sizes_report = {
        "songTitle": "李宇春 - 下个, 路口, 见 (纯人声·纯净去残事件音节版 V8.3)",
        "durationSec": round(duration_sec, 3),
        "referenceWav48k16bStereoBytes": wav_equiv_bytes,
        "referenceMp3Bytes": orig_mp3_bytes,
        "v8InitialUltraBytes": 47375,
        "v8InitialExpressiveBytes": 121429,
        "v8UltraBytes": v8_ultra_bytes,
        "v8ExpressiveBytes": v8_expr_bytes,
        "atlasStats": atlas_stats,
        "sharedCoreBytes": {
            "speakerCore": len(comp_speaker),
            "syllableAtlas": len(comp_atlas),
            "spaceFdn": len(comp_fdn),
            "totalShared": shared_bytes,
        },
        "profiles": {
            pid: {
                "totalCompressedBytes": m["totalCompressedBytes"],
                "totalKB": round(m["totalCompressedBytes"] / 1024.0, 2),
                "eventStreamBytes": m["packets"]["eventStream"]["compressedBytes"],
                "equivalentKbps": m["equivalentKbps"],
                "reductionVsWavPct": round(100.0 * (1.0 - m["totalCompressedBytes"] / wav_equiv_bytes), 3),
                "reductionVsV8Pct": round(
                    100.0 * (1.0 - m["totalCompressedBytes"] / (v8_ultra_bytes if pid == "ultra" else v8_expr_bytes)), 2
                ),
                "reductionVsV8InitialPct": round(
                    100.0 * (1.0 - m["totalCompressedBytes"] / (47375.0 if pid == "ultra" else 121429.0)), 2
                ),
                "logMelSnrDryDb": m["logMelSnrDryDb"],
                "logMelSnrFullDb": m["logMelSnrFullDb"],
                "envelopeCorrelation": m["envelopeCorrelation"],
                "purity": m["purity"],
                "bytesPerEventAvg": m["bytesPerEventAvg"],
            }
            for pid, m in profiles_summary.items()
        },
    }
    (PROJECT_ROOT / "data/sizes.json").write_text(
        json.dumps(sizes_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[6/6] Completed V8.3 compilation in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
