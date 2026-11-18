#!/usr/bin/env python3
"""End-to-End V8.5 8-Track Coarticulated Vocal Anatomy & Syllable-Atlas Compiler."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import librosa
import numpy as np
import scipy.signal as signal

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
    unpack_speaker_core,
    unpack_syllable_atlas,
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


def compute_mel80_and_hiband_db(ref: np.ndarray, syn: np.ndarray, sr: int = 48000) -> tuple[float, float]:
    r_m = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s_m = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = min(len(r_m), len(s_m))
    Sr = np.abs(librosa.stft(r_m[:L], n_fft=2048, hop_length=256))
    Sy = np.abs(librosa.stft(s_m[:L], n_fft=2048, hop_length=256))
    frms = np.sqrt(np.mean(Sr ** 2, axis=0))
    act = frms > np.max(frms) * 0.015
    eps = float(np.max(Sr) * 1e-4 + 1e-9)
    fb80 = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=80, fmin=80.0, fmax=15000.0)
    Mr = 20.0 * np.log10(np.maximum(fb80 @ Sr[:, act], eps))
    My = 20.0 * np.log10(np.maximum(fb80 @ Sy[:, act], eps))
    mel80_err = round(float(np.sqrt(np.mean((Mr - My) ** 2))), 2)

    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    hi = (freqs >= 2200.0) & (freqs <= 10000.0)
    pr = float(np.sum(Sr[hi][:, act] ** 2))
    py = float(np.sum(Sy[hi][:, act] ** 2))
    hi_bias = round(10.0 * math.log10((py + 1e-12) / (pr + 1e-12)), 2)
    return mel80_err, hi_bias


def compute_narrowband_harmonic_metrics(
    ref_mono: np.ndarray,
    syn_mono: np.ndarray,
    f0: np.ndarray,
    voiced: np.ndarray,
    sr: int = 48000,
    hop: int = 256,
) -> tuple[float, float]:
    """Measures exact 4096-pt Blackman-Harris H1..H10 per-harmonic RMSE and bias in dB across all voiced frames."""
    win_l = 3072
    n_fft = 4096
    bin_hz = sr / float(n_fft)
    w_bh = signal.windows.blackmanharris(win_l, sym=False).astype(np.float32)
    norm = 2.0 / float(np.sum(w_bh))

    L = min(len(ref_mono), len(syn_mono))
    v_frames = np.flatnonzero(voiced & (f0 > 80.0))
    v_frames = v_frames[(v_frames * hop + win_l // 2) < L]
    v_frames = v_frames[::4]  # stride-4 sample across entire song for fast exact audit

    r_pad = np.pad(ref_mono[:L].astype(np.float32), (win_l // 2, win_l // 2))
    s_pad = np.pad(syn_mono[:L].astype(np.float32), (win_l // 2, win_l // 2))

    diffs = []
    for f in v_frames:
        st = int(f * hop)
        mr = np.abs(np.fft.rfft(r_pad[st : st + win_l] * w_bh, n=n_fft)) * norm
        ms = np.abs(np.fft.rfft(s_pad[st : st + win_l] * w_bh, n=n_fft)) * norm
        pk_frame = float(np.max(mr))
        if pk_frame < 1e-3:
            continue
        ff = float(f0[f])
        cs_r = np.insert(np.cumsum(mr ** 2), 0, 0.0)
        cs_s = np.insert(np.cumsum(ms ** 2), 0, 0.0)
        for h in range(1, 11):
            hc_hz = h * ff
            hbw = min(0.18 * ff, max(38.0, 0.032 * hc_hz))
            lo_b = max(1, int(math.floor((hc_hz - hbw) / bin_hz)))
            hi_b = min(n_fft // 2 - 2, int(math.ceil((hc_hz + hbw) / bin_hz)))
            if hi_b > lo_b:
                ar = math.sqrt(max(0.0, float(cs_r[hi_b + 1] - cs_r[lo_b])))
                ay = math.sqrt(max(0.0, float(cs_s[hi_b + 1] - cs_s[lo_b])))
                if ar > pk_frame * 0.03:
                    diffs.append(20.0 * math.log10((ay + 1e-7) / (ar + 1e-7)))
    arr = np.asarray(diffs, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    return round(float(np.sqrt(np.mean(arr ** 2))), 2), round(float(np.mean(arr)), 2)


def compute_envelope_corr(ref: np.ndarray, syn: np.ndarray, hop: int = 512) -> float:
    r_m = 0.5 * (ref[:, 0] + ref[:, 1]) if ref.ndim == 2 else ref
    s_m = 0.5 * (syn[:, 0] + syn[:, 1]) if syn.ndim == 2 else syn
    L = (min(len(r_m), len(s_m)) // hop) * hop
    er = np.sqrt(np.mean(r_m[:L].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    es = np.sqrt(np.mean(s_m[:L].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    c = np.corrcoef(er, es)[0, 1]
    return round(float(c), 4)


def main() -> None:
    t0 = time.time()
    config = json.loads((PROJECT_ROOT / "schemas/vocal_genesis_config.json").read_text(encoding="utf-8"))
    lexicon = load_phoneme_lexicon(PROJECT_ROOT / "schemas/phoneme_lexicon.json")

    mp3_path = WORKSPACE_ROOT / "mp3_output/李宇春 - 下个, 路口, 见.mp3"
    lrc_path = WORKSPACE_ROOT / "mp3_output/李宇春 - 下个, 路口, 见.lrc"

    sr = int(config["sampleRate"])
    hop = int(config["analysisHop"])

    print("[1/6] Decoding master & separating 3-stream vocal anatomy (Lead + True Stereo Backing + Diffuse Reverb)...")
    master = decode_stereo(mp3_path, sr=sr)
    n_samples = len(master)
    duration_sec = n_samples / float(sr)
    anatomy = extract_clean_vocal_anatomy(master, sr=sr, hop=hop)

    print("[2/6] Running Octave-Locked Viterbi F0 + 40-Cepstrum Exact-Harmonic DAP + Valley-HNR analysis...")
    features = analyze_vocal_signal(anatomy["lead_direct"], config, backing_stereo=anatomy["backing_stereo"])

    print("[3/6] Aligning LRC syllables & detecting Intro/Interlude/Outro hook events...")
    all_lines, sung_lines = parse_lrc_with_tokens(lrc_path, lexicon, duration_sec)
    raw_events = segment_song_into_events(sung_lines, features, config)

    print("[4/6] Learning Global Speaker Timbre Core, 40-Cepstrum Syllable Atlas & Non-Clipping SVD Basis...")
    speaker, prototypes, raw_events, ref_peak, atlas_stats = build_speaker_and_syllable_atlas(
        raw_events, features, anatomy, config
    )

    raw_speaker = pack_speaker_core(speaker, sr, n_samples, hop, ref_peak)
    comp_speaker = compress_lzma1_alone(raw_speaker)
    assert decompress_lzma1_alone(comp_speaker) == raw_speaker

    raw_atlas = pack_syllable_atlas(prototypes, atlas_q_step=0.22)
    comp_atlas = compress_lzma1_alone(raw_atlas)
    assert decompress_lzma1_alone(comp_atlas) == raw_atlas

    raw_fdn, _ = fit_and_pack_space_fdn(anatomy["lead_direct"], anatomy["reverb_residual"], sr=sr)
    comp_fdn = compress_lzma1_alone(raw_fdn)
    assert decompress_lzma1_alone(comp_fdn) == raw_fdn

    shared_dir = PROJECT_ROOT / "data/shared"
    write_b122_js_module(shared_dir / "speaker_core.js", comp_speaker)
    write_b122_js_module(shared_dir / "syllable_atlas.js", comp_atlas)
    write_b122_js_module(shared_dir / "space_fdn.js", comp_fdn)

    shared_bytes = len(comp_speaker) + len(comp_atlas) + len(comp_fdn)
    print(
        f"    Shared Core: speaker={len(comp_speaker)} B + syllable_atlas({len(prototypes)} protos)={len(comp_atlas)} B + space_fdn={len(comp_fdn)} B -> {shared_bytes/1024:.2f} KB"
    )
    print(f"    Atlas Stats: {json.dumps(atlas_stats, ensure_ascii=False)}")

    wav_equiv_bytes = n_samples * 2 * 2
    orig_mp3_bytes = mp3_path.stat().st_size

    lead_dry_keys = (
        "lead_sub_f0",
        "lead_vowel_body",
        "lead_harmonic_sheen",
        "lead_plosive_burst",
        "lead_sibilant",
        "lead_velvet_breath",
    )

    speaker_q, _ = unpack_speaker_core(raw_speaker)
    prototypes_q = unpack_syllable_atlas(raw_atlas)

    profiles_summary: dict[str, dict] = {}
    for prof_id, prof_spec in config["profiles"].items():
        print(f"[5/6] Encoding & synthesizing 8-track V8.5 profile `{prof_id}` ({prof_spec['label']})...")
        t_p = time.time()
        events_enc = encode_events_for_profile(
            raw_events, speaker_q, prototypes_q, features, anatomy, prof_spec, ref_peak
        )
        raw_ev = pack_event_stream(events_enc, prof_spec)
        comp_ev = compress_lzma1_alone(raw_ev)
        assert decompress_lzma1_alone(comp_ev) == raw_ev

        prof_dir = PROJECT_ROOT / f"data/{prof_id}"
        write_b122_js_module(prof_dir / "event_stream.js", comp_ev)

        syn_res = synthesize_from_packets(raw_speaker, raw_atlas, raw_ev, raw_fdn, config)
        mix_syn = syn_res["mix_stereo"]
        dry_stereo = sum(syn_res["layers"][k] for k in lead_dry_keys)
        dry_syn = 0.5 * (dry_stereo[:, 0] + dry_stereo[:, 1])

        mel_snr_full = compute_log_mel_snr_db(anatomy["vocals_stereo"], mix_syn, sr=sr)
        mel_snr_dry = compute_log_mel_snr_db(anatomy["lead_direct"], dry_syn, sr=sr)
        mel80_err, hi_bias_db = compute_mel80_and_hiband_db(anatomy["lead_direct"], dry_syn, sr=sr)
        harm10_rmse, harm10_bias = compute_narrowband_harmonic_metrics(
            anatomy["lead_direct"], dry_syn, features["f0"], features["voiced"], sr=sr, hop=hop
        )
        env_corr = compute_envelope_corr(anatomy["vocals_stereo"], mix_syn)
        peak_mix = round(float(np.max(np.abs(mix_syn))), 4)

        tot_comp = shared_bytes + len(comp_ev)
        tot_raw = len(raw_speaker) + len(raw_atlas) + len(raw_fdn) + len(raw_ev)
        kbps = round((tot_comp * 8.0) / (duration_sec * 1000.0), 2)

        manifest = {
            "version": "v8.5",
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
            "mel80RmseDb": mel80_err,
            "hiBandBiasDb": hi_bias_db,
            "harm10RmseDb": harm10_rmse,
            "harm10BiasDb": harm10_bias,
            "envelopeCorrelation": env_corr,
            "peakAmplitude": peak_mix,
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
            f"    -> `{prof_id}`: total={tot_comp:,} B ({tot_comp/1024:.2f} KB | {kbps} kbps | {manifest['bytesPerEventAvg']} B/event), "
            f"Dry Mel-SNR={mel_snr_dry} dB, Mel80-Err={mel80_err} dB, H1..10 RMSE={harm10_rmse} dB (bias {harm10_bias:+.2f} dB), "
            f"HiBias={hi_bias_db:+.2f} dB, Full Mel-SNR={mel_snr_full} dB, Peak={peak_mix} ({time.time() - t_p:.1f}s)"
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
        json.dumps({"version": "v8.5", "default": "expressive", "available": list(config["profiles"].keys())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sizes_report = {
        "version": "v8.5",
        "songTitle": "李宇春 - 下个, 路口, 见 (纯人声·八轨连音协同衍生版 V8.5)",
        "durationSec": round(duration_sec, 3),
        "referenceWav48k16bStereoBytes": wav_equiv_bytes,
        "referenceMp3Bytes": orig_mp3_bytes,
        "v7FiveTrackCompactBytes": 587038,
        "v7MergedSingleTrackBytes": 336076,
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
                "reductionVsV7CompactPct": round(100.0 * (1.0 - m["totalCompressedBytes"] / 587038.0), 2),
                "logMelSnrDryDb": m["logMelSnrDryDb"],
                "logMelSnrFullDb": m["logMelSnrFullDb"],
                "mel80RmseDb": m["mel80RmseDb"],
                "harm10RmseDb": m["harm10RmseDb"],
                "harm10BiasDb": m["harm10BiasDb"],
                "hiBandBiasDb": m["hiBandBiasDb"],
                "envelopeCorrelation": m["envelopeCorrelation"],
                "peakAmplitude": m["peakAmplitude"],
                "bytesPerEventAvg": m["bytesPerEventAvg"],
            }
            for pid, m in profiles_summary.items()
        },
    }
    (PROJECT_ROOT / "data/sizes.json").write_text(
        json.dumps(sizes_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[6/6] Completed V8.5 8-track compilation in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
