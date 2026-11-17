#!/usr/bin/env python3
"""WEC Standalone Decoder & Audio Runner (`wec_run.py`).

Usage:
  python3 wec_run.py encode   # Pack V9R2.8 -> 下个路口见_V9R2.8.wec + verify 100% lossless round-trip
  python3 wec_run.py run 下个路口见_V9R2.8.wec [--wav out.wav] [--play]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from wec_codec import decode_wec, encode_project_to_wec

ROOT = Path(__file__).resolve().parent
V9R28 = ROOT.parent / "下个路口见_V9R2.8_零电流纯净高保真版"
DEFAULT_WEC = ROOT / "下个路口见_V9R2.8.wec"


def verify_lossless(project_dir: Path, wec_path: Path) -> None:
    decoded = decode_wec(wec_path)
    for rel, dec_val in decoded.items():
        orig_txt = (project_dir / rel).read_text(encoding="utf-8")
        orig_val = json.loads(orig_txt.split("=", 1)[1].rstrip().rstrip(";"))
        assert dec_val == orig_val, f"Round-trip mismatch in {rel}!"
    print(f"[PASS] 100% lossless round-trip verified across all {len(decoded)} modules.")


def run_wec(wec_path: Path, out_wav: Path, play_audio: bool = False) -> None:
    t0 = time.time()
    objs = decode_wec(wec_path)
    t_dec = time.time() - t0

    meta = objs["data/studio-metadata.js"]
    render_args = {
        "config": meta["config"],
        "speakerModel": objs["models/vocal-speaker-model.js"],
        "phonemeAtlas": objs["models/vocal-phoneme-atlas.js"],
        "backingModel": objs["models/backing-model.js"],
        "spaceFdn": objs["models/space-fdn.js"],
        "vocalScore": objs["data/vocal-score.js"],
        "backingScore": objs["data/backing-score.js"],
        "bandModels": {
            "bass": objs["models/bass-model.js"],
            "drums": objs["models/drum-model.js"],
            "poly": objs["models/poly-model.js"],
        },
        "bandScores": {
            "bass": objs["data/bass-score.js"],
            "drums": objs["data/drum-score.js"],
            "poly": objs["data/poly-score.js"],
        },
        "bandGains": objs["data/band-gains.js"],
    }

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        args_json = td_p / "args.json"
        args_json.write_text(json.dumps(render_args, ensure_ascii=False), encoding="utf-8")
        mix_f32 = td_p / "mix.f32"
        dsp_uri = (ROOT / "dsp/v9r2-master-engine.js").as_uri()
        js = f"""
import fs from 'node:fs';
import {{ renderV9r2AllLayers }} from '{dsp_uri}';
const args = JSON.parse(fs.readFileSync({json.dumps(str(args_json))}, 'utf-8'));
const r = renderV9r2AllLayers(args);
const N = r.numSamples;
const mixL = new Float32Array(N);
const mixR = new Float32Array(N);
for (const chs of Object.values(r.layers)) {{
  const c0 = chs[0], c1 = chs.length > 1 ? chs[1] : chs[0];
  for (let i = 0; i < N; i++) {{ mixL[i] += c0[i]; mixR[i] += c1[i]; }}
}}
const interleaved = new Float32Array(N * 2);
for (let i = 0; i < N; i++) {{
  interleaved[i * 2] = Math.max(-1, Math.min(1, mixL[i] * 0.96));
  interleaved[i * 2 + 1] = Math.max(-1, Math.min(1, mixR[i] * 0.96));
}}
fs.writeFileSync({json.dumps(str(mix_f32))}, Buffer.from(interleaved.buffer));
console.log(JSON.stringify({{ synthMs: r.synthMs, sampleRate: r.sampleRate, numSamples: N }}));
"""
        p = subprocess.run(["node", "--max-old-space-size=8192", "--input-type=module", "-e", js], capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr[-2000:])
        info = json.loads(p.stdout.strip())
        pcm_f = np.fromfile(mix_f32, dtype=np.float32)
        pcm_i16 = np.clip(np.round(pcm_f * 32767.0), -32768, 32767).astype("<i2")

        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(int(info["sampleRate"]))
            wf.writeframes(pcm_i16.tobytes())

    t_tot = time.time() - t0
    dur_s = info["numSamples"] / info["sampleRate"]
    print(f"[WEC RUN] {wec_path.name} ({wec_path.stat().st_size / 1024:.2f} KB)")
    print(f"          decode={t_dec*1000:.1f}ms | dsp_synth={info['synthMs']}ms | total={t_tot:.2f}s")
    print(f"          output={out_wav} ({dur_s:.2f}s @ {info['sampleRate']}Hz stereo 16-bit WAV)")

    if play_audio:
        subprocess.run(["afplay", str(out_wav)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="WEC Encoder & Audio Runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("encode")
    p_run = sub.add_parser("run")
    p_run.add_argument("wec", nargs="?", default=str(DEFAULT_WEC))
    p_run.add_argument("--wav", default=str(ROOT / "下个路口见_WEC解码输出_48kHz.wav"))
    p_run.add_argument("--play", action="store_true")
    args = ap.parse_args()

    if args.cmd == "encode":
        stats = encode_project_to_wec(V9R28, DEFAULT_WEC)
        print(f"[WEC ENCODE] {DEFAULT_WEC.name}: {stats['wecKB']} KB ({stats['wecBytes']} bytes) | pre-LZMA binary={stats['binaryPayloadKB']} KB")
        verify_lossless(V9R28, DEFAULT_WEC)
    elif args.cmd == "run":
        run_wec(Path(args.wec), Path(args.wav), args.play)


if __name__ == "__main__":
    main()
