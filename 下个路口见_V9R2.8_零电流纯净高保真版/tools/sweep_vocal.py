#!/usr/bin/env python3
"""Sweep a synthesis-side config parameter (no re-analysis) and report lead/vocals Mel SNR."""
import sys, json
from pathlib import Path
ROOT = Path("/Users/weckham/Desktop/MUSIC/下个路口见_V9R2.1_纯代码最高音质引擎")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import exp_vocal as ev

key = sys.argv[1]
vals = [float(v) for v in sys.argv[2].split(",")]
cfg = json.loads((ROOT / "schemas/v9r2_genesis_config.json").read_text(encoding="utf-8"))
st = ev.load_prep()
for v in vals:
    c = dict(cfg); c.update(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}); c[key] = v
    r = ev.render_vocals(c)
    print(key, v, "lead", ev.mel_snr(st["lead_mono"], r["lead"]), "backing", ev.mel_snr(st["backing_stereo"], r["backing"]), "space", ev.mel_snr(st["space_stereo"], r["space"]), "vocals", ev.mel_snr(st["vocals_stereo"], r["vocals"]), flush=True)
