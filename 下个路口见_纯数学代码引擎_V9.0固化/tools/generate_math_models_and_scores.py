#!/usr/bin/env python3
"""Generates 100% human-readable plain-text JS mathematical models & scores (Zero LZMA, Zero Binary)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = Path("/Users/weckham/Desktop/MUSIC/下个路口见_事件音节波码")
sys.path.insert(0, str(V8_ROOT))

from parsers.lrc_tokenizer import load_phoneme_lexicon, parse_lrc_with_tokens
from services.event_codec import (
    decode_base122,
    decompress_lzma1_alone,
    unpack_event_stream,
    unpack_speaker_core,
    unpack_syllable_atlas,
)
from services.syllable_atlas_builder import (
    build_envelope_knot_positions,
    build_uniform_knot_positions,
)

DB_FLOOR = -78.0
DB_CEIL = 4.0

INITIALS_ORDER = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
]

# Physical consonant acoustic turbulence & DAC coarticulation resistance table:
# [isPlosive(0/1), dacR(0..1), f1Hz, bw1Oct, g1, f2Hz, bw2Oct, g2]
CONSONANT_PHYSICS: dict[str, list[float]] = {
    "":   [0, 0.18, 1200, 1.8, 0.05, 3200, 1.6, 0.03],
    "y":  [0, 0.35, 2350, 1.2, 0.08, 3600, 1.3, 0.05],
    "w":  [0, 0.35, 750,  1.2, 0.08, 2200, 1.5, 0.04],
    "m":  [0, 0.32, 320,  1.1, 0.10, 1150, 1.4, 0.05],
    "n":  [0, 0.48, 360,  1.1, 0.10, 1650, 1.3, 0.06],
    "l":  [0, 0.42, 420,  1.2, 0.10, 2450, 1.3, 0.06],
    "r":  [0, 0.58, 1750, 0.9, 0.28, 2850, 1.0, 0.22],
    "f":  [0, 0.45, 1650, 1.5, 0.36, 6800, 1.4, 0.42],
    "h":  [0, 0.28, 1350, 1.4, 0.38, 3800, 1.5, 0.28],
    "b":  [1, 0.42, 680,  1.4, 0.52, 1950, 1.5, 0.28],
    "p":  [1, 0.46, 750,  1.4, 0.68, 2600, 1.5, 0.46],
    "d":  [1, 0.65, 1750, 1.2, 0.56, 3900, 1.3, 0.38],
    "t":  [1, 0.68, 1900, 1.2, 0.72, 4600, 1.3, 0.55],
    "g":  [1, 0.72, 1450, 1.1, 0.58, 2900, 1.3, 0.36],
    "k":  [1, 0.75, 1550, 1.1, 0.74, 3400, 1.4, 0.52],
    "j":  [1, 0.82, 3600, 0.85, 0.58, 6400, 0.95, 0.48],
    "q":  [1, 0.85, 3800, 0.85, 0.76, 6900, 0.95, 0.66],
    "x":  [0, 0.80, 4100, 0.80, 0.68, 7400, 0.90, 0.62],
    "zh": [1, 0.76, 2550, 0.80, 0.62, 4800, 0.95, 0.46],
    "ch": [1, 0.78, 2700, 0.80, 0.78, 5200, 0.95, 0.62],
    "sh": [0, 0.74, 2850, 0.78, 0.72, 5400, 0.92, 0.58],
    "z":  [1, 0.70, 5200, 0.75, 0.60, 8600, 0.85, 0.52],
    "c":  [1, 0.74, 5500, 0.75, 0.78, 9200, 0.85, 0.68],
    "s":  [0, 0.72, 5800, 0.72, 0.74, 9600, 0.82, 0.66],
}


def _load_v8_pkt(rel_path: str) -> bytes:
    txt = (V8_ROOT / rel_path).read_text(encoding="utf-8")
    b122 = txt[txt.index('"') + 1 : txt.rindex('"')]
    return decompress_lzma1_alone(decode_base122(b122))


def split_initial(py: str) -> str:
    if py.startswith("HOOK"):
        return ""
    for cand in INITIALS_ORDER:
        if py.startswith(cand):
            return cand
    if py.startswith("y"):
        return "y"
    if py.startswith("w"):
        return "w"
    return ""


def q_db_to_float(q: int) -> float:
    if q <= 0:
        return -80.0
    return DB_FLOOR + ((q - 1.0) / 254.0) * (DB_CEIL - DB_FLOOR)


def compact_pitch_repr(full_midi: np.ndarray) -> float | list[float]:
    """Represents steady notes as a single scalar, linear glides as 2 points, and curves as 4..7 points."""
    L = len(full_midi)
    if L <= 1 or float(np.max(full_midi) - np.min(full_midi)) < 0.08:
        return round(float(np.median(full_midi)), 2)
    # Test 2-point linear glide
    x = np.arange(L, dtype=np.float64)
    p2 = [round(float(full_midi[0]), 2), round(float(full_midi[-1]), 2)]
    lin2 = np.interp(x, [0, L - 1], p2)
    if float(np.sqrt(np.mean((full_midi - lin2) ** 2))) < 0.08:
        return p2
    # Test 4-point contour vs 6-point contour for melismas
    for n_pts in (4, 6):
        xp = np.linspace(0, L - 1, n_pts)
        vp = [round(float(v), 2) for v in np.interp(xp, x, full_midi)]
        approx = np.interp(x, xp, vp)
        if n_pts == 6 or float(np.sqrt(np.mean((full_midi - approx) ** 2))) < 0.09:
            return vp
    return p2


def main() -> None:
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    raw_spk = _load_v8_pkt("data/shared/speaker_core.js")
    raw_atl = _load_v8_pkt("data/shared/syllable_atlas.js")
    raw_ev = _load_v8_pkt("data/expressive/event_stream.js")

    spk, meta = unpack_speaker_core(raw_spk)
    protos = unpack_syllable_atlas(raw_atl)
    ev_res = unpack_event_stream(raw_ev)
    events = ev_res[0] if isinstance(ev_res, tuple) else ev_res

    sr = int(meta["sampleRate"])
    n_samples = int(meta["numSamples"])
    hop = int(meta["hop"])
    ref_peak = float(meta["refPeak"])
    dur_sec = round(n_samples / float(sr), 3)

    # 1. `models/singer-physics.js`
    M_MODES = 12
    base_erb_modes = [round(float(x), 2) for x in spk.mean_mcep[:M_MODES]]
    glottal_phase_24 = [round(float(x), 3) for x in spk.glottal_phase[:24]]
    aper_base = [round(float(x), 1) for x in spk.aper_mean]
    fdn_delays = [1103, 1361, 1697, 2029, 2411, 2833, 3259, 3761]

    singer_js = f"""// Pure Mathematical Singer & Studio Acoustic Constants (Zero Binary / Zero LZMA)
export const SINGER_PHYSICS = {{
  sampleRate: {sr},
  hopSamples: {hop},
  durationSec: {dur_sec},
  totalSamples: {n_samples},
  refPeak: {round(ref_peak, 5)},
  erbFreqMinHz: 75.0,
  erbFreqMaxHz: 16000.0,
  numModes: {M_MODES},
  modeStepDb: 0.4,

  // 12-mode Glasberg-Moore ERB Fourier vocal-tract baseline resonance (dB):
  // L_base(f) = sum_{{m=1..12}} baseErbModes[m-1] * cos(m * pi * u_ERB(f))
  baseErbModes: {json.dumps(base_erb_modes, separators=(",", ":"))},

  // Fant LF glottal + Velvet phase dispersion constants H1..H24 (radians)
  glottalPhase24: {json.dumps(glottal_phase_24, separators=(",", ":"))},

  // 8-band Klatt glottal open-phase HNR baseline (dB)
  aperHnrBaseDb: {json.dumps(aper_base, separators=(",", ":"))},

  // 2nd-order critically damped articulatory ODE constants (rad/s)
  coarticulationOmegaVowel: 38.0,
  coarticulationOmegaConsonant: 66.0,
  anticipatoryLookaheadSec: 0.068,

  // Studio 4-tap early reflections + 8-line prime Householder FDN
  earlyReflections: {{
    delaysSamples: [638, 1094, 1685, 2390],
    gains: [0.24, -0.18, 0.14, -0.10],
  }},
  fdnReverb: {{
    delaysSamples: {json.dumps(fdn_delays, separators=(",", ":"))},
    rt60LowSec: 1.25,
    rt60HighSec: 0.50,
    dampingCutoffHz: 4200.0,
    wetSendVocal: 0.16,
    wetSendBand: 0.10,
  }},
}};
"""
    (PROJECT_ROOT / "models/singer-physics.js").write_text(singer_js, encoding="utf-8")

    # 2. `models/mandarin-phonetics.js` (Mode step = 0.4 dB -> compact 1-2 digit ints, 2.95 dB accuracy!)
    M_CONS = 8
    syllable_entries = []
    proto_by_id = {}
    for p in protos:
        proto_by_id[p.syllable_id] = p
        ini = split_initial(p.pinyin)
        c_vec = np.mean(p.cons_mcep_knots[:, :M_CONS], axis=0)
        vk = p.vowel_mcep_knots[:, :M_MODES]
        v_stages = [vk[0], vk[2], vk[4]]
        c_ints = [int(np.round(x * 2.5)) for x in c_vec]  # 1 unit = 0.4 dB
        v_ints = [[int(np.round(x * 2.5)) for x in st] for st in v_stages]
        syllable_entries.append(
            f'  "{p.pinyin}":["{ini}",{json.dumps(c_ints, separators=(",", ":"))},{json.dumps(v_ints, separators=(",", ":"))}]'
        )

    cons_lines = [
        f'  "{k}":{json.dumps(v, separators=(",", ":"))}'
        for k, v in CONSONANT_PHYSICS.items()
    ]
    phonetics_js = (
        "// Pure Mathematical Mandarin Phonetics & Cavity Mode Atlas (Zero Binary / Zero LZMA)\n"
        "// Mode unit = 0.40 dB in Glasberg-Moore ERB Fourier cosine space:\n"
        "//   H_vocal(f, t) = exp((ln(10)/20) * sum_{m=1..12} (0.4*v_m(t)) * exp(-j * m * pi * u_ERB(f)))\n"
        "// Syllable schema: [initialKey, consonantModes8[], [onsetModes12[], nucleusModes12[], codaModes12[]]]\n\n"
        "export const MANDARIN_CONSONANTS = {\n"
        + ",\n".join(cons_lines)
        + "\n};\n\n"
        "export const MANDARIN_SYLLABLES = {\n"
        + ",\n".join(syllable_entries)
        + "\n};\n"
    )
    (PROJECT_ROOT / "models/mandarin-phonetics.js").write_text(phonetics_js, encoding="utf-8")

    # 3. `models/instrument-physics.js` — Physical differential & waveguide synthesis constants for the 5 band stems
    inst_js = """// Pure Mathematical Instrumental Physical Modeling Constants (Zero Samples / Zero Binary)
export const INSTRUMENT_PHYSICS = {
  // Track 06: Membrane Kick + Bessel Dual-Mode Snare + 4-Tap Studio Handclap
  kick: {
    fInfHz: 51.5,
    fStartHz: 164.0,
    pitchTauSec: 0.024,
    clickHz: 920.0,
    clickTauSec: 0.0032,
    ampDecaySec: 0.215,
    drive: 1.32,
  },
  snare: {
    mode1Hz: 188.0,
    mode2Hz: 334.0,
    mode1TauSec: 0.062,
    mode2TauSec: 0.038,
    wireCenterHz: 2750.0,
    wireQ: 0.95,
    wireTauFastSec: 0.088,
    wireTauSlowSec: 0.185,
    clapOffsetsMs: [0.0, 9.5, 19.2, 30.8],
    clapCenterHz: 1180.0,
    clapQ: 1.75,
    clapTauSec: 0.095,
  },

  // Track 07: 6-Oscillator Non-Harmonic Metallic Cymbal Matrix + Stereo Shaker
  cymbal: {
    metallicFreqsHz: [291.5, 432.2, 524.8, 742.2, 766.8, 1136.0],
    highpassHz: 6800.0,
    closedTauSec: 0.028,
    openTauSec: 0.165,
    crashTauSec: 1.35,
    shakerCenterHz: 5200.0,
    shakerTauSec: 0.024,
    swing16thSec: 0.0055,
  },

  // Track 08: Fingered J-Bass + Chamberlin 2-Pole State-Variable Envelope Filter
  bass: {
    pickupCombRatio: 0.18,
    subSineMix: 0.52,
    svfBaseCutoffHz: 185.0,
    svfEnvModHz: 920.0,
    svfEnvTauSec: 0.068,
    svfQ: 1.42,
    ampReleaseSec: 0.045,
    kickDuckDb: -2.4,
    kickDuckTauSec: 0.065,
  },

  // Track 09: Extended Karplus-Strong Digital Waveguide Clean Stratocaster Funk Guitar
  guitar: {
    pickupPosRatio: 0.14,
    openLoopLoss: 0.9945,
    palmMuteLoss: 0.885,
    brightnessBlend: 0.34,
    strumSpreadMs: 5.8,
    stereoDoubleDelayMs: 8.6,
  },

  // Track 10: 2-Operator FM Rhodes Electric Piano + 3-Voice Detuned Stereo Chorus Pad
  keys: {
    fmModRatio: 1.0,
    fmTineRatio: 7.0,
    fmIndexBase: 1.45,
    fmIndexTauSec: 0.28,
    fmTineIndex: 0.26,
    fmTineTauSec: 0.055,
    epDecaySec: 1.45,
    padDetuneCents: [-8.5, 0.0, 8.5],
    padCutoffHz: 1750.0,
    padAttackSec: 0.18,
    padReleaseSec: 0.42,
  },
};
"""
    (PROJECT_ROOT / "models/instrument-physics.js").write_text(inst_js, encoding="utf-8")

    # 4. `data/band-score.js` — Exact 98.000 BPM 85-bar arrangement & harmonic/rhythmic patterns
    band_js = """// Pure-Code Musical Accompaniment Score for 《李宇春 - 下个, 路口, 见》
// Locked Studio Clock: 98.000 BPM (1 bar = 2.44898s, Bar 0 Downbeat = 1.7493s, Pickup at 0.372s)

export const BAND_SCORE = {
  bpm: 98.0,
  bar0Sec: 1.7493,
  totalBars: 83,

  // Harmonic Voicings (MIDI note numbers: Guitar upper funk triad/9th + Keys warm 4-voice Rhodes/Pad)
  chords: {
    Dm9:    { bass: 38, altBass: 50, keys: [50, 57, 60, 64], guitar: [65, 69, 72, 76] }, // D2 | D3 A3 C4 E4 | F4 A4 C5 E5
    Bbmaj7: { bass: 34, altBass: 46, keys: [46, 53, 57, 62], guitar: [62, 65, 69, 74] }, // Bb1| Bb2 F3 A3 D4| D4 F4 A4 D5
    Cadd9:  { bass: 36, altBass: 48, keys: [48, 55, 60, 64], guitar: [64, 67, 72, 74] }, // C2 | C3 G3 C4 E4 | E4 G4 C5 D5
    Am7:    { bass: 33, altBass: 45, keys: [45, 52, 57, 60], guitar: [60, 64, 67, 72] }, // A1 | A2 E3 A3 C4 | C4 E4 G4 C5
    D9end:  { bass: 38, altBass: 50, keys: [50, 54, 57, 60, 64], guitar: [62, 66, 69, 72, 76] }, // Final Picardy D9 chord (Bar 80)
  },

  // 4-Bar Harmonic Cycle Templates ([firstHalfChord, secondHalfChord] per bar)
  progressions: {
    STD4:  [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Dm9"]],
    ROLL4: [["Dm9", "Bbmaj7"], ["Cadd9", "Am7"], ["Dm9", "Bbmaj7"], ["Cadd9", "Am7"]],
  },

  // 16-Step Drum Groove Patterns:
  //   k = Kick vel (0..9), s = Snare/Clap vel (0..9), h = Hat ('c'=closed, 'o'=open offbeat, 'p'=pedal, '.'=rest), sh = Shaker (0..9)
  drumPatterns: {
    INTRO_A: {
      k:  [9,0,0,0, 9,0,0,0, 9,0,0,0, 8,0,0,0],
      s:  [0,0,0,2, 8,0,2,0, 0,0,2,0, 9,2,0,0],
      h:  "cc.occ.occ.occ.o",
      sh: [3,2,4,3, 3,2,5,3, 3,2,4,3, 3,2,5,3],
    },
    VERSE: {
      k:  [9,0,0,0, 9,0,0,0, 9,0,0,0, 8,0,2,0],
      s:  [0,0,0,3, 8,0,2,0, 0,0,3,0, 9,2,0,1],
      h:  "cc.occ.occ.occ.o",
      sh: [4,3,5,6, 4,3,6,4, 4,3,6,3, 5,4,6,4],
    },
    FILL_BAR: {
      k:  [9,0,0,0, 9,0,0,0, 9,0,4,0, 8,0,6,0],
      s:  [0,0,0,3, 8,0,3,0, 0,3,5,4, 9,6,8,9],
      h:  "cc.occ.occ.occcc",
      sh: [4,3,5,5, 4,3,6,4, 5,4,6,5, 6,6,7,7],
    },
    CHORUS: {
      k:  [9,0,0,0, 9,0,0,0, 9,0,0,0, 9,0,3,0],
      s:  [0,2,0,4, 9,0,4,0, 0,0,4,3, 9,3,5,2],
      h:  "cc.occ.occ.occ.o",
      sh: [5,4,6,7, 5,4,8,5, 5,4,7,4, 6,6,8,5],
    },
    BREAK_V2: {
      k:  [4,0,0,0, 0,0,0,0, 3,0,0,0, 0,0,0,0],
      s:  [0,0,0,0, 3,0,0,0, 0,0,0,0, 3,0,0,0],
      h:  "c..pc..pc..pc..p",
      sh: [3,2,4,3, 3,2,4,3, 3,2,4,3, 3,2,4,3],
    },
    BUILD_V2: {
      k:  [7,0,0,0, 8,0,0,0, 8,0,5,0, 9,0,6,0],
      s:  [0,0,0,2, 6,0,3,0, 4,0,5,3, 8,5,7,8],
      h:  "cc.occ.occcccccc",
      sh: [4,3,5,5, 5,4,6,5, 6,5,7,6, 7,7,8,8],
    },
  },

  // 16-Step Bass Groove Patterns per Bar (steps 0..7 use chord[0], steps 8..15 use chord[1]):
  //   tuple [step, durSteps, useOctave(0/1), vel(0..1), chromaticApproach(-1/0/+1)]
  bassPatterns: {
    BOUNCE: [
      [0, 2.6, 0, 0.92, 0], [3, 0.9, 1, 0.55, 0], [4, 2.5, 0, 0.85, 0], [6, 1.5, 0, 0.68, 0],
      [8, 2.6, 0, 0.94, 0], [11, 0.9, 1, 0.58, 0], [12, 2.4, 0, 0.88, 0], [14, 1.5, 0, 0.72, 1],
    ],
    CHORUS_DRIVE: [
      [0, 2.2, 0, 0.96, 0], [2, 1.4, 1, 0.66, 0], [4, 2.2, 0, 0.90, 0], [6, 1.6, 1, 0.74, 0],
      [8, 2.2, 0, 0.96, 0], [10, 1.4, 1, 0.68, 0], [12, 2.2, 0, 0.92, 0], [14, 1.6, 0, 0.78, 1],
    ],
    LIGHT: [
      [0, 3.2, 0, 0.82, 0], [4, 2.8, 0, 0.72, 0],
      [8, 3.2, 0, 0.84, 0], [12, 2.8, 0, 0.74, 0],
    ],
  },

  // 16-Step Clean Funk Guitar Strum Patterns:
  //   '.'=sustain/rest, 'D'=open downstroke, 'U'=open upstroke, 'm'=palm-mute chuck, 'A'=accent funk upstroke
  guitarPatterns: {
    FUNK_A: "D.mUDUA.D.mADmU.",
    FUNK_B: "D.mUDUA.D.mUDUAU",
    SPARSE: "D..UD.m.D..UD.m.",
  },

  // Pre-Bar-0 2-Beat Studio Pickup (0.372s .. 1.7493s)
  pickupHits: [
    { sec: 0.372, type: "snare", vel: 0.45 },
    { sec: 0.525, type: "kick_bass", midi: 38, dur: 0.52, vel: 0.88 },
    { sec: 0.984, type: "snare", vel: 0.52 },
    { sec: 1.137, type: "kick_bass", midi: 38, dur: 0.52, vel: 0.86 },
    { sec: 1.596, type: "snare", vel: 0.62 },
  ],

  // Full 83-Bar Song Structure Map (Bars 0..82 -> covers 1.75s to 205.0s + natural ringout to 210.37s)
  sections: [
    { name: "Intro",        startBar: 0,  bars: 8,  prog: "STD4",  drum: "INTRO_A",  bass: "BOUNCE",       gtr: "FUNK_A", epGain: 0.0,  padGain: 0.0,  crashFirst: true,  gtrEnterHalfBar: 1 },
    { name: "Verse 1",      startBar: 8,  bars: 8,  prog: "STD4",  drum: "VERSE",    bass: "BOUNCE",       gtr: "FUNK_A", epGain: 0.18, padGain: 0.0,  crashFirst: false, fillLast: true },
    { name: "Pre-Chorus 1", startBar: 16, bars: 8,  prog: "STD4",  drum: "VERSE",    bass: "BOUNCE",       gtr: "FUNK_B", epGain: 0.48, padGain: 0.24, crashFirst: true,  fillLast: true },
    { name: "Chorus 1",     startBar: 24, bars: 8,  prog: "ROLL4", drum: "CHORUS",   bass: "CHORUS_DRIVE", gtr: "FUNK_B", epGain: 0.56, padGain: 0.46, crashFirst: true,  fillLast: true, resolveLastDm: true },
    { name: "Interlude",    startBar: 32, bars: 8,  prog: "STD4",  drum: "VERSE",    bass: "BOUNCE",       gtr: "FUNK_B", epGain: 0.52, padGain: 0.22, crashFirst: true,  fillLast: false },
    { name: "Verse 2A",     startBar: 40, bars: 1,  prog: "STD4",  drum: "VERSE",    bass: "LIGHT",        gtr: "SPARSE", epGain: 0.22, padGain: 0.0,  crashFirst: false, fillLast: false },
    { name: "Verse 2 Drop", startBar: 41, bars: 3,  prog: "STD4",  drum: "BREAK_V2", bass: "LIGHT",        gtr: "SPARSE", epGain: 0.25, padGain: 0.0,  crashFirst: false, fillLast: false, progOffset: 1 },
    { name: "Verse 2B",     startBar: 44, bars: 4,  prog: "STD4",  drum: "VERSE",    bass: "BOUNCE",       gtr: "FUNK_A", epGain: 0.30, padGain: 0.12, crashFirst: false, fillLast: true,  firstBarDrum: "BUILD_V2" },
    { name: "Pre-Chorus 2", startBar: 48, bars: 8,  prog: "STD4",  drum: "VERSE",    bass: "BOUNCE",       gtr: "FUNK_B", epGain: 0.52, padGain: 0.30, crashFirst: true,  fillLast: true },
    { name: "Chorus 2",     startBar: 56, bars: 8,  prog: "ROLL4", drum: "CHORUS",   bass: "CHORUS_DRIVE", gtr: "FUNK_B", epGain: 0.60, padGain: 0.50, crashFirst: true,  fillLast: true, resolveLastDm: true },
    { name: "Chorus 3",     startBar: 64, bars: 8,  prog: "ROLL4", drum: "CHORUS",   bass: "CHORUS_DRIVE", gtr: "FUNK_B", epGain: 0.64, padGain: 0.56, crashFirst: true,  fillLast: true, resolveLastDm: true },
    { name: "Outro Groove", startBar: 72, bars: 8,  prog: "STD4",  drum: "CHORUS",   bass: "CHORUS_DRIVE", gtr: "FUNK_B", epGain: 0.72, padGain: 0.68, crashFirst: true,  fillLast: true },
    { name: "Final Chord",  startBar: 80, bars: 3,  prog: "FINAL", drum: "FINAL",    bass: "FINAL",        gtr: "FINAL",  epGain: 0.75, padGain: 0.55, crashFirst: true,  fillLast: false },
  ],
};
"""
    (PROJECT_ROOT / "data/band-score.js").write_text(band_js, encoding="utf-8")

    # 5. `data/vocal-score.js` — Compact, clean, human-readable 285 vocal notes!
    pca_basis = spk.mcep_pca_basis
    note_lines = []
    for ev in events:
        L = int(ev.total_frames)
        start_ms = int(round(ev.start_frame * hop * 1000.0 / sr))
        dur_ms = int(round(L * hop * 1000.0 / sr))
        vot_ms = int(round(ev.cons_frames * hop * 1000.0 / sr))

        # Reconstruct full MIDI contour safely (handles empty f0_step_q on short hooks!)
        cur_m = ev.base_midi_q / 50.0
        if len(ev.f0_step_q) == 0:
            full_midi = np.full(L, cur_m, dtype=np.float64)
        else:
            n_fk = len(ev.f0_step_q) + 1
            kpos_f = build_uniform_knot_positions(L, n_fk)
            midi_k = np.full(len(kpos_f), cur_m, dtype=np.float64)
            for j in range(min(len(ev.f0_step_q), len(midi_k) - 1)):
                cur_m += ev.f0_step_q[j] * 0.025
                midi_k[j + 1] = cur_m
            full_midi = np.interp(np.arange(L), kpos_f, midi_k) if len(kpos_f) > 0 else np.full(L, cur_m)

        pitch_repr = compact_pitch_repr(full_midi)

        # Reconstruct dB envelope contour -> 4 integers in dB (1 dB step, human JND!)
        kpos_e = build_envelope_knot_positions(L, int(ev.cons_frames), len(ev.gain_h_db_q))
        g_db_k = np.array([q_db_to_float(int(q)) for q in ev.gain_h_db_q[: len(kpos_e)]], dtype=np.float64)
        full_gdb = np.interp(np.arange(L), kpos_e, g_db_k) if len(kpos_e) > 0 else np.full(L, -60.0)
        n_g = 3 if L <= 28 else 4
        g_pos = np.linspace(0, max(0, L - 1), n_g)
        g_vals = [int(round(float(x))) for x in np.interp(g_pos, np.arange(L), full_gdb)]

        # 4-mode timbre expression shift in 0.4 dB units (or 0 if neutral!)
        if ev.expr_mcep_q.size > 0:
            k_use = ev.expr_mcep_q.shape[1]
            mid_row = ev.expr_mcep_q[len(ev.expr_mcep_q) // 2]
            delta_cep = (mid_row.astype(np.float64) * 0.14) @ pca_basis[:k_use, :4]
            t4 = [int(np.clip(np.round(x * 2.5), -35, 35)) for x in delta_cep[:4]]
            timbre_repr: int | list[int] = 0 if all(abs(v) <= 1 for v in t4) else t4
        else:
            timbre_repr = 0

        bg_q = max(int(ev.back_gain_l_q), int(ev.back_gain_r_q))
        bg_db = int(round(q_db_to_float(bg_q))) if bg_q > 4 else -80

        flags = (
            (1 if ev.legato_prev else 0)
            | (2 if ev.legato_next else 0)
            | (4 if ev.is_plosive else 0)
            | (8 if ev.is_hook else 0)
        )

        p_obj = proto_by_id[ev.syllable_id]
        disp = ev.display_text.replace('"', '\\"')
        note_lines.append(
            f'  [{start_ms},{dur_ms},{vot_ms},"{disp}","{p_obj.pinyin}",{json.dumps(pitch_repr, separators=(",", ":"))},{json.dumps(g_vals, separators=(",", ":"))},{json.dumps(timbre_repr, separators=(",", ":"))},{bg_db},{flags}]'
        )

    lexicon = load_phoneme_lexicon(V8_ROOT / "schemas/phoneme_lexicon.json")
    _, sung_lines = parse_lrc_with_tokens(
        Path("/Users/weckham/Desktop/MUSIC/mp3_output/李宇春 - 下个, 路口, 见.lrc"),
        lexicon,
        dur_sec,
    )
    lrc_entries = [
        f'  [{int(round(ln["startSec"] * 1000.0))}, "{ln["text"].replace(chr(34), chr(92)+chr(34))}"]'
        for ln in sung_lines
    ]

    vocal_score_js = (
        "// Pure-Code Readable Vocal Score for 《李宇春 - 下个, 路口, 见》 (285 Sung Notes)\n"
        "// Tuple: [startMs, durMs, votMs, hanzi, pinyin, midiPitch, gainDb[], timbreDelta4, backingDb, flags]\n"
        "// Flags: 1=legatoPrev(--), 2=legatoNext(--), 4=plosiveBurst, 8=hookVocal\n\n"
        "export const VOCAL_LYRIC_LINES = [\n"
        + ",\n".join(lrc_entries)
        + "\n];\n\n"
        "export const VOCAL_SCORE = [\n"
        + ",\n".join(note_lines)
        + "\n];\n"
    )
    (PROJECT_ROOT / "data/vocal-score.js").write_text(vocal_score_js, encoding="utf-8")

    tot = 0
    for rel in (
        "models/singer-physics.js",
        "models/mandarin-phonetics.js",
        "models/instrument-physics.js",
        "data/band-score.js",
        "data/vocal-score.js",
    ):
        sz = (PROJECT_ROOT / rel).stat().st_size
        tot += sz
        print(f"{rel:32s}: {sz:6d} B ({sz/1024:.2f} KB)")
    print(f"{'TOTAL PLAIN-TEXT MODELS+SCORES':32s}: {tot:6d} B ({tot/1024:.2f} KB)")


if __name__ == "__main__":
    main()
