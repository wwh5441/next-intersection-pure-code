// Pure-Code Musical Accompaniment Score for 《李宇春 - 下个, 路口, 见》
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
