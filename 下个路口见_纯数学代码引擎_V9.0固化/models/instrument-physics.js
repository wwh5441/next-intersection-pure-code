// Pure Mathematical Instrumental Physical Modeling Constants (Zero Samples / Zero Binary)
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
