// Pure Mathematical Singer & Studio Acoustic Constants (Zero Binary / Zero LZMA)
export const SINGER_PHYSICS = {
  sampleRate: 48000,
  hopSamples: 256,
  durationSec: 210.373,
  totalSamples: 10097920,
  refPeak: 1.0,
  erbFreqMinHz: 75.0,
  erbFreqMaxHz: 16000.0,
  numModes: 12,
  modeStepDb: 0.4,

  // 12-mode Glasberg-Moore ERB Fourier vocal-tract baseline resonance (dB):
  // L_base(f) = sum_{m=1..12} baseErbModes[m-1] * cos(m * pi * u_ERB(f))
  baseErbModes: [4.85,-2.42,1.95,-1.65,1.12,-0.84,0.58,-0.42,0.28,-0.19,0.12,-0.08],

  // Fant LF glottal + Velvet phase dispersion constants H1..H24 (radians)
  glottalPhase24: [0,0.34,0.79,1.32,1.94,2.65,0.22,1.11,2.08,3.12,1.04,2.25,0.38,1.76,0.09,1.62,0.12,1.81,0.47,2.31,1.14,0.03,2.08,1.11],

  // 8-band Klatt glottal open-phase HNR baseline (dB)
  aperHnrBaseDb: [-28.5,-26.0,-23.5,-20.5,-17.0,-13.5,-10.8,-8.4],

  // 2nd-order critically damped articulatory ODE constants (rad/s)
  coarticulationOmegaVowel: 38.0,
  coarticulationOmegaConsonant: 66.0,
  anticipatoryLookaheadSec: 0.068,

  // Studio 4-tap early reflections + 8-line prime Householder FDN
  earlyReflections: {
    delaysSamples: [638, 1094, 1685, 2390],
    gains: [0.24, -0.18, 0.14, -0.10],
  },
  fdnReverb: {
    delaysSamples: [1103,1361,1697,2029,2411,2833,3259,3761],
    rt60LowSec: 1.25,
    rt60HighSec: 0.50,
    dampingCutoffHz: 4200.0,
    wetSendVocal: 0.16,
    wetSendBand: 0.10,
  },
};
