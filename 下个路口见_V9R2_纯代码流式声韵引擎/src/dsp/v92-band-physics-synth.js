// V9.2 Pure-Physics 3-Track Band Synthesizer adapted to V9R2 Layers (`src/dsp/v92-band-physics-synth.js`)
// Drums: swept-kick modal ODE + 164 Hz woodblock + percussive spectral shapes
// Bass: 18-harmonic inharmonic stiff-string equation + true slide pitch curves
// Guitar/Keys: 56-note coupled-rotor oscillators + strum transients + stereo FDN
import { synthesizePureMathDrums } from './band-physics/math-drums-synth.js';
import { synthesizePureMathBass, synthesizePureMathGuitarAndKeys } from './band-physics/math-bass-guitar-synth.js';

function sumStereo(pairs, n) {
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  for (const [a, b] of pairs) {
    for (let i = 0; i < n; i++) {
      L[i] += a[i];
      R[i] += b[i];
    }
  }
  return [L, R];
}

function scaleInPlace(chans, g) {
  for (const c of chans) for (let i = 0; i < c.length; i++) c[i] *= g;
}

export function synthesizeV92PhysicsBand(instrumentPhysics, bandScore, bandGains, sampleRate, numSamples) {
  const drums = synthesizePureMathDrums(bandScore, instrumentPhysics, numSamples);
  const bass = synthesizePureMathBass(bandScore, instrumentPhysics, numSamples);
  const harm = synthesizePureMathGuitarAndKeys(bandScore, instrumentPhysics, sampleRate, numSamples);

  const drumsSt = sumStereo([drums.drum_kick_snare, drums.drum_hihat_perc], numSamples);
  const [bL, bR] = bass.bass_groove;
  const bassMono = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) bassMono[i] = 0.5 * (bL[i] + bR[i]);
  const gtrSt = sumStereo([harm.guitar_funk_riff, harm.keys_rhodes_pad], numSamples);

  // Three scalar loudness constants calibrated against the original 48kHz stems
  scaleInPlace([bassMono], bandGains.bass);
  scaleInPlace(drumsSt, bandGains.drums);
  scaleInPlace(gtrSt, bandGains.guitarSynth);

  return {
    bass: [bassMono],
    drums: drumsSt,
    guitar_synth: gtrSt,
  };
}
