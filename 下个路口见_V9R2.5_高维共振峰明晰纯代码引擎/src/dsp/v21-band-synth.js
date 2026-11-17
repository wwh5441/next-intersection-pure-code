// V9R2.1 Pure-Code Band Synthesizer (`src/dsp/v21-band-synth.js`)
// 09 bass: harmonic-model oscillator bank | 10 drums: hit events x spectro-temporal templates
// 11 guitar+keys: polyphonic note runs x learned harmonic profiles
import { synthesizeBassHarmonic } from './bass-harmonic-synth.js';
import { synthesizeDrumEvents } from './drum-event-synth.js';
import { synthesizePolyNotes } from './poly-note-synth.js';

export function synthesizeV21Band(bandModels, bandScores, bandGains, numSamples, sr = 48000) {
  const bass = synthesizeBassHarmonic(bandModels.bass, bandScores.bass, numSamples, sr);
  const g = bandGains.bass ?? 1.0;
  if (g !== 1.0) for (let i = 0; i < numSamples; i++) bass[i] *= g;
  const drums = synthesizeDrumEvents(bandModels.drums, bandScores.drums, bandGains.drums, numSamples, sr);
  // Stereo-linked soft-knee transient safety ceiling after calibrated drum mix
  const dL = drums[0];
  const dR = drums[1];
  const knee = 0.78;
  const ceil = 0.92;
  const span = ceil - knee;
  for (let i = 0; i < numSamples; i++) {
    const pk = Math.max(Math.abs(dL[i]), Math.abs(dR[i]));
    if (pk > knee) {
      const sg = (knee + span * Math.tanh((pk - knee) / span)) / pk;
      dL[i] *= sg;
      dR[i] *= sg;
    }
  }
  const poly = synthesizePolyNotes(bandModels.poly, bandScores.poly, numSamples, sr, bandGains.poly);
  return {
    bass: [bass],
    drums,
    guitar_synth: poly,
  };
}
