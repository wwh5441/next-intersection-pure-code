// V9R2.1 Discrete Guitar Pluck & Keys/Rhodes Note Synthesizer (`src/dsp/poly-note-synth.js`)
// Synthesizes separated Guitar (inst=0, with physical pick/plectrum transient) and Keys/Rhodes (inst=1)
// from compact 9-integer discrete musical note events `[inst, midiIdx, start, dur, pkDbQ, atk, decayQ, sustQ, pan]`.
import { createMulberry32 } from './fft-core.js';

export function synthesizePolyNotes(model, score, numSamples, sr = 48000, gains = { guitar: 1.0, keys: 1.0, pick: 0.35 }) {
  const hop = model.hop;
  const outL = new Float32Array(numSamples);
  const outR = new Float32Array(numSamples);
  const nH = model.nHarm;
  const tun = Math.pow(2.0, model.tuning / 12.0);
  const qDb = model.qDb ?? 0.5;
  const twoPi = 2.0 * Math.PI;
  const rng = createMulberry32(4242);

  const buildProfiles = (mat) =>
    mat.map((row) => {
      const a = new Float64Array(nH);
      let sq = 0.0;
      for (let h = 0; h < nH; h++) {
        const v = Math.pow(10.0, (row[h] * 0.5) / 20.0);
        a[h] = v;
        sq += v * v;
      }
      const norm = 1.0 / Math.sqrt(0.5 * sq + 1e-18);
      for (let h = 0; h < nH; h++) a[h] *= norm;
      return a;
    });

  const guitarProf = buildProfiles(model.guitarHarmDbQ);
  const keysProf = buildProfiles(model.keysHarmDbQ);
  const gGuitar = gains.guitar ?? gains.tonal ?? 1.0;
  const gKeys = gains.keys ?? gains.tonal ?? 1.0;
  const gPick = gains.pick ?? 0.30;

  let startAcc = 0;
  for (const [inst, j, dStart, dur, pkDbQ, atk, decayQ, sustQ, pan = 8] of score.notes) {
    startAcc += dStart;
    const start = startAcc;
    const instGain = inst === 0 ? gGuitar : gKeys;
    if (instGain <= 0.0) continue;

    const th = (pan / 16.0) * (Math.PI * 0.5);
    const gL = Math.cos(th) * Math.SQRT2;
    const gR = Math.sin(th) * Math.SQRT2;
    const midi = model.midiLo + j;
    const f0 = 440.0 * Math.pow(2.0, (midi - 69) / 12.0) * tun;
    const prof = (inst === 0 ? guitarProf : keysProf)[j];
    const nAct = Math.min(nH, Math.floor(18500.0 / f0));
    const ampPk = Math.pow(10.0, (pkDbQ * qDb) / 20.0) * instGain;
    const sustFrac = sustQ * 0.1;

    const s0 = Math.max(0, start * hop);
    const durSamples = Math.max(hop, dur * hop);
    const sEnd = Math.min(numSamples, s0 + durSamples + Math.round(0.025 * sr));
    const atkSamples = Math.max(24, Math.min(durSamples >> 1, atk * hop));
    const relSamples = Math.min(Math.round(0.035 * sr), Math.max(hop, durSamples >> 2));
    const tauDec = Math.max(hop, decayQ * hop);

    let phi = rng() * twoPi;
    const w = (twoPi * f0) / sr;
    const c1 = Math.cos(w);
    const s1 = Math.sin(w);
    let cph = Math.cos(phi);
    let sph = Math.sin(phi);

    // Physical plectrum/pick burst on sharp acoustic guitar plucks
    if (inst === 0 && gPick > 0.0 && atk <= 3 && midi >= 52) {
      const pickLen = Math.min(sEnd - s0, Math.round(0.014 * sr));
      const pickTau = 0.0032 * sr;
      const pickAmp = ampPk * gPick * Math.min(1.4, 0.7 + (midi - 52) / 36.0);
      let lp = 0.0;
      let hp = 0.0;
      for (let i = 0; i < pickLen; i++) {
        const wn = (rng() * 2.0 - 1.0) * pickAmp * Math.exp(-i / pickTau);
        lp += 0.45 * (wn - lp); // ~5.5 kHz gentle bright roll
        hp = lp - 0.85 * hp; // ~1.2 kHz highpass so pick never muddies bass
        outL[s0 + i] += gL * hp;
        outR[s0 + i] += gR * hp;
      }
    }

    for (let s = s0; s < sEnd; s++) {
      const t = s - s0;
      let env;
      if (t < atkSamples) {
        env = 0.5 * (1.0 - Math.cos((Math.PI * t) / atkSamples));
      } else if (t < durSamples) {
        const td = t - atkSamples;
        env = sustFrac + (1.0 - sustFrac) * Math.exp(-td / tauDec);
      } else {
        const tr = t - durSamples;
        const baseE = sustFrac + (1.0 - sustFrac) * Math.exp(-(durSamples - atkSamples) / tauDec);
        env = baseE * Math.max(0.0, 1.0 - tr / relSamples);
      }
      const nc = cph * c1 - sph * s1;
      sph = sph * c1 + cph * s1;
      cph = nc;
      if (env <= 1e-5) continue;

      const two = 2.0 * cph;
      let sp = 0.0;
      let sh = sph;
      let accS = 0.0;
      for (let h = 0; h < nAct; h++) {
        accS += prof[h] * sh;
        const sn = two * sh - sp;
        sp = sh;
        sh = sn;
      }
      const v = ampPk * env * accS;
      outL[s] += gL * v;
      outR[s] += gR * v;
    }
  }
  return [outL, outR];
}
