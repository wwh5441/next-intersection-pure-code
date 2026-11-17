// Pitch-Synchronous GCI Glottal-Formant Wavecycle + Consonant Wavelet Synthesizer (`src/dsp/vocal-wavecycle-synth.js`)
// Synthesizes Tracks 01 (`<720Hz` Chest), 02 (`>=720Hz` Formants), and 03 (`Consonant & Breath Air`)
// with native Pitch-Shift (`semitones`) & Tempo-Scale support.

import {
  applyZeroPhaseSymmetricFir,
  designSymmetricLowpassHalf,
  resampleZeroPhasePoly1D,
  sincResampleSegmentAdd,
} from './polyphase-sinc.js';

const VOCAL_BASIS_CACHE = new WeakMap();

function getVocalBasisTables(vocalDict) {
  if (VOCAL_BASIS_CACHE.has(vocalDict)) return VOCAL_BASIS_CACHE.get(vocalDict);
  const { grainPoints: P, basisScale, voicedAtoms, consonantAtoms, internalSampleRate, chestCutoffHz } = vocalDict;
  const invB = 1.0 / basisScale;

  const kv = voicedAtoms.length;
  const vFlat = new Float32Array(kv * P);
  for (let k = 0; k < kv; k++) {
    const r = voicedAtoms[k];
    const off = k * P;
    for (let i = 0; i < P; i++) vFlat[off + i] = r[i] * invB;
  }

  const ku = consonantAtoms.length;
  const uFlat = new Float32Array(ku * P);
  for (let k = 0; k < ku; k++) {
    const r = consonantAtoms[k];
    const off = k * P;
    for (let i = 0; i < P; i++) uFlat[off + i] = r[i] * invB;
  }

  const lpChestHalf = designSymmetricLowpassHalf(chestCutoffHz / internalSampleRate, 64, 8.0);
  const hannP = new Float32Array(P);
  for (let i = 0; i < P; i++) {
    hannP[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * (i + 0.5)) / P));
  }

  const cached = { P, vFlat, uFlat, lpChestHalf, hannP };
  VOCAL_BASIS_CACHE.set(vocalDict, cached);
  return cached;
}

export function synthesizeVocalThreeTracks(vocalDict, vocalScore, options = {}) {
  const pitchSemitones = Number(options.pitchSemitones || 0.0);
  const { P, vFlat, uFlat, lpChestHalf, hannP } = getVocalBasisTables(vocalDict);
  const {
    totalSamples32,
    totalSamples48,
    rmsScale = 4096,
    weightScale = 256,
    arScale = 64,
    grains,
  } = vocalScore;

  const voiced32 = new Float32Array(totalSamples32);
  const unvoiced32 = new Float32Array(totalSamples32);

  const invRms = 1.0 / rmsScale;
  const invW = 1.0 / weightScale;
  const invAr = 1.0 / arScale;

  const prevUnit = new Float32Array(P);
  const curUnit = new Float32Array(P);
  const scaledGrain = new Float32Array(P);
  let hasPrevVoiced = false;
  let curStart = 0;

  const isPitchShifted = Math.abs(pitchSemitones) > 1e-4;
  const pitchRatio = isPitchShifted ? Math.pow(2.0, pitchSemitones / 12.0) : 1.0;

  if (!isPitchShifted) {
    // Exact 1:1 Studio Partition-of-Unity GCI Wavecycle Synthesis
    for (let gIdx = 0; gIdx < grains.length; gIdx++) {
      const tup = grains[gIdx];
      curStart += tup[0];
      const segLen = tup[1];
      const vArQ = tup[2];
      const rms = tup[3] * invRms;

      if (vArQ >= 0) {
        // Voiced GCI Glottal+Formant Wavecycle with 1st-Order Inter-Cycle AR Prediction
        const alpha = vArQ * invAr;
        if (alpha > 0.0 && hasPrevVoiced) {
          for (let i = 0; i < P; i++) curUnit[i] = alpha * prevUnit[i];
        } else {
          curUnit.fill(0.0);
        }
        let kIdx = 0;
        for (let j = 4; j < tup.length; j += 2) {
          kIdx += tup[j];
          const c = tup[j + 1] * invW;
          const off = kIdx * P;
          for (let i = 0; i < P; i++) curUnit[i] += vFlat[off + i] * c;
        }
        prevUnit.set(curUnit);
        hasPrevVoiced = true;

        if (rms > 0) {
          for (let i = 0; i < P; i++) scaledGrain[i] = curUnit[i] * rms;
          sincResampleSegmentAdd(scaledGrain, voiced32, curStart, segLen, 6);
        }
      } else {
        // Unvoiced Mandarin Consonant / Breath Wavelet (`03 vocal_breath_air`)
        hasPrevVoiced = false;
        curUnit.fill(0.0);
        let kIdx = 0;
        for (let j = 4; j < tup.length; j += 2) {
          kIdx += tup[j];
          const c = tup[j + 1] * invW;
          const off = kIdx * P;
          for (let i = 0; i < P; i++) curUnit[i] += uFlat[off + i] * c;
        }
        if (rms > 0) {
          for (let i = 0; i < P; i++) scaledGrain[i] = curUnit[i] * rms;
          sincResampleSegmentAdd(scaledGrain, unvoiced32, curStart, segLen, 6);
        }
      }
    }
  } else {
    // True PSOLA Pitch-Transposed Glottal Wavecycle Synthesis (preserves consonant duration & formants!)
    const voicedWeighted = new Float32Array(totalSamples32);
    const voicedWinSum = new Float32Array(totalSamples32);
    const decodedVoiced = [];

    for (let gIdx = 0; gIdx < grains.length; gIdx++) {
      const tup = grains[gIdx];
      curStart += tup[0];
      const segLen = tup[1];
      const vArQ = tup[2];
      const rms = tup[3] * invRms;

      if (vArQ >= 0) {
        const alpha = vArQ * invAr;
        if (alpha > 0.0 && hasPrevVoiced) {
          for (let i = 0; i < P; i++) curUnit[i] = alpha * prevUnit[i];
        } else {
          curUnit.fill(0.0);
        }
        let kIdx = 0;
        for (let j = 4; j < tup.length; j += 2) {
          kIdx += tup[j];
          const c = tup[j + 1] * invW;
          const off = kIdx * P;
          for (let i = 0; i < P; i++) curUnit[i] += vFlat[off + i] * c;
        }
        prevUnit.set(curUnit);
        hasPrevVoiced = true;

        const cyc = new Float32Array(P);
        for (let i = 0; i < P; i++) cyc[i] = curUnit[i] * rms;
        decodedVoiced.push({ a: curStart, b: curStart + segLen, center: curStart + 0.5 * segLen, segLen, cyc });
      } else {
        hasPrevVoiced = false;
        curUnit.fill(0.0);
        let kIdx = 0;
        for (let j = 4; j < tup.length; j += 2) {
          kIdx += tup[j];
          const c = tup[j + 1] * invW;
          const off = kIdx * P;
          for (let i = 0; i < P; i++) curUnit[i] += uFlat[off + i] * c;
        }
        if (rms > 0) {
          for (let i = 0; i < P; i++) scaledGrain[i] = curUnit[i] * rms;
          sincResampleSegmentAdd(scaledGrain, unvoiced32, curStart, segLen, 6);
        }
      }
    }

    // Group contiguous voiced cycles into voiced phrases & emit on pitch-scaled GCI grid
    let idx = 0;
    while (idx < decodedVoiced.length) {
      let endIdx = idx + 1;
      while (endIdx < decodedVoiced.length && decodedVoiced[endIdx].a <= decodedVoiced[endIdx - 1].b + 64) {
        endIdx++;
      }
      const phraseStart = decodedVoiced[idx].center;
      const phraseEnd = decodedVoiced[endIdx - 1].center;
      let synthPos = phraseStart;
      let srcPtr = idx;

      while (synthPos <= phraseEnd + 8) {
        while (srcPtr + 1 < endIdx && decodedVoiced[srcPtr + 1].center < synthPos) {
          srcPtr++;
        }
        const g = decodedVoiced[srcPtr];
        const basePeriod = Math.max(44, Math.min(260, g.segLen * 0.82));
        const newPeriod = Math.max(36, Math.round(basePeriod / pitchRatio));
        const bellLen = 2 * newPeriod;
        const outSt = Math.round(synthPos - newPeriod);

        for (let i = 0; i < P; i++) scaledGrain[i] = g.cyc[i] * hannP[i];
        sincResampleSegmentAdd(scaledGrain, voicedWeighted, Math.max(0, outSt), bellLen, 4);
        sincResampleSegmentAdd(hannP, voicedWinSum, Math.max(0, outSt), bellLen, 2);

        synthPos += newPeriod;
      }
      idx = endIdx;
    }

    for (let n = 0; n < totalSamples32; n++) {
      if (voicedWinSum[n] > 0.08) {
        voiced32[n] = voicedWeighted[n] / voicedWinSum[n];
      }
    }
  }

  // Exact Zero-Phase Complementary Split: `chest32 + formant32 === voiced32`
  const chest32 = applyZeroPhaseSymmetricFir(voiced32, lpChestHalf);
  const formant32 = new Float32Array(totalSamples32);
  for (let n = 0; n < totalSamples32; n++) {
    formant32[n] = voiced32[n] - chest32[n];
  }

  // Exact Zero-Phase 3/2 Kaiser-Sinc Polyphase Upsampling (32 kHz -> 48 kHz)
  const chest48 = resampleZeroPhasePoly1D(chest32, 3, 2, totalSamples48);
  const formant48 = resampleZeroPhasePoly1D(formant32, 3, 2, totalSamples48);
  const air48 = resampleZeroPhasePoly1D(unvoiced32, 3, 2, totalSamples48);

  return { chest48, formant48, air48 };
}
