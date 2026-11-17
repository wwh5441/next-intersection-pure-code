// 98.0000 BPM Drum Groove Motif + Hit Atom (`06`) & 6-Band Studio Cymbal Shimmer (`07`) Synthesizer
// (`src/dsp/drum-percussion-synth.js`)

import { synthesizeLappedStereoEvents } from './lapped-atom-synth.js';
import {
  applyZeroPhaseSymmetricFir,
  designSymmetricBandpassHalf,
  resampleZeroPhasePolyStereo,
} from './polyphase-sinc.js';

const BAR_MOTIF_CACHE = new WeakMap();
const CYMBAL_PREP_CACHE = new WeakMap();

export function getDecodedBarPrototypes(atomDict, barMotifSpec) {
  if (BAR_MOTIF_CACHE.has(barMotifSpec)) return BAR_MOTIF_CACHE.get(barMotifSpec);
  const { motifCount, barLen } = barMotifSpec;
  const [concatL, concatR] = synthesizeLappedStereoEvents(atomDict, barMotifSpec.protoEvents);

  const protos = new Array(motifCount);
  for (let m = 0; m < motifCount; m++) {
    const off = m * barLen;
    const pL = new Float32Array(barLen);
    const pR = new Float32Array(barLen);
    let energy = 0.0;
    for (let i = 0; i < barLen; i++) {
      const l = concatL[off + i] || 0.0;
      const r = concatR[off + i] || 0.0;
      pL[i] = l;
      pR[i] = r;
      energy += l * l + r * r;
    }
    const invNorm = 1.0 / (Math.sqrt(energy) + 1e-12);
    for (let i = 0; i < barLen; i++) {
      pL[i] *= invNorm;
      pR[i] *= invNorm;
    }
    protos[m] = [pL, pR];
  }
  BAR_MOTIF_CACHE.set(barMotifSpec, protos);
  return protos;
}

export function synthesizeBarMotifPlusFill(atomDict, barMotifSpec, arrangementSpec, upRatio, totalSamples48) {
  const protos = getDecodedBarPrototypes(atomDict, barMotifSpec);
  const { barLen, exactSubBar, subBar0 } = barMotifSpec;
  const { totalSubSamples, scaleQuant = 512, barSchedule, fillEvents } = arrangementSpec;

  const subL = new Float32Array(totalSubSamples);
  const subR = new Float32Array(totalSubSamples);
  const invScale = 1.0 / scaleQuant;

  for (let b = 0; b < barSchedule.length; b++) {
    const [motifIdx, scaleQ] = barSchedule[b];
    if (motifIdx < 0 || scaleQ <= 0) continue;
    const gain = scaleQ * invScale;
    const [pL, pR] = protos[motifIdx];
    const st = Math.round(subBar0 + b * exactSubBar);
    const endI = Math.min(barLen, totalSubSamples - st);
    if (st >= 0 && endI > 0) {
      for (let i = 0; i < endI; i++) {
        subL[st + i] += pL[i] * gain;
        subR[st + i] += pR[i] * gain;
      }
    }
  }

  if (fillEvents && fillEvents.midFrames && fillEvents.midFrames.length > 0) {
    const [fL, fR] = synthesizeLappedStereoEvents(atomDict, fillEvents);
    const nAdd = Math.min(totalSubSamples, fL.length);
    for (let i = 0; i < nAdd; i++) {
      subL[i] += fL[i];
      subR[i] += fR[i];
    }
  }

  return resampleZeroPhasePolyStereo(subL, subR, upRatio, 1, totalSamples48);
}

function getPreparedCymbalCarrierAndMotifs(cymbalDict) {
  if (CYMBAL_PREP_CACHE.has(cymbalDict)) return CYMBAL_PREP_CACHE.get(cymbalDict);

  const {
    sampleRate = 48000,
    beatLoopLen,
    carrierScale = 16384,
    carrierBeatL,
    carrierBeatR,
    bandEdgesHz,
    basisScale = 16384,
    spectralBasis,
    barEnvFrames,
    barEnvMotifCoeffs,
  } = cymbalDict;

  const pad = 128;
  const paddedLen = beatLoopLen + 2 * pad;
  const rawL = new Float32Array(paddedLen);
  const rawR = new Float32Array(paddedLen);
  const invC = 1.0 / carrierScale;

  for (let i = 0; i < paddedLen; i++) {
    const srcIdx = ((i - pad) % beatLoopLen + beatLoopLen) % beatLoopLen;
    rawL[i] = carrierBeatL[srcIdx] * invC;
    rawR[i] = carrierBeatR[srcIdx] * invC;
  }

  const numBands = bandEdgesHz.length;
  const bandCarrierL = new Array(numBands);
  const bandCarrierR = new Array(numBands);

  for (let b = 0; b < numBands; b++) {
    const [fLo, fHi] = bandEdgesHz[b];
    const bpHalf = designSymmetricBandpassHalf(fLo, fHi, sampleRate, 96, 8.0);
    const bpL = applyZeroPhaseSymmetricFir(rawL, bpHalf).subarray(pad, pad + beatLoopLen);
    const bpR = applyZeroPhaseSymmetricFir(rawR, bpHalf).subarray(pad, pad + beatLoopLen);

    // Pure linear bandpass normalized to global RMS = 1.0 (zero nonlinear sliding-RMS AM distortion!)
    let sumL = 0.0;
    let sumR = 0.0;
    for (let i = 0; i < beatLoopLen; i++) {
      sumL += bpL[i] * bpL[i];
      sumR += bpR[i] * bpR[i];
    }
    const invRmsL = 1.0 / Math.sqrt(sumL / beatLoopLen + 1e-12);
    const invRmsR = 1.0 / Math.sqrt(sumR / beatLoopLen + 1e-12);
    const cL = new Float32Array(beatLoopLen);
    const cR = new Float32Array(beatLoopLen);
    for (let i = 0; i < beatLoopLen; i++) {
      cL[i] = bpL[i] * invRmsL;
      cR[i] = bpR[i] * invRmsR;
    }
    bandCarrierL[b] = cL;
    bandCarrierR[b] = cR;
  }

  // Decode spectral basis (`K_e` x `2 * numBands`)
  const ke = spectralBasis.length;
  const dim = 2 * numBands;
  const basisFlat = new Float32Array(ke * dim);
  const invB = 1.0 / basisScale;
  for (let k = 0; k < ke; k++) {
    for (let d = 0; d < dim; d++) {
      basisFlat[k * dim + d] = spectralBasis[k][d] * invB;
    }
  }

  // Decode & unit-normalize 1-bar envelope motif prototypes (`motifCount` x `barEnvFrames * dim`)
  const motifCount = barEnvMotifCoeffs.length;
  const envProtos = new Array(motifCount);
  for (let m = 0; m < motifCount; m++) {
    const coeffRows = barEnvMotifCoeffs[m];
    const proto = new Float32Array(barEnvFrames * dim);
    let energy = 0.0;
    for (let f = 0; f < barEnvFrames; f++) {
      const cRow = coeffRows[f];
      const off = f * dim;
      for (let d = 0; d < dim; d++) {
        let val = 0.0;
        for (let k = 0; k < ke; k++) {
          val += cRow[k] * basisFlat[k * dim + d];
        }
        const clamped = val > 0.0 ? val : 0.0;
        proto[off + d] = clamped;
        energy += clamped * clamped;
      }
    }
    const invN = 1.0 / (Math.sqrt(energy) + 1e-12);
    for (let i = 0; i < proto.length; i++) proto[i] *= invN;
    envProtos[m] = proto;
  }

  const prepared = { beatLoopLen, numBands, dim, ke, basisFlat, barEnvFrames, envProtos, bandCarrierL, bandCarrierR };
  CYMBAL_PREP_CACHE.set(cymbalDict, prepared);
  return prepared;
}

export function synthesizeHiHatPercTrack(cymbalDict, hihatScore, totalSamples48) {
  const {
    beatLoopLen,
    numBands,
    dim,
    ke,
    basisFlat,
    barEnvFrames,
    envProtos,
    bandCarrierL,
    bandCarrierR,
  } = getPreparedCymbalCarrierAndMotifs(cymbalDict);

  const {
    envHop = 512,
    exactBarFrames,
    bar0Frame,
    bar0Sample,
    scaleQuant = 1024,
    varQuant = 4096,
    barSchedule,
    varFrames,
  } = hihatScore;

  const numFrames = Math.ceil(totalSamples48 / envHop) + 1;
  const envGrid = new Float32Array(numFrames * dim);
  const invScale = 1.0 / scaleQuant;

  // 1. Expand 98.0000 BPM 1-Bar Hi-Hat/Shaker Envelope Motifs
  for (let b = 0; b < barSchedule.length; b++) {
    const [mIdx, sQ] = barSchedule[b];
    if (mIdx < 0 || sQ <= 0) continue;
    const gain = sQ * invScale;
    const proto = envProtos[mIdx];
    const sf = Math.round(bar0Frame + b * exactBarFrames);
    const fEnd = Math.min(barEnvFrames, numFrames - sf);
    if (sf >= 0 && fEnd > 0) {
      for (let f = 0; f < fEnd; f++) {
        const dstOff = (sf + f) * dim;
        const srcOff = f * dim;
        for (let d = 0; d < dim; d++) {
          envGrid[dstOff + d] = proto[srcOff + d] * gain;
        }
      }
    }
  }

  // 2. Apply Sparse Turnaround Crash / Open-Hat / Shaker Variations
  if (varFrames && varFrames.length > 0) {
    const invVar = 1.0 / varQuant;
    let fIdx = 0;
    for (let i = 0; i < varFrames.length; i++) {
      const tup = varFrames[i];
      fIdx += tup[0];
      if (fIdx < 0 || fIdx >= numFrames) continue;
      const dstOff = fIdx * dim;
      for (let d = 0; d < dim; d++) {
        let delta = 0.0;
        for (let k = 0; k < ke; k++) {
          delta += tup[1 + k] * invVar * basisFlat[k * dim + d];
        }
        const updated = envGrid[dstOff + d] + delta;
        envGrid[dstOff + d] = updated > 0.0 ? updated : 0.0;
      }
    }
  }

  // 3. Modulate 6-Band Real Studio Cymbal/Shaker Carrier at 48 kHz with smooth frame-to-frame linear ramp
  const outL = new Float32Array(totalSamples48);
  const outR = new Float32Array(totalSamples48);
  const invHop = 1.0 / envHop;

  for (let f = 0; f < numFrames - 1; f++) {
    const st = f * envHop;
    const endN = Math.min(envHop, totalSamples48 - st);
    if (endN <= 0) break;
    const off0 = f * dim;
    const off1 = (f + 1) * dim;

    for (let b = 0; b < numBands; b++) {
      const gL0 = envGrid[off0 + b];
      const gL1 = envGrid[off1 + b];
      const gR0 = envGrid[off0 + numBands + b];
      const gR1 = envGrid[off1 + numBands + b];
      if (gL0 + gL1 + gR0 + gR1 < 1e-7) continue;

      const uL = bandCarrierL[b];
      const uR = bandCarrierR[b];
      let phase = (((st - bar0Sample) % beatLoopLen) + beatLoopLen) % beatLoopLen;

      // Exact block-RMS calibration against `bandCarrier` on this exact 512-sample beat phase!
      let cSqL = 0.0;
      let cSqR = 0.0;
      let pScan = phase;
      for (let n = 0; n < endN; n += 2) {
        cSqL += uL[pScan] * uL[pScan];
        cSqR += uR[pScan] * uR[pScan];
        pScan += 2;
        if (pScan >= beatLoopLen) pScan -= beatLoopLen;
      }
      const cnt = Math.max(1, Math.ceil(endN / 2));
      const cRmsL = Math.max(0.22, Math.sqrt(cSqL / cnt));
      const cRmsR = Math.max(0.22, Math.sqrt(cSqR / cnt));

      let gL = gL0 / cRmsL;
      let gR = gR0 / cRmsR;
      const dGL = (gL1 / cRmsL - gL) * invHop;
      const dGR = (gR1 / cRmsR - gR) * invHop;

      for (let n = 0; n < endN; n++) {
        outL[st + n] += uL[phase] * gL;
        outR[st + n] += uR[phase] * gR;
        gL += dGL;
        gR += dGR;
        phase++;
        if (phase === beatLoopLen) phase = 0;
      }
    }
  }

  return [outL, outR];
}
