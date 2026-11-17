// V9.3 Unfolded Explicit Sparse Orthonormal Cosine & 6-Stem Partition-of-Unity Decoder
// Pure Mathematical Reconstruction: c_k = sign(q_k) * 10^((0.5*u_b - 90)/20) * |q_k|^(4/3)

import { ORTHONORMAL_PHYSICS } from '../../models/orthonormal-basis-physics.js';
import { LONG_BINS, MODE_FOUR_SHORT } from './radix2-fft.js';

const POW43_MAX = 4096;

export function createSeriesDecoder() {
  const stepLut = new Float64Array(256);
  stepLut[0] = 0.0;
  for (let u = 1; u < 256; u++) {
    stepLut[u] = Math.pow(10.0, (0.5 * u + ORTHONORMAL_PHYSICS.stepMinDb) / 20.0);
  }

  const pow43Lut = new Float64Array(POW43_MAX);
  for (let q = 0; q < POW43_MAX; q++) {
    pow43Lut[q] = Math.pow(q, 4.0 / 3.0);
  }

  const slicesLong = ORTHONORMAL_PHYSICS.erbSlicesLong;
  const slicesShort = ORTHONORMAL_PHYSICS.erbSlicesShort;
  const numBands = slicesLong.length; // 32

  const binToBandLong = new Uint8Array(LONG_BINS);
  for (let b = 0; b < numBands; b++) {
    const [s0, s1] = slicesLong[b];
    for (let k = s0; k < s1; k++) binToBandLong[k] = b;
  }

  const binToBandShort = new Uint8Array(LONG_BINS);
  for (let b = 0; b < numBands; b++) {
    const [s0, s1] = slicesShort[b];
    for (let k = s0; k < s1; k++) binToBandShort[k] = b;
  }

  function decodeChannelFrame(frameEntry, outCoeffs2048, outStepCodes32) {
    outCoeffs2048.fill(0.0);
    outStepCodes32.fill(0);
    if (!frameEntry || frameEntry === 0) {
      return 0; // MODE_ONLY_LONG silent
    }
    const mode = frameEntry[0];
    const scArr = frameEntry[1];
    const pairs = frameEntry[2];
    for (let b = 0; b < scArr.length && b < numBands; b++) {
      outStepCodes32[b] = scArr[b];
    }

    const bMap = mode === MODE_FOUR_SHORT ? binToBandShort : binToBandLong;
    let k = -1;
    for (let i = 0; i < pairs.length; i += 2) {
      k += pairs[i] + 1;
      const q = pairs[i + 1];
      const sc = stepLut[outStepCodes32[bMap[k]]];
      const aq = q < 0 ? -q : q;
      const mag = aq < POW43_MAX ? pow43Lut[aq] : Math.pow(aq, 4.0 / 3.0);
      outCoeffs2048[k] = (q < 0 ? -sc : sc) * mag;
    }
    return mode;
  }

  // Unpacks 6x32 integer Partition-of-Unity matrices W_mid (sum===128) & W_side (sum===128, W_side[0]===0)
  function decodeStemWeightsFrame(partEntry, midSteps32, sideSteps32, wMid6x32, wSide6x32) {
    wMid6x32.fill(0);
    wSide6x32.fill(0);
    if (!partEntry || partEntry === 0) {
      for (let b = 0; b < numBands; b++) {
        wMid6x32[5 * numBands + b] = 128;
        wSide6x32[5 * numBands + b] = 128;
      }
      return;
    }

    const mFlat = partEntry[0] || [];
    const sFlat = partEntry[1] || [];
    let pM = 0;
    for (let b = 0; b < numBands; b++) {
      if (midSteps32[b] > 0 && pM + 5 <= mFlat.length) {
        let s5 = 0;
        for (let i = 0; i < 5; i++) {
          const w = mFlat[pM++];
          wMid6x32[i * numBands + b] = w;
          s5 += w;
        }
        wMid6x32[5 * numBands + b] = 128 - s5;
      } else {
        wMid6x32[5 * numBands + b] = 128;
      }
    }

    let pS = 0;
    for (let b = 0; b < numBands; b++) {
      if (sideSteps32[b] > 0 && pS + 4 <= sFlat.length) {
        let s4 = 0;
        wSide6x32[0 * numBands + b] = 0; // 01 Lead Vocal is pure center
        for (let i = 1; i < 5; i++) {
          const w = sFlat[pS++];
          wSide6x32[i * numBands + b] = w;
          s4 += w;
        }
        wSide6x32[5 * numBands + b] = 128 - s4;
      } else {
        wSide6x32[5 * numBands + b] = 128;
      }
    }
  }

  return {
    numBands,
    slicesLong,
    slicesShort,
    decodeChannelFrame,
    decodeStemWeightsFrame,
  };
}
