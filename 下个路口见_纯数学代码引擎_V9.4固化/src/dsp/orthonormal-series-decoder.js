// V9.4 Lossless Macro-Unfolding Orthonormal Cosine & 6-Stem Partition Decoder
// 100% Bijective / Bit-Exact Identical Reconstruction to V9.3 (0.000000 dB Difference)

import { ORTHONORMAL_PHYSICS } from '../../models/orthonormal-basis-physics.js';
import {
  SINGLE_PAIR_BOOK,
  DUAL_PAIR_BOOK,
  PARTITION_MID_BOOK,
  PARTITION_SIDE_BOOK,
} from '../../models/sparse-macro-book.js';
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

  const nSingles = SINGLE_PAIR_BOOK.length; // 512

  function decodeChannelFrame(frameEntry, outCoeffs2048, outStepCodes32) {
    outCoeffs2048.fill(0.0);
    outStepCodes32.fill(0);
    if (!frameEntry || frameEntry === 0) {
      return 0;
    }

    const mode = frameEntry[0];
    const scArr = frameEntry[1];
    const toks = frameEntry[2];
    for (let b = 0; b < scArr.length && b < numBands; b++) {
      outStepCodes32[b] = scArr[b];
    }

    const bMap = mode === MODE_FOUR_SHORT ? binToBandShort : binToBandLong;
    let k = -1;

    const putCoeff = (skip, q) => {
      k += skip + 1;
      const sc = stepLut[outStepCodes32[bMap[k]]];
      const aq = q < 0 ? -q : q;
      const mag = aq < POW43_MAX ? pow43Lut[aq] : Math.pow(aq, 4.0 / 3.0);
      outCoeffs2048[k] = (q < 0 ? -sc : sc) * mag;
    };

    for (let i = 0; i < toks.length; ) {
      const code = toks[i++];
      if (code === 0) {
        const s = toks[i++];
        const q = toks[i++];
        putCoeff(s, q);
      } else if (code <= nSingles) {
        const pair = SINGLE_PAIR_BOOK[code - 1];
        putCoeff(pair[0], pair[1]);
      } else {
        const quad = DUAL_PAIR_BOOK[code - 1 - nSingles];
        putCoeff(quad[0], quad[1]);
        putCoeff(quad[2], quad[3]);
      }
    }
    return mode;
  }

  function unpackMidTuples(toks, outFlat) {
    let wPos = 0;
    for (let i = 0; i < toks.length; ) {
      const c = toks[i++];
      if (c < 0) {
        const run = -c;
        const vec = PARTITION_MID_BOOK[toks[i++] - 1];
        for (let r = 0; r < run; r++) {
          outFlat[wPos++] = vec[0];
          outFlat[wPos++] = vec[1];
          outFlat[wPos++] = vec[2];
          outFlat[wPos++] = vec[3];
          outFlat[wPos++] = vec[4];
        }
      } else if (c > 0) {
        const vec = PARTITION_MID_BOOK[c - 1];
        outFlat[wPos++] = vec[0];
        outFlat[wPos++] = vec[1];
        outFlat[wPos++] = vec[2];
        outFlat[wPos++] = vec[3];
        outFlat[wPos++] = vec[4];
      } else {
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
      }
    }
    return wPos;
  }

  function unpackSideTuples(toks, outFlat) {
    let wPos = 0;
    for (let i = 0; i < toks.length; ) {
      const c = toks[i++];
      if (c < 0) {
        const run = -c;
        const vec = PARTITION_SIDE_BOOK[toks[i++] - 1];
        for (let r = 0; r < run; r++) {
          outFlat[wPos++] = vec[0];
          outFlat[wPos++] = vec[1];
          outFlat[wPos++] = vec[2];
          outFlat[wPos++] = vec[3];
        }
      } else if (c > 0) {
        const vec = PARTITION_SIDE_BOOK[c - 1];
        outFlat[wPos++] = vec[0];
        outFlat[wPos++] = vec[1];
        outFlat[wPos++] = vec[2];
        outFlat[wPos++] = vec[3];
      } else {
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
        outFlat[wPos++] = toks[i++];
      }
    }
    return wPos;
  }

  const scratchMidFlat = new Int32Array(32 * 5);
  const scratchSideFlat = new Int32Array(32 * 4);

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

    const lenM = unpackMidTuples(partEntry[0] || [], scratchMidFlat);
    const lenS = unpackSideTuples(partEntry[1] || [], scratchSideFlat);

    let pM = 0;
    for (let b = 0; b < numBands; b++) {
      if (midSteps32[b] > 0 && pM + 5 <= lenM) {
        let s5 = 0;
        for (let i = 0; i < 5; i++) {
          const w = scratchMidFlat[pM++];
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
      if (sideSteps32[b] > 0 && pS + 4 <= lenS) {
        let s4 = 0;
        wSide6x32[0 * numBands + b] = 0;
        for (let i = 1; i < 5; i++) {
          const w = scratchSideFlat[pS++];
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
