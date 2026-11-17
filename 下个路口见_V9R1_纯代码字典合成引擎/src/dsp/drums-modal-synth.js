// V9R1 Pure Mathematical 2-Track Drum & Percussion Modal/Transient Synthesizer
// Driven by Expanded Bar-Motif Dictionary Trajectories (`100fps`):
//   06 drum_kick_snare (58.59Hz Swept-Sine Kick ODE + 164.06Hz Woodblock/Snare Modal Body + <3.3kHz Transients)
//   07 drum_hihat_perc (64-Prototype >=3.3kHz Hi-Hat / Crash / Shaker Transients)

import {
  ComplexInverseFFT,
  DeterministicRng,
  interpPiecewiseLinear,
  upsample2xPolyphase,
} from './fft-minphase.js';

const TWO_PI = 2.0 * Math.PI;
const N_OLA = 512;
const HALF_OLA = N_OLA >> 1;
const N_BINS = HALF_OLA + 1;
const LEAD_PAD = 8;

function dequantizeDrumRms(q) {
  return q > 0.0 ? Math.pow(10.0, (q / 3.2 - 75.0) / 20.0) : 0.0;
}

export function synthesizeV9r1Drums(drumTimeline, instPhysics, nSamples) {
  const model = instPhysics.drums;
  const sr24 = model.sampleRate;
  const hop = model.hopSamples;
  const { nFrames, kickQ, bodyQ, highQ, shapeIdx } = drumTimeline;

  const bandCenters = Float64Array.from(model.bandCentersHz);
  const nBands = bandCenters.length;
  const nShapes = model.percShapeCodebookQ.length;
  const invShapeScale = 1.0 / model.shapeScaleDiv;

  const olaFreqs = new Float64Array(N_BINS);
  const wLow = new Float64Array(N_BINS);
  const wHigh = new Float64Array(N_BINS);
  const fLow = model.crossoverLowHz;
  const fHigh = model.crossoverHighHz;

  for (let i = 0; i < N_BINS; i++) {
    const f = (i * sr24) / N_OLA;
    olaFreqs[i] = f;
    let wl = 1.0;
    if (f >= fHigh) wl = 0.0;
    else if (f > fLow) wl = 0.5 * (1.0 + Math.cos((Math.PI * (f - fLow)) / (fHigh - fLow)));
    wLow[i] = wl;
    wHigh[i] = 1.0 - wl;
  }

  const protoCalMag = new Float32Array(nShapes * N_BINS);
  const tmpBand = new Float64Array(nBands);
  const tmpSpec = new Float64Array(N_BINS);
  for (let k = 0; k < nShapes; k++) {
    const qRow = model.percShapeCodebookQ[k];
    for (let b = 0; b < nBands; b++) tmpBand[b] = Math.max(0.0, qRow[b] * invShapeScale);
    const rightTail = tmpBand[nBands - 1] * 0.5;
    let sumInner2 = 0.0;
    for (let i = 0; i < N_BINS; i++) {
      const f = olaFreqs[i];
      const v = f < 210.0 ? 0.0 : interpPiecewiseLinear(bandCenters, tmpBand, f, 0.0, rightTail);
      tmpSpec[i] = v;
      if (i > 0 && i < HALF_OLA) sumInner2 += v * v;
    }
    const irfftRms = Math.sqrt((2.0 / (N_OLA * N_OLA)) * sumInner2 + 1e-12);
    const invCal = 1.0 / irfftRms;
    const baseOff = k * N_BINS;
    for (let i = 0; i < N_BINS; i++) protoCalMag[baseOff + i] = tmpSpec[i] * invCal;
  }

  const winOla = new Float64Array(N_OLA);
  const olaNorm = 1.0 / Math.sqrt((0.375 * N_OLA) / hop);
  for (let n = 0; n < N_OLA; n++) {
    winOla[n] = 0.5 * (1.0 - Math.cos((TWO_PI * n) / N_OLA)) * olaNorm;
  }

  const n24 = Math.ceil(nSamples / 2) + N_OLA + 32;
  const low24Pad = new Float32Array(LEAD_PAD + n24);
  const high24Pad = new Float32Array(LEAD_PAD + n24);

  const fft512 = new ComplexInverseFFT(N_OLA);
  const lowRe = new Float64Array(N_BINS), lowIm = new Float64Array(N_BINS);
  const highRe = new Float64Array(N_BINS), highIm = new Float64Array(N_BINS);
  const rng = new DeterministicRng(404);

  let kickPhase = 0.0;
  let bodyPhase = 0.0;
  const invHop = 1.0 / hop;
  const twoPiOverSr = TWO_PI / sr24;
  const kickBaseHz = model.kickBaseHz;
  const snareBaseHz = model.snareBaseHz;

  for (let t = 0; t < nFrames - 1; t++) {
    const s0 = LEAD_PAD + t * hop;
    if (s0 + hop >= low24Pad.length) break;

    const k0 = dequantizeDrumRms(kickQ[t]) * 1.41421356;
    const k1 = dequantizeDrumRms(kickQ[t + 1]) * 1.41421356;
    if (k0 > 1e-5 || k1 > 1e-5) {
      const boost = Math.max(0.0, (k1 - k0) / Math.max(k1, 0.02));
      let ph = kickPhase;
      for (let i = 0; i < hop; i++) {
        const a = i * invHop;
        ph += twoPiOverSr * (kickBaseHz + 72.0 * boost * (1.0 - a));
        if (ph >= TWO_PI) ph -= TWO_PI;
        low24Pad[s0 + i] += (k0 + (k1 - k0) * a) * Math.sin(ph);
      }
      kickPhase = ph;
    }

    const b0 = dequantizeDrumRms(bodyQ[t]) * 1.41421356;
    const b1 = dequantizeDrumRms(bodyQ[t + 1]) * 1.41421356;
    if (b0 > 1e-5 || b1 > 1e-5) {
      const boost = Math.max(0.0, (b1 - b0) / Math.max(b1, 0.02));
      let ph = bodyPhase;
      for (let i = 0; i < hop; i++) {
        const a = i * invHop;
        ph += twoPiOverSr * (snareBaseHz + 25.0 * boost * (1.0 - a));
        if (ph >= TWO_PI) ph -= TWO_PI;
        low24Pad[s0 + i] += (b0 + (b1 - b0) * a) * Math.sin(ph);
      }
      bodyPhase = ph;
    }

    const highRms = dequantizeDrumRms(highQ[t]);
    if (highRms > 1e-5) {
      const cStartRaw = t * hop + (hop >> 1) - HALF_OLA;
      if (cStartRaw >= 0 && LEAD_PAD + cStartRaw + N_OLA < low24Pad.length) {
        const sIdx = Math.max(0, Math.min(nShapes - 1, shapeIdx[t]));
        const pOff = sIdx * N_BINS;

        lowRe[0] = 0.0; lowIm[0] = 0.0;
        highRe[0] = 0.0; highIm[0] = 0.0;
        const nyqMag = protoCalMag[pOff + HALF_OLA] * highRms;
        lowRe[HALF_OLA] = nyqMag * wLow[HALF_OLA]; lowIm[HALF_OLA] = 0.0;
        highRe[HALF_OLA] = nyqMag * wHigh[HALF_OLA]; highIm[HALF_OLA] = 0.0;

        for (let i = 1; i < HALF_OLA; i++) {
          const mag = protoCalMag[pOff + i] * highRms;
          const ang = rng.nextFloat01() * TWO_PI;
          const cr = mag * Math.cos(ang);
          const ci = mag * Math.sin(ang);
          lowRe[i] = cr * wLow[i];
          lowIm[i] = ci * wLow[i];
          highRe[i] = cr * wHigh[i];
          highIm[i] = ci * wHigh[i];
        }

        fft512.dualRealInverse(lowRe, lowIm, highRe, highIm);
        const reTime = fft512.re, imTime = fft512.im;
        const dstBase = LEAD_PAD + cStartRaw;
        for (let n = 0; n < N_OLA; n++) {
          const w = winOla[n];
          low24Pad[dstBase + n] += reTime[n] * w;
          high24Pad[dstBase + n] += imTime[n] * w;
        }
      }
    }
  }

  const ksMono = new Float32Array(nSamples);
  const hhMono = new Float32Array(nSamples);
  upsample2xPolyphase(low24Pad, ksMono);
  upsample2xPolyphase(high24Pad, hhMono);

  const ksL = new Float32Array(nSamples), ksR = new Float32Array(nSamples);
  const hhL = new Float32Array(nSamples), hhR = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const vKs = ksMono[i];
    const sideKs = i >= 29 ? 0.035 * (vKs - ksMono[i - 29]) : 0.0;
    ksL[i] = vKs + sideKs;
    ksR[i] = vKs - sideKs;

    const vHh = hhMono[i];
    const sideHh = i >= 19 ? 0.11 * (vHh - hhMono[i - 19]) : 0.0;
    hhL[i] = vHh + sideHh;
    hhR[i] = vHh - sideHh;
  }

  return {
    drum_kick_snare: [ksL, ksR],
    drum_hihat_perc: [hhL, hhR],
  };
}
