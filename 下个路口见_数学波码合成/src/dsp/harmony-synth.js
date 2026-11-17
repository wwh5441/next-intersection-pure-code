import { Radix2FFT, XorShift32, interpPiecewise } from './fft.js';
import { StereoFdnReverb } from './fdn-reverb.js';
import { StreamingResampler } from './upsampler.js';

const HARM_RING_SIZE = 4096;
const HARM_RING_MASK = HARM_RING_SIZE - 1;
const HARM_N_OLA = 512;

export class HarmonyWaveSynth {
  constructor(model, streamData, outputSampleRate) {
    this.model = model;
    this.data = streamData;
    this.sr = model.sampleRate;
    this.hop = model.hopSamples;
    this.hopS = this.hop >> 1;
    this.nFrames = streamData.nFrames;
    this.nNotes = streamData.nNotes;
    this.maxHarm = model.maxHarmonics;
    this.qLevels = streamData.qLevels;
    this.maxNoteAmp = streamData.maxNoteAmp;
    this.outputSampleRate = outputSampleRate;

    this.warmW = Float32Array.from(model.warmHarmonicWeights);
    this.brightW = Float32Array.from(model.brightHarmonicWeights);
    this.strumCenters = Float32Array.from(model.strumCentersHz);

    // Precompute exact coupled-rotor step constants for all 56 MIDI notes (zero Math.sin per sample!)
    this.f0s = new Float32Array(this.nNotes);
    this.cosStep = new Float32Array(this.nNotes);
    this.sinStep = new Float32Array(this.nNotes);
    this.noteHarms = new Uint8Array(this.nNotes);
    this.rotorRe = new Float32Array(this.nNotes);
    this.rotorIm = new Float32Array(this.nNotes);

    const initRng = new XorShift32(505);
    for (let n = 0; n < this.nNotes; n++) {
      const midi = streamData.midiMin + n;
      const f0 = 440 * Math.pow(2, (midi - 69) / 12);
      this.f0s[n] = f0;
      const omega = (2 * Math.PI * f0) / this.sr;
      this.cosStep[n] = Math.cos(omega);
      this.sinStep[n] = Math.sin(omega);
      this.noteHarms[n] = Math.min(this.maxHarm, Math.floor(4350 / f0));
      const p0 = initRng.nextFloat() * 2 * Math.PI;
      this.rotorRe[n] = Math.cos(p0);
      this.rotorIm[n] = Math.sin(p0);
    }

    this.fft512 = new Radix2FFT(HARM_N_OLA);
    this.re512 = new Float32Array(HARM_N_OLA);
    this.im512 = new Float32Array(HARM_N_OLA);
    this.winOla = new Float32Array(HARM_N_OLA);
    for (let i = 0; i < HARM_N_OLA; i++) this.winOla[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / HARM_N_OLA));
    this.olaFreqs = new Float32Array((HARM_N_OLA >> 1) + 1);
    for (let i = 0; i <= HARM_N_OLA >> 1; i++) this.olaFreqs[i] = (i * this.sr) / HARM_N_OLA;
    this.olaNorm = 1 / Math.sqrt((0.375 * HARM_N_OLA) / this.hopS);

    this.w0 = new Float32Array(this.maxHarm);
    this.w1 = new Float32Array(this.maxHarm);
    this.a0 = new Float32Array(this.nNotes);
    this.a1 = new Float32Array(this.nNotes);
    this.strumBands = new Float32Array(4);
    this.specS = new Float32Array((HARM_N_OLA >> 1) + 1);

    this.ring = new Float32Array(HARM_RING_SIZE);
    this.prng = new XorShift32(606);
    this.renderedFrame = -1;
    this.clearedUpTo = 0;

    this.resampler = new StreamingResampler(this.sr, outputSampleRate, (idx) => this._sampleInternal(idx));
    this.fdn = new StereoFdnReverb(model.stereoAcoustics, outputSampleRate);
    this.stereoOut = new Float32Array(2);
    this.framesPerOutSample = this.sr / (outputSampleRate * this.hop);
    this.reset(0);
  }

  _evalChordFrame(t, outA, outW) {
    const d = this.data;
    if (t < 0 || t >= this.nFrames || d.harmQ[t] === 0) {
      outA.fill(0);
      outW.fill(0);
      return;
    }
    const harmRms = Math.pow(10, (d.harmQ[t] / 3.2 - 75) / 20);
    const br = d.brightQ[t] / 255;
    for (let k = 0; k < this.maxHarm; k++) {
      outW[k] = this.warmW[k] * (1 - br) + this.brightW[k] * br;
    }

    const invQL = 1 / this.qLevels;
    let varSum = 0;
    for (let n = 0; n < this.nNotes; n++) {
      const q = d.noteAmpQ[n * this.nFrames + t];
      if (q === 0) {
        outA[n] = 0;
        continue;
      }
      const norm = q * invQL;
      const amp = norm * norm * this.maxNoteAmp;
      outA[n] = amp;
      const K = this.noteHarms[n];
      for (let k = 0; k < K; k++) {
        const hk = amp * outW[k];
        varSum += 0.5 * hk * hk;
      }
    }

    if (varSum > 1e-12) {
      const scale = Math.sqrt((harmRms * harmRms) / varSum);
      for (let n = 0; n < this.nNotes; n++) outA[n] *= scale;
    }
  }

  _renderStrumSubframe(ts) {
    const d = this.data;
    const nSFrames = this.nFrames * 2;
    if (ts < 0 || ts >= nSFrames) return;
    let totVar = 0;
    for (let b = 0; b < 4; b++) {
      const q = d.strumQ[b * nSFrames + ts];
      const rms = q > 0 ? Math.pow(10, (q / 3.2 - 78) / 20) : 0;
      this.strumBands[b] = rms;
      totVar += rms * rms;
    }
    if (totVar <= 1e-10) return;

    const cStart = ts * this.hopS + (this.hopS >> 1) - (HARM_N_OLA >> 1);
    if (cStart < 0) return;

    const nBins = this.olaFreqs.length;
    let sumInner2 = 0;
    const rightTail = this.strumBands[3] * 0.4;
    for (let i = 0; i < nBins; i++) {
      const f = this.olaFreqs[i];
      const v = f < 1950 ? 0 : interpPiecewise(this.strumCenters, this.strumBands, f, 0, rightTail);
      this.specS[i] = v;
      if (i > 0 && i < nBins - 1) sumInner2 += v * v;
    }

    const irfftRms = Math.sqrt((2 / (HARM_N_OLA * HARM_N_OLA)) * sumInner2 + 1e-12);
    const cal = Math.sqrt(totVar) / irfftRms;
    const re = this.re512;
    const im = this.im512;
    re[0] = 0;
    im[0] = 0;
    re[HARM_N_OLA >> 1] = this.specS[HARM_N_OLA >> 1] * cal;
    im[HARM_N_OLA >> 1] = 0;
    for (let i = 1; i < HARM_N_OLA >> 1; i++) {
      const mag = this.specS[i] * cal;
      const ang = this.prng.nextFloat() * 2 * Math.PI;
      const cr = mag * Math.cos(ang);
      const ci = mag * Math.sin(ang);
      re[i] = cr;
      im[i] = ci;
      re[HARM_N_OLA - i] = cr;
      im[HARM_N_OLA - i] = -ci;
    }
    this.fft512.transform(re, im, true);
    const norm = this.olaNorm;
    for (let n = 0; n < HARM_N_OLA; n++) {
      this.ring[(cStart + n) & HARM_RING_MASK] += re[n] * this.winOla[n] * norm;
    }
  }

  _renderFrame(t) {
    if (t < 0 || t >= this.nFrames - 1) return;
    this._evalChordFrame(t, this.a0, this.w0);
    this._evalChordFrame(t + 1, this.a1, this.w1);

    const s0 = t * this.hop;
    const hop = this.hop;
    const invHop = 1 / hop;

    // 1. Polyphonic Coupled-Rotor Harmonic Chord Oscillators (zero Math.sin per sample!)
    for (let n = 0; n < this.nNotes; n++) {
      const aStart = this.a0[n];
      const aEnd = this.a1[n];
      const K = this.noteHarms[n];
      if ((aStart <= 1e-6 && aEnd <= 1e-6) || K === 0) continue;

      let r1 = this.rotorRe[n];
      let i1 = this.rotorIm[n];
      const cs = this.cosStep[n];
      const ss = this.sinStep[n];
      const da = (aEnd - aStart) * invHop;

      for (let s = 0; s < hop; s++) {
        const a = s * invHop;
        const noteAmp = aStart + da * s;
        let rk = r1;
        let ik = i1;
        let sampleSum = 0;

        for (let k = 0; k < K; k++) {
          const wk = this.w0[k] + (this.w1[k] - this.w0[k]) * a;
          sampleSum += noteAmp * wk * ik;
          const nrk = rk * r1 - ik * i1;
          ik = ik * r1 + rk * i1;
          rk = nrk;
        }
        this.ring[(s0 + s) & HARM_RING_MASK] += sampleSum;

        const nextR1 = r1 * cs - i1 * ss;
        i1 = i1 * cs + r1 * ss;
        r1 = nextR1;
      }

      // Periodically re-normalize rotor to prevent float drift over minutes
      const magInv = 1 / Math.hypot(r1, i1);
      this.rotorRe[n] = r1 * magInv;
      this.rotorIm[n] = i1 * magInv;
    }

    // 2. Render both 2x-rate Acoustic Guitar Strum Subframes
    this._renderStrumSubframe(t * 2);
    this._renderStrumSubframe(t * 2 + 1);
  }

  _sampleInternal(sampleIdx) {
    if (sampleIdx < 0) return 0;
    const curFrame = Math.floor(sampleIdx / this.hop);
    if (curFrame >= this.nFrames - 1) return 0;

    if (sampleIdx < this.clearedUpTo || curFrame > this.renderedFrame + 6) {
      this.ring.fill(0);
      this.renderedFrame = Math.max(-1, curFrame - 2);
      this.clearedUpTo = Math.max(0, (this.renderedFrame + 1) * this.hop - HARM_N_OLA);
    }

    while (this.clearedUpTo < sampleIdx - HARM_N_OLA) {
      this.ring[this.clearedUpTo & HARM_RING_MASK] = 0;
      this.clearedUpTo++;
    }

    const neededFrame = Math.min(this.nFrames - 2, curFrame + 2);
    while (this.renderedFrame < neededFrame) {
      this.renderedFrame++;
      this._renderFrame(this.renderedFrame);
    }

    return this.ring[sampleIdx & HARM_RING_MASK];
  }

  reset(targetSampleIndex = 0) {
    this.ring.fill(0);
    const srcIdx = Math.floor((targetSampleIndex * this.sr) / this.outputSampleRate);
    const curFrame = Math.floor(srcIdx / this.hop);
    this.renderedFrame = Math.max(-1, curFrame - 2);
    this.clearedUpTo = Math.max(0, srcIdx - HARM_N_OLA);
    this.resampler.reset(targetSampleIndex);
    this.fdn.reset();
  }

  stereoAt(outputSampleIndex) {
    const mono = this.resampler.sampleAt(outputSampleIndex);
    const fPos = outputSampleIndex * this.framesPerOutSample;
    const f0 = Math.min(this.nFrames - 1, Math.max(0, fPos | 0));
    const f1 = Math.min(this.nFrames - 1, f0 + 1);
    const frac = fPos - f0;
    const sq = this.data.sideQ[f0] + (this.data.sideQ[f1] - this.data.sideQ[f0]) * frac;
    const sideRatio = (sq / 255) * 1.4;
    this.fdn.processSample(mono, sideRatio, this.stereoOut);
    return this.stereoOut;
  }
}
