import { Radix2FFT, XorShift32, interpPiecewise } from './v5-fft.js';
import { StreamingResampler } from './v5-upsampler.js';

const DRUM_RING_SIZE = 4096;
const DRUM_RING_MASK = DRUM_RING_SIZE - 1;
const DRUM_N_OLA = 512;

export class DrumWaveSynth {
  constructor(model, streamData, outputSampleRate) {
    this.model = model;
    this.data = streamData;
    this.sr = model.sampleRate;
    this.hop = model.hopSamples;
    this.nFrames = streamData.nFrames;
    this.nComp = streamData.nComp;
    this.nBands = streamData.nBands;
    this.kickHz = streamData.kickHz;
    this.snareHz = streamData.snareHz;

    this.fft512 = new Radix2FFT(DRUM_N_OLA);
    this.re512 = new Float32Array(DRUM_N_OLA);
    this.im512 = new Float32Array(DRUM_N_OLA);
    this.winOla = new Float32Array(DRUM_N_OLA);
    for (let i = 0; i < DRUM_N_OLA; i++) {
      this.winOla[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / DRUM_N_OLA));
    }
    this.olaFreqs = new Float32Array((DRUM_N_OLA >> 1) + 1);
    for (let i = 0; i <= DRUM_N_OLA >> 1; i++) this.olaFreqs[i] = (i * this.sr) / DRUM_N_OLA;
    this.olaNoiseNorm = 1 / Math.sqrt((0.375 * DRUM_N_OLA) / this.hop);

    this.melFrame = new Float32Array(this.nBands);
    this.specMag = new Float32Array((DRUM_N_OLA >> 1) + 1);
    this.ring = new Float32Array(DRUM_RING_SIZE);
    this.kickPhase = 0;
    this.bodyPhase = 0;
    this.prng = new XorShift32(404);
    this.renderedFrame = -1;
    this.clearedUpTo = 0;

    this.resampler = new StreamingResampler(this.sr, outputSampleRate, (idx) => this._sampleInternal(idx));
    this.reset(0);
  }

  _deqRms(q) {
    return q > 0 ? Math.pow(10, (q / 3.2 - 75) / 20) : 0;
  }

  _renderFrame(t) {
    if (t < 0 || t >= this.nFrames - 1) return;
    const d = this.data;
    const s0 = t * this.hop;
    const hop = this.hop;
    const invHop = 1 / hop;
    const twoPiOverSr = (2 * Math.PI) / this.sr;

    // 1. Swept-Sine Sub/Kick Modal Wave
    const k0 = this._deqRms(d.kickQ[t]) * 1.4142;
    const k1 = this._deqRms(d.kickQ[t + 1]) * 1.4142;
    if (k0 > 1e-5 || k1 > 1e-5) {
      const boost = Math.max(0, (k1 - k0) / Math.max(k1, 0.02));
      let ph = this.kickPhase;
      for (let i = 0; i < hop; i++) {
        const a = i * invHop;
        const fk = this.kickHz + 72 * boost * (1 - a);
        ph += twoPiOverSr * fk;
        if (ph >= 2 * Math.PI) ph -= 2 * Math.PI;
        this.ring[(s0 + i) & DRUM_RING_MASK] += (k0 + (k1 - k0) * a) * Math.sin(ph);
      }
      this.kickPhase = ph;
    }

    // 2. Snare / Woodblock Tonal Body Modal Wave
    const b0 = this._deqRms(d.bodyQ[t]) * 1.4142;
    const b1 = this._deqRms(d.bodyQ[t + 1]) * 1.4142;
    if (b0 > 1e-5 || b1 > 1e-5) {
      const boost = Math.max(0, (b1 - b0) / Math.max(b1, 0.02));
      let ph = this.bodyPhase;
      for (let i = 0; i < hop; i++) {
        const a = i * invHop;
        const fb = this.snareHz + 25 * boost * (1 - a);
        ph += twoPiOverSr * fb;
        if (ph >= 2 * Math.PI) ph -= 2 * Math.PI;
        this.ring[(s0 + i) & DRUM_RING_MASK] += (b0 + (b1 - b0) * a) * Math.sin(ph);
      }
      this.bodyPhase = ph;
    }

    // 3. Crisp 512-pt Percussive Transient Wave (> 210 Hz)
    const highRms = this._deqRms(d.highQ[t]);
    if (highRms > 1e-5) {
      const cStart = s0 + (hop >> 1) - (DRUM_N_OLA >> 1);
      if (cStart >= 0) {
        const mf = this.melFrame;
        const wDrum = d.wDrum;
        const hRec = d.hRec;
        for (let b = 0; b < this.nBands; b++) {
          let sum = 0;
          const wRow = b * this.nComp;
          for (let c = 0; c < this.nComp; c++) {
            sum += wDrum[wRow + c] * hRec[c * this.nFrames + t];
          }
          mf[b] = sum;
        }

        const nBins = this.olaFreqs.length;
        let sumInner2 = 0;
        const rightTail = mf[this.nBands - 1] * 0.5;
        for (let i = 0; i < nBins; i++) {
          const f = this.olaFreqs[i];
          const v = f < 210 ? 0 : interpPiecewise(d.bandCenters, mf, f, 0, rightTail);
          this.specMag[i] = v;
          if (i > 0 && i < nBins - 1) sumInner2 += v * v;
        }

        const irfftRms = Math.sqrt((2 / (DRUM_N_OLA * DRUM_N_OLA)) * sumInner2 + 1e-12);
        const calScale = highRms / irfftRms;

        const re = this.re512;
        const im = this.im512;
        re[0] = 0;
        im[0] = 0;
        re[DRUM_N_OLA >> 1] = this.specMag[DRUM_N_OLA >> 1] * calScale;
        im[DRUM_N_OLA >> 1] = 0;
        for (let i = 1; i < DRUM_N_OLA >> 1; i++) {
          const mag = this.specMag[i] * calScale;
          const ang = this.prng.nextFloat() * 2 * Math.PI;
          const cr = mag * Math.cos(ang);
          const ci = mag * Math.sin(ang);
          re[i] = cr;
          im[i] = ci;
          re[DRUM_N_OLA - i] = cr;
          im[DRUM_N_OLA - i] = -ci;
        }
        this.fft512.transform(re, im, true);
        const norm = this.olaNoiseNorm;
        for (let n = 0; n < DRUM_N_OLA; n++) {
          this.ring[(cStart + n) & DRUM_RING_MASK] += re[n] * this.winOla[n] * norm;
        }
      }
    }
  }

  _sampleInternal(sampleIdx) {
    if (sampleIdx < 0) return 0;
    const curFrame = Math.floor(sampleIdx / this.hop);
    if (curFrame >= this.nFrames - 1) return 0;

    if (sampleIdx < this.clearedUpTo || curFrame > this.renderedFrame + 8) {
      this.ring.fill(0);
      this.renderedFrame = Math.max(-1, curFrame - 2);
      this.clearedUpTo = Math.max(0, (this.renderedFrame + 1) * this.hop - DRUM_N_OLA);
    }

    while (this.clearedUpTo < sampleIdx - DRUM_N_OLA) {
      this.ring[this.clearedUpTo & DRUM_RING_MASK] = 0;
      this.clearedUpTo++;
    }

    const neededFrame = Math.min(this.nFrames - 2, curFrame + 2);
    while (this.renderedFrame < neededFrame) {
      this.renderedFrame++;
      this._renderFrame(this.renderedFrame);
    }

    return this.ring[sampleIdx & DRUM_RING_MASK];
  }

  reset(targetSampleIndex = 0) {
    this.ring.fill(0);
    const srcIdx = Math.floor((targetSampleIndex * this.sr) / this.resampler.targetRate);
    const curFrame = Math.floor(srcIdx / this.hop);
    this.renderedFrame = Math.max(-1, curFrame - 2);
    this.clearedUpTo = Math.max(0, srcIdx - DRUM_N_OLA);
    this.resampler.reset(targetSampleIndex);
  }

  valueAt(outputSampleIndex) {
    return this.resampler.sampleAt(outputSampleIndex);
  }
}
