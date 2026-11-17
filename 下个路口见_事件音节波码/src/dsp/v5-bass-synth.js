import { XorShift32 } from './v5-fft.js';
import { StreamingResampler } from './v5-upsampler.js';

const BASS_RING_SIZE = 2048;
const BASS_RING_MASK = BASS_RING_SIZE - 1;

export class BassWaveSynth {
  constructor(model, streamData, outputSampleRate) {
    this.model = model;
    this.data = streamData;
    this.sr = model.sampleRate;
    this.hop = model.hopSamples;
    this.maxHarm = model.maxHarmonics;
    this.inharmB = model.inharmonicityB;
    this.nFrames = streamData.nFrames;
    this.staticProf = Float32Array.from(model.staticHarmonicLogProfile);

    this.kStretch = new Float32Array(this.maxHarm);
    this.harmPhases = new Float32Array(this.maxHarm);
    for (let k = 0; k < this.maxHarm; k++) {
      const idx = k + 1;
      this.kStretch[k] = idx * Math.sqrt(1 + this.inharmB * idx * idx);
      this.harmPhases[k] = -0.22 * idx;
    }

    this.amps0 = new Float32Array(this.maxHarm);
    this.amps1 = new Float32Array(this.maxHarm);
    this.ring = new Float32Array(BASS_RING_SIZE);
    this.lpState = 0;
    this.prng = new XorShift32(202);
    this.renderedFrame = -1;
    this.clearedUpTo = 0;

    this.resampler = new StreamingResampler(this.sr, outputSampleRate, (idx) => this._sampleInternal(idx));
    this.reset(0);
  }

  _evalFrameAmps(t, outAmps) {
    const d = this.data;
    if (t < 0 || t >= this.nFrames || d.rmsQ[t] === 0 || d.f0Q[t] <= 0) {
      outAmps.fill(0);
      return { f0: 0, noiseAmp: 0 };
    }
    const rms = Math.pow(10, (d.rmsQ[t] / 3.2 - 75) / 20);
    const f0 = 32.7032 * Math.pow(2, (d.f0Q[t] - 1) / 384);
    const nr = d.nrQ[t] / 255;
    const cbOff = d.toneIdx[t] * this.maxHarm;
    const nyq = this.sr * 0.46;

    let hVar = 1e-12;
    for (let k = 0; k < this.maxHarm; k++) {
      if (this.kStretch[k] * f0 >= nyq) {
        outAmps[k] = 0;
      } else {
        const logV = Math.max(-12, Math.min(8, this.staticProf[k] + d.cbTone[cbOff + k]));
        const rel = Math.exp(logV);
        outAmps[k] = rel;
        hVar += 0.5 * rel * rel;
      }
    }
    const scale = Math.sqrt((rms * rms * (1 - nr * nr)) / hVar);
    for (let k = 0; k < this.maxHarm; k++) outAmps[k] *= scale;
    return { f0, noiseAmp: rms * nr };
  }

  _renderFrame(t) {
    if (t < 0 || t >= this.nFrames - 1) return;
    const st0 = this._evalFrameAmps(t, this.amps0);
    const st1 = this._evalFrameAmps(t + 1, this.amps1);

    const s0 = t * this.hop;
    const hop = this.hop;
    const invHop = 1 / hop;
    const fc = st0.f0;
    const fn = st1.f0;

    for (let i = 0; i < hop; i++) this.ring[(s0 + i) & BASS_RING_MASK] = 0;

    if (fc > 30 || fn > 30) {
      const fs = fc > 30 ? fc : fn;
      const fe = fn > 30 ? fn : fc;
      const twoPiOverSr = (2 * Math.PI) / this.sr;

      for (let i = 0; i < hop; i++) {
        const a = i * invHop;
        const dphi0 = twoPiOverSr * (fs + (fe - fs) * a);
        let acc = 0;
        for (let k = 0; k < this.maxHarm; k++) {
          let ph = this.harmPhases[k] + this.kStretch[k] * dphi0;
          if (ph >= 2 * Math.PI) ph -= 2 * Math.PI;
          this.harmPhases[k] = ph;
          const amp = this.amps0[k] + (this.amps1[k] - this.amps0[k]) * a;
          if (amp > 0) acc += amp * Math.cos(ph);
        }
        this.ring[(s0 + i) & BASS_RING_MASK] = acc;
      }
    }

    const nAmp = 0.5 * (st0.noiseAmp + st1.noiseAmp);
    if (nAmp > 1e-5) {
      let lp = this.lpState;
      for (let i = 0; i < hop; i++) {
        lp = 0.65 * lp + 0.35 * this.prng.nextSigned();
        this.ring[(s0 + i) & BASS_RING_MASK] += lp * nAmp * 1.6;
      }
      this.lpState = lp;
    }
  }

  _sampleInternal(sampleIdx) {
    if (sampleIdx < 0) return 0;
    const curFrame = Math.floor(sampleIdx / this.hop);
    if (curFrame >= this.nFrames - 1) return 0;

    if (sampleIdx < this.clearedUpTo || curFrame > this.renderedFrame + 6) {
      this.ring.fill(0);
      this.renderedFrame = Math.max(-1, curFrame - 1);
      this.clearedUpTo = Math.max(0, curFrame * this.hop);
    }
    this.clearedUpTo = sampleIdx;

    while (this.renderedFrame <= curFrame) {
      this.renderedFrame++;
      this._renderFrame(this.renderedFrame);
    }
    return this.ring[sampleIdx & BASS_RING_MASK];
  }

  reset(targetSampleIndex = 0) {
    this.ring.fill(0);
    const srcIdx = Math.floor((targetSampleIndex * this.sr) / this.resampler.targetRate);
    const curFrame = Math.floor(srcIdx / this.hop);
    this.renderedFrame = Math.max(-1, curFrame - 2);
    this.clearedUpTo = Math.max(0, srcIdx - 64);
    this.resampler.reset(targetSampleIndex);
  }

  valueAt(outputSampleIndex) {
    return this.resampler.sampleAt(outputSampleIndex);
  }
}
