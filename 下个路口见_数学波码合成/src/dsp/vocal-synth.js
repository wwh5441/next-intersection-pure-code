import { Radix2FFT, XorShift32, interpPiecewise } from './fft.js';
import { StreamingResampler } from './upsampler.js';

const RING_SIZE = 4096;
const RING_MASK = RING_SIZE - 1;
const N_OLA = 512;

export class VocalWaveSynth {
  constructor(model, streamData, outputSampleRate) {
    this.model = model;
    this.data = streamData;
    this.sr = model.sampleRate;
    this.hop = model.hopSamples;
    this.maxHarm = model.maxHarmonics;
    this.nFrames = streamData.nFrames;
    this.nMels = streamData.nMels;
    this.rank = streamData.rank;

    this.melCenters = Float32Array.from(model.melCentersHz);
    this.apCenters = Float32Array.from(model.apCentersHz);
    this.tStatic = Float32Array.from(model.staticTimbreLogMel);
    this.tPitch = Float32Array.from(model.pitchSlopeLogMel);
    this.tEffort = Float32Array.from(model.effortSlopeLogMel);
    this.meanVg = model.meanVoicedGainLog;

    this.fft64 = new Radix2FFT(64);
    this.re64 = new Float32Array(64);
    this.im64 = new Float32Array(64);
    this.grid33 = new Float32Array(33);
    for (let i = 0; i < 33; i++) this.grid33[i] = (i * (this.sr * 0.5)) / 32;

    this.fft512 = new Radix2FFT(N_OLA);
    this.re512 = new Float32Array(N_OLA);
    this.im512 = new Float32Array(N_OLA);
    this.winOla = new Float32Array(N_OLA);
    for (let i = 0; i < N_OLA; i++) {
      this.winOla[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N_OLA));
    }
    this.olaFreqs = new Float32Array((N_OLA >> 1) + 1);
    for (let i = 0; i <= N_OLA >> 1; i++) {
      this.olaFreqs[i] = (i * this.sr) / N_OLA;
    }
    this.olaNoiseNorm = 1 / Math.sqrt((0.375 * N_OLA) / this.hop);

    // Scratch frame buffers (zero GC in AudioWorklet)
    this.env0 = new Float32Array(this.nMels);
    this.env1 = new Float32Array(this.nMels);
    this.hCal0 = new Float32Array(this.nMels);
    this.hCal1 = new Float32Array(this.nMels);
    this.ap0 = new Float32Array(8);
    this.ap1 = new Float32Array(8);
    this.nSpec0 = new Float32Array((N_OLA >> 1) + 1);
    this.minP0 = new Float32Array(33);
    this.minP1 = new Float32Array(33);
    this.u0 = new Float32Array(this.maxHarm);
    this.v0 = new Float32Array(this.maxHarm);
    this.du = new Float32Array(this.maxHarm);
    this.dv = new Float32Array(this.maxHarm);
    this.zScratch = new Float32Array(this.rank);

    this.ring = new Float32Array(RING_SIZE);
    this.phase0 = 0;
    this.prng = new XorShift32(101);
    this.renderedFrame = -1;
    this.clearedUpTo = 0;
    this.cachedFrameIdx = -1;

    this.resampler = new StreamingResampler(this.sr, outputSampleRate, (idx) => this._sampleInternal(idx));
    this.reset(0);
  }

  _minPhaseFromEnv(envMel, outMp33) {
    const re = this.re64;
    const im = this.im64;
    im.fill(0);
    for (let i = 0; i < 33; i++) {
      const e = interpPiecewise(this.melCenters, envMel, this.grid33[i]) + 1e-6;
      re[i] = Math.log(Math.max(e, 1e-6));
    }
    for (let i = 1; i < 32; i++) {
      re[64 - i] = re[i];
    }
    this.fft64.transform(re, im, true);
    im.fill(0);
    for (let i = 1; i < 32; i++) re[i] *= 2;
    for (let i = 33; i < 64; i++) re[i] = 0;
    this.fft64.transform(re, im, false);
    for (let i = 0; i < 33; i++) outMp33[i] = -im[i];
  }

  _evalFrameState(t, outEnv, outHCal, outAp, outNSpec, computeNoiseSpec) {
    const d = this.data;
    if (t < 0 || t >= this.nFrames || d.rmsQ[t] === 0) {
      outEnv.fill(0);
      outHCal.fill(0);
      outAp.fill(1);
      if (computeNoiseSpec) outNSpec.fill(0);
      return { active: false, f0: 0 };
    }

    const rmsTarget = Math.pow(10, (d.rmsQ[t] / 3.2 - 75) / 20);
    const f0q = d.f0Q[t];
    const f0 = f0q > 0 ? 110 * Math.pow(2, (f0q - 1) / 288) : 0;
    const gainLog = d.gainQ[t] / 14 - 14;
    const f0Oct = f0 > 80 ? Math.log2(f0 / 261.63) : 0;
    const eff = gainLog - this.meanVg;

    // Reconstruct pronunciation vector Z from RVQ stages
    const rank = this.rank;
    const z = this.zScratch;
    z.fill(0);
    for (let s = 0; s < d.nStages; s++) {
      const code = d.pronIndices[s][t] * rank;
      const cb = d.codebooks[s];
      for (let r = 0; r < rank; r++) z[r] += cb[code + r];
    }

    // Fuse 3D Speaker Timbre Field + Pronunciation Basis U_pron * Z
    const uPron = d.uPron;
    for (let m = 0; m < this.nMels; m++) {
      let pVal = 0;
      const uRow = m * rank;
      for (let r = 0; r < rank; r++) pVal += uPron[uRow + r] * z[r];
      const logE = this.tStatic[m] + this.tPitch[m] * f0Oct + this.tEffort[m] * eff + pVal;
      outEnv[m] = Math.exp(Math.max(-16, Math.min(12, logE)));
    }

    const apBase = d.apIdx[t] * 8;
    for (let b = 0; b < 8; b++) outAp[b] = d.cbAp[apBase + b];

    // Parseval exact energy calibration between voiced harmonics & turbulent consonant/breath noise
    const nBins = this.olaFreqs.length;
    let sumEf2 = 1e-12;
    let sumEfAp2 = 0;
    let sumInnerN2 = 0;

    for (let i = 0; i < nBins; i++) {
      const f = this.olaFreqs[i];
      const ef = interpPiecewise(this.melCenters, outEnv, f);
      const ap = f0 > 80 ? interpPiecewise(this.apCenters, outAp, f) : 1;
      sumEf2 += ef * ef;
      const rawN = ef * ap;
      sumEfAp2 += rawN * rawN;
      if (computeNoiseSpec) outNSpec[i] = rawN;
      if (i > 0 && i < nBins - 1) sumInnerN2 += rawN * rawN;
    }

    const tVar = rmsTarget * rmsTarget;
    if (f0 > 80) {
      const K = Math.min(this.maxHarm, Math.floor((this.sr * 0.47) / f0));
      let harmVar = 1e-12;
      for (let k = 1; k <= K; k++) {
        const hf = k * f0;
        const ak = interpPiecewise(this.melCenters, outEnv, hf);
        const apk = interpPiecewise(this.apCenters, outAp, hf);
        const hk2 = ak * ak * Math.max(0, 1 - apk * apk);
        harmVar += 0.5 * hk2;
      }
      const apFrac = sumEfAp2 / sumEf2;
      const hScale = Math.sqrt((tVar * Math.max(0, 1 - apFrac)) / harmVar);
      for (let m = 0; m < this.nMels; m++) outHCal[m] = outEnv[m] * hScale;

      if (computeNoiseSpec) {
        const irfftRms = Math.sqrt((2 / (N_OLA * N_OLA)) * sumInnerN2 + 1e-12);
        const nScale = Math.sqrt(tVar * apFrac) / irfftRms;
        for (let i = 0; i < nBins; i++) outNSpec[i] *= nScale;
      }
    } else {
      outHCal.fill(0);
      if (computeNoiseSpec) {
        const irfftRms = Math.sqrt((2 / (N_OLA * N_OLA)) * sumInnerN2 + 1e-12);
        const nScale = rmsTarget / irfftRms;
        for (let i = 0; i < nBins; i++) outNSpec[i] *= nScale;
      }
    }
    return { active: true, f0 };
  }

  _renderFrame(t) {
    if (t < 0 || t >= this.nFrames - 1) return;
    const st0 = this._evalFrameState(t, this.env0, this.hCal0, this.ap0, this.nSpec0, true);
    const st1 = this._evalFrameState(t + 1, this.env1, this.hCal1, this.ap1, null, false);

    const s0 = t * this.hop;
    const hop = this.hop;
    const invHop = 1 / hop;
    const fc = st0.f0;
    const fn = st1.f0;

    // 1. Voiced Harmonic Wave with Cepstral Minimum Phase & Trigonometric Angle-Addition Recurrence
    if (fc > 80 || fn > 80) {
      const fStart = fc > 80 ? fc : fn;
      const fEnd = fn > 80 ? fn : fc;
      const vw0 = fc > 80 ? 1 : 0;
      const vw1 = fn > 80 ? 1 : 0;

      this._minPhaseFromEnv(this.env0, this.minP0);
      this._minPhaseFromEnv(this.env1, this.minP1);

      const K = Math.min(this.maxHarm, Math.floor((this.sr * 0.47) / Math.max(90, 0.5 * (fStart + fEnd))));
      for (let k = 1; k <= K; k++) {
        const idx = k - 1;
        const hf0 = k * fStart;
        const hf1 = k * fEnd;
        const a0 = interpPiecewise(this.melCenters, this.hCal0, hf0) * vw0;
        const a1 = interpPiecewise(this.melCenters, this.hCal1, hf1) * vw1;
        const apv0 = interpPiecewise(this.apCenters, this.ap0, hf0);
        const apv1 = interpPiecewise(this.apCenters, this.ap1, hf1);
        const hg0 = a0 * Math.sqrt(Math.max(0, 1 - apv0 * apv0));
        const hg1 = a1 * Math.sqrt(Math.max(0, 1 - apv1 * apv1));

        const mp0 = interpPiecewise(this.grid33, this.minP0, hf0);
        const mp1 = interpPiecewise(this.grid33, this.minP1, hf1);

        const uStart = hg0 * Math.cos(mp0);
        const vStart = hg0 * Math.sin(mp0);
        const uEnd = hg1 * Math.cos(mp1);
        const vEnd = hg1 * Math.sin(mp1);

        this.u0[idx] = uStart;
        this.v0[idx] = vStart;
        this.du[idx] = (uEnd - uStart) * invHop;
        this.dv[idx] = (vEnd - vStart) * invHop;
      }

      const twoPiOverSr = (2 * Math.PI) / this.sr;
      const df = (fEnd - fStart) * invHop;
      let phi = this.phase0;

      for (let i = 0; i < hop; i++) {
        const fInst = fStart + df * i;
        phi += twoPiOverSr * fInst;
        if (phi >= 2 * Math.PI) phi -= 2 * Math.PI;

        const c1 = Math.cos(phi);
        const s1 = Math.sin(phi);
        let ck = c1;
        let sk = s1;
        let sum = 0;

        for (let k = 0; k < K; k++) {
          const uk = this.u0[k] + this.du[k] * i;
          const vk = this.v0[k] + this.dv[k] * i;
          sum += uk * ck - vk * sk;
          const nck = ck * c1 - sk * s1;
          sk = sk * c1 + ck * s1;
          ck = nck;
        }
        this.ring[(s0 + i) & RING_MASK] += sum;
      }
      this.phase0 = phi;
    }

    // 2. Turbulent Mandarin Consonant & Breath Wave Grain (512-pt Real IFFT)
    if (st0.active) {
      const cStart = s0 + (hop >> 1) - (N_OLA >> 1);
      if (cStart >= 0) {
        const re = this.re512;
        const im = this.im512;
        re[0] = this.nSpec0[0];
        im[0] = 0;
        re[N_OLA >> 1] = this.nSpec0[N_OLA >> 1];
        im[N_OLA >> 1] = 0;
        for (let b = 1; b < N_OLA >> 1; b++) {
          const mag = this.nSpec0[b];
          const ang = this.prng.nextFloat() * 2 * Math.PI;
          const cr = mag * Math.cos(ang);
          const ci = mag * Math.sin(ang);
          re[b] = cr;
          im[b] = ci;
          re[N_OLA - b] = cr;
          im[N_OLA - b] = -ci;
        }
        this.fft512.transform(re, im, true);
        const scale = this.olaNoiseNorm;
        for (let n = 0; n < N_OLA; n++) {
          this.ring[(cStart + n) & RING_MASK] += re[n] * this.winOla[n] * scale;
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
      this.clearedUpTo = Math.max(0, (this.renderedFrame + 1) * this.hop - N_OLA);
    }

    while (this.clearedUpTo < sampleIdx - N_OLA) {
      this.ring[this.clearedUpTo & RING_MASK] = 0;
      this.clearedUpTo++;
    }

    const neededFrame = Math.min(this.nFrames - 2, curFrame + 2);
    while (this.renderedFrame < neededFrame) {
      this.renderedFrame++;
      this._renderFrame(this.renderedFrame);
    }

    return this.ring[sampleIdx & RING_MASK];
  }

  reset(targetSampleIndex = 0) {
    this.ring.fill(0);
    const srcIdx = Math.floor((targetSampleIndex * this.sr) / this.resampler.targetRate);
    const curFrame = Math.floor(srcIdx / this.hop);
    this.renderedFrame = Math.max(-1, curFrame - 2);
    this.clearedUpTo = Math.max(0, srcIdx - N_OLA);
    this.resampler.reset(targetSampleIndex);
  }

  valueAt(outputSampleIndex) {
    return this.resampler.sampleAt(outputSampleIndex);
  }
}
