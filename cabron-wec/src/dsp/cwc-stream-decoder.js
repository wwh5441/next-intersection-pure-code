// CabronWEC (CWC-4X) Zero-Allocation Streaming Frame Decoder
// Decodes individual 512-sample (11.61ms) overlap-add stereo frames on demand
// with deterministic per-frame PNS seeding (seek-invariant) and live 5-layer stem mixing.

import {
  B62_INV,
  CWC_HOP_M,
  CWC_FRAME_N,
  GAIN_LUT,
  MID_BANDS,
  PNS_LUT,
  PNS_MID_BANDS,
  PNS_SIDE_BANDS,
  SIDE_BANDS,
  createMDCTProcessor,
  unpackQuantizedVec,
} from './mdct-core.js';

export class CWCStreamDecoder {
  constructor(totalFrames = 0) {
    this.totalFrames = totalFrames;
    this.frameStrings = new Array(totalFrames);
    this.loadedCount = 0;

    this.mdct = createMDCTProcessor(CWC_FRAME_N);
    this.decMidV = new Int32Array(240);
    this.decSideV = new Int32Array(168);
    this.specMid = new Float64Array(512);
    this.specSide = new Float64Array(512);
    this.outMidN = new Float64Array(CWC_FRAME_N);
    this.outSideN = new Float64Array(CWC_FRAME_N);

    // Preallocated overlap-add tails (second half of previous frame)
    this.olaL = new Float32Array(CWC_HOP_M);
    this.olaR = new Float32Array(CWC_HOP_M);
    this.noiseScratch = new Float64Array(160);

    // 5-layer stem gain/mute/solo state -> effective gains [0..1+]
    this.stemGains = new Float32Array([1, 1, 1, 1, 1]);
    this.stemMutes = new Uint8Array(5);
    this.stemSolos = new Uint8Array(5);
    this.effectiveGains = new Float32Array([1, 1, 1, 1, 1]);
    this.masterGain = 0.98;

    // Telemetry outputs updated on every decoded frame
    this.layerRms = new Float32Array(5);
    this.spectrum32 = new Float32Array(32);
    this.currentFrame = 0;
  }

  setTotalFrames(totalFrames) {
    if (this.frameStrings.length !== totalFrames) {
      const next = new Array(totalFrames);
      const copyLen = Math.min(totalFrames, this.frameStrings.length);
      for (let i = 0; i < copyLen; i++) next[i] = this.frameStrings[i];
      this.frameStrings = next;
    }
    this.totalFrames = totalFrames;
  }

  ingestChunk(startFrame, framesArray) {
    let newlyLoaded = 0;
    for (let i = 0; i < framesArray.length; i++) {
      const idx = startFrame + i;
      if (idx < this.totalFrames) {
        if (this.frameStrings[idx] === undefined) newlyLoaded++;
        this.frameStrings[idx] = framesArray[i];
      }
    }
    this.loadedCount += newlyLoaded;
  }

  hasFrame(frameIdx) {
    if (frameIdx < 0 || frameIdx >= this.totalFrames) return true;
    return this.frameStrings[frameIdx] !== undefined;
  }

  setStemControls({ gains, mutes, solos, masterGain }) {
    if (gains) for (let i = 0; i < 5; i++) this.stemGains[i] = gains[i];
    if (mutes) for (let i = 0; i < 5; i++) this.stemMutes[i] = mutes[i] ? 1 : 0;
    if (solos) for (let i = 0; i < 5; i++) this.stemSolos[i] = solos[i] ? 1 : 0;
    if (typeof masterGain === 'number') this.masterGain = masterGain;

    let anySolo = false;
    for (let i = 0; i < 5; i++) {
      if (this.stemSolos[i]) {
        anySolo = true;
        break;
      }
    }
    for (let i = 0; i < 5; i++) {
      const active = anySolo ? this.stemSolos[i] === 1 : this.stemMutes[i] === 0;
      this.effectiveGains[i] = active ? this.stemGains[i] : 0;
    }
  }

  /**
   * Clickless seek: resets OLA buffer and runs 2 silent pre-roll frames before targetFrame
   * so TDAC overlap state is exact at the seek boundary.
   */
  seekToFrame(targetFrame) {
    const clamped = Math.max(0, Math.min(Math.max(0, this.totalFrames - 1), targetFrame | 0));
    this.olaL.fill(0);
    this.olaR.fill(0);
    const preStart = Math.max(0, clamped - 2);
    const dummyL = new Float32Array(CWC_HOP_M);
    const dummyR = new Float32Array(CWC_HOP_M);
    for (let f = preStart; f < clamped; f++) {
      this.decodeFrame(f, dummyL, dummyR, 0);
    }
    this.currentFrame = clamped;
    return clamped;
  }

  /**
   * Synthesizes exactly 512 stereo samples (1 MDCT hop = 11.61ms) into outL/outR at offset.
   * Zero allocations during steady-state streaming.
   */
  decodeFrame(frameIdx, outL, outR, outOffset = 0) {
    this.currentFrame = frameIdx;
    const M = CWC_HOP_M;
    const fStr = frameIdx >= 0 && frameIdx < this.totalFrames ? this.frameStrings[frameIdx] : '';
    const specM = this.specMid;
    const specS = this.specSide;
    specM.fill(0);
    specS.fill(0);

    const g0 = this.effectiveGains[0];
    const g1 = this.effectiveGains[1];
    const g2 = this.effectiveGains[2];
    const g3 = this.effectiveGains[3];
    const g4 = this.effectiveGains[4];
    const layerPow = this._layerPow || (this._layerPow = new Float64Array(5));
    layerPow.fill(0);

    // Deterministic frame-indexed PRNG so PNS noise never drifts on seek
    let seed = ((frameIdx + 1) * 0x9e3779b1) ^ 0x2468ace1;
    const nextNoise = () => {
      seed ^= seed << 13;
      seed >>>= 0;
      seed ^= seed >>> 17;
      seed >>>= 0;
      seed ^= seed << 5;
      seed >>>= 0;
      const u1 = (seed >>> 0) / 4294967296 - 0.5;
      seed ^= seed << 13;
      seed >>>= 0;
      seed ^= seed >>> 17;
      seed >>>= 0;
      seed ^= seed << 5;
      seed >>>= 0;
      const u2 = (seed >>> 0) / 4294967296 - 0.5;
      return (u1 + u2) * 2.4494897; // zero-mean ~unit variance
    };

    if (fStr && fStr.length > 0) {
      const slash = fStr.indexOf('/');
      const eff = this.effectiveGains;

      // 1. Mid Tonal Core (0..240 bins = 0..10.33 kHz)
      unpackQuantizedVec(fStr, 12, slash, this.decMidV);
      for (let bi = 0; bi < MID_BANDS.length; bi++) {
        const qG = B62_INV[fStr.charCodeAt(bi)];
        if (qG <= 0) continue;
        const bd = MID_BANDS[bi];
        const step = GAIN_LUT[qG] / bd.qMax;
        const lg = eff[bd.layer];
        let pow = 0;
        for (let k = bd.lo; k < bd.hi; k++) {
          const val = this.decMidV[k] * step;
          pow += val * val;
          specM[k] = val * lg;
        }
        layerPow[bd.layer] += pow;
      }

      // 2. Mid High-Band Perceptual Noise Substitution (10.3..19.6 kHz, Layer 3)
      for (let pi = 0; pi < PNS_MID_BANDS.length; pi++) {
        const qDb = B62_INV[fStr.charCodeAt(9 + pi)];
        if (qDb <= 0) continue;
        const [lo, hi] = PNS_MID_BANDS[pi];
        const c = hi - lo;
        const rms = PNS_LUT[qDb];
        let np = 0;
        for (let u = 0; u < c; u++) {
          const n = nextNoise();
          this.noiseScratch[u] = n;
          np += n * n;
        }
        const sc = np > 1e-9 ? rms * Math.sqrt(c / np) : 0;
        let bandP = 0;
        for (let u = 0; u < c; u++) {
          const val = this.noiseScratch[u] * sc;
          bandP += val * val;
          specM[lo + u] = val * g3;
        }
        layerPow[3] += bandP;
      }

      // 3. Side Stereo Field (1..168 tonal + High-Band Side PNS, Layer 4)
      const sOff = slash + 1;
      unpackQuantizedVec(fStr, sOff + 6, fStr.length, this.decSideV);
      let sideP = 0;
      for (let bi = 0; bi < SIDE_BANDS.length; bi++) {
        const qG = B62_INV[fStr.charCodeAt(sOff + bi)];
        if (qG <= 0) continue;
        const bd = SIDE_BANDS[bi];
        const step = GAIN_LUT[qG] / bd.qMax;
        for (let k = bd.lo; k < bd.hi; k++) {
          const val = this.decSideV[k] * step;
          sideP += val * val;
          specS[k] = val * g4;
        }
      }
      for (let pi = 0; pi < PNS_SIDE_BANDS.length; pi++) {
        const qDb = B62_INV[fStr.charCodeAt(sOff + 4 + pi)];
        if (qDb <= 0) continue;
        const [lo, hi] = PNS_SIDE_BANDS[pi];
        const c = hi - lo;
        const rms = PNS_LUT[qDb];
        let np = 0;
        for (let u = 0; u < c; u++) {
          const n = nextNoise();
          this.noiseScratch[u] = n;
          np += n * n;
        }
        const sc = np > 1e-9 ? rms * Math.sqrt(c / np) : 0;
        for (let u = 0; u < c; u++) {
          const val = this.noiseScratch[u] * sc;
          sideP += val * val;
          specS[lo + u] = val * g4;
        }
      }
      layerPow[4] += sideP;
    }

    // Update layer RMS telemetry & 32-band logarithmic spectrum telemetry
    // Note: orthonormal energy scale = 1 / (M * M)
    const normRms = 1 / Math.sqrt(M * M);
    this.layerRms[0] = Math.min(1, Math.sqrt(layerPow[0]) * normRms * g0 * 1.8);
    this.layerRms[1] = Math.min(1, Math.sqrt(layerPow[1]) * normRms * g1 * 2.2);
    this.layerRms[2] = Math.min(1, Math.sqrt(layerPow[2]) * normRms * g2 * 3.8);
    this.layerRms[3] = Math.min(1, Math.sqrt(layerPow[3]) * normRms * g3 * 5.2);
    this.layerRms[4] = Math.min(1, Math.sqrt(layerPow[4]) * normRms * g4 * 4.5);

    for (let b = 0; b < 32; b++) {
      const lo = Math.floor(Math.pow(455, b / 32)) - 1;
      const hi = Math.max(lo + 1, Math.floor(Math.pow(455, (b + 1) / 32)));
      let e = 0;
      for (let k = Math.max(0, lo); k < Math.min(455, hi); k++) {
        e += specM[k] * specM[k] + specS[k] * specS[k];
      }
      const db = Math.log10(Math.sqrt(e / (hi - lo + 1)) * 0.0625 + 1e-4);
      this.spectrum32[b] = Math.max(0, Math.min(1, (db + 3.2) / 3.4));
    }

    // Run 2 IMDCT transforms (Mid + Side)
    this.mdct.inverse(specM, this.outMidN);
    this.mdct.inverse(specS, this.outSideN);

    // Princen-Bradley TDAC overlap-add + Mid/Side -> L/R matrix + soft ceiling
    const mg = this.masterGain;
    for (let n = 0; n < M; n++) {
      const curL = this.outMidN[n] + this.outSideN[n];
      const curR = this.outMidN[n] - this.outSideN[n];
      const l = (this.olaL[n] + curL) * mg;
      const r = (this.olaR[n] + curR) * mg;
      outL[outOffset + n] = l > 0.995 ? 0.995 : (l < -0.995 ? -0.995 : l);
      outR[outOffset + n] = r > 0.995 ? 0.995 : (r < -0.995 ? -0.995 : r);

      this.olaL[n] = this.outMidN[M + n] + this.outSideN[M + n];
      this.olaR[n] = this.outMidN[M + n] - this.outSideN[M + n];
    }
  }
}
