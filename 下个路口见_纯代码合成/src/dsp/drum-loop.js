import { SpectralStream } from './spectral-stream.js';
import { interpolate } from './resampler.js';

export class DrumLoopSynth {
  constructor(loopSpec, buffer, outputRate) {
    this.rate = loopSpec.sampleRate;
    this.ratio = this.rate / outputRate;
    this.barSamples = loopSpec.barSamples;
    this.fadeSamples = loopSpec.fadeSamples || 64;
    this.totalSamples = loopSpec.totalSamples;
    this.bars = loopSpec.bars; // [[startSample, protoIdx, gainByte], ...]

    // Decode compact prototype bank (~9.8s) once into contiguous Float32Array for O(1) playback
    const protoStream = new SpectralStream(buffer, this.rate);
    this.protoPcm = new Float32Array(protoStream.sampleCount);
    for (let i = 0; i < this.protoPcm.length; i++) {
      this.protoPcm[i] = protoStream.read(i);
    }

    this.fadeIn = Float32Array.from(
      { length: this.fadeSamples },
      (_, i) => 0.5 - 0.5 * Math.cos((Math.PI * i) / this.fadeSamples)
    );
    this.cursor = 0;
  }

  reset() {
    this.cursor = 0;
  }

  _evalBar(entry, sampleIdx) {
    const start = entry[0];
    const local = sampleIdx - start;
    const end = Math.min(this.totalSamples, start + this.barSamples);
    const len = end - start;
    if (local < 0 || local >= len) return 0;
    const protoIdx = entry[1];
    const gain = entry[2] / 128;
    let val = this.protoPcm[protoIdx * this.barSamples + local] * gain;
    if (len > 2 * this.fadeSamples) {
      if (local < this.fadeSamples) val *= this.fadeIn[local];
      else if (local >= len - this.fadeSamples) val *= 1 - this.fadeIn[local - (len - this.fadeSamples)];
    }
    return val;
  }

  read(index) {
    if (index < 0 || index >= this.totalSamples || this.bars.length === 0) return 0;
    // Fast monotonic cursor + binary search fallback on seek
    let c = this.cursor;
    if (c >= this.bars.length || this.bars[c][0] > index || (c + 1 < this.bars.length && this.bars[c + 1][0] <= index)) {
      let lo = 0;
      let hi = this.bars.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (this.bars[mid][0] <= index) {
          c = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      this.cursor = c;
    }
    let sum = this._evalBar(this.bars[c], index);
    if (c > 0 && index < this.bars[c - 1][0] + this.barSamples) {
      sum += this._evalBar(this.bars[c - 1], index);
    }
    return sum;
  }

  valueAt(outputFrame) {
    return interpolate(this, outputFrame * this.ratio);
  }
}
