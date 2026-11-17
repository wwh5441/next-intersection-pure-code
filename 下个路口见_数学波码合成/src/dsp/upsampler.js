const PHASES = 128;
const TAPS = 12;
const HALF = TAPS >> 1;

function buildPolyphaseTable(cutoff) {
  const table = new Float32Array(PHASES * TAPS);
  for (let p = 0; p < PHASES; p++) {
    const frac = p / PHASES;
    let sum = 0;
    for (let t = 0; t < TAPS; t++) {
      const x = t - (HALF - 1) - frac;
      const wx = 0.54 - 0.46 * Math.cos((2 * Math.PI * (x + HALF)) / TAPS);
      const sx = Math.abs(x) < 1e-7 ? cutoff : Math.sin(Math.PI * cutoff * x) / (Math.PI * x);
      const v = wx * sx;
      table[p * TAPS + t] = v;
      sum += v;
    }
    const inv = 1 / (sum || 1);
    for (let t = 0; t < TAPS; t++) table[p * TAPS + t] *= inv;
  }
  return table;
}

export class StreamingResampler {
  constructor(sourceRate, targetRate, pullSampleFn) {
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.ratio = sourceRate / targetRate;
    this.pullSampleFn = pullSampleFn;
    this.table = buildPolyphaseTable(Math.min(0.94, targetRate / sourceRate));
    this.ring = new Float32Array(32);
    this.sourceCursor = -1;
    this.loadedIndex = -1;
  }

  reset(targetSampleIndex = 0) {
    this.ring.fill(0);
    const srcPos = targetSampleIndex * this.ratio;
    const base = Math.floor(srcPos);
    this.loadedIndex = base - HALF - 1;
    for (let k = this.loadedIndex + 1; k <= base + HALF + 1; k++) {
      this.ring[k & 31] = k >= 0 ? this.pullSampleFn(k) : 0;
      this.loadedIndex = k;
    }
  }

  sampleAt(targetSampleIndex) {
    if (this.sourceRate === this.targetRate) {
      return this.pullSampleFn(targetSampleIndex);
    }
    const srcPos = targetSampleIndex * this.ratio;
    const base = Math.floor(srcPos);
    const needMax = base + HALF;
    if (needMax - this.loadedIndex > 28) {
      this.loadedIndex = base - HALF - 1;
      for (let k = this.loadedIndex + 1; k <= needMax; k++) {
        this.ring[k & 31] = k >= 0 ? this.pullSampleFn(k) : 0;
        this.loadedIndex = k;
      }
    } else {
      while (this.loadedIndex < needMax) {
        this.loadedIndex++;
        this.ring[this.loadedIndex & 31] = this.loadedIndex >= 0 ? this.pullSampleFn(this.loadedIndex) : 0;
      }
    }
    const frac = srcPos - base;
    const phase = Math.min(PHASES - 1, (frac * PHASES) | 0) * TAPS;
    let acc = 0;
    const startIdx = base - (HALF - 1);
    for (let t = 0; t < TAPS; t++) {
      acc += this.ring[(startIdx + t) & 31] * this.table[phase + t];
    }
    return acc;
  }
}
