// Pure Mathematical Radix-2 Complex FFT & Deterministic PRNG (`src/dsp/fft-core.js`)

export class Radix2ComplexFFT {
  constructor(size) {
    this.n = size;
    const bits = Math.round(Math.log2(size));
    this.rev = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let x = i;
      let r = 0;
      for (let b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>>= 1;
      }
      this.rev[i] = r;
    }
    this.cosTab = new Float64Array(size >> 1);
    this.sinTab = new Float64Array(size >> 1);
    for (let k = 0; k < size >> 1; k++) {
      const ang = (-2.0 * Math.PI * k) / size;
      this.cosTab[k] = Math.cos(ang);
      this.sinTab[k] = Math.sin(ang);
    }
  }

  transform(re, im, inverse = false) {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (i < j) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    const cosTab = this.cosTab;
    const sinTab = this.sinTab;
    const signFlip = inverse ? -1.0 : 1.0;

    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const twIdx = k * step;
          const wr = cosTab[twIdx];
          const wi = sinTab[twIdx] * signFlip;
          const a = i + k;
          const b = a + half;
          const br = re[b];
          const bi = im[b];
          const tr = br * wr - bi * wi;
          const ti = br * wi + bi * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
    if (inverse) {
      const invN = 1.0 / n;
      for (let i = 0; i < n; i++) {
        re[i] *= invN;
        im[i] *= invN;
      }
    }
  }
}

export function createMulberry32(seed) {
  let a = seed >>> 0;
  return function nextRand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
}
