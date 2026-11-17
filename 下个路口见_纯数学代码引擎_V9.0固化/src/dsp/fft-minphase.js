// Pure Mathematical Radix-2 Complex IFFT, Deterministic PRNG & ERB Minimum-Phase Evaluator

const TWO_PI = 2.0 * Math.PI;

export class ComplexInverseFFT {
  constructor(n) {
    this.n = n;
    const bits = Math.round(Math.log2(n));
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let x = i;
      let r = 0;
      for (let b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>>= 1;
      }
      this.rev[i] = r;
    }
    this.cosTab = new Float64Array(n >> 1);
    this.sinTab = new Float64Array(n >> 1);
    for (let k = 0; k < (n >> 1); k++) {
      const ang = (TWO_PI * k) / n;
      this.cosTab[k] = Math.cos(ang);
      this.sinTab[k] = Math.sin(ang);
    }
  }

  inverseReal(reIn, imIn, timeOut) {
    const n = this.n;
    const rev = this.rev;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    const half = n >> 1;

    for (let k = 0; k <= half; k++) {
      re[k] = reIn[k];
      im[k] = imIn[k];
    }
    for (let k = 1; k < half; k++) {
      re[n - k] = reIn[k];
      im[n - k] = -imIn[k];
    }

    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }

    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < halfLen; j++) {
          const idx = j * step;
          const wr = this.cosTab[idx];
          const wi = this.sinTab[idx];
          const uR = re[i + j];
          const uI = im[i + j];
          const vR = re[i + j + halfLen] * wr - im[i + j + halfLen] * wi;
          const vI = re[i + j + halfLen] * wi + im[i + j + halfLen] * wr;
          re[i + j] = uR + vR;
          im[i + j] = uI + vI;
          re[i + j + halfLen] = uR - vR;
          im[i + j + halfLen] = uI - vI;
        }
      }
    }

    const invN = 1.0 / n;
    for (let i = 0; i < n; i++) {
      timeOut[i] = re[i] * invN;
    }
  }
}

export class DeterministicRng {
  constructor(seed = 0x9E3779B9) {
    this.state = seed >>> 0 || 1;
  }

  uniform() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return (this.state / 2147483648.0) - 1.0;
  }

  gaussian() {
    // Marsaglia polar or Irwin-Hall 4-uniform fast zero-mean unit-variance approximation
    return 0.8660254 * (this.uniform() + this.uniform() + this.uniform() + this.uniform());
  }
}

export function erbNormCoord(freqHz, fMin = 75.0, fMax = 16000.0) {
  const e0 = 21.4 * Math.log10(1.0 + 0.00437 * fMin);
  const e1 = 21.4 * Math.log10(1.0 + 0.00437 * fMax);
  const fc = Math.max(fMin, Math.min(fMax, freqHz));
  return (21.4 * Math.log10(1.0 + 0.00437 * fc) - e0) / (e1 - e0);
}
