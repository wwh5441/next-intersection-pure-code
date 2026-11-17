const kernels = new Map();

class ForwardFFT {
  constructor(size) {
    this.size = size;
    this.reverse = new Uint32Array(size);
    const bits = Math.log2(size);
    for (let i = 0; i < size; i++) {
      let value = i;
      let reversed = 0;
      for (let b = 0; b < bits; b++) {
        reversed = (reversed << 1) | (value & 1);
        value >>>= 1;
      }
      this.reverse[i] = reversed;
    }
    this.cos = new Float64Array(size / 2);
    this.sin = new Float64Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / size);
      this.sin[i] = Math.sin((2 * Math.PI * i) / size);
    }
  }

  transform(real, imag) {
    const n = this.size;
    for (let i = 0; i < n; i++) {
      const j = this.reverse[i];
      if (j > i) {
        let v = real[i];
        real[i] = real[j];
        real[j] = v;
        v = imag[i];
        imag[i] = imag[j];
        imag[j] = v;
      }
    }
    for (let width = 2; width <= n; width *= 2) {
      const half = width / 2;
      const stride = n / width;
      for (let base = 0; base < n; base += width) {
        for (let j = 0; j < half; j++) {
          const p = base + j;
          const q = p + half;
          const twiddle = j * stride;
          const c = this.cos[twiddle];
          const s = this.sin[twiddle];
          const tr = real[q] * c + imag[q] * s;
          const ti = imag[q] * c - real[q] * s;
          real[q] = real[p] - tr;
          imag[q] = imag[p] - ti;
          real[p] += tr;
          imag[p] += ti;
        }
      }
    }
  }
}

class FastImdct {
  constructor(n) {
    this.n = n;
    this.h = n >> 1;
    this.fft = new ForwardFFT(this.h);
    this.preCos = new Float64Array(this.h);
    this.preSin = new Float64Array(this.h);
    this.postCos = new Float64Array(this.h);
    this.postSin = new Float64Array(this.h);
    for (let r = 0; r < this.h; r++) {
      const a1 = (Math.PI * r) / n;
      this.preCos[r] = Math.cos(a1);
      this.preSin[r] = Math.sin(a1);
      const a2 = (Math.PI * (r + 0.25)) / n;
      this.postCos[r] = Math.cos(a2);
      this.postSin[r] = Math.sin(a2);
    }
    this.real = new Float64Array(this.h);
    this.imag = new Float64Array(this.h);
    this.dct = new Float64Array(n);
  }

  inverse(coeffs, out2n) {
    const { n, h, real, imag, dct, preCos, preSin, postCos, postSin } = this;
    for (let r = 0; r < h; r++) {
      const ar = coeffs[2 * r];
      const ai = coeffs[n - 1 - 2 * r];
      real[r] = ar * preCos[r] + ai * preSin[r];
      imag[r] = ai * preCos[r] - ar * preSin[r];
    }
    this.fft.transform(real, imag);
    for (let p = 0; p < h; p++) {
      const yr = real[p] * postCos[p] + imag[p] * postSin[p];
      const yi = imag[p] * postCos[p] - real[p] * postSin[p];
      dct[2 * p] = yr;
      dct[n - 1 - 2 * p] = -yi;
    }
    const scale = 2 / n;
    for (let m = 0; m < h; m++) {
      out2n[m] = dct[h + m] * scale;
      out2n[h + m] = -dct[n - 1 - m] * scale;
      out2n[n + m] = -dct[h - 1 - m] * scale;
      out2n[n + h + m] = -dct[m] * scale;
    }
  }
}

export function imdctPlan(n) {
  if (!kernels.has(n)) kernels.set(n, new FastImdct(n));
  return kernels.get(n);
}
