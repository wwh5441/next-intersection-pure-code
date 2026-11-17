export class ComplexInverseFFT {
  constructor(size) {
    this.size = size;
    this.invN = 1.0 / size;
    this.reverse = new Uint32Array(size);
    const bits = Math.log2(size);
    for (let i = 0; i < size; i++) {
      let val = i;
      let rev = 0;
      for (let b = 0; b < bits; b++) {
        rev = (rev << 1) | (val & 1);
        val >>>= 1;
      }
      this.reverse[i] = rev;
    }
    this.cos = new Float64Array(size >> 1);
    this.sin = new Float64Array(size >> 1);
    for (let i = 0; i < size >> 1; i++) {
      const ang = (2.0 * Math.PI * i) / size;
      this.cos[i] = Math.cos(ang);
      this.sin[i] = Math.sin(ang);
    }
    this.re = new Float64Array(size);
    this.im = new Float64Array(size);
  }

  /**
   * Simultaneous Dual Real-IFFT:
   * Given conjugate-symmetric positive spectra `(hRe, hIm)` and `(aRe, aIm)` of length `size/2 + 1`,
   * computes `real(IFFT) = h(n)` and `imag(IFFT) = a(n)` in a single pass!
   */
  dualRealInverse(hRe, hIm, aRe, aIm) {
    const N = this.size;
    const half = N >> 1;
    const { re, im, reverse, cos, sin, invN } = this;

    // F[k] = H[k] + i * A[k] = (hRe[k] - aIm[k]) + i * (hIm[k] + aRe[k])
    re[0] = hRe[0];
    im[0] = aRe[0];
    re[half] = hRe[half];
    im[half] = aRe[half];

    for (let k = 1; k < half; k++) {
      const hr = hRe[k];
      const hi = hIm[k];
      const ar = aRe[k];
      const ai = aIm[k];
      re[k] = hr - ai;
      im[k] = hi + ar;
      // At N - k: H[N-k] = hr - i*hi, A[N-k] = ar - i*ai -> F[N-k] = (hr + ai) + i*(-hi + ar)
      const nk = N - k;
      re[nk] = hr + ai;
      im[nk] = -hi + ar;
    }

    for (let i = 0; i < N; i++) {
      const j = reverse[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }

    // Inverse DFT uses exp(+i * 2*pi * j * k / N)
    for (let width = 2; width <= N; width <<= 1) {
      const wHalf = width >> 1;
      const stride = N / width;
      for (let base = 0; base < N; base += width) {
        for (let j = 0; j < wHalf; j++) {
          const p = base + j;
          const q = p + wHalf;
          const tw = j * stride;
          const c = cos[tw];
          const s = sin[tw];
          const tr = re[q] * c - im[q] * s;
          const ti = im[q] * c + re[q] * s;
          re[q] = re[p] - tr;
          im[q] = im[p] - ti;
          re[p] += tr;
          im[p] += ti;
        }
      }
    }

    for (let i = 0; i < N; i++) {
      re[i] *= invN;
      im[i] *= invN;
    }
  }
}
