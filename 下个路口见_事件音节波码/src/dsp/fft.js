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

  _bitReverse() {
    const N = this.size;
    const { re, im, reverse } = this;
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
  }

  /** In-place forward DFT (`exp(-i * 2*pi * j * k / N)`, unnormalized). */
  forwardInPlace() {
    const N = this.size;
    const { re, im, cos, sin } = this;
    this._bitReverse();
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
          const tr = re[q] * c + im[q] * s;
          const ti = im[q] * c - re[q] * s;
          re[q] = re[p] - tr;
          im[q] = im[p] - ti;
          re[p] += tr;
          im[p] += ti;
        }
      }
    }
  }

  /** In-place inverse DFT (`exp(+i * 2*pi * j * k / N) / N`). */
  inverseInPlace() {
    const N = this.size;
    const { re, im, cos, sin, invN } = this;
    this._bitReverse();
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

  /**
   * Simultaneous Dual Real-Forward FFT:
   * Packs two windowed real chunks `(x0 * wIn, x1 * wIn)` of length `winLen` (zero-padded to `size`),
   * computes both positive-frequency spectra `(outRe0, outIm0)` and `(outRe1, outIm1)` in ONE FFT pass!
   */
  dualRealForward(src0, src1, start, wIn, winLen, outRe0, outIm0, outRe1, outIm1) {
    const N = this.size;
    const half = N >> 1;
    const { re, im } = this;
    for (let u = 0; u < winLen; u++) {
      const w = wIn[u];
      re[u] = src0[start + u] * w;
      im[u] = src1[start + u] * w;
    }
    re.fill(0.0, winLen, N);
    im.fill(0.0, winLen, N);

    this.forwardInPlace();

    outRe0[0] = re[0];
    outIm0[0] = 0.0;
    outRe1[0] = im[0];
    outIm1[0] = 0.0;

    outRe0[half] = re[half];
    outIm0[half] = 0.0;
    outRe1[half] = im[half];
    outIm1[half] = 0.0;

    for (let k = 1; k < half; k++) {
      const nk = N - k;
      const zr = re[k];
      const zi = im[k];
      const znr = re[nk];
      const zni = im[nk];
      outRe0[k] = 0.5 * (zr + znr);
      outIm0[k] = 0.5 * (zi - zni);
      outRe1[k] = 0.5 * (zi + zni);
      outIm1[k] = 0.5 * (znr - zr);
    }
  }

  /**
   * Simultaneous Dual Real-IFFT:
   * Given conjugate-symmetric positive spectra `(hRe, hIm)` and `(aRe, aIm)` of length `size/2 + 1`,
   * computes `real(IFFT) = h(n)` and `imag(IFFT) = a(n)` in a single pass!
   */
  dualRealInverse(hRe, hIm, aRe, aIm) {
    const N = this.size;
    const half = N >> 1;
    const { re, im } = this;

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
      const nk = N - k;
      re[nk] = hr + ai;
      im[nk] = -hi + ar;
    }

    this.inverseInPlace();
  }
}
