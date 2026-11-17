// Pure Mathematical Radix-2 Dual-Real FFT/IFFT, Polyphase Resampling, PWL/RLE Decoders & ERB Warping

const TWO_PI = 2.0 * Math.PI;
const POLY_PHASES = 128;
const POLY_TAPS = 12;
const POLY_HALF = POLY_TAPS >> 1;

export class ComplexInverseFFT {
  constructor(size) {
    this.size = size;
    this.n = size;
    this.invN = 1.0 / size;
    this.reverse = new Uint32Array(size);
    const bits = Math.round(Math.log2(size));
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
      const ang = (TWO_PI * i) / size;
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
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
  }

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

    outRe0[0] = re[0]; outIm0[0] = 0.0;
    outRe1[0] = im[0]; outIm1[0] = 0.0;
    outRe0[half] = re[half]; outIm0[half] = 0.0;
    outRe1[half] = im[half]; outIm1[half] = 0.0;

    for (let k = 1; k < half; k++) {
      const nk = N - k;
      const zr = re[k], zi = im[k];
      const znr = re[nk], zni = im[nk];
      outRe0[k] = 0.5 * (zr + znr);
      outIm0[k] = 0.5 * (zi - zni);
      outRe1[k] = 0.5 * (zi + zni);
      outIm1[k] = 0.5 * (znr - zr);
    }
  }

  dualRealInverse(hRe, hIm, aRe, aIm) {
    const N = this.size;
    const half = N >> 1;
    const { re, im } = this;
    re[0] = hRe[0]; im[0] = aRe[0];
    re[half] = hRe[half]; im[half] = aRe[half];
    for (let k = 1; k < half; k++) {
      const hr = hRe[k], hi = hIm[k];
      const ar = aRe[k], ai = aIm[k];
      re[k] = hr - ai;
      im[k] = hi + ar;
      const nk = N - k;
      re[nk] = hr + ai;
      im[nk] = -hi + ar;
    }
    this.inverseInPlace();
  }

  inverseReal(reIn, imIn, timeOut) {
    const N = this.size;
    const half = N >> 1;
    const { re, im } = this;
    for (let k = 0; k <= half; k++) {
      re[k] = reIn[k];
      im[k] = imIn[k];
    }
    for (let k = 1; k < half; k++) {
      re[N - k] = reIn[k];
      im[N - k] = -imIn[k];
    }
    this.inverseInPlace();
    for (let i = 0; i < N; i++) timeOut[i] = re[i];
  }
}

export class DeterministicRng {
  constructor(seed = 0x9E3779B9) {
    this.state = seed >>> 0 || 1;
  }

  nextFloat01() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return (this.state >>> 0) / 4294967296.0;
  }

  uniform() {
    return this.nextFloat01() * 2.0 - 1.0;
  }

  gaussian() {
    return 0.8660254 * (this.uniform() + this.uniform() + this.uniform() + this.uniform());
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

export function erbNormCoord(freqHz, fMin = 75.0, fMax = 16000.0) {
  const e0 = 21.4 * Math.log10(1.0 + 0.00437 * fMin);
  const e1 = 21.4 * Math.log10(1.0 + 0.00437 * fMax);
  const fc = Math.max(fMin, Math.min(fMax, freqHz));
  return (21.4 * Math.log10(1.0 + 0.00437 * fc) - e0) / (e1 - e0);
}

export function interpPiecewiseLinear(xs, ys, x, leftVal = ys[0], rightVal = ys[ys.length - 1]) {
  const n = xs.length;
  if (x <= xs[0]) return leftVal;
  if (x >= xs[n - 1]) return rightVal;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] + 1e-12);
  return ys[lo] + (ys[hi] - ys[lo]) * t;
}

export function decodePwlTimeline(pair, nFrames, div = 1.0, gate = 0.0) {
  const out = new Float32Array(nFrames);
  if (!pair || !Array.isArray(pair[0]) || pair[0].length === 0) return out;
  const dts = pair[0];
  const dvs = pair[1];
  const mLen = dts.length;
  const thresh = 0.5 * div;

  let idx0 = dts[0] | 0;
  let vq = dvs[0] | 0;
  let val0 = vq * div;
  if (idx0 >= 0 && idx0 < nFrames) {
    out[idx0] = gate > 0 ? (val0 > thresh ? val0 + gate : 0.0) : val0;
  }

  for (let m = 1; m < mLen; m++) {
    const idx1 = idx0 + (dts[m] | 0);
    vq += dvs[m] | 0;
    const val1 = vq * div;
    const span = idx1 - idx0;
    if (span > 0) {
      const inv = 1.0 / span;
      const sEnd = idx1 < nFrames ? idx1 : nFrames - 1;
      for (let s = idx0 + 1; s <= sEnd; s++) {
        const v = val0 + (val1 - val0) * ((s - idx0) * inv);
        out[s] = gate > 0 ? (v > thresh ? v + gate : 0.0) : v;
      }
    }
    idx0 = idx1;
    val0 = val1;
  }
  return out;
}

export function decodeRleTimeline(pair, nFrames, stride = 1) {
  const out = new Int32Array(nFrames);
  if (!pair || !Array.isArray(pair[0]) || pair[0].length === 0) return out;
  const lens = pair[0];
  const dvals = pair[1];
  let pos = 0;
  let cur = 0;
  for (let m = 0; m < lens.length && pos < nFrames; m++) {
    cur += dvals[m] | 0;
    const end = Math.min(nFrames, pos + (lens[m] | 0) * stride);
    out.fill(cur, pos, end);
    pos = end;
  }
  if (pos < nFrames) out.fill(cur, pos, nFrames);
  return out;
}

function buildWindowedSincTable(cutoff = 0.94) {
  const table = new Float32Array(POLY_PHASES * POLY_TAPS);
  for (let p = 0; p < POLY_PHASES; p++) {
    const frac = p / POLY_PHASES;
    let sum = 0.0;
    for (let t = 0; t < POLY_TAPS; t++) {
      const x = t - (POLY_HALF - 1) - frac;
      const wx = 0.54 - 0.46 * Math.cos((TWO_PI * (x + POLY_HALF)) / POLY_TAPS);
      const sx = Math.abs(x) < 1e-7 ? cutoff : Math.sin(Math.PI * cutoff * x) / (Math.PI * x);
      const v = wx * sx;
      table[p * POLY_TAPS + t] = v;
      sum += v;
    }
    const inv = 1.0 / (sum || 1.0);
    for (let t = 0; t < POLY_TAPS; t++) table[p * POLY_TAPS + t] *= inv;
  }
  return table;
}

const SINC_TABLE_094 = buildWindowedSincTable(0.94);

// Fast zero-branch 2x polyphase upsampler (24kHz -> 48kHz). `srcPad8` has 8 leading zero samples.
export function upsample2xPolyphase(srcPad8, dst48k) {
  const nOut = dst48k.length;
  const p0 = SINC_TABLE_094.subarray(0, 12);
  const p1 = SINC_TABLE_094.subarray(64 * 12, 65 * 12);
  const pairs = nOut >> 1;
  for (let m = 0; m < pairs; m++) {
    const b = m + 3; // 8 - (HALF - 1) = 3
    let a0 = 0.0;
    let a1 = 0.0;
    for (let t = 0; t < 12; t++) {
      const sv = srcPad8[b + t];
      a0 += sv * p0[t];
      a1 += sv * p1[t];
    }
    const o = m << 1;
    dst48k[o] = a0;
    dst48k[o + 1] = a1;
  }
}

// Fast zero-branch 4x polyphase upsampler (12kHz -> 48kHz). `srcPad8` has 8 leading zero samples.
export function upsample4xPolyphase(srcPad8, dst48k) {
  const nOut = dst48k.length;
  const p0 = SINC_TABLE_094.subarray(0, 12);
  const p1 = SINC_TABLE_094.subarray(32 * 12, 33 * 12);
  const p2 = SINC_TABLE_094.subarray(64 * 12, 65 * 12);
  const p3 = SINC_TABLE_094.subarray(96 * 12, 97 * 12);
  const quads = nOut >> 2;
  for (let m = 0; m < quads; m++) {
    const b = m + 3;
    let a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (let t = 0; t < 12; t++) {
      const sv = srcPad8[b + t];
      a0 += sv * p0[t];
      a1 += sv * p1[t];
      a2 += sv * p2[t];
      a3 += sv * p3[t];
    }
    const o = m << 2;
    dst48k[o] = a0;
    dst48k[o + 1] = a1;
    dst48k[o + 2] = a2;
    dst48k[o + 3] = a3;
  }
}
