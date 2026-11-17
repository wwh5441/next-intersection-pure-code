export class Radix2FFT {
  constructor(size) {
    this.size = size;
    const bits = Math.round(Math.log2(size));
    this.rev = new Uint16Array(size);
    for (let i = 0; i < size; i++) {
      let x = i;
      let r = 0;
      for (let b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
      }
      this.rev[i] = r;
    }
    this.cosTable = new Float32Array(size >> 1);
    this.sinTable = new Float32Array(size >> 1);
    for (let k = 0; k < size >> 1; k++) {
      const a = (-2 * Math.PI * k) / size;
      this.cosTable[k] = Math.cos(a);
      this.sinTable[k] = Math.sin(a);
    }
  }

  transform(re, im, inverse = false) {
    const n = this.size;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }

    const sign = inverse ? -1 : 1;
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < half; j++) {
          const tw = j * step;
          const wr = this.cosTable[tw];
          const wi = sign * this.sinTable[tw];
          const u = i + j;
          const v = u + half;
          const tr = wr * re[v] - wi * im[v];
          const ti = wr * im[v] + wi * re[v];
          re[v] = re[u] - tr;
          im[v] = im[u] - ti;
          re[u] += tr;
          im[u] += ti;
        }
      }
    }

    if (inverse) {
      const invN = 1 / n;
      for (let i = 0; i < n; i++) {
        re[i] *= invN;
        im[i] *= invN;
      }
    }
  }
}

export class XorShift32 {
  constructor(seed = 123456789) {
    this.state = seed >>> 0 || 1;
  }

  nextFloat() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return (this.state >>> 0) / 4294967296;
  }

  nextSigned() {
    return this.nextFloat() * 2 - 1;
  }
}

export function interpPiecewise(xs, ys, x, leftVal = ys[0], rightVal = ys[ys.length - 1]) {
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
