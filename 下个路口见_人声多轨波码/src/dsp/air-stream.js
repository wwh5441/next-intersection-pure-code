const AIR_MIN_DB = -64.0;
const AIR_STEP_DB = 0.45;
const SQRT6 = Math.sqrt(6);

const AIR_RMS_LUT = Float32Array.from({ length: 128 }, (_, c) =>
  c === 0 ? 0 : 10 ** ((AIR_MIN_DB + (c - 1) * AIR_STEP_DB) / 20)
);

function design2StageBandpass(fLo, fHi, sr) {
  const f0 = Math.sqrt(fLo * fHi);
  const q = Math.max(0.55, f0 / Math.max(200, fHi - fLo));
  const w0 = (2 * Math.PI * f0) / sr;
  const alpha = Math.sin(w0) / (2 * q);
  const invA0 = 1 / (1 + alpha);
  const b0 = alpha * invA0;
  const a1 = (-2 * Math.cos(w0)) * invA0;
  const a2 = (1 - alpha) * invA0;

  // Compute exact 2-stage impulse L2 norm so unit-variance noise yields exact 1.0 RMS
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  let u1 = 0, u2 = 0, v1 = 0, v2 = 0;
  let sumSq = 0;
  for (let n = 0; n < 1024; n++) {
    const x0 = n === 0 ? 1 : 0;
    const y0 = b0 * (x0 - x2) - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x0; y2 = y1; y1 = y0;
    const v0 = b0 * (y0 - u2) - a1 * v1 - a2 * v2;
    u2 = u1; u1 = y0; v2 = v1; v1 = v0;
    sumSq += v0 * v0;
  }
  return { b0, a1, a2, invNorm: 1 / Math.sqrt(sumSq + 1e-18) };
}

export class HighAirStream {
  constructor(buffer, outputRate) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
    if (magic !== 'VWA7') throw new Error(`Invalid high-air vocal packet: ${magic}`);

    this.rate = view.getUint32(4, true);
    this.sampleCount = view.getUint32(8, true);
    this.hop = view.getUint32(12, true);
    this.nHops = view.getUint32(16, true);

    this.filters = [];
    for (let b = 0; b < 4; b++) {
      const fLo = view.getUint16(20 + b * 4, true);
      const fHi = view.getUint16(22 + b * 4, true);
      this.filters.push(design2StageBandpass(fLo, fHi, outputRate));
    }
    this.codes = new Uint8Array(buffer, 36, this.nHops * 4);
    this.states = new Float64Array(4 * 8);
    this.reset();
  }

  reset() {
    this.states.fill(0);
    this.rng = 0x6d2b79f5;
  }

  _nextUnitNoise() {
    let s = this.rng | 0;
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    const u1 = (s >>> 0) * 2.3283064365386963e-10;
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    const u2 = (s >>> 0) * 2.3283064365386963e-10;
    this.rng = s || 1;
    return (u1 + u2 - 1) * SQRT6;
  }

  valueAt(frame) {
    if (frame < 0 || frame >= this.sampleCount) return 0;
    const pos = frame / this.hop;
    const h0 = Math.floor(pos);
    const h1 = Math.min(h0 + 1, this.nHops - 1);
    const base0 = h0 * 4;
    const base1 = h1 * 4;

    const c00 = this.codes[base0], c10 = this.codes[base1];
    const c01 = this.codes[base0 + 1], c11 = this.codes[base1 + 1];
    const c02 = this.codes[base0 + 2], c12 = this.codes[base1 + 2];
    const c03 = this.codes[base0 + 3], c13 = this.codes[base1 + 3];
    if ((c00 | c10 | c01 | c11 | c02 | c12 | c03 | c13) === 0) {
      return 0;
    }

    const frac = pos - h0;
    const invF = 1 - frac;
    const env0 = AIR_RMS_LUT[c00] * invF + AIR_RMS_LUT[c10] * frac;
    const env1 = AIR_RMS_LUT[c01] * invF + AIR_RMS_LUT[c11] * frac;
    const env2 = AIR_RMS_LUT[c02] * invF + AIR_RMS_LUT[c12] * frac;
    const env3 = AIR_RMS_LUT[c03] * invF + AIR_RMS_LUT[c13] * frac;

    const w = this._nextUnitNoise();
    const envs = [env0, env1, env2, env3];
    const st = this.states;
    let out = 0;

    for (let b = 0; b < 4; b++) {
      const e = envs[b];
      if (e === 0) continue;
      const { b0, a1, a2, invNorm } = this.filters[b];
      const o = b * 8;
      const y0 = b0 * (w - st[o + 1]) - a1 * st[o + 2] - a2 * st[o + 3];
      st[o + 1] = st[o];
      st[o] = w;
      st[o + 3] = st[o + 2];
      st[o + 2] = y0;

      const v0 = b0 * (y0 - st[o + 5]) - a1 * st[o + 6] - a2 * st[o + 7];
      st[o + 5] = st[o + 4];
      st[o + 4] = y0;
      st[o + 7] = st[o + 6];
      st[o + 6] = v0;

      out += v0 * invNorm * e;
    }
    return out;
  }
}
