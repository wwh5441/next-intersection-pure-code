// Exact Zero-Phase Kaiser-Sinc Polyphase Resamplers & Linear-Phase Crossovers (`src/dsp/polyphase-sinc.js`)
// Bit-exact counterpart to Python's `resample_zero_phase_poly` (>82 dB stopband attenuation, 0.000 sample shift).

const POLY_CACHE = new Map();

export function besselI0(x) {
  let sum = 1.0;
  let term = 1.0;
  const halfX2 = 0.25 * x * x;
  for (let k = 1; k <= 26; k++) {
    term *= halfX2 / (k * k);
    sum += term;
  }
  return sum;
}

function sincNorm(x) {
  if (Math.abs(x) < 1e-12) return 1.0;
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

export function getPolyphaseKernel(up, down, tapsPerStep = 16, beta = 8.6) {
  const key = `${up}:${down}:${tapsPerStep}:${beta}`;
  if (POLY_CACHE.has(key)) return POLY_CACHE.get(key);

  const maxUD = Math.max(up, down);
  const fc = 0.465 / maxUD;
  const halfLen = tapsPerStep * maxUD;
  const fullLen = 2 * halfLen + 1;
  const h = new Float64Array(fullLen);
  const invI0Beta = 1.0 / besselI0(beta);

  for (let i = 0; i < fullLen; i++) {
    const m = i - halfLen;
    const r = m / halfLen;
    const win = besselI0(beta * Math.sqrt(Math.max(0.0, 1.0 - r * r))) * invI0Beta;
    h[i] = up * (2.0 * fc * sincNorm(2.0 * fc * m)) * win;
  }

  const branches = new Array(up);
  let maxAbsK = 0;
  for (let p = 0; p < up; p++) {
    const kMin = Math.ceil((-halfLen - p) / up);
    const kMax = Math.floor((halfLen - p) / up);
    const len = kMax - kMin + 1;
    const w = new Float32Array(len);
    let dc = 0.0;
    for (let t = 0; t < len; t++) {
      const k = kMin + t;
      const val = h[k * up + p + halfLen];
      w[t] = val;
      dc += val;
    }
    const invDc = dc !== 0 ? 1.0 / dc : 1.0;
    for (let t = 0; t < len; t++) w[t] *= invDc;
    branches[p] = { kMin, kMax, len, w };
    maxAbsK = Math.max(maxAbsK, Math.abs(kMin), Math.abs(kMax));
  }

  const spec = { up, down, branches, maxAbsK };
  POLY_CACHE.set(key, spec);
  return spec;
}

export function resampleZeroPhasePoly1D(x, up, down, targetLen, tapsPerStep = 16, beta = 8.6) {
  const nIn = x.length;
  const nOut = targetLen || Math.round((nIn * up) / down);
  const out = new Float32Array(nOut);
  if (nIn === 0 || nOut === 0) return out;
  if (up === 1 && down === 1) {
    out.set(x.subarray(0, Math.min(nIn, nOut)));
    return out;
  }

  const { branches, maxAbsK } = getPolyphaseKernel(up, down, tapsPerStep, beta);
  const safeLo = maxAbsK;
  const safeHi = nIn - 1 - maxAbsK;

  for (let n = 0; n < nOut; n++) {
    const num = n * down;
    const q = Math.floor(num / up);
    const p = num - q * up;
    const br = branches[p];
    const w = br.w;
    const len = br.len;
    const baseIdx = q - br.kMin;
    let acc = 0.0;

    if (q >= safeLo && q <= safeHi) {
      for (let t = 0; t < len; t++) {
        acc += x[baseIdx - t] * w[t];
      }
    } else {
      for (let t = 0; t < len; t++) {
        let idx = baseIdx - t;
        if (idx < 0) idx = 0;
        else if (idx >= nIn) idx = nIn - 1;
        acc += x[idx] * w[t];
      }
    }
    out[n] = acc;
  }
  return out;
}

export function resampleZeroPhasePolyStereo(inL, inR, up, down, targetLen, tapsPerStep = 16, beta = 8.6) {
  return [
    resampleZeroPhasePoly1D(inL, up, down, targetLen, tapsPerStep, beta),
    resampleZeroPhasePoly1D(inR, up, down, targetLen, tapsPerStep, beta),
  ];
}

export function designSymmetricLowpassHalf(cutoffCycles, halfOrder = 64, beta = 8.0) {
  const wHalf = new Float32Array(halfOrder + 1);
  const invI0 = 1.0 / besselI0(beta);
  let dc = 0.0;
  for (let m = 0; m <= halfOrder; m++) {
    const r = m / halfOrder;
    const win = besselI0(beta * Math.sqrt(Math.max(0.0, 1.0 - r * r))) * invI0;
    const val = 2.0 * cutoffCycles * sincNorm(2.0 * cutoffCycles * m) * win;
    wHalf[m] = val;
    dc += m === 0 ? val : 2.0 * val;
  }
  const invDc = 1.0 / (dc || 1.0);
  for (let m = 0; m <= halfOrder; m++) wHalf[m] *= invDc;
  return wHalf;
}

export function designSymmetricBandpassHalf(fLoHz, fHiHz, sampleRate, halfOrder = 32, beta = 7.2) {
  const lpHi = designSymmetricLowpassHalf(fHiHz / sampleRate, halfOrder, beta);
  const lpLo = designSymmetricLowpassHalf(fLoHz / sampleRate, halfOrder, beta);
  const bp = new Float32Array(halfOrder + 1);
  for (let m = 0; m <= halfOrder; m++) bp[m] = lpHi[m] - lpLo[m];
  return bp;
}

export function applyZeroPhaseSymmetricFir(x, wHalf) {
  const n = x.length;
  const halfOrder = wHalf.length - 1;
  const out = new Float32Array(n);
  const w0 = wHalf[0];
  const safeEnd = n - halfOrder;

  for (let i = 0; i < n; i++) {
    let acc = x[i] * w0;
    if (i >= halfOrder && i < safeEnd) {
      for (let m = 1; m <= halfOrder; m++) {
        acc += wHalf[m] * (x[i - m] + x[i + m]);
      }
    } else {
      for (let m = 1; m <= halfOrder; m++) {
        const iL = i - m < 0 ? 0 : i - m;
        const iR = i + m >= n ? n - 1 : i + m;
        acc += wHalf[m] * (x[iL] + x[iR]);
      }
    }
    out[i] = acc;
  }
  return out;
}

// Fast Precomputed 4096-Phase 12-Tap Windowed-Sinc Lookup Table (eliminates 160M Math.sin/cos calls!)
const SINC_HALF_TAPS = 6;
const SINC_PHASES = 4096;
const SINC_TAPS = 2 * SINC_HALF_TAPS; // k = -5 .. +6
const SINC_LUT = (() => {
  const tbl = new Float32Array(SINC_PHASES * SINC_TAPS);
  for (let p = 0; p < SINC_PHASES; p++) {
    const fr = p / SINC_PHASES;
    const base = p * SINC_TAPS;
    let sum = 0.0;
    for (let t = 0; t < SINC_TAPS; t++) {
      const k = t - SINC_HALF_TAPS + 1;
      const d = fr - k;
      const ad = d < 0 ? -d : d;
      let val = 0.0;
      if (ad < SINC_HALF_TAPS) {
        const win = 0.5 + 0.5 * Math.cos((Math.PI * d) / SINC_HALF_TAPS);
        const s = ad < 1e-9 ? 1.0 : Math.sin(Math.PI * d) / (Math.PI * d);
        val = s * win;
      }
      tbl[base + t] = val;
      sum += val;
    }
    const inv = sum !== 0 ? 1.0 / sum : 1.0;
    for (let t = 0; t < SINC_TAPS; t++) tbl[base + t] *= inv;
  }
  return tbl;
})();

export function sincResampleSegmentAdd(grain, out, outStart, segLen) {
  const P = grain.length;
  if (segLen <= 0 || outStart >= out.length) return;
  const scale = P / segLen;
  const iEnd = Math.min(segLen, out.length - outStart);
  const pMax = P - 1;

  for (let j = 0; j < iEnd; j++) {
    const pos = (j + 0.5) * scale - 0.5;
    const ic = Math.floor(pos);
    const ph = Math.min(SINC_PHASES - 1, ((pos - ic) * SINC_PHASES) | 0);
    const lutOff = ph * SINC_TAPS;
    let sum = 0.0;

    if (ic >= SINC_HALF_TAPS - 1 && ic + SINC_HALF_TAPS <= pMax) {
      const gBase = ic - SINC_HALF_TAPS + 1;
      for (let t = 0; t < SINC_TAPS; t++) {
        sum += grain[gBase + t] * SINC_LUT[lutOff + t];
      }
    } else {
      for (let t = 0; t < SINC_TAPS; t++) {
        let idx = ic - SINC_HALF_TAPS + 1 + t;
        if (idx < 0) idx = 0;
        else if (idx > pMax) idx = pMax;
        sum += grain[idx] * SINC_LUT[lutOff + t];
      }
    }
    out[outStart + j] += sum;
  }
}
