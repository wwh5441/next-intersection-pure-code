// CabronWEC (CWC-4X) Pure Mathematical Transform & Psychoacoustic Codec Core
// Zero DOM / Zero Node dependencies — safe inside AudioWorkletGlobalScope, Web Workers, Browsers, and Node.js.

export const CWC_SAMPLE_RATE = 44100;
export const CWC_FRAME_N = 1024;
export const CWC_HOP_M = 512;
export const CWC_BINS = 512;
export const CWC_BPM = 98.0;
export const CWC_BAR_SEC = (4 * 60) / CWC_BPM; // 2.44897959s (108,000 samples)

// Alphabet for 100% escape-free readable JS string representation (no quotes, backslash, '/', '|')
export const POS_CH = '123456789abcdefghijklmnopqrstuv'; // +1 .. +31
export const NEG_CH = 'wxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!'; // -1 .. -31
export const Z_CH = '.#$%&()*,:;<=>?@_~';                // 1 .. 18 consecutive zeros
export const B62 = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';

// 24 Bark Critical Band edges (Hz)
export const BARK_EDGES_HZ = [
  0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
  1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 22050,
];

// 9 Mid Scale-Factor Bands (0..240 bins = 0..10,336 Hz) with orthogonal 5-layer stem ownership
// Layer mapping:
//   0 -> L01 Sub & Kick/Bass (0..215 Hz)
//   1 -> L02 Rhythm & Chord Body (215..947 Hz)
//   2 -> L03 Vocal Formants & Lead (947..4,478 Hz)
//   3 -> L04 Brilliance & Velvet Air (4,478..19,595 Hz, Mid bands + High PNS)
//   4 -> L05 Stereo Field & Room Space (Side channel 43..16,795 Hz)
export const MID_BANDS = [
  { lo: 0,   hi: 5,   qMax: 90, layer: 0, label: '0–215 Hz' },
  { lo: 5,   hi: 12,  qMax: 62, layer: 1, label: '215–516 Hz' },
  { lo: 12,  hi: 22,  qMax: 44, layer: 1, label: '516–947 Hz' },
  { lo: 22,  hi: 36,  qMax: 31, layer: 2, label: '947–1,550 Hz' },
  { lo: 36,  hi: 56,  qMax: 24, layer: 2, label: '1,550–2,411 Hz' },
  { lo: 56,  hi: 84,  qMax: 16, layer: 2, label: '2,411–3,617 Hz' },
  { lo: 84,  hi: 104, qMax: 14, layer: 2, label: '3,617–4,478 Hz' },
  { lo: 104, hi: 168, qMax: 11, layer: 3, label: '4,478–7,235 Hz' },
  { lo: 168, hi: 240, qMax: 8,  layer: 3, label: '7,235–10,336 Hz' },
];

export const SIDE_BANDS = [
  { lo: 1,   hi: 12,  qMax: 24, layer: 4 },
  { lo: 12,  hi: 36,  qMax: 16, layer: 4 },
  { lo: 36,  hi: 84,  qMax: 12, layer: 4 },
  { lo: 84,  hi: 168, qMax: 8,  layer: 4 },
];

export const PNS_MID_BANDS = [
  [240, 300],
  [300, 375],
  [375, 455],
];

export const PNS_SIDE_BANDS = [
  [168, 260],
  [260, 390],
];

export const CWC_STEMS = [
  { id: 'sub',   layer: 0, code: 'L01', name: 'Sub & Kick / Bass',       band: '0 – 215 Hz (Mid)',         color: '#06B6D4' },
  { id: 'body',  layer: 1, code: 'L02', name: 'Rhythm & Chord Body',     band: '215 – 947 Hz (Mid)',       color: '#38BDF8' },
  { id: 'vocal', layer: 2, code: 'L03', name: 'Vocal Formants & Lead',   band: '947 – 4,478 Hz (Mid)',     color: '#F59E0B' },
  { id: 'air',   layer: 3, code: 'L04', name: 'Brilliance & Velvet Air', band: '4.48 – 19.6 kHz (Mid+PNS)', color: '#A78BFA' },
  { id: 'space', layer: 4, code: 'L05', name: 'Stereo Field & Room',     band: '43 – 16.8 kHz (Side)',     color: '#34D399' },
];

// Precomputed fast lookup tables for branchless token unpacking
export const TOK_TYPE = new Int16Array(128);
export const Z_LEN = new Uint8Array(128);
export const B62_INV = new Int8Array(128).fill(-1);
export const GAIN_LUT = new Float64Array(62);
export const PNS_LUT = new Float64Array(36);

for (let i = 0; i < POS_CH.length; i++) TOK_TYPE[POS_CH.charCodeAt(i)] = i + 1;
for (let i = 0; i < NEG_CH.length; i++) TOK_TYPE[NEG_CH.charCodeAt(i)] = -(i + 1);
for (let i = 0; i < Z_CH.length; i++) Z_LEN[Z_CH.charCodeAt(i)] = i + 1;
for (let i = 0; i < B62.length; i++) B62_INV[B62.charCodeAt(i)] = i;
for (let q = 1; q < 62; q++) GAIN_LUT[q] = Math.pow(10, (q * 1.45 - 34) / 20);
for (let q = 1; q < 36; q++) PNS_LUT[q] = Math.pow(10, (q * 2.2 - 48) / 20);

export function packQuantizedVec(arr) {
  let out = '';
  let z = 0;
  const flush = () => {
    while (z > 0) {
      const t = Math.min(18, z);
      out += Z_CH[t - 1];
      z -= t;
    }
  };
  for (let i = 0; i < arr.length; i++) {
    const q = arr[i];
    if (q === 0) {
      z++;
      continue;
    }
    flush();
    if (q >= 1 && q <= 31) out += POS_CH[q - 1];
    else if (q <= -1 && q >= -31) out += NEG_CH[-q - 1];
    else if (q >= 32 && q <= 93) out += '+' + B62[q - 32];
    else if (q <= -32 && q >= -93) out += '-' + B62[-q - 32];
    else throw new RangeError(`CWC token out of bounds [-93..93]: ${q}`);
  }
  return out;
}

export function unpackQuantizedVec(str, start, end, outArr) {
  outArr.fill(0);
  let idx = 0;
  const maxLen = outArr.length;
  for (let p = start; p < end && idx < maxLen; p++) {
    const c = str.charCodeAt(p);
    const zl = c < 128 ? Z_LEN[c] : 0;
    if (zl > 0) {
      idx += zl;
      continue;
    }
    if (c === 43) {
      outArr[idx++] = 32 + B62_INV[str.charCodeAt(++p)];
      continue;
    }
    if (c === 45) {
      outArr[idx++] = -(32 + B62_INV[str.charCodeAt(++p)]);
      continue;
    }
    outArr[idx++] = c < 128 ? TOK_TYPE[c] : 0;
  }
}

/**
 * Exact O(N log N) Princen-Bradley TDAC MDCT / IMDCT via N/4 complex FFT
 * Uses the Vorbis slope-continuous sine window for >65 dB stopband rejection.
 */
export function createMDCTProcessor(N = CWC_FRAME_N) {
  const M = N >> 1;
  const Q = N >> 2;
  const win = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const s = Math.sin((Math.PI * (n + 0.5)) / N);
    win[n] = Math.sin(0.5 * Math.PI * s * s);
  }

  const bitRev = new Uint32Array(Q);
  const logQ = Math.log2(Q) | 0;
  for (let i = 0; i < Q; i++) {
    let x = i;
    let r = 0;
    for (let b = 0; b < logQ; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    bitRev[i] = r;
  }

  const twRe = new Float64Array(Q >> 1);
  const twIm = new Float64Array(Q >> 1);
  for (let i = 0; i < (Q >> 1); i++) {
    const a = (-2 * Math.PI * i) / Q;
    twRe[i] = Math.cos(a);
    twIm[i] = Math.sin(a);
  }

  const cPre = new Float64Array(Q);
  const sPre = new Float64Array(Q);
  const cPost = new Float64Array(Q);
  const sPost = new Float64Array(Q);
  for (let m = 0; m < Q; m++) {
    const a = (-Math.PI * (2 * m + 0.5)) / N;
    cPre[m] = Math.cos(a);
    sPre[m] = Math.sin(a);
  }
  for (let r = 0; r < Q; r++) {
    const a = (-2 * Math.PI * (r + 0.125)) / N;
    cPost[r] = Math.cos(a);
    sPost[r] = Math.sin(a);
  }

  const re = new Float64Array(Q);
  const im = new Float64Array(Q);
  const wBuf = new Float64Array(N);
  const folded = new Float64Array(M);

  function dct4(x, out, scale) {
    for (let m = 0; m < Q; m++) {
      const xr = x[2 * m];
      const xi = x[M - 1 - 2 * m];
      re[m] = xr * cPre[m] - xi * sPre[m];
      im[m] = xr * sPre[m] + xi * cPre[m];
    }
    for (let i = 0; i < Q; i++) {
      const j = bitRev[i];
      if (i < j) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    for (let size = 2; size <= Q; size <<= 1) {
      const half = size >> 1;
      const step = Q / size;
      for (let i = 0; i < Q; i += size) {
        for (let k = 0; k < half; k++) {
          const wr = twRe[k * step];
          const wi = twIm[k * step];
          const u = i + k;
          const v = u + half;
          const tr = re[v] * wr - im[v] * wi;
          const ti = re[v] * wi + im[v] * wr;
          re[v] = re[u] - tr;
          im[v] = im[u] - ti;
          re[u] += tr;
          im[u] += ti;
        }
      }
    }
    for (let r = 0; r < Q; r++) {
      const yr = re[r];
      const yi = im[r];
      const c = cPost[r];
      const s = sPost[r];
      out[2 * r] = (yr * c - yi * s) * scale;
      out[M - 1 - 2 * r] = -(yr * s + yi * c) * scale;
    }
  }

  return {
    N,
    M,
    win,
    forward(sig, offset, specOut) {
      const len = sig.length;
      for (let n = 0; n < N; n++) {
        const idx = offset + n;
        wBuf[n] = (idx >= 0 && idx < len ? sig[idx] : 0) * win[n];
      }
      for (let i = 0; i < Q; i++) {
        folded[i] = -wBuf[M + Q - 1 - i] - wBuf[M + Q + i];
        folded[Q + i] = wBuf[i] - wBuf[M - 1 - i];
      }
      dct4(folded, specOut, 1.0);
    },
    inverse(specIn, outN) {
      dct4(specIn, folded, 2.0 / M);
      for (let i = 0; i < Q; i++) {
        const a = folded[i];
        const b = folded[Q + i];
        outN[i] = b * win[i];
        outN[M - 1 - i] = -b * win[M - 1 - i];
        outN[M + Q - 1 - i] = -a * win[M + Q - 1 - i];
        outN[M + Q + i] = -a * win[M + Q + i];
      }
    },
  };
}
