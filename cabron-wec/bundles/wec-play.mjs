#!/usr/bin/env node
// CabronWEC (CWC-5X) Standalone Zero-Dependency .wec Player & WAV Renderer (wec-play.mjs)
// Usage:
//   node wec-play.mjs inspect <song.wec>
//   node wec-play.mjs verify  <song.wec>
//   node wec-play.mjs render  <song.wec> [output.wav] [--solo=0..4] [--mute=0..4]
const CWC_SAMPLE_RATE = 44100;
const CWC_FRAME_N = 4096;
const CWC_HOP_M = 2048; // 50% overlap hop (~46.44 ms long / 8 x 5.80 ms short)
const CWC_SHORT_N = 512;
const CWC_SHORT_M = 256;
const CWC_SHORT_K = CWC_HOP_M / CWC_SHORT_M; // 8 short sub-blocks per frame
const CWC_BPM = 96.0;
const CWC_BAR_SEC = (60 / CWC_BPM) * 4; // 2.500s per 4/4 bar
const WIN_LONG = 0;
const WIN_START = 1;
const WIN_SHORT = 2;
const WIN_STOP = 3;
const WIN_NAMES = ['LONG_4096', 'START_BRIDGE', 'SHORT_8x512', 'STOP_BRIDGE'];
const BARK_EDGES_HZ = [
0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500,
];
const CWC_STEMS = [
{ id: 'sub', code: 'L01', name: 'Sub & Bass Groove', band: '0 – 517 Hz', color: '#06B6D4' },
{ id: 'body', code: 'L02', name: 'Rhythm Guitar & Warmth', band: '517 – 1,550 Hz', color: '#38BDF8' },
{ id: 'vocal', code: 'L03', name: 'Lead Vocal & Formants', band: '1.55 – 4.48 kHz', color: '#F59E0B' },
{ id: 'air', code: 'L04', name: 'Harmonic Air & Velvet', band: '4.48 – 19.6 kHz', color: '#A78BFA' },
{ id: 'side', code: 'L05', name: 'Stereo Side Space', band: 'M/S Pan + Room', color: '#10B981' },
];
const MID_BANDS = [
{ lo: 0, hi: 24, layer: 0, qMax: 35, group: 0, label: 'Sub 0–258Hz' },
{ lo: 24, hi: 48, layer: 0, qMax: 33, group: 0, label: 'Bass 258–517Hz' },
{ lo: 48, hi: 96, layer: 1, qMax: 26, group: 0, label: 'LowMid 517–1034Hz' },
{ lo: 96, hi: 144, layer: 1, qMax: 24, group: 1, label: 'Body 1.03–1.55kHz' },
{ lo: 144, hi: 208, layer: 2, qMax: 19, group: 1, label: 'Presence 1.55–2.24kHz' },
{ lo: 208, hi: 296, layer: 2, qMax: 18, group: 1, label: 'Formant 2.24–3.19kHz' },
{ lo: 296, hi: 409, layer: 2, qMax: 14, group: 1, label: 'Clarity 3.19–4.40kHz' },
{ lo: 409, hi: 594, layer: 3, qMax: 9, group: 2, label: 'Brilliance 4.40–6.40kHz' },
{ lo: 594, hi: 882, layer: 3, qMax: 6, group: 2, label: 'Sibilance 6.40–9.50kHz' },
{ lo: 882, hi: 1115, layer: 3, qMax: 5, group: 2, label: 'CrispAir 9.50–12.00kHz' },
];
const PNS_MID_BANDS = [
[1115, 1440],
[1440, 1820],
];
const SIDE_BANDS = [
{ lo: 4, hi: 48, layer: 4, qMax: 18, label: 'SideLow 43–517Hz' },
{ lo: 48, hi: 144, layer: 4, qMax: 14, label: 'SideBody 517–1550Hz' },
{ lo: 144, hi: 368, layer: 4, qMax: 9, label: 'SideVocal 1.55–3.96kHz' },
{ lo: 368, hi: 672, layer: 4, qMax: 5, label: 'SideAir 3.96–7.24kHz' },
{ lo: 672, hi: 1115, layer: 4, qMax: 0, label: 'SidePan 7.24–12.00kHz' },
];
const PNS_SIDE_BANDS = [
[1115, 1560],
];
const CWC_NUM_SCALES = 23;
const CWC_NUM_PANS = 5;
const GAIN_LUT = new Float64Array(211);
for (let q = 1; q <= 210; q++) {
GAIN_LUT[q] = Math.pow(10, (q * 0.5 - 64) / 20);
}
const WHISPER_LUT = new Float64Array(46);
for (let q = 1; q <= 45; q++) {
WHISPER_LUT[q] = Math.pow(10, (q * 1.4 - 82) / 20);
}
const PNS_LUT = new Float64Array(221);
for (let q = 1; q <= 220; q++) {
PNS_LUT[q] = Math.pow(10, (q * 0.44 - 92) / 20);
}
const BIN_TO_MID_BAND = new Int8Array(CWC_HOP_M).fill(-1);
for (let bi = 0; bi < MID_BANDS.length; bi++) {
for (let k = MID_BANDS[bi].lo; k < MID_BANDS[bi].hi; k++) BIN_TO_MID_BAND[k] = bi;
}
const BIN_TO_SIDE_BAND = new Int8Array(CWC_HOP_M).fill(-1);
for (let bi = 0; bi < 4; bi++) {
for (let k = SIDE_BANDS[bi].lo; k < SIDE_BANDS[bi].hi; k++) BIN_TO_SIDE_BAND[k] = bi;
}
function createOrthonormalDCT4(M) {
const H = M >> 1;
const scale = Math.sqrt(2.0 / M);
const twCos = new Float64Array(H);
const twSin = new Float64Array(H);
for (let k = 0; k < H; k++) {
const angle = (Math.PI * (2 * k + 0.5)) / (2 * M);
twCos[k] = Math.cos(angle);
twSin[k] = Math.sin(angle);
}
const bitRev = new Int32Array(H);
const bits = Math.round(Math.log2(H));
for (let i = 0; i < H; i++) {
let r = 0;
let x = i;
for (let b = 0; b < bits; b++) {
r = (r << 1) | (x & 1);
x >>= 1;
}
bitRev[i] = r;
}
const fftCos = new Float64Array(H);
const fftSin = new Float64Array(H);
for (let i = 0; i < H; i++) {
const angle = (-2 * Math.PI * i) / H;
fftCos[i] = Math.cos(angle);
fftSin[i] = Math.sin(angle);
}
const re = new Float64Array(H);
const im = new Float64Array(H);
return function dct4(u, out) {
for (let n = 0; n < H; n++) {
const x0 = u[2 * n];
const x1 = u[M - 1 - 2 * n];
const j = bitRev[n];
re[j] = x0 * twCos[n] + x1 * twSin[n];
im[j] = x1 * twCos[n] - x0 * twSin[n];
}
for (let len = 2; len <= H; len <<= 1) {
const half = len >> 1;
const step = H / len;
for (let i = 0; i < H; i += len) {
for (let j = 0; j < half; j++) {
const tw = j * step;
const wr = fftCos[tw];
const wi = fftSin[tw];
const tr = re[i + j + half] * wr - im[i + j + half] * wi;
const ti = re[i + j + half] * wi + im[i + j + half] * wr;
re[i + j + half] = re[i + j] - tr;
im[i + j + half] = im[i + j] - ti;
re[i + j] += tr;
im[i + j] += ti;
}
}
}
for (let k = 0; k < H; k++) {
const r = re[k];
const m = im[k];
out[2 * k] = (r * twCos[k] + m * twSin[k]) * scale;
out[M - 1 - 2 * k] = (r * twSin[k] - m * twCos[k]) * scale;
}
};
}
function createAdaptiveMDCTProcessor(NL = CWC_FRAME_N, NS = CWC_SHORT_N) {
const ML = NL >> 1;
const MS = NS >> 1;
const K = ML / MS;
const pad = (ML - MS) >> 1;
const dctL = createOrthonormalDCT4(ML);
const dctS = createOrthonormalDCT4(MS);
const winLong = new Float64Array(NL);
const winStart = new Float64Array(NL);
const winStop = new Float64Array(NL);
const winShort = new Float64Array(NS);
for (let n = 0; n < NL; n++) winLong[n] = Math.sin((Math.PI / NL) * (n + 0.5));
for (let n = 0; n < NS; n++) winShort[n] = Math.sin((Math.PI / NS) * (n + 0.5));
for (let n = 0; n < ML; n++) winStart[n] = winLong[n];
for (let n = ML; n < ML + pad; n++) winStart[n] = 1.0;
for (let n = 0; n < MS; n++) winStart[ML + pad + n] = winShort[MS + n];
for (let n = 0; n < MS; n++) winStop[pad + n] = winShort[n];
for (let n = pad + MS; n < ML; n++) winStop[n] = 1.0;
for (let n = ML; n < NL; n++) winStop[n] = winLong[n];
const wins = [winLong, winStart, null, winStop];
const foldL = new Float64Array(ML);
const foldS = new Float64Array(MS);
const subSpec = new Float64Array(MS);
const subTime = new Float64Array(NS);
function forwardSingle(pcm, offset, w, M, dct, fold, out) {
const H = M >> 1;
const len = pcm.length;
for (let n = 0; n < H; n++) {
const i3 = offset + M + H + n;
const i2 = offset + M + H - 1 - n;
const x3 = (i3 >= 0 && i3 < len ? pcm[i3] : 0) * w[M + H + n];
const x2 = (i2 >= 0 && i2 < len ? pcm[i2] : 0) * w[M + H - 1 - n];
fold[n] = -x3 - x2;
const i0 = offset + n;
const i1 = offset + M - 1 - n;
const x0 = (i0 >= 0 && i0 < len ? pcm[i0] : 0) * w[n];
const x1 = (i1 >= 0 && i1 < len ? pcm[i1] : 0) * w[M - 1 - n];
fold[H + n] = x0 - x1;
}
dct(fold, out);
}
function inverseSingle(spec, w, M, dct, fold, out) {
const H = M >> 1;
dct(spec, fold);
for (let n = 0; n < H; n++) {
out[n] = fold[H + n] * w[n];
out[M - 1 - n] = -fold[H + n] * w[M - 1 - n];
out[M + H + n] = -fold[n] * w[M + H + n];
out[M + H - 1 - n] = -fold[n] * w[M + H - 1 - n];
}
}
return {
NL,
ML,
NS,
MS,
K,
pad,
forward(pcm, offset, winType = WIN_LONG, specOut) {
if (winType !== WIN_SHORT) {
forwardSingle(pcm, offset, wins[winType], ML, dctL, foldL, specOut);
} else {
for (let s = 0; s < K; s++) {
forwardSingle(pcm, offset + pad + s * MS, winShort, MS, dctS, foldS, subSpec);
for (let j = 0; j < MS; j++) {
specOut[j * K + s] = subSpec[j];
}
}
}
},
inverse(spec, winType = WIN_LONG, timeOut) {
if (winType !== WIN_SHORT) {
inverseSingle(spec, wins[winType], ML, dctL, foldL, timeOut);
} else {
timeOut.fill(0);
for (let s = 0; s < K; s++) {
for (let j = 0; j < MS; j++) {
subSpec[j] = spec[j * K + s];
}
inverseSingle(subSpec, winShort, MS, dctS, foldS, subTime);
const base = pad + s * MS;
for (let n = 0; n < NS; n++) {
timeOut[base + n] += subTime[n];
}
}
}
},
};
}
const createMDCTProcessor = createAdaptiveMDCTProcessor;
const RC_TOP = 1 << 24;
const RC_BOT = 1 << 16;
const TOT_BITS = 14;
const TOT = 1 << TOT_BITS;
function rangeDecodeStream(buf, offset = 0) {
let pos = offset;
const len =
((buf[pos] << 24) | (buf[pos + 1] << 16) | (buf[pos + 2] << 8) | buf[pos + 3]) >>> 0;
pos += 4;
if (len === 0) return { data: new Uint8Array(0), nextOffset: pos };
const alphaSize = buf[pos++] + 1;
const maskBytes = Math.ceil(alphaSize / 8);
const active = new Uint8Array(alphaSize);
for (let mb = 0; mb < maskBytes; mb++) {
const m = buf[pos++];
for (let b = 0; b < 8; b++) {
if (m & (1 << b) && mb * 8 + b < alphaSize) active[mb * 8 + b] = 1;
}
}
const freq = new Uint16Array(alphaSize);
const cum = new Uint16Array(alphaSize + 1);
const lut = new Uint8Array(TOT);
for (let s = 0; s < alphaSize; s++) {
if (active[s]) {
const b0 = buf[pos++];
const v = b0 & 128 ? (b0 & 127) | (buf[pos++] << 7) : b0;
freq[s] = v + 1;
}
cum[s + 1] = cum[s] + freq[s];
lut.fill(s, cum[s], cum[s + 1]);
}
let low = 0;
let range = 0xffffffff;
let code = 0;
for (let i = 0; i < 4; i++) {
code = ((code << 8) | (buf[pos++] || 0)) >>> 0;
}
const out = new Uint8Array(len);
for (let i = 0; i < len; i++) {
const r = (range >>> TOT_BITS) >>> 0;
const count = Math.min(TOT - 1, (((code - low) >>> 0) / r) | 0);
const s = lut[count];
out[i] = s;
low = (low + ((r * cum[s]) >>> 0)) >>> 0;
range = (r * freq[s]) >>> 0;
while (
((low ^ ((low + range) >>> 0)) >>> 0) < RC_TOP ||
(range < RC_BOT && ((range = (-low & (RC_BOT - 1)) >>> 0), true))
) {
code = ((code << 8) | (buf[pos++] || 0)) >>> 0;
range = (range << 8) >>> 0;
low = (low << 8) >>> 0;
}
}
return { data: out, nextOffset: pos };
}
const WEC_MAGIC = 'CWC5';
const WEC_HEADER_BYTES = 32;
const WEC_VERSION = 1;
const CRC32_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
let c = i;
for (let k = 0; k < 8; k++) {
c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
}
CRC32_TABLE[i] = c >>> 0;
}
function computeCrc32(bytes, start = 0, end = bytes.length) {
let crc = 0xffffffff;
for (let i = start; i < end; i++) {
crc = CRC32_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
}
return (crc ^ 0xffffffff) >>> 0;
}
function decodeFrameHeaders(buf, startOff, F, activeScaleMask) {
let off = startOff;
const next = () => {
const r = rangeDecodeStream(buf, off);
off = r.nextOffset;
return r.data;
};
const winTypes = next();
const actByBand = new Array(CWC_NUM_SCALES).fill(null);
for (let i = 0; i < CWC_NUM_SCALES; i++) {
if (activeScaleMask & (1 << i)) actByBand[i] = next();
}
const scDeltasTonal = next();
const scDeltasEnv = next();
const panDeltas = new Array(CWC_NUM_PANS);
for (let b = 0; b < CWC_NUM_PANS; b++) panDeltas[b] = next();
const scales = new Uint8Array(F * CWC_NUM_SCALES);
const lastNz = new Int32Array(CWC_NUM_SCALES).fill(130);
let tPos = 0;
let ePos = 0;
for (let f = 0; f < F; f++) {
const base = f * CWC_NUM_SCALES;
let prevBandDelta = 0;
for (let i = 0; i < CWC_NUM_SCALES; i++) {
const act = actByBand[i];
if (!act || act[f] === 0) {
prevBandDelta = 0;
continue;
}
const isEnv = (i >= 10 && i < 12) || i >= 16;
const d = isEnv ? scDeltasEnv[ePos++] : scDeltasTonal[tPos++];
const pred = Math.max(1, Math.min(254, lastNz[i] + (prevBandDelta >> 1)));
const c = (pred + d) & 255;
scales[base + i] = c;
prevBandDelta = c - lastNz[i];
lastNz[i] = c;
}
}
const pans = new Int8Array(F * CWC_NUM_PANS);
for (let b = 0; b < CWC_NUM_PANS; b++) {
let prev = 16;
const pArr = panDeltas[b];
for (let f = 0; f < F; f++) {
prev = (prev + pArr[f]) & 63;
pans[f * CWC_NUM_PANS + b] = prev - 16;
}
}
return { winTypes, scales, pans, nextOffset: off };
}
function decodeAtomsMode0(buf, startOff, F, winTypes, scales) {
let off = startOff;
const next = () => {
const r = rangeDecodeStream(buf, off);
off = r.nextOffset;
return r.data;
};
const binCtx = new Array(48);
for (let i = 0; i < 48; i++) binCtx[i] = next();
const sBinCtx = new Array(12);
for (let i = 0; i < 12; i++) sBinCtx[i] = next();
const extMag = next();
const signLen =
((buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]) >>> 0;
off += 4;
const signBytes = buf.subarray(off, off + signLen);
const maxNz = signLen * 8 + 16;
const tmpMidIdx = new Uint16Array(maxNz);
const tmpMidQ = new Int8Array(maxNz);
const tmpSideIdx = new Uint16Array(maxNz);
const tmpSideQ = new Int8Array(maxNz);
const midOff = new Uint32Array(F + 1);
const sideOff = new Uint32Array(F + 1);
const ctxPos = new Int32Array(48);
const sCtxPos = new Int32Array(12);
let extMagPos = 0;
let signBitIdx = 0;
let mWrite = 0;
let sWrite = 0;
const popSign = () => {
const neg = (signBytes[signBitIdx >>> 3] >>> (signBitIdx & 7)) & 1;
signBitIdx++;
return neg === 1;
};
const prevM = new Uint8Array(2048);
const curM = new Uint8Array(2048);
const prevS = new Uint8Array(2048);
const curS = new Uint8Array(2048);
for (let f = 0; f < F; f++) {
midOff[f] = mWrite;
sideOff[f] = sWrite;
curM.fill(0);
curS.fill(0);
const scBase = f * CWC_NUM_SCALES;
const sameWin = f > 0 && winTypes[f] === 0 && winTypes[f - 1] === 0;
for (let bi = 0; bi < MID_BANDS.length; bi++) {
const scVal = scales[scBase + bi];
if (scVal === 0 || scVal > 210) continue;
const bd = MID_BANDS[bi];
const grp = bi < 2 ? 0 : bi < 4 ? 1 : bi < 6 ? 2 : 3;
let adj = 0;
for (let k = bd.lo; k < bd.hi; k++) {
const pm = sameWin ? prevM[k] : 0;
const pCls = pm === 0 ? 0 : pm <= 2 ? 1 : 2;
const nei =
sameWin && ((k > 0 && prevM[k - 1] > 0) || (k < 2047 && prevM[k + 1] > 0)) ? 1 : 0;
const ctx = grp * 12 + pCls * 4 + adj * 2 + nei;
const sym = binCtx[ctx][ctxPos[ctx]++];
if (sym === 0) {
adj = 0;
} else {
const absQ = sym === 8 ? 8 + extMag[extMagPos++] : sym;
const neg = popSign();
curM[k] = absQ;
tmpMidIdx[mWrite] = k;
tmpMidQ[mWrite] = neg ? -absQ : absQ;
mWrite++;
adj = 1;
}
}
}
prevM.set(curM);
for (let bi = 0; bi < 4; bi++) {
if (scales[scBase + 12 + bi] === 0) continue;
const bd = SIDE_BANDS[bi];
let adj = 0;
for (let k = bd.lo; k < bd.hi; k++) {
const pm = sameWin ? prevS[k] : 0;
const pCls = pm === 0 ? 0 : pm <= 2 ? 1 : 2;
const nei =
sameWin && ((k > 0 && prevS[k - 1] > 0) || (k < 2047 && prevS[k + 1] > 0)) ? 1 : 0;
const ctx = pCls * 4 + adj * 2 + nei;
const sym = sBinCtx[ctx][sCtxPos[ctx]++];
if (sym === 0) {
adj = 0;
} else {
const absQ = sym === 8 ? 8 + extMag[extMagPos++] : sym;
const neg = popSign();
curS[k] = absQ;
tmpSideIdx[sWrite] = k;
tmpSideQ[sWrite] = neg ? -absQ : absQ;
sWrite++;
adj = 1;
}
}
}
prevS.set(curS);
}
midOff[F] = mWrite;
sideOff[F] = sWrite;
return {
midOff,
midIdx: tmpMidIdx.slice(0, mWrite),
midQ: tmpMidQ.slice(0, mWrite),
sideOff,
sideIdx: tmpSideIdx.slice(0, sWrite),
sideQ: tmpSideQ.slice(0, sWrite),
};
}
function decodeAtomsMode1(buf, startOff, F, winTypes, scales) {
let off = startOff;
const next = () => {
const r = rangeDecodeStream(buf, off);
off = r.nextOffset;
return r.data;
};
const tokCtx = new Array(16);
for (let i = 0; i < 16; i++) tokCtx[i] = next();
const sTokCtx = new Array(4);
for (let i = 0; i < 4; i++) sTokCtx[i] = next();
const extRun = next();
const extMag = next();
const signLen =
((buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3]) >>> 0;
off += 4;
const signBytes = buf.subarray(off, off + signLen);
const maxNz = signLen * 8 + 16;
const tmpMidIdx = new Uint16Array(maxNz);
const tmpMidQ = new Int8Array(maxNz);
const tmpSideIdx = new Uint16Array(maxNz);
const tmpSideQ = new Int8Array(maxNz);
const midOff = new Uint32Array(F + 1);
const sideOff = new Uint32Array(F + 1);
const tokPos = new Int32Array(16);
const sTokPos = new Int32Array(4);
let extRunPos = 0;
let extMagPos = 0;
let signBitIdx = 0;
let mWrite = 0;
let sWrite = 0;
const popSign = () => {
const neg = (signBytes[signBitIdx >>> 3] >>> (signBitIdx & 7)) & 1;
signBitIdx++;
return neg === 1;
};
const prevMidMag = new Uint8Array(2048);
const curMidMag = new Uint8Array(2048);
for (let f = 0; f < F; f++) {
midOff[f] = mWrite;
sideOff[f] = sWrite;
curMidMag.fill(0);
const scBase = f * CWC_NUM_SCALES;
const sameWin = f > 0 && winTypes[f] === 0 && winTypes[f - 1] === 0;
for (let bi = 0; bi < MID_BANDS.length; bi++) {
const scVal = scales[scBase + bi];
if (scVal === 0 || scVal > 210) continue;
const bd = MID_BANDS[bi];
const grp = Math.min(3, bi >> 1);
let adj = 0;
let k = bd.lo;
while (k < bd.hi) {
const hot = sameWin && prevMidMag[k] > 0 ? 1 : 0;
const cIdx = grp * 4 + adj * 2 + hot;
const sym = tokCtx[cIdx][tokPos[cIdx]++];
if (sym === 255) break;
let z = sym >>> 3;
let qMinus1 = sym & 7;
if (z === 30) {
while (true) {
const b = extRun[extRunPos++];
z += b;
if (b < 255) break;
}
}
if (qMinus1 === 7) qMinus1 += extMag[extMagPos++];
k += z;
const absQ = qMinus1 + 1;
const neg = popSign();
curMidMag[k] = absQ;
tmpMidIdx[mWrite] = k;
tmpMidQ[mWrite] = neg ? -absQ : absQ;
mWrite++;
adj = z === 0 ? 1 : 0;
k++;
}
}
prevMidMag.set(curMidMag);
for (let bi = 0; bi < 4; bi++) {
if (scales[scBase + 12 + bi] === 0) continue;
const bd = SIDE_BANDS[bi];
let adj = 0;
let k = bd.lo;
while (k < bd.hi) {
const cIdx = (bi > 1 ? 2 : 0) + adj;
const sym = sTokCtx[cIdx][sTokPos[cIdx]++];
if (sym === 255) break;
let z = sym >>> 3;
let qMinus1 = sym & 7;
if (z === 30) {
while (true) {
const b = extRun[extRunPos++];
z += b;
if (b < 255) break;
}
}
if (qMinus1 === 7) qMinus1 += extMag[extMagPos++];
k += z;
const absQ = qMinus1 + 1;
const neg = popSign();
tmpSideIdx[sWrite] = k;
tmpSideQ[sWrite] = neg ? -absQ : absQ;
sWrite++;
adj = z === 0 ? 1 : 0;
k++;
}
}
}
midOff[F] = mWrite;
sideOff[F] = sWrite;
return {
midOff,
midIdx: tmpMidIdx.slice(0, mWrite),
midQ: tmpMidQ.slice(0, mWrite),
sideOff,
sideIdx: tmpSideIdx.slice(0, sWrite),
sideQ: tmpSideQ.slice(0, sWrite),
};
}
function inspectWecHeader(inputBuf) {
const bytes =
inputBuf instanceof Uint8Array
? inputBuf
: new Uint8Array(inputBuf.buffer || inputBuf, inputBuf.byteOffset || 0, inputBuf.byteLength);
if (bytes.length < WEC_HEADER_BYTES) {
throw new Error(`Invalid .wec file: expected >= 32 bytes, got ${bytes.length}`);
}
const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
if (magic !== WEC_MAGIC) {
throw new Error(`Unsupported .wec magic "${magic}" (expected "${WEC_MAGIC}")`);
}
const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const version = dv.getUint8(4);
const entropyMode = dv.getUint8(5);
const tierCode = dv.getUint16(6, true);
const sampleRate = dv.getUint32(8, true);
const totalFrames = dv.getUint32(12, true);
const totalSamples = dv.getUint32(16, true);
const activeScaleMask = dv.getUint32(20, true);
const payloadBytes = dv.getUint32(24, true);
const storedCrc32 = dv.getUint32(28, true);
if (bytes.length < WEC_HEADER_BYTES + payloadBytes) {
throw new Error(
`Truncated .wec payload: header declares ${payloadBytes} B, got ${bytes.length - WEC_HEADER_BYTES} B`
);
}
const crcInput = new Uint8Array(28 + payloadBytes);
crcInput.set(bytes.subarray(0, 28), 0);
crcInput.set(bytes.subarray(WEC_HEADER_BYTES, WEC_HEADER_BYTES + payloadBytes), 28);
const computedCrc32 = computeCrc32(crcInput);
const crcOk = computedCrc32 === storedCrc32;
return {
magic,
version,
entropyMode,
entropyModeName:
entropyMode === 0 ? '48-Ctx Inter-Frame Per-Bin' : '16-Ctx Inter-Frame 2D Run-Mag',
tierCode,
tierLabel:
tierCode <= 500 ? 'CWC-500K' : tierCode <= 1000 ? 'CWC-1M' : 'CWC-HiFi',
sampleRate,
totalFrames,
totalSamples,
durationSec: Number((totalSamples / sampleRate).toFixed(3)),
activeScaleMask,
payloadBytes,
fileBytes: WEC_HEADER_BYTES + payloadBytes,
crc32Hex: storedCrc32.toString(16).padStart(8, '0'),
crcOk,
};
}
function unpackWecContainer(inputBuf) {
const bytes =
inputBuf instanceof Uint8Array
? inputBuf
: new Uint8Array(inputBuf.buffer || inputBuf, inputBuf.byteOffset || 0, inputBuf.byteLength);
const header = inspectWecHeader(bytes);
if (!header.crcOk) {
throw new Error(`CRC-32 mismatch in .wec container (expected 0x${header.crc32Hex})`);
}
const payload = bytes.subarray(WEC_HEADER_BYTES, WEC_HEADER_BYTES + header.payloadBytes);
const F = header.totalFrames;
const hdrDecoded = decodeFrameHeaders(payload, 0, F, header.activeScaleMask);
const atomsDecoded =
header.entropyMode === 0
? decodeAtomsMode0(payload, hdrDecoded.nextOffset, F, hdrDecoded.winTypes, hdrDecoded.scales)
: decodeAtomsMode1(payload, hdrDecoded.nextOffset, F, hdrDecoded.winTypes, hdrDecoded.scales);
return {
startFrame: 0,
frameCount: F,
winTypes: hdrDecoded.winTypes,
scales: hdrDecoded.scales,
pans: hdrDecoded.pans,
midOff: atomsDecoded.midOff,
midIdx: atomsDecoded.midIdx,
midQ: atomsDecoded.midQ,
sideOff: atomsDecoded.sideOff,
sideIdx: atomsDecoded.sideIdx,
sideQ: atomsDecoded.sideQ,
packedLength: header.fileBytes,
header,
};
}
function computeShortSubblockEnv(spec, winType, subEnvOut) {
subEnvOut.fill(1);
if (winType !== WIN_SHORT) return;
let maxE = 1e-12;
for (let s = 0; s < 8; s++) {
let e = 0;
for (let k = 72 + s; k < 1115; k += 8) {
const v = spec[k];
e += v * v;
}
subEnvOut[s] = Math.sqrt(e);
if (subEnvOut[s] > maxE) maxE = subEnvOut[s];
}
let sqSum = 0;
for (let s = 0; s < 8; s++) {
const r = Math.max(0.14, subEnvOut[s] / maxE);
subEnvOut[s] = r;
sqSum += r * r;
}
const norm = Math.sqrt(8 / Math.max(1e-12, sqSum));
for (let s = 0; s < 8; s++) subEnvOut[s] *= norm;
}
function synthesizeHarmonicSbrBand(
rawM,
bd,
qFill,
scaleVal,
winType,
baseTop,
subEnv,
nextNoise,
scratch,
outSpec = null,
layerGain = 1,
srcLoOverride = -1
) {
let zCount = 0;
for (let k = bd.lo; k < bd.hi; k++) {
if (rawM[k] === 0) zCount++;
}
if (zCount === 0 || qFill <= 0) return 0;
const srcLo =
srcLoOverride >= 0
? srcLoOverride
: bd.lo >= 882
? 296
: bd.lo >= 594
? 208
: bd.lo >= 409
? 144
: 72;
const top = Math.max(srcLo + 64, Math.min(bd.lo, baseTop));
const span8 = Math.max(16, ((top - srcLo) >> 3) << 3);
const shift = Math.max(8, ((bd.hi - top + 7) >> 3) << 3);
let patchP = 0;
let srcMax = 0;
const canPatch = scaleVal === 0 || bd.lo >= 409;
if (canPatch) {
for (let k = bd.lo; k < bd.hi; k++) {
if (rawM[k] !== 0) continue;
let sk = k - shift;
if (sk < srcLo) sk = srcLo + (((sk - srcLo) % span8 + span8) % span8);
const sv = sk < top ? rawM[sk] : 0;
const a = Math.abs(sv);
if (a > srcMax) srcMax = a;
patchP += sv * sv;
}
}
const usePatch = canPatch && patchP > 1e-10;
const patchRms = usePatch ? Math.sqrt(patchP / zCount) : 0;
const srcCrest = patchRms > 1e-8 ? srcMax / patchRms : 0;
const noiseMix = usePatch ? Math.max(0, Math.min(0.10, (2.6 - srcCrest) * 0.10)) : 0.35;
const patchMix = 1 - noiseMix;
const patchNorm = usePatch ? Math.sqrt(zCount / patchP) : 0;
const targetRms = PNS_LUT[Math.min(220, qFill)] * (usePatch ? 1 : 0.35);
let mixP = 0;
let u = 0;
for (let k = bd.lo; k < bd.hi; k++) {
if (rawM[k] !== 0) continue;
let v;
if (usePatch) {
let sk = k - shift;
if (sk < srcLo) sk = srcLo + (((sk - srcLo) % span8 + span8) % span8);
const sv = sk < top ? rawM[sk] * patchNorm : 0;
v = noiseMix > 0 ? patchMix * sv + noiseMix * nextNoise() : sv;
} else {
v = nextNoise();
}
if (winType === WIN_SHORT) v *= subEnv[k & 7];
scratch[u++] = v;
mixP += v * v;
}
const scN = targetRms * Math.sqrt(zCount / (mixP + 1e-12));
let bp = 0;
u = 0;
for (let k = bd.lo; k < bd.hi; k++) {
if (rawM[k] === 0) {
const val = scratch[u++] * scN;
rawM[k] = val;
if (outSpec) outSpec[k] = val * layerGain;
bp += val * val;
}
}
return bp;
}
class CWCStreamDecoder {
constructor(totalFrames = 0) {
this.mdct = createAdaptiveMDCTProcessor(CWC_FRAME_N);
this.totalFrames = totalFrames;
this.chunks = [];
this.frameToChunk = new Int16Array(Math.max(1, totalFrames)).fill(-1);
this.loadedCount = 0;
this.rawMid = new Float64Array(CWC_HOP_M);
this.specMid = new Float64Array(CWC_HOP_M);
this.specSide = new Float64Array(CWC_HOP_M);
this.outMidN = new Float64Array(CWC_FRAME_N);
this.outSideN = new Float64Array(CWC_FRAME_N);
this.olaL = new Float64Array(CWC_HOP_M);
this.olaR = new Float64Array(CWC_HOP_M);
this.noiseScratch = new Float64Array(640);
this.subEnv = new Float64Array(8);
this.bandSteps = new Float64Array(MID_BANDS.length);
this.sideSteps = new Float64Array(4);
this._layerPow = new Float64Array(5);
this.stemGains = new Float32Array([1, 1, 1, 1, 1]);
this.stemMutes = new Uint8Array(5);
this.stemSolos = new Uint8Array(5);
this.effectiveGains = new Float32Array([1, 1, 1, 1, 1]);
this.masterGain = 0.98;
this.layerRms = new Float32Array(5);
this.spectrum32 = new Float32Array(32);
this.currentFrame = 0;
}
setTotalFrames(totalFrames) {
if (this.frameToChunk.length < totalFrames) {
const next = new Int16Array(totalFrames).fill(-1);
next.set(this.frameToChunk.subarray(0, Math.min(totalFrames, this.frameToChunk.length)));
this.frameToChunk = next;
}
this.totalFrames = totalFrames;
}
ingestChunk(chunkBundle) {
if (!chunkBundle || typeof chunkBundle.startFrame !== 'number') return;
const { startFrame, frameCount } = chunkBundle;
const endFrame = Math.min(this.totalFrames, startFrame + frameCount);
if (this.frameToChunk.length < endFrame) {
this.setTotalFrames(endFrame);
}
const slot = this.chunks.length;
this.chunks.push(chunkBundle);
let newlyLoaded = 0;
for (let f = startFrame; f < endFrame; f++) {
if (this.frameToChunk[f] < 0) newlyLoaded++;
this.frameToChunk[f] = slot;
}
this.loadedCount += newlyLoaded;
}
hasFrame(frameIdx) {
if (frameIdx < 0 || frameIdx >= this.totalFrames) return true;
return this.frameToChunk[frameIdx] >= 0;
}
inspectFrame(frameIdx) {
if (frameIdx < 0 || frameIdx >= this.totalFrames) return null;
const slot = this.frameToChunk[frameIdx];
if (slot < 0) return null;
const ch = this.chunks[slot];
const lf = frameIdx - ch.startFrame;
const winType = ch.winTypes[lf];
const scBase = lf * CWC_NUM_SCALES;
const panBase = lf * CWC_NUM_PANS;
const mStart = ch.midOff[lf];
const mEnd = ch.midOff[lf + 1];
const sStart = ch.sideOff[lf];
const sEnd = ch.sideOff[lf + 1];
const scalesM = [];
for (let bi = 0; bi < MID_BANDS.length; bi++) {
const v = ch.scales[scBase + bi];
const sbrV = bi >= 4 ? ch.scales[scBase + 17 + (bi - 4)] : 0;
const name = MID_BANDS[bi].label.split(' ')[0];
if (v > 0 && v <= 210) scalesM.push(`${name}:${v}${sbrV ? `+F${sbrV}` : ''}`);
else if (sbrV > 0) scalesM.push(`${name}:SBR${sbrV}`);
else scalesM.push(`${name}:0`);
}
const pnsM = [
`PNS[12.0–15.5k]=${ch.scales[scBase + 10]}`,
`PNS[15.5–19.6k]=${ch.scales[scBase + 11]}`,
];
const pans = [];
for (let bi = 0; bi < CWC_NUM_PANS; bi++) {
pans.push(`Pan${bi}=${(ch.pans[panBase + bi] / 20).toFixed(2)}`);
}
const sampleAtoms = [];
const showN = Math.min(18, mEnd - mStart);
for (let p = 0; p < showN; p++) {
const k = ch.midIdx[mStart + p];
const q = ch.midQ[mStart + p];
sampleAtoms.push(`X[${k}]=${q > 0 ? '+' : ''}${q}`);
}
return {
winName: WIN_NAMES[winType] || 'LONG_4096',
midNzCount: mEnd - mStart,
sideNzCount: sEnd - sStart,
scalesM,
pnsM,
pans,
atomsPreview: sampleAtoms.join('  ') + (mEnd - mStart > showN ? `  … (+${mEnd - mStart - showN} sparse atoms)` : ''),
};
}
setStemControls(opts = {}) {
const { gains, mutes, solos, masterGain } = opts;
if (gains) for (let i = 0; i < 5; i++) this.stemGains[i] = gains[i];
if (mutes) for (let i = 0; i < 5; i++) this.stemMutes[i] = mutes[i] ? 1 : 0;
if (solos) for (let i = 0; i < 5; i++) this.stemSolos[i] = solos[i] ? 1 : 0;
if (typeof masterGain === 'number') this.masterGain = masterGain;
let anySolo = false;
for (let i = 0; i < 5; i++) {
if (this.stemSolos[i]) {
anySolo = true;
break;
}
}
for (let i = 0; i < 5; i++) {
const active = anySolo ? this.stemSolos[i] === 1 : this.stemMutes[i] === 0;
this.effectiveGains[i] = active ? this.stemGains[i] : 0;
}
}
seekToFrame(targetFrame) {
const clamped = Math.max(0, Math.min(Math.max(0, this.totalFrames - 1), targetFrame | 0));
this.olaL.fill(0);
this.olaR.fill(0);
const preStart = Math.max(0, clamped - 2);
const dummyL = this._dummyL || (this._dummyL = new Float32Array(CWC_HOP_M));
const dummyR = this._dummyR || (this._dummyR = new Float32Array(CWC_HOP_M));
for (let f = preStart; f < clamped; f++) {
this.decodeFrame(f, dummyL, dummyR, 0);
}
this.currentFrame = clamped;
return clamped;
}
decodeFrame(frameIdx, outL, outR, outOffset = 0) {
this.currentFrame = frameIdx;
const M = CWC_HOP_M;
const rawM = this.rawMid;
const specM = this.specMid;
const specS = this.specSide;
rawM.fill(0);
specM.fill(0);
specS.fill(0);
const eff = this.effectiveGains;
const g0 = eff[0];
const g1 = eff[1];
const g2 = eff[2];
const g3 = eff[3];
const g4 = eff[4];
const layerPow = this._layerPow;
layerPow.fill(0);
let winType = WIN_LONG;
const slot = frameIdx >= 0 && frameIdx < this.totalFrames ? this.frameToChunk[frameIdx] : -1;
let seed = ((frameIdx + 1) * 0x9e3779b1) ^ 0x2468ace1;
const nextNoise = () => {
seed ^= seed << 13;
seed >>>= 0;
seed ^= seed >>> 17;
seed >>>= 0;
seed ^= seed << 5;
seed >>>= 0;
const u1 = (seed >>> 0) / 4294967296 - 0.5;
seed ^= seed << 13;
seed >>>= 0;
seed ^= seed >>> 17;
seed >>>= 0;
seed ^= seed << 5;
seed >>>= 0;
const u2 = (seed >>> 0) / 4294967296 - 0.5;
return (u1 + u2) * 2.4494897;
};
if (slot >= 0) {
const ch = this.chunks[slot];
const lf = frameIdx - ch.startFrame;
winType = ch.winTypes[lf];
const scBase = lf * CWC_NUM_SCALES;
const panBase = lf * CWC_NUM_PANS;
for (let bi = 0; bi < MID_BANDS.length; bi++) {
const scVal = ch.scales[scBase + bi];
if (scVal > 0 && scVal <= 210) {
this.bandSteps[bi] = GAIN_LUT[scVal] / MID_BANDS[bi].qMax;
} else {
this.bandSteps[bi] = 0;
if (scVal > 210) {
const bd = MID_BANDS[bi];
const c = bd.hi - bd.lo;
const rms = WHISPER_LUT[Math.min(45, scVal - 210)];
let np = 0;
for (let u = 0; u < c; u++) {
const n = nextNoise();
this.noiseScratch[u] = n;
np += n * n;
}
const scN = rms * Math.sqrt(c / (np + 1e-12));
const lg = eff[bd.layer];
let bp = 0;
for (let u = 0; u < c; u++) {
const val = this.noiseScratch[u] * scN;
rawM[bd.lo + u] = val;
specM[bd.lo + u] = val * lg;
bp += val * val;
}
layerPow[bd.layer] += bp;
}
}
}
const mStart = ch.midOff[lf];
const mEnd = ch.midOff[lf + 1];
for (let p = mStart; p < mEnd; p++) {
const k = ch.midIdx[p];
const q = ch.midQ[p];
const bi = BIN_TO_MID_BAND[k];
const step = this.bandSteps[bi];
const val = (q < 0 ? q + 0.14 : q - 0.14) * step;
rawM[k] = val;
const layer = MID_BANDS[bi].layer;
specM[k] = val * eff[layer];
layerPow[layer] += val * val;
}
computeShortSubblockEnv(rawM, winType, this.subEnv);
let highestReadyEnd = 296;
for (let bi = 0; bi < 10; bi++) {
if (ch.scales[scBase + bi] > 0 && ch.scales[scBase + bi] <= 210) {
highestReadyEnd = Math.max(highestReadyEnd, MID_BANDS[bi].hi);
}
}
for (let fi = 0; fi < 6; fi++) {
const qFill = ch.scales[scBase + 17 + fi];
const bi = 4 + fi;
const bd = MID_BANDS[bi];
if (qFill > 0 && ch.scales[scBase + bi] <= 210) {
const bp = synthesizeHarmonicSbrBand(
rawM,
bd,
qFill,
ch.scales[scBase + bi],
winType,
highestReadyEnd,
this.subEnv,
nextNoise,
this.noiseScratch,
specM,
eff[bd.layer]
);
layerPow[bd.layer] += bp;
highestReadyEnd = Math.max(highestReadyEnd, bd.hi);
}
}
const pnsBd = this._pnsBd || (this._pnsBd = { lo: 0, hi: 0 });
for (let pi = 0; pi < PNS_MID_BANDS.length; pi++) {
const qDb = ch.scales[scBase + 10 + pi];
if (qDb <= 0) continue;
pnsBd.lo = PNS_MID_BANDS[pi][0];
pnsBd.hi = PNS_MID_BANDS[pi][1];
const bp = synthesizeHarmonicSbrBand(
rawM,
pnsBd,
qDb,
0,
winType,
1115,
this.subEnv,
nextNoise,
this.noiseScratch,
specM,
g3,
409
);
layerPow[3] += bp;
}
let sideP = 0;
for (let bi = 0; bi < CWC_NUM_PANS; bi++) {
const bd = SIDE_BANDS[bi];
const recPan = ch.pans[panBase + bi] * 0.05;
if (bi < 4) {
const scVal = ch.scales[scBase + 12 + bi];
this.sideSteps[bi] = scVal > 0 && scVal <= 210 ? GAIN_LUT[scVal] / bd.qMax : 0;
}
if (recPan !== 0) {
for (let k = bd.lo; k < bd.hi; k++) {
const pred = recPan * rawM[k];
specS[k] = pred;
}
}
}
const sStart = ch.sideOff[lf];
const sEnd = ch.sideOff[lf + 1];
for (let p = sStart; p < sEnd; p++) {
const k = ch.sideIdx[p];
const q = ch.sideQ[p];
const bi = BIN_TO_SIDE_BAND[k];
const step = this.sideSteps[bi];
specS[k] += (q < 0 ? q + 0.14 : q - 0.14) * step;
}
for (let k = 4; k < 1115; k++) {
const v = specS[k];
sideP += v * v;
specS[k] = v * g4;
}
for (let pi = 0; pi < PNS_SIDE_BANDS.length; pi++) {
const qDb = ch.scales[scBase + 16 + pi];
if (qDb <= 0) continue;
const [lo, hi] = PNS_SIDE_BANDS[pi];
const c = hi - lo;
const rms = PNS_LUT[Math.min(220, qDb)];
let spP = 0;
for (let u = 0; u < c; u++) {
const sv = rawM[643 + u] * 0.6;
const sVal = g4 > 0 ? specS[643 + u] / g4 : 0;
const v = sVal !== 0 ? sVal : (u & 8 ? sv : -sv);
spP += v * v;
}
const useCoherent = spP > 1e-10;
let np = 0;
for (let u = 0; u < c; u++) {
let v;
if (useCoherent) {
const sVal = g4 > 0 ? specS[643 + u] / g4 : 0;
v = sVal !== 0 ? sVal : rawM[643 + u] * 0.45 * (u & 8 ? 1 : -1);
} else {
v = nextNoise() * 0.35;
}
if (winType === WIN_SHORT) v *= this.subEnv[(lo + u) & 7];
this.noiseScratch[u] = v;
np += v * v;
}
const scN = (useCoherent ? rms : rms * 0.35) * Math.sqrt(c / (np + 1e-12));
for (let u = 0; u < c; u++) {
const val = this.noiseScratch[u] * scN;
sideP += val * val;
specS[lo + u] = val * g4;
}
}
layerPow[4] += sideP;
}
const invSqrtM = 1 / Math.sqrt(M);
this.layerRms[0] = Math.min(1, Math.sqrt(layerPow[0]) * invSqrtM * g0 * 1.9);
this.layerRms[1] = Math.min(1, Math.sqrt(layerPow[1]) * invSqrtM * g1 * 2.3);
this.layerRms[2] = Math.min(1, Math.sqrt(layerPow[2]) * invSqrtM * g2 * 3.8);
this.layerRms[3] = Math.min(1, Math.sqrt(layerPow[3]) * invSqrtM * g3 * 5.2);
this.layerRms[4] = Math.min(1, Math.sqrt(layerPow[4]) * invSqrtM * g4 * 4.5);
for (let b = 0; b < 32; b++) {
const lo = Math.floor(Math.pow(1820, b / 32)) - 1;
const hi = Math.max(lo + 1, Math.floor(Math.pow(1820, (b + 1) / 32)));
let e = 0;
for (let k = Math.max(0, lo); k < Math.min(1820, hi); k++) {
e += specM[k] * specM[k] + specS[k] * specS[k];
}
const db = Math.log10(Math.sqrt(e / (hi - lo + 1)) + 1e-4);
this.spectrum32[b] = Math.max(0, Math.min(1, (db + 3.2) / 3.4));
}
this.mdct.inverse(specM, winType, this.outMidN);
this.mdct.inverse(specS, winType, this.outSideN);
const mg = this.masterGain;
for (let n = 0; n < M; n++) {
const curL = this.outMidN[n] + this.outSideN[n];
const curR = this.outMidN[n] - this.outSideN[n];
const l = (this.olaL[n] + curL) * mg;
const r = (this.olaR[n] + curR) * mg;
outL[outOffset + n] = (l > 0.995 ? 0.995 : l < -0.995 ? -0.995 : l) + 0;
outR[outOffset + n] = (r > 0.995 ? 0.995 : r < -0.995 ? -0.995 : r) + 0;
this.olaL[n] = this.outMidN[M + n] + this.outSideN[M + n];
this.olaR[n] = this.outMidN[M + n] - this.outSideN[M + n];
}
}
}

function renderWecToWavBuffer(wecBytes, stemOpts = {}) {
  const bundle = unpackWecContainer(wecBytes);
  const { sampleRate, totalFrames, totalSamples } = bundle.header;
  const dec = new CWCStreamDecoder(totalFrames);
  dec.ingestChunk(bundle);
  dec.setStemControls(stemOpts);

  const wav = new ArrayBuffer(44 + totalSamples * 4);
  const dv = new DataView(wav);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF');
  dv.setUint32(4, 36 + totalSamples * 4, true);
  ws(8, 'WAVEfmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 2, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 4, true);
  dv.setUint16(32, 4, true);
  dv.setUint16(34, 16, true);
  ws(36, 'data');
  dv.setUint32(40, totalSamples * 4, true);

  const hopL = new Float32Array(CWC_HOP_M);
  const hopR = new Float32Array(CWC_HOP_M);
  let byteOff = 44;
  let written = 0;
  for (let f = 0; f < totalFrames && written < totalSamples; f++) {
    dec.decodeFrame(f, hopL, hopR, 0);
    const take = Math.min(CWC_HOP_M, totalSamples - written);
    for (let i = 0; i < take; i++) {
      dv.setInt16(byteOff, Math.max(-32767, Math.min(32767, Math.round(hopL[i] * 32767))), true);
      dv.setInt16(byteOff + 2, Math.max(-32767, Math.min(32767, Math.round(hopR[i] * 32767))), true);
      byteOff += 4;
    }
    written += take;
  }
  return new Uint8Array(wav);
}

export { inspectWecHeader, unpackWecContainer, renderWecToWavBuffer, CWCStreamDecoder };

const isMain = typeof process !== 'undefined' && process.argv?.[1] &&
  import.meta.url.endsWith(process.argv[1].split(/[\\/]/).pop());

if (isMain) {
  const fsMod = await import('node:fs');
  const [cmd = 'inspect', inFile, ...rest] = process.argv.slice(2);
  if (!inFile) {
    console.log('Usage: node wec-play.mjs <inspect|verify|render> <file.wec> [out.wav] [--solo=N] [--mute=N]');
    process.exit(1);
  }
  const buf = fsMod.readFileSync(inFile);
  if (cmd === 'inspect') {
    console.log(JSON.stringify(inspectWecHeader(buf), null, 2));
  } else if (cmd === 'verify') {
    const t0 = performance.now();
    const unpacked = unpackWecContainer(buf);
    const ms = (performance.now() - t0).toFixed(1);
    console.log(JSON.stringify({
      status: 'OK',
      crc32Hex: unpacked.header.crc32Hex,
      tierLabel: unpacked.header.tierLabel,
      frames: unpacked.frameCount,
      midAtoms: unpacked.midIdx.length,
      sideAtoms: unpacked.sideIdx.length,
      unpackMs: Number(ms),
    }, null, 2));
  } else if (cmd === 'render') {
    const outFile = rest.find((a) => !a.startsWith('--')) || inFile.replace(/\.wec$/i, '') + '.wav';
    const solos = [false, false, false, false, false];
    const mutes = [false, false, false, false, false];
    for (const arg of rest) {
      if (arg.startsWith('--solo=')) solos[Number(arg.slice(7))] = true;
      if (arg.startsWith('--mute=')) mutes[Number(arg.slice(7))] = true;
    }
    const t0 = performance.now();
    const wavBytes = renderWecToWavBuffer(buf, { solos, mutes });
    fsMod.writeFileSync(outFile, wavBytes);
    const sec = ((performance.now() - t0) / 1000).toFixed(2);
    console.log(`Rendered ${outFile} (${(wavBytes.length / 1048576).toFixed(2)} MiB) in ${sec}s`);
  } else {
    console.error(`Unknown command: ${cmd}`);
    process.exit(1);
  }
}
