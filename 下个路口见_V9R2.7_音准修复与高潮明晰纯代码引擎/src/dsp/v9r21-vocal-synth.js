// V9R2.1 Ground-Up Direct-ERB Vocal & True Stereo Harmony Synthesizer (`src/dsp/v9r21-vocal-synth.js`)
// Rebuilt from scratch without any V8.5 code:
// - Tracks 01-03: Direct 64-ERB vocal-tract harmonic synthesis + true Rosenberg glottal pulse phase
// - Tracks 04-06: 1024-pt Kolmogorov real-cepstrum minimum-phase WOLA consonants & velvet glottal breath
// - Track 07: True independent Left/Right stereo harmony pitch & timbre runs (not unison Lead cloning)
// - Track 08: 4-tap early reflections + 8-line prime Householder FDN space
import { Radix2ComplexFFT, createMulberry32 } from './fft-core.js';

const TWO_PI = 2.0 * Math.PI;
const CONS_ANCHORS = [0.0, 0.5, 1.0];
const VOWEL_ANCHORS = [0.0, 0.22, 0.5, 0.78, 1.0];

function hzToErb(f) {
  return 21.4 * Math.log10(1.0 + 0.00437 * f);
}

function dcBlock60HzInPlace(x, sr) {
  const r = Math.exp((-TWO_PI * 60.0) / sr);
  const g = 0.5 * (1.0 + r);
  let x1 = 0.0;
  let y1 = 0.0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = g * (xi - x1) + r * y1;
    x[i] = yi;
    x1 = xi;
    y1 = yi;
  }
}

function decodeProtoPca(rows, basis, nErb, stepDb) {
  const out = new Array(rows.length);
  const kAtlas = basis ? basis.length : 0;
  for (let i = 0; i < rows.length; i++) {
    const v = new Float32Array(nErb);
    const row = rows[i];
    for (let d = 0; d < row.length && d < kAtlas; d++) {
      const w = row[d] * stepDb;
      if (w === 0) continue;
      const b = basis[d];
      for (let e = 0; e < nErb; e++) v[e] += w * b[e];
    }
    out[i] = v;
  }
  return out;
}

function evalProtoOnEvent(proto, meanErb, aperMean, L, cf, stepDb, atlasBasis) {
  const nErb = meanErb.length;
  const nAper = aperMean.length;
  const vot = L > 1 && cf > 0 ? Math.min(L - 1, cf) : 0;
  const predE = new Float32Array(L * nErb);
  const predA = new Float32Array(L * nAper);
  const usePca = !!(atlasBasis && proto.consErbPca);
  const consSrc = usePca ? decodeProtoPca(proto.consErbPca, atlasBasis, nErb, stepDb) : proto.consErb;
  const vowelSrc = usePca ? decodeProtoPca(proto.vowelErbPca, atlasBasis, nErb, stepDb) : proto.vowelErb;
  const erbScale = usePca ? 1.0 : stepDb;

  if (vot > 0) {
    const cMax = CONS_ANCHORS.length - 1;
    for (let t = 0; t < vot; t++) {
      const u = vot > 1 ? (t / (vot - 1)) * cMax : 0.0;
      const i0 = Math.min(cMax - 1, Math.max(0, Math.floor(u)));
      const fr = u - i0;
      const c0 = consSrc[i0];
      const c1 = consSrc[i0 + 1];
      const eOff = t * nErb;
      for (let e = 0; e < nErb; e++) predE[eOff + e] = meanErb[e] + ((1.0 - fr) * c0[e] + fr * c1[e]) * erbScale;
      const a0 = proto.consAper[i0];
      const a1 = proto.consAper[i0 + 1];
      const aOff = t * nAper;
      for (let a = 0; a < nAper; a++) predA[aOff + a] = aperMean[a] + ((1.0 - fr) * a0[a] + fr * a1[a]) * stepDb;
    }
  }

  const vLen = L - vot;
  if (vLen > 0) {
    const vMax = VOWEL_ANCHORS.length - 1;
    for (let k = 0; k < vLen; k++) {
      const t = vot + k;
      const u = vLen > 1 ? (k / (vLen - 1)) * vMax : 0.0;
      const i0 = Math.min(vMax - 1, Math.max(0, Math.floor(u)));
      const fr = u - i0;
      const v0 = vowelSrc[i0];
      const v1 = vowelSrc[i0 + 1];
      const eOff = t * nErb;
      for (let e = 0; e < nErb; e++) predE[eOff + e] = meanErb[e] + ((1.0 - fr) * v0[e] + fr * v1[e]) * erbScale;
      const a0 = proto.vowelAper[i0];
      const a1 = proto.vowelAper[i0 + 1];
      const aOff = t * nAper;
      for (let a = 0; a < nAper; a++) predA[aOff + a] = aperMean[a] + ((1.0 - fr) * a0[a] + fr * a1[a]) * stepDb;
    }
  }
  return { predE, predA };
}

function buildCoarticulatedErbGrids(events, prototypes, stepDb, meanErb, aperMean, totalFrames, atlasBasis) {
  const nErb = meanErb.length;
  const nAper = aperMean.length;
  const baseErb = new Float32Array(totalFrames * nErb);
  const baseAper = new Float32Array(totalFrames * nAper);
  baseAper.fill(-3.6);

  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    const L = ev.L;
    const sF = ev.s;
    const span = Math.min(totalFrames, sF + L) - sF;
    if (L <= 0 || span <= 0) continue;
    const { predE, predA } = evalProtoOnEvent(prototypes[ev.sid], meanErb, aperMean, L, ev.cf, stepDb, atlasBasis);
    baseErb.set(predE.subarray(0, span * nErb), sF * nErb);
    baseAper.set(predA.subarray(0, span * nAper), sF * nAper);
  }

  const seams = [];
  for (let i = 1; i < events.length; i++) {
    const prev = events[i - 1];
    const curr = events[i];
    const b = curr.s;
    if (prev.L >= 2 && curr.L >= 2 && prev.s + prev.L === b && b >= 1 && b < totalFrames) {
      const wPre = Math.min(18, Math.floor((prev.L * 12) / 25), b);
      const wPost = Math.min(curr.pl ? 4 : 10, Math.floor((curr.L * 9) / 25), totalFrames - b);
      if (wPre >= 1 && wPost >= 1) {
        const dErb = new Float32Array(nErb);
        const dAper = new Float32Array(nAper);
        const bE = b * nErb;
        const pE = (b - 1) * nErb;
        for (let e = 0; e < nErb; e++) dErb[e] = baseErb[bE + e] - baseErb[pE + e];
        const bA = b * nAper;
        const pA = (b - 1) * nAper;
        for (let a = 0; a < nAper; a++) dAper[a] = baseAper[bA + a] - baseAper[pA + a];
        seams.push({ b, wPre, wPost, dErb, dAper });
      }
    }
  }

  for (const { b, wPre, wPost, dErb, dAper } of seams) {
    const invTot = 1.0 / (wPre + wPost);
    const gPre = wPre * invTot;
    const gPost = wPost * invTot;
    for (let r = 0; r < wPre; r++) {
      const w = 0.5 * gPre * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPre));
      const f = b - 1 - r;
      const eOff = f * nErb;
      const aOff = f * nAper;
      for (let e = 0; e < nErb; e++) baseErb[eOff + e] += w * dErb[e];
      for (let a = 0; a < nAper; a++) baseAper[aOff + a] += w * dAper[a];
    }
    for (let r = 0; r < wPost; r++) {
      const w = 0.5 * gPost * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPost));
      const f = b + r;
      const eOff = f * nErb;
      const aOff = f * nAper;
      for (let e = 0; e < nErb; e++) baseErb[eOff + e] -= w * dErb[e];
      for (let a = 0; a < nAper; a++) baseAper[aOff + a] -= w * dAper[a];
    }
  }
  return { baseErb, baseAper };
}

function buildExprKnots(L, cf) {
  if (L <= 1) return Int32Array.of(0);
  const last = L - 1;
  const vot = Math.min(last, Math.max(1, cf));
  const split = Math.min(last, vot + 16);
  const set = new Set([0, vot >> 1, vot, Math.max(vot, last - 3), last]);
  for (let t = vot; t <= split; t += 4) set.add(t);
  for (let t = split; t <= last; t += 8) set.add(t);
  return Int32Array.from(Array.from(set).sort((a, b) => a - b));
}

function interp1dKnots(pos, vals, L, cosine = false) {
  const out = new Float32Array(L);
  const K = pos.length;
  if (K === 0 || L === 0) return out;
  if (K === 1) {
    out.fill(vals[0]);
    return out;
  }
  let seg = 0;
  for (let t = 0; t < L; t++) {
    while (seg < K - 2 && t > pos[seg + 1]) seg++;
    const p0 = pos[seg];
    const p1 = pos[seg + 1];
    if (t <= p0) out[t] = vals[seg];
    else if (t >= p1) out[t] = vals[seg + 1];
    else {
      let fr = (t - p0) / (p1 - p0);
      if (cosine) fr = 0.5 - 0.5 * Math.cos(Math.PI * fr);
      out[t] = (1.0 - fr) * vals[seg] + fr * vals[seg + 1];
    }
  }
  return out;
}

function interp2dKnots(pos, vals, dims, L) {
  const out = new Float32Array(L * dims);
  const K = pos.length;
  if (K === 0 || L === 0) return out;
  if (K === 1) {
    for (let t = 0; t < L; t++) out.set(vals.subarray(0, dims), t * dims);
    return out;
  }
  let seg = 0;
  for (let t = 0; t < L; t++) {
    while (seg < K - 2 && t > pos[seg + 1]) seg++;
    const p0 = pos[seg];
    const p1 = pos[seg + 1];
    const oT = t * dims;
    if (t <= p0) out.set(vals.subarray(seg * dims, (seg + 1) * dims), oT);
    else if (t >= p1) out.set(vals.subarray((seg + 1) * dims, (seg + 2) * dims), oT);
    else {
      const fr = (t - p0) / (p1 - p0);
      const o0 = seg * dims;
      const o1 = (seg + 1) * dims;
      for (let d = 0; d < dims; d++) out[oT + d] = (1.0 - fr) * vals[o0 + d] + fr * vals[o1 + d];
    }
  }
  return out;
}

function bridgeShortF0Gaps(f0, maxGap) {
  const N = f0.length;
  let i = 0;
  while (i < N) {
    if (f0[i] <= 60.0) {
      let j = i;
      while (j < N && f0[j] <= 60.0) j++;
      const gap = j - i;
      if (i > 0 && j < N && gap <= maxGap) {
        const l0 = Math.log(f0[i - 1]);
        const l1 = Math.log(f0[j]);
        if (Math.abs(l1 - l0) <= Math.log(2.0) * (7.0 / 12.0)) {
          for (let k = 0; k < gap; k++) f0[i + k] = Math.exp(l0 + ((l1 - l0) * (k + 1)) / (gap + 1));
        }
      }
      i = j;
    } else {
      i++;
    }
  }
}

function synthesizeBackingVoiceChannel(model, notes, numSamples, sr, glottPsi) {
  const hop = model.hop;
  const nH = model.nHarm;
  const gainStep = model.gainStepDb ?? 0.5;
  const timbres = model.timbresDbQ.map((row) => {
    const a = new Float64Array(nH);
    let sq = 0.0;
    for (let h = 0; h < nH; h++) {
      const v = Math.pow(10.0, (row[h] * 0.5) / 20.0);
      a[h] = v;
      sq += v * v;
    }
    const sc = 1.0 / Math.sqrt(0.5 * sq + 1e-18);
    for (let h = 0; h < nH; h++) a[h] *= sc;
    return a;
  });

  const out = new Float32Array(numSamples);
  const nf = Math.floor(numSamples / hop) + 2;
  const f0g = new Float32Array(nf);
  const gg = new Float32Array(nf);
  const tidg = new Int16Array(nf);

  let curS = 0;
  for (const [ds, tid, dk, dc, dg] of notes) {
    curS += ds;
    let k = 0;
    let c = 0;
    let g = 0;
    const kf = [];
    const kc = [];
    const kg = [];
    for (let i = 0; i < dk.length; i++) {
      k += dk[i];
      c += dc[i];
      g += dg[i];
      kf.push(curS + k);
      kc.push(c);
      kg.push(g * gainStep);
    }
    for (let i = 0; i < kf.length; i++) {
      const fA = kf[i];
      const fB = i + 1 < kf.length ? kf[i + 1] : fA + 1;
      const i2 = Math.min(i + 1, kf.length - 1);
      for (let f = fA; f < fB && f < nf; f++) {
        const t = fB > fA ? (f - fA) / (fB - fA) : 0.0;
        const cents = kc[i] + (kc[i2] - kc[i]) * t;
        f0g[f] = model.f0RefHz * Math.pow(2.0, cents / 1200.0);
        gg[f] = Math.pow(10.0, (kg[i] + (kg[i2] - kg[i]) * t) / 20.0);
        tidg[f] = tid;
      }
    }
  }

  const cosPsi = Float64Array.from({ length: nH }, (_, h) => Math.cos(glottPsi[h] || 0.0));
  const sinPsi = Float64Array.from({ length: nH }, (_, h) => Math.sin(glottPsi[h] || 0.0));
  let phi = 0.0;
  const amp = new Float64Array(nH);
  const dAmp = new Float64Array(nH);

  for (let f = 0; f < nf - 1; f++) {
    const fa = f0g[f];
    const fb = f0g[f + 1];
    if (fa <= 0.0 && fb <= 0.0) {
      phi = 0.0;
      continue;
    }
    const s0 = f * hop;
    if (s0 >= numSamples) break;
    const fs = fa > 0.0 ? fa : fb;
    const fe = fb > 0.0 ? fb : fa;
    const profA = timbres[tidg[f]] || timbres[0];
    const profB = timbres[tidg[f + 1]] || profA;
    const ga = fa > 0.0 ? gg[f] : 0.0;
    const gb = fb > 0.0 ? gg[f + 1] : 0.0;
    for (let h = 0; h < nH; h++) {
      const a0 = ga * profA[h];
      const a1 = gb * profB[h];
      amp[h] = a0;
      dAmp[h] = (a1 - a0) / hop;
    }
    const end = Math.min(hop, numSamples - s0);
    for (let s = 0; s < end; s++) {
      const fc = fs + ((fe - fs) * s) / hop;
      phi += (TWO_PI * fc) / sr;
      if (phi >= TWO_PI) phi -= TWO_PI;
      const c1 = Math.cos(phi);
      const s1 = Math.sin(phi);
      const two = 2.0 * c1;
      let cp = 1.0;
      let sp = 0.0;
      let ch = c1;
      let sh = s1;
      let acc = 0.0;
      const nAct = Math.min(nH, Math.floor(12500.0 / Math.max(fc, 1.0)));
      for (let h = 0; h < nAct; h++) {
        const fh = (h + 1) * fc;
        const lowTrim = fh < 300.0 ? 0.52 + 0.48 * (fh / 300.0) : fh > 1500.0 && fh < 6500.0 ? 1.15 : 1.0;
        acc += (amp[h] + dAmp[h] * s) * lowTrim * (ch * cosPsi[h] - sh * sinPsi[h]);
        const cn = two * ch - cp;
        const sn = two * sh - sp;
        cp = ch;
        sp = sh;
        ch = cn;
        sh = sn;
      }
      out[s0 + s] = acc;
    }
  }
  return out;
}

function buildRosenbergGlottalPhases(maxHarm, tp, te) {
  const nGl = 512;
  const re = new Float64Array(nGl);
  const im = new Float64Array(nGl);
  let prev = 0.0;
  for (let i = 0; i < nGl; i++) {
    const t = i / nGl;
    let gl = 0.0;
    if (t < tp) gl = 0.5 * (1.0 - Math.cos((Math.PI * t) / tp));
    else if (t < te) gl = Math.cos((0.5 * Math.PI * (t - tp)) / (te - tp));
    re[i] = gl - prev;
    prev = gl;
  }
  const fft = new Radix2ComplexFFT(nGl);
  fft.transform(re, im, false);
  const cosA = new Float64Array(maxHarm);
  const sinA = new Float64Array(maxHarm);
  for (let h = 0; h < maxHarm; h++) {
    const ph = Math.atan2(im[h + 1], re[h + 1]);
    cosA[h] = Math.cos(ph);
    sinA[h] = Math.sin(ph);
  }
  return { cosA, sinA };
}

function synthesizeSpaceFdn(spaceFdn, dryMono, inLp = 0.48) {
  const N = dryMono.length;
  const envHop = spaceFdn.envHop;
  const halfHop = envHop * 0.5;
  const gmEnv = Float32Array.from(spaceFdn.qMid, (v) => (v / 255.0) * spaceFdn.baseWetGain);
  const gsEnv = Float32Array.from(spaceFdn.qSide, (v) => (v / 255.0) * spaceFdn.baseWetGain * 1.15);
  const nEnv = gmEnv.length;

  const delays = [1487, 1747, 2029, 2377, 2713, 3083, 3517, 3989];
  const preDelay = spaceFdn.preDelaySamples;
  const rt60 = spaceFdn.rt60Sec;
  const damp = Math.max(spaceFdn.dampLp, 0.54);
  const sr = spaceFdn.sampleRate;

  const erDelaysL = [547, 991, 1453, 1997, 3072];
  const erGainsL = [0.22, -0.17, 0.13, -0.09, 0.065];
  const erDelaysR = [613, 1087, 1571, 2137, 4224];
  const erGainsR = [0.21, 0.16, -0.13, 0.09, -0.055];

  const gains = delays.map((d) => Math.pow(10.0, (-3.0 * d) / (rt60 * sr)));
  const bufs = delays.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const lpSt = new Float64Array(8);

  // Two cascaded Schroeder allpass diffusers (eliminates metallic comb ringing in FDN tail)
  const ap1Buf = new Float32Array(149);
  const ap2Buf = new Float32Array(379);
  let ap1Idx = 0;
  let ap2Idx = 0;

  const preLen = Math.max(preDelay + 4500, 8192);
  const preBuf = new Float32Array(preLen);
  let preW = 0;
  let inState = 0.0;

  const outL = new Float32Array(N);
  const outR = new Float32Array(N);
  const normOut = 1.0 / Math.sqrt(4.0);
  const y = new Float64Array(8);

  for (let n = 0; n < N; n++) {
    inState = (1.0 - inLp) * dryMono[n] + inLp * inState;
    preBuf[preW] = inState;

    let erL = 0.0;
    let erR = 0.0;
    for (let k = 0; k < 5; k++) {
      let rl = preW - erDelaysL[k];
      if (rl < 0) rl += preLen;
      erL += erGainsL[k] * preBuf[rl];
      let rr = preW - erDelaysR[k];
      if (rr < 0) rr += preLen;
      erR += erGainsR[k] * preBuf[rr];
    }

    let rIdx = preW - preDelay;
    if (rIdx < 0) rIdx += preLen;
    let xPre = preBuf[rIdx];
    preW++;
    if (preW >= preLen) preW = 0;

    // 2-stage allpass diffusion
    const d1 = ap1Buf[ap1Idx];
    const v1 = xPre - 0.56 * d1;
    ap1Buf[ap1Idx] = v1;
    xPre = d1 + 0.56 * v1;
    ap1Idx++;
    if (ap1Idx >= 149) ap1Idx = 0;

    const d2 = ap2Buf[ap2Idx];
    const v2 = xPre - 0.52 * d2;
    ap2Buf[ap2Idx] = v2;
    xPre = d2 + 0.52 * v2;
    ap2Idx++;
    if (ap2Idx >= 379) ap2Idx = 0;

    let sumY = 0.0;
    for (let k = 0; k < 8; k++) {
      const yk = bufs[k][ptrs[k]];
      y[k] = yk;
      sumY += yk;
    }
    const house = 0.25 * sumY;
    let rawL = 0.0;
    let rawR = 0.0;
    for (let k = 0; k < 8; k++) {
      const fb = y[k] - house;
      const st = (1.0 - damp) * (fb * gains[k]) + damp * lpSt[k];
      lpSt[k] = st;
      bufs[k][ptrs[k]] = xPre * 0.35 + st;
      let p = ptrs[k] + 1;
      if (p >= delays[k]) p = 0;
      ptrs[k] = p;
      if ((k & 1) === 0) rawL += y[k];
      else rawR += y[k];
    }
    rawL *= normOut;
    rawR *= normOut;
    const rawM = 0.5 * (rawL + rawR);
    const rawS = 0.5 * (rawL - rawR);

    const ef = (n - halfHop) / envHop;
    let gm;
    let gs;
    if (ef <= 0) {
      gm = gmEnv[0];
      gs = gsEnv[0];
    } else if (ef >= nEnv - 1) {
      gm = gmEnv[nEnv - 1];
      gs = gsEnv[nEnv - 1];
    } else {
      const e0 = ef | 0;
      const fr = ef - e0;
      gm = (1.0 - fr) * gmEnv[e0] + fr * gmEnv[e0 + 1];
      gs = (1.0 - fr) * gsEnv[e0] + fr * gsEnv[e0 + 1];
    }
    outL[n] = rawM * gm + rawS * gs + erL * gm;
    outR[n] = rawM * gm - rawS * gs + erR * gm;
  }
  return [outL, outR];
}

export function synthesizeV9r21VocalAnatomy({
  speakerModel,
  phonemeAtlas,
  backingModel,
  spaceFdn,
  vocalScore,
  backingScore,
  config,
}) {
  const sr = speakerModel.sampleRate;
  const nSamples = speakerModel.numSamples;
  const hop = speakerModel.hop;
  const nErb = speakerModel.nErb;
  const nAper = speakerModel.nAper;
  const maxHarm = speakerModel.maxHarmonics;
  const refPeak = speakerModel.refPeak;
  const erbHz = Float64Array.from(speakerModel.erbFreqsHz);
  const erbMin = hzToErb(erbHz[0]);
  const erbMax = hzToErb(erbHz[nErb - 1]);
  const invErbSpan = (nErb - 1) / (erbMax - erbMin);

  const meanErb = Float32Array.from(speakerModel.meanErbDb);
  const aperMean = Float32Array.from(speakerModel.aperMean);
  const glottPsi = Float64Array.from(speakerModel.glottalPhase);
  const basisScale = 1.0 / speakerModel.basisQuantDiv;

  const kPca = vocalScore.pcaDims;
  const kAper = vocalScore.aperExprBands;
  const gainStepDb = vocalScore.gainStepDb ?? 0.25;
  const qStep = vocalScore.quantStepDb;

  const basisE = new Float32Array(kPca * nErb);
  for (let r = 0; r < kPca; r++) {
    const row = speakerModel.erbPcaBasisQ[r];
    for (let c = 0; c < nErb; c++) basisE[r * nErb + c] = row[c] * basisScale;
  }
  const basisA = new Float32Array(kAper * nAper);
  for (let r = 0; r < kAper; r++) {
    const row = speakerModel.aperPcaBasisQ[r];
    for (let c = 0; c < nAper; c++) basisA[r * nAper + c] = row[c] * basisScale;
  }

  const prototypes = phonemeAtlas.prototypes;
  const atlasBasis = phonemeAtlas.atlasErbBasisQ
    ? phonemeAtlas.atlasErbBasisQ.map((row) => Float32Array.from(row, (v) => v / (phonemeAtlas.basisQuantDiv || 4096)))
    : null;
  const events = vocalScore.events;
  const totalFrames = Math.floor(nSamples / hop) + 6;
  const qVec = vocalScore.quantStepVec || null;
  const emDelta = !!vocalScore.emDelta;

  const { baseErb, baseAper } = buildCoarticulatedErbGrids(
    events,
    prototypes,
    phonemeAtlas.atlasStepDb,
    meanErb,
    aperMean,
    totalFrames,
    atlasBasis
  );

  const gainGrid = new Float32Array(totalFrames);
  const vdegGrid = new Float32Array(totalFrames);
  const f0Grid = new Float32Array(totalFrames);
  const plosiveGrid = new Float32Array(totalFrames);
  const erbGrid = new Float32Array(baseErb);
  const aperGrid = new Float32Array(baseAper);
  const uiEvents = [];

  for (let idx = 0; idx < events.length; idx++) {
    const ev = events[idx];
    const L = ev.L;
    if (L <= 0) continue;
    const sF = ev.s;
    const cf = ev.cf;

    // Reconstruct sparse gain curve (0.25 dB delta-knots)
    const gPos = new Int32Array(ev.gk.length);
    const gVals = new Float32Array(ev.gk.length);
    let gPosAcc = 0;
    let gAcc = 0;
    for (let i = 0; i < ev.gk.length; i++) {
      gPosAcc += ev.gk[i] || 0;
      gAcc += ev.gq[i] || 0;
      gPos[i] = gPosAcc;
      const db = gAcc * gainStepDb;
      gVals[i] = db <= -79.5 ? 0.0 : refPeak * Math.pow(10.0, db / 20.0);
    }
    const gainCurve = interp1dKnots(gPos, gVals, L, false);

    // Reconstruct sparse voicing curve
    const vPos = new Int32Array(ev.vk.length);
    const vVals = new Float32Array(ev.vk.length);
    let vPosAcc = 0;
    let vAcc = 0;
    for (let i = 0; i < ev.vk.length; i++) {
      vPosAcc += ev.vk[i] || 0;
      vAcc += ev.vq[i] || 0;
      vPos[i] = vPosAcc;
      vVals[i] = Math.max(0.0, Math.min(1.0, vAcc * 0.01));
    }
    const vdegCurve = interp1dKnots(vPos, vVals, L, false);
    const vOn = Math.max(0, Math.min(L, ev.vo ?? 0));
    if (vOn > 0) {
      for (let i = 0; i < vOn; i++) vdegCurve[i] = 0.0;
      if (vOn < L) vdegCurve[vOn] *= 0.65;
    }

    const unvoicedOnly = vOn >= L;
    const f0Curve = new Float32Array(L);
    if (!unvoicedOnly) {
      const mPos = new Int32Array(ev.fk.length);
      const mVals = new Float32Array(ev.fk.length);
      let mPosAcc = 0;
      let cAcc = 0;
      for (let i = 0; i < ev.fk.length; i++) {
        mPosAcc += ev.fk[i] || 0;
        cAcc += ev.fc[i] || 0;
        mPos[i] = mPosAcc;
        mVals[i] = cAcc * 0.01;
      }
      const midiCurve = interp1dKnots(mPos, mVals, L, true);
      let midiSum = 0.0;
      for (let i = 0; i < L; i++) {
        midiSum += midiCurve[i];
        f0Curve[i] = 440.0 * Math.pow(2.0, (midiCurve[i] - 69.0) / 12.0);
      }
      uiEvents.push({
        eventIndex: idx,
        syllableId: ev.sid,
        text: ev.t,
        pinyin: ev.py,
        isHook: !!ev.hk,
        legatoPrev: !!ev.lp,
        legatoNext: !!ev.ln,
        startSec: (sF * hop) / sr,
        endSec: ((sF + L) * hop) / sr,
        meanMidi: midiSum / L,
      });
    } else {
      vdegCurve.fill(0.0);
    }

    const exprPos = buildExprKnots(L, cf);
    const nKnots = exprPos.length;
    const deKnots = new Float32Array(nKnots * nErb);
    const daKnots = new Float32Array(nKnots * nAper);
    const emAcc = new Int32Array(kPca);
    for (let k = 0; k < nKnots; k++) {
      const rowE = ev.em[k] || [];
      const eOff = k * nErb;
      for (let d = 0; d < kPca; d++) {
        const qVal = d < rowE.length ? rowE[d] : 0;
        const valQ = emDelta ? (emAcc[d] += qVal) : qVal;
        if (valQ === 0) continue;
        const w = valQ * (qVec ? qVec[d] : qStep);
        const bOff = d * nErb;
        for (let e = 0; e < nErb; e++) deKnots[eOff + e] += w * basisE[bOff + e];
      }
      const rowA = ev.ea[k] || [];
      const aOff = k * nAper;
      for (let d = 0; d < rowA.length && d < kAper; d++) {
        const w = rowA[d] * (qStep * 1.25);
        if (w === 0) continue;
        const bOff = d * nAper;
        for (let a = 0; a < nAper; a++) daKnots[aOff + a] += w * basisA[bOff + a];
      }
    }

    const deCurve = interp2dKnots(exprPos, deKnots, nErb, L);
    const daCurve = interp2dKnots(exprPos, daKnots, nAper, L);
    const span = Math.min(totalFrames, sF + L) - sF;
    if (span <= 0) continue;

    for (let i = 0; i < span; i++) {
      const gf = sF + i;
      gainGrid[gf] = gainCurve[i];
      vdegGrid[gf] = vdegCurve[i];
      f0Grid[gf] = f0Curve[i];
      const eOff = gf * nErb;
      const iE = i * nErb;
      for (let e = 0; e < nErb; e++) erbGrid[eOff + e] += deCurve[iE + e];
      const aOff = gf * nAper;
      const iA = i * nAper;
      for (let a = 0; a < nAper; a++) {
        aperGrid[aOff + a] = Math.max(-5.5, Math.min(-1.78, aperGrid[aOff + a] + daCurve[iA + a]));
      }
    }

    if (ev.pl) {
      const burstLen = Math.min(span, Math.max(2, Math.min(cf, 7)));
      for (let r = 0; r < burstLen; r++) plosiveGrid[sF + r] = r === burstLen - 1 && burstLen > 2 ? 0.5 : 1.0;
    }
  }

  // Micro-taper on phrase entrances/exits
  const gOrig = new Float32Array(gainGrid);
  const fade3 = [0.35, 0.72, 0.93];
  for (let f = 1; f < totalFrames - 1; f++) {
    if (gOrig[f] > 0.0 && gOrig[f - 1] === 0.0) {
      if (plosiveGrid[f] > 0.0) gainGrid[f] *= 0.65;
      else for (let kf = 0; kf < 3 && f + kf < totalFrames; kf++) if (gOrig[f + kf] > 0.0) gainGrid[f + kf] *= fade3[kf];
    }
    if (gOrig[f] > 0.0 && gOrig[f + 1] === 0.0) {
      for (let kf = 0; kf < 3 && f - kf >= 0; kf++) if (gOrig[f - kf] > 0.0) gainGrid[f - kf] *= fade3[kf];
    }
  }
  bridgeShortF0Gaps(f0Grid, 32);

  // Pass 2: Tracks 01-03 Direct 64-ERB Harmonic Synthesis with Rosenberg Glottal Pulse Phase
  const nFft = 1024;
  const halfW = nFft >> 1;
  const nBins = halfW + 1;
  const bufLen = nSamples + nFft + hop;
  const subMono = new Float32Array(bufLen);
  const bodyMono = new Float32Array(bufLen);
  const sheenMono = new Float32Array(bufLen);
  const phiArr = new Float32Array(bufLen);
  const periodArr = new Int32Array(bufLen);

  // Dynamic pitch & effort Rosenberg glottal pulse phases (soft chest Oq=0.68 <-> pressed belt Oq=0.44)
  const glSoft = buildRosenbergGlottalPhases(maxHarm, 0.47, 0.68);
  const glPress = buildRosenbergGlottalPhases(maxHarm, 0.38, 0.55);
  const cosPsi = Float64Array.from({ length: maxHarm }, (_, h) => Math.cos(glottPsi[h] || 0.0));
  const sinPsi = Float64Array.from({ length: maxHarm }, (_, h) => Math.sin(glottPsi[h] || 0.0));
  const sharpErb = new Float64Array(nErb);

  const subA = new Float64Array(maxHarm);
  const bodyA = new Float64Array(maxHarm);
  const sheenA = new Float64Array(maxHarm);
  const subB = new Float64Array(maxHarm);
  const bodyB = new Float64Array(maxHarm);
  const sheenB = new Float64Array(maxHarm);
  const dbH = new Float64Array(maxHarm);
  const relA = new Float64Array(maxHarm);
  const hfClampDb = config.hfClampDb ?? 2.2;

  function evalLeadFrame(fIdx, outSub, outBody, outSheen) {
    outSub.fill(0.0);
    outBody.fill(0.0);
    outSheen.fill(0.0);
    const vMix = Math.max(0.0, Math.min(1.0, (vdegGrid[fIdx] - 0.1) / 0.55));
    const gH = gainGrid[fIdx] * Math.sqrt(vMix);
    const ff = f0Grid[fIdx];
    if (gH <= 1e-7 || ff <= 60.0) return 0;
    const nAct = Math.min(maxHarm, Math.floor(19200.0 / ff));
    if (nAct <= 0) return 0;

    // Relaxed ERB formant contour + subtle chest warmth & anti-pinch relief
    const eOff = fIdx * nErb;
    for (let e = 0; e < nErb; e++) {
      let val = erbGrid[eOff + e];
      const fE = erbHz[e];
      if (fE >= 180.0 && fE <= 460.0) {
        const uWarm = (fE - 300.0) / 140.0;
        val += 0.8 * Math.exp(-0.5 * uWarm * uWarm);
      } else if (fE >= 1500.0 && fE <= 2700.0) {
        const uPinch = (fE - 2100.0) / 420.0;
        val -= 0.5 * Math.exp(-0.5 * uPinch * uPinch);
      }
      if (fE >= 350.0 && fE <= 4200.0) {
        let sumN = 0.0;
        let cntN = 0;
        for (let k = Math.max(0, e - 2); k <= Math.min(nErb - 1, e + 2); k++) {
          sumN += erbGrid[eOff + k];
          cntN++;
        }
        const localM = sumN / cntN;
        sharpErb[e] = val + 0.06 * Math.max(-6.0, Math.min(6.0, val - localM));
      } else {
        sharpErb[e] = val;
      }
    }

    // Update relaxed glottal phase + natural high-frequency phase dispersion above 1.4 kHz
    const press = Math.max(0.0, Math.min(1.0, 0.40 * ((ff - 210.0) / 240.0) + 0.45 * (gH / (0.34 * refPeak))));
    for (let h = 0; h < nAct; h++) {
      const cx = (1.0 - press) * glSoft.cosA[h] + press * glPress.cosA[h];
      const sx = (1.0 - press) * glSoft.sinA[h] + press * glPress.sinA[h];
      const invR = 1.0 / (Math.hypot(cx, sx) + 1e-12);
      const fh = (h + 1) * ff;
      if (fh > 1400.0) {
        const disp = Math.min(1.0, (fh - 1400.0) / 3600.0) * 0.48 * Math.sin((h + 1) * 1.618034);
        const cd = Math.cos(disp);
        const sd = Math.sin(disp);
        const c0 = cx * invR;
        const s0 = sx * invR;
        cosPsi[h] = c0 * cd - s0 * sd;
        sinPsi[h] = s0 * cd + c0 * sd;
      } else {
        cosPsi[h] = cx * invR;
        sinPsi[h] = sx * invR;
      }
    }

    let lowRefDb = -120.0;
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      const u = Math.max(0.0, Math.min(nErb - 1.0001, (hzToErb(fh) - erbMin) * invErbSpan));
      const i0 = u | 0;
      const fr = u - i0;
      const d = (1.0 - fr) * sharpErb[i0] + fr * sharpErb[i0 + 1];
      dbH[h] = d;
      if (h < 4 && d > lowRefDb) lowRefDb = d;
    }

    let sumSq = 0.0;
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      let d = dbH[h];
      if (fh > 1300.0 && fh <= 4500.0) {
        d += 1.6 * ((fh - 1300.0) / 3200.0);
      } else if (fh > 4500.0 && fh <= 11500.0) {
        d += 1.6 + 0.35 * Math.min(1.0, (fh - 4500.0) / 3000.0);
      }
      if (fh > 4800.0) {
        const maxHiDb = lowRefDb - (hfClampDb + 3.0 * Math.log2(fh / 4800.0));
        if (d > maxHiDb) d = maxHiDb;
      }
      const hfTap = Math.max(0.0, Math.min(1.0, (19200.0 - fh) / 1800.0));
      const a = Math.pow(10.0, d / 20.0) * hfTap;
      relA[h] = a;
      sumSq += a * a;
    }
    const scale = gH / Math.sqrt(0.5 * sumSq + 1e-18);
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      const amp = relA[h] * scale;
      const rSub = fh / 720.0;
      const rSub6 = rSub * rSub * rSub * rSub * rSub * rSub;
      const wSub = 1.0 / (1.0 + rSub6);
      const rHi = 3200.0 / fh;
      const rHi6 = rHi * rHi * rHi * rHi * rHi * rHi;
      const wHi = 1.0 / (1.0 + rHi6);
      const wMid = Math.max(0.0, 1.0 - wSub - wHi);
      outSub[h] = amp * wSub;
      outBody[h] = amp * wMid;
      outSheen[h] = amp * wHi;
    }
    return nAct;
  }

  const twoPiOverSr = TWO_PI / sr;
  const invHop = 1.0 / hop;
  const nHops = Math.floor(nSamples / hop) + 1;
  let phi = 0.0;
  let nActA = evalLeadFrame(0, subA, bodyA, sheenA);
  const rngJit = createMulberry32(20260711);
  let jitState = 0.0;

  for (let f = 0; f < nHops; f++) {
    const nActB = evalLeadFrame(f + 1, subB, bodyB, sheenB);
    const f0A = f0Grid[f];
    const f0B = f0Grid[f + 1];
    if (f0A <= 60.0 && f0B <= 60.0) {
      phi = 0.0;
      subA.set(subB);
      bodyA.set(bodyB);
      sheenA.set(sheenB);
      nActA = nActB;
      continue;
    }
    let fS = f0A;
    let fE = f0B;
    if (f0A <= 60.0 && f0B > 60.0) {
      phi = 0.0;
      fS = f0B;
      fE = f0B;
    } else if (f0A > 60.0 && f0B <= 60.0) {
      fS = f0A;
      fE = f0A;
    }

    const base = f * hop;
    const df = (fE - fS) * invHop;
    const nH = Math.max(nActA, nActB);
    const jitDepth = vdegGrid[f] > 0.72 && gainGrid[f] > 0.08 * refPeak ? 1.0 : 0.0;
    for (let s = 0; s < hop; s++) {
      jitState += 0.0042 * ((rngJit() - 0.5) * 2.0 - jitState);
      const fc = (fS + df * s) * (1.0 + 0.0019 * jitDepth * jitState);
      phi += twoPiOverSr * fc;
      if (phi >= TWO_PI) phi -= TWO_PI;
      phiArr[base + s] = phi;
      periodArr[base + s] = Math.round(sr / fc);
      if (nH > 0) {
        const frac = s * invHop;
        const c1 = Math.cos(phi);
        const s1 = Math.sin(phi);
        const twoC1 = 2.0 * c1;
        let chPrev = 1.0;
        let shPrev = 0.0;
        let ch = c1;
        let sh = s1;
        let accSub = 0.0;
        let accBody = 0.0;
        let accSheen = 0.0;
        for (let h = 0; h < nH; h++) {
          const wv = ch * cosPsi[h] - sh * sinPsi[h];
          accSub += (subA[h] + (subB[h] - subA[h]) * frac) * wv;
          accBody += (bodyA[h] + (bodyB[h] - bodyA[h]) * frac) * wv;
          accSheen += (sheenA[h] + (sheenB[h] - sheenA[h]) * frac) * wv;
          const chNext = twoC1 * ch - chPrev;
          const shNext = twoC1 * sh - shPrev;
          chPrev = ch;
          shPrev = sh;
          ch = chNext;
          sh = shNext;
        }
        const shim = 1.0 + 0.011 * jitDepth * jitState;
        subMono[base + s] = accSub * shim;
        bodyMono[base + s] = accBody * shim;
        sheenMono[base + s] = accSheen * shim;
      }
    }
    subA.set(subB);
    bodyA.set(bodyB);
    sheenA.set(sheenB);
    nActA = nActB;
  }

  // Pass 3: Tracks 04-06 High-Definition 1024-pt Kolmogorov Real-Cepstrum Minimum-Phase WOLA
  const rngUnv = createMulberry32(20261117);
  const eUnv = new Float32Array(bufLen);
  const sqrt6 = Math.sqrt(6.0);
  for (let i = 0; i < bufLen; i++) eUnv[i] = (rngUnv() + rngUnv() - 1.0) * sqrt6;
  dcBlock60HzInPlace(eUnv, sr);

  const rngBr = createMulberry32(20260921);
  const wMod = new Float32Array(bufLen);
  const normMod = 1.0 / Math.sqrt(1.0 + 0.5 * 0.42 * 0.42);
  for (let i = 0; i < bufLen; i++) {
    const wn = (rngBr() + rngBr() - 1.0) * sqrt6;
    const mod = periodArr[i] > 0 ? (1.0 + 0.42 * Math.cos(phiArr[i])) * normMod : 1.0;
    wMod[i] = wn * mod;
  }
  const eBreath = new Float32Array(wMod);
  const normComb = 1.0 / Math.sqrt(1.0 + 0.58 * 0.58);
  for (let i = 0; i < bufLen; i++) {
    const T0 = periodArr[i];
    if (T0 > 0 && i >= T0) eBreath[i] = (wMod[i] - 0.58 * wMod[i - T0]) * normComb;
  }
  dcBlock60HzInPlace(eBreath, sr);

  // At 75% overlap (nFft=1024, hop=256), wAna * wSyn sums to 0.5 * 2.0 = 1.0 identically!
  const wAna = Float64Array.from({ length: nFft }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / nFft));
  const wSyn = Float64Array.from({ length: nFft }, (_, i) => 0.5 * Math.sin((Math.PI * (i + 0.5)) / nFft));

  const freqs = Float64Array.from({ length: nBins }, (_, k) => (k * sr) / nFft);
  const logFreqs = Float64Array.from({ length: nBins }, (_, k) => Math.log(Math.max(freqs[k], 1.0)));
  const binErbPos = Float64Array.from({ length: nBins }, (_, k) =>
    Math.max(0.0, Math.min(nErb - 1.0001, (hzToErb(Math.max(80.0, freqs[k])) - erbMin) * invErbSpan))
  );
  const bandTaper = Float64Array.from({ length: nBins }, (_, k) => {
    const f = freqs[k];
    const lo = Math.max(0.0, Math.min(1.0, (f - 65.0) / 55.0));
    const hi = Math.max(0.0, Math.min(1.0, (20200.0 - f) / 1400.0));
    return lo * hi;
  });
  const sibHp = Float64Array.from({ length: nBins }, (_, k) => 0.25 + 0.75 / (1.0 + Math.pow(360.0 / Math.max(freqs[k], 1.0), 3.0)));

  const aperEdges = new Float64Array(nAper + 1);
  const logRatioA = Math.log(speakerModel.aperFreqMax / speakerModel.aperFreqMin) / nAper;
  for (let b = 0; b <= nAper; b++) aperEdges[b] = speakerModel.aperFreqMin * Math.exp(b * logRatioA);
  const logAperC = Float64Array.from({ length: nAper }, (_, b) => 0.5 * (Math.log(aperEdges[b]) + Math.log(aperEdges[b + 1])));

  const fft1024 = new Radix2ComplexFFT(nFft);
  const cepRe = new Float64Array(nFft);
  const cepIm = new Float64Array(nFft);
  const reW = new Float64Array(nFft);
  const imW = new Float64Array(nFft);
  const hEnv = new Float64Array(nBins);
  const cosPh = new Float64Array(nBins);
  const sinPh = new Float64Array(nBins);

  const plosMono = new Float32Array(bufLen);
  const sibMono = new Float32Array(bufLen);
  const breathMono = new Float32Array(bufLen);
  const twoOverNfft = 2.0 / nFft;
  const dbToLn = Math.LN10 / 20.0;
  const breathScale = Math.pow(10.0, (config.breathGainDb ?? 3.0) / 20.0);

  // Deterministic physical plosive release burst (b/p/d/t/g/k crisp lip/tongue pop)
  const popLen = Math.round(0.0032 * sr);
  const popTau = 0.00075 * sr;
  const popW1 = (TWO_PI * 1350.0) / sr;
  const popW2 = (TWO_PI * 2650.0) / sr;
  for (let idx = 0; idx < events.length; idx++) {
    const ev = events[idx];
    if (!ev.pl || ev.L < 4) continue;
    const relF = ev.s + Math.max(0, Math.min(ev.L - 1, ev.cf - 1));
    const gPop = Math.min(0.14, gainGrid[relF] * 0.30);
    if (gPop <= 1e-4) continue;
    const s0 = relF * hop;
    if (s0 + popLen >= nSamples) continue;
    for (let i = 0; i < popLen; i++) {
      const env = Math.sin(Math.min(1.0, i / 14.0) * Math.PI * 0.5) * Math.exp(-i / popTau);
      plosMono[s0 + i] += gPop * env * (0.65 * Math.sin(popW1 * i) + 0.35 * Math.sin(popW2 * i));
    }
  }

  for (let f = 0; f < totalFrames; f++) {
    const gTot = gainGrid[f];
    if (gTot <= 1e-7) continue;
    const center = f * hop;
    if (center >= nSamples) break;
    const start = center - halfW;
    if (start < 0 || start + nFft > bufLen) continue;

    const vd = vdegGrid[f];
    const vMix = Math.max(0.0, Math.min(1.0, (vd - 0.1) / 0.55));
    // Gate broadband unvoiced noise out of voiced vowels (vd >= 0.62) while preserving consonant transitions
    const unvGate = vd >= 0.62 ? 0.0 : Math.min(1.0, (0.62 - vd) / 0.35);
    const gUnv = gTot * Math.sqrt(1.0 - vMix) * unvGate;
    const gHarm = gTot * Math.sqrt(vMix);
    const fVal = f0Grid[f];
    if (gUnv <= 1e-7 && (gHarm <= 1e-7 || fVal <= 60.0)) continue;

    // Exact 1024-pt Kolmogorov real-cepstrum minimum phase from 64-ERB log envelope
    const eOff = f * nErb;
    cepIm.fill(0.0);
    for (let k = 0; k < nBins; k++) {
      const u = binErbPos[k];
      const i0 = u | 0;
      const fr = u - i0;
      const dDb = (1.0 - fr) * erbGrid[eOff + i0] + fr * erbGrid[eOff + i0 + 1];
      const lnMag = Math.max(-14.0, Math.min(7.5, dDb * dbToLn));
      cepRe[k] = lnMag;
      if (k > 0 && k < halfW) cepRe[nFft - k] = lnMag;
      hEnv[k] = Math.exp(lnMag) * bandTaper[k];
    }
    fft1024.transform(cepRe, cepIm, true);
    for (let n = 1; n < halfW; n++) cepRe[n] *= 2.0;
    for (let n = halfW + 1; n < nFft; n++) cepRe[n] = 0.0;
    cepIm.fill(0.0);
    fft1024.transform(cepRe, cepIm, false);
    for (let k = 0; k < nBins; k++) {
      cosPh[k] = Math.cos(cepIm[k]);
      sinPh[k] = Math.sin(cepIm[k]);
    }

    if (gUnv > 1e-7) {
      const pFrac = plosiveGrid[f];
      let powU = 0.0;
      for (let k = 1; k < nBins; k++) {
        const hu = hEnv[k] * (pFrac + (1.0 - pFrac) * sibHp[k]);
        powU += hu * hu;
      }
      const scaleU = gUnv / Math.sqrt(twoOverNfft * powU + 1e-18);
      for (let i = 0; i < nFft; i++) {
        reW[i] = eUnv[start + i] * wAna[i];
        imW[i] = 0.0;
      }
      fft1024.transform(reW, imW, false);
      reW[0] = 0.0;
      imW[0] = 0.0;
      for (let k = 1; k < nBins; k++) {
        const mag = scaleU * hEnv[k] * (pFrac + (1.0 - pFrac) * sibHp[k]);
        const hr = mag * cosPh[k];
        const hi = mag * sinPh[k];
        const xr = reW[k];
        const xi = imW[k];
        const yr = xr * hr - xi * hi;
        const yi = xr * hi + xi * hr;
        reW[k] = yr;
        imW[k] = yi;
        if (k < halfW) {
          reW[nFft - k] = yr;
          imW[nFft - k] = -yi;
        }
      }
      fft1024.transform(reW, imW, true);
      for (let i = 0; i < nFft; i++) {
        const yu = reW[i] * wSyn[i];
        if (pFrac > 0.0) plosMono[start + i] += pFrac * yu;
        if (pFrac < 1.0) sibMono[start + i] += (1.0 - pFrac) * yu;
      }
    }

    if (gHarm > 1e-7 && fVal > 60.0) {
      let powE = 0.0;
      for (let k = 1; k < nBins; k++) powE += hEnv[k] * hEnv[k];
      const invNormE = gHarm / Math.sqrt(twoOverNfft * powE + 1e-18);
      for (let i = 0; i < nFft; i++) {
        reW[i] = eBreath[start + i] * wAna[i];
        imW[i] = 0.0;
      }
      fft1024.transform(reW, imW, false);
      reW[0] = 0.0;
      imW[0] = 0.0;
      const apOff = f * nAper;
      let segA = 0;
      for (let k = 1; k < nBins; k++) {
        const lf = logFreqs[k];
        while (segA < nAper - 2 && lf > logAperC[segA + 1]) segA++;
        let apLog;
        if (lf <= logAperC[0]) apLog = aperGrid[apOff];
        else if (lf >= logAperC[nAper - 1]) apLog = aperGrid[apOff + nAper - 1];
        else {
          const fr = (lf - logAperC[segA]) / (logAperC[segA + 1] - logAperC[segA]);
          apLog = (1.0 - fr) * aperGrid[apOff + segA] + fr * aperGrid[apOff + segA + 1];
        }
        const hnrAmp = Math.pow(10.0, 0.5 * apLog) * breathScale;
        const subCut = Math.max(0.0, Math.min(1.0, (freqs[k] - 0.85 * fVal) / Math.max(0.35 * fVal, 35.0)));
        const mag = invNormE * hEnv[k] * hnrAmp * subCut;
        const hr = mag * cosPh[k];
        const hi = mag * sinPh[k];
        const xr = reW[k];
        const xi = imW[k];
        const yr = xr * hr - xi * hi;
        const yi = xr * hi + xi * hr;
        reW[k] = yr;
        imW[k] = yi;
        if (k < halfW) {
          reW[nFft - k] = yr;
          imW[nFft - k] = -yi;
        }
      }
      fft1024.transform(reW, imW, true);
      for (let i = 0; i < nFft; i++) breathMono[start + i] += reW[i] * wSyn[i];
    }
  }

  // Pass 4: Track 07 True Independent Left/Right Stereo Harmony Voices
  const backL = synthesizeBackingVoiceChannel(backingModel, backingScore.leftNotes, nSamples, sr, glottPsi);
  const backR = synthesizeBackingVoiceChannel(backingModel, backingScore.rightNotes, nSamples, sr, glottPsi);
  const backGain = config.backingGain ?? 1.0;
  if (backGain !== 1.0) {
    for (let i = 0; i < nSamples; i++) {
      backL[i] *= backGain;
      backR[i] *= backGain;
    }
  }

  const trackSub = subMono.subarray(0, nSamples);
  const trackBody = bodyMono.subarray(0, nSamples);
  const trackSheen = sheenMono.subarray(0, nSamples);
  const trackPlos = plosMono.subarray(0, nSamples);
  const trackSib = sibMono.subarray(0, nSamples);
  const trackBreath = breathMono.subarray(0, nSamples);

  // Pass 5: Track 08 Stereo Early Reflections + 8-Line Prime FDN Space
  const spaceSend = new Float32Array(nSamples);
  const sibSend = config.spaceSibSend ?? 0.16;
  for (let i = 0; i < nSamples; i++) {
    spaceSend[i] = trackSub[i] + trackBody[i] + 0.5 * trackSheen[i] + sibSend * trackSib[i] + 0.22 * (backL[i] + backR[i]);
  }
  const [spaceL, spaceR] = synthesizeSpaceFdn(spaceFdn, spaceSend, config.spaceInLp ?? 0.44);
  const spGain = config.spaceGain ?? 2.0;
  for (let i = 0; i < nSamples; i++) {
    spaceL[i] *= spGain;
    spaceR[i] *= spGain;
  }

  return {
    layers: {
      lead_sub_f0: [new Float32Array(trackSub)],
      lead_vowel_body: [new Float32Array(trackBody)],
      lead_harmonic_sheen: [new Float32Array(trackSheen)],
      lead_plosive_burst: [new Float32Array(trackPlos)],
      lead_sibilant: [new Float32Array(trackSib)],
      lead_velvet_breath: [new Float32Array(trackBreath)],
      backing_harmony: [backL, backR],
      vocal_space: [spaceL, spaceR],
    },
    uiEvents,
    prototypesCount: prototypes.length,
  };
}
