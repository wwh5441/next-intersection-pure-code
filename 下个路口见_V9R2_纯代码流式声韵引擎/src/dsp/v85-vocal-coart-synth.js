// V8.5 -> V9R2 8-Track Coarticulated Vocal Anatomy & Syllable-Atlas Synthesizer (`src/dsp/v85-vocal-coart-synth.js`)
// 100% Plain-Text Mathematical Equations — Zero Binary Blobs

import { Radix2ComplexFFT, createMulberry32 } from './fft-core.js';

const DB_FLOOR = -78.0;
const DB_CEIL = 4.0;
const CONS_ANCHORS_LEN = 3;
const VOWEL_ANCHORS_LEN = 5;
const TWO_PI = 2.0 * Math.PI;
const FDN_DELAYS_48K = [1499, 1777, 2081, 2437, 2801, 3217, 3671, 4177];

function uint8ToGainScalar(q, refPeak) {
  if (q <= 0) return 0.0;
  const db = DB_FLOOR + ((q - 1.0) / 254.0) * (DB_CEIL - DB_FLOOR);
  return refPeak * Math.pow(10.0, db / 20.0);
}

function uint8ArrToGain(qArr, refPeak) {
  const out = new Float32Array(qArr.length);
  for (let i = 0; i < qArr.length; i++) {
    out[i] = uint8ToGainScalar(qArr[i], refPeak);
  }
  return out;
}

export function buildEnvelopeKnotPositions(totalFrames, consFrames, vowelStep) {
  if (totalFrames <= 1) return Int32Array.of(0);
  const last = totalFrames - 1;
  const vot = Math.min(last, Math.max(1, consFrames));
  const step = Math.max(1, vowelStep);
  const coda = Math.max(vot, last - Math.max(2, Math.floor(step / 2)));
  const set = new Set([0, Math.floor(vot / 2), vot, coda, last]);
  let cur = vot + step;
  while (cur < last) {
    set.add(cur);
    cur += step;
  }
  return Int32Array.from([...set].sort((a, b) => a - b));
}

export function buildAdaptiveExprKnotPositions(totalFrames, consFrames, exprStep) {
  if (totalFrames <= 1) return Int32Array.of(0);
  const last = totalFrames - 1;
  const vot = Math.min(last, Math.max(1, consFrames));
  const step = Math.max(1, exprStep);
  if (step === 1) {
    const all = new Int32Array(totalFrames);
    for (let i = 0; i < totalFrames; i++) all[i] = i;
    return all;
  }
  const halfS = Math.max(2, Math.floor(step / 2));
  const glide = Math.min(last, vot + halfS);
  const coda = Math.max(glide, last - halfS);
  const set = new Set([0, Math.floor(vot / 2), vot, glide, coda, last]);
  let cur = glide + step;
  while (cur < last) {
    set.add(cur);
    cur += step;
  }
  return Int32Array.from([...set].sort((a, b) => a - b));
}

export function buildUniformKnotPositions(totalFrames, step) {
  if (totalFrames <= 1) return Int32Array.of(0);
  const s = Math.max(1, step);
  if (s === 1) {
    const all = new Int32Array(totalFrames);
    for (let i = 0; i < totalFrames; i++) all[i] = i;
    return all;
  }
  const last = totalFrames - 1;
  const pts = [];
  for (let p = 0; p < last; p += s) pts.push(p);
  if (pts.length === 0 || pts[pts.length - 1] !== last) pts.push(last);
  return Int32Array.from(pts);
}

function interp1dKnots(knotPos, knotVals, totalFrames) {
  if (knotPos.length === totalFrames) return Float32Array.from(knotVals);
  const out = new Float32Array(totalFrames);
  if (totalFrames <= 1 || knotPos.length <= 1) {
    out.fill(knotVals.length ? knotVals[0] : 0.0);
    return out;
  }
  let seg = 0;
  const maxSeg = knotPos.length - 2;
  for (let x = 0; x < totalFrames; x++) {
    while (seg < maxSeg && x > knotPos[seg + 1]) seg++;
    const x0 = knotPos[seg];
    const x1 = knotPos[seg + 1];
    const frac = x1 > x0 ? (x - x0) / (x1 - x0) : 0.0;
    out[x] = (1.0 - frac) * knotVals[seg] + frac * knotVals[seg + 1];
  }
  return out;
}

function interp2dKnots(knotPos, knotMat, cols, totalFrames) {
  if (knotPos.length === totalFrames) return knotMat;
  const out = new Float32Array(totalFrames * cols);
  if (totalFrames <= 1 || knotPos.length <= 1) {
    for (let x = 0; x < totalFrames; x++) {
      for (let c = 0; c < cols; c++) out[x * cols + c] = knotMat[c] || 0.0;
    }
    return out;
  }
  let seg = 0;
  const maxSeg = knotPos.length - 2;
  for (let x = 0; x < totalFrames; x++) {
    while (seg < maxSeg && x > knotPos[seg + 1]) seg++;
    const x0 = knotPos[seg];
    const x1 = knotPos[seg + 1];
    const frac = x1 > x0 ? (x - x0) / (x1 - x0) : 0.0;
    const invF = 1.0 - frac;
    const r0 = seg * cols;
    const r1 = (seg + 1) * cols;
    const dst = x * cols;
    for (let c = 0; c < cols; c++) {
      out[dst + c] = invF * knotMat[r0 + c] + frac * knotMat[r1 + c];
    }
  }
  return out;
}

function reconstructMidiKnots(baseMidiQ, f0Steps) {
  const n = 1 + f0Steps.length;
  const out = new Float32Array(n);
  let cur = baseMidiQ * 0.02;
  out[0] = cur;
  for (let k = 0; k < f0Steps.length; k++) {
    cur += f0Steps[k] * 0.025;
    out[k + 1] = cur;
  }
  return out;
}

function buildWarpedCosineLut(nCep, nLut = 4096) {
  const lut = new Float32Array(nLut * nCep);
  const invMax = 1.0 / (nLut - 1);
  for (let i = 0; i < nLut; i++) {
    const u = i * invMax;
    const rowOff = i * nCep;
    for (let m = 0; m < nCep; m++) {
      lut[rowOff + m] = Math.cos(Math.PI * (m + 1) * u);
    }
  }
  return lut;
}

function precomputeUnvoicedMinphaseTables(config) {
  const sr = config.sampleRate;
  const nFft = config.fftSize; // 512
  const nBins = (nFft >> 1) + 1;
  const nCep = config.cepstralCoeffs;
  const fMin = config.harmFreqMin;
  const fMax = config.harmFreqMax;

  const e0 = 21.4 * Math.log10(1.0 + 0.00437 * fMin);
  const e1 = 21.4 * Math.log10(1.0 + 0.00437 * fMax);
  const invESpan = 1.0 / (e1 - e0);
  const ln10Over20 = Math.LN10 / 20.0;

  const freqs = new Float64Array(nBins);
  const logFreqs = new Float64Array(nBins);
  const bandTaper = new Float32Array(nBins);
  const sibHp = new Float32Array(nBins);
  const bLn = new Float32Array(nCep * nBins);
  const phiRad = new Float32Array(nCep * nBins);

  for (let k = 0; k < nBins; k++) {
    const fk = (k * sr) / nFft;
    freqs[k] = fk;
    logFreqs[k] = Math.log(Math.max(fk, 75.0));
    if (k > 0) {
      const hiT = Math.max(0.0, Math.min(1.0, (19500.0 - fk) / 3000.0));
      const loT = Math.max(0.0, Math.min(1.0, (fk - 95.0) / 90.0));
      bandTaper[k] = hiT * loT;
      sibHp[k] = Math.max(0.15, Math.min(1.0, (fk - 900.0) / 1100.0));
    }
    const fClamp = Math.max(fMin, Math.min(fMax, fk));
    const uBin = (21.4 * Math.log10(1.0 + 0.00437 * fClamp) - e0) * invESpan;
    for (let m = 0; m < nCep; m++) {
      const ang = Math.PI * (m + 1) * uBin;
      bLn[m * nBins + k] = ln10Over20 * Math.cos(ang);
      phiRad[m * nBins + k] = -ln10Over20 * Math.sin(ang);
    }
  }
  return { bLn, phiRad, bandTaper, sibHp, freqs, logFreqs, e0, invESpan };
}

function dcBlock60HzInPlace(x, sr) {
  const rDc = Math.exp((-TWO_PI * 60.0) / sr);
  const gDc = 0.5 * (1.0 + rDc);
  let prevX = 0.0;
  let prevY = 0.0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = gDc * (xi - prevX) + rDc * prevY;
    prevX = xi;
    prevY = yi;
    x[i] = yi;
  }
}

function smoothFrameTrack(track) {
  const n = track.length;
  const tmp = new Float32Array(n);
  const out = new Float32Array(n);
  for (let pass = 0; pass < 2; pass++) {
    const src = pass === 0 ? track : tmp;
    const dst = pass === 0 ? tmp : out;
    for (let i = 0; i < n; i++) {
      const xm2 = src[Math.max(0, i - 2)];
      const xm1 = src[Math.max(0, i - 1)];
      const x0 = src[i];
      const xp1 = src[Math.min(n - 1, i + 1)];
      const xp2 = src[Math.min(n - 1, i + 2)];
      dst[i] = 0.08 * (xm2 + xp2) + 0.17 * (xm1 + xp1) + 0.5 * x0;
    }
  }
  return out;
}

function bridgeShortF0Gaps(f0Grid, maxGapFrames = 36) {
  let prevVoiced = -1;
  for (let f = 0; f < f0Grid.length; f++) {
    if (f0Grid[f] > 60.0) {
      if (prevVoiced >= 0) {
        const gap = f - prevVoiced - 1;
        if (gap >= 1 && gap <= maxGapFrames) {
          const va = f0Grid[prevVoiced];
          const vb = f0Grid[f];
          if (Math.abs(Math.log2(vb / va)) < 0.75) {
            for (let g = 1; g <= gap; g++) {
              f0Grid[prevVoiced + g] = va + ((vb - va) * g) / (gap + 1);
            }
          }
        }
      }
      prevVoiced = f;
    }
  }
}

function evalPrototypeOnEvent(proto, meanMcep, aperMean, nCep, nAper, totalFrames, consFrames, qScale) {
  const L = totalFrames;
  const vot = L > 1 && consFrames > 0 ? Math.min(L - 1, consFrames) : 0;
  const predM = new Float32Array(L * nCep);
  const predA = new Float32Array(L * nAper);

  if (vot > 0) {
    for (let i = 0; i < vot; i++) {
      const u = vot > 1 ? (i / (vot - 1)) * (CONS_ANCHORS_LEN - 1) : 0.0;
      const i0 = Math.min(CONS_ANCHORS_LEN - 2, Math.floor(u));
      const i1 = i0 + 1;
      const f = u - i0;
      const invF = 1.0 - f;
      const cm0 = proto.consMcep[i0];
      const cm1 = proto.consMcep[i1];
      const ca0 = proto.consAper[i0];
      const ca1 = proto.consAper[i1];
      const mOff = i * nCep;
      const aOff = i * nAper;
      for (let c = 0; c < nCep; c++) {
        predM[mOff + c] = meanMcep[c] + (invF * cm0[c] + f * cm1[c]) * qScale;
      }
      for (let b = 0; b < nAper; b++) {
        predA[aOff + b] = aperMean[b] + (invF * ca0[b] + f * ca1[b]) * qScale;
      }
    }
  }

  const vLen = L - vot;
  if (vLen > 0) {
    for (let j = 0; j < vLen; j++) {
      const i = vot + j;
      const u = vLen > 1 ? (j / (vLen - 1)) * (VOWEL_ANCHORS_LEN - 1) : 0.0;
      const i0 = Math.min(VOWEL_ANCHORS_LEN - 2, Math.floor(u));
      const i1 = i0 + 1;
      const f = u - i0;
      const invF = 1.0 - f;
      const vm0 = proto.vowelMcep[i0];
      const vm1 = proto.vowelMcep[i1];
      const va0 = proto.vowelAper[i0];
      const va1 = proto.vowelAper[i1];
      const mOff = i * nCep;
      const aOff = i * nAper;
      for (let c = 0; c < nCep; c++) {
        predM[mOff + c] = meanMcep[c] + (invF * vm0[c] + f * vm1[c]) * qScale;
      }
      for (let b = 0; b < nAper; b++) {
        predA[aOff + b] = aperMean[b] + (invF * va0[b] + f * va1[b]) * qScale;
      }
    }
  }
  return { predM, predA };
}

function buildCoarticulatedPrototypeGrids(events, prototypes, atlasStepDb, meanMcep, aperMean, nCep, nAper, totalFramesSong) {
  const baseCep = new Float32Array(totalFramesSong * nCep);
  const baseAper = new Float32Array(totalFramesSong * nAper);
  baseAper.fill(-3.6);

  for (let eIdx = 0; eIdx < events.length; eIdx++) {
    const ev = events[eIdx];
    const L = ev.totalFrames;
    if (L <= 0) continue;
    const sF = ev.startFrame;
    const fEnd = Math.min(totalFramesSong, sF + L);
    const span = fEnd - sF;
    if (span <= 0) continue;
    const { predM, predA } = evalPrototypeOnEvent(
      prototypes[ev.syllableId],
      meanMcep,
      aperMean,
      nCep,
      nAper,
      L,
      ev.consFrames,
      atlasStepDb
    );
    baseCep.set(predM.subarray(0, span * nCep), sF * nCep);
    baseAper.set(predA.subarray(0, span * nAper), sF * nAper);
  }

  // Bidirectional DAC `--` Anticipatory Coarticulation across contiguous syllable seams
  const seams = [];
  for (let i = 1; i < events.length; i++) {
    const prev = events[i - 1];
    const curr = events[i];
    const b = curr.startFrame;
    if (prev.totalFrames >= 2 && curr.totalFrames >= 2 && prev.startFrame + prev.totalFrames === b && b >= 1 && b < totalFramesSong) {
      const wPre = Math.min(18, Math.floor((prev.totalFrames * 12) / 25), b);
      const wPost = Math.min(curr.isPlosive ? 4 : 10, Math.floor((curr.totalFrames * 9) / 25), totalFramesSong - b);
      if (wPre >= 1 && wPost >= 1) {
        const dCep = new Float32Array(nCep);
        const dAper = new Float32Array(nAper);
        const bM = b * nCep;
        const bmM = (b - 1) * nCep;
        const bA = b * nAper;
        const bmA = (b - 1) * nAper;
        for (let c = 0; c < nCep; c++) dCep[c] = baseCep[bM + c] - baseCep[bmM + c];
        for (let a = 0; a < nAper; a++) dAper[a] = baseAper[bA + a] - baseAper[bmA + a];
        seams.push({ b, wPre, wPost, dCep, dAper });
      }
    }
  }

  for (let s = 0; s < seams.length; s++) {
    const { b, wPre, wPost, dCep, dAper } = seams[s];
    const invTot = 1.0 / (wPre + wPost);
    const gPre = wPre * invTot;
    const gPost = wPost * invTot;
    for (let r = 0; r < wPre; r++) {
      const w = 0.5 * gPre * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPre));
      const mOff = (b - 1 - r) * nCep;
      const aOff = (b - 1 - r) * nAper;
      for (let c = 0; c < nCep; c++) baseCep[mOff + c] += w * dCep[c];
      for (let a = 0; a < nAper; a++) baseAper[aOff + a] += w * dAper[a];
    }
    for (let r = 0; r < wPost; r++) {
      const w = 0.5 * gPost * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPost));
      const mOff = (b + r) * nCep;
      const aOff = (b + r) * nAper;
      for (let c = 0; c < nCep; c++) baseCep[mOff + c] -= w * dCep[c];
      for (let a = 0; a < nAper; a++) baseAper[aOff + a] -= w * dAper[a];
    }
  }

  return { baseCep, baseAper };
}

export function synthesizeSpaceFdnPlain(spaceModel, dryMono) {
  const { sampleRate: sr, numSamples: nSamples, envHop, rt60Sec, dampLp, baseWetGain, preDelaySamples: preDelay, qMid, qSide } = spaceModel;
  const nEnv = qMid.length;
  const gmEnv = new Float32Array(nEnv);
  const gsEnv = new Float32Array(nEnv);
  for (let i = 0; i < nEnv; i++) {
    gmEnv[i] = (qMid[i] / 255.0) * baseWetGain;
    gsEnv[i] = (qSide[i] / 255.0) * baseWetGain;
  }

  const delays = FDN_DELAYS_48K;
  const bLps = delays.map((d) => (1.0 - dampLp) * Math.pow(10.0, (-3.0 * d) / (rt60Sec * sr)));
  const bufs = delays.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const lp = new Float64Array(8);

  const ringSize = 4096;
  const ringMask = ringSize - 1;
  const warmRing = new Float64Array(ringSize);

  const outL = new Float32Array(nSamples);
  const outR = new Float32Array(nSamples);

  let prevX = 0.0;
  let prevHp = 0.0;
  let prevInLp = 0.0;
  const hpAlpha = 0.973;
  const inLp = 0.54;
  const oneMinusInLp = 1.0 - inLp;
  const reads = new Float64Array(8);
  const damped = new Float64Array(8);
  const halfHop = envHop * 0.5;
  const dryLen = dryMono.length;

  for (let n = 0; n < nSamples; n++) {
    const xn = n < dryLen ? dryMono[n] : 0.0;
    const ynHp = hpAlpha * (prevHp + xn - prevX);
    prevX = xn;
    prevHp = ynHp;
    const yWarm = oneMinusInLp * ynHp + inLp * prevInLp;
    prevInLp = yWarm;

    const wPtr = n & ringMask;
    warmRing[wPtr] = yWarm;

    const yn = n >= preDelay ? warmRing[(n - preDelay) & ringMask] : 0.0;
    const erL =
      (n >= 542 ? 0.11 * warmRing[(n - 542) & ringMask] : 0.0) -
      (n >= 1186 ? 0.075 * warmRing[(n - 1186) & ringMask] : 0.0);
    const erR =
      (n >= 859 ? 0.105 * warmRing[(n - 859) & ringMask] : 0.0) -
      (n >= 1589 ? 0.07 * warmRing[(n - 1589) & ringMask] : 0.0);

    let sumD = 0.0;
    for (let i = 0; i < 8; i++) {
      const b = bufs[i];
      const p = ptrs[i];
      const r = b[p];
      reads[i] = r;
      const st = bLps[i] * r + dampLp * lp[i];
      lp[i] = st;
      damped[i] = st;
      sumD += st;
    }

    const house = 0.25 * sumD;
    const excScaled = 0.25 * yn;
    for (let i = 0; i < 8; i++) {
      const b = bufs[i];
      const p = ptrs[i];
      const sign = (i & 1) === 0 ? excScaled : -excScaled;
      b[p] = damped[i] - house + sign;
      ptrs[i] = p + 1 === delays[i] ? 0 : p + 1;
    }

    const rawM = 0.5 * (reads[0] + reads[2] + reads[4] + reads[6]);
    const rawS = 0.5 * (reads[1] - reads[3] + reads[5] - reads[7]);

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

    const mVal = rawM * gm;
    const sVal = rawS * gs;
    outL[n] = mVal + sVal + erL * gm;
    outR[n] = mVal - sVal + erR * gm;
  }

  return [outL, outR];
}

/**
 * Synthesizes V8.5 8-Track Vocal Anatomy (`01..06` 1ch center + `07..08` 2ch stereo)
 * from plain-text JS `singerCore`, `syllableAtlas`, `spaceFdn`, `vocalEvents`,
 */
export function synthesizeV85VocalAnatomy(
  singerCore,
  syllableAtlas,
  spaceFdn,
  vocalEventsObj,
  config
) {
  const sr = singerCore.sampleRate;
  const nSamples = singerCore.numSamples;
  const hop = singerCore.hop;
  const refPeak = singerCore.refPeak;
  const nFft = config.fftSize; // 512
  const halfW = nFft >> 1;
  const nBins = (nFft >> 1) + 1;
  const nCep = config.cepstralCoeffs;
  const nAper = config.aperiodicBands;
  const maxHarm = config.maxHarmonics;
  const fMin = config.harmFreqMin;
  const fMax = config.harmFreqMax;

  const meanMcep = Float32Array.from(singerCore.meanMcep);
  const aperMean = Float32Array.from(singerCore.aperMean);
  const glottPsi = Float64Array.from(singerCore.glottalPhase);
  const basisScale = 1.0 / singerCore.basisQuantDiv;

  const kPca = vocalEventsObj.pcaDims;
  const kAper = vocalEventsObj.aperExprBands;
  const exprStep = vocalEventsObj.exprKnotStepFrames;
  const f0Step = vocalEventsObj.f0KnotStepFrames;
  const envStep = vocalEventsObj.envKnotStepFrames;
  const qStep = vocalEventsObj.quantStepDb;

  const basisM = new Float32Array(kPca * nCep);
  for (let r = 0; r < kPca; r++) {
    const row = singerCore.mcepPcaBasisQ[r];
    for (let c = 0; c < nCep; c++) basisM[r * nCep + c] = row[c] * basisScale;
  }
  const basisA = new Float32Array(kAper * nAper);
  for (let r = 0; r < kAper; r++) {
    const row = singerCore.aperPcaBasisQ[r];
    for (let c = 0; c < nAper; c++) basisA[r * nAper + c] = row[c] * basisScale;
  }

  const prototypes = syllableAtlas.prototypes;
  const events = vocalEventsObj.events;
  const totalFramesSong = Math.floor(nSamples / hop) + 6;

  const { baseCep, baseAper } = buildCoarticulatedPrototypeGrids(
    events,
    prototypes,
    syllableAtlas.atlasStepDb,
    meanMcep,
    aperMean,
    nCep,
    nAper,
    totalFramesSong
  );

  const gainGrid = new Float32Array(totalFramesSong);
  const vdegGrid = new Float32Array(totalFramesSong);
  const f0Grid = new Float32Array(totalFramesSong);
  const plosiveGrid = new Float32Array(totalFramesSong);
  const cepGrid = new Float32Array(baseCep);
  const aperGrid = new Float32Array(baseAper);
  const backLGrid = new Float32Array(totalFramesSong);
  const backRGrid = new Float32Array(totalFramesSong);
  const backSemiGrid = new Float32Array(totalFramesSong);

  const uiEvents = [];

  for (let idx = 0; idx < events.length; idx++) {
    const ev = events[idx];
    const L = ev.totalFrames;
    if (L <= 0) continue;
    const sF = ev.startFrame;
    const cf = ev.consFrames;

    const envPos = buildEnvelopeKnotPositions(L, cf, envStep);
    const gainCurve = interp1dKnots(envPos, uint8ArrToGain(ev.gainHQ, refPeak), L);
    const vdegKnots = new Float32Array(ev.gainAQ.length);
    for (let i = 0; i < ev.gainAQ.length; i++) vdegKnots[i] = ev.gainAQ[i] / 255.0;
    const vdegCurve = interp1dKnots(envPos, vdegKnots, L);

    const unvoicedOnly = ev.voicedOnsetFrame >= L;
    let f0Curve;
    if (!unvoicedOnly) {
      const f0Pos = buildUniformKnotPositions(L, f0Step);
      const midiKnots = reconstructMidiKnots(ev.baseMidiQ, ev.f0StepQ);
      const midiCurve = interp1dKnots(f0Pos, midiKnots, L);
      f0Curve = new Float32Array(L);
      for (let i = 0; i < L; i++) {
        f0Curve[i] = 440.0 * Math.pow(2.0, (midiCurve[i] - 69.0) / 12.0);
      }
      uiEvents.push({
        eventIndex: idx,
        syllableId: ev.syllableId,
        text: ev.text,
        pinyin: ev.pinyin,
        isHook: !!ev.isHook,
        legatoPrev: !!ev.legatoPrev,
        legatoNext: !!ev.legatoNext,
        startSec: (sF * hop) / sr,
        endSec: ((sF + L) * hop) / sr,
        meanMidi: midiKnots[Math.floor(midiKnots.length / 2)],
      });
    } else {
      f0Curve = new Float32Array(L);
      vdegCurve.fill(0.0);
    }

    const exprPos = buildAdaptiveExprKnotPositions(L, cf, exprStep);
    const nKnots = exprPos.length;
    const dmKnots = new Float32Array(nKnots * nCep);
    const daKnots = new Float32Array(nKnots * nAper);
    for (let k = 0; k < nKnots; k++) {
      const rowM = ev.exprMcepQ[k] || [];
      const mOutOff = k * nCep;
      for (let d = 0; d < rowM.length && d < kPca; d++) {
        const w = rowM[d] * qStep;
        if (w === 0) continue;
        const bOff = d * nCep;
        for (let c = 0; c < nCep; c++) dmKnots[mOutOff + c] += w * basisM[bOff + c];
      }
      const rowA = ev.exprAperQ[k] || [];
      const aOutOff = k * nAper;
      for (let d = 0; d < rowA.length && d < kAper; d++) {
        const w = rowA[d] * (qStep * 1.25);
        if (w === 0) continue;
        const bOff = d * nAper;
        for (let a = 0; a < nAper; a++) daKnots[aOutOff + a] += w * basisA[bOff + a];
      }
    }

    const dmCurve = interp2dKnots(exprPos, dmKnots, nCep, L);
    const daCurve = interp2dKnots(exprPos, daKnots, nAper, L);

    const fEnd = Math.min(totalFramesSong, sF + L);
    const span = fEnd - sF;
    if (span <= 0) continue;

    for (let i = 0; i < span; i++) {
      const gf = sF + i;
      gainGrid[gf] = gainCurve[i];
      vdegGrid[gf] = Math.max(0.0, Math.min(1.0, vdegCurve[i]));
      f0Grid[gf] = f0Curve[i];
      const mOff = gf * nCep;
      const iM = i * nCep;
      for (let c = 0; c < nCep; c++) cepGrid[mOff + c] += dmCurve[iM + c];
      const aOff = gf * nAper;
      const iA = i * nAper;
      for (let a = 0; a < nAper; a++) {
        aperGrid[aOff + a] = Math.max(-4.8, Math.min(-1.55, aperGrid[aOff + a] + daCurve[iA + a]));
      }
    }

    if (ev.isPlosive) {
      const burstLen = Math.min(span, Math.max(2, Math.min(cf, 7)));
      for (let r = 0; r < burstLen; r++) {
        plosiveGrid[sF + r] = r === burstLen - 1 && burstLen > 2 ? 0.5 : 1.0;
      }
    }

    const blG = uint8ToGainScalar(ev.backGainLQ, refPeak) * 0.68;
    const brG = uint8ToGainScalar(ev.backGainRQ, refPeak) * 0.68;
    if ((blG > 1e-6 || brG > 1e-6) && !unvoicedOnly) {
      const vEnv = new Float32Array(L);
      for (let i = 0; i < L; i++) {
        const vm = Math.max(0.0, Math.min(1.0, (vdegCurve[i] - 0.1) / 0.55));
        vEnv[i] = gainCurve[i] * Math.sqrt(vm);
      }
      const sortedV = Float32Array.from(vEnv).sort();
      const p70 = sortedV[Math.min(L - 1, Math.floor(0.7 * (L - 1)))];
      if (p70 > 1e-6) {
        for (let i = 0; i < span; i++) {
          const sh = Math.min(1.3, vEnv[i] / p70);
          backLGrid[sF + i] = blG * sh;
          backRGrid[sF + i] = brG * sh;
          backSemiGrid[sF + i] = ev.backPitchQ * 0.1;
        }
      }
    }
  }

  // Seam continuity polish on `--` bridges + phrase entrance/exit micro-taper
  const gOrig = new Float32Array(gainGrid);
  const fade3 = [0.35, 0.72, 0.93];
  for (let f = 1; f < totalFramesSong - 1; f++) {
    if (gOrig[f] > 0.0 && gOrig[f - 1] === 0.0) {
      if (plosiveGrid[f] > 0.0) {
        gainGrid[f] *= 0.65;
      } else {
        for (let kf = 0; kf < 3; kf++) {
          if (f + kf < totalFramesSong && gOrig[f + kf] > 0.0) gainGrid[f + kf] *= fade3[kf];
        }
      }
    }
    if (gOrig[f] > 0.0 && gOrig[f + 1] === 0.0) {
      for (let kf = 0; kf < 3; kf++) {
        if (f - kf >= 0 && gOrig[f - kf] > 0.0) gainGrid[f - kf] *= fade3[kf];
      }
    }
  }

  bridgeShortF0Gaps(f0Grid, 36);

  const f0BackGrid = new Float32Array(totalFramesSong);
  for (let f = 0; f < totalFramesSong; f++) {
    if (f0Grid[f] > 60.0) {
      f0BackGrid[f] = f0Grid[f] * Math.pow(2.0, backSemiGrid[f] / 12.0);
    }
  }
  const backLSmooth = smoothFrameTrack(backLGrid);
  const backRSmooth = smoothFrameTrack(backRGrid);

  // Pass 2: Tracks 01 (`lead_sub_f0`), 02 (`lead_vowel_body`), 03 (`lead_harmonic_sheen`) — Fast Chebyshev Harmonics
  const bufLen = nSamples + nFft + hop;
  const subMono = new Float32Array(bufLen);
  const bodyMono = new Float32Array(bufLen);
  const sheenMono = new Float32Array(bufLen);
  const phiArr = new Float32Array(bufLen);
  const periodArr = new Int32Array(bufLen);

  const bCosLut = buildWarpedCosineLut(nCep, 4096);
  const { bLn, phiRad, bandTaper, sibHp, freqs, logFreqs, e0, invESpan } = precomputeUnvoicedMinphaseTables(config);

  const cosPsi = new Float64Array(maxHarm);
  const sinPsi = new Float64Array(maxHarm);
  for (let h = 0; h < maxHarm; h++) {
    cosPsi[h] = Math.cos(glottPsi[h] || 0.0);
    sinPsi[h] = Math.sin(glottPsi[h] || 0.0);
  }

  const subA = new Float64Array(maxHarm);
  const bodyA = new Float64Array(maxHarm);
  const sheenA = new Float64Array(maxHarm);
  const subB = new Float64Array(maxHarm);
  const bodyB = new Float64Array(maxHarm);
  const sheenB = new Float64Array(maxHarm);
  const dbH = new Float64Array(maxHarm);
  const relA = new Float64Array(maxHarm);

  function evalLeadFrame(fIdx, outSub, outBody, outSheen) {
    outSub.fill(0.0);
    outBody.fill(0.0);
    outSheen.fill(0.0);
    const vMix = Math.max(0.0, Math.min(1.0, (vdegGrid[fIdx] - 0.1) / 0.55));
    const gH = gainGrid[fIdx] * Math.sqrt(vMix);
    const ff = f0Grid[fIdx];
    if (gH <= 1e-7 || ff <= 60.0) return 0;
    const nAct = Math.min(maxHarm, Math.floor(19000.0 / ff));
    if (nAct <= 0) return 0;

    const cepOff = fIdx * nCep;
    let lowRefDb = -120.0;
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      const fcC = Math.max(fMin, Math.min(fMax, fh));
      const uH = (21.4 * Math.log10(1.0 + 0.00437 * fcC) - e0) * invESpan;
      const uIdx = Math.max(0, Math.min(4095, Math.round(uH * 4095.0)));
      const lutOff = uIdx * nCep;
      let sumDb = 0.0;
      for (let m = 0; m < nCep; m++) sumDb += bCosLut[lutOff + m] * cepGrid[cepOff + m];
      dbH[h] = sumDb;
      if (h < 6 && sumDb > lowRefDb) lowRefDb = sumDb;
    }

    let sumSq = 0.0;
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      let d = dbH[h];
      if (fh > 2800.0 && d > lowRefDb - 2.5) d = lowRefDb - 2.5;
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

    for (let s = 0; s < hop; s++) {
      const fc = fS + df * s;
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
        subMono[base + s] = accSub;
        bodyMono[base + s] = accBody;
        sheenMono[base + s] = accSheen;
      }
    }

    subA.set(subB);
    bodyA.set(bodyB);
    sheenA.set(sheenB);
    nActA = nActB;
  }

  // Pass 3: Tracks 04 (`lead_plosive_burst`), 05 (`lead_sibilant`), 06 (`lead_velvet_breath`) — 512-pt Hilbert Min-Phase WOLA
  const rngUnv = createMulberry32(20261115);
  const eUnv = new Float32Array(bufLen);
  const sqrt6 = Math.sqrt(6.0);
  for (let i = 0; i < bufLen; i++) eUnv[i] = (rngUnv() + rngUnv() - 1.0) * sqrt6;
  dcBlock60HzInPlace(eUnv, sr);

  const rngBr = createMulberry32(20260804);
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
    if (T0 > 0 && i >= T0) {
      eBreath[i] = (wMod[i] - 0.58 * wMod[i - T0]) * normComb;
    }
  }
  dcBlock60HzInPlace(eBreath, sr);

  const wSqrt = Float32Array.from({ length: nFft }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / nFft));
  const plosMono = new Float32Array(bufLen);
  const sibMono = new Float32Array(bufLen);
  const breathMono = new Float32Array(bufLen);

  const aperEdges = new Float64Array(nAper + 1);
  const logRatioA = Math.log(config.aperFreqMax / config.aperFreqMin) / nAper;
  for (let b = 0; b <= nAper; b++) aperEdges[b] = config.aperFreqMin * Math.exp(b * logRatioA);
  const logAperC = new Float64Array(nAper);
  for (let b = 0; b < nAper; b++) logAperC[b] = 0.5 * (Math.log(aperEdges[b]) + Math.log(aperEdges[b + 1]));

  const fft512 = new Radix2ComplexFFT(nFft);
  const reW = new Float64Array(nFft);
  const imW = new Float64Array(nFft);
  const hEnv = new Float64Array(nBins);
  const cosPh = new Float64Array(nBins);
  const sinPh = new Float64Array(nBins);
  const twoOverNfft = 2.0 / nFft;

  for (let f = 0; f < totalFramesSong; f++) {
    const gTot = gainGrid[f];
    if (gTot <= 1e-7) continue;
    const center = f * hop;
    if (center >= nSamples) break;
    const start = center - halfW;
    if (start < 0 || start + nFft > bufLen) continue;

    const cepOff = f * nCep;
    for (let k = 0; k < nBins; k++) {
      let sumLn = 0.0;
      let sumPh = 0.0;
      for (let m = 0; m < nCep; m++) {
        const cm = cepGrid[cepOff + m];
        const idx = m * nBins + k;
        sumLn += cm * bLn[idx];
        sumPh += cm * phiRad[idx];
      }
      const clp = Math.max(-14.0, Math.min(8.0, sumLn));
      hEnv[k] = Math.exp(clp) * bandTaper[k];
      cosPh[k] = Math.cos(sumPh);
      sinPh[k] = Math.sin(sumPh);
    }

    const vMix = Math.max(0.0, Math.min(1.0, (vdegGrid[f] - 0.1) / 0.55));
    const gUnv = gTot * Math.sqrt(1.0 - vMix);
    const gHarm = gTot * Math.sqrt(vMix);

    if (gUnv > 1e-7) {
      const pFrac = plosiveGrid[f];
      let powU = 0.0;
      for (let k = 1; k < nBins; k++) {
        const hu = hEnv[k] * (pFrac + (1.0 - pFrac) * sibHp[k]);
        powU += hu * hu;
      }
      const scaleU = gUnv / Math.sqrt(twoOverNfft * powU + 1e-18);
      for (let i = 0; i < nFft; i++) {
        reW[i] = eUnv[start + i] * wSqrt[i];
        imW[i] = 0.0;
      }
      fft512.transform(reW, imW, false);
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
      fft512.transform(reW, imW, true);
      for (let i = 0; i < nFft; i++) {
        const yu = reW[i] * wSqrt[i];
        if (pFrac > 0.0) plosMono[start + i] += pFrac * yu;
        if (pFrac < 1.0) sibMono[start + i] += (1.0 - pFrac) * yu;
      }
    }

    const fVal = f0Grid[f];
    if (gHarm > 1e-7 && fVal > 60.0) {
      let powE = 0.0;
      for (let k = 1; k < nBins; k++) powE += hEnv[k] * hEnv[k];
      const invNormE = gHarm / Math.sqrt(twoOverNfft * powE + 1e-18);
      for (let i = 0; i < nFft; i++) {
        reW[i] = eBreath[start + i] * wSqrt[i];
        imW[i] = 0.0;
      }
      fft512.transform(reW, imW, false);
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
        const hnrAmp = Math.pow(10.0, 0.5 * apLog);
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
      fft512.transform(reW, imW, true);
      for (let i = 0; i < nFft; i++) {
        breathMono[start + i] += reW[i] * wSqrt[i];
      }
    }
  }

  // Pass 4: Track 07 (`backing_harmony`) — Independent L/R True Stereo Chorus Voices
  const maxHBack = 28;
  function synthBackingChannel(gainArr, frameLag, detuneMult, vibHz, vibCos, initPhi) {
    const outCh = new Float32Array(nSamples);
    const ampA = new Float64Array(maxHBack);
    const ampB = new Float64Array(maxHBack);
    const relBack = new Float64Array(maxHBack);

    function evalBack(fIdx, outAmp) {
      outAmp.fill(0.0);
      const gB = gainArr[fIdx];
      const fSrc = Math.max(0, fIdx - frameLag);
      const ffBase = f0BackGrid[fSrc];
      if (gB <= 1e-6 || ffBase <= 60.0) return { ff: 0.0, nAct: 0 };
      const tSec = (fIdx * hop) / sr;
      const vib = vibCos ? Math.cos(TWO_PI * vibHz * tSec) : Math.sin(TWO_PI * vibHz * tSec);
      const ff = ffBase * detuneMult * (1.0 + 0.0018 * vib);
      const nAct = Math.min(maxHBack, Math.floor(7600.0 / ff));
      if (nAct <= 0) return { ff, nAct: 0 };
      const cepOff = fSrc * nCep;
      let sumSq = 0.0;
      for (let h = 0; h < nAct; h++) {
        const fh = (h + 1) * ff;
        const fcC = Math.max(fMin, Math.min(fMax, fh));
        const uH = (21.4 * Math.log10(1.0 + 0.00437 * fcC) - e0) * invESpan;
        const uIdx = Math.max(0, Math.min(4095, Math.round(uH * 4095.0)));
        const lutOff = uIdx * nCep;
        let d = 0.0;
        for (let m = 0; m < nCep; m++) d += bCosLut[lutOff + m] * cepGrid[cepOff + m];
        const warmRoll = (1.0 / (1.0 + (fh / 3800.0) * (fh / 3800.0))) * Math.max(0.0, Math.min(1.0, (7600.0 - fh) / 1200.0));
        const a = Math.pow(10.0, d / 20.0) * warmRoll;
        relBack[h] = a;
        sumSq += a * a;
      }
      const sc = gB / Math.sqrt(0.5 * sumSq + 1e-18);
      for (let h = 0; h < nAct; h++) outAmp[h] = relBack[h] * sc;
      return { ff, nAct };
    }

    let phiC = initPhi;
    let stA = evalBack(0, ampA);
    for (let f = 0; f < nHops; f++) {
      const stB = evalBack(f + 1, ampB);
      if (stA.ff <= 60.0 && stB.ff <= 60.0) {
        phiC = initPhi;
        ampA.set(ampB);
        stA = stB;
        continue;
      }
      let fS = stA.ff;
      let fE = stB.ff;
      if (stA.ff <= 60.0 && stB.ff > 60.0) fS = stB.ff;
      else if (stA.ff > 60.0 && stB.ff <= 60.0) fE = stA.ff;

      const base = f * hop;
      const endS = Math.min(hop, nSamples - base);
      const df = (fE - fS) * invHop;
      const nH = Math.max(stA.nAct, stB.nAct);
      for (let s = 0; s < endS; s++) {
        phiC += twoPiOverSr * (fS + df * s);
        if (phiC >= TWO_PI) phiC -= TWO_PI;
        if (nH > 0) {
          const frac = s * invHop;
          const c1 = Math.cos(phiC);
          const s1 = Math.sin(phiC);
          const twoC1 = 2.0 * c1;
          let chPrev = 1.0;
          let shPrev = 0.0;
          let ch = c1;
          let sh = s1;
          let acc = 0.0;
          for (let h = 0; h < nH; h++) {
            acc += (ampA[h] + (ampB[h] - ampA[h]) * frac) * (ch * cosPsi[h] - sh * sinPsi[h]);
            const chNext = twoC1 * ch - chPrev;
            const shNext = twoC1 * sh - shPrev;
            chPrev = ch;
            shPrev = sh;
            ch = chNext;
            sh = shNext;
          }
          outCh[base + s] = acc;
        }
      }
      ampA.set(ampB);
      stA = stB;
    }
    return outCh;
  }

  const backL = synthBackingChannel(backLSmooth, 3, Math.pow(2.0, 6.5 / 1200.0), 5.1, false, 1.7);
  const backR = synthBackingChannel(backRSmooth, 5, Math.pow(2.0, -6.5 / 1200.0), 5.4, true, 4.2);

  const trackSub = subMono.subarray(0, nSamples);
  const trackBody = bodyMono.subarray(0, nSamples);
  const trackSheen = sheenMono.subarray(0, nSamples);
  const trackPlos = plosMono.subarray(0, nSamples);
  const trackSib = sibMono.subarray(0, nSamples);
  const trackBreath = breathMono.subarray(0, nSamples);

  // Pass 5: Track 08 (`vocal_space`) — 4-Tap Early Reflections + 48kHz 8-Line Prime Householder FDN
  const spaceSendMono = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    spaceSendMono[i] =
      trackSub[i] + trackBody[i] + 0.5 * trackSheen[i] + 0.2 * (backL[i] + backR[i]);
  }
  const [spaceL, spaceR] = synthesizeSpaceFdnPlain(spaceFdn, spaceSendMono);

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
