import { ComplexInverseFFT } from './fft.js';
import { synthesizeSpaceFdn } from './reverb-fdn.js';
import {
  buildAdaptiveExprKnotPositions,
  buildEnvelopeKnotPositions,
  buildUniformKnotPositions,
  reconstructMidiKnots,
  unpackEventStream,
  unpackSpeakerCore,
  unpackSyllableAtlas,
} from './unpacker.js';

const DB_FLOOR = -78.0;
const DB_CEIL = 4.0;
const VOWEL_ANCHORS_LEN = 5;
const CONS_ANCHORS_LEN = 3;
const TWO_PI = 2.0 * Math.PI;

function uint8ToGain(qArr, refPeak) {
  const out = new Float32Array(qArr.length);
  for (let i = 0; i < qArr.length; i++) {
    const q = qArr[i];
    if (q > 0) {
      const db = DB_FLOOR + ((q - 1.0) / 254.0) * (DB_CEIL - DB_FLOOR);
      out[i] = refPeak * Math.pow(10.0, db / 20.0);
    }
  }
  return out;
}

function uint8ScalarToGain(q, refPeak) {
  if (q <= 0) return 0.0;
  const db = DB_FLOOR + ((q - 1.0) / 254.0) * (DB_CEIL - DB_FLOOR);
  return refPeak * Math.pow(10.0, db / 20.0);
}

function interp1dKnots(knotPos, knotVals, totalFrames) {
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
    if (k === 0) {
      bandTaper[k] = 0.0;
      sibHp[k] = 0.0;
    } else {
      const hiT = Math.max(0.0, Math.min(1.0, (18500.0 - fk) / 3500.0));
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
  const k0 = 0.08;
  const k1 = 0.17;
  const k2 = 0.5;
  for (let pass = 0; pass < 2; pass++) {
    const src = pass === 0 ? track : tmp;
    const dst = pass === 0 ? tmp : out;
    for (let i = 0; i < n; i++) {
      const xm2 = src[Math.max(0, i - 2)];
      const xm1 = src[Math.max(0, i - 1)];
      const x0 = src[i];
      const xp1 = src[Math.min(n - 1, i + 1)];
      const xp2 = src[Math.min(n - 1, i + 2)];
      dst[i] = k0 * (xm2 + xp2) + k1 * (xm1 + xp1) + k2 * x0;
    }
  }
  return out;
}

function bridgeShortF0Gaps(f0Grid, maxGapFrames = 12) {
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

function createMulberry32(seed) {
  let a = seed >>> 0;
  return function nextRand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  };
}

export function synthesizeVocalGenesis(rawSpeaker, rawAtlas, rawEvents, rawFdn, config) {
  const speaker = unpackSpeakerCore(rawSpeaker);
  const atlas = unpackSyllableAtlas(rawAtlas);
  const { meta: evMeta, events } = unpackEventStream(rawEvents, atlas.prototypes);

  const sr = speaker.sampleRate;
  const nSamples = speaker.numSamples;
  const hop = speaker.hop;
  const refPeak = speaker.refPeak;

  const nFft = config.fftSize; // 512
  const halfW = nFft >> 1; // 256
  const nBins = halfW + 1; // 257
  const nCep = config.cepstralCoeffs; // 40
  const nAper = config.aperiodicBands; // 8
  const maxHarm = config.maxHarmonics || 64;
  const fMin = config.harmFreqMin;
  const fMax = config.harmFreqMax;

  const bCosLut = buildWarpedCosineLut(nCep, 4096);
  const { bLn, phiRad, bandTaper, sibHp, freqs, logFreqs, e0, invESpan } =
    precomputeUnvoicedMinphaseTables(config);

  const aperMinLog = Math.log(config.aperFreqMin);
  const aperMaxLog = Math.log(config.aperFreqMax);
  const logAperC = new Float64Array(nAper);
  for (let b = 0; b < nAper; b++) {
    logAperC[b] = aperMinLog + ((aperMaxLog - aperMinLog) * (b + 0.5)) / nAper;
  }

  const kPca = evMeta.kPca;
  const kAper = evMeta.kAper;
  const exprStep = evMeta.exprStep;
  const f0Step = evMeta.f0Step;
  const envStep = evMeta.envStep;
  const pfBeta = evMeta.formantPostFilterBeta || 0.0;
  const qStep = evMeta.qStepDb;

  const pfWeights = new Float32Array(nCep);
  pfWeights.fill(1.0);
  if (pfBeta > 1e-6) {
    for (let m = 2; m < nCep; m++) {
      const uM = (m - 2) / Math.max(1.0, nCep - 3);
      pfWeights[m] = 1.0 + pfBeta * Math.sin(Math.PI * Math.min(1.0, uM));
    }
  }

  const totalFramesSong = Math.floor(nSamples / hop) + 6;
  const gainGrid = new Float32Array(totalFramesSong);
  const vdegGrid = new Float32Array(totalFramesSong);
  const f0Grid = new Float32Array(totalFramesSong);
  const plosiveGrid = new Float32Array(totalFramesSong);
  const cepGrid = new Float32Array(totalFramesSong * nCep);
  const aperGrid = new Float32Array(totalFramesSong * nAper);
  aperGrid.fill(-3.6);
  const panTrack = new Float32Array(totalFramesSong);
  const backLGrid = new Float32Array(totalFramesSong);
  const backRGrid = new Float32Array(totalFramesSong);
  const backSemiGrid = new Float32Array(totalFramesSong);

  const uiEvents = [];

  // Pass 1a: Continuous Phrase-Level Coarticulated Prototype Grids (`x-i-a--y-i--g-e--l-u--k-ou--j-ian`)
  const baseCepGrid = new Float32Array(totalFramesSong * nCep);
  const baseAperGrid = new Float32Array(totalFramesSong * nAper);
  baseAperGrid.fill(-3.6);

  for (let ei = 0; ei < events.length; ei++) {
    const ev = events[ei];
    const L = ev.totalFrames;
    if (L <= 0) continue;
    const sF = ev.startFrame;
    const cf = ev.consFrames;
    const proto = atlas.prototypes[ev.syllableId];
    const fEndClip = Math.min(totalFramesSong, sF + L);
    const span = fEndClip - sF;
    if (span <= 0) continue;

    const vot = L > 1 && cf > 0 ? Math.min(L - 1, cf) : 0;
    const vLen = L - vot;
    for (let r = 0; r < span; r++) {
      const fAbs = sF + r;
      const cepBase = fAbs * nCep;
      const aperBase = fAbs * nAper;
      if (r < vot) {
        const uC = vot > 1 ? (r / (vot - 1)) * (CONS_ANCHORS_LEN - 1) : 0.0;
        const ic0 = Math.min(CONS_ANCHORS_LEN - 2, uC | 0);
        const fc = uC - ic0;
        for (let c = 0; c < nCep; c++) {
          baseCepGrid[cepBase + c] =
            speaker.meanMcep[c] +
            (1.0 - fc) * proto.consMcepKnots[ic0 * nCep + c] +
            fc * proto.consMcepKnots[(ic0 + 1) * nCep + c];
        }
        for (let b = 0; b < nAper; b++) {
          baseAperGrid[aperBase + b] =
            speaker.aperMean[b] +
            (1.0 - fc) * proto.consAperKnots[ic0 * nAper + b] +
            fc * proto.consAperKnots[(ic0 + 1) * nAper + b];
        }
      } else {
        const uV = vLen > 1 ? ((r - vot) / (vLen - 1)) * (VOWEL_ANCHORS_LEN - 1) : 0.0;
        const iv0 = Math.min(VOWEL_ANCHORS_LEN - 2, uV | 0);
        const fv = uV - iv0;
        for (let c = 0; c < nCep; c++) {
          baseCepGrid[cepBase + c] =
            speaker.meanMcep[c] +
            (1.0 - fv) * proto.vowelMcepKnots[iv0 * nCep + c] +
            fv * proto.vowelMcepKnots[(iv0 + 1) * nCep + c];
        }
        for (let b = 0; b < nAper; b++) {
          baseAperGrid[aperBase + b] =
            speaker.aperMean[b] +
            (1.0 - fv) * proto.vowelAperKnots[iv0 * nAper + b] +
            fv * proto.vowelAperKnots[(iv0 + 1) * nAper + b];
        }
      }
    }
  }

  // DAC Asymmetric `--` Coarticulation: previous character's rhyme bends into next character's initial!
  const protoSeams = [];
  for (let i = 1; i < events.length; i++) {
    const prevEv = events[i - 1];
    const currEv = events[i];
    const b = currEv.startFrame;
    if (
      prevEv.totalFrames >= 2 &&
      currEv.totalFrames >= 2 &&
      prevEv.startFrame + prevEv.totalFrames === b &&
      b >= 1 &&
      b < totalFramesSong
    ) {
      const wPre = Math.min(18, Math.floor((prevEv.totalFrames * 12) / 25), b);
      const wPost = Math.min(currEv.isPlosive ? 4 : 10, Math.floor((currEv.totalFrames * 9) / 25), totalFramesSong - b);
      if (wPre >= 1 && wPost >= 1) {
        const dCep = new Float32Array(nCep);
        const dAper = new Float32Array(nAper);
        const cOff0 = (b - 1) * nCep;
        const cOff1 = b * nCep;
        for (let c = 0; c < nCep; c++) dCep[c] = baseCepGrid[cOff1 + c] - baseCepGrid[cOff0 + c];
        const aOff0 = (b - 1) * nAper;
        const aOff1 = b * nAper;
        for (let ab = 0; ab < nAper; ab++) dAper[ab] = baseAperGrid[aOff1 + ab] - baseAperGrid[aOff0 + ab];
        protoSeams.push({ b, wPre, wPost, dCep, dAper });
      }
    }
  }
  for (let si = 0; si < protoSeams.length; si++) {
    const { b, wPre, wPost, dCep, dAper } = protoSeams[si];
    const invTot = 1.0 / (wPre + wPost);
    const gPre = wPre * invTot;
    const gPost = wPost * invTot;
    for (let r = 0; r < wPre; r++) {
      const w = 0.5 * gPre * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPre));
      const fIdx = b - 1 - r;
      const cOff = fIdx * nCep;
      const aOff = fIdx * nAper;
      for (let c = 0; c < nCep; c++) baseCepGrid[cOff + c] += w * dCep[c];
      for (let ab = 0; ab < nAper; ab++) baseAperGrid[aOff + ab] += w * dAper[ab];
    }
    for (let r = 0; r < wPost; r++) {
      const w = 0.5 * gPost * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPost));
      const fIdx = b + r;
      const cOff = fIdx * nCep;
      const aOff = fIdx * nAper;
      for (let c = 0; c < nCep; c++) baseCepGrid[cOff + c] -= w * dCep[c];
      for (let ab = 0; ab < nAper; ab++) baseAperGrid[aOff + ab] -= w * dAper[ab];
    }
  }

  // Pass 1b: Decode Per-Event Expression & Envelope Knots onto Coarticulated Grids
  for (let ei = 0; ei < events.length; ei++) {
    const ev = events[ei];
    const L = ev.totalFrames;
    if (L <= 0) continue;
    const sF = ev.startFrame;
    const cf = ev.consFrames;
    const proto = atlas.prototypes[ev.syllableId];

    const envPos = buildEnvelopeKnotPositions(L, cf, envStep);
    const gainCurve = interp1dKnots(envPos, uint8ToGain(ev.gainHQ, refPeak), L);
    const vdegKnots = new Float32Array(ev.gainAQ.length);
    for (let k = 0; k < vdegKnots.length; k++) vdegKnots[k] = ev.gainAQ[k] / 255.0;
    const vdegCurve = interp1dKnots(envPos, vdegKnots, L);
    for (let r = 0; r < L; r++) {
      if (vdegCurve[r] < 0.0) vdegCurve[r] = 0.0;
      else if (vdegCurve[r] > 1.0) vdegCurve[r] = 1.0;
    }

    const f0Curve = new Float32Array(L);
    if (!ev.unvoicedOnly) {
      const f0Pos = buildUniformKnotPositions(L, f0Step);
      const midiKnots = reconstructMidiKnots(ev.baseMidiQ, ev.f0StepQ);
      const midiCurve = interp1dKnots(f0Pos, midiKnots, L);
      for (let r = 0; r < L; r++) {
        f0Curve[r] = 440.0 * Math.pow(2.0, (midiCurve[r] - 69.0) / 12.0);
      }
    } else {
      vdegCurve.fill(0.0);
    }

    const exprPos = buildAdaptiveExprKnotPositions(L, cf, exprStep);
    const nx = exprPos.length;
    const dmKnots = new Float32Array(nx * nCep);
    for (let k = 0; k < nx; k++) {
      for (let m = 0; m < kPca; m++) {
        const w = ev.exprMcepQ[k * kPca + m] * qStep;
        const bOff = m * nCep;
        for (let c = 0; c < nCep; c++) {
          dmKnots[k * nCep + c] += w * speaker.mcepPcaBasis[bOff + c];
        }
      }
    }
    const daKnots = new Float32Array(nx * nAper);
    const qStepA = qStep * 1.25;
    for (let k = 0; k < nx; k++) {
      for (let m = 0; m < kAper; m++) {
        const w = ev.exprAperQ[k * kAper + m] * qStepA;
        const bOff = m * nAper;
        for (let b = 0; b < nAper; b++) {
          daKnots[k * nAper + b] += w * speaker.aperPcaBasis[bOff + b];
        }
      }
    }

    const panVal = ev.stereoPanQ / 64.0;
    let exSeg = 0;

    const uiPitch = [];
    const uiStep = Math.max(1, Math.floor(L / 8));
    for (let r = 0; r < L; r += uiStep) {
      if (f0Curve[r] > 60.0) uiPitch.push(Math.round(f0Curve[r] * 10) / 10);
    }
    uiEvents.push({
      index: ev.index,
      startSec: Number(((sF * hop) / sr).toFixed(3)),
      endSec: Number((((sF + L) * hop) / sr).toFixed(3)),
      consSec: Number(((cf * hop) / sr).toFixed(3)),
      syllableId: ev.syllableId,
      pinyin: proto.pinyin,
      initialClass: proto.initialClass,
      displayText: ev.displayText,
      isHook: ev.isHook,
      legatoNext: ev.legatoNext,
      midi: Number((ev.baseMidiQ * 0.02).toFixed(2)),
      pitchPreview: uiPitch,
    });

    const fEndClip = Math.min(totalFramesSong, sF + L);
    const span = fEndClip - sF;
    if (span <= 0) continue;

    if (ev.isPlosive) {
      const burstLen = Math.min(span, Math.max(2, Math.min(cf, 7)));
      for (let r = 0; r < burstLen; r++) {
        plosiveGrid[sF + r] = r === burstLen - 1 && burstLen > 2 ? 0.5 : 1.0;
      }
    }

    const blG = uint8ScalarToGain(ev.backGainLQ, refPeak) * 0.68;
    const brG = uint8ScalarToGain(ev.backGainRQ, refPeak) * 0.68;
    if ((blG > 1e-6 || brG > 1e-6) && !ev.unvoicedOnly) {
      const vEnv = new Float32Array(L);
      for (let r = 0; r < L; r++) {
        const vm = Math.max(0.0, Math.min(1.0, (vdegCurve[r] - 0.1) / 0.55));
        vEnv[r] = gainCurve[r] * Math.sqrt(vm);
      }
      const sortedV = Float32Array.from(vEnv).sort();
      const p70 = sortedV[Math.min(L - 1, Math.floor(0.7 * (L - 1)))];
      if (p70 > 1e-6) {
        const bSemi = ev.backPitchQ * 0.1;
        for (let r = 0; r < span; r++) {
          const envShape = Math.min(1.3, vEnv[r] / p70);
          backLGrid[sF + r] = blG * envShape;
          backRGrid[sF + r] = brG * envShape;
          backSemiGrid[sF + r] = bSemi;
        }
      }
    }

    for (let r = 0; r < span; r++) {
      const fAbs = sF + r;
      while (exSeg < nx - 2 && r > exprPos[exSeg + 1]) exSeg++;
      const exNext = Math.min(nx - 1, exSeg + 1);
      const ex0 = exprPos[exSeg];
      const ex1 = exprPos[exNext];
      const exFrac = ex1 > ex0 ? (r - ex0) / (ex1 - ex0) : 0.0;

      gainGrid[fAbs] = gainCurve[r];
      vdegGrid[fAbs] = vdegCurve[r];
      f0Grid[fAbs] = f0Curve[r];
      panTrack[fAbs] = panVal;

      const cepBase = fAbs * nCep;
      const aperBase = fAbs * nAper;
      for (let c = 0; c < nCep; c++) {
        const dVal = (1.0 - exFrac) * dmKnots[exSeg * nCep + c] + exFrac * dmKnots[exNext * nCep + c];
        cepGrid[cepBase + c] = (baseCepGrid[cepBase + c] + dVal) * pfWeights[c];
      }
      for (let b = 0; b < nAper; b++) {
        const dA = (1.0 - exFrac) * daKnots[exSeg * nAper + b] + exFrac * daKnots[exNext * nAper + b];
        aperGrid[aperBase + b] = Math.max(-4.8, Math.min(-1.7, baseAperGrid[aperBase + b] + dA));
      }
    }
  }

  // Post-decode DAC `--` seam polish across contiguous syllable bridges
  const postSeams = [];
  for (let i = 1; i < events.length; i++) {
    const prevEv = events[i - 1];
    const currEv = events[i];
    const b = currEv.startFrame;
    if (
      prevEv.totalFrames >= 2 &&
      currEv.totalFrames >= 2 &&
      prevEv.startFrame + prevEv.totalFrames === b &&
      b >= 1 &&
      b < totalFramesSong
    ) {
      const wPre = Math.min(6, Math.floor(prevEv.totalFrames / 2), b);
      const wPost = Math.min(currEv.isPlosive ? 2 : 5, Math.floor(currEv.totalFrames / 2), totalFramesSong - b);
      if (wPre >= 1 && wPost >= 1) {
        const dc = new Float32Array(nCep);
        const da = new Float32Array(nAper);
        const c0 = (b - 1) * nCep;
        const c1 = b * nCep;
        for (let c = 0; c < nCep; c++) dc[c] = cepGrid[c1 + c] - cepGrid[c0 + c];
        const a0 = (b - 1) * nAper;
        const a1 = b * nAper;
        for (let ab = 0; ab < nAper; ab++) da[ab] = aperGrid[a1 + ab] - aperGrid[a0 + ab];
        postSeams.push({
          b,
          wPre,
          wPost,
          isPlos: currEv.isPlosive,
          dc,
          da,
          dg: gainGrid[b] - gainGrid[b - 1],
          dv: vdegGrid[b] - vdegGrid[b - 1],
        });
      }
    }
  }
  for (let si = 0; si < postSeams.length; si++) {
    const { b, wPre, wPost, isPlos, dc, da, dg, dv } = postSeams[si];
    const invTot = 1.0 / (wPre + wPost);
    const gPre = wPre * invTot;
    const gPost = wPost * invTot;
    for (let r = 0; r < wPre; r++) {
      const w = 0.5 * gPre * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPre));
      const idxL = b - 1 - r;
      const cOff = idxL * nCep;
      const aOff = idxL * nAper;
      for (let c = 0; c < nCep; c++) cepGrid[cOff + c] += w * dc[c];
      for (let ab = 0; ab < nAper; ab++) {
        aperGrid[aOff + ab] = Math.max(-4.8, Math.min(-1.7, aperGrid[aOff + ab] + w * da[ab]));
      }
      if (!isPlos) {
        gainGrid[idxL] = Math.max(0.0, gainGrid[idxL] + w * dg);
        vdegGrid[idxL] = Math.max(0.0, Math.min(1.0, vdegGrid[idxL] + w * dv));
      }
    }
    for (let r = 0; r < wPost; r++) {
      const w = 0.5 * gPost * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wPost));
      const idxR = b + r;
      const cOff = idxR * nCep;
      const aOff = idxR * nAper;
      for (let c = 0; c < nCep; c++) cepGrid[cOff + c] -= w * dc[c];
      for (let ab = 0; ab < nAper; ab++) {
        aperGrid[aOff + ab] = Math.max(-4.8, Math.min(-1.7, aperGrid[aOff + ab] - w * da[ab]));
      }
      if (!isPlos) {
        gainGrid[idxR] = Math.max(0.0, gainGrid[idxR] - w * dg);
        vdegGrid[idxR] = Math.max(0.0, Math.min(1.0, vdegGrid[idxR] - w * dv));
      }
    }
  }

  // Physiological vocal-tract & airflow inertia across interior active non-plosive frames
  const cepSrc = Float32Array.from(cepGrid);
  const aperSrc = Float32Array.from(aperGrid);
  const gainSrc = Float32Array.from(gainGrid);
  const vdegSrc = Float32Array.from(vdegGrid);
  for (let f = 1; f < totalFramesSong - 1; f++) {
    if (gainSrc[f - 1] > 0.0 && gainSrc[f] > 0.0 && gainSrc[f + 1] > 0.0 && plosiveGrid[f] === 0.0) {
      const cOff = f * nCep;
      const cPrev = (f - 1) * nCep;
      const cNext = (f + 1) * nCep;
      for (let c = 0; c < nCep; c++) {
        cepGrid[cOff + c] = 0.2 * cepSrc[cPrev + c] + 0.6 * cepSrc[cOff + c] + 0.2 * cepSrc[cNext + c];
      }
      const aOff = f * nAper;
      const aPrev = (f - 1) * nAper;
      const aNext = (f + 1) * nAper;
      for (let ab = 0; ab < nAper; ab++) {
        const av = 0.2 * aperSrc[aPrev + ab] + 0.6 * aperSrc[aOff + ab] + 0.2 * aperSrc[aNext + ab];
        aperGrid[aOff + ab] = Math.max(-4.8, Math.min(-1.7, av));
      }
      gainGrid[f] = 0.2 * gainSrc[f - 1] + 0.6 * gainSrc[f] + 0.2 * gainSrc[f + 1];
      vdegGrid[f] = Math.max(0.0, Math.min(1.0, 0.2 * vdegSrc[f - 1] + 0.6 * vdegSrc[f] + 0.2 * vdegSrc[f + 1]));
    }
  }

  // Smooth 3-frame micro-taper ONLY at isolated phrase entrances/exits
  const gainOrig = Float32Array.from(gainGrid);
  const fade3 = [0.35, 0.72, 0.93];
  for (let f = 1; f < totalFramesSong - 1; f++) {
    if (gainOrig[f] > 0.0 && gainOrig[f - 1] === 0.0) {
      if (plosiveGrid[f] > 0.0) {
        gainGrid[f] *= 0.65;
      } else {
        for (let kf = 0; kf < 3; kf++) {
          if (f + kf < totalFramesSong && gainOrig[f + kf] > 0.0) gainGrid[f + kf] *= fade3[kf];
        }
      }
    }
    if (gainOrig[f] > 0.0 && gainOrig[f + 1] === 0.0) {
      for (let kf = 0; kf < 3; kf++) {
        if (f - kf >= 0 && gainOrig[f - kf] > 0.0) gainGrid[f - kf] *= fade3[kf];
      }
    }
  }

  // Continuous breath-phrase glottal phase & anticipatory pitch portamento (`--` pitch ribbon)
  bridgeShortF0Gaps(f0Grid, 36);
  const f0Seams = [];
  for (let i = 1; i < events.length; i++) {
    const prevEv = events[i - 1];
    const currEv = events[i];
    const b = currEv.startFrame;
    if (
      prevEv.totalFrames >= 2 &&
      currEv.totalFrames >= 2 &&
      prevEv.startFrame + prevEv.totalFrames === b &&
      b >= 1 &&
      b < totalFramesSong
    ) {
      const fa = f0Grid[b - 1];
      const fb = f0Grid[b];
      if (fa > 60.0 && fb > 60.0) {
        const dlp = Math.log2(fb / fa);
        if (Math.abs(dlp) < 0.65) {
          const wp = Math.min(6, Math.floor(prevEv.totalFrames / 2), b);
          const wq = Math.min(4, Math.floor(currEv.totalFrames / 2), totalFramesSong - b);
          if (wp >= 1 && wq >= 1) f0Seams.push({ b, wp, wq, dlp });
        }
      }
    }
  }
  for (let si = 0; si < f0Seams.length; si++) {
    const { b, wp, wq, dlp } = f0Seams[si];
    const invT = 1.0 / (wp + wq);
    const gp = wp * invT;
    const gq = wq * invT;
    for (let r = 0; r < wp; r++) {
      const idxL = b - 1 - r;
      if (f0Grid[idxL] > 60.0) {
        const w = 0.5 * gp * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wp));
        f0Grid[idxL] *= Math.pow(2.0, w * dlp);
      }
    }
    for (let r = 0; r < wq; r++) {
      const idxR = b + r;
      if (f0Grid[idxR] > 60.0) {
        const w = 0.5 * gq * (1.0 + Math.cos((Math.PI * (r + 0.5)) / wq));
        f0Grid[idxR] *= Math.pow(2.0, -w * dlp);
      }
    }
  }
  const f0Src = Float32Array.from(f0Grid);
  for (let f = 1; f < totalFramesSong - 1; f++) {
    if (f0Src[f - 1] > 60.0 && f0Src[f] > 60.0 && f0Src[f + 1] > 60.0) {
      const lpLead = 0.22 * Math.log2(f0Src[f - 1]) + 0.56 * Math.log2(f0Src[f]) + 0.22 * Math.log2(f0Src[f + 1]);
      f0Grid[f] = Math.pow(2.0, lpLead);
    }
  }

  const f0BackRaw = new Float32Array(totalFramesSong);
  for (let f = 0; f < totalFramesSong; f++) {
    if (f0Grid[f] > 60.0) {
      f0BackRaw[f] = f0Grid[f] * Math.pow(2.0, backSemiGrid[f] / 12.0);
    }
  }
  const f0BackGrid = Float32Array.from(f0BackRaw);
  for (let f = 1; f < totalFramesSong - 1; f++) {
    if (f0BackRaw[f - 1] > 60.0 && f0BackRaw[f] > 60.0 && f0BackRaw[f + 1] > 60.0) {
      const lpSm =
        0.25 * Math.log2(f0BackRaw[f - 1]) +
        0.5 * Math.log2(f0BackRaw[f]) +
        0.25 * Math.log2(f0BackRaw[f + 1]);
      f0BackGrid[f] = Math.pow(2.0, lpSm);
    }
  }

  const backLSmooth = smoothFrameTrack(backLGrid);
  const backRSmooth = smoothFrameTrack(backRGrid);
  const panSmooth = smoothFrameTrack(panTrack);

  // Pass 2: Tracks 01 (`lead_sub_f0`), 02 (`lead_vowel_body`), 03 (`lead_harmonic_sheen`) — Chebyshev Harmonics
  const bufLen = nSamples + nFft + hop;
  const subMono = new Float32Array(bufLen);
  const bodyMono = new Float32Array(bufLen);
  const sheenMono = new Float32Array(bufLen);
  const phiArr = new Float32Array(bufLen);
  const periodArr = new Int32Array(bufLen);

  const cStat = new Float64Array(maxHarm);
  const sStat = new Float64Array(maxHarm);
  for (let h = 0; h < maxHarm; h++) {
    const psi = h < speaker.glottalPhase.length ? speaker.glottalPhase[h] : 0.0;
    cStat[h] = Math.cos(psi);
    sStat[h] = -Math.sin(psi);
  }

  const subA = new Float64Array(maxHarm);
  const bodyA = new Float64Array(maxHarm);
  const sheenA = new Float64Array(maxHarm);
  const subB = new Float64Array(maxHarm);
  const bodyB = new Float64Array(maxHarm);
  const sheenB = new Float64Array(maxHarm);
  const relScratch = new Float64Array(maxHarm);
  const curSub = new Float64Array(maxHarm);
  const curBody = new Float64Array(maxHarm);
  const curSheen = new Float64Array(maxHarm);
  const dSub = new Float64Array(maxHarm);
  const dBody = new Float64Array(maxHarm);
  const dSheen = new Float64Array(maxHarm);

  const twoPiOverSr = TWO_PI / sr;
  const invHop = 1.0 / hop;
  const nHops = Math.floor(nSamples / hop) + 1;

  function evalLeadFrame(fIdx, outSub, outBody, outSheen) {
    outSub.fill(0.0);
    outBody.fill(0.0);
    outSheen.fill(0.0);
    const vMix = Math.max(0.0, Math.min(1.0, (vdegGrid[fIdx] - 0.1) / 0.55));
    const gH = gainGrid[fIdx] * Math.sqrt(vMix);
    const ff = f0Grid[fIdx];
    if (gH <= 1e-7 || ff <= 60.0) return 0;
    const nAct = Math.min(maxHarm, Math.floor(15799.0 / ff));
    if (nAct <= 0) return 0;

    const cepOff = fIdx * nCep;
    let sumRelSq = 0.0;
    let lowRefDb = -120.0;
    const lowLim = Math.min(nAct, 6);
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      const fcC = Math.max(fMin, Math.min(fMax, fh));
      const uH = (21.4 * Math.log10(1.0 + 0.00437 * fcC) - e0) * invESpan;
      const uIdx = Math.max(0, Math.min(4095, Math.round(uH * 4095.0))) * nCep;
      let dbH = 0.0;
      for (let m = 0; m < nCep; m++) {
        dbH += bCosLut[uIdx + m] * cepGrid[cepOff + m];
      }
      if (h < lowLim && dbH > lowRefDb) lowRefDb = dbH;
      if (fh > 2800.0 && dbH > lowRefDb - 3.0) dbH = lowRefDb - 3.0;
      const hfTap = Math.max(0.0, Math.min(1.0, (15800.0 - fh) / 1600.0));
      const ra = Math.pow(10.0, dbH / 20.0) * hfTap;
      relScratch[h] = ra;
      sumRelSq += ra * ra;
    }

    const scale = gH / Math.sqrt(0.5 * sumRelSq + 1e-18);
    for (let h = 0; h < nAct; h++) {
      const fh = (h + 1) * ff;
      const amp = relScratch[h] * scale;
      const rs = fh / 720.0;
      const rs2 = rs * rs;
      const wSub = 1.0 / (1.0 + rs2 * rs2 * rs2);
      const rh = 3200.0 / fh;
      const rh2 = rh * rh;
      const wHi = 1.0 / (1.0 + rh2 * rh2 * rh2);
      const wMid = Math.max(0.0, 1.0 - wSub - wHi);
      outSub[h] = amp * wSub;
      outBody[h] = amp * wMid;
      outSheen[h] = amp * wHi;
    }
    return nAct;
  }

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

    let fS;
    let fE;
    if (f0A <= 60.0 && f0B > 60.0) {
      phi = 0.0;
      fS = f0B;
      fE = f0B;
    } else if (f0A > 60.0 && f0B <= 60.0) {
      fS = f0A;
      fE = f0A;
    } else {
      fS = f0A;
      fE = f0B;
    }

    const base = f * hop;
    const df = (fE - fS) * invHop;
    const nH = Math.max(nActA, nActB);

    if (nH > 0) {
      for (let h = 0; h < nH; h++) {
        curSub[h] = subA[h];
        curBody[h] = bodyA[h];
        curSheen[h] = sheenA[h];
        dSub[h] = (subB[h] - subA[h]) * invHop;
        dBody[h] = (bodyB[h] - bodyA[h]) * invHop;
        dSheen[h] = (sheenB[h] - sheenA[h]) * invHop;
      }

      for (let s = 0; s < hop; s++) {
        const idx = base + s;
        const fc = fS + df * s;
        phi += twoPiOverSr * fc;
        if (phi >= TWO_PI) phi -= TWO_PI * Math.floor(phi / TWO_PI);
        phiArr[idx] = phi;
        periodArr[idx] = Math.round(sr / fc);

        const c1 = Math.cos(phi);
        const s1 = Math.sin(phi);
        const twoC1 = 2.0 * c1;
        let cm2 = 1.0;
        let cm1 = c1;
        let sm2 = 0.0;
        let sm1 = s1;

        const w0 = cStat[0] * c1 + sStat[0] * s1;
        let accSub = curSub[0] * w0;
        let accBody = curBody[0] * w0;
        let accSheen = curSheen[0] * w0;
        curSub[0] += dSub[0];
        curBody[0] += dBody[0];
        curSheen[0] += dSheen[0];

        for (let h = 1; h < nH; h++) {
          const ch = twoC1 * cm1 - cm2;
          const sh = twoC1 * sm1 - sm2;
          cm2 = cm1;
          cm1 = ch;
          sm2 = sm1;
          sm1 = sh;
          const wh = cStat[h] * ch + sStat[h] * sh;
          accSub += curSub[h] * wh;
          accBody += curBody[h] * wh;
          accSheen += curSheen[h] * wh;
          curSub[h] += dSub[h];
          curBody[h] += dBody[h];
          curSheen[h] += dSheen[h];
        }

        subMono[idx] = accSub;
        bodyMono[idx] = accBody;
        sheenMono[idx] = accSheen;
      }
    } else {
      for (let s = 0; s < hop; s++) {
        const idx = base + s;
        const fc = fS + df * s;
        phi += twoPiOverSr * fc;
        if (phi >= TWO_PI) phi -= TWO_PI * Math.floor(phi / TWO_PI);
        phiArr[idx] = phi;
        periodArr[idx] = Math.round(sr / fc);
      }
    }

    subA.set(subB);
    bodyA.set(bodyB);
    sheenA.set(sheenB);
    nActA = nActB;
  }

  // Pass 3: Tracks 04 (`lead_plosive_burst`), 05 (`lead_sibilant`), 06 (`lead_velvet_breath`) — 512-pt Dual-Real WOLA
  const randUnv = createMulberry32(20261115);
  const eUnv = new Float32Array(bufLen);
  for (let i = 0; i < bufLen; i += 2) {
    const u1 = Math.max(1e-12, randUnv());
    const u2 = randUnv();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const th = TWO_PI * u2;
    eUnv[i] = r * Math.cos(th);
    if (i + 1 < bufLen) eUnv[i + 1] = r * Math.sin(th);
  }
  dcBlock60HzInPlace(eUnv, sr);

  const randBr = createMulberry32(20260804);
  const wMod = new Float32Array(bufLen);
  const modNorm = 1.0 / Math.sqrt(1.0 + 0.5 * 0.42 * 0.42);
  for (let i = 0; i < bufLen; i += 2) {
    const u1 = Math.max(1e-12, randBr());
    const u2 = randBr();
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const th = TWO_PI * u2;
    let g0 = r * Math.cos(th);
    let g1 = r * Math.sin(th);
    if (periodArr[i] > 0) g0 *= (1.0 + 0.42 * Math.cos(phiArr[i])) * modNorm;
    wMod[i] = g0;
    if (i + 1 < bufLen) {
      if (periodArr[i + 1] > 0) g1 *= (1.0 + 0.42 * Math.cos(phiArr[i + 1])) * modNorm;
      wMod[i + 1] = g1;
    }
  }

  const eBreath = new Float32Array(bufLen);
  const combNorm = 1.0 / Math.sqrt(1.0 + 0.58 * 0.58);
  for (let i = 0; i < bufLen; i++) {
    const T0 = periodArr[i];
    if (T0 > 0 && i >= T0) {
      eBreath[i] = (wMod[i] - 0.58 * wMod[i - T0]) * combNorm;
    } else {
      eBreath[i] = wMod[i];
    }
  }
  dcBlock60HzInPlace(eBreath, sr);

  const fft = new ComplexInverseFFT(nFft);
  const wSqrt = new Float32Array(nFft);
  for (let u = 0; u < nFft; u++) {
    wSqrt[u] = Math.sin((Math.PI * (u + 0.5)) / nFft);
  }

  const plosMono = new Float32Array(bufLen);
  const sibMono = new Float32Array(bufLen);
  const breathMono = new Float32Array(bufLen);

  const xuRe = new Float64Array(nBins);
  const xuIm = new Float64Array(nBins);
  const xbRe = new Float64Array(nBins);
  const xbIm = new Float64Array(nBins);
  const yuRe = new Float64Array(nBins);
  const yuIm = new Float64Array(nBins);
  const ybRe = new Float64Array(nBins);
  const ybIm = new Float64Array(nBins);
  const hEnv = new Float64Array(nBins);
  const phMin = new Float64Array(nBins);
  const hUnv = new Float64Array(nBins);
  const hBr = new Float64Array(nBins);
  const twoOverNfft = 2.0 / nFft;

  for (let f = 0; f < totalFramesSong; f++) {
    const gTot = gainGrid[f];
    if (gTot <= 1e-7) continue;
    const center = f * hop;
    if (center >= nSamples) break;
    const start = center - halfW;
    if (start < 0 || start + nFft > bufLen) continue;

    const vMix = Math.max(0.0, Math.min(1.0, (vdegGrid[f] - 0.1) / 0.55));
    const gUnv = gTot * Math.sqrt(1.0 - vMix);
    const gHarm = gTot * Math.sqrt(vMix);
    const fVal = f0Grid[f];
    const doUnv = gUnv > 1e-7;
    const doBr = gHarm > 1e-7 && fVal > 60.0;
    if (!doUnv && !doBr) continue;

    const cepOff = f * nCep;
    hEnv.fill(0.0);
    phMin.fill(0.0);
    for (let m = 0; m < nCep; m++) {
      const cm = cepGrid[cepOff + m];
      if (Math.abs(cm) < 1e-8) continue;
      const bOff = m * nBins;
      for (let k = 0; k < nBins; k++) {
        hEnv[k] += cm * bLn[bOff + k];
        phMin[k] += cm * phiRad[bOff + k];
      }
    }
    hEnv[0] = 0.0;
    for (let k = 1; k < nBins; k++) {
      hEnv[k] = Math.exp(Math.max(-14.0, Math.min(8.0, hEnv[k]))) * bandTaper[k];
    }

    const pFrac = plosiveGrid[f];
    if (doUnv) {
      let powU = 0.0;
      hUnv[0] = 0.0;
      for (let k = 1; k < nBins; k++) {
        const tilt = pFrac + (1.0 - pFrac) * sibHp[k];
        const hu = hEnv[k] * tilt;
        hUnv[k] = hu;
        powU += hu * hu;
      }
      const scaleU = gUnv / Math.sqrt(twoOverNfft * powU + 1e-18);
      for (let k = 1; k < nBins; k++) hUnv[k] *= scaleU;
    }

    if (doBr) {
      let powE = 0.0;
      for (let k = 1; k < nBins; k++) powE += hEnv[k] * hEnv[k];
      const scaleE = gHarm / Math.sqrt(twoOverNfft * powE + 1e-18);
      const aperOff = f * nAper;
      const subLo = 0.85 * fVal;
      const subSpan = Math.max(0.35 * fVal, 35.0);
      let aCur = 0;
      hBr[0] = 0.0;
      for (let k = 1; k < nBins; k++) {
        const lf = logFreqs[k];
        let apLog;
        if (lf <= logAperC[0]) {
          apLog = aperGrid[aperOff];
        } else if (lf >= logAperC[nAper - 1]) {
          apLog = aperGrid[aperOff + nAper - 1];
        } else {
          while (aCur < nAper - 2 && lf > logAperC[aCur + 1]) aCur++;
          const fr = (lf - logAperC[aCur]) / (logAperC[aCur + 1] - logAperC[aCur]);
          apLog = (1.0 - fr) * aperGrid[aperOff + aCur] + fr * aperGrid[aperOff + aCur + 1];
        }
        const hnrAmp = Math.pow(10.0, 0.5 * apLog);
        const subCut = Math.max(0.0, Math.min(1.0, (freqs[k] - subLo) / subSpan));
        hBr[k] = hEnv[k] * scaleE * hnrAmp * subCut;
      }
    }

    fft.dualRealForward(eUnv, eBreath, start, wSqrt, nFft, xuRe, xuIm, xbRe, xbIm);

    yuRe.fill(0.0);
    yuIm.fill(0.0);
    ybRe.fill(0.0);
    ybIm.fill(0.0);
    for (let k = 1; k < nBins - 1; k++) {
      const cPh = Math.cos(phMin[k]);
      const sPh = Math.sin(phMin[k]);
      if (doUnv) {
        const huRe = hUnv[k] * cPh;
        const huIm = hUnv[k] * sPh;
        const xr = xuRe[k];
        const xi = xuIm[k];
        yuRe[k] = xr * huRe - xi * huIm;
        yuIm[k] = xr * huIm + xi * huRe;
      }
      if (doBr) {
        const hbRe = hBr[k] * cPh;
        const hbIm = hBr[k] * sPh;
        const xr = xbRe[k];
        const xi = xbIm[k];
        ybRe[k] = xr * hbRe - xi * hbIm;
        ybIm[k] = xr * hbIm + xi * hbRe;
      }
    }

    fft.dualRealInverse(yuRe, yuIm, ybRe, ybIm);
    if (doUnv) {
      const sibFrac = 1.0 - pFrac;
      for (let u = 0; u < nFft; u++) {
        const yw = fft.re[u] * wSqrt[u];
        if (pFrac > 0.0) plosMono[start + u] += pFrac * yw;
        if (sibFrac > 0.0) sibMono[start + u] += sibFrac * yw;
      }
    }
    if (doBr) {
      for (let u = 0; u < nFft; u++) {
        breathMono[start + u] += fft.im[u] * wSqrt[u];
      }
    }
  }

  // Pass 4: Track 07 (`backing_harmony`) — Independent L/R True Stereo Chorus Voices
  const maxHBack = 28;
  const backA = new Float64Array(maxHBack);
  const backB = new Float64Array(maxHBack);
  const curBack = new Float64Array(maxHBack);
  const dBack = new Float64Array(maxHBack);

  function synthBackingChannel(gainArr, frameLag, detuneMult, vibHz, vibCos, initPhi) {
    const outCh = new Float32Array(nSamples);
    let ffOut = 0.0;

    function evalBackFrame(fIdx, outAmp) {
      outAmp.fill(0.0);
      const gB = gainArr[fIdx];
      const fSrc = Math.max(0, fIdx - frameLag);
      const ffBase = f0BackGrid[fSrc];
      if (gB <= 1e-6 || ffBase <= 60.0) {
        ffOut = 0.0;
        return 0;
      }
      const tSec = (fIdx * hop) / sr;
      const vib = vibCos ? Math.cos(TWO_PI * vibHz * tSec) : Math.sin(TWO_PI * vibHz * tSec);
      const ff = ffBase * detuneMult * (1.0 + 0.0018 * vib);
      ffOut = ff;
      const nAct = Math.min(maxHBack, Math.floor(7600.0 / ff));
      if (nAct <= 0) return 0;

      const cepOff = fSrc * nCep;
      let sumSq = 0.0;
      for (let h = 0; h < nAct; h++) {
        const fh = (h + 1) * ff;
        const fcC = Math.max(fMin, Math.min(fMax, fh));
        const uH = (21.4 * Math.log10(1.0 + 0.00437 * fcC) - e0) * invESpan;
        const uIdx = Math.max(0, Math.min(4095, Math.round(uH * 4095.0))) * nCep;
        let dbH = 0.0;
        for (let m = 0; m < nCep; m++) dbH += bCosLut[uIdx + m] * cepGrid[cepOff + m];
        const rB = fh / 3800.0;
        const warmRoll = (1.0 / (1.0 + rB * rB)) * Math.max(0.0, Math.min(1.0, (7600.0 - fh) / 1200.0));
        const ra = Math.pow(10.0, dbH / 20.0) * warmRoll;
        relScratch[h] = ra;
        sumSq += ra * ra;
      }
      const scale = gB / Math.sqrt(0.5 * sumSq + 1e-18);
      for (let h = 0; h < nAct; h++) outAmp[h] = relScratch[h] * scale;
      return nAct;
    }

    let phiC = initPhi;
    let na = evalBackFrame(0, backA);
    let ffA = ffOut;
    for (let f = 0; f < nHops; f++) {
      const nb = evalBackFrame(f + 1, backB);
      const ffB = ffOut;
      if (ffA <= 60.0 && ffB <= 60.0) {
        phiC = initPhi;
        backA.set(backB);
        ffA = ffB;
        na = nb;
        continue;
      }
      let fS;
      let fE;
      if (ffA <= 60.0 && ffB > 60.0) {
        fS = ffB;
        fE = ffB;
      } else if (ffA > 60.0 && ffB <= 60.0) {
        fS = ffA;
        fE = ffA;
      } else {
        fS = ffA;
        fE = ffB;
      }

      const base = f * hop;
      const lim = Math.min(hop, nSamples - base);
      const df = (fE - fS) * invHop;
      const nH = Math.max(na, nb);
      if (nH > 0 && lim > 0) {
        for (let h = 0; h < nH; h++) {
          curBack[h] = backA[h];
          dBack[h] = (backB[h] - backA[h]) * invHop;
        }
        for (let s = 0; s < lim; s++) {
          const fc = fS + df * s;
          phiC += twoPiOverSr * fc;
          if (phiC >= TWO_PI) phiC -= TWO_PI * Math.floor(phiC / TWO_PI);

          const c1 = Math.cos(phiC);
          const s1 = Math.sin(phiC);
          const twoC1 = 2.0 * c1;
          let cm2 = 1.0;
          let cm1 = c1;
          let sm2 = 0.0;
          let sm1 = s1;

          let acc = curBack[0] * (cStat[0] * c1 + sStat[0] * s1);
          curBack[0] += dBack[0];
          for (let h = 1; h < nH; h++) {
            const ch = twoC1 * cm1 - cm2;
            const sh = twoC1 * sm1 - sm2;
            cm2 = cm1;
            cm1 = ch;
            sm2 = sm1;
            sm1 = sh;
            acc += curBack[h] * (cStat[h] * ch + sStat[h] * sh);
            curBack[h] += dBack[h];
          }
          outCh[base + s] = acc;
        }
      }
      backA.set(backB);
      ffA = ffB;
      na = nb;
    }
    return outCh;
  }

  const backL = synthBackingChannel(backLSmooth, 3, Math.pow(2.0, 6.5 / 1200.0), 5.1, false, 1.7);
  const backR = synthBackingChannel(backRSmooth, 5, Math.pow(2.0, -6.5 / 1200.0), 5.4, true, 4.2);

  // Pan Lead tracks (`01..06`) + Build Warm Space Send (`08`)
  const subL = new Float32Array(nSamples);
  const subR = new Float32Array(nSamples);
  const bodyL = new Float32Array(nSamples);
  const bodyR = new Float32Array(nSamples);
  const sheenL = new Float32Array(nSamples);
  const sheenR = new Float32Array(nSamples);
  const plosL = new Float32Array(nSamples);
  const plosR = new Float32Array(nSamples);
  const sibL = new Float32Array(nSamples);
  const sibR = new Float32Array(nSamples);
  const breathL = new Float32Array(nSamples);
  const breathR = new Float32Array(nSamples);
  const spaceSendMono = new Float32Array(nSamples);

  for (let n = 0; n < nSamples; n++) {
    const idxF = n * invHop;
    const f0I = Math.min(totalFramesSong - 2, idxF | 0);
    const frac = idxF - f0I;
    const pan = (1.0 - frac) * panSmooth[f0I] + frac * panSmooth[f0I + 1];
    const ang = (pan + 1.0) * (Math.PI * 0.25);
    const gL = Math.SQRT2 * Math.cos(ang);
    const gR = Math.SQRT2 * Math.sin(ang);

    const sm = subMono[n];
    const bm = bodyMono[n];
    const shm = sheenMono[n];
    const plm = plosMono[n];
    const sbm = sibMono[n];
    const brm = breathMono[n];

    subL[n] = sm * gL;
    subR[n] = sm * gR;
    bodyL[n] = bm * gL;
    bodyR[n] = bm * gR;
    sheenL[n] = shm * gL;
    sheenR[n] = shm * gR;
    plosL[n] = plm * gL;
    plosR[n] = plm * gR;
    sibL[n] = sbm * gL;
    sibR[n] = sbm * gR;
    breathL[n] = brm * gL;
    breathR[n] = brm * gR;

    spaceSendMono[n] = sm + bm + 0.5 * shm + 0.2 * (backL[n] + backR[n]);
  }

  const [spaceL, spaceR] = synthesizeSpaceFdn(rawFdn, spaceSendMono);

  let peakRaw = 0.0;
  for (let n = 0; n < nSamples; n++) {
    const ml = Math.abs(subL[n] + bodyL[n] + sheenL[n] + plosL[n] + sibL[n] + breathL[n] + backL[n] + spaceL[n]);
    const mr = Math.abs(subR[n] + bodyR[n] + sheenR[n] + plosR[n] + sibR[n] + breathR[n] + backR[n] + spaceR[n]);
    if (ml > peakRaw) peakRaw = ml;
    if (mr > peakRaw) peakRaw = mr;
  }
  if (peakRaw > 0.94) {
    const nLh = Math.ceil(nSamples / hop);
    const pkH = new Float32Array(nLh);
    for (let hi = 0; hi < nLh; hi++) {
      const s0 = hi * hop;
      const s1 = Math.min(nSamples, s0 + hop);
      let mp = 0.0;
      for (let n = s0; n < s1; n++) {
        const ml = Math.abs(subL[n] + bodyL[n] + sheenL[n] + plosL[n] + sibL[n] + breathL[n] + backL[n] + spaceL[n]);
        const mr = Math.abs(subR[n] + bodyR[n] + sheenR[n] + plosR[n] + sibR[n] + breathR[n] + backR[n] + spaceR[n]);
        if (ml > mp) mp = ml;
        if (mr > mp) mp = mr;
      }
      pkH[hi] = mp;
    }
    const gKnots = new Float32Array(nLh + 1);
    gKnots.fill(1.0);
    for (let ki = 0; ki <= nLh; ki++) {
      const pLoc = Math.max(pkH[Math.max(0, ki - 1)], pkH[Math.min(nLh - 1, ki)]);
      if (pLoc > 0.94) gKnots[ki] = 0.94 / pLoc;
    }
    const allBufs = [
      subL, subR, bodyL, bodyR, sheenL, sheenR,
      plosL, plosR, sibL, sibR, breathL, breathR,
      backL, backR, spaceL, spaceR,
    ];
    for (let hi = 0; hi < nLh; hi++) {
      const ga = gKnots[hi];
      const gb = gKnots[hi + 1];
      if (ga >= 0.999999 && gb >= 0.999999) continue;
      const s0 = hi * hop;
      const s1 = Math.min(nSamples, s0 + hop);
      const dg = (gb - ga) * invHop;
      for (let bi = 0; bi < allBufs.length; bi++) {
        const arr = allBufs[bi];
        let gCur = ga;
        for (let n = s0; n < s1; n++) {
          arr[n] *= gCur;
          gCur += dg;
        }
      }
    }
  }

  return {
    sampleRate: sr,
    numSamples: nSamples,
    layers: {
      lead_sub_f0: [subL, subR],
      lead_vowel_body: [bodyL, bodyR],
      lead_harmonic_sheen: [sheenL, sheenR],
      lead_plosive_burst: [plosL, plosR],
      lead_sibilant: [sibL, sibR],
      lead_velvet_breath: [breathL, breathR],
      backing_harmony: [backL, backR],
      vocal_space: [spaceL, spaceR],
    },
    uiEvents,
    prototypes: atlas.prototypes.map((p) => ({
      id: p.id,
      pinyin: p.pinyin,
      initialClass: p.initialClass,
      count: p.count,
    })),
  };
}
