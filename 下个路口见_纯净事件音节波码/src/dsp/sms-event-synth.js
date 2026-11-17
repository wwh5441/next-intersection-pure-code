import { ComplexInverseFFT } from './fft.js';
import { synthesizeSpaceFdn } from './reverb-fdn.js';
import {
  buildEnvelopeKnotPositions,
  buildUniformKnotPositions,
  unpackEventStream,
  unpackSpeakerCore,
  unpackSyllableAtlas,
} from './unpacker.js';

const DB_FLOOR = -58.0;
const DB_CEIL = 4.0;
const F0_CENTS_UNIT = 5.0;
const VOWEL_ANCHORS_LEN = 5;
const CONS_ANCHORS_LEN = 4;
const TWO_PI = 2.0 * Math.PI;

function hzToErb(f) {
  return 21.4 * Math.log10(1.0 + 0.00437 * f);
}

function erbToHz(e) {
  return (Math.pow(10.0, e / 21.4) - 1.0) / 0.00437;
}

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

function buildBh92ColaWindow(winLen, hop) {
  const wBh = new Float64Array(winLen);
  for (let u = 0; u < winLen; u++) {
    const ang = (TWO_PI * u) / winLen;
    wBh[u] =
      0.35875 -
      0.48829 * Math.cos(ang) +
      0.14128 * Math.cos(2.0 * ang) -
      0.01168 * Math.cos(3.0 * ang);
  }
  const nOv = Math.floor(winLen / hop);
  const wOla = new Float32Array(winLen);
  for (let u = 0; u < winLen; u++) {
    let denom = 0.0;
    for (let r = 0; r < nOv; r++) {
      const idx = (u + r * hop) % winLen;
      denom += wBh[idx] * wBh[idx];
    }
    wOla[u] = wBh[u] / denom;
  }
  return wOla;
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

  const nFft = config.fftSize;       // 2048
  const winLen = config.windowSize;  // 1024
  const halfW = winLen >> 1;
  const nErb = config.harmonicBands; // 40
  const nCep = config.cepstralCoeffs; // 18
  const nAper = config.aperiodicBands; // 16
  const maxHarm = config.maxHarmonics; // 40
  const fHMin = config.harmFreqMin;
  const fHMax = config.harmFreqMax;

  const erbMinV = hzToErb(fHMin);
  const erbSpanV = hzToErb(fHMax) - erbMinV;
  const logErb = new Float64Array(nErb);
  for (let b = 0; b < nErb; b++) {
    const e = erbMinV + (erbSpanV * b) / (nErb - 1);
    logErb[b] = Math.log(erbToHz(e));
  }

  const mcepBasis = new Float64Array(nCep * nErb);
  const cepNorm = Math.sqrt(2.0 / nErb);
  for (let m = 1; m <= nCep; m++) {
    for (let b = 0; b < nErb; b++) {
      mcepBasis[(m - 1) * nErb + b] = cepNorm * Math.cos((Math.PI * m * (b + 0.5)) / nErb);
    }
  }

  // 4-term Blackman-Harris (-92dB sidelobes) exact COLA synthesis window
  const wOla = buildBh92ColaWindow(winLen, hop);
  let eOlaSum = 0.0;
  for (let u = 0; u < winLen; u++) eOlaSum += wOla[u] * wOla[u];
  const eOlaOverHop = eOlaSum / hop;

  const binHz = sr / nFft;
  const aperMinLog = Math.log(config.aperFreqMin);
  const aperMaxLog = Math.log(config.aperFreqMax);
  const aperB0 = new Int32Array(nAper);
  const aperB1 = new Int32Array(nAper);
  const aperScale = new Float64Array(nAper);
  const vowelSheenMask = new Float64Array(nAper);
  for (let b = 0; b < nAper; b++) {
    const lo = Math.exp(aperMinLog + ((aperMaxLog - aperMinLog) * b) / nAper);
    const hi = Math.exp(aperMinLog + ((aperMaxLog - aperMinLog) * (b + 1)) / nAper);
    const fc = Math.sqrt(lo * hi);
    vowelSheenMask[b] = Math.max(0.0, Math.min(1.0, (fc - 5000.0) / 2800.0));
    const b0 = Math.max(3, Math.floor(lo / binHz));
    const b1 = Math.min((nFft >> 1) - 1, Math.max(b0 + 1, Math.ceil(hi / binHz)));
    aperB0[b] = b0;
    aperB1[b] = b1;
    aperScale[b] = nFft / Math.sqrt(2.0 * (b1 - b0) * eOlaOverHop);
  }

  const ifft = new ComplexInverseFFT(nFft);
  const halfBinsPlus1 = (nFft >> 1) + 1;
  const a1Re = new Float64Array(halfBinsPlus1);
  const a1Im = new Float64Array(halfBinsPlus1);
  const a2Re = new Float64Array(halfBinsPlus1);
  const a2Im = new Float64Array(halfBinsPlus1);

  const totalFramesSong = Math.floor(nSamples / hop) + 4;
  const uGrid = new Float32Array(totalFramesSong * maxHarm);
  const vGrid = new Float32Array(totalFramesSong * maxHarm);
  const f0Grid = new Float32Array(totalFramesSong);
  const nhGrid = new Uint8Array(totalFramesSong);
  const vDomGrid = new Float32Array(totalFramesSong);

  const aperMono = new Float32Array(nSamples + nFft);
  const glottalMod = new Float32Array(nSamples + hop);
  glottalMod.fill(1.0);
  const panTrack = new Float32Array(totalFramesSong);
  const widthTrack = new Float32Array(totalFramesSong);

  let pendingCenter = -1;
  const flushDualNoise = (center2) => {
    ifft.dualRealInverse(a1Re, a1Im, a2Re, a2Im);
    if (pendingCenter >= 0) {
      const st1 = pendingCenter - halfW;
      const s0 = Math.max(0, st1);
      const g0 = s0 - st1;
      const span = Math.min(aperMono.length, st1 + winLen) - s0;
      for (let u = 0; u < span; u++) {
        const gu = g0 + u;
        const srcIdx = gu < halfW ? nFft - halfW + gu : gu - halfW;
        aperMono[s0 + u] += ifft.re[srcIdx] * wOla[gu];
      }
    }
    if (center2 >= 0) {
      const st2 = center2 - halfW;
      const s0 = Math.max(0, st2);
      const g0 = s0 - st2;
      const span = Math.min(aperMono.length, st2 + winLen) - s0;
      for (let u = 0; u < span; u++) {
        const gu = g0 + u;
        const srcIdx = gu < halfW ? nFft - halfW + gu : gu - halfW;
        aperMono[s0 + u] += ifft.im[srcIdx] * wOla[gu];
      }
    }
    pendingCenter = -1;
  };

  const kPca = evMeta.kPca;
  const kAper = evMeta.kAper;
  const exprStep = evMeta.exprStep;
  const f0Step = evMeta.f0Step;
  const envStep = evMeta.envStep;
  const qStep = evMeta.qStepDb;

  const rand = createMulberry32(20261115);
  const minPhaseScale = -(Math.LN10 / 20.0) * cepNorm;
  const invHop = 1.0 / hop;
  const twoPiOverSr = TWO_PI / sr;
  const rolloffStart = 5400.0;
  const rolloffSpan = Math.max(200.0, fHMax - rolloffStart);

  const curMcep = new Float64Array(nCep);
  const curErb = new Float64Array(nErb);
  const curAper = new Float64Array(nAper);
  const relAmps = new Float64Array(maxHarm);
  const psiArr = new Float64Array(maxHarm);
  const uiEvents = [];

  // Pass 1: Decode purified events onto global quadrature grid + batch BH92 COLA consonants/sheen
  for (let ei = 0; ei < events.length; ei++) {
    const ev = events[ei];
    const L = ev.totalFrames;
    if (L <= 0) continue;
    const sF = ev.startFrame;
    const cf = ev.consFrames;
    const proto = atlas.prototypes[ev.syllableId];

    const envPos = buildEnvelopeKnotPositions(L, cf, envStep);
    const ghCurve = interp1dKnots(envPos, uint8ToGain(ev.gainHQ, refPeak), L);
    const gaCurve = interp1dKnots(envPos, uint8ToGain(ev.gainAQ, refPeak), L);

    if (!ev.legatoPrev && L >= 4) {
      ghCurve[0] *= 0.42;
      gaCurve[0] *= 0.55;
    }
    if (!ev.legatoNext && L >= 4) {
      ghCurve[L - 1] *= 0.42;
      gaCurve[L - 1] *= 0.55;
    }

    let f0Curve;
    if (!ev.unvoicedOnly) {
      const f0Pos = buildUniformKnotPositions(L, f0Step);
      const deltaSemi = new Float32Array(ev.f0DeltaQ.length);
      for (let k = 0; k < deltaSemi.length; k++) deltaSemi[k] = (ev.f0DeltaQ[k] * F0_CENTS_UNIT) / 100.0;
      const semiInterp = interp1dKnots(f0Pos, deltaSemi, L);
      f0Curve = new Float32Array(L);
      const baseMidi = ev.baseMidiQ * 0.25;
      for (let r = 0; r < L; r++) {
        f0Curve[r] = 440.0 * Math.pow(2.0, (baseMidi + semiInterp[r] - 69.0) / 12.0);
      }
    } else {
      f0Curve = new Float32Array(L);
    }

    const exprPos = buildUniformKnotPositions(L, exprStep);
    const nx = exprPos.length;
    const dmKnots = new Float32Array(nx * nCep);
    if (!ev.unvoicedOnly && ev.exprMcepQ.length > 0) {
      for (let k = 0; k < nx; k++) {
        for (let m = 0; m < kPca; m++) {
          const w = ev.exprMcepQ[k * kPca + m] * qStep;
          const bOff = m * nCep;
          for (let c = 0; c < nCep; c++) {
            dmKnots[k * nCep + c] += w * speaker.mcepPcaBasis[bOff + c];
          }
        }
      }
    }

    const daKnots = new Float32Array(nx * nAper);
    if (!ev.silentAper && ev.exprAperQ.length > 0) {
      const qStepA = qStep * 1.35;
      for (let k = 0; k < nx; k++) {
        for (let m = 0; m < kAper; m++) {
          const w = ev.exprAperQ[k * kAper + m] * qStepA;
          const bOff = m * nAper;
          for (let b = 0; b < nAper; b++) {
            daKnots[k * nAper + b] += w * speaker.aperPcaBasis[bOff + b];
          }
        }
      }
    }

    const panVal = ev.stereoPanQ / 64.0;
    const widthVal = ev.stereoWidthQ / 255.0;
    for (let r = 0; r < L && sF + r < totalFramesSong; r++) {
      panTrack[sF + r] = panVal;
      widthTrack[sF + r] = widthVal;
    }

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
      midi: Number((ev.baseMidiQ * 0.25).toFixed(2)),
      pitchPreview: uiPitch,
    });

    const vot = L > 1 ? Math.min(L - 1, Math.max(1, cf)) : 0;
    let exSeg = 0;

    for (let r = 0; r < L; r++) {
      const fAbs = sF + r;
      const center = fAbs * hop;
      if (center >= nSamples || fAbs >= totalFramesSong) break;

      while (exSeg < nx - 2 && r > exprPos[exSeg + 1]) exSeg++;
      const ex0 = exprPos[exSeg];
      const exNext = Math.min(nx - 1, exSeg + 1);
      const ex1 = exprPos[exNext];
      const exFrac = ex1 > ex0 ? (r - ex0) / (ex1 - ex0) : 0.0;

      const uV = r <= vot || L - vot <= 1 ? 0.0 : (r - vot) / (L - vot - 1);
      const idxV = uV * (VOWEL_ANCHORS_LEN - 1);
      const iv0 = Math.min(VOWEL_ANCHORS_LEN - 2, idxV | 0);
      const fv = idxV - iv0;

      for (let c = 0; c < nCep; c++) {
        const pVal =
          speaker.meanMcep[c] +
          (1.0 - fv) * proto.vowelMcepKnots[iv0 * nCep + c] +
          fv * proto.vowelMcepKnots[(iv0 + 1) * nCep + c];
        const dVal = (1.0 - exFrac) * dmKnots[exSeg * nCep + c] + exFrac * dmKnots[exNext * nCep + c];
        curMcep[c] = pVal + dVal;
      }

      if (r < vot && vot > 0) {
        const uC = vot > 1 ? (r / (vot - 1)) * (CONS_ANCHORS_LEN - 1) : 0.0;
        const ic0 = Math.min(CONS_ANCHORS_LEN - 2, uC | 0);
        const fc = uC - ic0;
        for (let b = 0; b < nAper; b++) {
          const pA =
            speaker.aperMean[b] +
            (1.0 - fc) * proto.consAperKnots[ic0 * nAper + b] +
            fc * proto.consAperKnots[(ic0 + 1) * nAper + b];
          const dA = (1.0 - exFrac) * daKnots[exSeg * nAper + b] + exFrac * daKnots[exNext * nAper + b];
          curAper[b] = pA + dA;
        }
      } else {
        for (let b = 0; b < nAper; b++) {
          const pA =
            speaker.aperMean[b] +
            (1.0 - fv) * proto.vowelAperKnots[iv0 * nAper + b] +
            fv * proto.vowelAperKnots[(iv0 + 1) * nAper + b];
          const dA = (1.0 - exFrac) * daKnots[exSeg * nAper + b] + exFrac * daKnots[exNext * nAper + b];
          curAper[b] = pA + dA;
        }
      }

      const ghVal = ghCurve[r];
      const gaRaw = gaCurve[r];
      const f0Val = f0Curve[r];
      let hasHarm = ghVal > 1e-6 && f0Val > 60.0 && r >= Math.max(0, ev.voicedOnsetFrame - 1);

      if (hasHarm) {
        const nH = Math.min(maxHarm, Math.floor(fHMax / f0Val));
        if (nH >= 1) {
          for (let b = 0; b < nErb; b++) {
            let s = 0.0;
            for (let m = 0; m < nCep; m++) s += curMcep[m] * mcepBasis[m * nErb + b];
            curErb[b] = s;
          }

          let sumSq = 1e-18;
          let erbCursor = 0;
          for (let hk = 0; hk < nH; hk++) {
            const fk = (hk + 1) * f0Val;
            const lf = Math.log(fk);
            let hDb;
            if (lf <= logErb[0]) hDb = curErb[0];
            else if (lf >= logErb[nErb - 1]) hDb = curErb[nErb - 1];
            else {
              while (erbCursor < nErb - 2 && lf > logErb[erbCursor + 1]) erbCursor++;
              const fr = (lf - logErb[erbCursor]) / (logErb[erbCursor + 1] - logErb[erbCursor]);
              hDb = (1.0 - fr) * curErb[erbCursor] + fr * curErb[erbCursor + 1];
            }
            const uRoll = Math.max(0.0, Math.min(1.0, (fk - rolloffStart) / rolloffSpan));
            const velvet = 0.5 * (1.0 + Math.cos(Math.PI * uRoll));
            const ra = Math.pow(10.0, hDb / 20.0) * (0.06 + 0.94 * velvet);
            relAmps[hk] = ra;
            sumSq += 0.5 * ra * ra;

            const thErb = Math.max(0.0, Math.min(Math.PI, ((hzToErb(fk) - erbMinV) / erbSpanV) * Math.PI));
            let tPh = 0.0;
            for (let m = 1; m <= nCep; m++) tPh += curMcep[m - 1] * Math.sin(m * thErb);
            psiArr[hk] = speaker.glottalPhase[hk] + tPh * minPhaseScale;
          }

          const scaleH = ghVal / Math.sqrt(sumSq);
          const gridBase = fAbs * maxHarm;
          for (let hk = 0; hk < nH; hk++) {
            const amp = scaleH * relAmps[hk];
            const psi = psiArr[hk];
            uGrid[gridBase + hk] = amp * Math.cos(psi);
            vGrid[gridBase + hk] = -amp * Math.sin(psi);
          }
          f0Grid[fAbs] = f0Val;
          nhGrid[fAbs] = nH;
          vDomGrid[fAbs] = ghVal / (ghVal + 1.5 * gaRaw + 1e-9);
        } else {
          hasHarm = false;
        }
      }

      // Voiced Vowel Zero-Hiss Law: <5kHz noise = 0.0, >5kHz velvet sheen <= 0.042 * ghVal
      const gaVal = hasHarm && r >= vot ? Math.min(gaRaw, ghVal * 0.042) : gaRaw;
      if (gaVal > 1e-6) {
        let sumBp = 0.0;
        for (let b = 0; b < nAper; b++) {
          let p = Math.pow(10.0, curAper[b] / 10.0);
          if (hasHarm && r >= vot) p *= vowelSheenMask[b];
          curAper[b] = p;
          sumBp += p;
        }
        if (sumBp > 1e-16) {
          const dstRe = pendingCenter < 0 ? a1Re : a2Re;
          const dstIm = pendingCenter < 0 ? a1Im : a2Im;
          dstRe.fill(0.0);
          dstIm.fill(0.0);
          for (let b = 0; b < nAper; b++) {
            const p = curAper[b];
            if (p <= 0.0) continue;
            const mag = gaVal * Math.sqrt(p / sumBp) * aperScale[b];
            if (mag <= 1e-8) continue;
            const b0 = aperB0[b];
            const b1 = aperB1[b];
            for (let k = b0; k < b1; k++) {
              const ph = TWO_PI * rand();
              dstRe[k] = mag * Math.cos(ph);
              dstIm[k] = mag * Math.sin(ph);
            }
          }
          if (pendingCenter < 0) {
            pendingCenter = center;
          } else {
            flushDualNoise(center);
          }
        }
      }
    }
  }

  if (pendingCenter >= 0) {
    a2Re.fill(0.0);
    a2Im.fill(0.0);
    flushDualNoise(-1);
  }

  // Pass 2: Global Sample-Continuous Complex Phasor Recurrence + Glottal Open-Phase Air Modulation
  const harmMono = new Float32Array(nSamples + hop);
  const uCur = new Float64Array(maxHarm);
  const vCur = new Float64Array(maxHarm);
  const du = new Float64Array(maxHarm);
  const dv = new Float64Array(maxHarm);
  const nHops = Math.floor(nSamples / hop) + 1;
  let phi0 = 0.0;

  for (let f = 0; f < nHops; f++) {
    const nh0 = nhGrid[f];
    const nh1 = nhGrid[f + 1];
    const nh = nh0 > nh1 ? nh0 : nh1;
    if (nh === 0) continue;

    let fStart;
    let fEnd;
    if (nh0 === 0 && nh1 > 0) {
      phi0 = 0.0;
      fStart = f0Grid[f + 1];
      fEnd = fStart;
    } else if (nh0 > 0 && nh1 === 0) {
      fStart = f0Grid[f];
      fEnd = fStart;
    } else {
      fStart = f0Grid[f];
      fEnd = f0Grid[f + 1];
    }

    const off0 = f * maxHarm;
    const off1 = (f + 1) * maxHarm;
    for (let k = 0; k < nh; k++) {
      const u0 = uGrid[off0 + k];
      const v0 = vGrid[off0 + k];
      uCur[k] = u0;
      vCur[k] = v0;
      du[k] = (uGrid[off1 + k] - u0) * invHop;
      dv[k] = (vGrid[off1 + k] - v0) * invHop;
    }

    const voicedBoth = nh0 > 0 && nh1 > 0;
    const mG = voicedBoth ? 0.45 * 0.5 * (vDomGrid[f] + vDomGrid[f + 1]) : 0.0;
    const gNorm = voicedBoth ? 1.0 / Math.sqrt(1.0 + 0.5 * mG * mG) : 1.0;

    const df = (fEnd - fStart) * invHop;
    const baseSample = f * hop;
    for (let s = 0; s < hop; s++) {
      const fCur = fStart + df * s;
      phi0 += twoPiOverSr * fCur;
      if (phi0 >= TWO_PI) phi0 -= TWO_PI;

      const z1r = Math.cos(phi0);
      const z1i = Math.sin(phi0);
      let zkr = z1r;
      let zki = z1i;
      let sum = 0.0;

      for (let k = 0; k < nh; k++) {
        sum += uCur[k] * zkr + vCur[k] * zki;
        uCur[k] += du[k];
        vCur[k] += dv[k];
        const nr = zkr * z1r - zki * z1i;
        zki = zkr * z1i + zki * z1r;
        zkr = nr;
      }
      const idxOut = baseSample + s;
      harmMono[idxOut] = sum;
      if (voicedBoth) {
        glottalMod[idxOut] = (1.0 + mG * z1r) * gNorm;
      }
    }
  }

  const drySum = new Float32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    aperMono[i] *= glottalMod[i];
    drySum[i] = harmMono[i] + aperMono[i];
  }

  const panSmooth = smoothFrameTrack(panTrack);
  const widthSmooth = smoothFrameTrack(widthTrack);

  const harmL = new Float32Array(nSamples);
  const harmR = new Float32Array(nSamples);
  const aperL = new Float32Array(nSamples);
  const aperR = new Float32Array(nSamples);
  const widthSample = new Float32Array(nSamples);

  for (let n = 0; n < nSamples; n++) {
    const idxF = n * invHop;
    const f0I = Math.min(panSmooth.length - 2, idxF | 0);
    const frac = idxF - f0I;
    const pan = (1.0 - frac) * panSmooth[f0I] + frac * panSmooth[f0I + 1];
    widthSample[n] = (1.0 - frac) * widthSmooth[f0I] + frac * widthSmooth[f0I + 1];
    const ang = (pan + 1.0) * (Math.PI * 0.25);
    const gL = Math.SQRT2 * Math.cos(ang);
    const gR = Math.SQRT2 * Math.sin(ang);
    harmL[n] = harmMono[n] * gL;
    harmR[n] = harmMono[n] * gR;
    aperL[n] = aperMono[n] * gL;
    aperR[n] = aperMono[n] * gR;
  }

  // Layer 3: Band-Limited (240Hz - 5.4kHz) Silky Stereo Chorus Doubler (zero sibilance splatter!)
  const bpDry = new Float32Array(nSamples);
  const hpA = Math.exp((-TWO_PI * 240.0) / sr);
  const lpA = Math.exp((-TWO_PI * 5400.0) / sr);
  const oneMLp = 1.0 - lpA;
  let px = 0.0;
  let hy = 0.0;
  let ly = 0.0;
  for (let n = 0; n < nSamples; n++) {
    const x = drySum[n];
    hy = hpA * (hy + x - px);
    px = x;
    ly = oneMLp * hy + lpA * ly;
    bpDry[n] = ly;
  }

  const dblL = new Float32Array(nSamples);
  const dblR = new Float32Array(nSamples);
  for (let n = 0; n < nSamples; n++) {
    const w = widthSample[n] * 0.28;
    if (w < 1e-4) continue;
    const dL = 595.0 + 34.0 * Math.sin(0.63 * twoPiOverSr * n);
    const dR = 868.0 + 38.0 * Math.cos(0.74 * twoPiOverSr * n);
    const pL = Math.max(0.0, Math.min(nSamples - 2.0, n - dL));
    const iL = pL | 0;
    const fL = pL - iL;
    dblL[n] = ((1.0 - fL) * bpDry[iL] + fL * bpDry[iL + 1]) * w;

    const pR = Math.max(0.0, Math.min(nSamples - 2.0, n - dR));
    const iR = pR | 0;
    const fR = pR - iR;
    dblR[n] = -((1.0 - fR) * bpDry[iR] + fR * bpDry[iR + 1]) * w;
  }

  // Layer 4: Native 48kHz Zero-Aliasing 8-Line Householder Studio Vocal Plate
  const [fdnL, fdnR] = synthesizeSpaceFdn(rawFdn, drySum);

  return {
    sampleRate: sr,
    numSamples: nSamples,
    layers: {
      harmonic_core: [harmL, harmR],
      consonant_air: [aperL, aperR],
      stereo_double: [dblL, dblR],
      space_fdn: [fdnL, fdnR],
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
