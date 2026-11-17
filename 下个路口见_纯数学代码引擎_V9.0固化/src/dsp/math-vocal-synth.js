// Pure Mathematical Vocal Tract, 2nd-Order `--` Coarticulation ODE & Fant LF Harmonic Synthesizer
// Generates 4 Orthogonal Vocal Stems:
//   01 vocal_chest_f0      (80 - 720 Hz chest fundamental & H1..H3)
//   02 vocal_formant_body  (720 - 15,600 Hz Mandarin cavity resonances + Velvet sheen)
//   03 vocal_consonant_air (23 Mandarin consonant Gaussian bursts/sibilants + Klatt open-phase breath)
//   04 vocal_chorus_harmony(True stereo double-tracked chorus backing harmony)

import { ComplexInverseFFT, DeterministicRng } from './fft-minphase.js';

const TWO_PI = 2.0 * Math.PI;
const LN10_OVER_20 = Math.LN10 / 20.0;

function dbToLin(db, refPeak = 1.0) {
  return db <= -79.5 ? 0.0 : refPeak * Math.pow(10.0, db / 20.0);
}

function interpArray(vals, nFrames) {
  const out = new Float32Array(nFrames);
  if (typeof vals === 'number') {
    out.fill(vals);
    return out;
  }
  const m = vals.length;
  if (m === 1 || nFrames <= 1) {
    out.fill(vals[0]);
    return out;
  }
  for (let i = 0; i < nFrames; i++) {
    const pos = (i * (m - 1)) / (nFrames - 1);
    const k = Math.min(m - 2, Math.floor(pos));
    const frac = pos - k;
    out[i] = vals[k] * (1.0 - frac) + vals[k + 1] * frac;
  }
  return out;
}

function buildErbCosineLut(numModes, fMin, fMax, nLut = 2048) {
  const lut = new Float32Array(nLut * numModes);
  const e0 = 21.4 * Math.log10(1.0 + 0.00437 * fMin);
  const e1 = 21.4 * Math.log10(1.0 + 0.00437 * fMax);
  for (let i = 0; i < nLut; i++) {
    const u = i / (nLut - 1);
    for (let m = 0; m < numModes; m++) {
      lut[i * numModes + m] = Math.cos(Math.PI * (m + 1) * u);
    }
  }
  return { lut, nLut, e0, invSpan: 1.0 / (e1 - e0) };
}

export function synthesizePureMathVocals(singerPhysics, phonetics, vocalScore) {
  const sr = singerPhysics.sampleRate;
  const hop = singerPhysics.hopSamples;
  const nSamples = singerPhysics.totalSamples;
  const nFrames = Math.floor(nSamples / hop) + 2;
  const M = singerPhysics.numModes; // 12
  const modeStep = singerPhysics.modeStepDb; // 0.4 dB
  const refPeak = singerPhysics.refPeak;
  const dtHop = hop / sr;

  const { MANDARIN_CONSONANTS, MANDARIN_SYLLABLES } = phonetics;

  // Frame-level continuous state tracks
  const f0Hz = new Float32Array(nFrames);
  const gainHarm = new Float32Array(nFrames);
  const vDeg = new Float32Array(nFrames);
  const activeMask = new Uint8Array(nFrames);
  const consGain = new Float32Array(nFrames);
  const plosiveFlag = new Uint8Array(nFrames);
  const backGain = new Float32Array(nFrames);
  const consTypeIdx = new Int16Array(nFrames).fill(-1);

  const consKeys = Object.keys(MANDARIN_CONSONANTS);
  const consKeyMap = new Map(consKeys.map((k, i) => [k, i]));

  // Target 12-mode vocal-tract trajectory before 2nd-order ODE coarticulation
  const targetModes = new Float32Array(nFrames * M);
  const omegaTrack = new Float32Array(nFrames).fill(singerPhysics.coarticulationOmegaVowel);

  // Initialize neutral vocal-tract baseline
  for (let f = 0; f < nFrames; f++) {
    for (let m = 0; m < M; m++) {
      targetModes[f * M + m] = singerPhysics.baseErbModes[m];
    }
  }

  // Pass 1: Lay down musical score notes & intra-syllable [onset -> nucleus -> coda] targets
  const parsedNotes = vocalScore.map((row) => {
    const [startMs, durMs, votMs, hanzi, pinyin, midiSpec, gainDbSpec, timbreSpec, backingDb, flags] = row;
    const fStart = Math.max(0, Math.min(nFrames - 2, Math.round((startMs * 0.001 * sr) / hop)));
    const L = Math.max(2, Math.min(nFrames - fStart, Math.round((durMs * 0.001 * sr) / hop)));
    const cFrames = Math.min(L - 1, Math.max(0, Math.round((votMs * 0.001 * sr) / hop)));
    return {
      fStart, L, cFrames, startMs, durMs, votMs, hanzi, pinyin,
      midiSpec, gainDbSpec, timbreSpec, backingDb,
      legatoPrev: (flags & 1) !== 0,
      legatoNext: (flags & 2) !== 0,
      isPlosive: (flags & 4) !== 0,
      isHook: (flags & 8) !== 0,
    };
  });

  for (let idx = 0; idx < parsedNotes.length; idx++) {
    const ev = parsedNotes[idx];
    const syl = MANDARIN_SYLLABLES[ev.pinyin] || MANDARIN_SYLLABLES.a;
    const iniKey = syl[0];
    const cModes8 = syl[1];
    const [vOnset, vNuc, vCoda] = syl[2];
    const consSpec = MANDARIN_CONSONANTS[iniKey] || MANDARIN_CONSONANTS[''];
    const cIdx = consKeyMap.get(iniKey) ?? 0;

    const midiArr = interpArray(ev.midiSpec, ev.L);
    const gainDbArr = interpArray(ev.gainDbSpec, ev.L);
    const isSteadyLong = typeof ev.midiSpec === 'number' && ev.durMs >= 420;
    const isUnvoicedInit = ev.cFrames >= 2 && iniKey !== '' && !['m', 'n', 'l', 'y', 'w'].includes(iniKey);

    // Look ahead to next contiguous syllable for anticipatory `--` coarticulation!
    let nextLocus = null;
    let nextDacR = 0.0;
    if (ev.legatoNext && idx + 1 < parsedNotes.length) {
      const nxt = parsedNotes[idx + 1];
      if (nxt.fStart - (ev.fStart + ev.L) <= 6) {
        const nxtSyl = MANDARIN_SYLLABLES[nxt.pinyin] || syl;
        const nxtCons = MANDARIN_CONSONANTS[nxtSyl[0]] || consSpec;
        nextDacR = nxtCons[1];
        nextLocus = nxtSyl[2][0]; // Onset cavity locus of next character!
      }
    }

    const anticFrames = nextLocus ? Math.min(16, Math.floor(ev.L * 0.45)) : 0;

    for (let x = 0; x < ev.L; x++) {
      const f = ev.fStart + x;
      if (f >= nFrames) break;
      activeMask[f] = 1;
      consTypeIdx[f] = cIdx;

      // Pitch with subtle natural 5.45 Hz physiological singer vibrato on sustained notes
      let mVal = midiArr[x];
      if (isSteadyLong) {
        const tLocal = x * dtHop;
        const vibEnv = Math.max(0.0, Math.min(1.0, (tLocal - 0.20) / 0.24));
        mVal += 0.13 * vibEnv * Math.sin(TWO_PI * 5.45 * tLocal + idx * 1.1);
      }
      f0Hz[f] = 440.0 * Math.pow(2.0, (mVal - 69.0) / 12.0);

      // Amplitude & Smooth Voicing Transition (VOT physics)
      const linG = dbToLin(gainDbArr[x], refPeak);
      gainHarm[f] = linG;

      if (isUnvoicedInit && x < ev.cFrames + 2) {
        const xi = Math.max(0.0, Math.min(1.0, (x - (ev.cFrames - 1.5)) / 3.5));
        const sCurve = 0.5 * (1.0 - Math.cos(Math.PI * xi));
        // Sibilants/fricatives have zero low buzz; stops keep tiny pre-voicing bar
        const preVoicedFloor = ['b', 'd', 'g'].includes(iniKey) ? 0.10 : 0.0;
        vDeg[f] = preVoicedFloor + (0.96 - preVoicedFloor) * sCurve;
        const burstBell = Math.sin(Math.PI * Math.min(1.0, (x + 0.5) / Math.max(1.5, ev.cFrames + 1.0)));
        consGain[f] = Math.max(linG, refPeak * 0.085) * (0.55 + 0.45 * burstBell);
        plosiveFlag[f] = consSpec[0] ? 1 : 0;
        omegaTrack[f] = singerPhysics.coarticulationOmegaConsonant;
      } else {
        vDeg[f] = 0.965;
        consGain[f] = linG * 0.08;
        plosiveFlag[f] = 0;
        omegaTrack[f] = singerPhysics.coarticulationOmegaVowel;
      }

      if (ev.backingDb > -75) {
        backGain[f] = dbToLin(ev.backingDb, refPeak);
      }

      // 3-Stage Mandarin Final Trajectory [onset -> nucleus -> coda] + Anticipatory `--` Bend!
      const vStart = Math.min(ev.cFrames, ev.L - 1);
      const uVowel = ev.L > vStart + 1 ? Math.max(0.0, (x - vStart) / (ev.L - 1 - vStart)) : 0.5;
      const anticW = (anticFrames > 0 && x >= ev.L - anticFrames)
        ? 0.52 * nextDacR * Math.pow((x - (ev.L - anticFrames) + 1) / anticFrames, 1.35)
        : 0.0;

      for (let m = 0; m < M; m++) {
        let vm;
        if (x < ev.cFrames && m < cModes8.length) {
          const cBlend = x / Math.max(1, ev.cFrames);
          vm = ((1.0 - cBlend) * cModes8[m] + cBlend * vOnset[m]) * modeStep;
        } else if (uVowel < 0.46) {
          const w = uVowel / 0.46;
          vm = ((1.0 - w) * vOnset[m] + w * vNuc[m]) * modeStep;
        } else {
          const w = (uVowel - 0.46) / 0.54;
          vm = ((1.0 - w) * vNuc[m] + w * vCoda[m]) * modeStep;
        }
        if (anticW > 0.0) {
          vm = (1.0 - anticW) * vm + anticW * (nextLocus[m] * modeStep);
        }
        if (Array.isArray(ev.timbreSpec) && m < ev.timbreSpec.length) {
          vm += ev.timbreSpec[m] * modeStep;
        }
        targetModes[f * M + m] = singerPhysics.baseErbModes[m] + vm;
      }
    }
  }

  // Pass 1b: Bridge intra-phrase micro-gaps (<= 34 frames = 181ms) so glottis & vocal tract stay connected!
  let f = 0;
  while (f < nFrames) {
    if (activeMask[f]) { f++; continue; }
    let gEnd = f;
    while (gEnd < nFrames && !activeMask[gEnd]) gEnd++;
    const gap = gEnd - f;
    if (f > 0 && gEnd < nFrames && gap <= 34) {
      for (let k = f; k < gEnd; k++) {
        const a = (k - f + 1) / (gap + 1);
        f0Hz[k] = f0Hz[f - 1] * Math.pow(f0Hz[gEnd] / Math.max(80.0, f0Hz[f - 1]), a);
        const bridgeG = Math.min(gainHarm[f - 1], gainHarm[gEnd]) * 0.45;
        gainHarm[k] = bridgeG;
        vDeg[k] = Math.min(vDeg[f - 1], vDeg[gEnd]) * 0.75;
        activeMask[k] = 1;
        consTypeIdx[k] = consTypeIdx[gEnd];
        for (let m = 0; m < M; m++) {
          targetModes[k * M + m] = (1.0 - a) * targetModes[(f - 1) * M + m] + a * targetModes[gEnd * M + m];
        }
      }
    }
    f = gEnd;
  }

  // Pass 1c: Smooth log-pitch portamento across contiguous `--` seams + 3-frame isolated phrase micro-fades
  const logF0 = new Float32Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    logF0[i] = f0Hz[i] > 50.0 ? Math.log2(f0Hz[i]) : 0.0;
  }
  const smoothLogF0 = new Float32Array(logF0);
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 1; i < nFrames - 1; i++) {
      if (logF0[i - 1] > 0 && logF0[i] > 0 && logF0[i + 1] > 0) {
        smoothLogF0[i] = 0.23 * logF0[i - 1] + 0.54 * logF0[i] + 0.23 * logF0[i + 1];
      }
    }
    logF0.set(smoothLogF0);
  }
  for (let i = 0; i < nFrames; i++) {
    if (smoothLogF0[i] > 0) f0Hz[i] = Math.pow(2.0, smoothLogF0[i]);
  }

  // Pass 2: Integrate Exact 2nd-Order Critically Damped Articulatory Differential Equation:
  //   q''(t) + 2*w*q'(t) + w^2*q(t) = w^2 * target(t)
  const smoothModes = new Float32Array(nFrames * M);
  const qPos = new Float64Array(M);
  const qVel = new Float64Array(M);
  for (let m = 0; m < M; m++) qPos[m] = targetModes[m];

  for (let i = 0; i < nFrames; i++) {
    const w = omegaTrack[i];
    const expW = Math.exp(-w * dtHop);
    for (let m = 0; m < M; m++) {
      const tgt = targetModes[i * M + m];
      const d0 = qPos[m] - tgt;
      const v0 = qVel[m];
      const d1 = (d0 + (v0 + w * d0) * dtHop) * expW;
      const v1 = (v0 - w * (v0 + w * d0) * dtHop) * expW;
      qPos[m] = tgt + d1;
      qVel[m] = v1;
      smoothModes[i * M + m] = qPos[m];
    }
  }

  // Pass 3: Sample-Accurate Fant LF + Velvet Harmonic Bank Synthesis (Tracks 01, 02, 04)
  const chestL = new Float32Array(nSamples);
  const bodyL = new Float32Array(nSamples);
  const harmBackL = new Float32Array(nSamples);
  const harmBackR = new Float32Array(nSamples);

  const { lut, nLut, e0, invSpan } = buildErbCosineLut(M, singerPhysics.erbFreqMinHz, singerPhysics.erbFreqMaxHz);
  const maxHarm = 52;
  const hAmpPrev = new Float32Array(maxHarm);
  const hAmpCurr = new Float32Array(maxHarm);
  const gPhaseArr = singerPhysics.glottalPhase24;
  const cosDisp = new Float64Array(maxHarm);
  const sinDisp = new Float64Array(maxHarm);
  const cosBackL = new Float64Array(18);
  const sinBackL = new Float64Array(18);
  const cosBackR = new Float64Array(18);
  const sinBackR = new Float64Array(18);
  for (let h = 1; h <= maxHarm; h++) {
    const disp = h <= gPhaseArr.length ? gPhaseArr[h - 1] : (0.618 * h * h);
    cosDisp[h - 1] = Math.cos(disp);
    sinDisp[h - 1] = Math.sin(disp);
    if (h <= 18) {
      cosBackL[h - 1] = Math.cos(0.4 * h);
      sinBackL[h - 1] = Math.sin(0.4 * h);
      cosBackR[h - 1] = Math.cos(0.9 * h);
      sinBackR[h - 1] = Math.sin(0.9 * h);
    }
  }
  const bTilt = new Float32Array(18);
  for (let h = 1; h <= 18; h++) {
    bTilt[h - 1] = h === 1 ? 0.85 : 1.0 / Math.pow(h, 0.25);
  }
  const wBack = new Float32Array(18);
  let glottalPhase = 0.0;
  let backPhaseL = 0.25;
  let backPhaseR = 0.61;
  let silentCount = 100;

  for (let i = 0; i < nFrames - 1; i++) {
    const s0 = i * hop;
    const s1 = Math.min(nSamples, s0 + hop);
    if (s1 <= s0) break;

    const g0 = gainHarm[i] * vDeg[i];
    const g1 = gainHarm[i + 1] * vDeg[i + 1];
    const fStart = f0Hz[i] > 60 ? f0Hz[i] : f0Hz[i + 1];
    const fEnd = f0Hz[i + 1] > 60 ? f0Hz[i + 1] : fStart;

    if ((g0 < 1e-6 && g1 < 1e-6) || fStart < 60.0) {
      silentCount++;
      if (silentCount > 36) glottalPhase = 0.0;
      hAmpPrev.fill(0.0);
      continue;
    }
    silentCount = 0;

    const fMid = 0.5 * (fStart + fEnd);
    const nH = Math.min(maxHarm, Math.floor(15400.0 / fMid));
    hAmpCurr.fill(0.0);

    // Evaluate ERB vocal-tract transfer function at each harmonic h*fMid
    let sumSq = 1e-12;
    let lowRefDb = -60.0;
    for (let h = 1; h <= nH; h++) {
      const fh = h * fMid;
      const u = Math.max(0.0, Math.min(1.0, (21.4 * Math.log10(1.0 + 0.00437 * fh) - e0) * invSpan));
      const lutIdx = Math.min(nLut - 1, Math.round(u * (nLut - 1))) * M;
      const modeOff = i * M;
      let db = 0.0;
      for (let m = 0; m < M; m++) {
        db += smoothModes[modeOff + m] * lut[lutIdx + m];
      }
      if (h <= 4 && db > lowRefDb) lowRefDb = db;
      // Physiological glottal spectral tilt guard above 2.8 kHz
      if (fh > 2800.0) {
        const tiltCap = lowRefDb - 3.5 - 8.0 * Math.log2(fh / 2800.0);
        if (db > tiltCap) db = tiltCap;
      }
      const amp = Math.pow(10.0, db / 20.0);
      hAmpCurr[h - 1] = amp;
      sumSq += amp * amp;
    }

    // Normalize harmonic weights so total RMS equals exact score envelope g(t)!
    const normScale = Math.SQRT2 / Math.sqrt(sumSq);
    for (let h = 0; h < nH; h++) {
      hAmpCurr[h] *= normScale;
    }

    const bg0 = backGain[i] * vDeg[i];
    const bg1 = backGain[i + 1] * vDeg[i + 1];
    const hasBack = bg0 > 1e-5 || bg1 > 1e-5;
    const nHB = Math.min(nH, 18);
    if (hasBack) {
      for (let h = 0; h < nHB; h++) wBack[h] = hAmpCurr[h] * bTilt[h];
    }
    const invLen = 1.0 / (s1 - s0);
    const hChestEnd = Math.min(nH, Math.floor(560.0 / fMid));
    const hXfadeEnd = Math.min(nH, Math.floor(720.0 / fMid));

    for (let s = s0; s < s1; s++) {
      const a = (s - s0) * invLen;
      const fInst = fStart + (fEnd - fStart) * a;
      const gInst = g0 + (g1 - g0) * a;
      glottalPhase += (TWO_PI * fInst) / sr;
      if (glottalPhase > TWO_PI) glottalPhase -= TWO_PI;

      // Exact Chebyshev / complex phasor recurrence: cos(h*theta + phi_h) = Re(e^{i*h*theta} * e^{i*phi_h})
      const c1 = Math.cos(glottalPhase);
      const s1p = Math.sin(glottalPhase);
      let ch = 1.0;
      let sh = 0.0;
      let vChest = 0.0;
      let vBody = 0.0;

      for (let h = 0; h < nH; h++) {
        const nextC = ch * c1 - sh * s1p;
        const nextS = sh * c1 + ch * s1p;
        ch = nextC;
        sh = nextS;
        const wH = hAmpPrev[h] + (hAmpCurr[h] - hAmpPrev[h]) * a;
        const sig = wH * (ch * cosDisp[h] - sh * sinDisp[h]);
        const h1 = h + 1;
        if (h1 <= hChestEnd) {
          vChest += sig;
        } else if (h1 <= hXfadeEnd) {
          const xFade = (720.0 - h1 * fInst) * (1.0 / 160.0);
          const xc = xFade < 0.0 ? 0.0 : (xFade > 1.0 ? 1.0 : xFade);
          vChest += sig * xc;
          vBody += sig * (1.0 - xc);
        } else {
          vBody += sig;
        }
      }
      chestL[s] = vChest * gInst;
      bodyL[s] = vBody * gInst;

      if (hasBack) {
        const bgInst = (bg0 + (bg1 - bg0) * a) * 0.58;
        backPhaseL += (TWO_PI * fInst * 1.00376) / sr;
        backPhaseR += (TWO_PI * fInst * 0.99625) / sr;
        if (backPhaseL > TWO_PI) backPhaseL -= TWO_PI;
        if (backPhaseR > TWO_PI) backPhaseR -= TWO_PI;
        const c1L = Math.cos(backPhaseL);
        const s1L = Math.sin(backPhaseL);
        const c1R = Math.cos(backPhaseR);
        const s1R = Math.sin(backPhaseR);
        let chL = 1.0, shL = 0.0, chR = 1.0, shR = 0.0;
        let bL = 0.0, bR = 0.0;
        for (let h = 0; h < nHB; h++) {
          const ncL = chL * c1L - shL * s1L;
          const nsL = shL * c1L + chL * s1L;
          chL = ncL; shL = nsL;
          const ncR = chR * c1R - shR * s1R;
          const nsR = shR * c1R + chR * s1R;
          chR = ncR; shR = nsR;
          const wH = wBack[h];
          bL += wH * (chL * cosBackL[h] - shL * sinBackL[h]);
          bR += wH * (chR * cosBackR[h] - shR * sinBackR[h]);
        }
        harmBackL[s] = bL * bgInst;
        harmBackR[s] = bR * bgInst;
      }
    }
    hAmpPrev.set(hAmpCurr);
  }

  // Pass 4: Analytical Dual-Gaussian Consonant Burst/Sibilant + Klatt Breath Synthesis (Track 03)
  const consAir = new Float32Array(nSamples);
  const nFft = 512;
  const nBins = (nFft >> 1) + 1;
  const ifft = new ComplexInverseFFT(nFft);
  const rng = new DeterministicRng(0x517CC1B7);
  const win = new Float32Array(nFft);
  for (let n = 0; n < nFft; n++) {
    win[n] = Math.sin((Math.PI * (n + 0.5)) / nFft); // Princen-Bradley sine window (sum of squares = 1 at hop=256!)
  }

  const reSpec = new Float64Array(nBins);
  const imSpec = new Float64Array(nBins);
  const rawMag = new Float64Array(nBins);
  const frameBuf = new Float64Array(nFft);
  const binFreqs = new Float64Array(nBins);
  const binLog2 = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {
    const fk = Math.max(60.0, (k * sr) / nFft);
    binFreqs[k] = fk;
    binLog2[k] = Math.log2(fk);
  }

  for (let i = 0; i < nFrames - 1; i++) {
    if (!activeMask[i]) continue;
    const unvFrac = Math.max(0.0, 1.0 - vDeg[i]);
    const cSpec = consTypeIdx[i] >= 0 ? MANDARIN_CONSONANTS[consKeys[consTypeIdx[i]]] : MANDARIN_CONSONANTS[''];
    const cAmp = consGain[i] * (unvFrac * 0.72 + 0.042 * vDeg[i]);
    if (cAmp < 1e-6) continue;

    const f1Log = Math.log2(Math.max(120.0, cSpec[2]));
    const bw1 = Math.max(0.35, cSpec[3]);
    const g1 = cSpec[4];
    const f2Log = Math.log2(Math.max(250.0, cSpec[5]));
    const bw2 = Math.max(0.35, cSpec[6]);
    const g2 = cSpec[7];
    const breathMix = vDeg[i] * 0.052 * gainHarm[i];

    reSpec[0] = 0.0; imSpec[0] = 0.0;
    rawMag.fill(0.0);
    let magSq = 1e-12;
    for (let k = 1; k < nBins; k++) {
      const fk = binFreqs[k];
      if (fk < 130.0 || fk > 17500.0) continue;
      const lf = binLog2[k];
      const d1 = (lf - f1Log) / bw1;
      const d2 = (lf - f2Log) / bw2;
      // Dual-Gaussian analytical consonant turbulence spectrum + warm velvet breath
      let env = cAmp * (g1 * Math.exp(-0.5 * d1 * d1) + g2 * Math.exp(-0.5 * d2 * d2));
      if (breathMix > 0.0 && fk > 900.0 && fk < 8500.0) {
        env += breathMix * Math.exp(-0.5 * Math.pow((lf - 11.3) / 1.4, 2.0));
      }
      rawMag[k] = env;
      magSq += env * env;
    }

    // Exact Parseval scaling for 1/N inverse DFT so time-domain RMS matches analytical cAmp + breathMix!
    const scale = ((cAmp + breathMix) * nFft * 0.72) / Math.sqrt(magSq * 2.0);
    for (let k = 1; k < nBins - 1; k++) {
      const m = rawMag[k] * scale;
      reSpec[k] = m * rng.gaussian();
      imSpec[k] = m * rng.gaussian();
    }
    reSpec[nBins - 1] = 0.0;
    imSpec[nBins - 1] = 0.0;

    ifft.inverseReal(reSpec, imSpec, frameBuf);
    const sBase = i * hop - (nFft >> 1);
    const plosiveTransient = plosiveFlag[i] ? 1.22 : 1.0;
    for (let n = 0; n < nFft; n++) {
      const dst = sBase + n;
      if (dst >= 0 && dst < nSamples) {
        consAir[dst] += frameBuf[n] * win[n] * plosiveTransient;
      }
    }
  }
  // Smooth studio de-esser saturation on extreme sibilant spikes while keeping linear breath/turbulence
  for (let s = 0; s < nSamples; s++) {
    const x = consAir[s];
    if (x > 0.06 || x < -0.06) {
      consAir[s] = 0.22 * Math.tanh(x / 0.22);
    }
  }

  // Apply subtle stereo chorus delay (14ms L / 23ms R) on Track 04 Chorus Harmony
  const dL = Math.round(0.014 * sr);
  const dR = Math.round(0.023 * sr);
  const backOutL = new Float32Array(nSamples);
  const backOutR = new Float32Array(nSamples);
  for (let s = dR; s < nSamples; s++) {
    backOutL[s] = harmBackL[s - dL];
    backOutR[s] = harmBackR[s - dR];
  }

  return {
    vocal_chest_f0: [chestL, new Float32Array(chestL)],
    vocal_formant_body: [bodyL, new Float32Array(bodyL)],
    vocal_consonant_air: [consAir, new Float32Array(consAir)],
    vocal_chorus_harmony: [backOutL, backOutR],
  };
}
