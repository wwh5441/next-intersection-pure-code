// V9R1 Pure-Code Bar-Motif Dictionary & Sparse Residual Parameter Expander (`motif-expander.js`)
// Expands 1-Bar Medoid Motifs (`BAND_MOTIF_ATLAS`) + 86-Bar Arrangement + Optional Sparse Residuals
// into continuous full-song parameter trajectories BEFORE phase-continuous ODE integration!

import {
  decodePwlTimeline,
  decodeSignedPwlTimeline,
  decodeRleTimeline,
} from './fft-minphase.js';

function applyGainScaleInto(dst, offset, src, len, gq, gate) {
  if (gq <= 0 || !src) {
    dst.fill(0.0, offset, offset + len);
    return;
  }
  const dq = 64.0 * Math.log10(Math.max(0.1, gq / 64.0));
  const halfGate = gate * 0.5;
  for (let i = 0; i < len; i++) {
    const v = src[i];
    if (v > halfGate) {
      const s = v + dq;
      dst[offset + i] = s <= 0.0 ? 0.0 : (s >= 255.0 ? 255.0 : s);
    } else {
      dst[offset + i] = 0.0;
    }
  }
}

function addClampedDelta(dst, offset, delta, len, minVal = 0.0, maxVal = 255.0) {
  for (let i = 0; i < len; i++) {
    const s = dst[offset + i] + delta[i];
    dst[offset + i] = s <= minVal ? minVal : (s >= maxVal ? maxVal : s);
  }
}

function holdF0AcrossSilence(f0Arr, rmsArr, gate = 10.0) {
  const n = f0Arr.length;
  let firstVoiced = -1;
  for (let i = 0; i < n; i++) {
    if (rmsArr[i] > gate && f0Arr[i] > 0.0) {
      firstVoiced = i;
      break;
    }
  }
  if (firstVoiced < 0) return;
  const initF0 = f0Arr[firstVoiced];
  for (let i = 0; i < firstVoiced; i++) f0Arr[i] = initF0;
  let lastF0 = initF0;
  for (let i = firstVoiced; i < n; i++) {
    if (rmsArr[i] > gate && f0Arr[i] > 0.0) {
      lastF0 = f0Arr[i];
    } else {
      f0Arr[i] = lastF0;
    }
  }
}

export function expandBandArrangementToTimelines(motifAtlas, arrangement, nNotes = 56) {
  const nBars = motifAtlas.totalBars;
  const totF = motifAtlas.totalFrames;
  const dCfg = motifAtlas.drums;
  const bCfg = motifAtlas.bass;
  const hCfg = motifAtlas.harmony;

  // 1. Pre-decode 1-Bar Motif Dictionaries once (`18` Drum + `22` Bass + `22` Harmony)
  const dBarF = dCfg.barFrames; // 245
  const drumDec = dCfg.motifs.map((m) => ({
    kick: decodePwlTimeline(m.kick, dBarF, dCfg.div, dCfg.gate),
    body: decodePwlTimeline(m.body, dBarF, dCfg.div, dCfg.gate),
    high: decodePwlTimeline(m.high, dBarF, dCfg.div, dCfg.gate),
    shape: decodeRleTimeline(m.shape, dBarF, dCfg.shapeStride),
  }));

  const bBarMax = bCfg.barFramesOdd; // 123
  const bassDec = bCfg.motifs.map((m) => ({
    rms: decodePwlTimeline(m.rms, bBarMax, bCfg.rmsDiv, bCfg.rmsGate),
    f0: decodePwlTimeline(m.f0, bBarMax, bCfg.f0Div, 0.0),
    nr: decodePwlTimeline(m.nr, bBarMax, bCfg.nrDiv, bCfg.nrGate),
    tone: decodeRleTimeline(m.tone, bBarMax, bCfg.toneStride || 2),
  }));

  const hBarF = hCfg.barFrames; // 98
  const sBarF = hCfg.strumFrames; // 196
  const harmDec = hCfg.motifs.map((m) => {
    const notesMap = new Map();
    if (m.notes) {
      for (const [kStr, pwl] of Object.entries(m.notes)) {
        notesMap.set(Number(kStr), decodePwlTimeline(pwl, hBarF, hCfg.noteDiv, hCfg.noteGate));
      }
    }
    return {
      harm: decodePwlTimeline(m.harm, hBarF, hCfg.envDiv, hCfg.envGate),
      bright: decodePwlTimeline(m.bright, hBarF, 4.0, 0.0),
      side: decodePwlTimeline(m.side, hBarF, 4.0, 0.0),
      strum: m.strum.map((sp) => decodePwlTimeline(sp, sBarF, hCfg.strumDiv, hCfg.strumGate)),
      notesMap,
    };
  });

  // 2. Allocate padded bar-commensurable full-song trajectories
  const padD = nBars * dBarF;
  const kickFull = new Float32Array(padD);
  const bodyFull = new Float32Array(padD);
  const highFull = new Float32Array(padD);
  const shapeFull = new Int32Array(padD);

  const padB = (nBars >> 1) * 245 + (nBars & 1) * bCfg.barFramesEven;
  const bassRmsFull = new Float32Array(padB);
  const bassF0Full = new Float32Array(padB);
  const bassNrFull = new Float32Array(padB);
  const bassToneFull = new Int32Array(padB);

  const padH = nBars * hBarF;
  const padS = nBars * sBarF;
  const harmFull = new Float32Array(padH);
  const brightFull = new Float32Array(padH);
  const sideFull = new Float32Array(padH);
  const strumFull = Array.from({ length: 4 }, () => new Float32Array(padS));
  const noteFull = Array.from({ length: nNotes }, () => new Float32Array(padH));

  const bars = arrangement.bars;
  const residuals = arrangement.residuals || {};

  // 3. Schedule all 86 bars + overlay sparse variation residuals
  for (let b = 0; b < nBars; b++) {
    const row = bars[b] || [0, 0, 0, 0, 0, 0];
    const [md, gd, mb, gb, mh, gh] = row;
    const barRes = residuals[String(b)] || null;

    // --- Drums (`245` frames/bar) ---
    const dOff = b * dBarF;
    if (md > 0 && md <= drumDec.length) {
      const dm = drumDec[md - 1];
      applyGainScaleInto(kickFull, dOff, dm.kick, dBarF, gd, dCfg.gate);
      applyGainScaleInto(bodyFull, dOff, dm.body, dBarF, gd, dCfg.gate);
      applyGainScaleInto(highFull, dOff, dm.high, dBarF, gd, dCfg.gate);
      shapeFull.set(dm.shape, dOff);
    }
    if (barRes && barRes.d) {
      const [dk, dbd, dhi] = barRes.d;
      if (dk && dk.length) {
        addClampedDelta(kickFull, dOff, decodeSignedPwlTimeline(dk, dBarF, dCfg.resDiv), dBarF, 0.0, 255.0);
      }
      if (dbd && dbd.length) {
        addClampedDelta(bodyFull, dOff, decodeSignedPwlTimeline(dbd, dBarF, dCfg.resDiv), dBarF, 0.0, 255.0);
      }
      if (dhi && dhi.length) {
        addClampedDelta(highFull, dOff, decodeSignedPwlTimeline(dhi, dBarF, dCfg.resDiv), dBarF, 0.0, 255.0);
      }
    }

    // --- Bass (`122` even / `123` odd frames = exact `245` per 2 bars) ---
    const bOff = (b >> 1) * 245 + ((b & 1) === 0 ? 0 : bCfg.barFramesEven);
    const bLen = (b & 1) === 0 ? bCfg.barFramesEven : bCfg.barFramesOdd;
    if (mb > 0 && mb <= bassDec.length) {
      const bm = bassDec[mb - 1];
      applyGainScaleInto(bassRmsFull, bOff, bm.rms, bLen, gb, bCfg.rmsGate);
      for (let i = 0; i < bLen; i++) {
        bassF0Full[bOff + i] = bm.f0[i];
        bassNrFull[bOff + i] = bm.nr[i];
        bassToneFull[bOff + i] = bm.tone[i];
      }
    }
    if (barRes && barRes.b) {
      const [dbRms, dbF0] = barRes.b;
      if (dbRms && dbRms.length) {
        addClampedDelta(bassRmsFull, bOff, decodeSignedPwlTimeline(dbRms, bBarMax, bCfg.resRmsDiv), bLen, 0.0, 255.0);
      }
      if (dbF0 && dbF0.length) {
        const df = decodeSignedPwlTimeline(dbF0, bBarMax, bCfg.resF0Div);
        for (let i = 0; i < bLen; i++) {
          if (bassRmsFull[bOff + i] > 10.0) {
            bassF0Full[bOff + i] = Math.max(0.0, bassF0Full[bOff + i] + df[i]);
          }
        }
      }
    }

    // --- Harmony: Guitar + Keys (`98` chord frames + `196` strum frames) ---
    const hOff = b * hBarF;
    const sOff = b * sBarF;
    if (mh > 0 && mh <= harmDec.length) {
      const hm = harmDec[mh - 1];
      applyGainScaleInto(harmFull, hOff, hm.harm, hBarF, gh, hCfg.envGate);
      for (let i = 0; i < hBarF; i++) {
        brightFull[hOff + i] = hm.bright[i];
        sideFull[hOff + i] = hm.side[i];
      }
      for (let sb = 0; sb < 4; sb++) {
        applyGainScaleInto(strumFull[sb], sOff, hm.strum[sb], sBarF, gh, hCfg.strumGate);
      }
      for (const [nIdx, nArr] of hm.notesMap.entries()) {
        if (nIdx >= 0 && nIdx < nNotes) {
          noteFull[nIdx].set(nArr, hOff);
        }
      }
    }
    if (barRes && barRes.h) {
      const [dhEnv, dnMap, dsList] = barRes.h;
      if (dhEnv && dhEnv.length) {
        addClampedDelta(harmFull, hOff, decodeSignedPwlTimeline(dhEnv, hBarF, hCfg.resDiv), hBarF, 0.0, 255.0);
      }
      if (dnMap) {
        for (const [nStr, dnPwl] of Object.entries(dnMap)) {
          const nIdx = Number(nStr);
          if (nIdx >= 0 && nIdx < nNotes && dnPwl && dnPwl.length) {
            addClampedDelta(noteFull[nIdx], hOff, decodeSignedPwlTimeline(dnPwl, hBarF, hCfg.resDiv), hBarF, 0.0, 127.0);
          }
        }
      }
      if (Array.isArray(dsList) && dsList.length === 4) {
        for (let sb = 0; sb < 4; sb++) {
          if (dsList[sb] && dsList[sb].length) {
            addClampedDelta(strumFull[sb], sOff, decodeSignedPwlTimeline(dsList[sb], sBarF, hCfg.strumDiv), sBarF, 0.0, 255.0);
          }
        }
      }
    }
  }

  holdF0AcrossSilence(bassF0Full, bassRmsFull, 10.0);

  return {
    drums: {
      nFrames: totF.drums,
      kickQ: kickFull.subarray(0, totF.drums),
      bodyQ: bodyFull.subarray(0, totF.drums),
      highQ: highFull.subarray(0, totF.drums),
      shapeIdx: shapeFull.subarray(0, totF.drums),
    },
    bass: {
      nFrames: totF.bass,
      rmsQ: bassRmsFull.subarray(0, totF.bass),
      f0Q: bassF0Full.subarray(0, totF.bass),
      nrQ: bassNrFull.subarray(0, totF.bass),
      toneIdx: bassToneFull.subarray(0, totF.bass),
    },
    harmony: {
      nFrames: totF.harmony,
      nSFrames: totF.strum,
      qLevels: hCfg.qLevels,
      maxNoteAmp: hCfg.maxNoteAmp,
      harmQ: harmFull.subarray(0, totF.harmony),
      brightQ: brightFull.subarray(0, totF.harmony),
      sideQ: sideFull.subarray(0, totF.harmony),
      strumCols: strumFull.map((col) => col.subarray(0, totF.strum)),
      noteCols: noteFull.map((col) => col.subarray(0, totF.harmony)),
    },
  };
}
