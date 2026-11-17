// Sine-Lapped Mid/Side Instrument Timbre Atom Dictionary Synthesizer (`src/dsp/lapped-atom-synth.js`)
// Pure mathematical overlap-add over instrument-specific orthonormal atom dictionaries (w^2[n] + w^2[n+hop] === 1).

const DECODED_DICT_CACHE = new WeakMap();
const SINE_WIN_CACHE = new Map();

export function getSineWindow(winLen) {
  if (SINE_WIN_CACHE.has(winLen)) return SINE_WIN_CACHE.get(winLen);
  const w = new Float32Array(winLen);
  for (let n = 0; n < winLen; n++) {
    w[n] = Math.sin((Math.PI * (n + 0.5)) / winLen);
  }
  SINE_WIN_CACHE.set(winLen, w);
  return w;
}

export function getDecodedAtomTables(dictSpec) {
  if (DECODED_DICT_CACHE.has(dictSpec)) return DECODED_DICT_CACHE.get(dictSpec);
  const { winLen, basisScale, midAtoms, sideAtoms = [] } = dictSpec;
  const invBasis = 1.0 / basisScale;

  const kMid = midAtoms.length;
  const midFlat = new Float32Array(kMid * winLen);
  for (let k = 0; k < kMid; k++) {
    const row = midAtoms[k];
    const off = k * winLen;
    for (let n = 0; n < winLen; n++) midFlat[off + n] = row[n] * invBasis;
  }

  const kSide = sideAtoms.length;
  const sideFlat = new Float32Array(kSide * winLen);
  for (let k = 0; k < kSide; k++) {
    const row = sideAtoms[k];
    const off = k * winLen;
    for (let n = 0; n < winLen; n++) sideFlat[off + n] = row[n] * invBasis;
  }

  const tables = { winLen, kMid, kSide, midFlat, sideFlat };
  DECODED_DICT_CACHE.set(dictSpec, tables);
  return tables;
}

function accumulateSparseFrames(outBuf, frames, atomFlat, winLen, hop, sineWin, invRms, invWeight) {
  if (!frames || frames.length === 0 || atomFlat.length === 0) return;
  const frameBuf = new Float32Array(winLen);
  let frameIdx = 0;
  const bufLen = outBuf.length;

  for (let i = 0; i < frames.length; i++) {
    const tup = frames[i];
    frameIdx += tup[0];
    const rms = tup[1] * invRms;
    if (rms <= 0) continue;

    frameBuf.fill(0.0);
    let atomIdx = 0;
    for (let j = 2; j < tup.length; j += 2) {
      atomIdx += tup[j];
      const coeff = tup[j + 1] * invWeight * rms;
      const off = atomIdx * winLen;
      for (let n = 0; n < winLen; n++) {
        frameBuf[n] += atomFlat[off + n] * coeff;
      }
    }

    const st = frameIdx * hop;
    const endN = Math.min(winLen, bufLen - st);
    if (st >= 0 && endN > 0) {
      for (let n = 0; n < endN; n++) {
        outBuf[st + n] += frameBuf[n] * sineWin[n];
      }
    }
  }
}

export function synthesizeLappedStereoEvents(dictSpec, eventStream) {
  const { winLen, hop } = dictSpec;
  const { totalSamples, pad = winLen, rmsScale = 4096, weightScale = 256, midFrames, sideFrames } = eventStream;
  const paddedLen = totalSamples + 2 * pad + winLen;
  const midPadded = new Float32Array(paddedLen);
  const sidePadded = new Float32Array(paddedLen);

  const { midFlat, sideFlat } = getDecodedAtomTables(dictSpec);
  const sineWin = getSineWindow(winLen);
  const invRms = 1.0 / rmsScale;
  const invWeight = 1.0 / weightScale;

  accumulateSparseFrames(midPadded, midFrames, midFlat, winLen, hop, sineWin, invRms, invWeight);
  accumulateSparseFrames(sidePadded, sideFrames, sideFlat, winLen, hop, sineWin, invRms, invWeight);

  const outL = new Float32Array(totalSamples);
  const outR = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    const m = midPadded[pad + i];
    const s = sidePadded[pad + i];
    outL[i] = m + s;
    outR[i] = m - s;
  }
  return [outL, outR];
}
