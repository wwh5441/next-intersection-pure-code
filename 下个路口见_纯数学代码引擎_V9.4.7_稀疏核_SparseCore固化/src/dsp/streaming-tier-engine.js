// Universal TDAC 2-Frame Locality Streaming Synthesis Engine & 48kHz/24-Bit WAV Bounce
// ~0.38 MB Live PCM RAM · Instant O(1) Random-Access Seek · Zero External Dependencies

import { LONG_BINS, createOrthonormalTdacKernel } from './radix2-fft.js';
import { createTierAlgebraicDecoder } from './algebraic-bit-decoder.js';

export function createStreamingTierEngine(tierPhysics, huffmanTables, tierFrames) {
  const sr = tierPhysics.sampleRate;
  const totalSamples = tierPhysics.totalSamples;
  const nFrames = tierPhysics.nFrames;
  const baseGainScale = tierPhysics.masterGainScale || 1.0;
  const useSoftCeiling = Boolean(tierPhysics.softCeiling);
  const postLowpassHz = Number(tierPhysics.postLowpassHz || 0);

  // C^1-continuous unit-slope algebraic ceiling at 0.82 -> asymptote 0.95
  function softClip(x) {
    const ax = x < 0.0 ? -x : x;
    if (ax <= 0.82) return x;
    const over = ax - 0.82;
    const y = 0.82 + (0.13 * over) / (0.13 + over);
    return x < 0.0 ? -y : y;
  }

  // Exact 4th-Order Maximally-Flat Butterworth Continuous-Time Anti-Sidelobe Low-Pass (-24 dB/oct)
  function buildBiquadCoeffs(fc, qVal) {
    const k = Math.tan((Math.PI * fc) / sr);
    const k2 = k * k;
    const norm = 1.0 / (1.0 + k / qVal + k2);
    const b0 = k2 * norm;
    return {
      b0,
      b1: 2.0 * b0,
      b2: b0,
      a1: 2.0 * (k2 - 1.0) * norm,
      a2: (1.0 - k / qVal + k2) * norm,
    };
  }

  const lpStage1 = postLowpassHz > 0 ? buildBiquadCoeffs(postLowpassHz, 0.5411961) : null;
  const lpStage2 = postLowpassHz > 0 ? buildBiquadCoeffs(postLowpassHz, 1.306563) : null;
  let lpExpectedSample = -999999;
  const lpState = new Float64Array(16); // [L1: x1,x2,y1,y2, L2: x1,x2,y1,y2, R1: ..., R2: ...]

  function applyButterworth4InPlace(bufL, bufR, count, startSample) {
    if (!lpStage1 || count <= 0) return;
    if (startSample !== lpExpectedSample) {
      const l0 = bufL[0];
      const r0 = bufR[0];
      for (let j = 0; j < 8; j++) lpState[j] = l0;
      for (let j = 8; j < 16; j++) lpState[j] = r0;
    }
    lpExpectedSample = startSample + count;

    const { b0: b0a, b1: b1a, b2: b2a, a1: a1a, a2: a2a } = lpStage1;
    const { b0: b0b, b1: b1b, b2: b2b, a1: a1b, a2: a2b } = lpStage2;

    let lx1a = lpState[0], lx2a = lpState[1], ly1a = lpState[2], ly2a = lpState[3];
    let lx1b = lpState[4], lx2b = lpState[5], ly1b = lpState[6], ly2b = lpState[7];
    let rx1a = lpState[8], rx2a = lpState[9], ry1a = lpState[10], ry2a = lpState[11];
    let rx1b = lpState[12], rx2b = lpState[13], ry1b = lpState[14], ry2b = lpState[15];

    for (let i = 0; i < count; i++) {
      const xl = bufL[i];
      const ula = b0a * xl + b1a * lx1a + b2a * lx2a - a1a * ly1a - a2a * ly2a;
      lx2a = lx1a; lx1a = xl; ly2a = ly1a; ly1a = ula;
      const ulb = b0b * ula + b1b * lx1b + b2b * lx2b - a1b * ly1b - a2b * ly2b;
      lx2b = lx1b; lx1b = ula; ly2b = ly1b; ly1b = ulb;
      bufL[i] = ulb;

      const xr = bufR[i];
      const ura = b0a * xr + b1a * rx1a + b2a * rx2a - a1a * ry1a - a2a * ry2a;
      rx2a = rx1a; rx1a = xr; ry2a = ry1a; ry1a = ura;
      const urb = b0b * ura + b1b * rx1b + b2b * rx2b - a1b * ry1b - a2b * ry2b;
      rx2b = rx1b; rx1b = ura; ry2b = ry1b; ry1b = urb;
      bufR[i] = urb;
    }

    lpState[0] = lx1a; lpState[1] = lx2a; lpState[2] = ly1a; lpState[3] = ly2a;
    lpState[4] = lx1b; lpState[5] = lx2b; lpState[6] = ly1b; lpState[7] = ly2b;
    lpState[8] = rx1a; lpState[9] = rx2a; lpState[10] = ry1a; lpState[11] = ry2a;
    lpState[12] = rx1b; lpState[13] = rx2b; lpState[14] = ry1b; lpState[15] = ry2b;
  }

  const kernel = createOrthonormalTdacKernel();
  const decoder = createTierAlgebraicDecoder(tierPhysics, huffmanTables);

  const cMid = new Float64Array(LONG_BINS);
  const cSide = new Float64Array(LONG_BINS);

  let cachedFrameIdx = -999;
  const cachedWinMid = new Float64Array(2 * LONG_BINS);
  const cachedWinSide = new Float64Array(2 * LONG_BINS);
  const nextWinMid = new Float64Array(2 * LONG_BINS);
  const nextWinSide = new Float64Array(2 * LONG_BINS);

  function synthesizeSingleFrameWindow(t, outWinMid4096, outWinSide4096) {
    outWinMid4096.fill(0.0);
    outWinSide4096.fill(0.0);
    if (t < 0 || t >= nFrames) return;

    const entry = tierFrames[t];
    if (!entry || entry === 0) return;

    const info = decoder.decodeFrame(t, entry, cMid, cSide);
    if (info.hasMid) {
      kernel.synthesizeFrameOverlapAdd(cMid, info.modeMid, outWinMid4096, 0);
    }
    if (info.hasSide) {
      kernel.synthesizeFrameOverlapAdd(cSide, info.modeSide, outWinSide4096, 0);
    }
  }

  function synthesizeSampleRange(startSample, length, mixParams, outL, outR) {
    const endSample = Math.min(totalSamples, startSample + length);
    const masterGain = (mixParams?.masterGain ?? 1.0) * baseGainScale;
    const stereoWidth = mixParams?.stereoWidth ?? 1.0;
    const channelMode = mixParams?.channelMode || 'stereo'; // 'stereo' | 'mid' | 'side'

    let curSample = startSample;
    while (curSample < endSample) {
      const hopIdx = Math.floor(curSample / LONG_BINS);
      const inHopOffset = curSample - hopIdx * LONG_BINS;
      const copyCount = Math.min(LONG_BINS - inHopOffset, endSample - curSample);

      if (cachedFrameIdx !== hopIdx) {
        synthesizeSingleFrameWindow(hopIdx, cachedWinMid, cachedWinSide);
      }
      synthesizeSingleFrameWindow(hopIdx + 1, nextWinMid, nextWinSide);

      const dstBase = curSample - startSample;
      if (channelMode === 'mid') {
        for (let i = 0; i < copyCount; i++) {
          const rawM = (cachedWinMid[LONG_BINS + inHopOffset + i] + nextWinMid[inHopOffset + i]) * masterGain;
          const m = useSoftCeiling ? softClip(rawM) : rawM;
          outL[dstBase + i] = m;
          outR[dstBase + i] = m;
        }
      } else if (channelMode === 'side') {
        for (let i = 0; i < copyCount; i++) {
          const rawSd =
            (cachedWinSide[LONG_BINS + inHopOffset + i] + nextWinSide[inHopOffset + i]) *
            masterGain *
            stereoWidth;
          const sd = useSoftCeiling ? softClip(rawSd) : rawSd;
          outL[dstBase + i] = sd;
          outR[dstBase + i] = -sd;
        }
      } else if (useSoftCeiling) {
        for (let i = 0; i < copyCount; i++) {
          const idxRight = LONG_BINS + inHopOffset + i;
          const idxLeft = inHopOffset + i;
          const m = (cachedWinMid[idxRight] + nextWinMid[idxLeft]) * masterGain;
          const sd = (cachedWinSide[idxRight] + nextWinSide[idxLeft]) * masterGain * stereoWidth;
          outL[dstBase + i] = softClip(m + sd);
          outR[dstBase + i] = softClip(m - sd);
        }
      } else {
        for (let i = 0; i < copyCount; i++) {
          const idxRight = LONG_BINS + inHopOffset + i;
          const idxLeft = inHopOffset + i;
          const m = (cachedWinMid[idxRight] + nextWinMid[idxLeft]) * masterGain;
          const sd = (cachedWinSide[idxRight] + nextWinSide[idxLeft]) * masterGain * stereoWidth;
          outL[dstBase + i] = m + sd;
          outR[dstBase + i] = m - sd;
        }
      }

      if (inHopOffset + copyCount === LONG_BINS) {
        cachedWinMid.set(nextWinMid);
        cachedWinSide.set(nextWinSide);
        cachedFrameIdx = hopIdx + 1;
      } else {
        cachedFrameIdx = hopIdx;
      }

      curSample += copyCount;
    }

    const synthesizedCount = Math.max(0, endSample - startSample);
    if (synthesizedCount > 0 && lpStage1) {
      applyButterworth4InPlace(outL, outR, synthesizedCount, startSample);
    }

    for (let i = synthesizedCount; i < length; i++) {
      outL[i] = 0.0;
      outR[i] = 0.0;
    }
  }

  function renderFullSongStereo(mixParams = null) {
    const outL = new Float32Array(totalSamples);
    const outR = new Float32Array(totalSamples);
    cachedFrameIdx = -999;
    lpExpectedSample = -999999;
    synthesizeSampleRange(0, totalSamples, mixParams, outL, outR);
    return [outL, outR];
  }

  return {
    tierId: tierPhysics.tierId,
    versionShort: tierPhysics.versionShort,
    sampleRate: sr,
    totalSamples,
    durationSec: totalSamples / sr,
    tierPhysics,
    huffmanTables,
    tierFrames,
    synthesizeSampleRange,
    renderFullSongStereo,
  };
}

export function encodeStereoWav24Bit(leftF32, rightF32, sampleRate = 48000) {
  const numSamples = Math.min(leftF32.length, rightF32.length);
  const bytesPerSample = 3;
  const numChannels = 2;
  const dataBytes = numSamples * numChannels * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  const writeAscii = (off, str) => {
    for (let i = 0; i < str.length; i++) u8[off + i] = str.charCodeAt(i);
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, 24, true);
  writeAscii(36, 'data');
  view.setUint32(40, dataBytes, true);

  let pos = 44;
  for (let i = 0; i < numSamples; i++) {
    const cL = Math.max(-1.0, Math.min(0.99999988, leftF32[i]));
    const cR = Math.max(-1.0, Math.min(0.99999988, rightF32[i]));
    const iL = Math.round(cL * 8388607.0);
    const iR = Math.round(cR * 8388607.0);
    u8[pos] = iL & 0xff;
    u8[pos + 1] = (iL >> 8) & 0xff;
    u8[pos + 2] = (iL >> 16) & 0xff;
    u8[pos + 3] = iR & 0xff;
    u8[pos + 4] = (iR >> 8) & 0xff;
    u8[pos + 5] = (iR >> 16) & 0xff;
    pos += 6;
  }

  return new Blob([buf], { type: 'audio/wav' });
}
