// Pure Mathematical 51-Bit (3x17-Bit) Safe-Integer + 4-Class Canonical Huffman + J-Orthogonal Stereo + SBR Decoder
// Zero LZMA · Zero Base122 · Zero Base64 · Zero Opaque Strings · 100% Plain-Text Integer Algebra

import { LONG_BINS, SHORT_BINS, MODE_FOUR_SHORT } from './radix2-fft.js';

const LUT_BITS = 14;
const LUT_SIZE = 1 << LUT_BITS; // 16384
const LUT_MASK = LUT_SIZE - 1;
const INV_2P17 = 1.0 / 131072.0;      // Exact 2^-17 in IEEE-754 float64
const INV_2P34 = 1.0 / 17179869184.0; // Exact 2^-34 in IEEE-754 float64
const POW43_MAX = 4096;

export function buildCanonicalHuffmanLut14(flatSymLenPairs) {
  const lut = new Int32Array(LUT_SIZE);
  const nPairs = flatSymLenPairs.length >> 1;
  if (nPairs === 0) return lut;

  let code = 0;
  let prevLen = flatSymLenPairs[1];
  for (let i = 0; i < nPairs; i++) {
    const sym = flatSymLenPairs[2 * i];
    const len = flatSymLenPairs[2 * i + 1];
    code <<= len - prevLen;
    prevLen = len;

    let rev = 0;
    for (let b = 0; b < len; b++) {
      rev = (rev << 1) | ((code >> b) & 1);
    }
    const packed = (sym << 4) | len;
    const step = 1 << len;
    for (let idx = rev; idx < LUT_SIZE; idx += step) {
      lut[idx] = packed;
    }
    code++;
  }
  return lut;
}

export function createTierAlgebraicDecoder(tierPhysics, huffmanTables) {
  const stepMinDb = tierPhysics.stepMinDb;
  const stepQuantum = tierPhysics.stepDbQuantum || 0.5;
  const stepGroup = tierPhysics.stepGroup || 1;
  const hasDirectSide = Boolean(tierPhysics.hasDirectSide);
  const slicesLong = tierPhysics.erbSlicesLong;
  const slicesShort = tierPhysics.erbSlicesShort;
  const ildPatches = tierPhysics.ildPatches || [];
  const sbrPatches = tierPhysics.sbrPatches || [];

  const stepLut = new Float64Array(256);
  stepLut[0] = 0.0;
  for (let u = 1; u < 256; u++) {
    stepLut[u] = Math.pow(10.0, (stepQuantum * u + stepMinDb) / 20.0);
  }

  const sbrRmsLut = new Float64Array(64);
  sbrRmsLut[0] = 0.0;
  for (let u = 1; u < 64; u++) {
    sbrRmsLut[u] = Math.pow(10.0, (1.5 * u - 85.0) / 20.0);
  }

  const pow43Lut = new Float64Array(POW43_MAX);
  for (let q = 0; q < POW43_MAX; q++) {
    pow43Lut[q] = Math.pow(q, 4.0 / 3.0);
  }

  const lutHdr = buildCanonicalHuffmanLut14(huffmanTables.H_HDR);
  const lutC1 = buildCanonicalHuffmanLut14(huffmanTables.H_C1);
  const lutC2 = buildCanonicalHuffmanLut14(huffmanTables.H_C2);
  const lutC3 = buildCanonicalHuffmanLut14(huffmanTables.H_C3);

  const chunks = new Uint32Array(8192);

  function decodeFrame(frameIndex, frameEntry, outMid2048, outSide2048) {
    outMid2048.fill(0.0);
    outSide2048.fill(0.0);
    if (!frameEntry || frameEntry === 0) {
      return { modeMid: 0, modeSide: 0, hasMid: false, hasSide: false };
    }

    let nChunks = 0;
    const nWords = frameEntry.length;
    for (let i = 0; i < nWords; i++) {
      const w = frameEntry[i];
      chunks[nChunks++] = w & 0x1ffff;
      chunks[nChunks++] = ((w * INV_2P17) | 0) & 0x1ffff;
      chunks[nChunks++] = (w * INV_2P34) | 0;
    }
    chunks[nChunks] = 0;
    chunks[nChunks + 1] = 0;
    chunks[nChunks + 2] = 0;

    let cPos = 0;
    let bitBuf = 0;
    let bitsInBuf = 0;

    const readHuff = (lut) => {
      if (bitsInBuf <= 15) {
        bitBuf = (bitBuf | ((chunks[cPos++] << bitsInBuf) >>> 0)) >>> 0;
        bitsInBuf += 17;
      }
      const entry = lut[bitBuf & LUT_MASK];
      const len = entry & 15;
      bitBuf >>>= len;
      bitsInBuf -= len;
      return entry >> 4;
    };

    const readBits = (n) => {
      if (n === 0) return 0;
      if (bitsInBuf <= 15) {
        bitBuf = (bitBuf | ((chunks[cPos++] << bitsInBuf) >>> 0)) >>> 0;
        bitsInBuf += 17;
      }
      const val = bitBuf & ((1 << n) - 1);
      bitBuf >>>= n;
      bitsInBuf -= n;
      return val;
    };

    const decodeChannelCoeffs = (outCoeffs) => {
      const hdr0 = readHuff(lutHdr);
      if (hdr0 === 0) {
        return { mode: 0, active: false };
      }
      const mode = hdr0 - 1;
      const firstBand = readHuff(lutHdr) - 20;
      const lastBand = firstBand + (readHuff(lutHdr) - 60);
      const slices = mode === MODE_FOUR_SHORT ? slicesShort : slicesLong;

      let curSc = 72;
      let groupRem = 0;

      for (let b = firstBand; b <= lastBand; b++) {
        const clsTok = readHuff(lutHdr);
        if (clsTok === 10) {
          continue;
        }
        const bCls = clsTok - 10;
        if (groupRem <= 0) {
          const diff = readHuff(lutHdr) - 300;
          curSc += diff;
          groupRem = stepGroup - 1;
        } else {
          groupRem--;
        }
        const scale = stepLut[curSc];
        const [s0, s1] = slices[b];
        let pos = s0;

        if (bCls === 1) {
          while (pos < s1) {
            const tok = readHuff(lutC1);
            if (tok === 16) break;
            if (tok > 16) {
              pos += (tok - 16) << 2;
              continue;
            }
            for (let sh = 3; sh >= 0; sh--) {
              if ((tok >> sh) & 1) {
                const sign = readBits(1);
                if (pos < s1) outCoeffs[pos] = sign ? -scale : scale;
              }
              pos++;
            }
          }
        } else if (bCls === 2) {
          while (pos < s1) {
            const tok = readHuff(lutC2);
            if (tok === 81) break;
            if (tok > 81) {
              pos += (tok - 81) << 2;
              continue;
            }
            const a0 = (tok / 27) | 0;
            const r0 = tok - a0 * 27;
            const a1 = (r0 / 9) | 0;
            const r1 = r0 - a1 * 9;
            const a2 = (r1 / 3) | 0;
            const a3 = r1 - a2 * 3;
            const quads = [a0, a1, a2, a3];
            for (let qIdx = 0; qIdx < 4; qIdx++) {
              const a = quads[qIdx];
              if (a !== 0) {
                const sign = readBits(1);
                const mag = scale * pow43Lut[a];
                if (pos < s1) outCoeffs[pos] = sign ? -mag : mag;
              }
              pos++;
            }
          }
        } else {
          while (pos < s1) {
            const tok = readHuff(lutC3);
            if (tok === 256) break;
            if (tok > 256) {
              pos += (tok - 256) << 2;
              continue;
            }
            for (let sh = 6; sh >= 0; sh -= 2) {
              const a = (tok >> sh) & 3;
              if (a !== 0) {
                const sign = readBits(1);
                let aq = a;
                if (a === 3) {
                  const rtok = readHuff(lutHdr);
                  if (rtok < 632) {
                    aq = 3 + (rtok - 600);
                  } else {
                    const bl = rtok - 632;
                    const lowBits = bl > 1 ? readBits(bl - 1) : 0;
                    aq = 3 + ((1 << (bl - 1)) | lowBits);
                  }
                }
                const mag = scale * (aq < POW43_MAX ? pow43Lut[aq] : Math.pow(aq, 4.0 / 3.0));
                if (pos < s1) outCoeffs[pos] = sign ? -mag : mag;
              }
              pos++;
            }
          }
        }
      }

      return { mode, active: true };
    };

    const mRes = decodeChannelCoeffs(outMid2048);
    let sRes = { mode: mRes.mode, active: false };
    if (hasDirectSide) {
      sRes = decodeChannelCoeffs(outSide2048);
    }

    // Pure-Math SBR High-Frequency Harmonic Mirroring (for compact tiers 5, 6, 7)
    if (mRes.active && sbrPatches.length > 0) {
      const isShort = mRes.mode === MODE_FOUR_SHORT;
      for (let p = 0; p < sbrPatches.length; p++) {
        const uSbr = readHuff(lutHdr) - 800;
        if (uSbr <= 0) continue;
        const targetRms = sbrRmsLut[uSbr];
        const [srcLoL, srcHiL, dstLoL, dstHiL] = sbrPatches[p];

        if (!isShort) {
          const dstLen = dstHiL - dstLoL;
          const srcLen = Math.max(1, srcHiL - srcLoL);
          let sumSq = 1e-18;
          for (let i = 0; i < dstLen; i++) {
            const srcVal = outMid2048[srcLoL + (i % srcLen)];
            const trig = Math.sin((frameIndex * 131 + dstLoL + i) * 1.61803398875);
            const v = 0.78 * srcVal + 0.22 * targetRms * trig;
            outMid2048[dstLoL + i] = v;
            sumSq += v * v;
          }
          const g = targetRms / Math.sqrt(sumSq / dstLen);
          for (let i = 0; i < dstLen; i++) {
            outMid2048[dstLoL + i] *= g;
          }
        } else {
          const sLo = srcLoL >> 2;
          const sHi = Math.max(sLo + 1, srcHiL >> 2);
          const dLo = dstLoL >> 2;
          const dHi = Math.max(dLo + 1, dstHiL >> 2);
          const dstLen = dHi - dLo;
          const srcLen = sHi - sLo;
          for (let w = 0; w < 4; w++) {
            const base = w * SHORT_BINS;
            let sumSq = 1e-18;
            for (let i = 0; i < dstLen; i++) {
              const srcVal = outMid2048[base + sLo + (i % srcLen)];
              const trig = Math.sin(((frameIndex * 4 + w) * 97 + dLo + i) * 1.61803398875);
              const v = 0.78 * srcVal + 0.22 * targetRms * trig;
              outMid2048[base + dLo + i] = v;
              sumSq += v * v;
            }
            const g = targetRms / Math.sqrt(sumSq / dstLen);
            for (let i = 0; i < dstLen; i++) {
              outMid2048[base + dLo + i] *= g;
            }
          }
        }
      }
    }

    // Exact J-Orthogonal Parametric / Intensity Stereo Synthesis: Side = pan * Mid + wid * J(Mid)
    if (mRes.active && ildPatches.length > 0) {
      if (!hasDirectSide || !sRes.active) {
        sRes.mode = mRes.mode;
      }
      const isShort = mRes.mode === MODE_FOUR_SHORT;
      for (let p = 0; p < ildPatches.length; p++) {
        const ildTok = readHuff(lutHdr) - 700;
        if (ildTok === 24) continue; // pan=0 (idx 4), wid=0 (idx 0) -> 4*6 + 0 = 24
        const panIdx = (ildTok / 6) | 0;
        const widIdx = ildTok - panIdx * 6;
        const pan = (panIdx - 4) * 0.16;
        const wid = widIdx * 0.14;
        if (pan === 0.0 && wid === 0.0) continue;
        sRes.active = true;

        const [k0L, k1L] = ildPatches[p];
        if (!isShort) {
          const kEnd = k1L - ((k1L - k0L) & 1);
          for (let k = k0L; k < kEnd; k += 2) {
            const u0 = outMid2048[k];
            const u1 = outMid2048[k + 1];
            outSide2048[k] = pan * u0 - wid * u1;
            outSide2048[k + 1] = pan * u1 + wid * u0;
          }
        } else {
          const s0 = k0L >> 2;
          const s1 = k1L >> 2;
          const sEnd = s1 - ((s1 - s0) & 1);
          for (let w = 0; w < 4; w++) {
            const base = w * SHORT_BINS;
            for (let k = s0; k < sEnd; k += 2) {
              const u0 = outMid2048[base + k];
              const u1 = outMid2048[base + k + 1];
              outSide2048[base + k] = pan * u0 - wid * u1;
              outSide2048[base + k + 1] = pan * u1 + wid * u0;
            }
          }
        }
      }
    }

    return {
      modeMid: mRes.mode,
      modeSide: sRes.mode,
      hasMid: mRes.active,
      hasSide: sRes.active,
    };
  }

  return {
    decodeFrame,
  };
}
