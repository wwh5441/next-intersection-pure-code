const UNPACK_LONG_WIDTHS = [
  16, 16, 16, 16, 20, 20, 24, 24, 28, 32, 36, 40,
  48, 56, 64, 76, 88, 104, 124, 148, 180, 220, 288, 364,
];
const UNPACK_SHORT_WIDTHS = [8, 8, 12, 12, 16, 20, 28, 36, 48, 68, 104, 152];

const UNPACK_EDGES_L = (() => {
  const e = new Int32Array(UNPACK_LONG_WIDTHS.length + 1);
  for (let i = 0; i < UNPACK_LONG_WIDTHS.length; i++) e[i + 1] = e[i] + UNPACK_LONG_WIDTHS[i];
  return e;
})();
const UNPACK_EDGES_S = (() => {
  const e = new Int32Array(UNPACK_SHORT_WIDTHS.length + 1);
  for (let i = 0; i < UNPACK_SHORT_WIDTHS.length; i++) e[i + 1] = e[i] + UNPACK_SHORT_WIDTHS[i];
  return e;
})();

export function unpackWavePacket(raw) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magic = String.fromCharCode(...raw.subarray(0, 4));

  if (magic === 'VWP7') {
    const longN = view.getUint16(8, true);
    const shortN = view.getUint16(10, true);
    const tCount = view.getUint32(12, true);
    const bLong = view.getUint8(24);
    const bShort = view.getUint8(25);
    const presentCount = view.getUint32(28, true);
    const nonzero = view.getUint32(32, true);

    let offset = 36;
    const actMaskStart = offset;
    offset += Math.ceil(tCount / 8);

    const actFrames = [];
    for (let t = 0; t < tCount; t++) {
      if ((raw[actMaskStart + (t >> 3)] & (128 >> (t & 7))) !== 0) {
        actFrames.push(t);
      }
    }
    const nAct = actFrames.length;
    const actModes = raw.subarray(offset, offset + nAct);
    offset += nAct;

    const stride = 1 + 48 + longN;
    const output = new Uint8Array(32 + tCount * stride);
    output.set(raw.subarray(0, 28), 0);
    output.set([86, 87, 81, 54], 0); // "VWQ6"

    const prevLong = new Int32Array(bLong);
    for (let idx = 0; idx < nAct; idx++) {
      const t = actFrames[idx];
      const dst = 32 + t * stride;
      const mode = actModes[idx];
      output[dst] = mode;
      if (mode !== 2) {
        for (let g = 0; g < bLong; g++) {
          const cur = (prevLong[g] + raw[offset++]) & 0xff;
          prevLong[g] = cur;
          output[dst + 1 + g] = cur;
        }
      } else {
        for (let g = 0; g < bShort; g++) {
          output[dst + 1 + g] = raw[offset++];
        }
        for (let m = 1; m < 4; m++) {
          const prevBase = dst + 1 + (m - 1) * bShort;
          const curBase = dst + 1 + m * bShort;
          for (let g = 0; g < bShort; g++) {
            output[curBase + g] = (output[prevBase + g] + raw[offset++]) & 0xff;
          }
        }
      }
    }

    const presStart = offset;
    offset += Math.ceil(presentCount / 8);
    const zzStart = offset;
    offset += nonzero;
    if (offset !== raw.byteLength) {
      throw new Error(`Truncated VWP7 stream: ${offset} != ${raw.byteLength}`);
    }

    let pCur = 0;
    let nzCur = 0;
    for (let idx = 0; idx < nAct; idx++) {
      const t = actFrames[idx];
      const scaleDst = 32 + t * stride + 1;
      const qDst = 32 + t * stride + 49;
      const mode = actModes[idx];

      if (mode !== 2) {
        for (let g = 0; g < bLong; g++) {
          if (output[scaleDst + g] === 0) continue;
          const sK = UNPACK_EDGES_L[g];
          const eK = UNPACK_EDGES_L[g + 1];
          for (let k = sK; k < eK; k++) {
            const bit = raw[presStart + (pCur >> 3)] & (128 >> (pCur & 7));
            pCur++;
            if (bit === 0) continue;
            const zz = raw[zzStart + nzCur++];
            const mag = (zz + 1) >> 1;
            output[qDst + k] = zz & 1 ? mag : 256 - mag;
          }
        }
      } else {
        for (let sub = 0; sub < 4; sub++) {
          const offK = sub * shortN;
          const scOff = sub * bShort;
          for (let g = 0; g < bShort; g++) {
            if (output[scaleDst + scOff + g] === 0) continue;
            const sK = offK + UNPACK_EDGES_S[g];
            const eK = offK + UNPACK_EDGES_S[g + 1];
            for (let k = sK; k < eK; k++) {
              const bit = raw[presStart + (pCur >> 3)] & (128 >> (pCur & 7));
              pCur++;
              if (bit === 0) continue;
              const zz = raw[zzStart + nzCur++];
              const mag = (zz + 1) >> 1;
              output[qDst + k] = zz & 1 ? mag : 256 - mag;
            }
          }
        }
      }
    }
    if (pCur !== presentCount || nzCur !== nonzero) {
      throw new Error('VWP7 active-band coefficient count mismatch');
    }
    return output.buffer;
  }

  if (magic === 'VWAP') {
    const nHops = view.getUint32(16, true);
    const actCount = view.getUint32(36, true);
    let offset = 40;
    const hopMaskStart = offset;
    offset += Math.ceil(nHops / 8);

    const actHops = [];
    for (let h = 0; h < nHops; h++) {
      if ((raw[hopMaskStart + (h >> 3)] & (128 >> (h & 7))) !== 0) {
        actHops.push(h);
      }
    }
    const output = new Uint8Array(36 + nHops * 4);
    output.set(raw.subarray(0, 36), 0);
    output.set([86, 87, 65, 55], 0); // "VWA7"

    if (actCount > 0) {
      for (let b = 0; b < 4; b++) {
        let acc = 0;
        for (let i = 0; i < actCount; i++) {
          acc = (acc + raw[offset++]) & 0xff;
          output[36 + actHops[i] * 4 + b] = acc;
        }
      }
    }
    return output.buffer;
  }

  if (magic === 'VWRP') {
    const nHops = view.getUint32(16, true);
    const output = new Uint8Array(32 + nHops * 2);
    output.set(raw.subarray(0, 32), 0);
    output.set([86, 87, 82, 55], 0); // "VWR7"
    let offset = 32;
    let accM = 0;
    for (let h = 0; h < nHops; h++) {
      accM = (accM + raw[offset++]) & 0xff;
      output[32 + h] = accM;
    }
    let accS = 0;
    for (let h = 0; h < nHops; h++) {
      accS = (accS + raw[offset++]) & 0xff;
      output[32 + nHops + h] = accS;
    }
    return output.buffer;
  }

  throw new Error(`Unsupported V7 vocal packet header: ${magic}`);
}
