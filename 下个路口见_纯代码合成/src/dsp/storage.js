export function unpackSpectrum(raw) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magic = String.fromCharCode(...raw.subarray(0, 4));
  if (magic !== 'MDP4') throw new Error(`Invalid V4 packed header: ${magic}`);

  const longN = view.getUint16(8, true);
  const tCount = view.getUint32(12, true);
  const bLong = view.getUint8(24);
  const bShort = view.getUint8(25);
  const nonzero = view.getUint32(28, true);

  let offset = 32;
  const modes = raw.subarray(offset, offset + tCount);
  offset += tCount;

  const stride = 1 + 48 + longN;
  const output = new Uint8Array(32 + tCount * stride);
  output.set(raw.subarray(0, 28), 0);
  output.set([77, 68, 81, 52], 0); // "MDQ4"

  const prevLong = new Int32Array(bLong);
  for (let t = 0; t < tCount; t++) {
    const dst = 32 + t * stride;
    const mode = modes[t];
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

  const totalBins = tCount * longN;
  const maskStart = offset;
  const maskBytes = Math.ceil(totalBins / 8);
  const signStart = maskStart + maskBytes;
  const signBytes = Math.ceil(nonzero / 8);
  const magStart = signStart + signBytes;
  if (raw.byteLength !== magStart + nonzero) {
    throw new Error(`Truncated MDP4 stream: ${raw.byteLength} != ${magStart + nonzero}`);
  }

  let nzIdx = 0;
  for (let t = 0; t < tCount; t++) {
    const qDst = 32 + t * stride + 49;
    const baseIdx = t * longN;
    for (let k = 0; k < longN; k++) {
      const bitIdx = baseIdx + k;
      if ((raw[maskStart + (bitIdx >> 3)] & (128 >> (bitIdx & 7))) === 0) continue;
      const mag = raw[magStart + nzIdx];
      const neg = raw[signStart + (nzIdx >> 3)] & (128 >> (nzIdx & 7));
      output[qDst + k] = neg ? 256 - mag : mag;
      nzIdx++;
    }
  }
  if (nzIdx !== nonzero) throw new Error('Coefficient count mismatch');
  return output.buffer;
}
