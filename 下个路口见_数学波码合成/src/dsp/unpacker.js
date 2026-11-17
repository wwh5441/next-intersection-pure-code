function ascii4(bytes, offset = 0) {
  return String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
}

function readF32LE(bytes, offset, count) {
  const out = new Float32Array(count);
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, count * 4);
  for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

function cumsumMod256(bytes, offset, count) {
  const out = new Uint8Array(count);
  let acc = 0;
  for (let i = 0; i < count; i++) {
    acc = (acc + bytes[offset + i]) & 0xff;
    out[i] = acc;
  }
  return out;
}

function cumsumI16(bytes, offset, count) {
  const out = new Int32Array(count);
  const dv = new DataView(bytes.buffer, bytes.byteOffset + offset, count * 2);
  let acc = 0;
  for (let i = 0; i < count; i++) {
    acc += dv.getInt16(i * 2, true);
    out[i] = acc;
  }
  return out;
}

export function unpackWaveStream(rawBuffer) {
  const bytes = rawBuffer instanceof Uint8Array ? rawBuffer : new Uint8Array(rawBuffer);
  const magic = ascii4(bytes, 0);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (magic === 'WVV5') {
    const nFrames = dv.getUint32(4, true);
    const nMels = bytes[8];
    const rank = bytes[9];
    const nStages = bytes[10];
    const nApCb = bytes[11];
    const stageSizes = [dv.getUint16(12, true), dv.getUint16(14, true), dv.getUint16(16, true)].slice(0, nStages);
    let off = 24;

    const stageScales = readF32LE(bytes, off, 3);
    off += 12;
    const uPron = readF32LE(bytes, off, nMels * rank);
    off += nMels * rank * 4;

    const codebooks = [];
    for (let s = 0; s < nStages; s++) {
      const sz = stageSizes[s];
      const sc = stageScales[s];
      const cb = new Float32Array(sz * rank);
      for (let i = 0; i < sz * rank; i++) {
        const signed = (bytes[off + i] << 24) >> 24;
        cb[i] = signed * sc;
      }
      off += sz * rank;
      codebooks.push(cb);
    }

    const cbAp = new Float32Array(nApCb * 8);
    for (let i = 0; i < nApCb * 8; i++) cbAp[i] = bytes[off + i] / 255;
    off += nApCb * 8;

    const rmsQ = cumsumMod256(bytes, off, nFrames);
    off += nFrames;
    const f0Q = cumsumI16(bytes, off, nFrames);
    off += nFrames * 2;
    const gainQ = cumsumMod256(bytes, off, nFrames);
    off += nFrames;

    const pronIndices = [];
    for (let s = 0; s < nStages; s++) {
      pronIndices.push(bytes.slice(off, off + nFrames));
      off += nFrames;
    }
    const apIdx = bytes.slice(off, off + nFrames);

    return {
      kind: 'vocal',
      nFrames,
      nMels,
      rank,
      nStages,
      uPron,
      codebooks,
      cbAp,
      rmsQ,
      f0Q,
      gainQ,
      pronIndices,
      apIdx,
    };
  }

  if (magic === 'WVB5') {
    const nFrames = dv.getUint32(4, true);
    const maxHarm = bytes[8];
    const nToneCb = bytes[9];
    const toneScale = dv.getFloat32(12, true);
    let off = 16;

    const cbTone = new Float32Array(nToneCb * maxHarm);
    for (let i = 0; i < nToneCb * maxHarm; i++) {
      const signed = (bytes[off + i] << 24) >> 24;
      cbTone[i] = signed * toneScale;
    }
    off += nToneCb * maxHarm;

    const rmsQ = cumsumMod256(bytes, off, nFrames);
    off += nFrames;
    const f0Q = cumsumI16(bytes, off, nFrames);
    off += nFrames * 2;
    const toneIdx = bytes.slice(off, off + nFrames);
    off += nFrames;
    const nrQ = bytes.slice(off, off + nFrames);

    return { kind: 'bass', nFrames, maxHarm, cbTone, rmsQ, f0Q, toneIdx, nrQ };
  }

  if (magic === 'WVD5') {
    const nFrames = dv.getUint32(4, true);
    const nComp = bytes[8];
    const nBands = bytes[9];
    const nProtos = bytes[10];
    const qLevels = bytes[11];
    const barFrames = dv.getUint16(12, true);
    const nBars = dv.getUint16(14, true);
    const kickHz = dv.getFloat32(16, true);
    const snareHz = dv.getFloat32(20, true);
    let off = 24;

    const bandCenters = readF32LE(bytes, off, nBands);
    off += nBands * 4;
    const wDrum = readF32LE(bytes, off, nBands * nComp);
    off += nBands * nComp * 4;
    const hMax = readF32LE(bytes, off, nComp);
    off += nComp * 4;

    const protoLen = nProtos * nComp * barFrames;
    const protoGrooves = bytes.slice(off, off + protoLen);
    off += protoLen;
    const barSchedule = bytes.slice(off, off + nBars);
    off += nBars;
    const dResH = bytes.slice(off, off + nComp * nFrames);
    off += nComp * nFrames;

    const kickQ = cumsumMod256(bytes, off, nFrames);
    off += nFrames;
    const bodyQ = cumsumMod256(bytes, off, nFrames);
    off += nFrames;
    const highQ = cumsumMod256(bytes, off, nFrames);

    const hRec = new Float32Array(nComp * nFrames);
    const invQL = 1 / qLevels;
    for (let c = 0; c < nComp; c++) {
      const hm = hMax[c];
      const cOff = c * nFrames;
      for (let t = 0; t < nFrames; t++) {
        const bIdx = Math.floor(t / barFrames);
        const fInBar = t - bIdx * barFrames;
        const pId = barSchedule[bIdx];
        const pred = protoGrooves[(pId * nComp + c) * barFrames + fInBar];
        const qVal = (pred + dResH[cOff + t]) & 0xff;
        const norm = qVal * invQL;
        hRec[cOff + t] = norm * norm * hm;
      }
    }

    return {
      kind: 'drums',
      nFrames,
      nComp,
      nBands,
      kickHz,
      snareHz,
      bandCenters,
      wDrum,
      hRec,
      kickQ,
      bodyQ,
      highQ,
    };
  }

  if (magic === 'WVH5') {
    const nFrames = dv.getUint32(4, true);
    const nNotes = bytes[8];
    const topVoices = bytes[9];
    const qLevels = bytes[10];
    const midiMin = bytes[11];
    const maxNoteAmp = dv.getFloat32(12, true);
    let off = 16;

    const noteAmpQ = new Uint8Array(nNotes * nFrames);
    for (let n = 0; n < nNotes; n++) {
      let acc = 0;
      const rowOff = n * nFrames;
      for (let t = 0; t < nFrames; t++) {
        acc = (acc + bytes[off + rowOff + t]) & 0xff;
        noteAmpQ[rowOff + t] = acc;
      }
    }
    off += nNotes * nFrames;

    const brightQ = bytes.slice(off, off + nFrames);
    off += nFrames;
    const harmQ = cumsumMod256(bytes, off, nFrames);
    off += nFrames;

    const nSFrames = nFrames * 2;
    const strumQ = new Uint8Array(4 * nSFrames);
    for (let b = 0; b < 4; b++) {
      let acc = 0;
      const bOff = b * nSFrames;
      for (let ts = 0; ts < nSFrames; ts++) {
        acc = (acc + bytes[off + bOff + ts]) & 0xff;
        strumQ[bOff + ts] = acc;
      }
    }
    off += 4 * nSFrames;

    const sideQ = cumsumMod256(bytes, off, nFrames);

    return {
      kind: 'harmony',
      nFrames,
      nNotes,
      topVoices,
      qLevels,
      midiMin,
      maxNoteAmp,
      noteAmpQ,
      brightQ,
      harmQ,
      strumQ,
      sideQ,
    };
  }

  throw new Error(`Unknown wave packet magic: ${magic}`);
}
