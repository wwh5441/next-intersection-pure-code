const ILLEGAL = [0, 10, 13, 34, 38, 92];

export function decodeBase122(strData) {
  const out = [];
  let curByte = 0;
  let bitOfByte = 0;

  function push7(byte7) {
    const shifted = byte7 << 1;
    curByte |= shifted >>> bitOfByte;
    bitOfByte += 7;
    if (bitOfByte >= 8) {
      out.push(curByte & 0xff);
      bitOfByte -= 8;
      curByte = (shifted << (7 - bitOfByte)) & 0xff;
    }
  }

  for (let i = 0; i < strData.length; i++) {
    const c = strData.charCodeAt(i);
    if (c > 127) {
      const illegalIdx = (c >>> 8) & 7;
      if (illegalIdx !== 7) {
        push7(ILLEGAL[illegalIdx]);
      }
      push7(c & 127);
    } else {
      push7(c);
    }
  }
  return Uint8Array.from(out);
}

export function buildEnvelopeKnotPositions(totalFrames, consFrames, vowelStep) {
  if (totalFrames <= 1) return Int32Array.of(0);
  const last = totalFrames - 1;
  const vot = Math.min(last, Math.max(1, consFrames));
  const step = Math.max(1, vowelStep);
  const coda = Math.max(vot, last - Math.max(2, Math.floor(step / 2)));
  const set = new Set([0, Math.floor(vot / 2), vot, coda, last]);
  let cur = vot + step;
  while (cur < last) {
    set.add(cur);
    cur += step;
  }
  return Int32Array.from([...set].sort((a, b) => a - b));
}

export function buildAdaptiveExprKnotPositions(totalFrames, consFrames, exprStep) {
  if (totalFrames <= 1) return Int32Array.of(0);
  const last = totalFrames - 1;
  const vot = Math.min(last, Math.max(1, consFrames));
  const step = Math.max(1, exprStep);
  const halfS = Math.max(2, Math.floor(step / 2));
  const glide = Math.min(last, vot + halfS);
  const coda = Math.max(glide, last - halfS);
  const set = new Set([0, Math.floor(vot / 2), vot, glide, coda, last]);
  let cur = glide + step;
  while (cur < last) {
    set.add(cur);
    cur += step;
  }
  return Int32Array.from([...set].sort((a, b) => a - b));
}

export function buildUniformKnotPositions(totalFrames, step) {
  if (totalFrames <= 1) return Int32Array.of(0);
  const last = totalFrames - 1;
  const s = Math.max(1, step);
  const pts = [];
  for (let p = 0; p < last; p += s) pts.push(p);
  if (pts.length === 0 || pts[pts.length - 1] !== last) pts.push(last);
  return Int32Array.from(pts);
}

export function reconstructMidiKnots(baseMidiQ, f0StepQ) {
  const nK = 1 + f0StepQ.length;
  const out = new Float32Array(nK);
  let cur = baseMidiQ * 0.02;
  out[0] = cur;
  for (let k = 0; k < f0StepQ.length; k++) {
    cur += f0StepQ[k] * 0.025;
    out[k + 1] = cur;
  }
  return out;
}

export function unpackSpeakerCore(raw) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magic = String.fromCharCode(raw[0], raw[1], raw[2], raw[3]);
  if (magic !== 'VGS8') throw new Error(`Invalid speaker packet magic: ${magic}`);

  const sampleRate = view.getUint32(4, true);
  const numSamples = view.getUint32(8, true);
  const hop = view.getUint32(12, true);
  const refPeak = view.getFloat32(16, true);
  const nCep = view.getUint16(20, true);
  const nHarm = view.getUint16(22, true);
  const nAper = view.getUint16(24, true);
  const kMcep = view.getUint16(26, true);
  const kAper = view.getUint16(28, true);

  let off = 32;
  const readI16Float = (count, div) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = view.getInt16(off, true) / div;
      off += 2;
    }
    return out;
  };

  const meanMcep = readI16Float(nCep, 256.0);
  const glottalPhase = readI16Float(nHarm, 2048.0);
  const mcepPcaBasis = readI16Float(kMcep * nCep, 2048.0);
  const aperMean = readI16Float(nAper, 256.0);
  const aperPcaBasis = readI16Float(kAper * nAper, 2048.0);

  return {
    sampleRate,
    numSamples,
    hop,
    refPeak,
    nCep,
    nHarm,
    nAper,
    kMcep,
    kAper,
    meanMcep,
    glottalPhase,
    mcepPcaBasis,
    aperMean,
    aperPcaBasis,
  };
}

export function unpackSyllableAtlas(raw) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magic = String.fromCharCode(raw[0], raw[1], raw[2], raw[3]);
  if (magic !== 'VGA8') throw new Error(`Invalid syllable atlas magic: ${magic}`);

  const nProtos = view.getUint16(4, true);
  const nCep = view.getUint8(6);
  const nAper = view.getUint8(7);
  const qStep = view.getFloat32(8, true);
  const metaLen = view.getUint16(12, true);

  let off = 14;
  const metaStr = new TextDecoder().decode(raw.subarray(off, off + metaLen));
  off += metaLen;
  const metaParts = metaLen > 0 ? metaStr.split('|') : [];

  const readI8Scaled = (count) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const v = (raw[off++] << 24) >> 24;
      out[i] = v * qStep;
    }
    return out;
  };

  const consMAll = readI8Scaled(nProtos * 3 * nCep);
  const consAAll = readI8Scaled(nProtos * 3 * nAper);
  const vowMAll = readI8Scaled(nProtos * 5 * nCep);
  const vowAAll = readI8Scaled(nProtos * 5 * nAper);

  const prototypes = [];
  for (let sid = 0; sid < nProtos; sid++) {
    const [pinyin, initialClass, cntStr] = (metaParts[sid] || 'a:glide:1').split(':');
    prototypes.push({
      id: sid,
      pinyin,
      initialClass,
      count: Number(cntStr) || 1,
      consMcepKnots: consMAll.subarray(sid * 3 * nCep, (sid + 1) * 3 * nCep),
      consAperKnots: consAAll.subarray(sid * 3 * nAper, (sid + 1) * 3 * nAper),
      vowelMcepKnots: vowMAll.subarray(sid * 5 * nCep, (sid + 1) * 5 * nCep),
      vowelAperKnots: vowAAll.subarray(sid * 5 * nAper, (sid + 1) * 5 * nAper),
    });
  }
  return { nProtos, nCep, nAper, prototypes };
}

export function unpackEventStream(raw, atlasPrototypes = null) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magic = String.fromCharCode(raw[0], raw[1], raw[2], raw[3]);
  if (magic !== 'VGE8') throw new Error(`Invalid event stream magic: ${magic}`);

  const nEv = view.getUint16(4, true);
  const kPca = view.getUint8(6);
  const kAper = view.getUint8(7);
  const exprStep = view.getUint8(8);
  const f0Step = view.getUint8(9);
  const envStep = view.getUint8(10);
  const pfBetaQ = view.getUint8(11);
  const qStepDb = view.getFloat32(12, true);
  const dispLen = view.getUint32(16, true);

  let off = 20;
  const dispStr = dispLen > 0 ? new TextDecoder().decode(raw.subarray(off, off + dispLen)) : '';
  const dispList = dispStr ? dispStr.split('\t') : new Array(nEv).fill('');
  off += dispLen;

  const gaps = new Uint16Array(nEv);
  for (let i = 0; i < nEv; i++, off += 2) gaps[i] = view.getUint16(off, true);
  const totals = new Uint16Array(nEv);
  for (let i = 0; i < nEv; i++, off += 2) totals[i] = view.getUint16(off, true);
  const cons = raw.subarray(off, off + nEv);
  off += nEv;
  const sids = raw.subarray(off, off + nEv);
  off += nEv;
  const flags = raw.subarray(off, off + nEv);
  off += nEv;
  const baseMidiZz = new Uint16Array(nEv);
  for (let i = 0; i < nEv; i++, off += 2) baseMidiZz[i] = view.getUint16(off, true);
  const pans = new Int8Array(nEv);
  for (let i = 0; i < nEv; i++) pans[i] = (raw[off++] << 24) >> 24;
  const backL = raw.subarray(off, off + nEv);
  off += nEv;
  const backR = raw.subarray(off, off + nEv);
  off += nEv;
  const backP = new Int8Array(nEv);
  for (let i = 0; i < nEv; i++) backP[i] = (raw[off++] << 24) >> 24;
  const vOnsets = new Uint16Array(nEv);
  for (let i = 0; i < nEv; i++, off += 2) vOnsets[i] = view.getUint16(off, true);

  const f0Lens = new Int32Array(nEv);
  const envLens = new Int32Array(nEv);
  const exprLens = new Int32Array(nEv);
  let totF0 = 0;
  let totEnv = 0;
  let totExpr = 0;

  for (let i = 0; i < nEv; i++) {
    const L = totals[i];
    const cf = cons[i];
    const unv = (flags[i] & 8) !== 0;
    f0Lens[i] = unv ? 0 : Math.max(0, buildUniformKnotPositions(L, f0Step).length - 1);
    envLens[i] = buildEnvelopeKnotPositions(L, cf, envStep).length;
    exprLens[i] = buildAdaptiveExprKnotPositions(L, cf, exprStep).length;
    totF0 += f0Lens[i];
    totEnv += envLens[i];
    totExpr += exprLens[i];
  }

  const f0Stream = raw.subarray(off, off + totF0);
  off += totF0;
  const ghStream = raw.subarray(off, off + totEnv);
  off += totEnv;
  const gaStream = raw.subarray(off, off + totEnv);
  off += totEnv;

  const mcepStreams = [];
  for (let m = 0; m < kPca; m++) {
    mcepStreams.push(raw.subarray(off, off + totExpr));
    off += totExpr;
  }
  const aperStreams = [];
  for (let m = 0; m < kAper; m++) {
    aperStreams.push(raw.subarray(off, off + totExpr));
    off += totExpr;
  }

  const events = [];
  let prevEnd = 0;
  let prevMidi = 3000;
  let cF0 = 0;
  let cEnv = 0;
  let cExpr = 0;

  for (let i = 0; i < nEv; i++) {
    const L = totals[i];
    const cf = cons[i];
    const fl = flags[i];
    const unv = (fl & 8) !== 0;
    const startFrame = prevEnd + gaps[i];
    prevEnd = startFrame + L;

    const zz = baseMidiZz[i];
    const dm = (zz >>> 1) ^ -(zz & 1);
    const baseMidiQ = prevMidi + dm;
    if (!unv) prevMidi = baseMidiQ;

    const nf = f0Lens[i];
    const f0StepQ = new Int8Array(nf);
    for (let k = 0; k < nf; k++) {
      f0StepQ[k] = f0Stream[cF0++] - 128;
    }

    const ne = envLens[i];
    const gainHQ = new Uint8Array(ne);
    const gainAQ = new Uint8Array(ne);
    let accH = 0;
    let accA = 0;
    for (let k = 0; k < ne; k++) {
      accH = (accH + ghStream[cEnv + k]) & 0xff;
      accA = (accA + gaStream[cEnv + k]) & 0xff;
      gainHQ[k] = accH;
      gainAQ[k] = accA;
    }
    cEnv += ne;

    const nx = exprLens[i];
    const exprMcepQ = new Int8Array(nx * kPca);
    for (let m = 0; m < kPca; m++) {
      let acc = 0;
      const col = mcepStreams[m];
      for (let k = 0; k < nx; k++) {
        acc = (acc + col[cExpr + k]) & 0xff;
        exprMcepQ[k * kPca + m] = acc - 128;
      }
    }
    const exprAperQ = new Int8Array(nx * kAper);
    for (let m = 0; m < kAper; m++) {
      let acc = 0;
      const col = aperStreams[m];
      for (let k = 0; k < nx; k++) {
        acc = (acc + col[cExpr + k]) & 0xff;
        exprAperQ[k * kAper + m] = acc - 128;
      }
    }
    cExpr += nx;

    const sid = sids[i];
    events.push({
      index: i,
      startFrame,
      consFrames: cf,
      totalFrames: L,
      syllableId: sid,
      pinyin: atlasPrototypes ? atlasPrototypes[sid]?.pinyin || '' : '',
      displayText: dispList[i] || '',
      legatoPrev: (fl & 1) !== 0,
      legatoNext: (fl & 2) !== 0,
      isHook: (fl & 4) !== 0,
      unvoicedOnly: unv,
      isPlosive: (fl & 16) !== 0,
      baseMidiQ,
      stereoPanQ: pans[i],
      backGainLQ: backL[i],
      backGainRQ: backR[i],
      backPitchQ: backP[i],
      voicedOnsetFrame: vOnsets[i],
      f0StepQ,
      gainHQ,
      gainAQ,
      exprMcepQ,
      exprAperQ,
    });
  }

  return {
    meta: {
      numEvents: nEv,
      kPca,
      kAper,
      exprStep,
      f0Step,
      envStep,
      formantPostFilterBeta: pfBetaQ / 200.0,
      qStepDb,
    },
    events,
  };
}
