import { unpackWavePacket } from './v7-storage.js';
import { CosineWaveStream } from './v7-wave-stream.js';
import { HighAirStream } from './v7-air-stream.js';

function decodeStreamBuffer(rawSub, bankLen, sampleRate) {
  const magic = String.fromCharCode(rawSub[0], rawSub[1], rawSub[2], rawSub[3]);
  const pkt = unpackWavePacket(rawSub);
  const stream =
    magic === 'VWA7' || magic === 'VWAP'
      ? new HighAirStream(pkt, sampleRate)
      : new CosineWaveStream(pkt, sampleRate);
  const out = new Float32Array(bankLen);
  for (let i = 0; i < bankLen; i++) {
    out[i] = stream.valueAt(i);
  }
  return out;
}

function parseVda8Packet(raw, sampleRate) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const magic = String.fromCharCode(raw[0], raw[1], raw[2], raw[3]);
  if (magic !== 'VDA8') {
    throw new Error(`Invalid VDA8 magic: ${magic}`);
  }
  const numProtos = view.getUint16(6, true);
  const segLen = view.getUint32(8, true);
  const stride = view.getUint32(12, true);
  const xfade = view.getUint16(16, true);
  const numEvents = view.getUint32(18, true);
  const nSamples = view.getUint32(22, true);
  const nStreams = view.getUint32(26, true);

  let off = 30;
  const bankLen = numProtos * stride;
  const streamBanks = [];
  for (let s = 0; s < nStreams; s++) {
    const sLen = view.getUint32(off, true);
    off += 4;
    const rawSub = raw.subarray(off, off + sLen);
    off += sLen;
    streamBanks.push(decodeStreamBuffer(rawSub, bankLen, sampleRate));
  }

  const starts = new Uint32Array(numEvents);
  const protoIds = new Uint16Array(numEvents);
  const gains = new Float32Array(numEvents);
  for (let i = 0; i < numEvents; i++) {
    starts[i] = view.getUint32(off, true);
    protoIds[i] = view.getUint16(off + 4, true);
    gains[i] = view.getFloat32(off + 6, true);
    off += 10;
  }

  return { numProtos, segLen, stride, xfade, numEvents, nSamples, streamBanks, starts, protoIds, gains };
}

function scheduleBank(bank, parsed, targetLen) {
  const { segLen, stride, xfade, numEvents, starts, protoIds, gains } = parsed;
  const out = new Float32Array(targetLen);
  const invXf = xfade > 1 ? Math.PI / (xfade - 1) : 0;

  for (let i = 0; i < numEvents; i++) {
    const p = starts[i];
    const protoOff = protoIds[i] * stride;
    const g = gains[i];
    const nextP = i + 1 < numEvents ? starts[i + 1] : targetLen;
    const avail = Math.min(segLen, targetLen - p, Math.max(0, nextP - p + xfade));
    if (avail <= 0) continue;

    const fadeIn = i > 0 && avail >= xfade;
    const fadeOut = i + 1 < numEvents && avail >= xfade;
    const fadeOutStart = avail - xfade;

    for (let k = 0; k < avail; k++) {
      let w = 1.0;
      if (fadeIn && k < xfade) w *= 0.5 * (1.0 - Math.cos(k * invXf));
      if (fadeOut && k >= fadeOutStart) {
        const rem = avail - 1 - k;
        w *= 0.5 * (1.0 - Math.cos(rem * invXf));
      }
      out[p + k] += g * w * bank[protoOff + k];
    }
  }
  return out;
}

export function synthesizeV8DictAccompaniment(rawTracks, numSamples, sampleRate = 48000) {
  // 1. Bass (mono: stream 0)
  const bassP = parseVda8Packet(rawTracks.bass, sampleRate);
  const bassMono = scheduleBank(bassP.streamBanks[0], bassP, numSamples);
  const bassL = bassMono;
  const bassR = new Float32Array(bassMono);

  // 2. Drums (mono: stream 0 body + stream 1 air)
  const drumsP = parseVda8Packet(rawTracks.drums, sampleRate);
  const dBank = drumsP.streamBanks[0];
  const dAir = drumsP.streamBanks[1];
  for (let i = 0; i < dBank.length; i++) dBank[i] += dAir[i];
  const drumsMono = scheduleBank(dBank, drumsP, numSamples);
  const drumsL = drumsMono;
  const drumsR = new Float32Array(drumsMono);

  // 3. Guitar/Synth (stereo M/S: stream 0 mid + stream 2 air, stream 1 side)
  const gtrP = parseVda8Packet(rawTracks.guitar_synth, sampleRate);
  const gMid = gtrP.streamBanks[0];
  const gSide = gtrP.streamBanks[1];
  const gAir = gtrP.streamBanks[2];
  for (let i = 0; i < gMid.length; i++) gMid[i] += gAir[i];
  const midOut = scheduleBank(gMid, gtrP, numSamples);
  const sideOut = scheduleBank(gSide, gtrP, numSamples);
  const gtrL = new Float32Array(numSamples);
  const gtrR = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    gtrL[i] = midOut[i] + sideOut[i];
    gtrR[i] = midOut[i] - sideOut[i];
  }

  return {
    bass: [bassL, bassR],
    drums: [drumsL, drumsR],
    guitar_synth: [gtrL, gtrR],
  };
}
