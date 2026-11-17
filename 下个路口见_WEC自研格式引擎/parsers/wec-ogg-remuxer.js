// Ephemeral In-Memory RFC 7845 Ogg-Opus Wrapper & OpusHead Builder
// Bridges raw RFC 6716 Opus packets stored in .wec directly to native C++ WebAudio / WebCodecs libopus

import { computeOggPageCrc32 } from './wec-crc32.js';

export function buildOpusHeadBytes(channels, preSkip, sampleRate = 48000, outputGainQ8 = 0) {
  const buf = new ArrayBuffer(19);
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  // "OpusHead"
  u8.set([0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64], 0);
  u8[8] = 1; // version
  u8[9] = channels;
  dv.setUint16(10, preSkip, true);
  dv.setUint32(12, sampleRate, true);
  dv.setInt16(16, outputGainQ8, true);
  u8[18] = 0; // channel mapping family 0 (mono/stereo)
  return u8;
}

function buildOpusTagsBytes() {
  const vendor = new TextEncoder().encode('WEC-v2-Opus');
  const buf = new ArrayBuffer(8 + 4 + vendor.length + 4);
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  u8.set([0x4f, 0x70, 0x75, 0x73, 0x54, 0x61, 0x67, 0x73], 0);
  dv.setUint32(8, vendor.length, true);
  u8.set(vendor, 12);
  dv.setUint32(12 + vendor.length, 0, true);
  return u8;
}

function makeOggPage(headerType, granulePosBigInt, serialNo, pageSeqNo, packets) {
  const segTable = [];
  let payloadBytes = 0;
  for (const pkt of packets) {
    let rem = pkt.byteLength;
    payloadBytes += rem;
    while (rem >= 255) {
      segTable.push(255);
      rem -= 255;
    }
    segTable.push(rem);
  }
  const pageLen = 27 + segTable.length + payloadBytes;
  const page = new Uint8Array(pageLen);
  const dv = new DataView(page.buffer);
  // "OggS"
  page[0] = 0x4f;
  page[1] = 0x67;
  page[2] = 0x67;
  page[3] = 0x53;
  page[4] = 0; // stream_structure_version
  page[5] = headerType;
  dv.setBigInt64(6, granulePosBigInt, true);
  dv.setUint32(14, serialNo >>> 0, true);
  dv.setUint32(18, pageSeqNo >>> 0, true);
  dv.setUint32(22, 0, true); // zero before CRC
  page[26] = segTable.length;
  for (let i = 0; i < segTable.length; i++) {
    page[27 + i] = segTable[i];
  }
  let pos = 27 + segTable.length;
  for (const pkt of packets) {
    page.set(pkt, pos);
    pos += pkt.byteLength;
  }
  const crc = computeOggPageCrc32(page);
  dv.setUint32(22, crc, true);
  return page;
}

export function wrapTrackOpusPacketsAsOggBuffer(track, totalSamples, sampleRate = 48000) {
  const serial = 0x57454332 + (track.trackIndex || 0);
  // Keep OpusHead gain at 0 and apply track linear gain cleanly in Float32 mixer to avoid clipping
  const opusHead = buildOpusHeadBytes(track.channels, track.preSkip, sampleRate, 0);
  const opusTags = buildOpusTagsBytes();

  const pages = [];
  let seq = 0;
  pages.push(makeOggPage(0x02, 0n, serial, seq++, [opusHead]));
  pages.push(makeOggPage(0x00, 0n, serial, seq++, [opusTags]));

  const nPkts = track.packetCount;
  const samplesPerPkt = Math.round((sampleRate * (track.frameDurationMs || 20)) / 1000);
  const maxGranule = BigInt(totalSamples + track.preSkip);

  // Group up to 40 packets per page (~800ms) so segment table length is always <= 255
  let batch = [];
  let batchSegs = 0;
  let cumGranule = 0n;

  for (let i = 0; i < nPkts; i++) {
    const pktOff = track.packetOffsets[i];
    const pktLen = track.packetLengths[i];
    const pktSlice = track.audioData.subarray(pktOff, pktOff + pktLen);
    const neededSegs = Math.floor(pktLen / 255) + 1;

    if (batch.length > 0 && (batchSegs + neededSegs > 250 || batch.length >= 45)) {
      pages.push(makeOggPage(0x00, cumGranule, serial, seq++, batch));
      batch = [];
      batchSegs = 0;
    }
    batch.push(pktSlice);
    batchSegs += neededSegs;
    cumGranule += BigInt(samplesPerPkt);
  }

  if (batch.length > 0) {
    const eosGranule = cumGranule > maxGranule ? maxGranule : cumGranule;
    pages.push(makeOggPage(0x04, eosGranule, serial, seq++, batch));
  }

  let totalLen = 0;
  for (const p of pages) totalLen += p.byteLength;
  const out = new Uint8Array(totalLen);
  let w = 0;
  for (const p of pages) {
    out.set(p, w);
    w += p.byteLength;
  }
  return out.buffer;
}
