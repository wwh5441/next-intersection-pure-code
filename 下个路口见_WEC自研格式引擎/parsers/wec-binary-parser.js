// WEC v2.0 Binary Container Parser (Supports WEC0 Pure-Minimal, WEC1 Dual-Track, WEC2 >=4+3 Unbounded)

import { WEC_SPEC_VERSION, WEC_ROLE_META, WEC_FEATURE_META } from '../models/wec-format-constants.js';
import { computeWecCrc32, formatHex32 } from './wec-crc32.js';

function decodeU8ExtLengths(u8, offset, byteLen, packetCount) {
  const lengths = new Uint16Array(packetCount);
  const offsets = new Uint32Array(packetCount + 1);
  let pos = offset;
  const end = offset + byteLen;
  let cum = 0;
  for (let i = 0; i < packetCount; i++) {
    if (pos >= end) throw new Error(`Truncated u8_ext packet index at packet ${i}`);
    const b = u8[pos++];
    let ln = b;
    if (b === 255) {
      ln = u8[pos] | (u8[pos + 1] << 8);
      pos += 2;
    }
    lengths[i] = ln;
    offsets[i] = cum;
    cum += ln;
  }
  offsets[packetCount] = cum;
  return { lengths, offsets, totalAudioBytes: cum };
}

async function inflateZlibBytes(u8Slice) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('Environment lacks Web Standard DecompressionStream("deflate")');
  }
  const ds = new DecompressionStream('deflate');
  const blob = new Blob([u8Slice]);
  const stream = blob.stream().pipeThrough(ds);
  const ab = await new Response(stream).arrayBuffer();
  return new Uint8Array(ab);
}

export function parseTimbBinary(rawU8) {
  const dv = new DataView(rawU8.buffer, rawU8.byteOffset, rawU8.byteLength);
  const magic = String.fromCharCode(rawU8[0], rawU8[1], rawU8[2], rawU8[3]);
  if (magic !== 'TIM1') {
    throw new Error(`Invalid TIMB binary header magic: ${magic}`);
  }
  const hopSamples = dv.getUint16(4, true);
  const timbVer = dv.getUint16(6, true);
  const frameCount = dv.getUint32(8, true);
  const meanF0Hz = dv.getFloat32(12, true);
  const f0MinHz = dv.getFloat32(16, true);
  const f0MaxHz = dv.getFloat32(20, true);
  const centroidHz = dv.getFloat32(24, true);

  const f0Cents = new Uint16Array(frameCount);
  const rmsQ = new Uint8Array(frameCount);
  const voicing = new Uint8Array(frameCount);
  const chest = new Uint8Array(frameCount);
  const f1 = new Uint8Array(frameCount);
  const f2 = new Uint8Array(frameCount);
  const air = new Uint8Array(frameCount);

  let p = 32;
  let voicedCount = 0;
  for (let i = 0; i < frameCount; i++) {
    f0Cents[i] = dv.getUint16(p, true);
    rmsQ[i] = rawU8[p + 2];
    voicing[i] = rawU8[p + 3];
    chest[i] = rawU8[p + 4];
    f1[i] = rawU8[p + 5];
    f2[i] = rawU8[p + 6];
    air[i] = rawU8[p + 7];
    if (voicing[i] & 1) voicedCount++;
    p += 8;
  }

  return {
    magic,
    hopSamples,
    frameCount,
    meanF0Hz: Math.round(meanF0Hz * 100) / 100,
    f0MinHz: Math.round(f0MinHz * 100) / 100,
    f0MaxHz: Math.round(f0MaxHz * 100) / 100,
    centroidHz: Math.round(centroidHz * 10) / 10,
    voicedFrames: voicedCount,
    f0Cents,
    rmsQ,
    voicing,
    chest,
    f1,
    f2,
    air,
  };
}

export function decodeOpusTocByte(toc) {
  const config = (toc >>> 3) & 0x1f;
  const stereo = ((toc >>> 2) & 1) === 1;
  const frameCountCode = toc & 0x03;

  let mode = 'CELT-only';
  if (config < 12) mode = 'SILK-only';
  else if (config < 16) mode = 'Hybrid (SILK+CELT)';

  let bandwidth = 'FB (20kHz)';
  if (config <= 3) bandwidth = 'NB (4kHz)';
  else if (config <= 7) bandwidth = 'MB (6kHz)';
  else if (config <= 11 || (config >= 16 && config <= 19)) {
    bandwidth = config <= 11 ? 'WB (8kHz)' : 'NB (4kHz)';
  } else if (config === 12 || config === 13 || (config >= 24 && config <= 27)) {
    bandwidth = 'SWB (12kHz)';
  } else if (config >= 20 && config <= 23) {
    bandwidth = 'WB (8kHz)';
  }

  return { config, mode, bandwidth, stereo, frameCountCode };
}

export async function parseWecContainerAsync(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer);
  if (u8.byteLength < 32) {
    throw new Error('File too small to be a valid .wec container (< 32 bytes)');
  }
  const dv = new DataView(arrayBuffer);
  const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  const version = u8[4];

  if (!['WEC0', 'WEC1', 'WEC2'].includes(magic)) {
    throw new Error(`Invalid .wec magic signature: "${magic}" (expected WEC0, WEC1, or WEC2)`);
  }
  if (version !== WEC_SPEC_VERSION) {
    throw new Error(`Unsupported WEC version ${version} (expected v${WEC_SPEC_VERSION} Opus container)`);
  }

  // Case 1: WEC0 Pure-Minimal 32-Byte Single-Stream Opus Container
  if (magic === 'WEC0') {
    const codecId = u8[5];
    const channels = u8[6];
    const frameDurationMs = u8[7];
    const sampleRate = dv.getUint32(8, true);
    const totalSamples = dv.getUint32(12, true);
    const preSkip = dv.getUint16(16, true);
    const outputGainQ8 = dv.getInt16(18, true);
    const packetCount = dv.getUint32(20, true);
    const indexByteLen = dv.getUint32(24, true);
    const storedCrc = dv.getUint32(28, true);
    const computedCrc = computeWecCrc32(u8, 28);
    if (storedCrc !== computedCrc) {
      throw new Error(`WEC0 CRC-32 mismatch: stored 0x${formatHex32(storedCrc)} vs computed 0x${formatHex32(computedCrc)}`);
    }

    const { lengths, offsets, totalAudioBytes } = decodeU8ExtLengths(u8, 32, indexByteLen, packetCount);
    const audioOffset = 32 + indexByteLen;
    const audioData = u8.subarray(audioOffset, audioOffset + totalAudioBytes);
    const roleMeta = WEC_ROLE_META[0x00];

    const track = {
      trackIndex: 0,
      roleId: 0x00,
      roleCode: roleMeta.code,
      roleZh: channels === 1 ? '单流最小核混音 (Mono Mix)' : roleMeta.zh,
      color: roleMeta.color,
      codecId,
      channels,
      frameDurationMs,
      preSkip,
      defaultGainDb: outputGainQ8 / 256.0,
      defaultPan: 0.0,
      targetBitrateKbps: Math.round(((totalAudioBytes * 8) / (totalSamples / sampleRate)) / 1000),
      packetCount,
      indexOffset: 32,
      indexByteLen,
      audioOffset,
      audioByteLen: totalAudioBytes,
      label: 'master_minimal',
      packetLengths: lengths,
      packetOffsets: offsets,
      audioData,
    };

    return {
      rawBytes: u8,
      fileByteLength: u8.byteLength,
      magic,
      family: 'WEC0',
      version,
      headerSizeBytes: 32,
      flags: 0,
      sampleRate,
      totalSamples,
      durationSec: totalSamples / sampleRate,
      bpm: 98.0,
      frameDurationMs,
      crcValid: true,
      storedCrcHex: `0x${formatHex32(storedCrc)}`,
      computedCrcHex: `0x${formatHex32(computedCrc)}`,
      tracks: [track],
      features: [],
      decodedFeatures: {},
    };
  }

  // Case 2: WEC1 (Dual-Track) or WEC2 (>= 4+3 Multi-Layer Master)
  if (u8.byteLength < 48) {
    throw new Error(`${magic} container truncated (< 48 bytes)`);
  }
  const flags = u8[5];
  const numAudioTracks = dv.getUint16(6, true);
  const numFeatureLayers = dv.getUint16(8, true);
  const frameDurationMs = dv.getUint16(10, true);
  const sampleRate = dv.getUint32(12, true);
  const totalSamples = dv.getUint32(16, true);
  const bpm = dv.getFloat32(20, true);
  const trackDirOffset = dv.getUint32(24, true);
  const featureDirOffset = dv.getUint32(28, true);
  const payloadOffset = dv.getUint32(32, true);
  const totalPayloadBytes = dv.getUint32(36, true);
  const storedCrc = dv.getUint32(44, true);
  const computedCrc = computeWecCrc32(u8, 44);

  if (storedCrc !== computedCrc) {
    throw new Error(`${magic} CRC-32 mismatch: stored 0x${formatHex32(storedCrc)} vs computed 0x${formatHex32(computedCrc)}`);
  }
  if (magic === 'WEC1' && (numAudioTracks < 1 || numAudioTracks > 2)) {
    throw new Error(`WEC1 tier violation: expected 1..2 audio tracks, found ${numAudioTracks}`);
  }
  if (magic === 'WEC2' && (numAudioTracks < 4 || numFeatureLayers < 3)) {
    throw new Error(
      `WEC2 tier violation: requires >=4 audio tracks and >=3 feature layers, found ${numAudioTracks}+${numFeatureLayers}`
    );
  }

  const decoderUtf8 = new TextDecoder('utf-8');
  const tracks = [];
  for (let ti = 0; ti < numAudioTracks; ti++) {
    const base = trackDirOffset + ti * 48;
    const trackIndex = u8[base];
    const roleId = u8[base + 1];
    const codecId = u8[base + 2];
    const channels = u8[base + 3];
    const preSkip = dv.getUint16(base + 4, true);
    const defaultGainQ8 = dv.getInt16(base + 6, true);
    const defaultPanQ8 = dv.getInt16(base + 8, true);
    const targetBitrateKbps = dv.getUint16(base + 10, true);
    const packetCount = dv.getUint32(base + 12, true);
    const indexOffset = dv.getUint32(base + 16, true);
    const indexByteLen = dv.getUint32(base + 20, true);
    const audioOffset = dv.getUint32(base + 24, true);
    const audioByteLen = dv.getUint32(base + 28, true);

    const rawLabelSlice = u8.subarray(base + 32, base + 48);
    let zeroIdx = rawLabelSlice.indexOf(0);
    if (zeroIdx < 0) zeroIdx = 16;
    const label = decoderUtf8.decode(rawLabelSlice.subarray(0, zeroIdx)) || `track_${ti}`;
    const roleMeta = WEC_ROLE_META[roleId] || WEC_ROLE_META[0xff];

    const { lengths, offsets } = decodeU8ExtLengths(u8, indexOffset, indexByteLen, packetCount);
    const audioData = u8.subarray(audioOffset, audioOffset + audioByteLen);

    tracks.push({
      trackIndex,
      roleId,
      roleCode: roleMeta.code,
      roleZh: roleMeta.zh,
      color: roleMeta.color,
      codecId,
      channels,
      frameDurationMs,
      preSkip,
      defaultGainDb: defaultGainQ8 / 256.0,
      defaultPan: defaultPanQ8 / 256.0,
      targetBitrateKbps,
      packetCount,
      indexOffset,
      indexByteLen,
      audioOffset,
      audioByteLen,
      label,
      packetLengths: lengths,
      packetOffsets: offsets,
      audioData,
    });
  }

  const features = [];
  const decodedFeatures = {};
  for (let fi = 0; fi < numFeatureLayers; fi++) {
    const base = featureDirOffset + fi * 32;
    const fourCC = String.fromCharCode(u8[base], u8[base + 1], u8[base + 2], u8[base + 3]).trim();
    const encoding = u8[base + 4];
    const layerFlags = u8[base + 5];
    const schemaVersion = dv.getUint16(base + 6, true);
    const dataOffset = dv.getUint32(base + 8, true);
    const compressedByteLen = dv.getUint32(base + 12, true);
    const uncompressedByteLen = dv.getUint32(base + 16, true);
    const itemCount = dv.getUint32(base + 20, true);
    const descSlice = u8.subarray(base + 24, base + 32);
    let z = descSlice.indexOf(0);
    if (z < 0) z = 8;
    const shortDesc = decoderUtf8.decode(descSlice.subarray(0, z));

    const storedSlice = u8.subarray(dataOffset, dataOffset + compressedByteLen);
    let rawSlice = storedSlice;
    if (encoding === 2 || encoding === 3) {
      rawSlice = await inflateZlibBytes(storedSlice);
    }

    let parsedPayload = null;
    if (fourCC === 'TIMB') {
      parsedPayload = parseTimbBinary(rawSlice);
    } else if (encoding === 1 || encoding === 3) {
      try {
        parsedPayload = JSON.parse(decoderUtf8.decode(rawSlice));
      } catch (_) {
        parsedPayload = null;
      }
    }

    const metaInfo = WEC_FEATURE_META[fourCC] || {
      name: `${fourCC} · 自定义扩展特征层`,
      desc: shortDesc || 'Custom TLV Layer',
    };

    const fObj = {
      layerIndex: fi,
      fourCC,
      title: metaInfo.name,
      description: metaInfo.desc,
      encoding,
      encodingName: ['RawBin', 'UTF8Json', 'ZlibBin', 'ZlibUTF8Json'][encoding] || `Code(${encoding})`,
      layerFlags,
      schemaVersion,
      dataOffset,
      compressedByteLen,
      uncompressedByteLen,
      itemCount,
      shortDesc,
      parsed: parsedPayload,
    };
    features.push(fObj);
    decodedFeatures[fourCC] = parsedPayload;
  }

  return {
    rawBytes: u8,
    fileByteLength: u8.byteLength,
    magic,
    family: magic,
    version,
    headerSizeBytes: 48,
    flags,
    numAudioTracks,
    numFeatureLayers,
    sampleRate,
    totalSamples,
    durationSec: totalSamples / sampleRate,
    bpm: Math.round(bpm * 100) / 100 || 98.0,
    frameDurationMs,
    trackDirOffset,
    featureDirOffset,
    payloadOffset,
    totalPayloadBytes,
    crcValid: true,
    storedCrcHex: `0x${formatHex32(storedCrc)}`,
    computedCrcHex: `0x${formatHex32(computedCrc)}`,
    tracks,
    features,
    decodedFeatures,
  };
}
