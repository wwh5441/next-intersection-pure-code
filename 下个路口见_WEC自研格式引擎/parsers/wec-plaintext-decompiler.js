// White-Box Plaintext Decompiler & RFC 6716 Opus Packet Inspector for WEC0 / WEC1 / WEC2

import { decodeOpusTocByte } from './wec-binary-parser.js';

export function midiCentsToNoteName(cents) {
  if (!cents || cents <= 0) return 'UV (清音/休止)';
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const midiFloat = cents / 100.0;
  const midiInt = Math.round(midiFloat);
  const diff = Math.round((midiFloat - midiInt) * 100);
  const octave = Math.floor(midiInt / 12) - 1;
  const note = names[((midiInt % 12) + 12) % 12];
  const sign = diff >= 0 ? `+${diff}` : `${diff}`;
  const hz = 440.0 * Math.pow(2.0, (midiFloat - 69.0) / 12.0);
  return `${note}${octave} (${sign}c · ${hz.toFixed(1)} Hz)`;
}

export function decompilePacketAtIndex(parsed, packetIdx) {
  if (!parsed || !parsed.tracks || parsed.tracks.length === 0) return null;
  const maxPkt = parsed.tracks[0].packetCount - 1;
  const k = Math.max(0, Math.min(maxPkt, packetIdx | 0));
  const timeSec = (k * (parsed.frameDurationMs || 20)) / 1000.0;

  const trackPackets = parsed.tracks.map((tr) => {
    const off = tr.packetOffsets[k];
    const len = tr.packetLengths[k];
    const slice = tr.audioData.subarray(off, off + len);
    const toc = len > 0 ? decodeOpusTocByte(slice[0]) : null;
    const hexPreview = Array.from(slice.subarray(0, Math.min(16, len)))
      .map((b) => b.toString(16).toUpperCase().padStart(2, '0'))
      .join(' ');
    return {
      trackIndex: tr.trackIndex,
      label: tr.label,
      roleZh: tr.roleZh,
      channels: tr.channels,
      packetByteOffsetInFile: tr.audioOffset + off,
      packetByteLen: len,
      instantaneousKbps: Math.round(((len * 8) / ((parsed.frameDurationMs || 20) / 1000.0)) / 10) / 100,
      opusToc: toc,
      first16BytesHex: hexPreview,
    };
  });

  let timbFrame = null;
  const timb = parsed.decodedFeatures && parsed.decodedFeatures.TIMB;
  if (timb && k < timb.frameCount) {
    const cents = timb.f0Cents[k];
    const vByte = timb.voicing[k];
    timbFrame = {
      voiced: (vByte & 1) === 1,
      sibilant: ((vByte >>> 1) & 1) === 1,
      harmonicityPct: Math.round(((vByte >>> 2) / 63.0) * 100),
      f0Cents: cents,
      pitchNote: midiCentsToNoteName(cents),
      rmsDb: Math.round(((timb.rmsQ[k] / 255.0) * 85.0 - 85.0) * 10) / 10,
      formantBands: {
        chest_140_420Hz: timb.chest[k],
        f1Vowel_470_1170Hz: timb.f1[k],
        f2Presence_1220_3185Hz: timb.f2[k],
        airBreath_3230_9375Hz: timb.air[k],
      },
    };
  }

  return {
    containerMagic: parsed.magic,
    specVersion: parsed.version,
    crc32Verified: parsed.storedCrcHex,
    packetIndex: k,
    timeWindowSec: `${timeSec.toFixed(3)}s – ${(timeSec + 0.02).toFixed(3)}s`,
    audioTracksCount: parsed.tracks.length,
    featureLayersCount: parsed.features.length,
    opusPacketsAtTick: trackPackets,
    timbVocalFeatureAtTick: timbFrame || 'N/A (WEC0 纯最小单流档不含声纹特征层)',
  };
}

export function buildContainerManifestJson(parsed) {
  if (!parsed) return {};
  return {
    magic: parsed.magic,
    specVersion: parsed.version,
    fileBytes: parsed.fileByteLength,
    headerBytes: parsed.headerSizeBytes,
    crc32: parsed.storedCrcHex,
    crcValid: parsed.crcValid,
    sampleRate: parsed.sampleRate,
    totalSamples: parsed.totalSamples,
    durationSec: Number(parsed.durationSec.toFixed(4)),
    bpm: parsed.bpm,
    tierContract:
      parsed.magic === 'WEC0'
        ? 'Pure Minimal (1 Opus Stream, 0 Feature Layers, 32B Header)'
        : parsed.magic === 'WEC1'
        ? 'Dual-Track Timbre/Vocal + Accompaniment (1..2 Tracks)'
        : `Multi-Stem Master (${parsed.tracks.length} Audio Tracks + ${parsed.features.length} Feature Layers, >= 4+3 satisfied)`,
    audioTrackDirectory: parsed.tracks.map((t) => ({
      index: t.trackIndex,
      label: t.label,
      role: t.roleCode,
      roleZh: t.roleZh,
      channels: t.channels,
      preSkipSamples: t.preSkip,
      defaultGainDb: Number(t.defaultGainDb.toFixed(2)),
      packetCount: t.packetCount,
      indexBytes: t.indexByteLen,
      opusPayloadBytes: t.audioByteLen,
      actualKbps: Number((((t.audioByteLen * 8) / parsed.durationSec) / 1000).toFixed(2)),
    })),
    featureLayerDirectory: parsed.features.map((f) => ({
      index: f.layerIndex,
      fourCC: f.fourCC,
      title: f.title,
      encoding: f.encodingName,
      compressedBytes: f.compressedByteLen,
      uncompressedBytes: f.uncompressedByteLen,
      itemCount: f.itemCount,
      shortDesc: f.shortDesc,
    })),
  };
}

export function formatBinaryHexView(parsed, packetIdx) {
  if (!parsed) return '';
  const u8 = parsed.rawBytes;
  const lines = [];
  const hdrEnd = Math.min(u8.length, parsed.magic === 'WEC0' ? 64 : Math.min(256, parsed.payloadOffset || 192));
  lines.push(`=== [1] ${parsed.magic} Binary Header & Directory Region (0x0000 .. 0x${hdrEnd.toString(16).padStart(4, '0')}) ===`);
  for (let off = 0; off < hdrEnd; off += 16) {
    const slice = u8.subarray(off, Math.min(hdrEnd, off + 16));
    const hex = Array.from(slice)
      .map((b) => b.toString(16).toUpperCase().padStart(2, '0'))
      .join(' ')
      .padEnd(47, ' ');
    const asc = Array.from(slice)
      .map((b) => (b >= 32 && b <= 126 ? String.fromCharCode(b) : '.'))
      .join('');
    lines.push(`0x${off.toString(16).toUpperCase().padStart(6, '0')}  ${hex}  |${asc}|`);
  }

  const k = Math.max(0, Math.min((parsed.tracks[0]?.packetCount || 1) - 1, packetIdx | 0));
  lines.push('');
  lines.push(`=== [2] Active RFC 6716 Opus Raw Packets @ Packet #${k} (${(k * 0.02).toFixed(2)}s) ===`);
  for (const tr of parsed.tracks) {
    const pOff = tr.audioOffset + tr.packetOffsets[k];
    const pLen = tr.packetLengths[k];
    const showLen = Math.min(48, pLen);
    const slice = u8.subarray(pOff, pOff + showLen);
    const toc = pLen > 0 ? decodeOpusTocByte(slice[0]) : null;
    lines.push(
      `Track[${tr.trackIndex}] ${tr.label} (${tr.roleZh}) | FileOffset=0x${pOff.toString(16).toUpperCase()} | Len=${pLen}B | TOC=${toc ? `${toc.mode} ${toc.bandwidth}` : 'Empty'}`
    );
    for (let off = 0; off < showLen; off += 16) {
      const row = slice.subarray(off, Math.min(showLen, off + 16));
      const hex = Array.from(row)
        .map((b) => b.toString(16).toUpperCase().padStart(2, '0'))
        .join(' ')
        .padEnd(47, ' ');
      lines.push(`  +0x${off.toString(16).toUpperCase().padStart(4, '0')}  ${hex}`);
    }
  }
  return lines.join('\n');
}
