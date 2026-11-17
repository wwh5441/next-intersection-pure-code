// Native RFC 6716 Opus Multi-Stem Audio Engine for WEC0 / WEC1 / WEC2
// Uses browser native C++ libopus via OfflineAudioContext.decodeAudioData + Sample-Accurate Multi-Stem Summing

import { wrapTrackOpusPacketsAsOggBuffer } from '../../parsers/wec-ogg-remuxer.js';

let sharedDecodeCtx = null;
function getOfflineDecodeContext(sampleRate = 48000) {
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!sharedDecodeCtx || sharedDecodeCtx.sampleRate !== sampleRate) {
    sharedDecodeCtx = new OfflineCtx(2, 128, sampleRate);
  }
  return sharedDecodeCtx;
}

function quantizeFloat32ToScaledInt16(f32, length) {
  let maxAbs = 1e-6;
  const n = Math.min(f32.length, length);
  for (let i = 0; i < n; i++) {
    const a = Math.abs(f32[i]);
    if (a > maxAbs) maxAbs = a;
  }
  const scaleToI16 = 32767.0 / maxAbs;
  const scaleToF32 = maxAbs / 32767.0;
  const out = new Int16Array(length);
  for (let i = 0; i < n; i++) {
    out[i] = Math.round(f32[i] * scaleToI16);
  }
  return { pcmI16: out, scale: scaleToF32 };
}

async function decodeSingleOpusTrack(track, totalSamples, sampleRate) {
  const oggBuffer = wrapTrackOpusPacketsAsOggBuffer(track, totalSamples, sampleRate);
  const ctx = getOfflineDecodeContext(sampleRate);
  const audioBuf = await ctx.decodeAudioData(oggBuffer);

  // Detect if browser already stripped OpusHead preSkip
  const rawFullLen = track.packetCount * Math.round((sampleRate * (track.frameDurationMs || 20)) / 1000);
  const startSkip = audioBuf.length >= rawFullLen ? track.preSkip : 0;
  const gainLinear = Math.pow(10.0, (track.defaultGainDb || 0.0) / 20.0);

  const ch0Raw = audioBuf.getChannelData(0).subarray(startSkip, startSkip + totalSamples);
  const q0 = quantizeFloat32ToScaledInt16(ch0Raw, totalSamples);
  q0.scale *= gainLinear;

  let q1 = null;
  if (track.channels >= 2 && audioBuf.numberOfChannels >= 2) {
    const ch1Raw = audioBuf.getChannelData(1).subarray(startSkip, startSkip + totalSamples);
    q1 = quantizeFloat32ToScaledInt16(ch1Raw, totalSamples);
    q1.scale *= gainLinear;
  }

  return {
    trackIndex: track.trackIndex,
    label: track.label,
    roleCode: track.roleCode,
    roleZh: track.roleZh,
    color: track.color,
    channels: track.channels,
    leftI16: q0.pcmI16,
    leftScale: q0.scale,
    rightI16: q1 ? q1.pcmI16 : q0.pcmI16,
    rightScale: q1 ? q1.scale : q0.scale,
  };
}

export async function createWecOpusEngineAsync(parsed) {
  const decodedStems = await Promise.all(
    parsed.tracks.map((tr) => decodeSingleOpusTrack(tr, parsed.totalSamples, parsed.sampleRate))
  );

  let pcmBytes = 0;
  for (const st of decodedStems) {
    pcmBytes += st.leftI16.byteLength + (st.channels >= 2 ? st.rightI16.byteLength : 0);
  }

  const liveLevelsDb = new Float32Array(decodedStems.length).fill(-90.0);

  function synthesizeSampleRange(startSample, count, mixParams, outL, outR) {
    outL.fill(0.0, 0, count);
    outR.fill(0.0, 0, count);
    if (startSample >= parsed.totalSamples || count <= 0) return;

    const validCount = Math.min(count, parsed.totalSamples - startSample);
    const stemStates = mixParams && mixParams.stemStates ? mixParams.stemStates : null;
    const anySolo = stemStates ? stemStates.some((s) => s && s.solo) : false;

    for (let ti = 0; ti < decodedStems.length; ti++) {
      const st = decodedStems[ti];
      const ctrl = stemStates && stemStates[ti] ? stemStates[ti] : null;
      const isActive = anySolo ? !!(ctrl && ctrl.solo) : !(ctrl && ctrl.muted);
      if (!isActive) {
        liveLevelsDb[ti] = -90.0;
        continue;
      }

      const faderDb = ctrl && typeof ctrl.gainDb === 'number' ? ctrl.gainDb : 0.0;
      if (faderDb <= -35.9) {
        liveLevelsDb[ti] = -90.0;
        continue;
      }
      const faderLin = faderDb === 0.0 ? 1.0 : Math.pow(10.0, faderDb / 20.0);
      const pan = ctrl && typeof ctrl.pan === 'number' ? Math.max(-1, Math.min(1, ctrl.pan)) : 0.0;
      const panL = pan <= 0 ? 1.0 : 1.0 - pan;
      const panR = pan >= 0 ? 1.0 : 1.0 + pan;

      const sL = st.leftScale * faderLin * panL;
      const sR = st.rightScale * faderLin * panR;
      const srcL = st.leftI16;
      const srcR = st.rightI16;

      let sumSq = 0.0;
      for (let i = 0; i < validCount; i++) {
        const idx = startSample + i;
        const vl = srcL[idx] * sL;
        const vr = srcR[idx] * sR;
        outL[i] += vl;
        outR[i] += vr;
        sumSq += 0.5 * (vl * vl + vr * vr);
      }
      const rms = Math.sqrt(sumSq / Math.max(1, validCount));
      liveLevelsDb[ti] = rms > 1e-5 ? Math.max(-80.0, 20.0 * Math.log10(rms)) : -90.0;
    }

    const masterGain = mixParams && typeof mixParams.masterGain === 'number' ? mixParams.masterGain : 1.0;
    const width = mixParams && typeof mixParams.stereoWidth === 'number' ? mixParams.stereoWidth : 1.0;
    const chMode = (mixParams && mixParams.channelMode) || 'stereo';

    for (let i = 0; i < validCount; i++) {
      let l = outL[i] * masterGain;
      let r = outR[i] * masterGain;
      if (width !== 1.0 || chMode !== 'stereo') {
        const mid = 0.5 * (l + r);
        const side = 0.5 * (l - r) * width;
        if (chMode === 'mono' || chMode === 'mid') {
          l = mid;
          r = mid;
        } else if (chMode === 'side') {
          l = side;
          r = -side;
        } else {
          l = mid + side;
          r = mid - side;
        }
      }
      // Soft transparent safety limiter above +/-0.985 to prevent DAC clipping on +6dB stem boosts
      if (l > 0.985) l = 0.985 + 0.015 * Math.tanh((l - 0.985) / 0.015);
      else if (l < -0.985) l = -0.985 + 0.015 * Math.tanh((l + 0.985) / 0.015);
      if (r > 0.985) r = 0.985 + 0.015 * Math.tanh((r - 0.985) / 0.015);
      else if (r < -0.985) r = -0.985 + 0.015 * Math.tanh((r + 0.985) / 0.015);
      outL[i] = l;
      outR[i] = r;
    }
  }

  return {
    parsed,
    magic: parsed.magic,
    family: parsed.family,
    sampleRate: parsed.sampleRate,
    totalSamples: parsed.totalSamples,
    durationSec: parsed.durationSec,
    bpm: parsed.bpm,
    stems: decodedStems,
    liveLevelsDb,
    pcmRamMb: Math.round((pcmBytes / (1024 * 1024)) * 100) / 100,
    synthesizeSampleRange,
  };
}

export function encodeStereoWav24Bit(engine, mixParams, onProgress = null) {
  const sr = engine.sampleRate;
  const totalSamples = engine.totalSamples;
  const bytesPerSample = 3;
  const dataBytes = totalSamples * 2 * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataBytes);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);

  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) u8[off + i] = s.charCodeAt(i);
  };
  writeStr(0, 'RIFF');
  dv.setUint32(4, 36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 2, true);
  dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * 2 * bytesPerSample, true);
  dv.setUint16(32, 2 * bytesPerSample, true);
  dv.setUint16(34, 24, true);
  writeStr(36, 'data');
  dv.setUint32(40, dataBytes, true);

  const block = 24000;
  const tmpL = new Float32Array(block);
  const tmpR = new Float32Array(block);
  let pos = 44;

  for (let s = 0; s < totalSamples; s += block) {
    const cnt = Math.min(block, totalSamples - s);
    engine.synthesizeSampleRange(s, cnt, mixParams, tmpL, tmpR);
    for (let i = 0; i < cnt; i++) {
      const sl = Math.max(-1, Math.min(1, tmpL[i]));
      const srVal = Math.max(-1, Math.min(1, tmpR[i]));
      let il = Math.round(sl * 8388607);
      let ir = Math.round(srVal * 8388607);
      if (il < 0) il += 16777216;
      if (ir < 0) ir += 16777216;
      u8[pos] = il & 0xff;
      u8[pos + 1] = (il >>> 8) & 0xff;
      u8[pos + 2] = (il >>> 16) & 0xff;
      u8[pos + 3] = ir & 0xff;
      u8[pos + 4] = (ir >>> 8) & 0xff;
      u8[pos + 5] = (ir >>> 16) & 0xff;
      pos += 6;
    }
    if (onProgress) onProgress(Math.min(1.0, (s + cnt) / totalSamples));
  }
  return new Blob([buf], { type: 'audio/wav' });
}
