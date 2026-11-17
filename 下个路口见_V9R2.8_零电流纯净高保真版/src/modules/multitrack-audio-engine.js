// 11-Track WebAudio Mixer, Spectrum Analyzer & 48kHz WAV Exporter (`src/modules/multitrack-audio-engine.js`)
export class MultiTrackAudioEngine {
  constructor(layers, onEnded) {
    this.layers = layers;
    this.onEnded = onEnded;
    this.ctx = null;
    this.masterGain = null;
    this.analyser = null;
    this.gainNodes = new Map();
    this.sourceNodes = [];
    this.profileBuffers = new Map(); // profileId -> { sampleRate, numSamples, durationSec, audioBuffers: Map }
    this.activeProfileId = null;
    this.playing = false;
    this.startOffsetSec = 0.0;
    this.ctxStartTime = 0.0;
  }

  ensureContext(sampleRate = 48000) {
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx({ sampleRate });
      this.masterGain = this.ctx.createGain();
      // -0.35 dB headroom ceiling prevents DAC inter-sample clipping at 1.0297 true peak
      this.masterGain.gain.value = 0.96;
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.78;
      this.masterGain.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);

      for (const l of this.layers) {
        const g = this.ctx.createGain();
        g.gain.value = 1.0;
        g.connect(this.masterGain);
        this.gainNodes.set(l.id, g);
      }
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  hasCachedProfile(profileId) {
    return this.profileBuffers.has(profileId);
  }

  storeSynthesizedProfile(profileId, { sampleRate, numSamples, layers }) {
    const rawChannels = new Map();

    for (const l of this.layers) {
      const chArrs = layers[l.id];
      if (!chArrs) continue;
      const floatChs = [];
      for (let c = 0; c < chArrs.length; c++) {
        floatChs.push(chArrs[c] instanceof Float32Array ? chArrs[c] : new Float32Array(chArrs[c]));
      }
      rawChannels.set(l.id, floatChs);
    }

    this.profileBuffers.set(profileId, {
      sampleRate,
      numSamples,
      durationSec: numSamples / sampleRate,
      audioBuffers: null,
      rawChannels,
    });
  }

  ensureProfileAudioBuffers(entry, ctx) {
    if (entry.audioBuffers) return entry.audioBuffers;
    const audioBuffers = new Map();
    for (const [layerId, floatChs] of entry.rawChannels.entries()) {
      const ab = ctx.createBuffer(floatChs.length, entry.numSamples, entry.sampleRate);
      for (let c = 0; c < floatChs.length; c++) {
        ab.copyToChannel(floatChs[c], c);
      }
      audioBuffers.set(layerId, ab);
    }
    entry.audioBuffers = audioBuffers;
    return audioBuffers;
  }

  activateProfile(profileId, layerParams) {
    if (!this.profileBuffers.has(profileId)) return false;
    const wasPlaying = this.playing;
    const pos = this.getCurrentTime();
    this.stopSourcesOnly();
    this.activeProfileId = profileId;
    this.applyLayerMix(layerParams);
    this.startOffsetSec = pos;
    if (wasPlaying) {
      this.play(pos);
    }
    return true;
  }

  applyLayerMix(layerParams) {
    if (!this.ctx) return;
    const anySolo = Object.values(layerParams).some((p) => p.solo);
    const now = this.ctx.currentTime;
    for (const l of this.layers) {
      const p = layerParams[l.id] || { gain: 1.0, mute: false, solo: false };
      const audible = anySolo ? p.solo && !p.mute : !p.mute;
      const targetGain = audible ? p.gain : 0.0;
      const node = this.gainNodes.get(l.id);
      if (node) {
        node.gain.setTargetAtTime(targetGain, now, 0.012);
      }
    }
  }

  stopSourcesOnly() {
    for (const src of this.sourceNodes) {
      try {
        src.onended = null;
        src.stop();
        src.disconnect();
      } catch (_) {}
    }
    this.sourceNodes = [];
    this.playing = false;
  }

  play(fromSec = null) {
    const entry = this.profileBuffers.get(this.activeProfileId);
    if (!entry) return;
    const ctx = this.ensureContext(entry.sampleRate);
    const audioBuffers = this.ensureProfileAudioBuffers(entry, ctx);
    this.stopSourcesOnly();

    const offset = Math.max(0.0, Math.min(entry.durationSec - 0.02, fromSec ?? this.startOffsetSec));
    this.startOffsetSec = offset;
    this.ctxStartTime = ctx.currentTime;
    this.playing = true;

    let first = true;
    for (const l of this.layers) {
      const buf = audioBuffers.get(l.id);
      const gNode = this.gainNodes.get(l.id);
      if (!buf || !gNode) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gNode);
      if (first) {
        first = false;
        src.onended = () => {
          if (this.playing) {
            this.playing = false;
            this.startOffsetSec = 0.0;
            if (this.onEnded) this.onEnded();
          }
        };
      }
      src.start(0, offset);
      this.sourceNodes.push(src);
    }
  }

  pause() {
    if (!this.playing) return;
    this.startOffsetSec = this.getCurrentTime();
    this.stopSourcesOnly();
  }

  stop() {
    this.stopSourcesOnly();
    this.startOffsetSec = 0.0;
  }

  seek(sec) {
    const entry = this.profileBuffers.get(this.activeProfileId);
    const dur = entry ? entry.durationSec : 210.373;
    const clamped = Math.max(0.0, Math.min(dur - 0.02, sec));
    if (this.playing) {
      this.play(clamped);
    } else {
      this.startOffsetSec = clamped;
    }
  }

  getCurrentTime() {
    const entry = this.profileBuffers.get(this.activeProfileId);
    const dur = entry ? entry.durationSec : 210.373;
    if (!this.playing || !this.ctx) return this.startOffsetSec;
    return Math.min(dur, this.startOffsetSec + (this.ctx.currentTime - this.ctxStartTime));
  }

  getSpectrumBytes(targetUint8) {
    if (!this.analyser) {
      targetUint8.fill(0);
      return;
    }
    this.analyser.getByteFrequencyData(targetUint8);
  }

  getInstantLayerRms(layerParams) {
    const out = {};
    const entry = this.profileBuffers.get(this.activeProfileId);
    if (!entry || !this.playing) {
      for (const l of this.layers) out[l.id] = 0.0;
      return out;
    }
    const sIdx = Math.floor(this.getCurrentTime() * entry.sampleRate);
    const anySolo = Object.values(layerParams).some((p) => p.solo);
    const win = 1024;

    for (const l of this.layers) {
      const p = layerParams[l.id] || { gain: 1.0, mute: false, solo: false };
      const audible = anySolo ? p.solo && !p.mute : !p.mute;
      if (!audible) {
        out[l.id] = 0.0;
        continue;
      }
      const chs = entry.rawChannels.get(l.id);
      if (!chs || !chs.length) {
        out[l.id] = 0.0;
        continue;
      }
      const c0 = chs[0];
      const end = Math.min(c0.length, sIdx + win);
      let sumSq = 0.0;
      let cnt = 0;
      for (let i = Math.max(0, sIdx); i < end; i += 4) {
        const v = c0[i] * p.gain;
        sumSq += v * v;
        cnt++;
      }
      out[l.id] = cnt > 0 ? Math.sqrt(sumSq / cnt) : 0.0;
    }
    return out;
  }

  exportCurrentMixWav16(layerParams) {
    const entry = this.profileBuffers.get(this.activeProfileId);
    if (!entry) return null;
    const { sampleRate, numSamples, rawChannels } = entry;
    const mixL = new Float32Array(numSamples);
    const mixR = new Float32Array(numSamples);
    const anySolo = Object.values(layerParams).some((p) => p.solo);

    for (const l of this.layers) {
      const p = layerParams[l.id] || { gain: 1.0, mute: false, solo: false };
      const audible = anySolo ? p.solo && !p.mute : !p.mute;
      if (!audible || p.gain <= 0.0) continue;
      const chs = rawChannels.get(l.id);
      if (!chs) continue;
      const g = p.gain * 0.96;
      const cL = chs[0];
      const cR = chs.length > 1 ? chs[1] : chs[0];
      for (let i = 0; i < numSamples; i++) {
        mixL[i] += cL[i] * g;
        mixR[i] += cR[i] * g;
      }
    }

    const dataBytes = numSamples * 4;
    const buf = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buf);
    const writeStr = (off, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataBytes, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 2, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 4, true);
    view.setUint16(32, 4, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, dataBytes, true);

    let off = 44;
    for (let i = 0; i < numSamples; i++) {
      const sL = Math.max(-1.0, Math.min(1.0, mixL[i]));
      const sR = Math.max(-1.0, Math.min(1.0, mixR[i]));
      view.setInt16(off, sL < 0 ? sL * 32768 : sL * 32767, true);
      view.setInt16(off + 2, sR < 0 ? sR * 32768 : sR * 32767, true);
      off += 4;
    }
    return new Blob([buf], { type: 'audio/wav' });
  }
}
