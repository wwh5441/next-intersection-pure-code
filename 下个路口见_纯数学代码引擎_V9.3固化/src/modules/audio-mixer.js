// V9.3 6-Stem Conservative WebAudio Synchronized Studio Mixer & 24-bit WAV Bounce Engine

import { STEM_MANIFEST, encodeStereoWav24Bit } from '../dsp/song-renderer.js';

export function dbToGain(db) {
  if (db <= -35.9) return 0.0;
  return Math.pow(10.0, db / 20.0);
}

export class StudioAudioMixer {
  constructor(store) {
    this.store = store;
    this.ctx = null;
    this.masterGain = null;
    this.stemBuffers = new Map();
    this.gainNodes = new Map();
    this.activeSources = [];
    this.startedAtCtxTime = 0.0;
    this.offsetSec = 0.0;
    this.durationSec = 210.373;
    this.sampleRate = 48000;
    this.numSamples = 0;
    this.rafId = null;
  }

  ensureContext(sampleRate = 48000) {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx({ sampleRate });
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 1.0;
      this.masterGain.connect(this.ctx.destination);

      for (const m of STEM_MANIFEST) {
        const g = this.ctx.createGain();
        g.gain.value = 1.0;
        g.connect(this.masterGain);
        this.gainNodes.set(m.id, g);
      }
    }
    return this.ctx;
  }

  getLiveOffsetSec() {
    if (this.store.getState().isPlaying && this.ctx) {
      return Math.min(
        this.durationSec,
        this.offsetSec + Math.max(0.0, this.ctx.currentTime - this.startedAtCtxTime)
      );
    }
    return this.offsetSec;
  }

  loadSynthesizedStems(sampleRate, numSamples, stemsObj) {
    const ctx = this.ensureContext(sampleRate);
    this.sampleRate = sampleRate;
    this.numSamples = numSamples;
    this.durationSec = numSamples / sampleRate;

    this.stemBuffers.clear();
    for (const m of STEM_MANIFEST) {
      const [L, R] = stemsObj[m.id];
      const buf = ctx.createBuffer(2, numSamples, sampleRate);
      buf.copyToChannel(L, 0);
      buf.copyToChannel(R, 1);
      this.stemBuffers.set(m.id, buf);
    }
    this.syncGainsFromStore();
  }

  getEffectiveStemGains() {
    const { stemStates } = this.store.getState();
    const anySolo = STEM_MANIFEST.some((m) => stemStates[m.id]?.solo);
    const gains = {};
    for (const m of STEM_MANIFEST) {
      const st = stemStates[m.id];
      const audible = anySolo ? Boolean(st?.solo) : !st?.muted;
      gains[m.id] = audible ? dbToGain(st?.gainDb ?? 0.0) : 0.0;
    }
    return gains;
  }

  syncGainsFromStore() {
    if (!this.ctx) return;
    const gains = this.getEffectiveStemGains();
    const now = this.ctx.currentTime;
    for (const m of STEM_MANIFEST) {
      const node = this.gainNodes.get(m.id);
      if (node) {
        node.gain.setTargetAtTime(gains[m.id], now, 0.012);
      }
    }
  }

  async play(fromSec = null) {
    const st = this.store.getState();
    if (!st.isReady || this.stemBuffers.size === 0) return;

    const ctx = this.ensureContext(this.sampleRate);
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    this._stopSourcesOnly();
    const startOffset = Math.max(
      0.0,
      Math.min(this.durationSec - 0.05, fromSec !== null ? fromSec : this.offsetSec)
    );
    this.offsetSec = startOffset;
    this.startedAtCtxTime = ctx.currentTime;

    const when = ctx.currentTime + 0.015;
    this.activeSources = [];
    for (const m of STEM_MANIFEST) {
      const buf = this.stemBuffers.get(m.id);
      const gainNode = this.gainNodes.get(m.id);
      if (!buf || !gainNode) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gainNode);
      src.start(when, startOffset);
      this.activeSources.push(src);
    }

    this.syncGainsFromStore();
    this.store.update({ isPlaying: true, currentTime: startOffset });
    this._startClockLoop();
  }

  pause() {
    if (!this.store.getState().isPlaying) return;
    if (this.ctx) {
      this.offsetSec = Math.min(
        this.durationSec,
        this.offsetSec + Math.max(0.0, this.ctx.currentTime - this.startedAtCtxTime)
      );
    }
    this._stopSourcesOnly();
    this._stopClockLoop();
    this.store.update({ isPlaying: false, currentTime: this.offsetSec });
  }

  async togglePlay() {
    if (this.store.getState().isPlaying) {
      this.pause();
    } else {
      const cur = this.offsetSec >= this.durationSec - 0.2 ? 0.0 : this.offsetSec;
      await this.play(cur);
    }
  }

  async seek(targetSec) {
    const clamped = Math.max(0.0, Math.min(this.durationSec, Number(targetSec) || 0.0));
    this.offsetSec = clamped;
    if (this.store.getState().isPlaying) {
      await this.play(clamped);
    } else {
      this.store.update({ currentTime: clamped });
    }
  }

  async exportCurrentMixWav(filename = '下个路口见_V9.3_纯数学48kHz_24bit母带.wav') {
    if (this.stemBuffers.size === 0 || this.store.getState().isExportingWav) return;
    this.store.update({ isExportingWav: true });
    try {
      await new Promise((r) => setTimeout(r, 20));
      const gains = this.getEffectiveStemGains();
      const mixL = new Float32Array(this.numSamples);
      const mixR = new Float32Array(this.numSamples);
      for (const m of STEM_MANIFEST) {
        const g = gains[m.id];
        if (g === 0.0) continue;
        const buf = this.stemBuffers.get(m.id);
        if (!buf) continue;
        const cL = buf.getChannelData(0);
        const cR = buf.getChannelData(1);
        for (let i = 0; i < this.numSamples; i++) {
          mixL[i] += cL[i] * g;
          mixR[i] += cR[i] * g;
        }
      }
      const wavBlob = encodeStereoWav24Bit(mixL, mixR, this.sampleRate);
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } finally {
      this.store.update({ isExportingWav: false });
    }
  }

  _stopSourcesOnly() {
    for (const s of this.activeSources) {
      try {
        s.stop();
        s.disconnect();
      } catch (_) {
        // Ignore stopped buffer source
      }
    }
    this.activeSources = [];
  }

  _startClockLoop() {
    this._stopClockLoop();
    const tick = () => {
      if (!this.store.getState().isPlaying || !this.ctx) return;
      const nowOffset = this.offsetSec + Math.max(0.0, this.ctx.currentTime - this.startedAtCtxTime);
      if (nowOffset >= this.durationSec) {
        this._stopSourcesOnly();
        this._stopClockLoop();
        this.offsetSec = this.durationSec;
        this.store.update({ isPlaying: false, currentTime: this.durationSec });
        return;
      }
      this.store.update({ currentTime: nowOffset });
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  _stopClockLoop() {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }
}
