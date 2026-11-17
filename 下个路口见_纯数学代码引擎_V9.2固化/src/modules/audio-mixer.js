// 10-Track WebAudio Synchronized Studio Mixer Engine with Instant A/B Tier Caching

import { STEM_MANIFEST } from '../dsp/song-renderer.js';

function dbToGain(db) {
  if (db <= -35.9) return 0.0;
  return Math.pow(10.0, db / 20.0);
}

export class TenTrackAudioMixer {
  constructor(store) {
    this.store = store;
    this.ctx = null;
    this.masterGain = null;
    this.stemBuffers = new Map();
    this.tierCache = new Map(); // tierId -> { buffers: Map, renderTimeMs, peakAmplitude }
    this.gainNodes = new Map();
    this.activeSources = [];
    this.startedAtCtxTime = 0.0;
    this.offsetSec = 0.0;
    this.durationSec = 210.373;
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

  hasCachedTier(tierId) {
    return this.tierCache.has(tierId);
  }

  getCachedTierMeta(tierId) {
    return this.tierCache.get(tierId) || null;
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

  loadSynthesizedStems(sampleRate, numSamples, stemsObj, tierId = 'expressive', meta = {}) {
    const ctx = this.ensureContext(sampleRate);
    this.durationSec = numSamples / sampleRate;

    const buffers = new Map();
    for (const m of STEM_MANIFEST) {
      const [L, R] = stemsObj[m.id];
      const buf = ctx.createBuffer(2, numSamples, sampleRate);
      buf.copyToChannel(L, 0);
      buf.copyToChannel(R, 1);
      buffers.set(m.id, buf);
    }

    this.tierCache.set(tierId, {
      buffers,
      renderTimeMs: meta.renderTimeMs || 0,
      peakAmplitude: meta.peakAmplitude || 0.94,
    });

    this.activateCachedTier(tierId);
  }

  async activateCachedTier(tierId) {
    const entry = this.tierCache.get(tierId);
    if (!entry) return false;
    const wasPlaying = this.store.getState().isPlaying;
    const liveSec = this.getLiveOffsetSec();

    this.stemBuffers = entry.buffers;
    this.syncGainsFromStore();

    if (wasPlaying) {
      await this.play(liveSec);
    }
    return true;
  }

  syncGainsFromStore() {
    if (!this.ctx) return;
    const { stemStates } = this.store.getState();
    const anySolo = STEM_MANIFEST.some((m) => stemStates[m.id]?.solo);
    const now = this.ctx.currentTime;

    for (const m of STEM_MANIFEST) {
      const st = stemStates[m.id];
      const node = this.gainNodes.get(m.id);
      if (!st || !node) continue;
      const audible = anySolo ? st.solo : !st.muted;
      const target = audible ? dbToGain(st.gainDb) : 0.0;
      node.gain.setTargetAtTime(target, now, 0.012);
    }
  }

  async play(fromSec = null) {
    const st = this.store.getState();
    if (!st.isReady || this.stemBuffers.size === 0) return;

    const ctx = this.ensureContext();
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

  _stopSourcesOnly() {
    for (const s of this.activeSources) {
      try {
        s.stop();
        s.disconnect();
      } catch (_) {
        // Ignore already stopped buffer source
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
