// V9.4.x Write-Once Immutable Chunk Streaming Mixer + Phase-Locked 15ms Live Tier Switching
// Zero AudioBuffer Reuse Race · Exact 0.25s (12,000-sample) Scheduling · ~0.38 MB Live PCM RAM

import { encodeStereoWav24Bit } from '../dsp/streaming-tier-engine.js';

export function dbToGain(db) {
  if (db <= -35.9) return 0.0;
  return Math.pow(10.0, db / 20.0);
}

const CHUNK_SAMPLES = 12000; // Exact 0.2500000000s @ 48,000 Hz (96 KB stereo PCM per chunk)
const LOOKAHEAD_SEC = 0.65;
const PUMP_INTERVAL_MS = 80;

export class StudioAudioMixer {
  constructor(store, initialEngine = null) {
    this.store = store;
    this.streamEngine = initialEngine;
    this.sampleRate = initialEngine ? initialEngine.sampleRate : 48000;
    this.numSamples = initialEngine ? initialEngine.totalSamples : 10097920;
    this.durationSec = this.numSamples / this.sampleRate;

    this.ctx = null;
    this.masterGain = null;

    this.currentGen = null;
    this.retiringGens = new Set();

    this.offsetSec = 0.0;
    this.anchorSongSec = 0.0;
    this.anchorCtxTime = 0.0;

    this.lastSyncMs = 0;
    this.syncTimer = null;
    this.rafId = null;

    this.scratchL = new Float32Array(CHUNK_SAMPLES);
    this.scratchR = new Float32Array(CHUNK_SAMPLES);
  }

  ensureContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx({ sampleRate: this.sampleRate });
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 1.0;
      this.masterGain.connect(this.ctx.destination);
    }
    return this.ctx;
  }

  getMixParams() {
    const st = this.store.getState();
    return {
      masterGain: dbToGain(st.masterGainDb ?? 0.0),
      stereoWidth: (st.stereoWidthPct ?? 100) / 100.0,
      channelMode: st.channelMode || 'stereo',
    };
  }

  setActiveEngine(newEngine, crossfadeIfPlaying = true) {
    if (!newEngine) return;
    this.streamEngine = newEngine;
    this.sampleRate = newEngine.sampleRate;
    this.numSamples = newEngine.totalSamples;
    this.durationSec = newEngine.durationSec;

    if (crossfadeIfPlaying && this.store.getState().isPlaying && this.ctx) {
      this._applyLiveCrossfadeImmediate();
    }
  }

  getLiveOffsetSec() {
    if (this.store.getState().isPlaying && this.ctx) {
      const elapsed = Math.max(0.0, this.ctx.currentTime - this.anchorCtxTime);
      return Math.min(this.durationSec, this.anchorSongSec + elapsed);
    }
    return this.offsetSec;
  }

  _pumpGeneration(gen) {
    if (!gen || gen.stopped || !this.ctx || !gen.engine) return;

    while (
      gen.nextGenSample < this.numSamples &&
      gen.startCtxTime + gen.scheduledSamples / this.sampleRate - this.ctx.currentTime < LOOKAHEAD_SEC
    ) {
      const rem = this.numSamples - gen.nextGenSample;
      const count = Math.min(CHUNK_SAMPLES, rem);

      gen.engine.synthesizeSampleRange(
        gen.nextGenSample,
        count,
        gen.mixParams,
        this.scratchL,
        this.scratchR
      );

      // Write-once immutable AudioBuffer per chunk (eliminates WebAudio shared-buffer overwrite races)
      const buf = this.ctx.createBuffer(2, count, this.sampleRate);
      if (count === CHUNK_SAMPLES) {
        buf.copyToChannel(this.scratchL, 0);
        buf.copyToChannel(this.scratchR, 1);
      } else {
        buf.copyToChannel(this.scratchL.subarray(0, count), 0);
        buf.copyToChannel(this.scratchR.subarray(0, count), 1);
      }

      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gen.busGain);

      const when = gen.startCtxTime + gen.scheduledSamples / this.sampleRate;
      src.onended = () => {
        try {
          src.disconnect();
        } catch (_) {}
        gen.sources.delete(src);
      };

      src.start(when);
      gen.sources.add(src);
      gen.scheduledSamples += count;
      gen.nextGenSample += count;
    }
  }

  _spawnGeneration(startSample, startCtxTime, fadeInSec = 0.005) {
    const ctx = this.ensureContext();
    const busGain = ctx.createGain();
    if (fadeInSec > 0.0) {
      busGain.gain.setValueAtTime(0.0, startCtxTime);
      busGain.gain.linearRampToValueAtTime(1.0, startCtxTime + fadeInSec);
    } else {
      busGain.gain.setValueAtTime(1.0, startCtxTime);
    }
    busGain.connect(this.masterGain);

    const gen = {
      engine: this.streamEngine,
      busGain,
      mixParams: this.getMixParams(),
      startSample,
      nextGenSample: startSample,
      scheduledSamples: 0,
      startCtxTime,
      sources: new Set(),
      timer: null,
      cleanupTimer: null,
      stopped: false,
    };

    this._pumpGeneration(gen);
    gen.timer = setInterval(() => this._pumpGeneration(gen), PUMP_INTERVAL_MS);
    return gen;
  }

  _retireGenerationCrossfade(gen, fadeStartCtxTime, fadeSec = 0.015) {
    if (!gen || gen.stopped) return;
    gen.stopped = true;
    if (gen.timer !== null) {
      clearInterval(gen.timer);
      gen.timer = null;
    }
    if (!this.ctx) {
      this._destroyGenerationImmediate(gen);
      return;
    }

    const t0 = Math.max(this.ctx.currentTime, fadeStartCtxTime);
    const t1 = t0 + fadeSec;
    try {
      gen.busGain.gain.cancelScheduledValues(t0);
      gen.busGain.gain.setValueAtTime(gen.busGain.gain.value, t0);
      gen.busGain.gain.linearRampToValueAtTime(0.0, t1);
    } catch (_) {}

    for (const s of gen.sources) {
      try {
        s.stop(t1 + 0.005);
      } catch (_) {}
    }

    this.retiringGens.add(gen);
    const delayMs = Math.max(25, Math.ceil((t1 - this.ctx.currentTime + 0.025) * 1000));
    gen.cleanupTimer = setTimeout(() => {
      this._destroyGenerationImmediate(gen);
    }, delayMs);
  }

  _destroyGenerationImmediate(gen) {
    if (!gen) return;
    gen.stopped = true;
    if (gen.timer !== null) {
      clearInterval(gen.timer);
      gen.timer = null;
    }
    if (gen.cleanupTimer !== null) {
      clearTimeout(gen.cleanupTimer);
      gen.cleanupTimer = null;
    }
    for (const s of gen.sources) {
      try {
        s.stop();
      } catch (_) {}
      try {
        s.disconnect();
      } catch (_) {}
    }
    gen.sources.clear();
    try {
      gen.busGain.disconnect();
    } catch (_) {}
    this.retiringGens.delete(gen);
  }

  _applyLiveCrossfadeImmediate() {
    if (!this.store.getState().isPlaying || !this.ctx || !this.streamEngine) return;
    this.lastSyncMs = performance.now();

    const handoffCtxTime = Math.max(this.ctx.currentTime + 0.014, this.anchorCtxTime);
    const elapsedSamples = Math.round((handoffCtxTime - this.anchorCtxTime) * this.sampleRate);
    const anchorSamples = Math.round(this.anchorSongSec * this.sampleRate);
    const handoffSample = Math.max(0, Math.min(this.numSamples, anchorSamples + elapsedSamples));
    if (handoffSample >= this.numSamples) return;

    const oldGen = this.currentGen;
    this.currentGen = this._spawnGeneration(handoffSample, handoffCtxTime, 0.015);
    if (oldGen) {
      this._retireGenerationCrossfade(oldGen, handoffCtxTime, 0.015);
    }
  }

  syncMixFromStore() {
    if (!this.store.getState().isPlaying || !this.ctx) return;
    const nowMs = performance.now();
    const elapsed = nowMs - this.lastSyncMs;

    if (elapsed >= 40) {
      if (this.syncTimer !== null) {
        clearTimeout(this.syncTimer);
        this.syncTimer = null;
      }
      this._applyLiveCrossfadeImmediate();
    } else if (this.syncTimer === null) {
      this.syncTimer = setTimeout(() => {
        this.syncTimer = null;
        this._applyLiveCrossfadeImmediate();
      }, 40 - elapsed);
    }
  }

  _startFreshAtSec(fromSec, crossfadePrev = false) {
    if (!this.streamEngine) return;
    const ctx = this.ensureContext();
    if (this.syncTimer !== null) {
      clearTimeout(this.syncTimer);
      this.syncTimer = null;
    }

    const clampedSec = Math.max(0.0, Math.min(this.durationSec - 0.02, fromSec));
    const startSample = Math.floor(clampedSec * this.sampleRate);
    const startCtxTime = ctx.currentTime + 0.015;

    if (this.currentGen) {
      if (crossfadePrev) {
        this._retireGenerationCrossfade(this.currentGen, startCtxTime, 0.012);
      } else {
        this._destroyGenerationImmediate(this.currentGen);
      }
      this.currentGen = null;
    }

    this.offsetSec = startSample / this.sampleRate;
    this.anchorSongSec = this.offsetSec;
    this.anchorCtxTime = startCtxTime;
    this.currentGen = this._spawnGeneration(startSample, startCtxTime, 0.006);
  }

  async play(fromSec = null) {
    if (!this.store.getState().isReady || !this.streamEngine) return;
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    const startOffset = fromSec !== null ? fromSec : this.offsetSec;
    this.store.update({ isPlaying: true, currentTime: startOffset });
    this._startFreshAtSec(startOffset, false);
    this._startClockLoop();
  }

  pause() {
    if (!this.store.getState().isPlaying) return;
    this.offsetSec = this.getLiveOffsetSec();
    this._stopAllImmediate();
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
      this._startFreshAtSec(clamped, true);
      this.store.update({ currentTime: clamped });
    } else {
      this.store.update({ currentTime: clamped });
    }
  }

  async exportCurrentMixWav(customName = null) {
    if (this.store.getState().isExportingWav || !this.streamEngine) return;
    const meta = this.store.getState().activeTierMeta;
    const tag = meta ? `${meta.versionCode}_${meta.tierName}` : 'V9.4.x';
    const filename = customName || `下个路口见_${tag}_纯数学48kHz_24bit.wav`;

    this.store.update({ isExportingWav: true });
    try {
      await new Promise((r) => setTimeout(r, 20));
      const [mixL, mixR] = this.streamEngine.renderFullSongStereo(this.getMixParams());
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

  _stopAllImmediate() {
    if (this.syncTimer !== null) {
      clearTimeout(this.syncTimer);
      this.syncTimer = null;
    }
    if (this.currentGen) {
      this._destroyGenerationImmediate(this.currentGen);
      this.currentGen = null;
    }
    for (const g of Array.from(this.retiringGens)) {
      this._destroyGenerationImmediate(g);
    }
    this.retiringGens.clear();
  }

  _startClockLoop() {
    this._stopClockLoop();
    const tick = () => {
      if (!this.store.getState().isPlaying || !this.ctx) return;
      if (this.currentGen) {
        this._pumpGeneration(this.currentGen);
      }
      const nowOffset = this.getLiveOffsetSec();
      if (nowOffset >= this.durationSec) {
        this._stopAllImmediate();
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
