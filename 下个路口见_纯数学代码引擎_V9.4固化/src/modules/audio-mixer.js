// V9.4 Real-Time Sliding-Window Streaming Mixer (Write-Once Immutable Chunks + Phase-Locked 15ms Crossfade)
// Zero AudioBuffer Reuse Race · Exact 0.25s (12,000-sample) IEEE-754 Drift-Free Scheduling · ~0.38 MB Live PCM RAM

import { STEM_MANIFEST, encodeStereoWav24Bit } from '../dsp/song-renderer.js';
import { createStreamingFrameEngine } from '../dsp/streaming-frame-engine.js';

export function dbToGain(db) {
  if (db <= -35.9) return 0.0;
  return Math.pow(10.0, db / 20.0);
}

const CHUNK_SAMPLES = 12000; // Exact 0.2500000000s @ 48,000 Hz (zero float drift; 96 KB stereo PCM per chunk)
const LOOKAHEAD_SEC = 0.65;  // ~3 chunks in flight (~0.28 MB AudioBuffers + 0.10 MB scratch = ~0.38 MB live RAM)
const PUMP_INTERVAL_MS = 80;

export class StudioAudioMixer {
  constructor(store) {
    this.store = store;
    this.streamEngine = createStreamingFrameEngine();
    this.sampleRate = this.streamEngine.sampleRate;
    this.numSamples = this.streamEngine.totalSamples;
    this.durationSec = this.streamEngine.durationSec;

    this.ctx = null;
    this.masterGain = null;

    this.currentGen = null;
    this.retiringGens = new Set();

    this.offsetSec = 0.0;
    this.anchorSongSec = 0.0;
    this.anchorCtxTime = 0.0;

    this.lastGainSyncMs = 0;
    this.gainSyncTimer = null;
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

  getEffectiveGainsArray() {
    const { stemStates } = this.store.getState();
    const anySolo = STEM_MANIFEST.some((m) => stemStates[m.id]?.solo);
    return STEM_MANIFEST.map((m) => {
      const st = stemStates[m.id];
      const audible = anySolo ? Boolean(st?.solo) : !st?.muted;
      return audible ? dbToGain(st?.gainDb ?? 0.0) : 0.0;
    });
  }

  getLiveOffsetSec() {
    if (this.store.getState().isPlaying && this.ctx) {
      const elapsed = Math.max(0.0, this.ctx.currentTime - this.anchorCtxTime);
      return Math.min(this.durationSec, this.anchorSongSec + elapsed);
    }
    return this.offsetSec;
  }

  _pumpGeneration(gen) {
    if (!gen || gen.stopped || !this.ctx) return;

    while (
      gen.nextGenSample < this.numSamples &&
      gen.startCtxTime + gen.scheduledSamples / this.sampleRate - this.ctx.currentTime < LOOKAHEAD_SEC
    ) {
      const rem = this.numSamples - gen.nextGenSample;
      const count = Math.min(CHUNK_SAMPLES, rem);

      this.streamEngine.synthesizeSampleRange(
        gen.nextGenSample,
        count,
        gen.gains6,
        this.scratchL,
        this.scratchR
      );

      // CRITICAL: Allocate a fresh write-once AudioBuffer per chunk BEFORE src.buffer acquisition.
      // Never mutate an already-acquired AudioBuffer via copyToChannel!
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
        } catch (_) {
          // Ignore already disconnected node
        }
        gen.sources.delete(src);
      };

      src.start(when);
      gen.sources.add(src);
      gen.scheduledSamples += count;
      gen.nextGenSample += count;
    }
  }

  _spawnGeneration(startSample, startCtxTime, fadeInSec = 0.004) {
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
      busGain,
      gains6: this.getEffectiveGainsArray(),
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
    } catch (_) {
      // Fallback immediate teardown
    }

    for (const s of gen.sources) {
      try {
        s.stop(t1 + 0.005);
      } catch (_) {
        // Ignore already stopped source
      }
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

  _applyLiveGainCrossfade() {
    if (!this.store.getState().isPlaying || !this.ctx || !this.currentGen) return;
    this.lastGainSyncMs = performance.now();

    // Phase-locked handoff: anchorSongSec & anchorCtxTime stay untouched for 0.000ms timeline drift!
    const handoffCtxTime = Math.max(this.ctx.currentTime + 0.014, this.anchorCtxTime);
    const elapsedSamples = Math.round((handoffCtxTime - this.anchorCtxTime) * this.sampleRate);
    const anchorSamples = Math.round(this.anchorSongSec * this.sampleRate);
    const handoffSample = Math.max(0, Math.min(this.numSamples, anchorSamples + elapsedSamples));
    if (handoffSample >= this.numSamples) return;

    const oldGen = this.currentGen;
    this.currentGen = this._spawnGeneration(handoffSample, handoffCtxTime, 0.015);
    this._retireGenerationCrossfade(oldGen, handoffCtxTime, 0.015);
  }

  syncGainsFromStore() {
    if (!this.store.getState().isPlaying || !this.ctx) return;
    const nowMs = performance.now();
    const elapsedMs = nowMs - this.lastGainSyncMs;

    if (elapsedMs >= 40) {
      if (this.gainSyncTimer !== null) {
        clearTimeout(this.gainSyncTimer);
        this.gainSyncTimer = null;
      }
      this._applyLiveGainCrossfade();
    } else if (this.gainSyncTimer === null) {
      this.gainSyncTimer = setTimeout(() => {
        this.gainSyncTimer = null;
        this._applyLiveGainCrossfade();
      }, 40 - elapsedMs);
    }
  }

  _startFreshAtSec(fromSec, crossfadePrev = false) {
    const ctx = this.ensureContext();
    if (this.gainSyncTimer !== null) {
      clearTimeout(this.gainSyncTimer);
      this.gainSyncTimer = null;
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
    if (!this.store.getState().isReady) return;
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

  async exportCurrentMixWav(filename = '下个路口见_V9.4_纯数学48kHz_24bit母带.wav') {
    if (this.store.getState().isExportingWav) return;
    this.store.update({ isExportingWav: true });
    try {
      await new Promise((r) => setTimeout(r, 20));
      const gains6 = this.getEffectiveGainsArray();
      const [mixL, mixR] = this.streamEngine.renderLinearMixFullSong(gains6);
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
    if (this.gainSyncTimer !== null) {
      clearTimeout(this.gainSyncTimer);
      this.gainSyncTimer = null;
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
