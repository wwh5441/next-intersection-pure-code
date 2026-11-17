// WEC v2.0 Multi-Stem Live Crossfade Mixer & Multi-Format Exporter (.wec / Manifest JSON / 24-bit WAV)

import { encodeStereoWav24Bit } from '../dsp/wec-opus-engine.js';
import {
  buildContainerManifestJson,
  decompilePacketAtIndex,
} from '../../parsers/wec-plaintext-decompiler.js';

export function dbToGain(db) {
  if (db <= -35.9) return 0.0;
  return Math.pow(10.0, db / 20.0);
}

const CHUNK_SAMPLES = 9600; // 200ms chunk (10 x 20ms Opus frames)
const LOOKAHEAD_SEC = 0.55;
const PUMP_INTERVAL_MS = 75;

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
      stemStates: st.stemStates ? st.stemStates.map((s) => ({ ...s })) : [],
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

      gen.engine.synthesizeSampleRange(gen.nextGenSample, count, gen.mixParams, this.scratchL, this.scratchR);

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
        try { src.disconnect(); } catch (_) {}
        gen.sources.delete(src);
      };

      src.start(when);
      gen.sources.add(src);
      gen.scheduledSamples += count;
      gen.nextGenSample += count;
    }
  }

  _spawnGeneration(startSample, startCtxTime, fadeInSec = 0.006) {
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
      try { s.stop(t1 + 0.005); } catch (_) {}
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
    if (gen.timer !== null) { clearInterval(gen.timer); gen.timer = null; }
    if (gen.cleanupTimer !== null) { clearTimeout(gen.cleanupTimer); gen.cleanupTimer = null; }
    for (const s of gen.sources) {
      try { s.stop(); } catch (_) {}
      try { s.disconnect(); } catch (_) {}
    }
    gen.sources.clear();
    try { gen.busGain.disconnect(); } catch (_) {}
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
    if (elapsed >= 35) {
      if (this.syncTimer !== null) { clearTimeout(this.syncTimer); this.syncTimer = null; }
      this._applyLiveCrossfadeImmediate();
    } else if (this.syncTimer === null) {
      this.syncTimer = setTimeout(() => {
        this.syncTimer = null;
        this._applyLiveCrossfadeImmediate();
      }, 35 - elapsed);
    }
  }

  _startFreshAtSec(fromSec, crossfadePrev = false) {
    if (!this.streamEngine) return;
    const ctx = this.ensureContext();
    if (this.syncTimer !== null) { clearTimeout(this.syncTimer); this.syncTimer = null; }

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
    if (ctx.state === 'suspended') await ctx.resume();
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
    const maxPkt = (this.streamEngine?.parsed?.tracks?.[0]?.packetCount || 10519) - 1;
    const pktIdx = Math.min(maxPkt, Math.floor(clamped / 0.02));
    this.offsetSec = clamped;
    if (this.store.getState().isPlaying) {
      this._startFreshAtSec(clamped, true);
      this.store.update({ currentTime: clamped, inspectedPacketIdx: pktIdx });
    } else {
      this.store.update({ currentTime: clamped, inspectedPacketIdx: pktIdx });
    }
  }

  downloadRawWecFile() {
    if (!this.streamEngine) return;
    const { parsed } = this.streamEngine;
    const meta = this.store.getState().activeTierMeta;
    const fname = (meta && meta.wecFilename) || `output_${parsed.magic}.wec`;
    const blob = new Blob([parsed.rawBytes], { type: 'application/x-wec-audio' });
    this._triggerBlobDownload(blob, fname);
  }

  exportPlaintextJsonSnapshot() {
    if (!this.streamEngine) return;
    const { parsed } = this.streamEngine;
    const st = this.store.getState();
    const pktIdx = st.inspectedPacketIdx || 0;
    const snapshot = {
      manifest: buildContainerManifestJson(parsed),
      inspectedPacket: decompilePacketAtIndex(parsed, pktIdx),
    };
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json;charset=utf-8' });
    const base = ((st.activeTierMeta && st.activeTierMeta.wecFilename) || 'container.wec').replace(/\.wec$/i, '');
    this._triggerBlobDownload(blob, `${base}_Pkt${pktIdx}_结构透视.json`);
  }

  async exportCurrentMixWav() {
    if (this.store.getState().isExportingWav || !this.streamEngine) return;
    const st = this.store.getState();
    const meta = st.activeTierMeta;
    const tag = meta ? `${meta.magic}_${meta.name}` : this.streamEngine.magic;
    const sceneSuffix = st.activeSceneId && st.activeSceneId !== 'master' ? `_${st.activeSceneId}` : '';
    const filename = `下个路口见_${tag}${sceneSuffix}_48kHz_24bit.wav`;

    this.store.update({ isExportingWav: true });
    try {
      await new Promise((r) => setTimeout(r, 25));
      const wavBlob = encodeStereoWav24Bit(this.streamEngine, this.getMixParams());
      this._triggerBlobDownload(wavBlob, filename);
    } finally {
      this.store.update({ isExportingWav: false });
    }
  }

  _triggerBlobDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  _stopAllImmediate() {
    if (this.syncTimer !== null) { clearTimeout(this.syncTimer); this.syncTimer = null; }
    if (this.currentGen) { this._destroyGenerationImmediate(this.currentGen); this.currentGen = null; }
    for (const g of Array.from(this.retiringGens)) this._destroyGenerationImmediate(g);
    this.retiringGens.clear();
  }

  _startClockLoop() {
    this._stopClockLoop();
    const tick = () => {
      if (!this.store.getState().isPlaying || !this.ctx) return;
      if (this.currentGen) this._pumpGeneration(this.currentGen);
      const nowOffset = this.getLiveOffsetSec();
      if (nowOffset >= this.durationSec) {
        this._stopAllImmediate();
        this._stopClockLoop();
        this.offsetSec = this.durationSec;
        this.store.update({ isPlaying: false, currentTime: this.durationSec });
        return;
      }
      const patch = { currentTime: nowOffset };
      if (this.store.getState().followPlayheadFrame) {
        const maxPkt = (this.streamEngine?.parsed?.tracks?.[0]?.packetCount || 10519) - 1;
        patch.inspectedPacketIdx = Math.min(maxPkt, Math.floor(nowOffset / 0.02));
      }
      this.store.update(patch);
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
