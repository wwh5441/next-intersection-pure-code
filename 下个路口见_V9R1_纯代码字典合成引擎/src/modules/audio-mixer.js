// V9R1 10-Track Low-RAM WebAudio Mixer & Worker Synthesis Controller (`src/modules/audio-mixer.js`)
import { STEM_MANIFEST, renderV9r1Song } from '../dsp/v9r1-renderer.js';
import { studioStore } from '../state/studio-store.js';

class V9r1AudioMixer {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.masterAnalyser = null;
    this.stemNodes = new Map(); // stemId -> { gainNode, analyser, meterBuf }
    this.sourceNodes = new Map();
    this.tierCache = new Map(); // tierId -> { audioBuffers: Map, envelopes: Map, stats }
    this.worker = null;
    this.reqCounter = 0;
    this.activeReqId = 0;

    this.startWallCtxTime = 0;
    this.startSongOffset = 0;
    this.rafId = 0;
  }

  ensureContext() {
    if (this.ctx) return this.ctx;
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioCtx({ sampleRate: 48000 });
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 1.0;

    this.masterAnalyser = this.ctx.createAnalyser();
    this.masterAnalyser.fftSize = 1024;
    this.masterMeterBuf = new Float32Array(1024);

    this.masterGain.connect(this.masterAnalyser);
    this.masterAnalyser.connect(this.ctx.destination);

    for (const m of STEM_MANIFEST) {
      const gainNode = this.ctx.createGain();
      const analyser = this.ctx.createAnalyser();
      analyser.fftSize = 512;
      gainNode.connect(analyser);
      analyser.connect(this.masterGain);
      this.stemNodes.set(m.id, {
        gainNode,
        analyser,
        meterBuf: new Float32Array(512),
      });
    }
    this.syncStemGains();
    return this.ctx;
  }

  computeStemEnvelope20ms(chs, sampleRate, numSamples) {
    const win = Math.max(1, Math.round(sampleRate * 0.02));
    const nFrames = Math.ceil(numSamples / win);
    const env = new Float32Array(nFrames);
    for (let f = 0; f < nFrames; f++) {
      const s0 = f * win;
      const s1 = Math.min(numSamples, s0 + win);
      let sq = 0.0;
      let cnt = 0;
      for (let c = 0; c < chs.length; c++) {
        const arr = chs[c];
        for (let i = s0; i < s1; i += 4) {
          const v = arr[i];
          sq += v * v;
          cnt++;
        }
      }
      env[f] = Math.sqrt(sq / Math.max(1, cnt));
    }
    return env;
  }

  buildTierBuffers(payload) {
    const { sampleRate, numSamples, stems } = payload;
    const audioBuffers = new Map();
    const envelopes = new Map();

    for (const m of STEM_MANIFEST) {
      const chs = stems[m.id];
      const numCh = chs.length;
      let buf;
      if (typeof AudioBuffer === 'function') {
        buf = new AudioBuffer({
          length: numSamples,
          numberOfChannels: numCh,
          sampleRate,
        });
      } else {
        const ctx = this.ensureContext();
        buf = ctx.createBuffer(numCh, numSamples, sampleRate);
      }
      for (let c = 0; c < numCh; c++) {
        buf.copyToChannel(chs[c], c);
      }
      audioBuffers.set(m.id, buf);
      envelopes.set(m.id, this.computeStemEnvelope20ms(chs, sampleRate, numSamples));
    }

    return {
      audioBuffers,
      envelopes,
      stats: {
        tierId: payload.tierId,
        sampleRate: payload.sampleRate,
        numSamples: payload.numSamples,
        durationSec: payload.durationSec,
        renderTimeMs: payload.renderTimeMs,
        peakAmplitude: payload.peakAmplitude,
        vocalLabel: payload.vocalProfile.shortLabel,
        bandLabel: payload.bandProfile.label,
      },
    };
  }

  async synthesizeViaWorkerOrMain(tierId, reqId) {
    const inlineWorkerSrc = window.__V9R1_WORKER_SOURCE__;
    const canUseModuleWorker =
      (Boolean(inlineWorkerSrc) || (!window.__V9R1_SINGLE_HTML__ && window.location.protocol !== 'file:')) &&
      typeof Worker === 'function';

    if (canUseModuleWorker) {
      try {
        if (!this.worker) {
          if (inlineWorkerSrc) {
            const blob = new Blob([inlineWorkerSrc], { type: 'application/javascript' });
            this.worker = new Worker(URL.createObjectURL(blob));
          } else {
            this.worker = new Worker(new URL('../dsp/v9r1-worker.js', import.meta.url), {
              type: 'module',
            });
          }
        }
        return await new Promise((resolve, reject) => {
          const handler = (ev) => {
            const msg = ev.data || {};
            if (msg.reqId !== reqId) return;
            if (msg.type === 'progress') {
              if (this.activeReqId === reqId) {
                studioStore.setState({
                  renderPct: msg.pct,
                  renderText: msg.text,
                });
              }
            } else if (msg.type === 'done') {
              this.worker.removeEventListener('message', handler);
              resolve(msg.payload);
            } else if (msg.type === 'error') {
              this.worker.removeEventListener('message', handler);
              reject(new Error(msg.message || 'Worker synthesis error'));
            }
          };
          this.worker.addEventListener('message', handler);
          this.worker.postMessage({ reqId, tierId });
        });
      } catch (workerErr) {
        console.warn('[V9R1] Worker fallback to main thread:', workerErr);
      }
    }

    // Fallback for standalone file:// or single-HTML bundle
    await new Promise((r) => setTimeout(r, 24));
    return renderV9r1Song(tierId, (pct, text) => {
      if (this.activeReqId === reqId) {
        studioStore.setState({ renderPct: pct, renderText: text });
      }
    });
  }

  async loadTier(tierId) {
    const wasPlaying = studioStore.getState().isPlaying;
    const savedTime = this.getCurrentTime();
    if (wasPlaying) {
      this.stopSourcesOnly();
      studioStore.setState({ isPlaying: false });
    }

    if (this.tierCache.has(tierId)) {
      const cached = this.tierCache.get(tierId);
      studioStore.setState({
        tierId,
        isRendering: false,
        renderPct: 1.0,
        renderText: `已从内存秒切 [${cached.stats.vocalLabel}]（0ms 缓存命中）`,
        renderStats: cached.stats,
        durationSec: cached.stats.durationSec,
        errorText: null,
      });
      this.startSongOffset = Math.min(savedTime, cached.stats.durationSec - 0.05);
      if (wasPlaying) {
        await this.play();
      }
      return;
    }

    const reqId = ++this.reqCounter;
    this.activeReqId = reqId;
    studioStore.setState({
      tierId,
      isRendering: true,
      renderPct: 0.05,
      renderText: `正在实时纯代码积分 [${tierId.toUpperCase()}] 10 轨声学波形...`,
      errorText: null,
    });

    try {
      const payload = await this.synthesizeViaWorkerOrMain(tierId, reqId);
      if (this.activeReqId !== reqId) return;

      const built = this.buildTierBuffers(payload);
      this.tierCache.set(tierId, built);

      studioStore.setState({
        tierId,
        isRendering: false,
        renderPct: 1.0,
        renderText: `合成就绪 · 耗时 ${(built.stats.renderTimeMs / 1000).toFixed(2)}s · 峰值 ${built.stats.peakAmplitude}`,
        renderStats: built.stats,
        durationSec: built.stats.durationSec,
      });

      this.startSongOffset = Math.min(savedTime, built.stats.durationSec - 0.05);
      if (wasPlaying) {
        await this.play();
      }
    } catch (err) {
      console.error('[V9R1] Synthesis failed:', err);
      studioStore.setState({
        isRendering: false,
        errorText: err && err.message ? err.message : String(err),
        renderText: '合成失败，请检查控制台日志',
      });
    }
  }

  getEffectiveGain(stemId) {
    const { stemStates } = studioStore.getState();
    const anySolo = Object.values(stemStates).some((s) => s.solo);
    const st = stemStates[stemId];
    if (!st) return 0.0;
    if (st.muted) return 0.0;
    if (anySolo && !st.solo) return 0.0;
    return st.gain;
  }

  syncStemGains() {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    for (const m of STEM_MANIFEST) {
      const nodeObj = this.stemNodes.get(m.id);
      if (!nodeObj) continue;
      const target = this.getEffectiveGain(m.id);
      nodeObj.gainNode.gain.setTargetAtTime(target, now, 0.012);
    }
  }

  stopSourcesOnly() {
    for (const src of this.sourceNodes.values()) {
      try {
        src.onended = null;
        src.stop();
        src.disconnect();
      } catch (_) {}
    }
    this.sourceNodes.clear();
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  async play() {
    const st = studioStore.getState();
    if (st.isRendering) return;
    const cached = this.tierCache.get(st.tierId);
    if (!cached) return;

    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    this.stopSourcesOnly();
    this.syncStemGains();

    let offset = this.startSongOffset;
    if (offset >= cached.stats.durationSec - 0.1) {
      offset = 0.0;
      this.startSongOffset = 0.0;
    }

    const startAt = ctx.currentTime + 0.02;
    this.startWallCtxTime = startAt;

    let firstSource = null;
    for (const m of STEM_MANIFEST) {
      const buf = cached.audioBuffers.get(m.id);
      const nodeObj = this.stemNodes.get(m.id);
      if (!buf || !nodeObj) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(nodeObj.gainNode);
      src.start(startAt, offset);
      this.sourceNodes.set(m.id, src);
      if (!firstSource) firstSource = src;
    }

    if (firstSource) {
      firstSource.onended = () => {
        const cur = this.getCurrentTime();
        if (cur >= cached.stats.durationSec - 0.15) {
          this.pause(0.0);
        }
      };
    }

    studioStore.setState({ isPlaying: true, currentTime: offset });
    this.startClockLoop();
  }

  pause(overrideOffset = null) {
    const cur = overrideOffset !== null ? overrideOffset : this.getCurrentTime();
    this.stopSourcesOnly();
    this.startSongOffset = cur;
    studioStore.setState({ isPlaying: false, currentTime: cur });
  }

  async togglePlay() {
    if (studioStore.getState().isPlaying) {
      this.pause();
    } else {
      await this.play();
    }
  }

  async seek(sec) {
    const st = studioStore.getState();
    const clamped = Math.max(0, Math.min(st.durationSec - 0.05, Number(sec) || 0));
    const wasPlaying = st.isPlaying;
    if (wasPlaying) {
      this.stopSourcesOnly();
    }
    this.startSongOffset = clamped;
    studioStore.setState({ currentTime: clamped });
    if (wasPlaying) {
      await this.play();
    }
  }

  getCurrentTime() {
    const st = studioStore.getState();
    if (!st.isPlaying || !this.ctx) {
      return this.startSongOffset;
    }
    const elapsed = Math.max(0, this.ctx.currentTime - this.startWallCtxTime);
    return Math.min(st.durationSec, this.startSongOffset + elapsed);
  }

  startClockLoop() {
    const tick = () => {
      if (!studioStore.getState().isPlaying) return;
      const t = this.getCurrentTime();
      studioStore.setState({ currentTime: t });
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  toggleMute(stemId) {
    const cur = studioStore.getState().stemStates[stemId];
    if (!cur) return;
    studioStore.updateStem(stemId, { muted: !cur.muted });
    this.syncStemGains();
  }

  toggleSolo(stemId) {
    const cur = studioStore.getState().stemStates[stemId];
    if (!cur) return;
    studioStore.updateStem(stemId, { solo: !cur.solo, muted: false });
    this.syncStemGains();
  }

  setStemGain(stemId, gain) {
    studioStore.updateStem(stemId, { gain: Number(gain) });
    this.syncStemGains();
  }

  resetAllStems() {
    studioStore.resetStems();
    this.syncStemGains();
  }

  getLiveLevels() {
    const st = studioStore.getState();
    const out = { masterRms: 0.0, stems: {} };
    const cached = this.tierCache.get(st.tierId);

    if (st.isPlaying && this.ctx && this.masterAnalyser) {
      this.masterAnalyser.getFloatTimeDomainData(this.masterMeterBuf);
      let mSq = 0.0;
      for (let i = 0; i < this.masterMeterBuf.length; i++) {
        mSq += this.masterMeterBuf[i] * this.masterMeterBuf[i];
      }
      out.masterRms = Math.sqrt(mSq / this.masterMeterBuf.length);

      for (const m of STEM_MANIFEST) {
        const nodeObj = this.stemNodes.get(m.id);
        if (!nodeObj) {
          out.stems[m.id] = 0.0;
          continue;
        }
        nodeObj.analyser.getFloatTimeDomainData(nodeObj.meterBuf);
        let sq = 0.0;
        for (let i = 0; i < nodeObj.meterBuf.length; i++) {
          sq += nodeObj.meterBuf[i] * nodeObj.meterBuf[i];
        }
        out.stems[m.id] = Math.sqrt(sq / nodeObj.meterBuf.length);
      }
      return out;
    }

    // Paused preview at current scrub time using precomputed 20ms stem RMS
    if (cached) {
      const frameIdx = Math.floor(st.currentTime / 0.02);
      let mSq = 0.0;
      for (const m of STEM_MANIFEST) {
        const env = cached.envelopes.get(m.id);
        const effGain = this.getEffectiveGain(m.id);
        const raw = env && frameIdx >= 0 && frameIdx < env.length ? env[frameIdx] : 0.0;
        const val = raw * effGain;
        out.stems[m.id] = val;
        mSq += val * val;
      }
      out.masterRms = Math.sqrt(mSq);
    } else {
      for (const m of STEM_MANIFEST) out.stems[m.id] = 0.0;
    }
    return out;
  }
}

export const audioMixer = new V9r1AudioMixer();
