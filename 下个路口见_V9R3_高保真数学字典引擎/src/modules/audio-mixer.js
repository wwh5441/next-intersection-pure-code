// 48kHz 10-Track Web Audio Studio Mixer & Worker Bridge (`src/modules/audio-mixer.js`)
import { V9R3_STEM_SPECS, synthesizeAllTenTracks, synthesizeVocalOnlyTracks } from '../dsp/engine-orchestrator.js';
import { studioStore } from '../state/studio-store.js';

class AudioMixerController {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.masterAnalyser = null;
    this.stemNodes = new Map();
    this.audioBuffers = new Map();
    this.activeSources = [];
    this.worker = null;
    this.reqCounter = 0;

    this.playStartCtxTime = 0;
    this.pauseOffsetSec = 0;
    this.rafId = null;
    this.freqBins = new Uint8Array(512);
    this.meterScratch = new Float32Array(256);

    studioStore.subscribe(() => this.syncStemGains());
  }

  ensureContext() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC({ sampleRate: 48000 });
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 1.0;

      this.masterAnalyser = this.ctx.createAnalyser();
      this.masterAnalyser.fftSize = 1024;
      this.masterAnalyser.smoothingTimeConstant = 0.78;
      this.freqBins = new Uint8Array(this.masterAnalyser.frequencyBinCount);

      this.masterGain.connect(this.masterAnalyser);
      this.masterAnalyser.connect(this.ctx.destination);

      for (const spec of V9R3_STEM_SPECS) {
        const gainNode = this.ctx.createGain();
        const analyser = this.ctx.createAnalyser();
        analyser.fftSize = 256;
        gainNode.connect(analyser);
        analyser.connect(this.masterGain);
        this.stemNodes.set(spec.id, { gainNode, analyser });
      }
      this.syncStemGains();
    }
    return this.ctx;
  }

  ensureWorker() {
    if (this.worker || window.__V9R3_SINGLE_FILE__) return this.worker;
    try {
      this.worker = new Worker(new URL('./synth-worker.js', import.meta.url), { type: 'module' });
    } catch {
      this.worker = null;
    }
    return this.worker;
  }

  tracksToAudioBuffers(tracksMap) {
    const ctx = this.ensureContext();
    for (const [stemId, chArrays] of Object.entries(tracksMap)) {
      const numCh = chArrays.length;
      const len = chArrays[0].length;
      const buf = ctx.createBuffer(numCh, len, 48000);
      for (let c = 0; c < numCh; c++) {
        buf.copyToChannel(chArrays[c], c);
      }
      this.audioBuffers.set(stemId, buf);
    }
  }

  async loadTier(tierId) {
    const wasPlaying = studioStore.getState().isPlaying;
    const savedPos = this.getCurrentTime();
    if (wasPlaying) this.pause();

    const pitchSemitones = studioStore.getState().pitchSemitones;
    studioStore.setState({
      tierId,
      isSynthesizing: true,
      synthProgress: 0.04,
      synthStatus: '正在启动 48kHz 纯数学字典引擎...',
    });

    const worker = this.ensureWorker();
    if (worker) {
      const reqId = ++this.reqCounter;
      await new Promise((resolve, reject) => {
        const handler = (ev) => {
          const msg = ev.data || {};
          if (msg.id !== reqId) return;
          if (msg.type === 'progress') {
            studioStore.setState({ synthProgress: msg.progress, synthStatus: msg.status });
          } else if (msg.type === 'done_all') {
            worker.removeEventListener('message', handler);
            this.tracksToAudioBuffers(msg.payload.tracks);
            studioStore.setState({
              isSynthesizing: false,
              synthProgress: 1.0,
              duration: msg.payload.durationSec,
              elapsedSynthMs: msg.payload.elapsedSynthMs,
              synthStatus: `10 轨 48kHz 纯数学字典就绪 (${msg.payload.elapsedSynthMs} ms)`,
            });
            resolve();
          } else if (msg.type === 'error') {
            worker.removeEventListener('message', handler);
            reject(new Error(msg.message));
          }
        };
        worker.addEventListener('message', handler);
        worker.postMessage({ id: reqId, cmd: 'synth_all', tierId, pitchSemitones });
      });
    } else {
      await new Promise((r) => setTimeout(r, 25));
      const res = synthesizeAllTenTracks(tierId, { pitchSemitones }, (progress, status) => {
        studioStore.setState({ synthProgress: progress, synthStatus: status });
      });
      this.tracksToAudioBuffers(res.tracks);
      studioStore.setState({
        isSynthesizing: false,
        synthProgress: 1.0,
        duration: res.durationSec,
        elapsedSynthMs: res.elapsedMs,
        synthStatus: `10 轨 48kHz 纯数学字典就绪 (${res.elapsedMs} ms)`,
      });
    }

    this.pauseOffsetSec = Math.min(savedPos, studioStore.getState().duration - 0.1);
    if (wasPlaying) await this.play();
  }

  async setVocalPitchSemitones(pitchSemitones) {
    const st = studioStore.getState();
    if (st.isSynthesizing || st.pitchSemitones === pitchSemitones) return;

    const wasPlaying = st.isPlaying;
    const savedPos = this.getCurrentTime();
    if (wasPlaying) this.pause();

    studioStore.setState({
      pitchSemitones,
      isSynthesizing: true,
      synthProgress: 0.25,
      synthStatus: `正在重排主唱 GCI 声门周期字典 (${pitchSemitones > 0 ? '+' : ''}${pitchSemitones} 半音)...`,
    });

    const worker = this.ensureWorker();
    if (worker) {
      const reqId = ++this.reqCounter;
      await new Promise((resolve, reject) => {
        const handler = (ev) => {
          const msg = ev.data || {};
          if (msg.id !== reqId) return;
          if (msg.type === 'progress') {
            studioStore.setState({ synthProgress: msg.progress, synthStatus: msg.status });
          } else if (msg.type === 'done_vocal') {
            worker.removeEventListener('message', handler);
            this.tracksToAudioBuffers(msg.payload.tracks);
            studioStore.setState({
              isSynthesizing: false,
              synthProgress: 1.0,
              synthStatus: `主唱 GCI 周期变调完成 (${pitchSemitones > 0 ? '+' : ''}${pitchSemitones} st · ${msg.payload.elapsedMs} ms)`,
            });
            resolve();
          } else if (msg.type === 'error') {
            worker.removeEventListener('message', handler);
            reject(new Error(msg.message));
          }
        };
        worker.addEventListener('message', handler);
        worker.postMessage({ id: reqId, cmd: 'synth_vocal', tierId: st.tierId, pitchSemitones });
      });
    } else {
      await new Promise((r) => setTimeout(r, 20));
      const res = synthesizeVocalOnlyTracks(st.tierId, pitchSemitones);
      this.tracksToAudioBuffers(res.tracks);
      studioStore.setState({
        isSynthesizing: false,
        synthProgress: 1.0,
        synthStatus: `主唱 GCI 周期变调完成 (${pitchSemitones > 0 ? '+' : ''}${pitchSemitones} st · ${res.elapsedMs} ms)`,
      });
    }

    this.pauseOffsetSec = savedPos;
    if (wasPlaying) await this.play();
  }

  syncStemGains() {
    if (!this.ctx) return;
    const { stemStates } = studioStore.getState();
    const anySolo = Object.values(stemStates).some((s) => s.solo);
    const now = this.ctx.currentTime;

    for (const spec of V9R3_STEM_SPECS) {
      const node = this.stemNodes.get(spec.id);
      if (!node) continue;
      const st = stemStates[spec.id];
      const audible = anySolo ? st.solo && !st.mute : !st.mute;
      const target = audible ? st.gain : 0.0;
      node.gainNode.gain.setTargetAtTime(target, now, 0.012);
    }
  }

  async play() {
    if (studioStore.getState().isSynthesizing || this.audioBuffers.size === 0) return;
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') await ctx.resume();

    this.stopSourcesOnly();
    const offset = Math.max(0, Math.min(this.pauseOffsetSec, studioStore.getState().duration - 0.05));
    const startAt = ctx.currentTime + 0.02;

    for (const spec of V9R3_STEM_SPECS) {
      const buf = this.audioBuffers.get(spec.id);
      const node = this.stemNodes.get(spec.id);
      if (!buf || !node) continue;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(node.gainNode);
      src.start(startAt, offset);
      this.activeSources.push(src);
    }

    this.playStartCtxTime = startAt - offset;
    studioStore.setState({ isPlaying: true, currentTime: offset });
    this.startClockLoop();
  }

  pause() {
    if (!studioStore.getState().isPlaying) return;
    this.pauseOffsetSec = this.getCurrentTime();
    this.stopSourcesOnly();
    studioStore.setState({ isPlaying: false, currentTime: this.pauseOffsetSec });
  }

  stop() {
    this.stopSourcesOnly();
    this.pauseOffsetSec = 0;
    studioStore.setState({ isPlaying: false, currentTime: 0 });
  }

  async seek(sec) {
    const dur = studioStore.getState().duration;
    const clamped = Math.max(0, Math.min(sec, dur - 0.05));
    this.pauseOffsetSec = clamped;
    studioStore.setState({ currentTime: clamped });
    if (studioStore.getState().isPlaying) {
      await this.play();
    }
  }

  stopSourcesOnly() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    for (const src of this.activeSources) {
      try {
        src.stop();
        src.disconnect();
      } catch {
        // ignore already stopped source
      }
    }
    this.activeSources = [];
  }

  getCurrentTime() {
    const st = studioStore.getState();
    if (!st.isPlaying || !this.ctx) return this.pauseOffsetSec;
    return Math.max(0, Math.min(this.ctx.currentTime - this.playStartCtxTime, st.duration));
  }

  startClockLoop() {
    const tick = () => {
      if (!studioStore.getState().isPlaying) return;
      const t = this.getCurrentTime();
      const dur = studioStore.getState().duration;
      if (t >= dur - 0.02) {
        this.stop();
        return;
      }
      studioStore.setState({ currentTime: t });
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  getMasterSpectrum() {
    if (!this.masterAnalyser || !studioStore.getState().isPlaying) {
      this.freqBins.fill(0);
      return this.freqBins;
    }
    this.masterAnalyser.getByteFrequencyData(this.freqBins);
    return this.freqBins;
  }

  getStemRmsLevel(stemId) {
    if (!studioStore.getState().isPlaying) return 0;
    const node = this.stemNodes.get(stemId);
    if (!node) return 0;
    node.analyser.getFloatTimeDomainData(this.meterScratch);
    let sum = 0.0;
    for (let i = 0; i < this.meterScratch.length; i++) {
      sum += this.meterScratch[i] * this.meterScratch[i];
    }
    const rms = Math.sqrt(sum / this.meterScratch.length);
    return Math.min(1.0, Math.pow(rms * 5.2, 0.62));
  }
}

export const audioMixer = new AudioMixerController();
