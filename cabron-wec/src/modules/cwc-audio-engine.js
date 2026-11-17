// CabronWEC (CWC-4X) Streaming WebAudio & Progressive Chunk Engine
// - Boots immediately on Chunk 0 (~520 KB gzipped) with 0ms pre-render latency
// - Streams frames in real time on a dedicated AudioWorklet thread (< 12 MB RAM)
// - Maintains an optional decoupled Original Reference MP3 A/B bus loaded strictly on demand

import { CWC_CHUNKS, CWC_TRACK_META } from '../../data/cwc-models.js';
import { CWC_HOP_M, CWC_SAMPLE_RATE } from '../dsp/mdct-core.js';
import { CWCStreamDecoder } from '../dsp/cwc-stream-decoder.js';
import { cwcStore } from '../state/cwc-store.js';

class CWCAudioEngine {
  constructor() {
    this.ctx = null;
    this.workletNode = null;
    this.cwcGain = null;
    this.analyser = null;
    this.refAudioEl = null;
    this.refSourceNode = null;
    this.refGain = null;

    this.chunkPromises = new Map();
    this.chunkFrameCache = new Map(); // Stores parsed frames per chunk for Inspector & WAV Export
    this.inspectorDecoder = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);

    // Live 60fps canvas telemetry (mutated without forcing full DOM re-renders)
    this.liveTelemetry = {
      spectrum32: new Float32Array(32),
      layerRms: new Float32Array(5),
      timeDomain: new Float32Array(256),
      currentSample: 0,
      currentFrame: 0,
      cpuMs: 0,
    };

    // ScriptProcessor fallback only if browser AudioWorklet fails in restricted environment
    this.fallbackNode = null;
    this.fallbackDecoder = null;
    this.fallbackSample = 0;
    this.fallbackPlaying = false;
    this.fallbackHopL = new Float32Array(CWC_HOP_M);
    this.fallbackHopR = new Float32Array(CWC_HOP_M);
    this.fallbackCachedFrame = -1;

    this.rafId = null;
    this.lastUiSyncMs = 0;
  }

  async init() {
    cwcStore.setState({
      durationSec: CWC_TRACK_META.durationSec,
      totalFrames: CWC_TRACK_META.totalFrames,
    });

    // Prioritize Chunk 0 immediately for sub-100ms instant ready state, then prefetch 1..7
    await this.loadChunk(0);
    cwcStore.setState({ isReady: true });
    this.prefetchRemainingChunks();
  }

  async loadChunk(chunkIdx) {
    if (chunkIdx < 0 || chunkIdx >= CWC_CHUNKS.length) return null;
    if (this.chunkPromises.has(chunkIdx)) return this.chunkPromises.get(chunkIdx);

    const promise = (async () => {
      const meta = CWC_CHUNKS[chunkIdx];
      const mod = await import(`../../data/${meta.file}`);
      const frames = mod.CWC_PACKED_FRAMES.split('|');
      this.chunkFrameCache.set(chunkIdx, { startFrame: mod.START_FRAME, frames });
      this.inspectorDecoder.ingestChunk(mod.START_FRAME, frames);

      if (this.workletNode) {
        this.workletNode.port.postMessage({
          type: 'chunk',
          startFrame: mod.START_FRAME,
          frames,
        });
      }
      if (this.fallbackDecoder) {
        this.fallbackDecoder.ingestChunk(mod.START_FRAME, frames);
      }

      const nextLoaded = [...cwcStore.getState().chunksLoaded];
      nextLoaded[chunkIdx] = true;
      cwcStore.setState({
        chunksLoaded: nextLoaded,
        loadedChunkCount: nextLoaded.filter(Boolean).length,
      });
      return { startFrame: mod.START_FRAME, frames };
    })();

    this.chunkPromises.set(chunkIdx, promise);
    return promise;
  }

  async prefetchRemainingChunks() {
    for (let i = 1; i < CWC_CHUNKS.length; i++) {
      await this.loadChunk(i);
    }
  }

  chunkIndexForFrame(frameIdx) {
    for (let i = 0; i < CWC_CHUNKS.length; i++) {
      const c = CWC_CHUNKS[i];
      if (frameIdx >= c.startFrame && frameIdx < c.startFrame + c.frameCount) return i;
    }
    return 0;
  }

  async ensureAudioContext() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') await this.ctx.resume();
      return this.ctx;
    }

    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC({ sampleRate: CWC_SAMPLE_RATE, latencyHint: 'interactive' });
    this.cwcGain = this.ctx.createGain();
    this.cwcGain.gain.value = cwcStore.getState().audioSource === 'cwc' ? 1 : 0;

    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.72;

    this.cwcGain.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);

    try {
      const workletUrl = new URL('../dsp/cwc-worklet.js', import.meta.url);
      await this.ctx.audioWorklet.addModule(workletUrl);
      this.workletNode = new AudioWorkletNode(this.ctx, 'cabron-wec-processor', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      this.workletNode.connect(this.cwcGain);
      this.workletNode.port.onmessage = (ev) => this.handleWorkletMessage(ev.data);
      this.workletNode.port.postMessage({
        type: 'init',
        totalSamples: CWC_TRACK_META.totalSamples,
        totalFrames: CWC_TRACK_META.totalFrames,
      });

      // Push already loaded chunks into the Worklet
      for (const [, entry] of this.chunkFrameCache.entries()) {
        this.workletNode.port.postMessage({
          type: 'chunk',
          startFrame: entry.startFrame,
          frames: entry.frames,
        });
      }
      this.syncStemsToWorklet();
      cwcStore.setState({ engineMode: 'AudioWorklet (128-sample)' });
    } catch (_err) {
      // Clean fallback streaming scheduler using CWCStreamDecoder directly
      this.setupFallbackProcessor();
      cwcStore.setState({ engineMode: 'WebAudio Stream Fallback' });
    }

    if (this.ctx.state === 'suspended') await this.ctx.resume();
    return this.ctx;
  }

  setupFallbackProcessor() {
    this.fallbackDecoder = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
    for (const [, entry] of this.chunkFrameCache.entries()) {
      this.fallbackDecoder.ingestChunk(entry.startFrame, entry.frames);
    }
    this.syncStemsToWorklet();
    const proc = this.ctx.createScriptProcessor(1024, 0, 2);
    proc.onaudioprocess = (ev) => {
      const outL = ev.outputBuffer.getChannelData(0);
      const outR = ev.outputBuffer.getChannelData(1);
      if (!this.fallbackPlaying) {
        outL.fill(0);
        outR.fill(0);
        return;
      }
      const t0 = performance.now();
      for (let i = 0; i < outL.length; i++) {
        if (this.fallbackSample >= CWC_TRACK_META.totalSamples) {
          outL.fill(0, i);
          outR.fill(0, i);
          this.fallbackPlaying = false;
          this.fallbackSample = 0;
          this.handleTrackEnded();
          break;
        }
        const fIdx = Math.floor(this.fallbackSample / CWC_HOP_M);
        const intra = this.fallbackSample - fIdx * CWC_HOP_M;
        if (fIdx !== this.fallbackCachedFrame) {
          if (this.fallbackDecoder.hasFrame(fIdx)) {
            this.fallbackDecoder.decodeFrame(fIdx, this.fallbackHopL, this.fallbackHopR, 0);
          } else {
            this.fallbackHopL.fill(0);
            this.fallbackHopR.fill(0);
          }
          this.fallbackCachedFrame = fIdx;
        }
        outL[i] = this.fallbackHopL[intra];
        outR[i] = this.fallbackHopR[intra];
        this.fallbackSample++;
      }
      this.liveTelemetry.cpuMs = performance.now() - t0;
      this.liveTelemetry.currentSample = this.fallbackSample;
      this.liveTelemetry.currentFrame = Math.floor(this.fallbackSample / CWC_HOP_M);
      this.liveTelemetry.layerRms.set(this.fallbackDecoder.layerRms);
      this.liveTelemetry.spectrum32.set(this.fallbackDecoder.spectrum32);
    };
    proc.connect(this.cwcGain);
    this.fallbackNode = proc;
  }

  handleWorkletMessage(msg) {
    if (!msg) return;
    if (msg.type === 'telemetry') {
      if (cwcStore.getState().audioSource === 'cwc') {
        this.liveTelemetry.currentSample = msg.currentSample;
        this.liveTelemetry.currentFrame = msg.currentFrame;
        this.liveTelemetry.cpuMs = typeof msg.cpuMs === 'number' ? msg.cpuMs : null;
        for (let i = 0; i < 5; i++) this.liveTelemetry.layerRms[i] = msg.layerRms[i];
        for (let i = 0; i < 32; i++) this.liveTelemetry.spectrum32[i] = msg.spectrum32[i];

        const now = performance.now();
        if (now - this.lastUiSyncMs > 75) {
          this.lastUiSyncMs = now;
          const timeSec = msg.currentSample / CWC_SAMPLE_RATE;
          const st = cwcStore.getState();
          cwcStore.setState({
            currentTimeSec: timeSec,
            currentFrame: msg.currentFrame,
            cpuMs: typeof msg.cpuMs === 'number' ? Number(msg.cpuMs.toFixed(3)) : null,
            underruns: msg.underruns || 0,
            stems: st.stems.map((s, idx) => ({ ...s, rms: msg.layerRms[idx] })),
          });
        }
      }
    } else if (msg.type === 'ended') {
      this.handleTrackEnded();
    }
  }

  handleTrackEnded() {
    this.liveTelemetry.currentSample = 0;
    this.liveTelemetry.currentFrame = 0;
    this.liveTelemetry.layerRms.fill(0);
    cwcStore.setState({
      isPlaying: false,
      currentTimeSec: 0,
      currentFrame: 0,
      stems: cwcStore.getState().stems.map((s) => ({ ...s, rms: 0 })),
    });
  }

  syncStemsToWorklet() {
    const st = cwcStore.getState();
    const controls = {
      gains: st.stems.map((s) => s.gain),
      mutes: st.stems.map((s) => s.muted),
      solos: st.stems.map((s) => s.solo),
      masterGain: st.masterGain,
    };
    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'stems', controls });
    }
    if (this.fallbackDecoder) {
      this.fallbackDecoder.setStemControls(controls);
    }
  }

  async ensureReferenceAudio() {
    if (this.refAudioEl) return this.refAudioEl;
    cwcStore.setState({ refLoading: true });
    await this.ensureAudioContext();
    const audio = new Audio('/ref-audio/master.mp3');
    audio.preload = 'auto';
    audio.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      const onCanPlay = () => { cleanup(); resolve(); };
      const onErr = () => { cleanup(); reject(new Error('Failed to load reference MP3')); };
      const cleanup = () => {
        audio.removeEventListener('canplay', onCanPlay);
        audio.removeEventListener('error', onErr);
      };
      audio.addEventListener('canplay', onCanPlay);
      audio.addEventListener('error', onErr);
      audio.load();
    });
    this.refSourceNode = this.ctx.createMediaElementSource(audio);
    this.refGain = this.ctx.createGain();
    this.refGain.gain.value = cwcStore.getState().audioSource === 'ref' ? 0.98 : 0;
    this.refSourceNode.connect(this.refGain);
    this.refGain.connect(this.analyser);
    audio.addEventListener('ended', () => {
      if (cwcStore.getState().audioSource === 'ref') this.handleTrackEnded();
    });
    this.refAudioEl = audio;
    cwcStore.setState({ refLoading: false, refLoaded: true });
    return audio;
  }

  async setAudioSource(source) {
    if (source === cwcStore.getState().audioSource) return;
    const curSec = cwcStore.getState().currentTimeSec;
    const wasPlaying = cwcStore.getState().isPlaying;

    if (source === 'ref') {
      try {
        await this.ensureReferenceAudio();
      } catch (err) {
        cwcStore.setState({ errorMsg: err.message, refLoading: false });
        return;
      }
    }

    cwcStore.setState({ audioSource: source });
    if (this.cwcGain) this.cwcGain.gain.setTargetAtTime(source === 'cwc' ? 1 : 0, this.ctx.currentTime, 0.015);
    if (this.refGain) this.refGain.gain.setTargetAtTime(source === 'ref' ? 0.98 : 0, this.ctx.currentTime, 0.015);

    if (wasPlaying) {
      await this.seek(curSec);
      await this.play();
    }
  }

  async play() {
    await this.ensureAudioContext();
    const st = cwcStore.getState();
    const curSample = Math.floor(st.currentTimeSec * CWC_SAMPLE_RATE);
    const targetChunk = this.chunkIndexForFrame(Math.floor(curSample / CWC_HOP_M));
    await this.loadChunk(targetChunk);
    if (targetChunk + 1 < CWC_CHUNKS.length) this.loadChunk(targetChunk + 1);

    if (st.audioSource === 'cwc') {
      if (this.refAudioEl && !this.refAudioEl.paused) this.refAudioEl.pause();
      if (this.workletNode) {
        this.workletNode.port.postMessage({ type: 'play', sampleOffset: curSample });
      } else {
        this.fallbackDecoder.seekToFrame(Math.floor(curSample / CWC_HOP_M));
        this.fallbackSample = curSample;
        this.fallbackCachedFrame = -1;
        this.fallbackPlaying = true;
      }
    } else {
      if (this.workletNode) this.workletNode.port.postMessage({ type: 'pause' });
      this.fallbackPlaying = false;
      const ref = await this.ensureReferenceAudio();
      ref.currentTime = st.currentTimeSec;
      await ref.play();
    }

    cwcStore.setState({ isPlaying: true, errorMsg: null });
    this.startUiLoop();
  }

  pause() {
    if (this.workletNode) this.workletNode.port.postMessage({ type: 'pause' });
    this.fallbackPlaying = false;
    if (this.refAudioEl && !this.refAudioEl.paused) this.refAudioEl.pause();
    cwcStore.setState({ isPlaying: false });
  }

  async togglePlay() {
    if (cwcStore.getState().isPlaying) this.pause();
    else await this.play();
  }

  async seek(timeSec) {
    const clamped = Math.max(0, Math.min(CWC_TRACK_META.durationSec - 0.05, timeSec));
    const targetSample = Math.floor(clamped * CWC_SAMPLE_RATE);
    const targetFrame = Math.floor(targetSample / CWC_HOP_M);
    const chunkIdx = this.chunkIndexForFrame(targetFrame);
    await this.loadChunk(chunkIdx);

    this.liveTelemetry.currentSample = targetSample;
    this.liveTelemetry.currentFrame = targetFrame;
    cwcStore.setState({
      currentTimeSec: clamped,
      currentFrame: targetFrame,
      inspectedFrame: targetFrame,
    });

    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'seek', sampleOffset: targetSample });
    } else if (this.fallbackDecoder) {
      this.fallbackDecoder.seekToFrame(targetFrame);
      this.fallbackSample = targetSample;
      this.fallbackCachedFrame = -1;
    }

    if (this.refAudioEl && cwcStore.getState().audioSource === 'ref') {
      this.refAudioEl.currentTime = clamped;
    }
  }

  startUiLoop() {
    if (this.rafId) return;
    const tick = () => {
      this.rafId = requestAnimationFrame(tick);
      if (this.analyser) {
        this.analyser.getFloatTimeDomainData(this.liveTelemetry.timeDomain);
      }
      const st = cwcStore.getState();
      if (st.audioSource === 'ref' && this.refAudioEl && st.isPlaying) {
        const t = this.refAudioEl.currentTime;
        const f = Math.floor((t * CWC_SAMPLE_RATE) / CWC_HOP_M);
        this.liveTelemetry.currentSample = Math.floor(t * CWC_SAMPLE_RATE);
        this.liveTelemetry.currentFrame = f;
        if (this.analyser) {
          const freq = new Uint8Array(this.analyser.frequencyBinCount);
          this.analyser.getByteFrequencyData(freq);
          for (let b = 0; b < 32; b++) {
            const idx = Math.min(freq.length - 1, Math.floor(Math.pow(freq.length, b / 32)));
            this.liveTelemetry.spectrum32[b] = freq[idx] / 255;
          }
        }
        const now = performance.now();
        if (now - this.lastUiSyncMs > 75) {
          this.lastUiSyncMs = now;
          cwcStore.setState({ currentTimeSec: t, currentFrame: f });
        }
      }
    };
    this.rafId = requestAnimationFrame(tick);
  }

  setStemGain(index, gain) {
    const stems = cwcStore.getState().stems.map((s, i) => (i === index ? { ...s, gain } : s));
    cwcStore.setState({ stems });
    this.syncStemsToWorklet();
  }

  toggleStemMute(index) {
    const stems = cwcStore.getState().stems.map((s, i) => (i === index ? { ...s, muted: !s.muted } : s));
    cwcStore.setState({ stems });
    this.syncStemsToWorklet();
  }

  toggleStemSolo(index) {
    const stems = cwcStore.getState().stems.map((s, i) => (i === index ? { ...s, solo: !s.solo } : s));
    cwcStore.setState({ stems });
    this.syncStemsToWorklet();
  }

  setMasterGain(masterGain) {
    cwcStore.setState({ masterGain });
    this.syncStemsToWorklet();
  }

  getFrameCodeString(frameIdx) {
    return this.inspectorDecoder.frameStrings[frameIdx] ?? null;
  }

  /**
   * Streams all 8 CWC chunks through CWCStreamDecoder with current stem mix
   * and builds a 16-bit PCM Stereo WAV download in ~2 seconds.
   */
  async exportWav() {
    if (cwcStore.getState().exportingWav) return;
    cwcStore.setState({ exportingWav: true, exportProgress: 0 });
    try {
      for (let i = 0; i < CWC_CHUNKS.length; i++) {
        await this.loadChunk(i);
        cwcStore.setState({ exportProgress: ((i + 1) / CWC_CHUNKS.length) * 0.25 });
      }
      const st = cwcStore.getState();
      const dec = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
      for (const [, entry] of this.chunkFrameCache.entries()) {
        dec.ingestChunk(entry.startFrame, entry.frames);
      }
      dec.setStemControls({
        gains: st.stems.map((s) => s.gain),
        mutes: st.stems.map((s) => s.muted),
        solos: st.stems.map((s) => s.solo),
        masterGain: st.masterGain,
      });

      const N = CWC_TRACK_META.totalSamples;
      const wav = new ArrayBuffer(44 + N * 4);
      const dv = new DataView(wav);
      const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
      ws(0, 'RIFF');
      dv.setUint32(4, 36 + N * 4, true);
      ws(8, 'WAVEfmt ');
      dv.setUint32(16, 16, true);
      dv.setUint16(20, 1, true);
      dv.setUint16(22, 2, true);
      dv.setUint32(24, CWC_SAMPLE_RATE, true);
      dv.setUint32(28, CWC_SAMPLE_RATE * 4, true);
      dv.setUint16(32, 4, true);
      dv.setUint16(34, 16, true);
      ws(36, 'data');
      dv.setUint32(40, N * 4, true);

      const hopL = new Float32Array(CWC_HOP_M);
      const hopR = new Float32Array(CWC_HOP_M);
      let byteOff = 44;
      let written = 0;

      for (let f = 0; f < CWC_TRACK_META.totalFrames && written < N; f++) {
        dec.decodeFrame(f, hopL, hopR, 0);
        const take = Math.min(CWC_HOP_M, N - written);
        for (let i = 0; i < take; i++) {
          dv.setInt16(byteOff, Math.max(-32767, Math.min(32767, Math.round(hopL[i] * 32767))), true);
          dv.setInt16(byteOff + 2, Math.max(-32767, Math.min(32767, Math.round(hopR[i] * 32767))), true);
          byteOff += 4;
        }
        written += take;
        if (f % 600 === 0) {
          cwcStore.setState({ exportProgress: 0.25 + 0.75 * (f / CWC_TRACK_META.totalFrames) });
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      const blob = new Blob([wav], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'CabronWEC_CWC4X_NextIntersection_44100Hz.wav';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    } finally {
      cwcStore.setState({ exportingWav: false, exportProgress: 1 });
    }
  }
}

export const cwcEngine = new CWCAudioEngine();
