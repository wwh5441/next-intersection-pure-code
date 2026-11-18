// CabronWEC (CWC-5X) Multi-Tier Streaming WebAudio & Progressive Chunk Engine
// - Supports instant clickless switching between:
//     1. `1m`   (< 1 MB Pure-Code Edition)
//     2. `500k` (< 500 KB Pure-Code Edition)
//     3. `hifi` (2.38 MB Reference Pure-Code Edition)
//     4. `ref`  (Original 9.62 MB 320 kbps MP3 Master)
// - Streams CSR frames in real time on a dedicated AudioWorklet thread

import { CWC_EDITIONS, CWC_TRACK_META } from '../../data/cwc-models.js';
import { unpackBase91Chunk } from '../dsp/cwc-entropy.js';
import { unpackWecContainer } from '../dsp/cwc-wec-container.js';
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

    this.chunkPromises = new Map(); // key: `${tier}:${chunkIdx}`
    this.chunkBundleCache = new Map(); // key: `${tier}:${chunkIdx}`
    this.loadedWecBundle = null;
    this.wecCache = new Map();
    this.inspectorDecoders = {
      '1m': new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
      '500k': new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
      hifi: new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
      wec: new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
    };

    this.liveTelemetry = {
      spectrum32: new Float32Array(32),
      layerRms: new Float32Array(5),
      timeDomain: new Float32Array(256),
      currentSample: 0,
      currentFrame: 0,
      cpuMs: 0,
    };

    this.fallbackNode = null;
    this.fallbackDecoders = null;
    this.fallbackSample = 0;
    this.fallbackPlaying = false;
    this.fallbackHopL = new Float32Array(CWC_HOP_M);
    this.fallbackHopR = new Float32Array(CWC_HOP_M);
    this.fallbackCachedFrame = -1;

    this.rafId = null;
    this.lastUiSyncMs = 0;
  }

  getEditionMeta(tier = cwcStore.getState().activeTier) {
    return CWC_EDITIONS[tier] || CWC_EDITIONS['1m'];
  }

  async init() {
    cwcStore.setState({
      durationSec: CWC_TRACK_META.durationSec,
      totalFrames: CWC_TRACK_META.totalFrames,
    });
    // Boot immediately on `< 1 MB` Chunk 0 (~90 KB gzip), then prefetch `< 1 MB` & `< 500 KB` in background!
    await this.loadChunk('1m', 0);
    cwcStore.setState({ isReady: true });
    this.prefetchTierChunks('1m').then(() => this.prefetchTierChunks('500k'));
  }

  syncActiveTierChunkFlags(tier = cwcStore.getState().activeTier) {
    const byTier = cwcStore.getState().chunksLoadedByTier;
    const arr = byTier[tier] || new Array(8).fill(false);
    cwcStore.setState({
      chunksLoaded: [...arr],
      loadedChunkCount: arr.filter(Boolean).length,
    });
  }

  async loadChunk(tier, chunkIdx) {
    if (tier === 'wec') return this.loadedWecBundle;
    const ed = this.getEditionMeta(tier);
    if (!ed || chunkIdx < 0 || chunkIdx >= ed.chunks.length) return null;
    const key = `${ed.id}:${chunkIdx}`;
    if (this.chunkPromises.has(key)) return this.chunkPromises.get(key);

    const promise = (async () => {
      const meta = ed.chunks[chunkIdx];
      const mod = await import(`../../data/${meta.file}`);
      const bundle = unpackBase91Chunk(mod.CWC_PACKED_CHUNK, mod.START_FRAME, mod.FRAME_COUNT);
      this.chunkBundleCache.set(key, bundle);
      this.inspectorDecoders[ed.id].ingestChunk(bundle);

      if (this.workletNode) {
        this.workletNode.port.postMessage({
          type: 'chunk',
          tier: ed.id,
          chunkBundle: bundle,
        });
      }
      if (this.fallbackDecoders && this.fallbackDecoders[ed.id]) {
        this.fallbackDecoders[ed.id].ingestChunk(bundle);
      }

      const st = cwcStore.getState();
      const nextTierArr = [...(st.chunksLoadedByTier[ed.id] || new Array(8).fill(false))];
      nextTierArr[chunkIdx] = true;
      const nextByTier = { ...st.chunksLoadedByTier, [ed.id]: nextTierArr };
      const patch = { chunksLoadedByTier: nextByTier };
      if (st.activeTier === ed.id) {
        patch.chunksLoaded = nextTierArr;
        patch.loadedChunkCount = nextTierArr.filter(Boolean).length;
      }
      cwcStore.setState(patch);
      return bundle;
    })();

    this.chunkPromises.set(key, promise);
    return promise;
  }

  async prefetchTierChunks(tier) {
    const ed = this.getEditionMeta(tier);
    for (let i = 0; i < ed.chunks.length; i++) {
      await this.loadChunk(ed.id, i);
    }
  }

  chunkIndexForFrame(frameIdx, tier = cwcStore.getState().activeTier) {
    const chunks = this.getEditionMeta(tier).chunks;
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
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
        tier: cwcStore.getState().activeTier,
        totalSamples: CWC_TRACK_META.totalSamples,
        totalFrames: CWC_TRACK_META.totalFrames,
      });
      for (const [key, bundle] of this.chunkBundleCache.entries()) {
        const [tier] = key.split(':');
        this.workletNode.port.postMessage({
          type: 'chunk',
          tier,
          chunkBundle: bundle,
        });
      }
      if (this.loadedWecBundle) {
        const curTier = cwcStore.getState().activeTier;
        this.workletNode.port.postMessage({
          type: 'wec-bundle',
          chunkBundle: this.loadedWecBundle,
        });
        if (curTier !== 'wec') {
          this.workletNode.port.postMessage({ type: 'tier', tier: curTier });
        }
      }
      this.syncStemsToWorklet();
      cwcStore.setState({ engineMode: 'AudioWorklet (128-sample)' });
    } catch (_err) {
      this.setupFallbackProcessor();
    }

    if (this.ctx.state === 'suspended') await this.ctx.resume();
    return this.ctx;
  }

  setupFallbackProcessor() {
    this.fallbackDecoders = {
      '1m': new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
      '500k': new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
      hifi: new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
      wec: new CWCStreamDecoder(CWC_TRACK_META.totalFrames),
    };
    for (const [key, bundle] of this.chunkBundleCache.entries()) {
      const [tier] = key.split(':');
      if (this.fallbackDecoders[tier]) this.fallbackDecoders[tier].ingestChunk(bundle);
    }
    if (this.loadedWecBundle) {
      this.fallbackDecoders.wec.ingestChunk(this.loadedWecBundle);
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
      const dec = this.fallbackDecoders[cwcStore.getState().activeTier] || this.fallbackDecoders['1m'];
      const total = CWC_TRACK_META.totalSamples;
      for (let i = 0; i < outL.length; i++) {
        if (this.fallbackSample >= total) {
          outL.fill(0, i);
          outR.fill(0, i);
          this.fallbackPlaying = false;
          this.handleTrackEnded();
          break;
        }
        const fIdx = Math.floor(this.fallbackSample / CWC_HOP_M);
        const intra = this.fallbackSample - fIdx * CWC_HOP_M;
        if (fIdx !== this.fallbackCachedFrame) {
          dec.decodeFrame(fIdx, this.fallbackHopL, this.fallbackHopR, 0);
          this.fallbackCachedFrame = fIdx;
        }
        outL[i] = this.fallbackHopL[intra];
        outR[i] = this.fallbackHopR[intra];
        this.fallbackSample++;
      }
      const curF = Math.floor(this.fallbackSample / CWC_HOP_M);
      this.handleWorkletMessage({
        type: 'telemetry',
        playing: this.fallbackPlaying,
        currentSample: this.fallbackSample,
        currentFrame: curF,
        loadedFrames: dec.loadedCount,
        underruns: 0,
        cpuMs: 0.12,
        layerRms: Array.from(dec.layerRms),
        spectrum32: Array.from(dec.spectrum32),
      });
    };
    proc.connect(this.cwcGain);
    this.fallbackNode = proc;
    cwcStore.setState({ engineMode: 'Streaming PCM Node' });
  }

  handleWorkletMessage(msg) {
    if (!msg) return;
    if (msg.type === 'ended') {
      this.handleTrackEnded();
      return;
    }
    if (msg.type === 'telemetry') {
      const st = cwcStore.getState();
      if (st.audioSource !== 'cwc') return;
      this.liveTelemetry.currentSample = msg.currentSample;
      this.liveTelemetry.currentFrame = msg.currentFrame;
      if (msg.spectrum32) this.liveTelemetry.spectrum32.set(msg.spectrum32);
      if (msg.layerRms) this.liveTelemetry.layerRms.set(msg.layerRms);
      if (typeof msg.cpuMs === 'number') this.liveTelemetry.cpuMs = msg.cpuMs;

      const now = performance.now();
      if (now - this.lastUiSyncMs > 75) {
        this.lastUiSyncMs = now;
        const stems = st.stems.map((s, i) => ({
          ...s,
          rms: msg.layerRms ? msg.layerRms[i] : 0,
        }));
        cwcStore.setState({
          currentTimeSec: msg.currentSample / CWC_SAMPLE_RATE,
          currentFrame: msg.currentFrame,
          cpuMs: msg.cpuMs,
          underruns: msg.underruns || 0,
          stems,
        });
      }
    }
  }

  handleTrackEnded() {
    cwcStore.setState({
      isPlaying: false,
      currentTimeSec: 0,
      currentFrame: 0,
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
    if (this.fallbackDecoders) {
      for (const dec of Object.values(this.fallbackDecoders)) {
        dec.setStemControls(controls);
      }
      this.fallbackCachedFrame = -1;
    }
  }

  async ensureReferenceAudio() {
    if (this.refAudioEl) return this.refAudioEl;
    await this.ensureAudioContext();
    cwcStore.setState({ refLoading: true });
    const audio = new Audio('/ref-audio/master.mp3');
    audio.preload = 'auto';
    audio.crossOrigin = 'anonymous';

    await new Promise((resolve, reject) => {
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onErr = () => {
        cleanup();
        reject(new Error('Reference MP3 stream failed to load'));
      };
      const cleanup = () => {
        audio.removeEventListener('canplay', onReady);
        audio.removeEventListener('error', onErr);
      };
      audio.addEventListener('canplay', onReady);
      audio.addEventListener('error', onErr);
      audio.load();
    });

    this.refGain = this.ctx.createGain();
    this.refGain.gain.value = 0;
    this.refSourceNode = this.ctx.createMediaElementSource(audio);
    this.refSourceNode.connect(this.refGain);
    this.refGain.connect(this.analyser);

    audio.addEventListener('ended', () => {
      if (cwcStore.getState().audioSource === 'ref') this.handleTrackEnded();
    });

    this.refAudioEl = audio;
    cwcStore.setState({ refLoading: false, refLoaded: true });
    return audio;
  }

  /**
   * Verifies CRC-32 and activates a binary `.wec` container buffer directly in AudioWorklet.
   */
  async loadWecBuffer(inputBuf, fileName = 'custom.wec', sourceTier = '500k') {
    const t0 = performance.now();
    const bytes = inputBuf instanceof Uint8Array ? inputBuf : new Uint8Array(inputBuf);
    const bundle = unpackWecContainer(bytes);
    const unpackMs = Number((performance.now() - t0).toFixed(1));
    this.loadedWecBundle = bundle;

    const freshInspector = new CWCStreamDecoder(bundle.frameCount);
    freshInspector.ingestChunk(bundle);
    this.inspectorDecoders.wec = freshInspector;

    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'wec-bundle', chunkBundle: bundle });
    }
    if (this.fallbackDecoders) {
      const freshFb = new CWCStreamDecoder(bundle.frameCount);
      freshFb.ingestChunk(bundle);
      this.fallbackDecoders.wec = freshFb;
      this.syncStemsToWorklet();
      const curFrame = Math.floor((cwcStore.getState().currentTimeSec * CWC_SAMPLE_RATE) / CWC_HOP_M);
      freshFb.seekToFrame(curFrame);
      this.fallbackCachedFrame = -1;
    }

    const allTrue = new Array(8).fill(true);
    const st = cwcStore.getState();
    cwcStore.setState({
      activeTier: 'wec',
      chunksLoadedByTier: { ...st.chunksLoadedByTier, wec: allTrue },
      chunksLoaded: allTrue,
      loadedChunkCount: 8,
      loadedWecMeta: {
        fileName,
        sourceTier,
        bytes: bundle.header.fileBytes,
        crc32Hex: bundle.header.crc32Hex,
        tierLabel: bundle.header.tierLabel,
        entropyModeName: bundle.header.entropyModeName,
        unpackMs,
      },
      errorMsg: null,
    });

    if (st.audioSource !== 'cwc') {
      await this.setAudioSource('cwc');
    }
    return bundle;
  }

  async fetchAndLoadWec(baseTier = '500k') {
    const ed = CWC_EDITIONS[baseTier] || CWC_EDITIONS['500k'];
    const wecUrl = ed.wec?.file || `/bundles/next-intersection-${baseTier}.wec`;
    const fileName = wecUrl.split('/').pop();
    cwcStore.setState({ tierSwitching: true });
    try {
      let bytes = this.wecCache.get(wecUrl);
      if (!bytes) {
        const res = await fetch(wecUrl);
        if (!res.ok) throw new Error(`Failed to fetch ${wecUrl} (HTTP ${res.status})`);
        const ctype = (res.headers.get('content-type') || '').toLowerCase();
        if (ctype.includes('text/html')) {
          throw new Error(`Server returned HTML instead of .wec binary for ${wecUrl}`);
        }
        bytes = new Uint8Array(await res.arrayBuffer());
        this.wecCache.set(wecUrl, bytes);
      }
      await this.loadWecBuffer(bytes, fileName, baseTier);
    } catch (err) {
      cwcStore.setState({ errorMsg: `.wec load failed: ${err.message}` });
    } finally {
      cwcStore.setState({ tierSwitching: false });
    }
  }

  /**
   * Unified selector: '1m' | '500k' | 'hifi' | 'wec-500k' | 'wec-1m' | 'wec-hifi' | 'wec' | 'ref'
   */
  async selectEdition(mode) {
    if (mode === 'ref') {
      await this.setAudioSource('ref');
      return;
    }
    if (mode && mode.startsWith('wec-')) {
      await this.fetchAndLoadWec(mode.slice(4));
      return;
    }
    if (mode === 'wec') {
      if (!this.loadedWecBundle) {
        await this.fetchAndLoadWec('500k');
      } else {
        cwcStore.setState({
          activeTier: 'wec',
          chunksLoaded: new Array(8).fill(true),
          loadedChunkCount: 8,
        });
        if (this.workletNode) this.workletNode.port.postMessage({ type: 'tier', tier: 'wec' });
        if (cwcStore.getState().audioSource !== 'cwc') await this.setAudioSource('cwc');
      }
      return;
    }
    const tier = CWC_EDITIONS[mode] ? mode : '1m';
    const st = cwcStore.getState();
    const curFrame = Math.floor((st.currentTimeSec * CWC_SAMPLE_RATE) / CWC_HOP_M);
    const curChunk = this.chunkIndexForFrame(curFrame, tier);

    cwcStore.setState({ tierSwitching: true });
    try {
      await this.loadChunk(tier, curChunk);
      this.prefetchTierChunks(tier);

      const tierArr = cwcStore.getState().chunksLoadedByTier[tier] || new Array(8).fill(false);
      cwcStore.setState({
        activeTier: tier,
        chunksLoaded: [...tierArr],
        loadedChunkCount: tierArr.filter(Boolean).length,
      });

      if (this.workletNode) {
        this.workletNode.port.postMessage({ type: 'tier', tier });
      }
      if (this.fallbackDecoders && this.fallbackDecoders[tier]) {
        this.fallbackDecoders[tier].seekToFrame(curFrame);
        this.fallbackCachedFrame = -1;
      }

      if (st.audioSource !== 'cwc') {
        await this.setAudioSource('cwc');
      }
    } finally {
      cwcStore.setState({ tierSwitching: false });
    }
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
    if (this.cwcGain && this.ctx) {
      this.cwcGain.gain.setTargetAtTime(source === 'cwc' ? 1 : 0, this.ctx.currentTime, 0.015);
    }
    if (this.refGain && this.ctx) {
      this.refGain.gain.setTargetAtTime(source === 'ref' ? 0.98 : 0, this.ctx.currentTime, 0.015);
    }

    if (wasPlaying) {
      await this.seek(curSec);
      await this.play();
    }
  }

  async play() {
    await this.ensureAudioContext();
    const st = cwcStore.getState();
    const tier = st.activeTier;
    const curSample = Math.floor(st.currentTimeSec * CWC_SAMPLE_RATE);
    const targetChunk = this.chunkIndexForFrame(Math.floor(curSample / CWC_HOP_M), tier);
    await this.loadChunk(tier, targetChunk);
    const chunks = this.getEditionMeta(tier).chunks;
    if (targetChunk + 1 < chunks.length) this.loadChunk(tier, targetChunk + 1);

    if (st.audioSource === 'cwc') {
      if (this.refAudioEl && !this.refAudioEl.paused) this.refAudioEl.pause();
      if (this.workletNode) {
        this.workletNode.port.postMessage({ type: 'tier', tier });
        this.workletNode.port.postMessage({ type: 'play', sampleOffset: curSample });
      } else if (this.fallbackDecoders) {
        this.fallbackDecoders[tier].seekToFrame(Math.floor(curSample / CWC_HOP_M));
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
    const st = cwcStore.getState();
    const tier = st.activeTier;
    const clamped = Math.max(0, Math.min(CWC_TRACK_META.durationSec - 0.05, timeSec));
    const targetSample = Math.floor(clamped * CWC_SAMPLE_RATE);
    const targetFrame = Math.floor(targetSample / CWC_HOP_M);
    const chunkIdx = this.chunkIndexForFrame(targetFrame, tier);
    await this.loadChunk(tier, chunkIdx);

    this.liveTelemetry.currentSample = targetSample;
    this.liveTelemetry.currentFrame = targetFrame;
    cwcStore.setState({
      currentTimeSec: clamped,
      currentFrame: targetFrame,
      inspectedFrame: targetFrame,
    });

    if (this.workletNode) {
      this.workletNode.port.postMessage({ type: 'seek', sampleOffset: targetSample });
    } else if (this.fallbackDecoders) {
      this.fallbackDecoders[tier].seekToFrame(targetFrame);
      this.fallbackSample = targetSample;
      this.fallbackCachedFrame = -1;
    }
    if (this.refAudioEl && st.audioSource === 'ref') {
      this.refAudioEl.currentTime = clamped;
    }
  }

  startUiLoop() {
    if (this.rafId) return;
    const freqScratch = new Uint8Array(256);
    const tick = () => {
      const st = cwcStore.getState();
      if (!st.isPlaying) {
        this.rafId = null;
        return;
      }
      this.rafId = requestAnimationFrame(tick);
      if (this.analyser) {
        this.analyser.getFloatTimeDomainData(this.liveTelemetry.timeDomain);
      }
      if (st.audioSource === 'ref' && this.refAudioEl && st.isPlaying) {
        const t = this.refAudioEl.currentTime;
        const f = Math.floor((t * CWC_SAMPLE_RATE) / CWC_HOP_M);
        this.liveTelemetry.currentSample = Math.floor(t * CWC_SAMPLE_RATE);
        this.liveTelemetry.currentFrame = f;
        if (this.analyser) {
          this.analyser.getByteFrequencyData(freqScratch);
          for (let b = 0; b < 32; b++) {
            const idx = Math.min(freqScratch.length - 1, Math.floor(Math.pow(freqScratch.length, b / 32)));
            this.liveTelemetry.spectrum32[b] = freqScratch[idx] / 255;
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

  inspectFrame(frameIdx, tier = cwcStore.getState().activeTier) {
    const dec = this.inspectorDecoders[tier] || this.inspectorDecoders['1m'];
    return dec.inspectFrame(frameIdx);
  }

  async exportWav() {
    if (cwcStore.getState().exportingWav) return;
    const tier = cwcStore.getState().activeTier;
    const ed = this.getEditionMeta(tier);
    cwcStore.setState({ exportingWav: true, exportProgress: 0 });
    try {
      const st = cwcStore.getState();
      const dec = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
      if (tier === 'wec' && this.loadedWecBundle) {
        dec.ingestChunk(this.loadedWecBundle);
        cwcStore.setState({ exportProgress: 0.25 });
      } else {
        for (let i = 0; i < ed.chunks.length; i++) {
          await this.loadChunk(tier, i);
          cwcStore.setState({ exportProgress: ((i + 1) / ed.chunks.length) * 0.25 });
        }
        for (let i = 0; i < ed.chunks.length; i++) {
          const b = this.chunkBundleCache.get(`${tier}:${i}`);
          if (b) dec.ingestChunk(b);
        }
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
      const ws = (o, s) => {
        for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
      };
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
        if (f % 200 === 0) {
          cwcStore.setState({ exportProgress: 0.25 + 0.75 * (f / CWC_TRACK_META.totalFrames) });
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      const blob = new Blob([wav], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `CabronWEC_${tier.toUpperCase()}_NextIntersection_44100Hz.wav`;
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
