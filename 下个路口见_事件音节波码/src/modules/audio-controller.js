export class MultiLayerAudioController {
  constructor(layerSpecs, onEnded) {
    this.layerSpecs = layerSpecs;
    this.onEnded = onEnded;
    this.ctx = null;
    this.masterGain = null;
    this.limiter = null;
    this.layerGains = new Map();
    this.layerAnalysers = new Map();
    this.audioBuffers = new Map();
    this.activeSources = [];

    this.duration = 0;
    this.playing = false;
    this.offsetSec = 0;
    this.startedAtCtxTime = 0;

    this.layerParams = new Map();
    for (const spec of layerSpecs) {
      this.layerParams.set(spec.id, {
        gain: spec.defaultGain ?? 1.0,
        muted: false,
        solo: false,
      });
    }
    this._meterScratch = new Float32Array(256);
  }

  _ensureContext(sampleRate = 48000) {
    if (!this.ctx) {
      const AudioCtx = globalThis.AudioContext || globalThis.webkitAudioContext;
      this.ctx = new AudioCtx({ sampleRate });

      // Studio soft ceiling limiter + -1.6dB headroom so 4-layer transient peaks never clip DAC
      this.limiter = this.ctx.createDynamicsCompressor();
      this.limiter.threshold.value = -2.0;
      this.limiter.knee.value = 6.0;
      this.limiter.ratio.value = 10.0;
      this.limiter.attack.value = 0.002;
      this.limiter.release.value = 0.09;

      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.84;
      this.masterGain.connect(this.limiter);
      this.limiter.connect(this.ctx.destination);

      for (const spec of this.layerSpecs) {
        const gNode = this.ctx.createGain();
        const analyser = this.ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.72;
        gNode.connect(analyser);
        analyser.connect(this.masterGain);
        this.layerGains.set(spec.id, gNode);
        this.layerAnalysers.set(spec.id, analyser);
      }
      this._applyLayerGains();
    }
  }

  loadSynthesizedLayers(layersObj, sampleRate, numSamples) {
    const wasPlaying = this.playing;
    const resumePos = this.getCurrentTime();
    if (wasPlaying) this.pause();

    this._ensureContext(sampleRate);
    this.audioBuffers.clear();
    for (const spec of this.layerSpecs) {
      const pair = layersObj[spec.id];
      if (!pair) continue;
      const buf = this.ctx.createBuffer(2, numSamples, sampleRate);
      buf.copyToChannel(pair[0], 0);
      buf.copyToChannel(pair[1], 1);
      this.audioBuffers.set(spec.id, buf);
    }
    this.duration = numSamples / sampleRate;
    this.offsetSec = Math.min(resumePos, Math.max(0, this.duration - 0.05));
    if (wasPlaying) {
      this.play(this.offsetSec);
    }
  }

  _applyLayerGains() {
    if (!this.ctx) return;
    let anySolo = false;
    for (const p of this.layerParams.values()) {
      if (p.solo) anySolo = true;
    }
    const now = this.ctx.currentTime;
    for (const [id, gNode] of this.layerGains.entries()) {
      const p = this.layerParams.get(id);
      const audible = anySolo ? p.solo : !p.muted;
      const target = audible ? p.gain : 0.0;
      gNode.gain.setTargetAtTime(target, now, 0.012);
    }
  }

  setLayerParam(layerId, patch) {
    const cur = this.layerParams.get(layerId);
    if (!cur) return;
    Object.assign(cur, patch);
    this._applyLayerGains();
  }

  getLayerParamsSnapshot() {
    const snap = {};
    for (const [k, v] of this.layerParams.entries()) {
      snap[k] = { ...v };
    }
    return snap;
  }

  async play(fromSec = this.offsetSec) {
    if (!this.audioBuffers.size) return;
    this._ensureContext();
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }
    this._stopSourcesOnly();

    const startOffset = Math.max(0, Math.min(fromSec, Math.max(0, this.duration - 0.02)));
    this.offsetSec = startOffset;
    this.startedAtCtxTime = this.ctx.currentTime;
    this.playing = true;

    let firstSource = null;
    for (const [id, buf] of this.audioBuffers.entries()) {
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(this.layerGains.get(id));
      src.start(0, startOffset);
      this.activeSources.push(src);
      if (!firstSource) firstSource = src;
    }

    if (firstSource) {
      const expectedRef = firstSource;
      firstSource.onended = () => {
        if (this.playing && this.activeSources.includes(expectedRef)) {
          const elapsed = this.getCurrentTime();
          if (elapsed >= this.duration - 0.15) {
            this.playing = false;
            this.offsetSec = 0;
            this.activeSources = [];
            if (this.onEnded) this.onEnded();
          }
        }
      };
    }
  }

  pause() {
    if (!this.playing) return;
    this.offsetSec = this.getCurrentTime();
    this.playing = false;
    this._stopSourcesOnly();
  }

  async seek(targetSec) {
    const clamped = Math.max(0, Math.min(targetSec, this.duration));
    this.offsetSec = clamped;
    if (this.playing) {
      await this.play(clamped);
    }
  }

  _stopSourcesOnly() {
    for (const s of this.activeSources) {
      try {
        s.onended = null;
        s.stop();
        s.disconnect();
      } catch {
        // ignore stopped source
      }
    }
    this.activeSources = [];
  }

  getCurrentTime() {
    if (!this.playing || !this.ctx) return this.offsetSec;
    return Math.min(this.duration, this.offsetSec + (this.ctx.currentTime - this.startedAtCtxTime));
  }

  getLayerRmsLevels() {
    const out = {};
    for (const [id, analyser] of this.layerAnalysers.entries()) {
      if (!this.playing) {
        out[id] = 0;
        continue;
      }
      analyser.getFloatTimeDomainData(this._meterScratch);
      let sq = 0;
      for (let i = 0; i < this._meterScratch.length; i++) {
        sq += this._meterScratch[i] * this._meterScratch[i];
      }
      out[id] = Math.min(1.0, Math.sqrt(sq / this._meterScratch.length) * 4.2);
    }
    return out;
  }
}
