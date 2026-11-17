import { loadVocalWaveStreams } from './loader.js';

export class VocalAudioController {
  constructor(manifest, store) {
    this.manifest = manifest;
    this.store = store;
    this.store.initTracks(manifest.tracks);
  }

  get running() {
    return this.store.state.running;
  }

  async ensureContext() {
    if (!this.context) {
      this.context = new AudioContext({ sampleRate: 48000, latencyHint: 'playback' });
      this.master = this.context.createGain();
      this.master.gain.value = this.store.state.volume;
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 2048;
      this.master.connect(this.analyser).connect(this.context.destination);
      const workletUrl = window.__VFS_WORKLET_DATA_URL || new URL('../dsp/worklet.js', import.meta.url);
      this.ready = this.context.audioWorklet.addModule(workletUrl);
    }
    await this.context.resume();
    await this.ready;
  }

  async ensureSynth() {
    if (this.node) return;
    this.store.patch({ loading: true, loadProgress: null });
    try {
      const payloads = await loadVocalWaveStreams(this.manifest, this.store.state.profile, (n, total) => {
        this.store.patch({ loadProgress: [n, total] });
      });
      this.node = new AudioWorkletNode(this.context, 'five-track-vocal-synth', {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        processorOptions: { manifest: this.manifest, payloads },
      });
      this.node.onprocessorerror = () => {
        this.onError?.(new Error('五轨人声实时合成线程遇到异常，请刷新重试。'));
      };
      this.node.port.onmessage = ({ data }) => {
        this.store.patch({
          time: data.time,
          running: data.running,
          computed: data.computed,
          meters: data.meters,
        });
      };
      this.node.connect(this.master);
      this.node.port.postMessage({ type: 'seek', time: this.store.state.time });
      this.applyGains();
    } finally {
      this.store.patch({ loading: false, loadProgress: null });
    }
  }

  async play() {
    await this.ensureContext();
    if (this.store.state.time >= this.manifest.duration - 0.05) {
      this.seek(0);
    }
    await this.ensureSynth();
    if (this.context.state !== 'running') {
      await this.context.resume();
    }
    this.node.port.postMessage({ type: 'play' });
    this.store.patch({ running: true });
  }

  pause() {
    this.node?.port.postMessage({ type: 'pause' });
    this.store.patch({ running: false });
  }

  seek(time) {
    const clamped = Math.max(0, Math.min(this.manifest.duration, time));
    this.node?.port.postMessage({ type: 'seek', time: clamped });
    this.store.patch({ time: clamped });
  }

  setVolume(value) {
    this.store.patch({ volume: value });
    if (this.master) {
      this.master.gain.setTargetAtTime(value, this.context.currentTime, 0.015);
    }
  }

  async setProfile(profile) {
    if (profile === this.store.state.profile) return;
    const wasPlaying = this.store.state.running;
    this.pause();
    if (this.node) {
      this.node.port.postMessage({ type: 'dispose' });
      this.node.disconnect();
      this.node.port.close();
      this.node = null;
    }
    const response = await fetch(`./data/${profile}/manifest.json`);
    if (!response.ok) throw new Error(`未能读取人声波码档位: ${profile}`);
    this.manifest = await response.json();
    this.store.initTracks(this.manifest.tracks);
    this.store.patch({ profile, meters: {} });
    this.onProfileChanged?.(this.manifest);
    if (wasPlaying) await this.play();
  }

  applyPreset(preset) {
    const enabled = {};
    for (const track of this.manifest.tracks) {
      enabled[track.id] = preset.tracks.includes(track.id);
    }
    this.store.patch({ enabled, solo: null, activePreset: preset.id });
    this.applyGains();
  }

  applyGains() {
    const { enabled, gains, solo } = this.store.state;
    for (const track of this.manifest.tracks) {
      const audible = enabled[track.id] && (!solo || solo === track.id);
      this.node?.port.postMessage({
        type: 'gain',
        track: track.id,
        value: audible ? gains[track.id] : 0,
      });
    }
  }

  toggleMute(id) {
    const next = { ...this.store.state.enabled, [id]: !this.store.state.enabled[id] };
    this.store.patch({ enabled: next, activePreset: null });
    this.applyGains();
  }

  toggleSolo(id) {
    const solo = this.store.state.solo === id ? null : id;
    this.store.patch({ solo, activePreset: null });
    this.applyGains();
  }

  setTrackGain(id, value) {
    const next = { ...this.store.state.gains, [id]: value };
    this.store.patch({ gains: next });
    this.applyGains();
  }
}
