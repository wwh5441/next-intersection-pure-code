// CabronWEC (CWC-4X) Reactive Studio Store

import { CWC_STEMS } from '../dsp/mdct-core.js';

const initialState = {
  isReady: false,
  isPlaying: false,
  audioSource: 'cwc', // 'cwc' | 'ref' (A/B comparison against original MP3)
  refLoading: false,
  refLoaded: false,
  currentTimeSec: 0,
  durationSec: 210.373,
  currentFrame: 0,
  totalFrames: 18122,
  masterGain: 0.98,
  stems: CWC_STEMS.map((s) => ({
    ...s,
    gain: 1.0,
    muted: false,
    solo: false,
    rms: 0,
  })),
  chunksLoaded: new Array(8).fill(false),
  loadedChunkCount: 0,
  cpuMs: null,
  underruns: 0,
  engineMode: 'AudioWorklet',
  activeDeckTab: 'equations', // 'equations' | 'metrics' | 'lyrics'
  inspectedFrame: 5000,
  exportingWav: false,
  exportProgress: 0,
  errorMsg: null,
};

class StudioStore {
  constructor() {
    this.state = initialState;
    this.listeners = new Set();
  }

  getState() {
    return this.state;
  }

  setState(patch) {
    this.state = typeof patch === 'function' ? patch(this.state) : { ...this.state, ...patch };
    for (const fn of this.listeners) fn(this.state);
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

export const cwcStore = new StudioStore();
