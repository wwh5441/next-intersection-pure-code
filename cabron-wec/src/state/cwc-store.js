// CabronWEC (CWC-5X) Observable Application Store
import { CWC_STEMS } from '../dsp/mdct-core.js';

class CWCStore {
  constructor() {
    this.state = {
      isReady: false,
      isPlaying: false,
      audioSource: 'cwc', // 'cwc' | 'ref'
      activeTier: '1m', // '1m' (<1MB) | '500k' (<500KB) | 'hifi' (2.3MB Reference)
      tierSwitching: false,
      refLoading: false,
      refLoaded: false,
      engineMode: 'AudioWorklet (128-sample)',
      currentTimeSec: 0,
      durationSec: 210.373,
      currentFrame: 0,
      totalFrames: 4532,
      inspectedFrame: 0,
      chunksLoadedByTier: {
        '1m': new Array(8).fill(false),
        '500k': new Array(8).fill(false),
        hifi: new Array(8).fill(false),
        wec: new Array(8).fill(false),
      },
      loadedWecMeta: null,
      chunksLoaded: new Array(8).fill(false),
      loadedChunkCount: 0,
      cpuMs: null,
      underruns: 0,
      masterGain: 0.98,
      exportingWav: false,
      exportProgress: 0,
      activeDeckTab: 'metrics',
      errorMsg: null,
      stems: CWC_STEMS.map((s) => ({
        ...s,
        gain: 1.0,
        muted: false,
        solo: false,
        rms: 0,
      })),
    };
    this.listeners = new Set();
  }

  getState() {
    return this.state;
  }

  setState(patch) {
    this.state = { ...this.state, ...patch };
    for (const fn of this.listeners) {
      fn(this.state);
    }
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

export const cwcStore = new CWCStore();
