// Reactive Studio State Store (`src/state/studio-store.js`)
import { DEFAULT_V9R3_TIER } from '../../data/tiers-registry.js';
import { SONG_STRUCTURE } from '../../data/song-structure.js';
import { V9R3_STEM_SPECS } from '../dsp/engine-orchestrator.js';

function createInitialStemStates() {
  const map = {};
  for (const s of V9R3_STEM_SPECS) {
    map[s.id] = { solo: false, mute: false, gain: 1.0 };
  }
  return map;
}

class StudioStore {
  constructor() {
    this.state = {
      tierId: DEFAULT_V9R3_TIER,
      pitchSemitones: 0,
      isSynthesizing: false,
      synthProgress: 0,
      synthStatus: '准备就绪',
      elapsedSynthMs: 0,
      isPlaying: false,
      currentTime: 0,
      duration: SONG_STRUCTURE.durationSec || 210.3733,
      stemStates: createInitialStemStates(),
      activeInspectorTab: 'vocal_voiced',
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

  updateStem(stemId, patch) {
    const prev = this.state.stemStates[stemId];
    if (!prev) return;
    this.setState({
      stemStates: {
        ...this.state.stemStates,
        [stemId]: { ...prev, ...patch },
      },
    });
  }

  setStemPreset(activeIds = null) {
    const next = {};
    for (const s of V9R3_STEM_SPECS) {
      const prev = this.state.stemStates[s.id];
      if (!activeIds) {
        next[s.id] = { ...prev, solo: false, mute: false, gain: 1.0 };
      } else {
        next[s.id] = { ...prev, solo: activeIds.includes(s.id), mute: false };
      }
    }
    this.setState({ stemStates: next });
  }

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

export const studioStore = new StudioStore();
