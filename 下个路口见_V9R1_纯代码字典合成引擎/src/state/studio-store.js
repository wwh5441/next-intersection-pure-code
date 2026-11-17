// V9R1 Studio Reactive State Store (`src/state/studio-store.js`)
import { STEM_MANIFEST } from '../dsp/v9r1-renderer.js';
import { DEFAULT_V9R1_TIER } from '../../data/tiers-registry.js';
import { SINGER_CORE } from '../../models/singer-core.js';

function createInitialStemStates() {
  const map = {};
  for (const m of STEM_MANIFEST) {
    map[m.id] = {
      gain: 1.0,
      muted: false,
      solo: false,
    };
  }
  return map;
}

const state = {
  tierId: DEFAULT_V9R1_TIER,
  isPlaying: false,
  currentTime: 0.0,
  durationSec: SINGER_CORE.durationSec,
  isRendering: true,
  renderPct: 0.02,
  renderText: '正在启动 V9R1 双字典纯代码合成工作线程...',
  renderStats: null,
  errorText: null,
  stemStates: createInitialStemStates(),
  inspectorTab: 'syllables',
  syllableFilter: '',
};

const listeners = new Set();

export const studioStore = {
  getState() {
    return state;
  },

  setState(patch) {
    Object.assign(state, patch);
    for (const fn of listeners) {
      fn(state);
    }
  },

  updateStem(stemId, patch) {
    const cur = state.stemStates[stemId];
    if (!cur) return;
    state.stemStates = {
      ...state.stemStates,
      [stemId]: { ...cur, ...patch },
    };
    for (const fn of listeners) {
      fn(state);
    }
  },

  resetStems() {
    state.stemStates = createInitialStemStates();
    for (const fn of listeners) {
      fn(state);
    }
  },

  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};
