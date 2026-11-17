// Reactive State Store for V9.3 48kHz Near-Lossless Orthonormal Pure-Math Studio

import { STEM_MANIFEST } from '../dsp/song-renderer.js';

export function createStudioStore() {
  const stemStates = {};
  for (const m of STEM_MANIFEST) {
    stemStates[m.id] = {
      gainDb: 0.0,
      muted: false,
      solo: false,
    };
  }

  const state = {
    versionLabel: 'V9.3',
    engineMode: 'full-buffer', // V9.3: full 6-stem pre-render; V9.4 upgrades to live streaming
    isReady: false,
    isPlaying: false,
    isExportingWav: false,
    progress: 0.0,
    statusText: '正在初始化 48kHz 正交局部余弦级数求解器...',
    currentTime: 0.0,
    duration: 210.373,
    bpm: 98.0,
    bar0Sec: 1.7493,
    renderTimeMs: 0,
    peakAmplitude: 0.94,
    pcmRamMb: 462.25,
    presetMode: 'all', // 'all' | 'vocal' | 'band'
    activeInspectorTab: 'equations',
    metrics: null,
    stemStates,
  };

  const listeners = new Set();

  function notify(changedKeys) {
    for (const fn of listeners) {
      fn(state, changedKeys);
    }
  }

  return {
    getState() {
      return state;
    },

    update(patch) {
      const keys = Object.keys(patch);
      for (const k of keys) {
        state[k] = patch[k];
      }
      notify(keys);
    },

    updateStem(stemId, patch) {
      const cur = state.stemStates[stemId];
      if (!cur) return;
      Object.assign(cur, patch);
      state.presetMode = 'custom';
      notify(['stemStates', 'presetMode']);
    },

    applyGroupPreset(mode) {
      state.presetMode = mode;
      for (const m of STEM_MANIFEST) {
        const st = state.stemStates[m.id];
        st.solo = false;
        if (mode === 'all') {
          st.muted = false;
        } else if (mode === 'vocal') {
          st.muted = m.group !== 'vocal';
        } else if (mode === 'band') {
          st.muted = m.group !== 'band';
        }
      }
      notify(['stemStates', 'presetMode']);
    },

    resetAllFaders() {
      state.presetMode = 'all';
      for (const m of STEM_MANIFEST) {
        const st = state.stemStates[m.id];
        st.gainDb = 0.0;
        st.muted = false;
        st.solo = false;
      }
      notify(['stemStates', 'presetMode']);
    },

    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
