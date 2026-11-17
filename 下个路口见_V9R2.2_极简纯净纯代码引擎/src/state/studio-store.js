// Reactive Studio State Store (`src/state/studio-store.js`)
export function createStudioStore(layers) {
  const layerParams = {};
  for (const l of layers) {
    layerParams[l.id] = { gain: l.defaultGain ?? 1.0, mute: false, solo: false };
  }

  let state = {
    profileId: 'master',
    rendering: false,
    progress: 0.0,
    statusText: '待命 (点击合成或播放启动 48kHz 十一轨无损重构)',
    ready: false,
    synthMs: 0,
    durationSec: 210.373,
    playing: false,
    currentTime: 0.0,
    highlightedSyllableId: null,
    layerParams,
  };

  const listeners = new Set();

  return {
    getState() {
      return state;
    },
    setState(patch) {
      state = { ...state, ...patch };
      for (const fn of listeners) fn(state);
    },
    updateLayer(layerId, patch) {
      const next = {
        ...state.layerParams,
        [layerId]: { ...state.layerParams[layerId], ...patch },
      };
      state = { ...state, layerParams: next };
      for (const fn of listeners) fn(state);
    },
    resetAllLayers() {
      const next = {};
      for (const l of layers) {
        next[l.id] = { gain: 1.0, mute: false, solo: false };
      }
      state = { ...state, layerParams: next };
      for (const fn of listeners) fn(state);
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
