// Reactive State Store for V9.4.1..V9.4.7 7-Tier Rate-Distortion Pure-Math Studio

export function createStudioStore(initialConfig = {}) {
  const state = {
    studioMode: initialConfig.studioMode || 'standalone', // 'unified' | 'standalone'
    activeTierId: initialConfig.activeTierId || 'v94_1',
    loadingTierId: null,
    tiersCatalog: initialConfig.tiersCatalog || [],
    loadedTierIds: [],
    activeTierMeta: initialConfig.activeTierMeta || null,

    isReady: false,
    isPlaying: false,
    isExportingWav: false,
    statusText: '正在挂载 48kHz 纯数学正交余弦流式引擎...',
    currentTime: 0.0,
    duration: 210.373333,
    bpm: 98.0,
    bar0Sec: 1.7493,
    renderTimeMs: 0,
    pcmRamMb: 0.38,

    channelMode: 'stereo', // 'stereo' | 'mid' | 'side'
    masterGainDb: 0.0,
    stereoWidthPct: 100,

    activeInspectorTab: 'equations',
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

    resetMonitorControls() {
      state.channelMode = 'stereo';
      state.masterGainDb = 0.0;
      state.stereoWidthPct = 100;
      notify(['channelMode', 'masterGainDb', 'stereoWidthPct']);
    },

    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
