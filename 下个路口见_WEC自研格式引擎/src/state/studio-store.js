// Reactive Unidirectional Store for WEC v2.0 (WEC0 / WEC1 / WEC2 Opus Multi-Stem & Feature Studio)

export function buildDefaultStemStates(trackCount) {
  const arr = [];
  for (let i = 0; i < trackCount; i++) {
    arr.push({ gainDb: 0.0, pan: 0.0, muted: false, solo: false });
  }
  return arr;
}

export function createStudioStore(catalog, initialTierMeta) {
  const initTracks = initialTierMeta ? initialTierMeta.audioTracks || 1 : 1;
  const state = {
    isReady: false,
    statusText: '正在载入 WEC v2 原生 Opus 容器...',
    renderTimeMs: 0,
    pcmRamMb: 0.0,
    bpm: 98.0,
    bar0Sec: 1.7493,
    loadingTierId: '',
    loadedTierIds: [],
    errorMsg: '',
    isPlaying: false,
    isExportingWav: false,
    currentTime: 0.0,
    duration: initialTierMeta ? initialTierMeta.durationSec || 210.373333 : 210.373333,
    tiersCatalog: catalog || [],
    familyFilter: 'ALL', // 'ALL' | 'WEC0' | 'WEC1' | 'WEC2'
    activeTierId: initialTierMeta ? initialTierMeta.tierId : 'wec2_6plus4',
    activeTierMeta: initialTierMeta || null,
    isCustomFile: false,
    stemStates: buildDefaultStemStates(initTracks),
    activeSceneId: 'master',
    masterGainDb: 0.0,
    stereoWidthPct: 100,
    channelMode: 'stereo',
    activeInspectorTab: 'packet', // 'packet' | 'manifest' | 'hex' | 'spec'
    followPlayheadFrame: true,
    inspectedPacketIdx: 3080, // ~61.6s chorus
  };

  const listeners = new Set();

  function getState() {
    return state;
  }

  function update(patch) {
    const changedKeys = [];
    for (const [k, v] of Object.entries(patch)) {
      if (state[k] !== v) {
        state[k] = v;
        changedKeys.push(k);
      }
    }
    if (changedKeys.length > 0) {
      for (const fn of listeners) {
        fn(state, changedKeys);
      }
    }
  }

  function setStemPatch(trackIndex, patch) {
    const next = state.stemStates.map((s, idx) => (idx === trackIndex ? { ...s, ...patch } : s));
    update({ stemStates: next, activeSceneId: 'custom' });
  }

  function applyScenePreset(sceneId, muteIndices = [], soloIndices = []) {
    const muteSet = new Set(muteIndices);
    const soloSet = new Set(soloIndices);
    const next = state.stemStates.map((s, idx) => ({
      ...s,
      gainDb: 0.0,
      muted: muteSet.has(idx),
      solo: soloSet.has(idx),
    }));
    update({ stemStates: next, activeSceneId: sceneId });
  }

  function resetMonitorControls() {
    const next = buildDefaultStemStates(state.stemStates.length);
    update({
      stemStates: next,
      activeSceneId: 'master',
      masterGainDb: 0.0,
      stereoWidthPct: 100,
      channelMode: 'stereo',
    });
  }

  function subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  return {
    getState,
    update,
    setStemPatch,
    applyScenePreset,
    resetMonitorControls,
    subscribe,
  };
}
