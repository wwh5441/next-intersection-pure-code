// Main Entry for WEC v2.0 (Opus-Powered WEC0 / WEC1 / WEC2 Multi-Stem & Feature Studio)

import { WEC_CATALOG } from '../data/wec-catalog.js';
import { createStudioStore, buildDefaultStemStates } from './state/studio-store.js';
import { createWecEngineRegistry } from './modules/wec-loader.js';
import { StudioAudioMixer } from './modules/audio-mixer.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountWecDropzoneBar } from './components/wec-dropzone-bar.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountScoreCanvas } from './components/score-canvas.js';
import { mountCompressionLadder } from './components/compression-ladder.js';
import { mountWecDecompilerPanel } from './components/wec-decompiler-panel.js';

async function bootstrapWecStudio() {
  const defaultMeta = WEC_CATALOG.find((x) => x.tierId === 'wec2_6plus4') || WEC_CATALOG[0];
  const store = createStudioStore(WEC_CATALOG, defaultMeta);
  const registry = createWecEngineRegistry(WEC_CATALOG);
  const mixer = new StudioAudioMixer(store, null);

  let activeBundle = null;
  const getActiveBundle = () => activeBundle;

  async function activateCatalogTier(tierId) {
    if (store.getState().loadingTierId) return;
    const t0 = performance.now();
    store.update({ loadingTierId: tierId, errorMsg: '' });
    try {
      const bundle = await registry.loadCatalogTier(tierId);
      activeBundle = bundle;
      const trackCount = bundle.parsed.tracks.length;
      store.update({
        stemStates: buildDefaultStemStates(trackCount),
        activeSceneId: 'master',
      });
      mixer.setActiveEngine(bundle.engine, true);
      const dt = Math.max(1, Math.round(performance.now() - t0));
      store.update({
        isReady: true,
        loadingTierId: '',
        activeTierId: tierId,
        activeTierMeta: bundle.meta,
        isCustomFile: false,
        duration: bundle.engine.durationSec,
        bpm: bundle.engine.bpm || 98.0,
        pcmRamMb: bundle.engine.pcmRamMb,
        renderTimeMs: dt,
        loadedTierIds: registry.getCachedIds(),
      });
    } catch (err) {
      store.update({
        loadingTierId: '',
        errorMsg: err && err.message ? err.message : String(err),
      });
    }
  }

  async function activateLocalWecFile(file) {
    const t0 = performance.now();
    store.update({ loadingTierId: file.name, errorMsg: '' });
    try {
      const bundle = await registry.loadLocalWecFile(file);
      activeBundle = bundle;
      const trackCount = bundle.parsed.tracks.length;
      store.update({
        stemStates: buildDefaultStemStates(trackCount),
        activeSceneId: 'master',
      });
      mixer.setActiveEngine(bundle.engine, true);
      const dt = Math.max(1, Math.round(performance.now() - t0));
      store.update({
        isReady: true,
        loadingTierId: '',
        activeTierId: bundle.meta.tierId,
        activeTierMeta: bundle.meta,
        isCustomFile: true,
        duration: bundle.engine.durationSec,
        bpm: bundle.engine.bpm || 98.0,
        pcmRamMb: bundle.engine.pcmRamMb,
        renderTimeMs: dt,
      });
    } catch (err) {
      store.update({
        loadingTierId: '',
        errorMsg: err && err.message ? err.message : String(err),
      });
    }
  }

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountWecDropzoneBar(document.getElementById('tier-selector-slot'), store, activateCatalogTier, activateLocalWecFile);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer, getActiveBundle);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer, getActiveBundle);
  mountCompressionLadder(document.getElementById('mixer-slot'), store, mixer, activateCatalogTier, getActiveBundle);
  mountWecDecompilerPanel(document.getElementById('inspector-slot'), store, getActiveBundle);

  await activateCatalogTier(defaultMeta.tierId);
}

bootstrapWecStudio();
