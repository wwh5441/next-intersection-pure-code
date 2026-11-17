// Unified 7-Tier Interactive Comparison Studio Entry Point (V9.4.1 原母带 Studio Master .. V9.4.7 稀疏核 Sparse Core)
import { TIERS_CATALOG } from '../data/tiers-catalog.js';
import { createStudioStore } from './state/studio-store.js';
import { StudioAudioMixer } from './modules/audio-mixer.js';
import { createTierEngineRegistry } from './modules/tier-loader.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountTierSelectorBar } from './components/tier-selector-bar.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountScoreCanvas } from './components/score-canvas.js';
import { mountStereoAndLadder } from './components/stereo-and-ladder.js';
import { mountFormulaInspector } from './components/formula-inspector.js';

async function bootstrapUnifiedStudio() {
  const tBoot = performance.now();
  const defaultMeta = TIERS_CATALOG[0]; // Boot directly on ① V9.4.1 原母带 Studio Master!
  const registry = createTierEngineRegistry();

  const store = createStudioStore({
    studioMode: 'unified',
    activeTierId: defaultMeta.tierId,
    tiersCatalog: TIERS_CATALOG,
    activeTierMeta: defaultMeta,
  });
  const mixer = new StudioAudioMixer(store, null);

  async function selectTier(tierId) {
    const curSt = store.getState();
    if (curSt.loadingTierId) return;
    if (curSt.isReady && curSt.activeTierId === tierId && mixer.streamEngine) return;

    const meta = TIERS_CATALOG.find((x) => x.tierId === tierId);
    if (!meta) return;

    const t0 = performance.now();
    store.update({
      loadingTierId: tierId,
      statusText: `正在加载 ${meta.versionCode}【${meta.tierName}】纯数学模块...`,
    });

    try {
      const eng = await registry.loadTierEngine(tierId);
      const warmL = new Float32Array(2048);
      const warmR = new Float32Array(2048);
      eng.synthesizeSampleRange(48000, 2048, null, warmL, warmR);
      const loadMs = Math.max(1, Math.round(performance.now() - t0));

      mixer.setActiveEngine(eng, true);
      store.update({
        activeTierId: tierId,
        activeTierMeta: meta,
        loadingTierId: null,
        loadedTierIds: registry.getCachedIds(),
        isReady: true,
        duration: eng.durationSec,
        renderTimeMs: loadMs,
        statusText: `${meta.versionCode}【${meta.tierName}】流式就绪`,
      });
    } catch (err) {
      console.error('Tier switch error:', err);
      store.update({
        loadingTierId: null,
        statusText: `加载失败: ${String(err)}`,
      });
    }
  }

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountTierSelectorBar(document.getElementById('tier-selector-slot'), store, selectTier);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer);
  mountStereoAndLadder(document.getElementById('mixer-slot'), store, mixer, selectTier);
  mountFormulaInspector(document.getElementById('inspector-slot'), store, () => mixer.streamEngine);

  await selectTier(defaultMeta.tierId);
  const totalBootMs = Math.max(1, Math.round(performance.now() - tBoot));
  store.update({ renderTimeMs: totalBootMs });

  window.__STUDIO_DEBUG__ = { store, mixer, registry, selectTier };
}

bootstrapUnifiedStudio();
