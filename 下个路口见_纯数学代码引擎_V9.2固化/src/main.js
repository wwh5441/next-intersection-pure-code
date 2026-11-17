// Main Entry Point for 《李宇春 - 下个，路口，见》 V9.1 Dual-Tier Pure Mathematical Code Studio

import { createStudioStore } from './state/studio-store.js';
import { TenTrackAudioMixer } from './modules/audio-mixer.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountScoreCanvas } from './components/score-canvas.js';
import { mountStemMixer } from './components/stem-mixer.js';
import { mountFormulaInspector } from './components/formula-inspector.js';
import { renderFullPureMathSong } from './dsp/song-renderer.js';

async function loadMetricsIfAvailable(store) {
  if (window.__INLINE_CODE_METRICS__) {
    store.update({ metrics: window.__INLINE_CODE_METRICS__ });
    return;
  }
  try {
    const res = await fetch('./data/code-metrics.json', { cache: 'no-store' });
    if (res.ok) {
      const metrics = await res.json();
      store.update({ metrics });
    }
  } catch (_) {
    // Optional metrics badge fallback already populated
  }
}

function createSynthesisCoordinator(store, mixer) {
  let worker = null;

  function ensureWorker() {
    if (worker) return worker;
    let workerUrl = null;
    if (window.__PURE_MATH_WORKER_SOURCE__) {
      const blob = new Blob([window.__PURE_MATH_WORKER_SOURCE__], { type: 'text/javascript' });
      workerUrl = URL.createObjectURL(blob);
    } else {
      workerUrl = new URL('./dsp/math-worker.js', import.meta.url);
    }
    worker = new Worker(workerUrl, { type: 'module' });
    worker.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === 'progress') {
        store.update({ progress: msg.progress, statusText: msg.text });
      } else if (msg.type === 'done') {
        applyFinishedResult(msg);
      } else if (msg.type === 'error') {
        console.warn('Worker fallback to main thread:', msg.message);
        const syncRes = renderFullPureMathSong(msg.tierId, (p, t) =>
          store.update({ progress: p, statusText: t })
        );
        applyFinishedResult(syncRes);
      }
    };
    worker.onerror = () => {
      const targetTier = store.getState().vocalTier;
      const syncRes = renderFullPureMathSong(targetTier, (p, t) =>
        store.update({ progress: p, statusText: t })
      );
      applyFinishedResult(syncRes);
    };
    return worker;
  }

  function applyFinishedResult(data) {
    const tierId = data.tierId || store.getState().vocalTier;
    mixer.loadSynthesizedStems(data.sampleRate, data.numSamples, data.stems, tierId, {
      renderTimeMs: data.renderTimeMs,
      peakAmplitude: data.peakAmplitude,
    });
    const prevCached = store.getState().cachedTiers;
    const cachedTiers = prevCached.includes(tierId) ? prevCached : [...prevCached, tierId];
    store.update({
      vocalTier: tierId,
      cachedTiers,
      isReady: true,
      isSwitchingTier: false,
      progress: 1.0,
      duration: data.durationSec,
      bpm: data.bpm,
      bar0Sec: data.bar0Sec,
      renderTimeMs: data.renderTimeMs,
      peakAmplitude: data.peakAmplitude,
      statusText: '10 轨纯数学合成就绪',
    });
  }

  async function selectVocalTier(tierId) {
    const st = store.getState();
    if (st.isSwitchingTier) return;
    if (!st.availableTiers.includes(tierId)) return;

    if (mixer.hasCachedTier(tierId)) {
      const meta = mixer.getCachedTierMeta(tierId);
      store.update({
        vocalTier: tierId,
        renderTimeMs: meta.renderTimeMs,
        peakAmplitude: meta.peakAmplitude,
        statusText: '已瞬时切换内存缓存档位 (0ms 无缝衔接)',
      });
      await mixer.activateCachedTier(tierId);
      return;
    }

    store.update({
      vocalTier: tierId,
      isSwitchingTier: true,
      progress: 0.05,
      statusText: '正在求解新档位人声声学微分方程 (复用 5 轨伴奏缓存)...',
    });

    try {
      const w = ensureWorker();
      w.postMessage({ cmd: 'render', tierId });
    } catch (_) {
      setTimeout(() => {
        const syncRes = renderFullPureMathSong(tierId, (p, t) =>
          store.update({ progress: p, statusText: t })
        );
        applyFinishedResult(syncRes);
      }, 20);
    }
  }

  return { selectVocalTier };
}

function bootstrapStudio() {
  const store = createStudioStore();
  const mixer = new TenTrackAudioMixer(store);
  const coordinator = createSynthesisCoordinator(store, mixer);

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer, coordinator);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer);
  mountStemMixer(document.getElementById('mixer-slot'), store, mixer);
  mountFormulaInspector(document.getElementById('inspector-slot'), store);

  window.__STUDIO_DEBUG__ = { store, mixer, coordinator };

  loadMetricsIfAvailable(store);
  coordinator.selectVocalTier(store.getState().vocalTier);
}

bootstrapStudio();
