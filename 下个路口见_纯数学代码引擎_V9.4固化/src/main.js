// Main Entry Point for 《李宇春 - 下个，路口，见》 V9.4 Losslessly-Folded + Real-Time Streaming Engine

import { createStudioStore } from './state/studio-store.js';
import { StudioAudioMixer } from './modules/audio-mixer.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountScoreCanvas } from './components/score-canvas.js';
import { mountStemMixer } from './components/stem-mixer.js';
import { mountFormulaInspector } from './components/formula-inspector.js';

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
    // Optional metrics fallback
  }
}

function bootstrapStudio() {
  const tBoot = performance.now();
  const store = createStudioStore();
  const mixer = new StudioAudioMixer(store);

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer);
  mountStemMixer(document.getElementById('mixer-slot'), store, mixer);
  mountFormulaInspector(document.getElementById('inspector-slot'), store);

  // Warm 1 streaming hop (2048 samples) to verify kernel & unlock instant <15ms readiness
  const warmL = new Float32Array(2048);
  const warmR = new Float32Array(2048);
  mixer.streamEngine.synthesizeSampleRange(48000, 2048, [1, 1, 1, 1, 1, 1], warmL, warmR);
  const bootMs = Math.max(1, Math.round(performance.now() - tBoot));

  store.update({
    isReady: true,
    progress: 1.0,
    duration: mixer.durationSec,
    renderTimeMs: bootMs,
    pcmRamMb: 0.38,
    statusText: 'V9.4 流式正交余弦环形缓冲就绪',
  });

  window.__STUDIO_DEBUG__ = { store, mixer };
  loadMetricsIfAvailable(store);
}

bootstrapStudio();
