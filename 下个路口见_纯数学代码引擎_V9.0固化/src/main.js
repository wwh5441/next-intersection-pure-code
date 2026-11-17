// Main Entry Point for 《李宇春 - 下个，路口，见》 V9 Pure Mathematical Code Studio

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

function startSynthesisEngine(store, mixer) {
  const applyFinishedResult = (data) => {
    mixer.loadSynthesizedStems(data.sampleRate, data.numSamples, data.stems);
    store.update({
      isReady: true,
      progress: 1.0,
      duration: data.durationSec,
      bpm: data.bpm,
      bar0Sec: data.bar0Sec,
      renderTimeMs: data.renderTimeMs,
      peakAmplitude: data.peakAmplitude,
      statusText: '10 轨纯数学合成就绪',
    });
  };

  // Support both modular ES Worker and standalone plain-text Blob Worker
  try {
    let workerUrl = null;
    if (window.__PURE_MATH_WORKER_SOURCE__) {
      const blob = new Blob([window.__PURE_MATH_WORKER_SOURCE__], { type: 'text/javascript' });
      workerUrl = URL.createObjectURL(blob);
    } else {
      workerUrl = new URL('./dsp/math-worker.js', import.meta.url);
    }

    const worker = new Worker(workerUrl, { type: 'module' });
    worker.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === 'progress') {
        store.update({ progress: msg.progress, statusText: msg.text });
      } else if (msg.type === 'done') {
        applyFinishedResult(msg);
        worker.terminate();
      } else if (msg.type === 'error') {
        console.warn('Worker fallback to main thread:', msg.message);
        worker.terminate();
        const syncRes = renderFullPureMathSong((p, t) => store.update({ progress: p, statusText: t }));
        applyFinishedResult(syncRes);
      }
    };
    worker.onerror = () => {
      worker.terminate();
      const syncRes = renderFullPureMathSong((p, t) => store.update({ progress: p, statusText: t }));
      applyFinishedResult(syncRes);
    };
    worker.postMessage({ cmd: 'render' });
  } catch (_) {
    setTimeout(() => {
      const syncRes = renderFullPureMathSong((p, t) => store.update({ progress: p, statusText: t }));
      applyFinishedResult(syncRes);
    }, 30);
  }
}

function bootstrapStudio() {
  const store = createStudioStore();
  const mixer = new TenTrackAudioMixer(store);

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer);
  mountStemMixer(document.getElementById('mixer-slot'), store, mixer);
  mountFormulaInspector(document.getElementById('inspector-slot'), store);

  window.__STUDIO_DEBUG__ = { store, mixer };

  loadMetricsIfAvailable(store);
  startSynthesisEngine(store, mixer);
}

bootstrapStudio();
