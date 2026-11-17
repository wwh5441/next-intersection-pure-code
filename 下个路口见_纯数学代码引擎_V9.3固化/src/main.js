// Main Entry Point for 《李宇春 - 下个，路口，见》 V9.3 48kHz Near-Lossless Orthonormal Pure-Math Engine

import { createStudioStore } from './state/studio-store.js';
import { StudioAudioMixer } from './modules/audio-mixer.js';
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
    // Optional metrics fallback
  }
}

function startSynthesis(store, mixer) {
  const applyFinished = (data) => {
    mixer.loadSynthesizedStems(data.sampleRate, data.numSamples, data.stems);
    // 6 stereo stems x 2 ch x numSamples x 4 bytes Float32
    const pcmRamMb = (6 * 2 * data.numSamples * 4) / 1048576;
    store.update({
      isReady: true,
      progress: 1.0,
      duration: data.durationSec,
      bpm: data.bpm,
      bar0Sec: data.bar0Sec,
      renderTimeMs: data.renderTimeMs,
      peakAmplitude: data.peakAmplitude,
      pcmRamMb,
      statusText: '6 轨 48kHz 几乎无损纯数学母带就绪',
    });
  };

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
        applyFinished(msg);
        worker.terminate();
      } else if (msg.type === 'error') {
        worker.terminate();
        const syncRes = renderFullPureMathSong((p, t) => store.update({ progress: p, statusText: t }));
        applyFinished(syncRes);
      }
    };
    worker.onerror = () => {
      worker.terminate();
      const syncRes = renderFullPureMathSong((p, t) => store.update({ progress: p, statusText: t }));
      applyFinished(syncRes);
    };
    worker.postMessage({ cmd: 'render' });
  } catch (_) {
    setTimeout(() => {
      const syncRes = renderFullPureMathSong((p, t) => store.update({ progress: p, statusText: t }));
      applyFinished(syncRes);
    }, 20);
  }
}

function bootstrapStudio() {
  const store = createStudioStore();
  const mixer = new StudioAudioMixer(store);

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer);
  mountStemMixer(document.getElementById('mixer-slot'), store, mixer);
  mountFormulaInspector(document.getElementById('inspector-slot'), store);

  window.__STUDIO_DEBUG__ = { store, mixer };

  loadMetricsIfAvailable(store);
  startSynthesis(store, mixer);
}

bootstrapStudio();
