// Background Web Worker for 10-Track Pure Mathematical Song Synthesis

import { renderFullPureMathSong, STEM_MANIFEST } from './song-renderer.js';

self.onmessage = (e) => {
  const { cmd } = e.data || {};
  if (cmd !== 'render') return;

  try {
    const res = renderFullPureMathSong((progress, text) => {
      self.postMessage({ type: 'progress', progress, text });
    });

    const transferList = [];
    for (const meta of STEM_MANIFEST) {
      const [L, R] = res.stems[meta.id];
      transferList.push(L.buffer, R.buffer);
    }

    self.postMessage(
      {
        type: 'done',
        sampleRate: res.sampleRate,
        numSamples: res.numSamples,
        durationSec: res.durationSec,
        bpm: res.bpm,
        bar0Sec: res.bar0Sec,
        renderTimeMs: res.renderTimeMs,
        peakAmplitude: res.peakAmplitude,
        stems: res.stems,
      },
      transferList
    );
  } catch (err) {
    self.postMessage({ type: 'error', message: err?.stack || String(err) });
  }
};
