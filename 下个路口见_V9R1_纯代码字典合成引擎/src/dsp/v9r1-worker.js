// V9R1 Zero-Copy Web Worker Synthesis Host (`v9r1-worker.js`)
// Keeps browser UI 100% responsive at 60fps during 3m30s 10-track ODE synthesis

import { renderV9r1Song } from './v9r1-renderer.js';

self.onmessage = (ev) => {
  const { reqId, tierId } = ev.data || {};
  try {
    const result = renderV9r1Song(tierId, (pct, text) => {
      self.postMessage({ type: 'progress', reqId, pct, text });
    });

    const transferList = [];
    for (const chList of Object.values(result.stems)) {
      for (const arr of chList) {
        if (arr && arr.buffer) transferList.push(arr.buffer);
      }
    }

    self.postMessage(
      {
        type: 'done',
        reqId,
        payload: {
          tierId: result.tierId,
          vocalProfile: result.vocalProfile,
          bandProfile: result.bandProfile,
          sampleRate: result.sampleRate,
          numSamples: result.numSamples,
          durationSec: result.durationSec,
          bpm: result.bpm,
          bar0Sec: result.bar0Sec,
          renderTimeMs: result.renderTimeMs,
          peakAmplitude: result.peakAmplitude,
          stems: result.stems,
        },
      },
      transferList
    );
  } catch (err) {
    self.postMessage({
      type: 'error',
      reqId,
      message: err && err.message ? err.message : String(err),
    });
  }
};
