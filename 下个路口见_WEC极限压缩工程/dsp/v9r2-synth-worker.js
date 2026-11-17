// Web Worker for V9R2.1 Pure-Code 11-Track Synthesis (`src/dsp/v9r2-synth-worker.js`)
import { getRenderArgs } from '../modules/pure-code-assets.js';
import { renderV9r2AllLayers } from './v9r2-master-engine.js';

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type !== 'SYNTH_PROFILE') return;
  try {
    const res = renderV9r2AllLayers({
      ...getRenderArgs(),
      onProgress: (progress, statusText) => self.postMessage({ type: 'PROGRESS', reqId: msg.reqId, progress, statusText }),
    });
    const transferList = [];
    const layers = {};
    for (const [id, chans] of Object.entries(res.layers)) {
      layers[id] = chans.map((c) => {
        transferList.push(c.buffer);
        return c.buffer;
      });
    }
    self.postMessage(
      {
        type: 'READY',
        reqId: msg.reqId,
        sampleRate: res.sampleRate,
        numSamples: res.numSamples,
        durationSec: res.durationSec,
        synthMs: res.synthMs,
        layers,
        uiEvents: res.uiEvents,
      },
      transferList
    );
  } catch (err) {
    self.postMessage({ type: 'ERROR', reqId: msg.reqId, error: err?.message || String(err) });
  }
};
