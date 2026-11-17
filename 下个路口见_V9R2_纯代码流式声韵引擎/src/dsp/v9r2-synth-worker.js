// Web Worker for V9R2 Pure-Code 11-Track Synthesis (`src/dsp/v9r2-synth-worker.js`)
import { SINGER_CORE } from '../../models/singer-core.js';
import { SYLLABLE_ATLAS } from '../../models/syllable-atlas.js';
import { SPACE_FDN } from '../../models/space-fdn.js';
import { INSTRUMENT_PHYSICS } from '../../models/instrument-physics.js';
import { STUDIO_METADATA } from '../../data/studio-metadata.js';
import { VOCAL_EVENTS_V85 } from '../../data/vocal-events-v85.js';
import { BAND_SCORE_EXPRESSIVE } from '../../data/band-score-expressive.js';
import { BAND_STEM_GAINS } from '../../data/band-stem-gains.js';
import { renderV9r2AllLayers } from './v9r2-master-engine.js';

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type !== 'SYNTH_PROFILE') return;
  try {
    const res = renderV9r2AllLayers({
      config: STUDIO_METADATA.config,
      singerCore: SINGER_CORE,
      syllableAtlas: SYLLABLE_ATLAS,
      spaceFdn: SPACE_FDN,
      vocalEvents: VOCAL_EVENTS_V85,
      instrumentPhysics: INSTRUMENT_PHYSICS,
      bandScore: BAND_SCORE_EXPRESSIVE,
      bandGains: BAND_STEM_GAINS,
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
        profileId: 'pure',
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
