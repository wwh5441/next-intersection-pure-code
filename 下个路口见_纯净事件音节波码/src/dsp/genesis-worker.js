import '../vendor/lzma-d-min.js';
import { synthesizeVocalGenesis } from './sms-event-synth.js';

self.onmessage = ({ data }) => {
  try {
    const t0 = performance.now();
    const rawSpeaker = Uint8Array.from(globalThis.LZMA.decompress(new Uint8Array(data.compSpeaker)));
    const rawAtlas = Uint8Array.from(globalThis.LZMA.decompress(new Uint8Array(data.compAtlas)));
    const rawEvents = Uint8Array.from(globalThis.LZMA.decompress(new Uint8Array(data.compEvents)));
    const rawFdn = Uint8Array.from(globalThis.LZMA.decompress(new Uint8Array(data.compFdn)));

    const result = synthesizeVocalGenesis(rawSpeaker, rawAtlas, rawEvents, rawFdn, data.config);
    const synthMs = Math.round(performance.now() - t0);

    const transferList = [];
    const serializedLayers = {};
    for (const [layerId, [chL, chR]] of Object.entries(result.layers)) {
      serializedLayers[layerId] = [chL.buffer, chR.buffer];
      transferList.push(chL.buffer, chR.buffer);
    }

    self.postMessage(
      {
        id: data.id,
        ok: true,
        synthMs,
        sampleRate: result.sampleRate,
        numSamples: result.numSamples,
        layers: serializedLayers,
        uiEvents: result.uiEvents,
        prototypes: result.prototypes,
      },
      transferList
    );
  } catch (err) {
    self.postMessage({
      id: data.id,
      ok: false,
      error: err?.message || String(err),
    });
  }
};
