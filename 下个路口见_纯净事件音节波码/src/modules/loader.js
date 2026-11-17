import '../vendor/lzma-d-min.js';
import { decodeBase122 } from '../dsp/unpacker.js';
import { synthesizeVocalGenesis } from '../dsp/sms-event-synth.js';

async function loadJsonResource(relPath) {
  if (globalThis.__VFS__ && globalThis.__VFS__.json[relPath]) {
    return JSON.parse(globalThis.__VFS__.json[relPath]);
  }
  const resp = await fetch(`./${relPath}`);
  if (!resp.ok) throw new Error(`Failed to fetch ${relPath}: ${resp.status}`);
  return resp.json();
}

async function loadCompressedPacket(relPath) {
  if (globalThis.__VFS__ && globalThis.__VFS__.b122[relPath]) {
    return decodeBase122(globalThis.__VFS__.b122[relPath]);
  }
  const mod = await import(`../../${relPath}`);
  return decodeBase122(mod.default);
}

function runInWorker(payload) {
  return new Promise((resolve, reject) => {
    if (globalThis.__VFS__ || typeof Worker === 'undefined') {
      reject(new Error('Inline VFS or Worker unavailable, using direct main-thread synthesis'));
      return;
    }
    const worker = new Worker(new URL('../dsp/genesis-worker.js', import.meta.url), { type: 'module' });
    const cleanup = () => worker.terminate();
    worker.onerror = (ev) => {
      cleanup();
      reject(new Error(ev.message || 'Worker synthesis error'));
    };
    worker.onmessage = ({ data }) => {
      cleanup();
      if (!data.ok) {
        reject(new Error(data.error || 'Unknown Worker failure'));
        return;
      }
      const hydratedLayers = {};
      for (const [id, [bufL, bufR]] of Object.entries(data.layers)) {
        hydratedLayers[id] = [new Float32Array(bufL), new Float32Array(bufR)];
      }
      resolve({
        synthMs: data.synthMs,
        sampleRate: data.sampleRate,
        numSamples: data.numSamples,
        layers: hydratedLayers,
        uiEvents: data.uiEvents,
        prototypes: data.prototypes,
      });
    };
    const cSpk = payload.compSpeaker.slice().buffer;
    const cAtl = payload.compAtlas.slice().buffer;
    const cEv = payload.compEvents.slice().buffer;
    const cFdn = payload.compFdn.slice().buffer;
    worker.postMessage(
      {
        id: 1,
        compSpeaker: cSpk,
        compAtlas: cAtl,
        compEvents: cEv,
        compFdn: cFdn,
        config: payload.config,
      },
      [cSpk, cAtl, cEv, cFdn]
    );
  });
}

function runDirectFallback({ compSpeaker, compAtlas, compEvents, compFdn, config }) {
  const t0 = performance.now();
  const rawSpeaker = Uint8Array.from(globalThis.LZMA.decompress(compSpeaker));
  const rawAtlas = Uint8Array.from(globalThis.LZMA.decompress(compAtlas));
  const rawEvents = Uint8Array.from(globalThis.LZMA.decompress(compEvents));
  const rawFdn = Uint8Array.from(globalThis.LZMA.decompress(compFdn));
  const res = synthesizeVocalGenesis(rawSpeaker, rawAtlas, rawEvents, rawFdn, config);
  return {
    synthMs: Math.round(performance.now() - t0),
    sampleRate: res.sampleRate,
    numSamples: res.numSamples,
    layers: res.layers,
    uiEvents: res.uiEvents,
    prototypes: res.prototypes,
  };
}

export class GenesisResourceLoader {
  constructor() {
    this.cache = new Map();
    this.sharedPromises = null;
  }

  async loadStudioMetadata() {
    const [config, profiles, sizes, lyrics, atlasCatalog] = await Promise.all([
      loadJsonResource('schemas/vocal_genesis_config.json'),
      loadJsonResource('data/profiles.json'),
      loadJsonResource('data/sizes.json'),
      loadJsonResource('data/lyrics.json'),
      loadJsonResource('data/atlas_catalog.json'),
    ]);
    return { config, profiles, sizes, lyrics, atlasCatalog };
  }

  async _loadSharedPackets() {
    if (!this.sharedPromises) {
      this.sharedPromises = Promise.all([
        loadCompressedPacket('data/shared/speaker_core.js'),
        loadCompressedPacket('data/shared/syllable_atlas.js'),
        loadCompressedPacket('data/shared/space_fdn.js'),
      ]);
    }
    return this.sharedPromises;
  }

  async synthesizeProfile(profileId, config) {
    if (this.cache.has(profileId)) {
      return this.cache.get(profileId);
    }
    const [[compSpeaker, compAtlas, compFdn], compEvents, manifest] = await Promise.all([
      this._loadSharedPackets(),
      loadCompressedPacket(`data/${profileId}/event_stream.js`),
      loadJsonResource(`data/${profileId}/manifest.json`),
    ]);

    let synthResult;
    try {
      synthResult = await runInWorker({ compSpeaker, compAtlas, compEvents, compFdn, config });
    } catch {
      synthResult = runDirectFallback({ compSpeaker, compAtlas, compEvents, compFdn, config });
    }

    const bundle = { manifest, ...synthResult };
    this.cache.set(profileId, bundle);
    return bundle;
  }
}
