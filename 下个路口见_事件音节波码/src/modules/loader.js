import '../vendor/lzma-d-min.js';
import { decodeBase122 } from '../dsp/unpacker.js';
import { synthesizeVocalGenesis } from '../dsp/sms-event-synth.js';
import { decodeV7Base122, synthesizeV7Accompaniment, synthesizeV5Accompaniment, synthesizeV8DictAccompaniment } from '../dsp/accomp-synth.js';

const V7_ACCOMP_STREAMS = ['bass', 'drums_body', 'drums_air', 'accomp_mid', 'accomp_side', 'accomp_air'];
const V8_DICT_ACCOMP_STREAMS = ['bass', 'drums', 'guitar_synth'];
const V5_ACCOMP_STREAMS = {
  drums: 'drums-wave.js',
  bass: 'bass-wave.js',
  harmony: 'harmony-wave.js',
};

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

async function loadV7CompressedPacket(relPath) {
  if (globalThis.__VFS__ && globalThis.__VFS__.b122[relPath]) {
    return decodeV7Base122(globalThis.__VFS__.b122[relPath]);
  }
  const mod = await import(`../../${relPath}`);
  return decodeV7Base122(mod.default);
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
    const transfer = [cSpk, cAtl, cEv, cFdn];
    const compAccompBufs = {};
    if (payload.compAccomp) {
      for (const [k, u8] of Object.entries(payload.compAccomp)) {
        const buf = u8.slice().buffer;
        compAccompBufs[k] = buf;
        transfer.push(buf);
      }
    }
    worker.postMessage(
      {
        id: 1,
        compSpeaker: cSpk,
        compAtlas: cAtl,
        compEvents: cEv,
        compFdn: cFdn,
        compAccomp: payload.compAccomp ? compAccompBufs : null,
        accompEngine: payload.accompEngine || 'v7',
        accompModels: payload.accompModels || null,
        config: payload.config,
      },
      transfer
    );
  });
}

function runDirectFallback({ compSpeaker, compAtlas, compEvents, compFdn, compAccomp, accompEngine, accompModels, config }) {
  const t0 = performance.now();
  const rawSpeaker = Uint8Array.from(globalThis.LZMA.decompress(compSpeaker));
  const rawAtlas = Uint8Array.from(globalThis.LZMA.decompress(compAtlas));
  const rawEvents = Uint8Array.from(globalThis.LZMA.decompress(compEvents));
  const rawFdn = Uint8Array.from(globalThis.LZMA.decompress(compFdn));
  const res = synthesizeVocalGenesis(rawSpeaker, rawAtlas, rawEvents, rawFdn, config);
  if (compAccomp) {
    const rawAccomp = {};
    for (const [k, u8] of Object.entries(compAccomp)) {
      rawAccomp[k] = Uint8Array.from(globalThis.LZMA.decompress(u8));
    }
    const accompLayers =
      accompEngine === 'v8_dict'
        ? synthesizeV8DictAccompaniment(rawAccomp, res.numSamples, res.sampleRate)
        : accompEngine === 'v5'
          ? synthesizeV5Accompaniment(rawAccomp, accompModels, res.numSamples, res.sampleRate)
          : synthesizeV7Accompaniment(rawAccomp, res.numSamples, res.sampleRate);
    Object.assign(res.layers, accompLayers);
  }
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
    const profCfg = config.profiles?.[profileId] || {};
    const accompEngine = profCfg.accompEngine || (profileId === 'ultra' ? 'v8_dict' : 'v7');
    const accompDir =
      profCfg.accompDir ||
      (accompEngine === 'v8_dict' ? 'accomp_v8_dict' : accompEngine === 'v5' ? 'accomp_v5_compact' : 'accomp_faithful');

    const vocalPromise = Promise.all([
      this._loadSharedPackets(),
      loadCompressedPacket(`data/${profileId}/event_stream.js`),
      loadJsonResource(`data/${profileId}/manifest.json`),
    ]);

    let accompModels = null;
    const compAccomp = {};
    if (accompEngine === 'v8_dict') {
      const [[shared, compEvents, manifest], ...accArr] = await Promise.all([
        vocalPromise,
        ...V8_DICT_ACCOMP_STREAMS.map((name) => loadCompressedPacket(`data/${accompDir}/${name}_dict.js`)),
      ]);
      V8_DICT_ACCOMP_STREAMS.forEach((name, idx) => {
        compAccomp[name] = accArr[idx];
      });
      var [compSpeaker, compAtlas, compFdn] = shared;
      var ev = compEvents;
      var mf = manifest;
    } else if (accompEngine === 'v5') {
      const entries = Object.entries(V5_ACCOMP_STREAMS);
      const [[shared, compEvents, manifest], models, ...accArr] = await Promise.all([
        vocalPromise,
        loadJsonResource(`data/${accompDir}/accomp_models.json`),
        ...entries.map(([, fname]) => loadV7CompressedPacket(`data/${accompDir}/${fname}`)),
      ]);
      accompModels = models;
      entries.forEach(([k], idx) => {
        compAccomp[k] = accArr[idx];
      });
      var [compSpeaker, compAtlas, compFdn] = shared;
      var ev = compEvents;
      var mf = manifest;
    } else {
      const [[shared, compEvents, manifest], ...accArr] = await Promise.all([
        vocalPromise,
        ...V7_ACCOMP_STREAMS.map((name) => loadV7CompressedPacket(`data/${accompDir}/${name}.js`)),
      ]);
      V7_ACCOMP_STREAMS.forEach((name, idx) => {
        compAccomp[name] = accArr[idx];
      });
      var [compSpeaker, compAtlas, compFdn] = shared;
      var ev = compEvents;
      var mf = manifest;
    }

    let synthResult;
    const payload = { compSpeaker, compAtlas, compEvents: ev, compFdn, compAccomp, accompEngine, accompModels, config };
    try {
      synthResult = await runInWorker(payload);
    } catch {
      synthResult = runDirectFallback(payload);
    }

    const bundle = { manifest: mf, ...synthResult };
    this.cache.set(profileId, bundle);
    return bundle;
  }
}
