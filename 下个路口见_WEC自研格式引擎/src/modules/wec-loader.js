// Native .wec (WEC0 / WEC1 / WEC2) Binary Loader with 3-Slot LRU Decoded Engine Cache

import { parseWecContainerAsync } from '../../parsers/wec-binary-parser.js';
import { createWecOpusEngineAsync } from '../dsp/wec-opus-engine.js';

const MAX_ENGINE_CACHE = 3;

export function createWecEngineRegistry(catalog) {
  const rawBufferCache = new Map();
  const engineLru = new Map();
  const catalogById = new Map(catalog.map((m) => [m.tierId, m]));

  function touchLru(key, bundle) {
    if (engineLru.has(key)) engineLru.delete(key);
    engineLru.set(key, bundle);
    while (engineLru.size > MAX_ENGINE_CACHE) {
      const oldestKey = engineLru.keys().next().value;
      engineLru.delete(oldestKey);
    }
  }

  async function loadCatalogTier(tierId) {
    if (engineLru.has(tierId)) {
      const hit = engineLru.get(tierId);
      touchLru(tierId, hit);
      return hit;
    }
    const meta = catalogById.get(tierId);
    if (!meta) {
      throw new Error(`未知的 WEC 档位 ID: ${tierId}`);
    }

    let ab = rawBufferCache.get(tierId);
    if (!ab) {
      const resp = await fetch(encodeURI(meta.wecUrl));
      if (!resp.ok) {
        throw new Error(`拉取 ${meta.wecFilename} 失败 (HTTP ${resp.status})`);
      }
      ab = await resp.arrayBuffer();
      rawBufferCache.set(tierId, ab);
    }

    const parsed = await parseWecContainerAsync(ab);
    const engine = await createWecOpusEngineAsync(parsed);
    const bundle = { engine, parsed, meta, isCustomFile: false };
    touchLru(tierId, bundle);
    return bundle;
  }

  async function loadLocalWecFile(file) {
    const ab = await file.arrayBuffer();
    const parsed = await parseWecContainerAsync(ab);
    const engine = await createWecOpusEngineAsync(parsed);
    const wecBytes = ab.byteLength;
    const dur = parsed.durationSec;
    const rawWav24Bytes = 44 + parsed.totalSamples * 2 * 3;

    const customMeta = {
      tierId: `local_${Date.now()}`,
      family: parsed.family,
      magic: parsed.magic,
      versionCode: `${parsed.magic} · Local`,
      name: file.name.replace(/\.wec$/i, ''),
      badge: `${parsed.magic} 本地载入 · ${parsed.tracks.length} 音轨 + ${parsed.features.length} 特征层`,
      wecFilename: file.name,
      wecBytes,
      equivKbps: Number(((wecBytes * 8.0) / dur / 1000.0).toFixed(2)),
      compressionRatioVs24BitWav: Number((rawWav24Bytes / wecBytes).toFixed(2)),
      audioTracks: parsed.tracks.length,
      featureLayers: parsed.features.length,
      featureTags: parsed.features.map((f) => f.fourCC),
      crcHex: parsed.storedCrcHex,
      passband: `${parsed.sampleRate} Hz · RFC 6716 Opus`,
      scenario: `CRC-32 ${parsed.storedCrcHex} 校验通过 · 本地拖拽直播`,
      sampleRate: parsed.sampleRate,
      totalSamples: parsed.totalSamples,
      durationSec: dur,
    };

    return { engine, parsed, meta: customMeta, isCustomFile: true };
  }

  function getCachedIds() {
    return Array.from(rawBufferCache.keys());
  }

  return { loadCatalogTier, loadLocalWecFile, getCachedIds };
}
