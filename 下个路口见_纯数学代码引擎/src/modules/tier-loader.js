// On-Demand Dynamic Pure-Math Tier Engine Loader & Memory Cache (V9.4.1 .. V9.4.7)
import { createStreamingTierEngine } from '../dsp/streaming-tier-engine.js';

export function createTierEngineRegistry() {
  const engineCache = new Map();

  async function loadTierEngine(tierId) {
    if (engineCache.has(tierId)) {
      return engineCache.get(tierId);
    }
    const mod = await import(`../../data/tiers/${tierId}/tier-entry.js`);
    const eng = createStreamingTierEngine(mod.TIER_PHYSICS, mod.HUFFMAN_TABLES, mod.TIER_FRAMES);
    engineCache.set(tierId, eng);
    return eng;
  }

  function getCachedIds() {
    return Array.from(engineCache.keys());
  }

  return {
    loadTierEngine,
    getCachedIds,
  };
}
