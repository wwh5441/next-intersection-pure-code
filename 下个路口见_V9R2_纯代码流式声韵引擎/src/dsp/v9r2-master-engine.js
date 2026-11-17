// V9R2 Pure-Code 11-Track Renderer (`src/dsp/v9r2-master-engine.js`)
// Vocals 01–08: V8.5 97-syllable atlas + `--` coarticulation parametric synthesis
// Band 09–11: V9.2 physical-model equations driven by symbolic parameter score
import { synthesizeV85VocalAnatomy } from './v85-vocal-coart-synth.js';
import { synthesizeV92PhysicsBand } from './v92-band-physics-synth.js';

export function renderV9r2AllLayers({
  config,
  singerCore,
  syllableAtlas,
  spaceFdn,
  vocalEvents,
  instrumentPhysics,
  bandScore,
  bandGains,
  onProgress = null,
}) {
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const t0 = now();
  if (onProgress) onProgress(0.08, '展开 97 音节声韵字典与 286 字 `--` 协同连音...');
  const vocalRes = synthesizeV85VocalAnatomy(singerCore, syllableAtlas, spaceFdn, vocalEvents, config);

  if (onProgress) onProgress(0.6, '求解鼓膜模态、弦振与耦合旋子物理方程...');
  const bandLayers = synthesizeV92PhysicsBand(
    instrumentPhysics,
    bandScore,
    bandGains,
    singerCore.sampleRate,
    singerCore.numSamples
  );

  const layers = { ...vocalRes.layers, ...bandLayers };
  applySmoothBusLimiter(layers, singerCore.numSamples, 0.94);

  if (onProgress) onProgress(1.0, '48kHz 十一轨纯代码合成完毕');
  return {
    sampleRate: singerCore.sampleRate,
    numSamples: singerCore.numSamples,
    durationSec: singerCore.numSamples / singerCore.sampleRate,
    synthMs: Math.round(now() - t0),
    layers,
    uiEvents: vocalRes.uiEvents,
    prototypesCount: vocalRes.prototypesCount,
  };
}

// Zero-overshoot hop-local bus limiter (V9.2): one shared gain curve applied to every track,
// so 11-track linear additivity is preserved.
function applySmoothBusLimiter(layers, nSamples, ceiling) {
  const hop = 256;
  const nBlocks = Math.ceil(nSamples / hop);
  const tracks = Object.values(layers).map((chs) => (chs.length > 1 ? chs : [chs[0], chs[0]]));
  const gBlock = new Float32Array(nBlocks).fill(1.0);
  for (let b = 0; b < nBlocks; b++) {
    const s1 = Math.min(nSamples, (b + 1) * hop);
    let pk = 0.0;
    for (let s = b * hop; s < s1; s++) {
      let l = 0.0;
      let r = 0.0;
      for (const [cl, cr] of tracks) {
        l += cl[s];
        r += cr[s];
      }
      const a = Math.max(Math.abs(l), Math.abs(r));
      if (a > pk) pk = a;
    }
    if (pk > ceiling) gBlock[b] = ceiling / pk;
  }
  const gSmooth = new Float32Array(nBlocks);
  let rel = 1.0;
  for (let b = 0; b < nBlocks; b++) {
    const target = Math.min(gBlock[b], b > 0 ? gBlock[b - 1] : 1.0, b + 1 < nBlocks ? gBlock[b + 1] : 1.0);
    rel = target < rel ? target : 0.85 * rel + 0.15 * target;
    gSmooth[b] = Math.min(gBlock[b], rel);
  }
  const unique = new Set();
  for (const chs of Object.values(layers)) for (const c of chs) unique.add(c);
  for (let b = 0; b < nBlocks; b++) {
    const g0 = Math.min(b > 0 ? gSmooth[b - 1] : 1.0, gSmooth[b]);
    const g1 = Math.min(gSmooth[b], b + 1 < nBlocks ? gSmooth[b + 1] : gSmooth[b]);
    if (g0 >= 0.9999 && g1 >= 0.9999) continue;
    const s0 = b * hop;
    const s1 = Math.min(nSamples, s0 + hop);
    const inv = 1.0 / Math.max(1, s1 - s0);
    for (let s = s0; s < s1; s++) {
      const g = g0 + (g1 - g0) * (s - s0) * inv;
      for (const c of unique) c[s] *= g;
    }
  }
}
