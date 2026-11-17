// V9R2.1 Pure-Code 11-Track Renderer (`src/dsp/v9r2-master-engine.js`)
// Vocals 01–08: Direct 64-ERB vocal tract + 97-syllable phoneme atlas + true L/R stereo harmony
// Band 09–11: bass harmonic model, physical/gated drum events, polyphonic note runs (all rebuilt from the MP3)
import { synthesizeV9r21VocalAnatomy } from './v9r21-vocal-synth.js';
import { synthesizeV21Band } from './v21-band-synth.js';

export function renderV9r2AllLayers({
  config,
  speakerModel,
  phonemeAtlas,
  backingModel,
  spaceFdn,
  vocalScore,
  backingScore,
  bandModels,
  bandScores,
  bandGains,
  onProgress = null,
}) {
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const t0 = now();
  if (onProgress) onProgress(0.08, '合成 64-ERB 直接共振峰主唱与 L/R 独立真和声...');
  const vocalRes = synthesizeV9r21VocalAnatomy({
    speakerModel,
    phonemeAtlas,
    backingModel,
    spaceFdn,
    vocalScore,
    backingScore,
    config,
  });
  if (onProgress) onProgress(0.55, '合成贝斯谐波、物理鼓组与吉他电钢复调音符...');
  const bandLayers = synthesizeV21Band(bandModels, bandScores, bandGains, speakerModel.numSamples, speakerModel.sampleRate);
  const layers = { ...vocalRes.layers, ...bandLayers };
  applySmoothBusLimiter(layers, speakerModel.numSamples, 0.96);
  if (onProgress) onProgress(1.0, '48kHz 十一轨纯代码合成完毕');
  return {
    sampleRate: speakerModel.sampleRate,
    numSamples: speakerModel.numSamples,
    durationSec: speakerModel.numSamples / speakerModel.sampleRate,
    synthMs: Math.round(now() - t0),
    layers,
    uiEvents: vocalRes.uiEvents,
    prototypesCount: vocalRes.prototypesCount,
  };
}

// Shared lookahead smooth-envelope safety limiter (keeps exact 11-track additivity with zero 187 Hz block modulation).
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
  // 4-block (~21 ms) lookahead min-envelope + 120 ms smooth release + 5-tap Hann smoothing
  const gMin = new Float32Array(nBlocks);
  const look = 4;
  for (let b = 0; b < nBlocks; b++) {
    let m = 1.0;
    const j0 = Math.max(0, b - 1);
    const j1 = Math.min(nBlocks - 1, b + look);
    for (let j = j0; j <= j1; j++) if (gBlock[j] < m) m = gBlock[j];
    gMin[b] = m;
  }
  const gRel = new Float32Array(nBlocks);
  let rel = 1.0;
  for (let b = 0; b < nBlocks; b++) {
    const target = gMin[b];
    rel = target < rel ? 0.35 * rel + 0.65 * target : 0.955 * rel + 0.045 * target;
    gRel[b] = Math.min(gBlock[b], rel);
  }
  const gSmooth = new Float32Array(nBlocks);
  const kWeights = [0.09, 0.24, 0.34, 0.24, 0.09];
  for (let b = 0; b < nBlocks; b++) {
    let acc = 0.0;
    for (let k = -2; k <= 2; k++) {
      const idx = Math.max(0, Math.min(nBlocks - 1, b + k));
      acc += kWeights[k + 2] * gRel[idx];
    }
    gSmooth[b] = Math.min(gBlock[b], acc);
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
