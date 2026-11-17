const LONG_BAND_WIDTHS = [
  16, 16, 16, 16, 20, 20, 24, 24, 28, 32, 36, 40,
  48, 56, 64, 76, 88, 104, 124, 148, 180, 220, 288, 364,
];
const SHORT_BAND_WIDTHS = [8, 8, 12, 12, 16, 20, 28, 36, 48, 68, 104, 152];

const SCALE_MIN_DB = -52.0;
const SCALE_STEP_DB = 0.50;

export const SCALE_LUT = Float32Array.from({ length: 256 }, (_, code) =>
  code === 0 ? 0 : 10 ** ((SCALE_MIN_DB + (code - 1) * SCALE_STEP_DB) / 20)
);

export function getBandEdges(n) {
  const widths = n === 2048 ? LONG_BAND_WIDTHS : SHORT_BAND_WIDTHS;
  const edges = new Int32Array(widths.length + 1);
  for (let i = 0; i < widths.length; i++) {
    edges[i + 1] = edges[i] + widths[i];
  }
  return edges;
}

export function getBandQmaxTable(baseQmax, numBands) {
  const out = new Int32Array(numBands);
  for (let g = 0; g < numBands; g++) {
    const u = g / Math.max(1, numBands - 1);
    let q;
    if (u <= 0.6) q = baseQmax;
    else if (u <= 0.8) q = Math.max(7, Math.round(baseQmax * 0.58));
    else q = Math.max(5, Math.round(baseQmax * 0.34));
    out[g] = Math.min(127, q);
  }
  return out;
}

export function applyCeltFolding1D(spectrum, edges, scales, foldStartBand, seedBase) {
  const numBands = edges.length - 1;
  if (foldStartBand >= numBands) return;
  const n = spectrum.length;
  const srcLo = n >> 3;
  const srcLen = n >> 2;
  for (let g = foldStartBand; g < numBands; g++) {
    const targetRms = scales[g];
    const startK = edges[g];
    const endK = edges[g + 1];
    if (targetRms <= 0) {
      spectrum.fill(0, startK, endK);
      continue;
    }
    const width = endK - startK;
    let sumSq = 0;
    for (let i = 0; i < width; i++) {
      const k = startK + i;
      const srcVal = spectrum[srcLo + (k % srcLen)];
      let h = (Math.imul(seedBase, 2654435761) + Math.imul(k, 2246822519)) >>> 0;
      h = Math.imul(h ^ (h >>> 15), 2246822519) >>> 0;
      const sign = (h & 0x80000000) !== 0 ? -1 : 1;
      const noiseUnit = ((h & 0xffff) / 32768 - 1) * 0.15;
      const v = sign * srcVal + noiseUnit * targetRms;
      spectrum[k] = v;
      sumSq += v * v;
    }
    const curRms = Math.sqrt(sumSq / width);
    if (curRms > 1e-12) {
      const gain = targetRms / curRms;
      for (let k = startK; k < endK; k++) spectrum[k] *= gain;
    } else {
      spectrum.fill(0, startK, endK);
    }
  }
}
