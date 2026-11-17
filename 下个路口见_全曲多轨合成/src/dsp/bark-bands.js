const BARK_LONG_WIDTHS = [
  16, 16, 16, 16, 20, 20, 24, 24, 28, 32, 36, 40,
  48, 56, 64, 76, 88, 104, 124, 148, 180, 220, 288, 364,
];
const BARK_SHORT_WIDTHS = [8, 8, 12, 12, 16, 20, 28, 36, 48, 68, 104, 152];

const V6_SCALE_MIN_DB = -58.0;
const V6_SCALE_STEP_DB = 0.45;

export const SCALE_LUT = Float32Array.from({ length: 256 }, (_, code) =>
  code === 0 ? 0 : 10 ** ((V6_SCALE_MIN_DB + (code - 1) * V6_SCALE_STEP_DB) / 20)
);

export function getBandEdges(n) {
  const widths = n === 2048 ? BARK_LONG_WIDTHS : BARK_SHORT_WIDTHS;
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
    if (u <= 0.65) q = baseQmax;
    else if (u <= 0.85) q = Math.max(9, Math.round(baseQmax * 0.68));
    else q = Math.max(7, Math.round(baseQmax * 0.45));
    out[g] = Math.min(127, q);
  }
  return out;
}
