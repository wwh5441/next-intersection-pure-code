const SINC_PHASES = 128;
const SINC_TAPS = 16;
const sincTable = new Float64Array((SINC_PHASES + 1) * SINC_TAPS);

function sincFn(x) {
  return Math.abs(x) < 1e-10 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
}

for (let phase = 0; phase <= SINC_PHASES; phase++) {
  let sum = 0;
  for (let tap = 0; tap < SINC_TAPS; tap++) {
    const x = tap - 7 - phase / SINC_PHASES;
    const value = Math.abs(x) >= 8 ? 0 : sincFn(x) * sincFn(x / 8);
    sincTable[phase * SINC_TAPS + tap] = value;
    sum += value;
  }
  for (let tap = 0; tap < SINC_TAPS; tap++) sincTable[phase * SINC_TAPS + tap] /= sum;
}

export function interpolateSinc(stream, sourcePosition) {
  const base = Math.floor(sourcePosition);
  const phase = Math.round((sourcePosition - base) * SINC_PHASES);
  const offset = phase * SINC_TAPS;
  let value = 0;
  for (let tap = 0; tap < SINC_TAPS; tap++) {
    value += stream.read(base + tap - 7) * sincTable[offset + tap];
  }
  return value;
}
