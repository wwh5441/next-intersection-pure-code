import { imdctPlan } from './fft.js';
import { interpolate } from './resampler.js';
import { SCALE_LUT, getBandEdges, getBandQmaxTable, applyCeltFolding1D } from './bark-bands.js';

const LONG_BINS = 2048;
const SHORT_BINS = 512;
const PAD_ZEROS = 768;

function buildWindows() {
  const sinL = Float64Array.from({ length: 2 * LONG_BINS }, (_, i) =>
    Math.sin((Math.PI * (i + 0.5)) / (2 * LONG_BINS))
  );
  const sinS = Float64Array.from({ length: 2 * SHORT_BINS }, (_, i) =>
    Math.sin((Math.PI * (i + 0.5)) / (2 * SHORT_BINS))
  );
  const lLeft = sinL.subarray(0, LONG_BINS);
  const lRight = sinL.subarray(LONG_BINS);
  const sLeft = new Float64Array(LONG_BINS);
  sLeft.set(sinS.subarray(0, SHORT_BINS), PAD_ZEROS);
  sLeft.fill(1, PAD_ZEROS + SHORT_BINS);
  const sRight = new Float64Array(LONG_BINS);
  sRight.fill(1, 0, PAD_ZEROS);
  sRight.set(sinS.subarray(SHORT_BINS), PAD_ZEROS);

  const make2L = (a, b) => {
    const w = new Float64Array(2 * LONG_BINS);
    w.set(a, 0);
    w.set(b, LONG_BINS);
    return w;
  };
  return {
    longModes: {
      0: make2L(lLeft, lRight),
      1: make2L(lLeft, sRight),
      3: make2L(sLeft, lRight),
      4: make2L(sLeft, sRight),
    },
    shortWin: sinS,
  };
}

const WINDOWS = buildWindows();

export class SpectralStream {
  constructor(buffer, outputRate) {
    this.data = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
    if (magic !== 'MDQ4') throw new Error(`Invalid spectral parameter format: ${magic}`);
    this.rate = this.data.getUint32(4, true);
    this.n = this.data.getUint16(8, true);
    this.shortN = this.data.getUint16(10, true);
    this.superFrames = this.data.getUint32(12, true);
    this.sampleCount = this.data.getUint32(16, true);
    this.baseQmax = this.data.getUint8(20);
    this.power = this.data.getUint8(21) / 100;
    this.foldLong = this.data.getUint8(22);
    this.foldShort = this.data.getUint8(23);
    this.stride = 1 + 48 + this.n;
    if (buffer.byteLength !== 32 + this.superFrames * this.stride) {
      throw new Error('Truncated MDQ4 spectral parameters');
    }

    this.ratio = this.rate / outputRate;
    this.imdctLong = imdctPlan(this.n);
    this.imdctShort = imdctPlan(this.shortN);
    this.edgesLong = getBandEdges(this.n);
    this.edgesShort = getBandEdges(this.shortN);
    this.qmaxLong = getBandQmaxTable(this.baseQmax, this.edgesLong.length - 1);
    this.qmaxShort = getBandQmaxTable(this.baseQmax, this.edgesShort.length - 1);

    this.invPower = 1 / this.power;
    this.specLong = new Float32Array(this.n);
    this.specShort = new Float32Array(this.shortN);
    this.scalesLong = new Float32Array(24);
    this.scalesShort = new Float32Array(12);
    this.timeLong = new Float64Array(2 * this.n);
    this.timeShort = new Float64Array(2 * this.shortN);
    this.reset();
  }

  reset() {
    this.frameCache = new Map();
    this.blockCache = new Map();
  }

  _decodeSubSpectrum(specBuf, scalesBuf, edges, qmaxTable, foldStart, scaleBaseOffset, qBaseOffset, seedBase) {
    specBuf.fill(0);
    const numBands = edges.length - 1;
    for (let g = 0; g < numBands; g++) {
      scalesBuf[g] = SCALE_LUT[this.data.getUint8(scaleBaseOffset + g)];
    }
    const codedBands = Math.min(foldStart, numBands);
    for (let g = 0; g < codedBands; g++) {
      const sc = scalesBuf[g];
      if (sc <= 0) continue;
      const qm = qmaxTable[g];
      const sK = edges[g];
      const eK = edges[g + 1];
      for (let k = sK; k < eK; k++) {
        const q = this.data.getInt8(qBaseOffset + k);
        if (q === 0) continue;
        specBuf[k] = Math.sign(q) * (Math.abs(q) / qm) ** this.invPower * sc;
      }
    }
    applyCeltFolding1D(specBuf, edges, scalesBuf, foldStart, seedBase);
  }

  inverseSuperFrame(index) {
    if (this.frameCache.has(index)) return this.frameCache.get(index);
    const output = new Float32Array(2 * this.n);
    if (index < 0 || index >= this.superFrames) return output;

    const base = 32 + index * this.stride;
    const mode = this.data.getUint8(base);
    const scaleBase = base + 1;
    const qBase = base + 49;

    if (mode !== 2) {
      this._decodeSubSpectrum(
        this.specLong,
        this.scalesLong,
        this.edgesLong,
        this.qmaxLong,
        this.foldLong,
        scaleBase,
        qBase,
        index * 4
      );
      this.imdctLong.inverse(this.specLong, this.timeLong);
      const win = WINDOWS.longModes[mode];
      for (let i = 0; i < output.length; i++) {
        output[i] = this.timeLong[i] * win[i];
      }
    } else {
      const winS = WINDOWS.shortWin;
      for (let sub = 0; sub < 4; sub++) {
        this._decodeSubSpectrum(
          this.specShort,
          this.scalesShort,
          this.edgesShort,
          this.qmaxShort,
          this.foldShort,
          scaleBase + sub * 12,
          qBase + sub * SHORT_BINS,
          index * 4 + sub
        );
        this.imdctShort.inverse(this.specShort, this.timeShort);
        const outOff = PAD_ZEROS + sub * SHORT_BINS;
        for (let i = 0; i < 2 * SHORT_BINS; i++) {
          output[outOff + i] += this.timeShort[i] * winS[i];
        }
      }
    }

    this.frameCache.set(index, output);
    if (this.frameCache.size > 4) this.frameCache.delete(this.frameCache.keys().next().value);
    return output;
  }

  block(index) {
    if (this.blockCache.has(index)) return this.blockCache.get(index);
    const a = this.inverseSuperFrame(index);
    const b = this.inverseSuperFrame(index + 1);
    const output = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) output[i] = a[i + this.n] + b[i];
    this.blockCache.set(index, output);
    if (this.blockCache.size > 4) this.blockCache.delete(this.blockCache.keys().next().value);
    return output;
  }

  read(index) {
    if (index < 0 || index >= this.sampleCount) return 0;
    const blk = Math.floor(index / this.n);
    return this.block(blk)[index - blk * this.n];
  }

  valueAt(outputFrame) {
    return interpolate(this, outputFrame * this.ratio);
  }
}
