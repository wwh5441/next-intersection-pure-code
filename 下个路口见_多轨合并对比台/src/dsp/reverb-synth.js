const FDN_DELAYS = [1423, 1657, 1901, 2179, 2477, 2803, 3163, 3559];
const GAIN_MIN_DB = -44.0;
const GAIN_STEP_DB = 0.25;

const REVERB_GAIN_LUT = Float32Array.from({ length: 256 }, (_, c) =>
  c === 0 ? 0 : 10 ** ((GAIN_MIN_DB + (c - 1) * GAIN_STEP_DB) / 20)
);

export class StudioFdnReverb {
  constructor(buffer, outputRate) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
    if (magic !== 'VWR7') throw new Error(`Invalid studio reverb packet: ${magic}`);

    this.rate = view.getUint32(4, true);
    this.sampleCount = view.getUint32(8, true);
    this.envHop = view.getUint32(12, true);
    this.nHops = view.getUint32(16, true);
    const rt60Low = view.getUint16(20, true) / 1000;
    const rt60High = view.getUint16(22, true) / 1000;
    this.preDelay = view.getUint16(24, true);

    this.codesMid = new Uint8Array(buffer, 32, this.nHops);
    this.codesSide = new Uint8Array(buffer, 32 + this.nHops, this.nHops);

    this.aHp = 1 - Math.exp((-2 * Math.PI * 220) / outputRate);
    this.aLp = 1 - Math.exp((-2 * Math.PI * 6400) / outputRate);

    this.gDc = Float64Array.from(FDN_DELAYS, (d) => 10 ** ((-3 * d) / (outputRate * rt60Low)));
    this.damp = Float64Array.from(FDN_DELAYS, (d, i) => {
      const gHf = 10 ** ((-3 * d) / (outputRate * rt60High));
      return Math.min(0.72, Math.max(0.08, 1 - gHf / this.gDc[i]));
    });

    this.preMask = 2047;
    this.preBuf = new Float32Array(2048);
    this.bufs = FDN_DELAYS.map((d) => new Float64Array(d));
    this.ptrs = new Int32Array(8);
    this.dState = new Float64Array(8);
    this.reset();
  }

  reset() {
    this.hpM = 0;
    this.lpM = 0;
    this.hpS = 0;
    this.lpS = 0;
    this.prePtr = 0;
    this.preBuf.fill(0);
    for (let i = 0; i < 8; i++) {
      this.bufs[i].fill(0);
      this.ptrs[i] = 0;
      this.dState[i] = 0;
    }
  }

  step(frame, dryMid, drySide) {
    this.hpM += this.aHp * (dryMid - this.hpM);
    const mHp = dryMid - this.hpM;
    this.lpM += this.aLp * (mHp - this.lpM);
    const mIn = this.lpM;

    this.hpS += this.aHp * (drySide - this.hpS);
    const sHp = drySide - this.hpS;
    this.lpS += this.aLp * (sHp - this.lpS);
    const sIn = this.lpS;

    const wPtr = this.prePtr;
    this.preBuf[wPtr] = mIn;
    const erMid =
      0.22 * this.preBuf[(wPtr - 576) & this.preMask] +
      0.14 * this.preBuf[(wPtr - 888) & this.preMask];
    const erSide =
      0.20 * this.preBuf[(wPtr - 672) & this.preMask] -
      0.15 * this.preBuf[(wPtr - 936) & this.preMask];
    const xLate = this.preBuf[(wPtr - this.preDelay) & this.preMask];
    this.prePtr = (wPtr + 1) & this.preMask;

    let z0, z1, z2, z3, z4, z5, z6, z7;
    const zArr = new Float64Array(8);
    for (let i = 0; i < 8; i++) {
      const b = this.bufs[i];
      const p = this.ptrs[i];
      const dm = this.damp[i];
      const st = (1 - dm) * b[p] + dm * this.dState[i];
      this.dState[i] = st;
      zArr[i] = st * this.gDc[i];
    }
    [z0, z1, z2, z3, z4, z5, z6, z7] = zArr;
    const fbSum = 0.25 * (z0 + z1 + z2 + z3 + z4 + z5 + z6 + z7);

    const xm = xLate * 0.35;
    const xs = sIn * 0.18;
    for (let i = 0; i < 8; i++) {
      const b = this.bufs[i];
      const p = this.ptrs[i];
      b[p] = xm + (i & 1 ? xs : -xs) + (zArr[i] - fbSum);
      this.ptrs[i] = p + 1 === FDN_DELAYS[i] ? 0 : p + 1;
    }

    const rawMid = 0.35 * (z0 + z1 - z2 - z3 + z4 + z5 - z6 - z7) + erMid;
    const rawSide = 0.35 * (z0 - z1 + z2 - z3 - z4 + z5 - z6 + z7) + erSide;

    if (frame < 0 || frame >= this.sampleCount) return [0, 0];
    const pos = frame / this.envHop;
    const h0 = Math.floor(pos);
    const h1 = Math.min(h0 + 1, this.nHops - 1);
    const frac = pos - h0;

    const gM = REVERB_GAIN_LUT[this.codesMid[h0]] * (1 - frac) + REVERB_GAIN_LUT[this.codesMid[h1]] * frac;
    const gS = REVERB_GAIN_LUT[this.codesSide[h0]] * (1 - frac) + REVERB_GAIN_LUT[this.codesSide[h1]] * frac;

    const wetM = rawMid * gM;
    const wetS = rawSide * gS;
    return [wetM + wetS, wetM - wetS];
  }
}
