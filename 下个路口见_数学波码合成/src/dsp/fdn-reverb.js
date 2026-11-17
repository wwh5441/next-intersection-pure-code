const BASE_DELAYS_48K = [1193, 1511, 1877, 2293];
const BASE_HAAS_48K = 629;

export class StereoFdnReverb {
  constructor(acoustics, sampleRate) {
    const rateScale = sampleRate / 48000;
    this.d0 = Math.max(64, Math.round(BASE_DELAYS_48K[0] * rateScale));
    this.d1 = Math.max(64, Math.round(BASE_DELAYS_48K[1] * rateScale));
    this.d2 = Math.max(64, Math.round(BASE_DELAYS_48K[2] * rateScale));
    this.d3 = Math.max(64, Math.round(BASE_DELAYS_48K[3] * rateScale));
    this.haasLen = Math.max(32, Math.round(BASE_HAAS_48K * rateScale));

    this.buf0 = new Float32Array(this.d0);
    this.buf1 = new Float32Array(this.d1);
    this.buf2 = new Float32Array(this.d2);
    this.buf3 = new Float32Array(this.d3);
    this.haasBuf = new Float32Array(this.haasLen);

    const rt60 = Math.max(0.25, acoustics.rt60Sec || 0.95);
    this.g0 = Math.pow(10, (-3 * (this.d0 / sampleRate)) / rt60);
    this.g1 = Math.pow(10, (-3 * (this.d1 / sampleRate)) / rt60);
    this.g2 = Math.pow(10, (-3 * (this.d2 / sampleRate)) / rt60);
    this.g3 = Math.pow(10, (-3 * (this.d3 / sampleRate)) / rt60);

    this.reverbWet = acoustics.reverbWet ?? 0.16;
    this.damping = acoustics.damping ?? 0.34;
    this.oneMinusDamp = 1 - this.damping;

    this.reset();
  }

  reset() {
    this.buf0.fill(0);
    this.buf1.fill(0);
    this.buf2.fill(0);
    this.buf3.fill(0);
    this.haasBuf.fill(0);
    this.lp0 = 0;
    this.lp1 = 0;
    this.lp2 = 0;
    this.lp3 = 0;
    this.i0 = 0;
    this.i1 = 0;
    this.i2 = 0;
    this.i3 = 0;
    this.ih = 0;
  }

  processSample(x, sideRatio, outPair) {
    const y0 = this.buf0[this.i0];
    const y1 = this.buf1[this.i1];
    const y2 = this.buf2[this.i2];
    const y3 = this.buf3[this.i3];

    // 4x4 Orthogonal Householder Mixing Matrix: H = I - 0.5 * 1 * 1^T
    const s = 0.5 * (y0 + y1 + y2 + y3);
    const d = this.damping;
    const omd = this.oneMinusDamp;

    this.lp0 = omd * (y0 - s) + d * this.lp0;
    this.lp1 = omd * (y1 - s) + d * this.lp1;
    this.lp2 = omd * (y2 - s) + d * this.lp2;
    this.lp3 = omd * (y3 - s) + d * this.lp3;

    const inp = x * 0.35;
    this.buf0[this.i0] = inp + this.lp0 * this.g0;
    this.buf1[this.i1] = inp - this.lp1 * this.g1;
    this.buf2[this.i2] = inp + this.lp2 * this.g2;
    this.buf3[this.i3] = inp - this.lp3 * this.g3;

    const hx = this.haasBuf[this.ih];
    this.haasBuf[this.ih] = x;

    this.i0 = this.i0 + 1 === this.d0 ? 0 : this.i0 + 1;
    this.i1 = this.i1 + 1 === this.d1 ? 0 : this.i1 + 1;
    this.i2 = this.i2 + 1 === this.d2 ? 0 : this.i2 + 1;
    this.i3 = this.i3 + 1 === this.d3 ? 0 : this.i3 + 1;
    this.ih = this.ih + 1 === this.haasLen ? 0 : this.ih + 1;

    const revMid = 0.5 * (y0 + y2) * this.reverbWet;
    const revSide = (0.55 * (y0 - y1 + y2 - y3) + 0.45 * (x - hx)) * sideRatio;
    const midVal = x + revMid;
    outPair[0] = midVal + revSide;
    outPair[1] = midVal - revSide;
  }
}
