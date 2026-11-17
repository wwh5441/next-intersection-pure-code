// V9R1 Pure Mathematical 4-Line Instrument Stereo FDN + 8-Line Prime Householder Studio Room FDN

const BASE_DELAYS_48K = [1193, 1511, 1877, 2293];
const BASE_HAAS_48K = 629;

export function applyInstrumentStereoFdn(mono48k, sideQ, nHarmFrames, hop24, sr24, acoustics, outSr) {
  const nSamples = mono48k.length;
  const outL = new Float32Array(nSamples);
  const outR = new Float32Array(nSamples);

  const rateScale = outSr / 48000.0;
  const d0 = Math.max(64, Math.round(BASE_DELAYS_48K[0] * rateScale));
  const d1 = Math.max(64, Math.round(BASE_DELAYS_48K[1] * rateScale));
  const d2 = Math.max(64, Math.round(BASE_DELAYS_48K[2] * rateScale));
  const d3 = Math.max(64, Math.round(BASE_DELAYS_48K[3] * rateScale));
  const haasLen = Math.max(32, Math.round(BASE_HAAS_48K * rateScale));

  const buf0 = new Float32Array(d0);
  const buf1 = new Float32Array(d1);
  const buf2 = new Float32Array(d2);
  const buf3 = new Float32Array(d3);
  const haasBuf = new Float32Array(haasLen);

  const rt60 = Math.max(0.25, acoustics.rt60Sec || 0.95);
  const g0 = Math.pow(10.0, (-3.0 * (d0 / outSr)) / rt60);
  const g1 = Math.pow(10.0, (-3.0 * (d1 / outSr)) / rt60);
  const g2 = Math.pow(10.0, (-3.0 * (d2 / outSr)) / rt60);
  const g3 = Math.pow(10.0, (-3.0 * (d3 / outSr)) / rt60);

  const reverbWet = acoustics.reverbWet ?? 0.16;
  const damping = acoustics.damping ?? 0.34;
  const omd = 1.0 - damping;
  const framesPerOutSample = sr24 / (outSr * hop24);

  let lp0 = 0.0, lp1 = 0.0, lp2 = 0.0, lp3 = 0.0;
  let i0 = 0, i1 = 0, i2 = 0, i3 = 0, ih = 0;

  for (let n = 0; n < nSamples; n++) {
    const x = mono48k[n];
    const y0 = buf0[i0], y1 = buf1[i1], y2 = buf2[i2], y3 = buf3[i3];

    const s = 0.5 * (y0 + y1 + y2 + y3);
    lp0 = omd * (y0 - s) + damping * lp0;
    lp1 = omd * (y1 - s) + damping * lp1;
    lp2 = omd * (y2 - s) + damping * lp2;
    lp3 = omd * (y3 - s) + damping * lp3;

    const inp = x * 0.35;
    buf0[i0] = inp + lp0 * g0;
    buf1[i1] = inp - lp1 * g1;
    buf2[i2] = inp + lp2 * g2;
    buf3[i3] = inp - lp3 * g3;

    const hx = haasBuf[ih];
    haasBuf[ih] = x;

    i0 = i0 + 1 === d0 ? 0 : i0 + 1;
    i1 = i1 + 1 === d1 ? 0 : i1 + 1;
    i2 = i2 + 1 === d2 ? 0 : i2 + 1;
    i3 = i3 + 1 === d3 ? 0 : i3 + 1;
    ih = ih + 1 === haasLen ? 0 : ih + 1;

    const fPos = n * framesPerOutSample;
    const f0 = fPos < nHarmFrames - 1 ? (fPos | 0) : nHarmFrames - 1;
    const f1 = f0 + 1 < nHarmFrames ? f0 + 1 : f0;
    const sq = sideQ[f0] + (sideQ[f1] - sideQ[f0]) * (fPos - f0);
    const sideRatio = (sq / 255.0) * 1.4;

    const revMid = 0.5 * (y0 + y2) * reverbWet;
    const revSide = (0.55 * (y0 - y1 + y2 - y3) + 0.45 * (x - hx)) * sideRatio;
    const midVal = x + revMid;
    outL[n] = midVal + revSide;
    outR[n] = midVal - revSide;
  }

  return [outL, outR];
}

export function synthesizeV9r1SpaceFdn(vocalSendMono, bandSendL, bandSendR, singerCore) {
  const sr = singerCore.sampleRate;
  const nSamples = vocalSendMono.length;
  const fdn = singerCore.fdnReverb;
  const envHop = fdn.envHopSamples;
  const rt60Sec = fdn.rt60Sec;
  const dampLp = fdn.dampLp;
  const baseWet = fdn.baseWetGain;
  const preDelay = fdn.preDelaySamples;

  const nEnv = fdn.gmEnvU8.length;
  const gmEnv = new Float32Array(nEnv);
  const gsEnv = new Float32Array(nEnv);
  for (let i = 0; i < nEnv; i++) {
    gmEnv[i] = (fdn.gmEnvU8[i] / 255.0) * baseWet;
    gsEnv[i] = (fdn.gsEnvU8[i] / 255.0) * baseWet;
  }

  const delays = fdn.delaysSamples;
  const bLps = delays.map((d) => (1.0 - dampLp) * Math.pow(10.0, (-3.0 * d) / (rt60Sec * sr)));
  const bufs = delays.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const lp = new Float64Array(8);

  const ringSize = 4096;
  const ringMask = ringSize - 1;
  const warmRing = new Float64Array(ringSize);
  const outL = new Float32Array(nSamples);
  const outR = new Float32Array(nSamples);

  let prevX = 0.0, prevHp = 0.0, prevInLp = 0.0;
  const hpAlpha = 0.973;
  const inLp = 0.54;
  const oneMinusInLp = 1.0 - inLp;
  const reads = new Float64Array(8);
  const damped = new Float64Array(8);
  const halfHop = envHop * 0.5;

  for (let n = 0; n < nSamples; n++) {
    const xn = vocalSendMono[n] + 0.10 * (bandSendL[n] + bandSendR[n]);
    const ynHp = hpAlpha * (prevHp + xn - prevX);
    prevX = xn;
    prevHp = ynHp;
    const yWarm = oneMinusInLp * ynHp + inLp * prevInLp;
    prevInLp = yWarm;

    const wPtr = n & ringMask;
    warmRing[wPtr] = yWarm;

    const yn = n >= preDelay ? warmRing[(n - preDelay) & ringMask] : 0.0;
    const erL =
      (n >= 542 ? 0.11 * warmRing[(n - 542) & ringMask] : 0.0) -
      (n >= 1186 ? 0.075 * warmRing[(n - 1186) & ringMask] : 0.0);
    const erR =
      (n >= 859 ? 0.105 * warmRing[(n - 859) & ringMask] : 0.0) -
      (n >= 1589 ? 0.07 * warmRing[(n - 1589) & ringMask] : 0.0);

    let sumD = 0.0;
    for (let i = 0; i < 8; i++) {
      const b = bufs[i], p = ptrs[i];
      const r = b[p];
      reads[i] = r;
      const st = bLps[i] * r + dampLp * lp[i];
      lp[i] = st;
      damped[i] = st;
      sumD += st;
    }

    const house = 0.25 * sumD;
    const excScaled = 0.25 * yn;
    for (let i = 0; i < 8; i++) {
      const b = bufs[i], p = ptrs[i];
      b[p] = damped[i] - house + ((i & 1) === 0 ? excScaled : -excScaled);
      ptrs[i] = p + 1 === delays[i] ? 0 : p + 1;
    }

    const rawM = 0.5 * (reads[0] + reads[2] + reads[4] + reads[6]);
    const rawS = 0.5 * (reads[1] - reads[3] + reads[5] - reads[7]);

    const ef = (n - halfHop) / envHop;
    let gm, gs;
    if (ef <= 0) {
      gm = gmEnv[0]; gs = gsEnv[0];
    } else if (ef >= nEnv - 1) {
      gm = gmEnv[nEnv - 1]; gs = gsEnv[nEnv - 1];
    } else {
      const e0 = ef | 0;
      const fr = ef - e0;
      gm = (1.0 - fr) * gmEnv[e0] + fr * gmEnv[e0 + 1];
      gs = (1.0 - fr) * gsEnv[e0] + fr * gsEnv[e0 + 1];
    }

    const mVal = rawM * gm;
    const sVal = rawS * gs;
    outL[n] = mVal + sVal + erL * gm;
    outR[n] = mVal - sVal + erR * gm;
  }

  return { studio_fdn_space: [outL, outR] };
}
