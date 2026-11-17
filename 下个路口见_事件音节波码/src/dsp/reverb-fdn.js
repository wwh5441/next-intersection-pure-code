const FDN_DELAYS_48K = [1499, 1777, 2081, 2437, 2801, 3217, 3671, 4177];

export function synthesizeSpaceFdn(rawPkt, dryMono) {
  const view = new DataView(rawPkt.buffer, rawPkt.byteOffset, rawPkt.byteLength);
  const magic = String.fromCharCode(rawPkt[0], rawPkt[1], rawPkt[2], rawPkt[3]);
  if (magic !== 'VGR8') throw new Error(`Invalid FDN magic: ${magic}`);

  const sr = view.getUint32(4, true);
  const nSamples = view.getUint32(8, true);
  const envHop = view.getUint32(12, true);
  const rt60Sec = view.getFloat32(16, true);
  const dampLp = view.getFloat32(20, true);
  const baseWetGain = view.getFloat32(24, true);
  const preDelay = view.getUint32(28, true);

  const nEnv = Math.ceil(nSamples / envHop);
  const gmEnv = new Float32Array(nEnv);
  const gsEnv = new Float32Array(nEnv);
  for (let i = 0; i < nEnv; i++) {
    gmEnv[i] = (rawPkt[32 + i] / 255.0) * baseWetGain;
    gsEnv[i] = (rawPkt[32 + nEnv + i] / 255.0) * baseWetGain;
  }

  const delays = FDN_DELAYS_48K;
  const bLps = delays.map((d) => (1.0 - dampLp) * Math.pow(10.0, (-3.0 * d) / (rt60Sec * sr)));
  const bufs = delays.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const lp = new Float64Array(8);

  const ringSize = 4096;
  const ringMask = ringSize - 1;
  const warmRing = new Float64Array(ringSize);

  const outL = new Float32Array(nSamples);
  const outR = new Float32Array(nSamples);

  let prevX = 0.0;
  let prevHp = 0.0;
  let prevInLp = 0.0;
  const hpAlpha = 0.973;
  const inLp = 0.54;
  const oneMinusInLp = 1.0 - inLp;
  const reads = new Float64Array(8);
  const damped = new Float64Array(8);
  const halfHop = envHop * 0.5;
  const dryLen = dryMono.length;

  for (let n = 0; n < nSamples; n++) {
    const xn = n < dryLen ? dryMono[n] : 0.0;
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
      const b = bufs[i];
      const p = ptrs[i];
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
      const b = bufs[i];
      const p = ptrs[i];
      const sign = (i & 1) === 0 ? excScaled : -excScaled;
      b[p] = damped[i] - house + sign;
      ptrs[i] = p + 1 === delays[i] ? 0 : p + 1;
    }

    const rawM = 0.5 * (reads[0] + reads[2] + reads[4] + reads[6]);
    const rawS = 0.5 * (reads[1] - reads[3] + reads[5] - reads[7]);

    const ef = (n - halfHop) / envHop;
    let gm;
    let gs;
    if (ef <= 0) {
      gm = gmEnv[0];
      gs = gsEnv[0];
    } else if (ef >= nEnv - 1) {
      gm = gmEnv[nEnv - 1];
      gs = gsEnv[nEnv - 1];
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

  return [outL, outR];
}
