const FDN_DELAYS_48K = [1499, 1777, 2081, 2437, 2801, 3217, 3671, 4177];

export function synthesizeSpaceFdn(rawPkt, dryMono) {
  const view = new DataView(rawPkt.buffer, rawPkt.byteOffset, rawPkt.byteLength);
  const magic = String.fromCharCode(rawPkt[0], rawPkt[1], rawPkt[2], rawPkt[3]);
  if (magic !== 'VGR9') throw new Error(`Invalid FDN magic: ${magic}`);

  const sr = view.getUint32(4, true);
  const nSamples = view.getUint32(8, true);
  const rt60Sec = view.getFloat32(12, true);
  const dampLp = view.getFloat32(16, true);
  const hpHz = view.getFloat32(20, true);
  const lpHz = view.getFloat32(24, true);
  const ratioM = view.getFloat32(28, true);
  const ratioS = view.getFloat32(32, true);
  const preDelay = view.getUint32(36, true);
  const duckAmount = view.getFloat32(40, true);

  const hpA = Math.exp((-2.0 * Math.PI * hpHz) / sr);
  const lpA = Math.exp((-2.0 * Math.PI * lpHz) / sr);
  const oneMLp = 1.0 - lpA;
  const oneMDamp = 1.0 - dampLp;

  const gains = FDN_DELAYS_48K.map((d) => Math.pow(10.0, (-3.0 * d) / (rt60Sec * sr)));
  const bufs = FDN_DELAYS_48K.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const lpState = new Float64Array(8);
  const reads = new Float64Array(8);
  const damped = new Float64Array(8);

  const outM = new Float32Array(nSamples);
  const outS = new Float32Array(nSamples);

  let prevX = 0.0;
  let hpY = 0.0;
  let lpY = 0.0;

  let drySqAct = 1e-16;
  let mSqAct = 1e-16;
  let sSqAct = 1e-16;
  let actCount = 0;

  for (let n = 0; n < nSamples; n++) {
    const xn = n >= preDelay ? dryMono[n - preDelay] : 0.0;
    hpY = hpA * (hpY + xn - prevX);
    prevX = xn;
    lpY = oneMLp * hpY + lpA * lpY;

    let sumD = 0.0;
    for (let i = 0; i < 8; i++) {
      const b = bufs[i];
      const p = ptrs[i];
      const r = b[p];
      reads[i] = r;
      const st = oneMDamp * (r * gains[i]) + dampLp * lpState[i];
      lpState[i] = st;
      damped[i] = st;
      sumD += st;
    }

    const house = 0.25 * sumD;
    const excScaled = 0.25 * lpY;
    for (let i = 0; i < 8; i++) {
      const b = bufs[i];
      const p = ptrs[i];
      b[p] = damped[i] - house + ((i & 1) === 0 ? excScaled : -excScaled);
      ptrs[i] = p + 1 === FDN_DELAYS_48K[i] ? 0 : p + 1;
    }

    const mv = 0.35 * (reads[0] + reads[2] + reads[4] + reads[6]);
    const sv = 0.35 * (reads[1] - reads[3] + reads[5] - reads[7]);
    outM[n] = mv;
    outS[n] = sv;

    const dv = dryMono[n];
    if (dv > 1e-4 || dv < -1e-4) {
      drySqAct += dv * dv;
      mSqAct += mv * mv;
      sSqAct += sv * sv;
      actCount++;
    }
  }

  let gainM = ratioM;
  let gainS = ratioS;
  if (actCount > 0) {
    const dryRms = Math.sqrt(drySqAct / actCount);
    const mRms = Math.sqrt(mSqAct / actCount);
    const sRms = Math.sqrt(sSqAct / actCount);
    gainM = (dryRms * ratioM) / Math.max(mRms, 1e-6);
    gainS = (dryRms * ratioS) / Math.max(sRms, 1e-6);
  }

  // Smooth 256-sample vocal clarity ducking
  const hopD = 256;
  const nD = Math.floor(nSamples / hopD) + 1;
  const envD = new Float32Array(nD);
  let pkD = 1e-9;
  for (let k = 0; k < nD; k++) {
    const s0 = k * hopD;
    const s1 = Math.min(nSamples, s0 + hopD);
    let sq = 1e-16;
    for (let i = s0; i < s1; i++) sq += dryMono[i] * dryMono[i];
    const r = Math.sqrt(sq / Math.max(1, s1 - s0));
    envD[k] = r;
    if (r > pkD) pkD = r;
  }

  const smD = new Float32Array(nD);
  let st = 0.0;
  for (let k = 0; k < nD; k++) {
    const tgt = Math.min(1.0, envD[k] / (pkD * 0.45));
    const coef = tgt > st ? 0.35 : 0.06;
    st += coef * (tgt - st);
    smD[k] = 1.0 - duckAmount * st;
  }

  const outL = new Float32Array(nSamples);
  const outR = new Float32Array(nSamples);
  for (let n = 0; n < nSamples; n++) {
    const idx = n / hopD;
    const i0 = Math.min(nD - 2, idx | 0);
    const fr = idx - i0;
    const duck = (1.0 - fr) * smD[i0] + fr * smD[i0 + 1];
    const mFinal = outM[n] * gainM * duck;
    const sFinal = outS[n] * gainS * duck;
    outL[n] = mFinal + sFinal;
    outR[n] = mFinal - sFinal;
  }

  return [outL, outR];
}
