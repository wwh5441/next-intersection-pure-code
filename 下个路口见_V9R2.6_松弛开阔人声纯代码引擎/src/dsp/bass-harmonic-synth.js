// V9R2.1 Discrete Bass Note Synthesizer (`src/dsp/bass-harmonic-synth.js`)
// Synthesizes discrete bass notes with smooth continuous-phase harmonic oscillators + 12 fingerstyle presets.
export function synthesizeBassHarmonic(model, score, numSamples, sr = 48000) {
  const hop = model.hop;
  const nH = model.nHarm;
  const twoPi = 2.0 * Math.PI;
  const gainStep = model.gainStepDb ?? 0.5;
  const timbres = model.timbresDbQ.map((row) => {
    const a = new Float64Array(nH);
    let sq = 0.0;
    for (let h = 0; h < nH; h++) {
      const v = Math.pow(10.0, (row[h] * 0.5) / 20.0);
      a[h] = v;
      sq += v * v;
    }
    const sc = 1.0 / Math.sqrt(0.5 * sq + 1e-18);
    for (let h = 0; h < nH; h++) a[h] *= sc;
    return a;
  });

  const nf = Math.floor(numSamples / hop) + 2;
  const f0g = new Float32Array(nf);
  const gg = new Float32Array(nf);
  const tidg = new Int16Array(nf);

  let curS = 0;
  for (const [ds, tid, dk, dc, dg] of score.notes) {
    curS += ds;
    let k = 0;
    let c = 0;
    let g = 0;
    const kf = [];
    const kc = [];
    const kg = [];
    for (let i = 0; i < dk.length; i++) {
      k += dk[i];
      c += dc[i];
      g += dg[i];
      kf.push(curS + k);
      kc.push(c);
      kg.push(g * gainStep);
    }
    for (let i = 0; i < kf.length; i++) {
      const fA = kf[i];
      const fB = i + 1 < kf.length ? kf[i + 1] : fA + 1;
      const i2 = Math.min(i + 1, kf.length - 1);
      for (let f = fA; f < fB && f < nf; f++) {
        const t = fB > fA ? (f - fA) / (fB - fA) : 0.0;
        const cents = kc[i] + (kc[i2] - kc[i]) * t;
        f0g[f] = model.f0RefHz * Math.pow(2.0, cents / 1200.0);
        gg[f] = Math.pow(10.0, (kg[i] + (kg[i2] - kg[i]) * t) / 20.0);
        tidg[f] = tid;
      }
    }
  }

  const out = new Float32Array(numSamples);
  let phi = 0.0;
  const amp = new Float64Array(nH);
  const dAmp = new Float64Array(nH);

  for (let f = 0; f < nf - 1; f++) {
    const fa = f0g[f];
    const fb = f0g[f + 1];
    if (fa <= 0.0 && fb <= 0.0) {
      phi = 0.0;
      continue;
    }
    const s0 = f * hop;
    if (s0 >= numSamples) break;
    const fs = fa > 0.0 ? fa : fb;
    const fe = fb > 0.0 ? fb : fa;
    const profA = timbres[tidg[f]] || timbres[0];
    const profB = timbres[tidg[f + 1]] || profA;
    const ga = fa > 0.0 ? gg[f] : 0.0;
    const gb = fb > 0.0 ? gg[f + 1] : 0.0;
    for (let h = 0; h < nH; h++) {
      const a0 = ga * profA[h];
      const a1 = gb * profB[h];
      amp[h] = a0;
      dAmp[h] = (a1 - a0) / hop;
    }
    const end = Math.min(hop, numSamples - s0);
    for (let s = 0; s < end; s++) {
      const fc = fs + ((fe - fs) * s) / hop;
      phi += (twoPi * fc) / sr;
      if (phi >= twoPi) phi -= twoPi;
      const c1 = Math.cos(phi);
      const s1 = Math.sin(phi);
      const two = 2.0 * c1;
      let cp = 1.0;
      let sp = 0.0;
      let ch = c1;
      let sh = s1;
      let acc = 0.0;
      const nAct = Math.min(nH, Math.floor(14000.0 / Math.max(fc, 1.0)));
      for (let h = 0; h < nAct; h++) {
        acc += (amp[h] + dAmp[h] * s) * sh;
        const cn = two * ch - cp;
        const sn = two * sh - sp;
        cp = ch;
        sp = sh;
        ch = cn;
        sh = sn;
      }
      out[s0 + s] = acc;
    }
  }
  return out;
}
