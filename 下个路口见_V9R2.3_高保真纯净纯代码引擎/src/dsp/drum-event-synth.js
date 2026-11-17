// V9R2.1 Clean Physical + Tight-Transient Drum Synthesizer (`src/dsp/drum-event-synth.js`)
// 1. `low` (Kick, 30-160 Hz): Pure time-domain swept-sine + 2nd-harmonic membrane (0 noise, 100% center punch).
// 2. `midModal` (Snare shell / Woodblock / Rim, 168-1350 Hz): Dual damped sinusoidal resonators (0 noise).
// 3. `midSnap` (Snare wires / Clap) & `high` (Hi-Hat / Shaker / Cymbal): 512-pt sine-windowed gated transients.
import { Radix2ComplexFFT, createMulberry32 } from './fft-core.js';

function hzToMel(f) {
  const fsp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fsp;
  const logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fsp;
}

function melToHz(m) {
  const fsp = 200 / 3;
  const minLogHz = 1000;
  const minLogMel = minLogHz / fsp;
  const logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fsp * m;
}

export function melFilterbank(sr, nfft, nMel, fmin, fmax) {
  const nBins = nfft / 2 + 1;
  const m0 = hzToMel(fmin);
  const m1 = hzToMel(fmax);
  const pts = [];
  for (let i = 0; i < nMel + 2; i++) pts.push(melToHz(m0 + ((m1 - m0) * i) / (nMel + 1)));
  const fb = [];
  for (let b = 0; b < nMel; b++) {
    const row = new Float64Array(nBins);
    const d0 = pts[b + 1] - pts[b];
    const d1 = pts[b + 2] - pts[b + 1];
    for (let k = 0; k < nBins; k++) {
      const f = (k * sr) / nfft;
      const lower = (f - pts[b]) / d0;
      const upper = (pts[b + 2] - f) / d1;
      row[k] = Math.max(0, Math.min(lower, upper));
    }
    fb.push(row);
  }
  return fb;
}

export function synthesizeDrumEvents(model, score, gains = {}, numSamples, sr = 48000) {
  const hop = model.hop;
  const nfft = model.nfft;
  const nBins = nfft / 2 + 1;
  const twoPi = 2.0 * Math.PI;
  const outL = new Float32Array(numSamples);
  const outR = new Float32Array(numSamples);

  // 1. Physical Kick Membrane (`low`) — 100% noise-free swept-sine + punch harmonic
  const gLow = gains.low ?? 1.0;
  if (gLow > 0 && score.low && score.low.frame) {
    const kEv = score.low;
    const atkSamples = Math.max(16, Math.round(0.0018 * sr));
    const tauPitch = 0.015 * sr;
    let fr = 0;
    for (let i = 0; i < kEv.frame.length; i++) {
      fr += kEv.frame[i];
      const s0 = fr * hop;
      if (s0 >= numSamples) break;
      const amp = Math.pow(10.0, (kEv.ampQ[i] * 0.25) / 20.0) * gLow;
      const fStart = kEv.fStart[i];
      const fBody = kEv.fBody[i];
      const df = fStart - fBody;
      const tauS = Math.max(0.02 * sr, kEv.tauMs[i] * 0.001 * sr);
      const tauH2 = 0.42 * tauS;
      const len = Math.min(numSamples - s0, Math.round(5.2 * tauS));
      let phi = 0.0;
      for (let t = 0; t < len; t++) {
        const fc = fBody + df * Math.exp(-t / tauPitch);
        phi += (twoPi * fc) / sr;
        if (phi >= twoPi) phi -= twoPi;
        const atk = t < atkSamples ? 0.5 * (1.0 - Math.cos((Math.PI * t) / atkSamples)) : 1.0;
        const env = atk * Math.exp(-t / tauS);
        const w = amp * env * (Math.sin(phi) + 0.16 * Math.exp(-t / tauH2) * Math.sin(2.0 * phi));
        outL[s0 + t] += w;
        outR[s0 + t] += w;
      }
    }
  }

  // 2. Modal Shell / Woodblock / Rimshot Resonators (`midModal`) — 100% noise-free tonal percussion
  const gModal = gains.midModal ?? 1.0;
  if (gModal > 0 && score.midModal && score.midModal.frame) {
    const mEv = score.midModal;
    const atkM = Math.max(12, Math.round(0.0012 * sr));
    let fr = 0;
    for (let i = 0; i < mEv.frame.length; i++) {
      fr += mEv.frame[i];
      const s0 = fr * hop;
      if (s0 >= numSamples) break;
      const amp = Math.pow(10.0, (mEv.ampQ[i] * 0.25) / 20.0) * gModal;
      const r2 = mEv.r2Pct[i] * 0.01;
      const norm = 1.0 / Math.sqrt(1.0 + r2 * r2);
      const w1 = (twoPi * mEv.f1[i]) / sr;
      const w2 = (twoPi * mEv.f2[i]) / sr;
      const tau1 = Math.max(0.008 * sr, mEv.tauMs[i] * 0.001 * sr);
      const tau2 = 0.72 * tau1;
      const th = ((mEv.pan[i] ?? 32) / 64.0) * (0.5 * Math.PI);
      const pL = Math.cos(th) * Math.SQRT2;
      const pR = Math.sin(th) * Math.SQRT2;
      const len = Math.min(numSamples - s0, Math.round(5.0 * tau1));
      let ph1 = 0.0;
      let ph2 = 0.0;
      for (let t = 0; t < len; t++) {
        ph1 += w1;
        ph2 += w2;
        const atk = t < atkM ? 0.5 * (1.0 - Math.cos((Math.PI * t) / atkM)) : 1.0;
        const v =
          amp *
          norm *
          atk *
          (Math.exp(-t / tau1) * Math.sin(ph1) + r2 * Math.exp(-t / tau2) * Math.sin(ph2));
        outL[s0 + t] += pL * v;
        outR[s0 + t] += pR * v;
      }
    }
  }

  // 3. Tight Gated Snare-Wire (`midSnap`) & Crisp Hi-Hat / Shaker (`high`) — 512-pt Sine-WOLA
  const nf = Math.floor(numSamples / hop) + 2;
  const fft = new Radix2ComplexFFT(nfft);
  const win = Float64Array.from({ length: nfft }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / nfft));
  const scaleWola = Math.sqrt(nfft); // restores unit-variance Parseval energy after 1/nfft IFFT
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);

  let famIdx = 0;
  for (const [name, fam] of Object.entries(model.families)) {
    const famGain = gains[name] ?? 1.0;
    const ev = score[name];
    if (famGain <= 0 || !ev || !ev.frame) {
      famIdx++;
      continue;
    }
    const nMel = fam.nMel;
    const T = fam.T;
    const fb = melFilterbank(sr, nfft, nMel, fam.band[0], fam.band[1]);
    const colSum = fb.map((row) => row.reduce((a, b) => a + b, 0) || 1.0);
    const binW = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) for (let b = 0; b < nMel; b++) binW[k] += fb[b][k];

    const tmplLin = fam.templatesDb.map((t) =>
      t.map((row) => row.map((v) => (v <= -170 ? 0.0 : Math.pow(10.0, (v * 0.5) / 10.0))))
    );
    const sideLin = fam.sideDb.map((row) => row.map((v) => Math.pow(10.0, v / 10.0)));

    const P = new Float32Array(nf * nMel);
    const PS = new Float32Array(nf * nMel);
    let fr = 0;
    for (let i = 0; i < ev.frame.length; i++) {
      fr += ev.frame[i];
      const k = ev.tmpl[i];
      const g = Math.pow(10.0, (ev.gainQ[i] * 0.25) / 10.0);
      const tk = tmplLin[k];
      const sk = sideLin[k];
      for (let t = 0; t < T && fr + t < nf; t++) {
        const base = (fr + t) * nMel;
        for (let b = 0; b < nMel; b++) {
          const pv = tk[b][t];
          if (pv === 0.0) continue;
          const v = g * pv;
          P[base + b] += v;
          PS[base + b] += v * sk[b];
        }
      }
    }

    const rngM = createMulberry32(2100 + famIdx * 29);
    const rngS = createMulberry32(6300 + famIdx * 43);
    const magM = new Float64Array(nBins);
    const magS = new Float64Array(nBins);

    for (let f = 0; f < nf; f++) {
      const base = f * nMel;
      let totP = 0.0;
      for (let b = 0; b < nMel; b++) totP += P[base + b];
      if (totP <= 1e-12) continue;

      for (let k = 0; k < nBins; k++) {
        if (binW[k] <= 0) {
          magM[k] = 0.0;
          magS[k] = 0.0;
          continue;
        }
        let sm = 0.0;
        let ss = 0.0;
        for (let b = 0; b < nMel; b++) {
          const w = fb[b][k];
          if (w === 0.0) continue;
          sm += (w * P[base + b]) / colSum[b];
          ss += (w * PS[base + b]) / colSum[b];
        }
        magM[k] = Math.sqrt(Math.max(0.0, sm / binW[k]));
        magS[k] = Math.sqrt(Math.max(0.0, ss / binW[k]));
      }

      const s0 = f * hop - (nfft >> 1);
      for (let pass = 0; pass < 2; pass++) {
        const mag = pass === 0 ? magM : magS;
        re.fill(0.0);
        im.fill(0.0);
        let energy = 0.0;
        const rng = pass === 0 ? rngM : rngS;
        for (let k = 1; k < nBins - 1; k++) {
          const mk = mag[k];
          if (mk === 0.0) continue;
          energy += mk;
          const ph = rng() * twoPi;
          re[k] = mk * Math.cos(ph);
          im[k] = mk * Math.sin(ph);
          re[nfft - k] = re[k];
          im[nfft - k] = -im[k];
        }
        if (energy === 0.0) continue;
        fft.transform(re, im, true);
        const outScale = scaleWola * famGain;
        for (let i = 0; i < nfft; i++) {
          const s = s0 + i;
          if (s < 0 || s >= numSamples) continue;
          const v = re[i] * win[i] * outScale;
          if (pass === 0) {
            outL[s] += v;
            outR[s] += v;
          } else {
            outL[s] += v;
            outR[s] -= v;
          }
        }
      }
    }
    famIdx++;
  }

  return [outL, outR];
}
