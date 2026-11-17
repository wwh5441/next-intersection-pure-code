// V9R1 Pure Mathematical 18-Harmonic Inharmonic Bass + 56-Note Coupled-Rotor Guitar/Keys & 80fps Strum Synth
// Driven by Expanded Bar-Motif Dictionary Trajectories:
//   08 bass_groove      -> 1-Channel Mono `[bassMono]` (18-Harmonic String f_k = k*f0*sqrt(1+B*k^2))
//   09 guitar_funk_riff -> 2-Channel Stereo `[gtrL, gtrR]` (Coupled-Rotor MIDI 62..95 + 80fps Strum + Stereo FDN)
//   10 keys_rhodes_pad  -> 2-Channel Stereo `[keysL, keysR]` (Coupled-Rotor MIDI 40..61 + Stereo FDN)

import {
  ComplexInverseFFT,
  DeterministicRng,
  interpPiecewiseLinear,
  upsample2xPolyphase,
  upsample4xPolyphase,
} from './fft-minphase.js';
import { applyInstrumentStereoFdn } from './space-fdn-synth.js';

const TWO_PI = 2.0 * Math.PI;
const N_OLA = 512;
const HALF_OLA = N_OLA >> 1;
const N_BINS = HALF_OLA + 1;
const LEAD_PAD = 8;

export function synthesizeV9r1Bass(bassTimeline, instPhysics, nSamples) {
  const model = instPhysics.bass;
  const sr12 = model.sampleRate;
  const hop = model.hopSamples;
  const maxHarm = model.maxHarmonics;
  const inharmB = model.inharmonicityB;
  const { nFrames, rmsQ, f0Q, nrQ, toneIdx } = bassTimeline;

  const staticProf = Float64Array.from(model.staticHarmonicLogProfile);
  const nCb = model.toneCodebookI8.length;
  const cbTone = new Float64Array(nCb * maxHarm);
  for (let c = 0; c < nCb; c++) {
    const row = model.toneCodebookI8[c];
    for (let k = 0; k < maxHarm; k++) cbTone[c * maxHarm + k] = row[k] * model.toneScale;
  }

  const kStretch = new Float64Array(maxHarm);
  const harmPhases = new Float64Array(maxHarm);
  for (let k = 0; k < maxHarm; k++) {
    const idx = k + 1;
    kStretch[k] = idx * Math.sqrt(1.0 + inharmB * idx * idx);
    harmPhases[k] = -0.22 * idx;
  }

  const amps0 = new Float64Array(maxHarm);
  const amps1 = new Float64Array(maxHarm);
  const nyq = sr12 * 0.46;

  function evalBassFrame(t, outAmps) {
    if (t < 0 || t >= nFrames || rmsQ[t] <= 0.0 || f0Q[t] <= 0.0) {
      outAmps.fill(0.0);
      return { f0: 0.0, noiseAmp: 0.0 };
    }
    const rms = Math.pow(10.0, (rmsQ[t] / 3.2 - 75.0) / 20.0);
    const f0 = 32.7032 * Math.pow(2.0, (f0Q[t] - 1.0) / 384.0);
    const nr = Math.min(0.98, Math.max(0.0, nrQ[t] / 255.0));
    const tIdx = Math.max(0, Math.min(nCb - 1, toneIdx[t]));
    const cbOff = tIdx * maxHarm;

    let hVar = 1e-12;
    for (let k = 0; k < maxHarm; k++) {
      if (kStretch[k] * f0 >= nyq) {
        outAmps[k] = 0.0;
      } else {
        const logV = Math.max(-12.0, Math.min(8.0, staticProf[k] + cbTone[cbOff + k]));
        const rel = Math.exp(logV);
        outAmps[k] = rel;
        hVar += 0.5 * rel * rel;
      }
    }
    const scale = Math.sqrt((rms * rms * (1.0 - nr * nr)) / hVar);
    for (let k = 0; k < maxHarm; k++) outAmps[k] *= scale;
    return { f0, noiseAmp: rms * nr };
  }

  const n12 = Math.ceil(nSamples / 4) + 32;
  const bass12Pad = new Float32Array(LEAD_PAD + n12);
  const rng = new DeterministicRng(202);
  let lpState = 0.0;
  const invHop = 1.0 / hop;
  const twoPiOverSr = TWO_PI / sr12;

  let st0 = evalBassFrame(0, amps0);
  for (let t = 0; t < nFrames - 1; t++) {
    const st1 = evalBassFrame(t + 1, amps1);
    const s0 = LEAD_PAD + t * hop;
    if (s0 + hop >= bass12Pad.length) break;

    const fc = st0.f0, fn = st1.f0;
    if (fc > 30.0 || fn > 30.0) {
      const fs = fc > 30.0 ? fc : fn;
      const fe = fn > 30.0 ? fn : fc;
      for (let i = 0; i < hop; i++) {
        const a = i * invHop;
        const dphi0 = twoPiOverSr * (fs + (fe - fs) * a);
        let acc = 0.0;
        for (let k = 0; k < maxHarm; k++) {
          let ph = harmPhases[k] + kStretch[k] * dphi0;
          if (ph >= TWO_PI) ph -= TWO_PI;
          harmPhases[k] = ph;
          const amp = amps0[k] + (amps1[k] - amps0[k]) * a;
          if (amp > 0.0) acc += amp * Math.cos(ph);
        }
        bass12Pad[s0 + i] = acc;
      }
    }

    const nAmp = 0.5 * (st0.noiseAmp + st1.noiseAmp);
    if (nAmp > 1e-5) {
      let lp = lpState;
      for (let i = 0; i < hop; i++) {
        lp = 0.65 * lp + 0.35 * rng.uniform();
        bass12Pad[s0 + i] += lp * nAmp * 1.6;
      }
      lpState = lp;
    }

    amps0.set(amps1);
    st0 = st1;
  }

  const bassMono = new Float32Array(nSamples);
  upsample4xPolyphase(bass12Pad, bassMono);
  return { bass_groove: [bassMono] };
}

export function synthesizeV9r1GuitarAndKeys(harmTimeline, instPhysics, outSr, nSamples) {
  const model = instPhysics.harmony;
  const sr24 = model.sampleRate;
  const hop = model.hopSamples;
  const hopS = hop >> 1;
  const { nFrames, nSFrames, qLevels, maxNoteAmp, harmQ, brightQ, sideQ, strumCols, noteCols } = harmTimeline;
  const nNotes = model.nNotes;
  const maxHarm = model.maxHarmonics;
  const splitIdx = model.guitarSplitMidi - model.midiMin;

  const warmW = Float64Array.from(model.warmHarmonicWeights);
  const brightW = Float64Array.from(model.brightHarmonicWeights);
  const strumCenters = Float64Array.from(model.strumCentersHz);

  // Coupled-rotor phase state remains 100% continuous across all 86 bar boundaries!
  const cosStep = new Float64Array(nNotes);
  const sinStep = new Float64Array(nNotes);
  const noteHarms = new Uint8Array(nNotes);
  const rotorRe = new Float64Array(nNotes);
  const rotorIm = new Float64Array(nNotes);
  const initRng = new DeterministicRng(505);

  for (let n = 0; n < nNotes; n++) {
    const midi = model.midiMin + n;
    const f0 = 440.0 * Math.pow(2.0, (midi - 69.0) / 12.0);
    const omega = (TWO_PI * f0) / sr24;
    cosStep[n] = Math.cos(omega);
    sinStep[n] = Math.sin(omega);
    noteHarms[n] = Math.min(maxHarm, Math.floor(4350.0 / f0));
    const p0 = initRng.nextFloat01() * TWO_PI;
    rotorRe[n] = Math.cos(p0);
    rotorIm[n] = Math.sin(p0);
  }

  const a0 = new Float64Array(nNotes), a1 = new Float64Array(nNotes);
  const w0 = new Float64Array(maxHarm), w1 = new Float64Array(maxHarm);
  const invQL = 1.0 / qLevels;

  function evalChordFrame(t, outA, outW) {
    if (t < 0 || t >= nFrames || harmQ[t] <= 0.0) {
      outA.fill(0.0);
      outW.fill(0.0);
      return;
    }
    const harmRms = Math.pow(10.0, (harmQ[t] / 3.2 - 75.0) / 20.0);
    const br = Math.min(1.0, Math.max(0.0, brightQ[t] / 255.0));
    for (let k = 0; k < maxHarm; k++) {
      outW[k] = warmW[k] * (1.0 - br) + brightW[k] * br;
    }

    let varSum = 0.0;
    for (let n = 0; n < nNotes; n++) {
      const q = noteCols[n][t];
      if (q <= 0.0) {
        outA[n] = 0.0;
        continue;
      }
      const norm = q * invQL;
      const amp = norm * norm * maxNoteAmp;
      outA[n] = amp;
      const K = noteHarms[n];
      for (let k = 0; k < K; k++) {
        const hk = amp * outW[k];
        varSum += 0.5 * hk * hk;
      }
    }
    if (varSum > 1e-12) {
      const scale = Math.sqrt((harmRms * harmRms) / varSum);
      for (let n = 0; n < nNotes; n++) outA[n] *= scale;
    }
  }

  const n24 = Math.ceil(nSamples / 2) + N_OLA + 32;
  const gtr24Pad = new Float32Array(LEAD_PAD + n24);
  const keys24Pad = new Float32Array(LEAD_PAD + n24);
  const invHop = 1.0 / hop;

  evalChordFrame(0, a0, w0);
  for (let t = 0; t < nFrames - 1; t++) {
    evalChordFrame(t + 1, a1, w1);
    const s0 = LEAD_PAD + t * hop;
    if (s0 + hop >= gtr24Pad.length) break;

    for (let n = 0; n < nNotes; n++) {
      const aStart = a0[n], aEnd = a1[n];
      const K = noteHarms[n];
      if ((aStart <= 1e-6 && aEnd <= 1e-6) || K === 0) continue;

      const dstBuf = n >= splitIdx ? gtr24Pad : keys24Pad;
      let r1 = rotorRe[n], i1 = rotorIm[n];
      const cs = cosStep[n], ss = sinStep[n];
      const da = (aEnd - aStart) * invHop;

      for (let s = 0; s < hop; s++) {
        const a = s * invHop;
        const noteAmp = aStart + da * s;
        let rk = r1, ik = i1;
        let sampleSum = 0.0;
        for (let k = 0; k < K; k++) {
          const wk = w0[k] + (w1[k] - w0[k]) * a;
          sampleSum += noteAmp * wk * ik;
          const nrk = rk * r1 - ik * i1;
          ik = ik * r1 + rk * i1;
          rk = nrk;
        }
        dstBuf[s0 + s] += sampleSum;

        const nextR1 = r1 * cs - i1 * ss;
        i1 = i1 * cs + r1 * ss;
        r1 = nextR1;
      }

      const magInv = 1.0 / Math.hypot(r1, i1);
      rotorRe[n] = r1 * magInv;
      rotorIm[n] = i1 * magInv;
    }

    a0.set(a1);
    w0.set(w1);
  }

  // Render 80fps 4-Band Acoustic/Funk Guitar Pick Strum Transients (>1950 Hz) -> Track 09
  const fft512 = new ComplexInverseFFT(N_OLA);
  const reIn = new Float64Array(N_BINS), imIn = new Float64Array(N_BINS);
  const timeOla = new Float64Array(N_OLA);
  const winOla = new Float64Array(N_OLA);
  const olaNorm = 1.0 / Math.sqrt((0.375 * N_OLA) / hopS);
  for (let n = 0; n < N_OLA; n++) {
    winOla[n] = 0.5 * (1.0 - Math.cos((TWO_PI * n) / N_OLA)) * olaNorm;
  }
  const olaFreqs = new Float64Array(N_BINS);
  for (let i = 0; i < N_BINS; i++) olaFreqs[i] = (i * sr24) / N_OLA;

  const strumBands = new Float64Array(4);
  const specS = new Float64Array(N_BINS);
  const strumRng = new DeterministicRng(606);

  for (let ts = 0; ts < nSFrames; ts++) {
    let totVar = 0.0;
    for (let b = 0; b < 4; b++) {
      const q = strumCols[b][ts];
      const rms = q > 0.0 ? Math.pow(10.0, (q / 3.2 - 78.0) / 20.0) : 0.0;
      strumBands[b] = rms;
      totVar += rms * rms;
    }
    if (totVar <= 1e-10) continue;

    const cStartRaw = ts * hopS + (hopS >> 1) - HALF_OLA;
    if (cStartRaw < 0 || LEAD_PAD + cStartRaw + N_OLA >= gtr24Pad.length) continue;

    let sumInner2 = 0.0;
    const rightTail = strumBands[3] * 0.4;
    for (let i = 0; i < N_BINS; i++) {
      const f = olaFreqs[i];
      const v = f < 1950.0 ? 0.0 : interpPiecewiseLinear(strumCenters, strumBands, f, 0.0, rightTail);
      specS[i] = v;
      if (i > 0 && i < HALF_OLA) sumInner2 += v * v;
    }

    const irfftRms = Math.sqrt((2.0 / (N_OLA * N_OLA)) * sumInner2 + 1e-12);
    const cal = Math.sqrt(totVar) / irfftRms;
    reIn[0] = 0.0; imIn[0] = 0.0;
    reIn[HALF_OLA] = specS[HALF_OLA] * cal;
    imIn[HALF_OLA] = 0.0;

    for (let i = 1; i < HALF_OLA; i++) {
      const mag = specS[i] * cal;
      const ang = strumRng.nextFloat01() * TWO_PI;
      reIn[i] = mag * Math.cos(ang);
      imIn[i] = mag * Math.sin(ang);
    }

    fft512.inverseReal(reIn, imIn, timeOla);
    const dstBase = LEAD_PAD + cStartRaw;
    for (let n = 0; n < N_OLA; n++) {
      gtr24Pad[dstBase + n] += timeOla[n] * winOla[n];
    }
  }

  const gtr48Mono = new Float32Array(nSamples);
  const keys48Mono = new Float32Array(nSamples);
  upsample2xPolyphase(gtr24Pad, gtr48Mono);
  upsample2xPolyphase(keys24Pad, keys48Mono);

  return {
    guitar_funk_riff: applyInstrumentStereoFdn(gtr48Mono, sideQ, nFrames, hop, sr24, model.stereoAcoustics, outSr),
    keys_rhodes_pad: applyInstrumentStereoFdn(keys48Mono, sideQ, nFrames, hop, sr24, model.stereoAcoustics, outSr),
  };
}
