// Pure Mathematical Fingered J-Bass (Chamberlin SVF) & Extended Karplus-Strong Funk Guitar Synthesizer
// Generates:
//   08 bass_groove      (Asymmetric pickup wave + 2-pole envelope-swept SVF + Kick sidechain lock)
//   09 guitar_funk_riff (4-string fractional-delay Karplus-Strong waveguide + Strat pickup comb + palm mute)

import { DeterministicRng } from './fft-minphase.js';

const TWO_PI = 2.0 * Math.PI;

function midiToHz(midi) {
  return 440.0 * Math.pow(2.0, (midi - 69.0) / 12.0);
}

function renderBassNote(buf, startS, durS, midi, vel, cfg, sr) {
  const nTot = Math.min(buf.length - startS, durS + Math.round(cfg.ampReleaseSec * sr));
  if (nTot <= 0) return;
  const freq = midiToHz(midi);
  let phase = 0.0;
  let lp = 0.0;
  let bp = 0.0;
  const qInv = 1.0 / cfg.svfQ;
  const combShift = TWO_PI * cfg.pickupCombRatio;

  for (let i = 0; i < nTot; i++) {
    const s = startS + i;
    if (s < 0) continue;
    const t = i / sr;
    phase += (TWO_PI * freq) / sr;
    if (phase >= TWO_PI) phase -= TWO_PI;

    // J-Bass asymmetric string harmonic wave + warm fundamental sine
    const saw1 = (phase / Math.PI) - 1.0;
    const p2 = phase + combShift;
    const saw2 = ((p2 >= TWO_PI ? p2 - TWO_PI : p2) / Math.PI) - 1.0;
    const rawStr = cfg.subSineMix * Math.sin(phase) + (1.0 - cfg.subSineMix) * (saw1 - 0.38 * saw2);

    // Dynamic envelope-modulated 2-pole Chamberlin State-Variable Filter
    const fc = cfg.svfBaseCutoffHz + cfg.svfEnvModHz * vel * Math.exp(-t / cfg.svfEnvTauSec);
    const fCoeff = 2.0 * Math.sin((Math.PI * Math.min(sr * 0.38, fc)) / sr);
    lp += fCoeff * bp;
    const hp = rawStr - lp - qInv * bp;
    bp += fCoeff * hp;

    const attack = 1.0 - Math.exp(-t / 0.0035);
    const rel = i > durS ? Math.exp(-(i - durS) / (cfg.ampReleaseSec * sr)) : Math.exp(-t / 1.65);
    buf[s] += Math.tanh(lp * 1.35) * attack * rel * vel * 0.28;
  }
}

function pluckKarplusStrongString(outBuf, startS, maxDurS, midi, lossRho, brightBlend, pickupRatio, vel, rng, sr) {
  const freq = midiToHz(midi);
  const exactDelay = sr / freq;
  const N = Math.max(8, Math.floor(exactDelay - 0.5));
  const frac = exactDelay - N;
  const allpassC = (1.0 - frac) / (1.0 + frac);

  const ring = new Float32Array(N + 2);
  const rawExcite = new Float32Array(N + 2);
  let lpInit = 0.0;
  for (let i = 0; i < N + 2; i++) {
    lpInit = 0.42 * lpInit + 0.58 * rng.uniform();
    rawExcite[i] = lpInit;
  }
  // Stratocaster bridge-middle pickup comb excitation
  const combLag = Math.max(1, Math.round(N * pickupRatio));
  for (let i = 0; i < N; i++) {
    const iLag = (i - combLag + N) % N;
    ring[i] = (rawExcite[i] - 0.72 * rawExcite[iLag]) * vel;
  }

  const endS = Math.min(outBuf.length, startS + maxDurS);
  let rdIdx = 0;
  let prevAvg = 0.0;
  let prevApIn = 0.0;
  let prevApOut = 0.0;

  for (let s = startS; s < endS; s++) {
    const y0 = ring[rdIdx];
    const nextIdx = (rdIdx + 1) % N;
    const y1 = ring[nextIdx];

    // Two-point string loop lowpass + Thiran 1st-order allpass fractional tuning
    const avg = (1.0 - brightBlend) * y0 + brightBlend * y1;
    const apOut = allpassC * avg + prevApIn - allpassC * prevApOut;
    prevApIn = avg;
    prevApOut = apOut;

    const damped = apOut * lossRho;
    ring[rdIdx] = damped;
    rdIdx = nextIdx;

    if (s >= 0) {
      // Gentle body high-pass removes mud below 150 Hz + subtle clean tube amp saturation
      const bodyOut = damped - 0.965 * prevAvg;
      prevAvg = damped;
      outBuf[s] += Math.tanh(bodyOut * 1.9) * 0.058;
    }
  }
}

export function synthesizePureMathBassAndGuitar(bandScore, instPhysics, kickSidechainEnv, sr, nSamples) {
  const bassMono = new Float32Array(nSamples);
  const gtrL = new Float32Array(nSamples);
  const gtrR = new Float32Array(nSamples);

  const rngL = new DeterministicRng(0x6774724C);
  const rngR = new DeterministicRng(0x67747252);
  const barSec = 240.0 / bandScore.bpm;
  const stepSec = barSec / 16.0;
  const bar0 = bandScore.bar0Sec;
  const bassCfg = instPhysics.bass;
  const gtrCfg = instPhysics.guitar;

  // Pre-Bar-0 2-beat bass pickup
  for (const hit of bandScore.pickupHits) {
    if (hit.type === 'kick_bass') {
      renderBassNote(bassMono, Math.round(hit.sec * sr), Math.round(hit.dur * sr), hit.midi, hit.vel, bassCfg, sr);
    }
  }

  for (const sec of bandScore.sections) {
    for (let bOff = 0; bOff < sec.bars; bOff++) {
      const barIdx = sec.startBar + bOff;
      const tBar = bar0 + barIdx * barSec;

      if (sec.prog === 'FINAL') {
        if (bOff === 0) {
          const ch = bandScore.chords.D9end;
          const s0 = Math.round(tBar * sr);
          renderBassNote(bassMono, s0, Math.round(3.8 * sr), ch.bass, 0.88, bassCfg, sr);
          const notes = ch.guitar;
          for (let nIdx = 0; nIdx < notes.length; nIdx++) {
            const stOff = Math.round(nIdx * 0.012 * sr);
            pluckKarplusStrongString(gtrL, s0 + stOff, Math.round(4.2 * sr), notes[nIdx], 0.9965, 0.25, gtrCfg.pickupPosRatio, 0.78, rngL, sr);
            pluckKarplusStrongString(gtrR, s0 + stOff + Math.round(0.009 * sr), Math.round(4.2 * sr), notes[nIdx], 0.9965, 0.27, gtrCfg.pickupPosRatio, 0.78, rngR, sr);
          }
        }
        continue;
      }

      const pIdx = ((bOff + (sec.progOffset || 0)) % 4);
      let [chName0, chName1] = bandScore.progressions[sec.prog][pIdx];
      if (sec.resolveLastDm && bOff === sec.bars - 1) {
        chName1 = 'Dm9';
      }
      const ch0 = bandScore.chords[chName0];
      const ch1 = bandScore.chords[chName1];

      // 1. Render Bass Groove Pattern
      const bPat = bandScore.bassPatterns[sec.bass];
      for (const [st, durSteps, useOct, vel, chrom] of bPat) {
        const ch = st < 8 ? ch0 : ch1;
        let midi = useOct ? ch.altBass : ch.bass;
        if (chrom !== 0 && st >= 14) {
          midi = ch1.bass + chrom;
        }
        const sStart = Math.round((tBar + st * stepSec) * sr);
        const sDur = Math.round(durSteps * stepSec * sr);
        renderBassNote(bassMono, sStart, sDur, midi, vel, bassCfg, sr);
      }

      // 2. Render Stereo Double-Tracked Extended Karplus-Strong Clean Funk Guitar
      const gPat = bandScore.guitarPatterns[sec.gtr];
      for (let st = 0; st < 16; st++) {
        if (sec.gtrEnterHalfBar && bOff === 0 && st < 8) continue; // Guitar enters cleanly on Bar 0 Beat 3 (2.974s)!
        const act = gPat[st];
        if (!act || act === '.') continue;

        const swing = (st & 1) ? 0.0055 : 0.0;
        const sStep = Math.round((tBar + st * stepSec + swing) * sr);
        if (sStep >= nSamples) continue;

        const ch = st < 8 ? ch0 : ch1;
        const notes = ch.guitar;
        const isMute = act === 'm';
        const isUp = act === 'U' || act === 'A';
        const isAcc = act === 'A';
        const rho = isMute ? gtrCfg.palmMuteLoss : gtrCfg.openLoopLoss;
        const vel = isAcc ? 0.92 : (isMute ? 0.62 : 0.76);
        const maxDurS = Math.round((isMute ? 0.095 : stepSec * 2.4) * sr);
        const spreadS = Math.round(gtrCfg.strumSpreadMs * 0.001 * sr);
        const dblS = Math.round(gtrCfg.stereoDoubleDelayMs * 0.001 * sr);

        for (let i = 0; i < notes.length; i++) {
          const strIdx = isUp ? (notes.length - 1 - i) : i;
          const sOnL = sStep + i * spreadS;
          const sOnR = sOnL + dblS;
          pluckKarplusStrongString(gtrL, sOnL, maxDurS, notes[strIdx], rho, gtrCfg.brightnessBlend, gtrCfg.pickupPosRatio, vel, rngL, sr);
          pluckKarplusStrongString(gtrR, sOnR, maxDurS, notes[strIdx], rho, gtrCfg.brightnessBlend * 1.05, gtrCfg.pickupPosRatio * 1.08, vel * 0.96, rngR, sr);
        }
      }
    }
  }

  // Apply Kick sidechain pocket ducking to Bass
  const bassL = new Float32Array(nSamples);
  const bassR = new Float32Array(nSamples);
  for (let s = 0; s < nSamples; s++) {
    const v = bassMono[s] * kickSidechainEnv[s];
    bassL[s] = v;
    bassR[s] = v;
  }

  return {
    bass_groove: [bassL, bassR],
    guitar_funk_riff: [gtrL, gtrR],
  };
}
