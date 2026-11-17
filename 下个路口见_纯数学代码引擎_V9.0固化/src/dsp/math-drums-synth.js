// Pure Mathematical Drum & Metallic Cymbal Synthesizer (Zero Samples / Zero Binary)
// Generates:
//   06 drum_kick_snare (Membrane ODE Kick + Dual-Bessel-Mode Snare + 4-Tap Stereo Handclap)
//   07 drum_hihat_perc (6-Oscillator Non-Harmonic Metallic Cymbal Matrix + Pedal Choke + Stereo Shaker)

import { DeterministicRng } from './fft-minphase.js';

const TWO_PI = 2.0 * Math.PI;

function renderOneKick(bufL, bufR, startSample, vel, cfg, sr) {
  const durS = Math.round(0.36 * sr);
  const endS = Math.min(bufL.length, startSample + durS);
  let phase = 0.0;
  let clickPhase = 0.0;
  for (let s = startSample; s < endS; s++) {
    if (s < 0) continue;
    const t = (s - startSample) / sr;
    const fInst = cfg.fInfHz + (cfg.fStartHz - cfg.fInfHz) * Math.exp(-t / cfg.pitchTauSec);
    phase += (TWO_PI * fInst) / sr;
    clickPhase += (TWO_PI * cfg.clickHz) / sr;

    const attack = 1.0 - Math.exp(-t / 0.0011);
    const bodyEnv = attack * Math.exp(-t / cfg.ampDecaySec);
    const clickEnv = Math.exp(-t / cfg.clickTauSec);
    const raw = (Math.sin(phase) * bodyEnv + 0.24 * Math.sin(clickPhase) * clickEnv) * vel;
    const sat = Math.tanh(raw * cfg.drive) * 0.42;
    bufL[s] += sat;
    bufR[s] += sat;
  }
}

function renderOneSnareAndClap(bufL, bufR, startSample, vel, isBackbeat, cfg, rng, sr) {
  const durS = Math.round(0.32 * sr);
  const endS = Math.min(bufL.length, startSample + durS);
  let p1 = 0.0;
  let p2 = 0.0;
  // Chamberlin 2-pole SVF states for snare wires & clap cavity
  const fWire = 2.0 * Math.sin((Math.PI * cfg.wireCenterHz) / sr);
  const qWire = 1.0 / cfg.wireQ;
  let lpW = 0.0;
  let bpW = 0.0;

  const fClap = 2.0 * Math.sin((Math.PI * cfg.clapCenterHz) / sr);
  const qClap = 1.0 / cfg.clapQ;
  let lpC = 0.0;
  let bpC = 0.0;

  for (let s = startSample; s < endS; s++) {
    if (s < 0) continue;
    const t = (s - startSample) / sr;
    const pitchDrop = 1.0 + 0.16 * Math.exp(-t / 0.012);
    p1 += (TWO_PI * cfg.mode1Hz * pitchDrop) / sr;
    p2 += (TWO_PI * cfg.mode2Hz * pitchDrop) / sr;

    const drumShell = 0.65 * Math.sin(p1) * Math.exp(-t / cfg.mode1TauSec)
                    + 0.35 * Math.sin(p2) * Math.exp(-t / cfg.mode2TauSec);

    const nIn = rng.uniform();
    lpW += fWire * bpW;
    const hpW = nIn - lpW - qWire * bpW;
    bpW += fWire * hpW;
    const wireEnv = 0.68 * Math.exp(-t / cfg.wireTauFastSec) + 0.32 * Math.exp(-t / cfg.wireTauSlowSec);
    const snareSig = (0.48 * drumShell + 0.56 * bpW * wireEnv) * vel * 0.34;

    let clapL = 0.0;
    let clapR = 0.0;
    if (isBackbeat) {
      const cIn = rng.uniform();
      lpC += fClap * bpC;
      const hpC = cIn - lpC - qClap * bpC;
      bpC += fClap * hpC;

      let cEnv = 0.0;
      const tMs = t * 1000.0;
      for (let k = 0; k < cfg.clapOffsetsMs.length; k++) {
        const dt = tMs - cfg.clapOffsetsMs[k];
        if (dt >= 0.0) {
          const tau = k === cfg.clapOffsetsMs.length - 1 ? cfg.clapTauSec : 0.0085;
          cEnv += (k === cfg.clapOffsetsMs.length - 1 ? 0.55 : 0.38) * Math.exp((-dt * 0.001) / tau);
        }
      }
      const clapSig = bpC * cEnv * vel * 0.19;
      clapL = clapSig * 1.08;
      clapR = clapSig * 0.92;
    }

    bufL[s] += snareSig + clapL;
    bufR[s] += snareSig + clapR;
  }
}

export function synthesizePureMathDrums(bandScore, instPhysics, sr, nSamples) {
  const kickSnareL = new Float32Array(nSamples);
  const kickSnareR = new Float32Array(nSamples);
  const hatPercL = new Float32Array(nSamples);
  const hatPercR = new Float32Array(nSamples);
  const kickSidechainEnv = new Float32Array(nSamples).fill(1.0);

  const rng = new DeterministicRng(0xD808F00D);
  const barSec = 240.0 / bandScore.bpm;
  const stepSec = barSec / 16.0;
  const bar0 = bandScore.bar0Sec;

  const kickCfg = instPhysics.kick;
  const snareCfg = instPhysics.snare;
  const cymCfg = instPhysics.cymbal;

  const applyKickDuck = (startS, strength) => {
    const len = Math.round(0.14 * sr);
    const endS = Math.min(nSamples, startS + len);
    for (let s = Math.max(0, startS); s < endS; s++) {
      const t = (s - startS) / sr;
      const duck = 1.0 - strength * 0.24 * Math.exp(-t / 0.058);
      if (duck < kickSidechainEnv[s]) kickSidechainEnv[s] = duck;
    }
  };

  // Pre-Bar-0 2-beat studio pickup
  for (const hit of bandScore.pickupHits) {
    const s0 = Math.round(hit.sec * sr);
    if (hit.type === 'snare') {
      renderOneSnareAndClap(kickSnareL, kickSnareR, s0, hit.vel, false, snareCfg, rng, sr);
    } else if (hit.type === 'kick_bass') {
      renderOneKick(kickSnareL, kickSnareR, s0, hit.vel, kickCfg, sr);
      applyKickDuck(s0, hit.vel);
    }
  }

  // Collect all hi-hat / crash / shaker events so closed hats naturally choke open hats!
  const hatEvents = [];
  const crashEvents = [];
  const shakerEvents = [];

  for (const sec of bandScore.sections) {
    for (let bOff = 0; bOff < sec.bars; bOff++) {
      const barIdx = sec.startBar + bOff;
      const tBar = bar0 + barIdx * barSec;

      if (sec.drum === 'FINAL') {
        if (bOff === 0) {
          const sDown = Math.round(tBar * sr);
          renderOneKick(kickSnareL, kickSnareR, sDown, 0.85, kickCfg, sr);
          applyKickDuck(sDown, 0.85);
          crashEvents.push({ sample: sDown, vel: 0.58 });
        }
        continue;
      }

      if (sec.crashFirst && bOff === 0) {
        crashEvents.push({ sample: Math.round(tBar * sr), vel: 0.52 });
      }

      let patKey = sec.drum;
      if (sec.firstBarDrum && bOff === 0) patKey = sec.firstBarDrum;
      if (sec.fillLast && bOff === sec.bars - 1) patKey = 'FILL_BAR';
      const pat = bandScore.drumPatterns[patKey];

      for (let st = 0; st < 16; st++) {
        const swing = (st & 1) ? cymCfg.swing16thSec : 0.0;
        const sStep = Math.round((tBar + st * stepSec + swing) * sr);
        if (sStep >= nSamples) continue;

        const kv = pat.k[st] / 9.0;
        if (kv > 0.0) {
          renderOneKick(kickSnareL, kickSnareR, sStep, kv, kickCfg, sr);
          applyKickDuck(sStep, kv);
        }

        const sv = pat.s[st] / 9.0;
        if (sv > 0.0) {
          const isBackbeat = st === 4 || st === 12;
          renderOneSnareAndClap(kickSnareL, kickSnareR, sStep, sv, isBackbeat, snareCfg, rng, sr);
        }

        const hc = pat.h[st];
        if (hc && hc !== '.') {
          hatEvents.push({ sample: sStep, type: hc, vel: hc === 'o' ? 0.68 : (st % 4 === 0 ? 0.58 : 0.42) });
        }

        const shv = pat.sh[st] / 9.0;
        if (shv > 0.0) {
          shakerEvents.push({ sample: sStep, vel: shv, panRight: (st & 1) === 1 });
        }
      }
    }
  }

  // Synthesize 6-oscillator non-harmonic metallic hi-hats with authentic pedal choke
  const mFreqs = cymCfg.metallicFreqsHz;
  const mPhases = new Float64Array(6);
  const fHp = 2.0 * Math.sin((Math.PI * Math.min(sr * 0.42, cymCfg.highpassHz)) / sr);
  let lpHat = 0.0;
  let bpHat = 0.0;

  for (let idx = 0; idx < hatEvents.length; idx++) {
    const ev = hatEvents[idx];
    const nextSample = idx + 1 < hatEvents.length ? hatEvents[idx + 1].sample : nSamples;
    const tau = ev.type === 'o' ? cymCfg.openTauSec : (ev.type === 'p' ? 0.018 : cymCfg.closedTauSec);
    const maxDur = Math.round(tau * 5.2 * sr);
    const chokeLimit = ev.type === 'o' ? Math.min(nSamples, nextSample + Math.round(0.012 * sr)) : nSamples;
    const endS = Math.min(nSamples, ev.sample + maxDur, chokeLimit);

    for (let s = ev.sample; s < endS; s++) {
      if (s < 0) continue;
      const t = (s - ev.sample) / sr;
      let sqSum = 0.0;
      for (let o = 0; o < 6; o++) {
        mPhases[o] += (TWO_PI * mFreqs[o]) / sr;
        if (mPhases[o] > TWO_PI) mPhases[o] -= TWO_PI;
        sqSum += mPhases[o] < Math.PI ? 1.0 : -1.0;
      }
      const metalIn = 0.55 * (sqSum / 6.0) + 0.45 * rng.uniform();
      lpHat += fHp * bpHat;
      const hpHat = metalIn - lpHat - 0.85 * bpHat;
      bpHat += fHp * hpHat;

      let chokeFade = 1.0;
      if (ev.type === 'o' && s > nextSample - Math.round(0.008 * sr)) {
        chokeFade = Math.max(0.0, (chokeLimit - s) / Math.max(1, 0.020 * sr));
      }
      const amp = Math.exp(-t / tau) * chokeFade * ev.vel * 0.145;
      hatPercL[s] += hpHat * amp * 0.90;
      hatPercR[s] += hpHat * amp * 1.10;
    }
  }

  // Synthesize section downbeat stereo Crash Cymbals
  for (const cr of crashEvents) {
    const endS = Math.min(nSamples, cr.sample + Math.round(2.4 * sr));
    let lpCr = 0.0;
    let bpCr = 0.0;
    const fCr = 2.0 * Math.sin((Math.PI * 5200.0) / sr);
    for (let s = cr.sample; s < endS; s++) {
      if (s < 0) continue;
      const t = (s - cr.sample) / sr;
      let sq = 0.0;
      for (let o = 0; o < 6; o++) {
        sq += Math.sin(TWO_PI * mFreqs[o] * 1.18 * t + o * 1.7) >= 0 ? 1.0 : -1.0;
      }
      const raw = 0.42 * (sq / 6.0) + 0.58 * rng.uniform();
      lpCr += fCr * bpCr;
      const hpCr = raw - lpCr - 0.75 * bpCr;
      bpCr += fCr * hpCr;
      const env = (1.0 - Math.exp(-t / 0.004)) * Math.exp(-t / cymCfg.crashTauSec) * cr.vel * 0.13;
      hatPercL[s] += hpCr * env * 1.08;
      hatPercR[s] += hpCr * env * 0.92;
    }
  }

  // Synthesize subtle 16th-note stereo Shaker groove
  const fSh = 2.0 * Math.sin((Math.PI * cymCfg.shakerCenterHz) / sr);
  let lpSh = 0.0;
  let bpSh = 0.0;
  const shDur = Math.round(0.055 * sr);
  for (const sh of shakerEvents) {
    const endS = Math.min(nSamples, sh.sample + shDur);
    for (let s = sh.sample; s < endS; s++) {
      if (s < 0) continue;
      const t = (s - sh.sample) / sr;
      lpSh += fSh * bpSh;
      const hpSh = rng.uniform() - lpSh - 0.65 * bpSh;
      bpSh += fSh * hpSh;
      const env = Math.sin(Math.PI * Math.min(1.0, t / 0.012)) * Math.exp(-t / cymCfg.shakerTauSec) * sh.vel * 0.055;
      hatPercL[s] += bpSh * env * (sh.panRight ? 0.72 : 1.18);
      hatPercR[s] += bpSh * env * (sh.panRight ? 1.18 : 0.72);
    }
  }

  return {
    drum_kick_snare: [kickSnareL, kickSnareR],
    drum_hihat_perc: [hatPercL, hatPercR],
    kickSidechainEnv,
  };
}
