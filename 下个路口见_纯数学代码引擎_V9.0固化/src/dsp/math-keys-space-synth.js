// Pure Mathematical 2-Op FM Rhodes, Stereo Chorus Pad & 8-Line Prime Householder FDN Reverb
// Generates:
//   10 keys_rhodes_pad  (Yamaha DX / Rhodes 2-op FM equation + 3x detuned analog saw chorus pad)
//   05 studio_fdn_space (4-tap early reflections + 8-line prime-delay Householder FDN studio room)

const TWO_PI = 2.0 * Math.PI;

function midiToHz(midi) {
  return 440.0 * Math.pow(2.0, (midi - 69.0) / 12.0);
}

function renderRhodesAndPadChord(bufL, bufR, startS, durS, notes, epGain, padGain, cfg, sr) {
  const relS = Math.round(cfg.padReleaseSec * sr);
  const endS = Math.min(bufL.length, startS + durS + relS);
  if (endS <= startS || (epGain <= 0.001 && padGain <= 0.001)) return;

  const nNotes = notes.length;
  const freqs = notes.map(midiToHz);
  const epDec = cfg.epDecaySec;
  const fPadCut = 1.0 - Math.exp((-TWO_PI * cfg.padCutoffHz) / sr);
  let padLpL = 0.0;
  let padLpR = 0.0;

  // Precompute detuned pad frequencies (-8.5, 0, +8.5 cents)
  const detRatios = cfg.padDetuneCents.map((c) => Math.pow(2.0, c / 1200.0));
  const padPhases = new Float64Array(nNotes * 3);
  for (let i = 0; i < padPhases.length; i++) padPhases[i] = (i * 0.37) % 1.0;

  // Exact multiplicative exponential recurrences (mathematically identical to exp(-t/tau), 8x faster!)
  let envEpBase = 1.0;
  const kEpDec = Math.exp(-1.0 / (epDec * sr));
  let envAttEp = 1.0;
  const kEpAtt = Math.exp(-1.0 / (0.004 * sr));
  let modIdx = cfg.fmIndexBase;
  const kModDec = Math.exp(-1.0 / (cfg.fmIndexTauSec * sr));
  let tineIdx = cfg.fmTineIndex;
  const kTineDec = Math.exp(-1.0 / (cfg.fmTineTauSec * sr));
  let envEpRel = 1.0;
  const kEpRel = Math.exp(-1.0 / (0.18 * sr));

  let envPadAtt = 1.0;
  const kPadAtt = Math.exp(-1.0 / (cfg.padAttackSec * sr));
  let envPadRel = 1.0;
  const kPadRel = Math.exp(-1.0 / (cfg.padReleaseSec * sr));

  const epPhases = new Float64Array(nNotes);
  const epInc = freqs.map((f) => (TWO_PI * f) / sr);
  let tremPhase = 0.0;
  const tremInc = (TWO_PI * 3.2) / sr;

  for (let s = Math.max(0, startS); s < endS; s++) {
    const i = s - startS;

    let epSumL = 0.0;
    let epSumR = 0.0;
    if (epGain > 0.001) {
      if (i > durS) envEpRel *= kEpRel;
      const envEp = (1.0 - envAttEp) * envEpBase * envEpRel;
      envAttEp *= kEpAtt;
      envEpBase *= kEpDec;
      modIdx *= kModDec;
      tineIdx *= kTineDec;
      tremPhase += tremInc;
      const autoPan = 0.5 + 0.28 * Math.sin(tremPhase);

      let epRaw = 0.0;
      for (let n = 0; n < nNotes; n++) {
        const wc = epPhases[n];
        epPhases[n] = wc + epInc[n];
        const mod = modIdx * Math.sin(wc) + tineIdx * Math.sin(wc * cfg.fmTineRatio);
        epRaw += Math.sin(wc + mod) * (n === 0 ? 0.85 : 0.65);
      }
      const sigEp = epRaw * envEp * epGain * 0.068;
      epSumL = sigEp * (0.65 + 0.55 * autoPan);
      epSumR = sigEp * (1.20 - 0.55 * autoPan);
    }

    let padOutL = 0.0;
    let padOutR = 0.0;
    if (padGain > 0.001) {
      if (i > durS) envPadRel *= kPadRel;
      const envPad = (1.0 - envPadAtt) * envPadRel * padGain * 0.036;
      envPadAtt *= kPadAtt;
      let rawL = 0.0;
      let rawR = 0.0;
      for (let n = 0; n < nNotes; n++) {
        const fBase = freqs[n];
        for (let d = 0; d < 3; d++) {
          const idx = n * 3 + d;
          let p = padPhases[idx] + (fBase * detRatios[d]) / sr;
          if (p >= 1.0) p -= 1.0;
          padPhases[idx] = p;
          const saw = 2.0 * p - 1.0;
          if (d === 0) { rawL += saw * 1.15; rawR += saw * 0.35; }
          else if (d === 1) { rawL += saw * 0.70; rawR += saw * 0.70; }
          else { rawL += saw * 0.35; rawR += saw * 1.15; }
        }
      }
      padLpL += fPadCut * (rawL - padLpL);
      padLpR += fPadCut * (rawR - padLpR);
      padOutL = padLpL * envPad;
      padOutR = padLpR * envPad;
    }

    bufL[s] += epSumL + padOutL;
    bufR[s] += epSumR + padOutR;
  }
}

export function synthesizePureMathKeys(bandScore, instPhysics, sr, nSamples) {
  const keysL = new Float32Array(nSamples);
  const keysR = new Float32Array(nSamples);
  const barSec = 240.0 / bandScore.bpm;
  const halfBarS = Math.round(0.5 * barSec * sr);
  const bar0 = bandScore.bar0Sec;
  const cfg = instPhysics.keys;

  for (const sec of bandScore.sections) {
    if (sec.epGain <= 0.001 && sec.padGain <= 0.001) continue;
    for (let bOff = 0; bOff < sec.bars; bOff++) {
      const barIdx = sec.startBar + bOff;
      const tBar = bar0 + barIdx * barSec;

      if (sec.prog === 'FINAL') {
        if (bOff === 0) {
          const ch = bandScore.chords.D9end;
          renderRhodesAndPadChord(keysL, keysR, Math.round(tBar * sr), Math.round(4.0 * sr), ch.keys, sec.epGain, sec.padGain, cfg, sr);
        }
        continue;
      }

      const pIdx = ((bOff + (sec.progOffset || 0)) % 4);
      let [chName0, chName1] = bandScore.progressions[sec.prog][pIdx];
      if (sec.resolveLastDm && bOff === sec.bars - 1) {
        chName1 = 'Dm9';
      }
      const s0 = Math.round(tBar * sr);
      const s1 = Math.round((tBar + 0.5 * barSec) * sr);
      renderRhodesAndPadChord(keysL, keysR, s0, halfBarS, bandScore.chords[chName0].keys, sec.epGain, sec.padGain, cfg, sr);
      renderRhodesAndPadChord(keysL, keysR, s1, halfBarS, bandScore.chords[chName1].keys, sec.epGain, sec.padGain, cfg, sr);
    }
  }

  return { keys_rhodes_pad: [keysL, keysR] };
}

export function synthesizePureMathSpaceFdn(vocalBusL, vocalBusR, bandBusL, bandBusR, singerPhysics) {
  const sr = singerPhysics.sampleRate;
  const nSamples = vocalBusL.length;
  const outL = new Float32Array(nSamples);
  const outR = new Float32Array(nSamples);

  const er = singerPhysics.earlyReflections;
  const fdn = singerPhysics.fdnReverb;
  const wVoc = fdn.wetSendVocal;
  const wBand = fdn.wetSendBand;

  // Build pre-de-essed reverb send bus (gentle 1-pole low-pass at 4.8 kHz keeps space silky warm)
  const sendL = new Float32Array(nSamples);
  const sendR = new Float32Array(nSamples);
  const aSend = 1.0 - Math.exp((-TWO_PI * 4800.0) / sr);
  let lpSL = 0.0;
  let lpSR = 0.0;
  for (let s = 0; s < nSamples; s++) {
    const inL = vocalBusL[s] * wVoc + bandBusL[s] * wBand;
    const inR = vocalBusR[s] * wVoc + bandBusR[s] * wBand;
    lpSL += aSend * (inL - lpSL);
    lpSR += aSend * (inR - lpSR);
    sendL[s] = lpSL;
    sendR[s] = lpSR;
  }

  // 4-Tap Near-Field Stereo Early Reflections
  const erD = er.delaysSamples;
  const erG = er.gains;
  for (let t = 0; t < erD.length; t++) {
    const dL = erD[t];
    const dR = erD[t] + ((t & 1) ? 73 : -59);
    const g = erG[t] * 0.55;
    for (let s = Math.max(dL, dR); s < nSamples; s++) {
      outL[s] += sendL[s - dL] * g;
      outR[s] += sendR[s - dR] * g;
    }
  }

  // 8-Line Prime-Delay Householder Feedback Delay Network: H = I - (2/8)*1*1^T
  const delays = fdn.delaysSamples;
  const lines = delays.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const dampState = new Float32Array(8);
  const gainsLow = delays.map((d) => Math.pow(10.0, (-3.0 * d) / (fdn.rt60LowSec * sr)));
  const gainsRatio = delays.map((d, idx) => Math.pow(10.0, (-3.0 * d) / (fdn.rt60HighSec * sr)) / gainsLow[idx]);
  const dampAlpha = 1.0 - Math.exp((-TWO_PI * fdn.dampingCutoffHz) / sr);
  const readVec = new Float64Array(8);

  for (let s = 0; s < nSamples; s++) {
    let sum8 = 0.0;
    for (let k = 0; k < 8; k++) {
      const y = lines[k][ptrs[k]];
      readVec[k] = y;
      sum8 += y;
    }
    const householder = 0.25 * sum8;
    const inL = sendL[s] * 0.35;
    const inR = sendR[s] * 0.35;

    let wetL = 0.0;
    let wetR = 0.0;
    for (let k = 0; k < 8; k++) {
      const fb = readVec[k] - householder;
      dampState[k] += dampAlpha * (fb - dampState[k]);
      const hi = fb - dampState[k];
      const decayed = gainsLow[k] * (dampState[k] + gainsRatio[k] * hi);
      const excite = (k & 1) ? inR : inL;
      lines[k][ptrs[k]] = decayed + excite * ((k & 2) ? -1.0 : 1.0);
      ptrs[k]++;
      if (ptrs[k] >= delays[k]) ptrs[k] = 0;

      if (k & 1) wetR += readVec[k];
      else wetL += readVec[k];
    }
    outL[s] += wetL * 0.56;
    outR[s] += wetR * 0.56;
  }

  return { studio_fdn_space: [outL, outR] };
}
