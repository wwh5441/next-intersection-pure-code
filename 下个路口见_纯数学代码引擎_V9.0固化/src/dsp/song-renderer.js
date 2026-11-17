// Pure-Code 10-Track Studio Song Renderer (5 Vocal Math Stems + 5 Band Math Stems)
// Zero Binary / Zero LZMA / Zero Base122 — 100% Mathematical Synthesis

import { SINGER_PHYSICS } from '../../models/singer-physics.js';
import { MANDARIN_CONSONANTS, MANDARIN_SYLLABLES } from '../../models/mandarin-phonetics.js';
import { INSTRUMENT_PHYSICS } from '../../models/instrument-physics.js';
import { BAND_SCORE } from '../../data/band-score.js';
import { VOCAL_LYRIC_LINES, VOCAL_SCORE } from '../../data/vocal-score.js';

import { synthesizePureMathVocals } from './math-vocal-synth.js';
import { synthesizePureMathDrums } from './math-drums-synth.js';
import { synthesizePureMathBassAndGuitar } from './math-bass-guitar-synth.js';
import { synthesizePureMathKeys, synthesizePureMathSpaceFdn } from './math-keys-space-synth.js';

export const STEM_MANIFEST = [
  { id: 'vocal_chest_f0',       group: 'vocal', num: '01', name: '人声 · 基音与胸腔共鸣',   formula: 'Fant LF H1–H3 (<720 Hz)',       color: '#e85d3f' },
  { id: 'vocal_formant_body',   group: 'vocal', num: '02', name: '人声 · 声韵母二阶连音核', formula: '12-ERB Fourier + 2nd-Order ODE', color: '#d9483b' },
  { id: 'vocal_consonant_air',  group: 'vocal', num: '03', name: '人声 · 辅音瞬态与真气声', formula: 'Dual-Gaussian VOT + Klatt Air',  color: '#2a9d8f' },
  { id: 'vocal_chorus_harmony', group: 'vocal', num: '04', name: '人声 · 副歌双声道真和声', formula: '±6.5c Stereo Detune Double',     color: '#6d597a' },
  { id: 'studio_fdn_space',     group: 'vocal', num: '05', name: '空间 · 早期反射与FDN混响', formula: '4-Tap ER + 8-Prime Householder', color: '#457b9d' },
  { id: 'drum_kick_snare',      group: 'band',  num: '06', name: '鼓组 · 膜振底鼓与军鼓拍手', formula: 'Pitch-ODE Kick + Bessel Snare', color: '#c98a2c' },
  { id: 'drum_hihat_perc',      group: 'band',  num: '07', name: '鼓组 · 非公度金属镲与沙锤', formula: '6-Square Ring Matrix + Choke',  color: '#b07d2b' },
  { id: 'bass_groove',          group: 'band',  num: '08', name: '贝斯 · 指弹电贝斯包络滤波', formula: 'Pickup Comb + Chamberlin SVF',   color: '#3a7d44' },
  { id: 'guitar_funk_riff',     group: 'band',  num: '09', name: '吉他 · 波导拨弦放克切分',   formula: 'Extended Karplus-Strong 4-Str', color: '#2b6cb0' },
  { id: 'keys_rhodes_pad',      group: 'band',  num: '10', name: '键盘 · FM电钢与合唱铺底',   formula: '2-Op DX FM + 3x Saw Chorus',     color: '#805ad5' },
];

function computeRmsStereo(chL, chR) {
  let sum = 0.0;
  const n = chL.length;
  for (let i = 0; i < n; i += 4) {
    sum += 0.5 * (chL[i] * chL[i] + chR[i] * chR[i]);
  }
  return Math.sqrt(sum / Math.max(1, Math.ceil(n / 4)));
}

function scaleStereoInPlace(pair, gain) {
  const [L, R] = pair;
  for (let i = 0; i < L.length; i++) {
    L[i] *= gain;
    R[i] *= gain;
  }
}

export function renderFullPureMathSong(onProgress = null) {
  const t0 = performance.now();
  const sr = SINGER_PHYSICS.sampleRate;
  const nSamples = SINGER_PHYSICS.totalSamples;

  if (onProgress) onProgress(0.08, '正在求解人声 12 模声道与二阶临界阻尼跨字连音微分方程...');
  const vocals = synthesizePureMathVocals(
    SINGER_PHYSICS,
    { MANDARIN_CONSONANTS, MANDARIN_SYLLABLES },
    VOCAL_SCORE
  );

  if (onProgress) onProgress(0.38, '正在积分 98.0 BPM 物理膜振底鼓、军鼓拍手与非公度金属镲片...');
  const drums = synthesizePureMathDrums(BAND_SCORE, INSTRUMENT_PHYSICS, sr, nSamples);

  if (onProgress) onProgress(0.60, '正在计算 Chamberlin SVF 指弹贝斯与 Karplus-Strong 波导放克吉他...');
  const bassGtr = synthesizePureMathBassAndGuitar(
    BAND_SCORE,
    INSTRUMENT_PHYSICS,
    drums.kickSidechainEnv,
    sr,
    nSamples
  );

  if (onProgress) onProgress(0.78, '正在合成双算子 FM 电钢琴与副歌立体声合唱铺底...');
  const keys = synthesizePureMathKeys(BAND_SCORE, INSTRUMENT_PHYSICS, sr, nSamples);

  // Calibrate instrumental studio mix levels to match 《下个, 路口, 见》 reference stem balance
  // Reference studio target: Vocals ~0.092, Kick+Snare ~0.088, Hat+Perc ~0.032, Bass ~0.068, Guitar ~0.056, Keys ~0.044
  const bandTargets = [
    [drums.drum_kick_snare, 0.086],
    [drums.drum_hihat_perc, 0.030],
    [bassGtr.bass_groove, 0.068],
    [bassGtr.guitar_funk_riff, 0.055],
    [keys.keys_rhodes_pad, 0.042],
  ];
  for (const [pair, tgtRms] of bandTargets) {
    const cur = computeRmsStereo(pair[0], pair[1]);
    if (cur > 1e-6) {
      scaleStereoInPlace(pair, tgtRms / cur);
    }
  }

  // Calibrate dry vocal bus so lead vocals sit clearly upfront above the band (~0.092 RMS)
  const vocDryL = new Float32Array(nSamples);
  const vocDryR = new Float32Array(nSamples);
  for (const k of ['vocal_chest_f0', 'vocal_formant_body', 'vocal_consonant_air', 'vocal_chorus_harmony']) {
    const [L, R] = vocals[k];
    for (let s = 0; s < nSamples; s++) {
      vocDryL[s] += L[s];
      vocDryR[s] += R[s];
    }
  }
  const curVocRms = computeRmsStereo(vocDryL, vocDryR);
  if (curVocRms > 1e-6) {
    const vGain = 0.094 / curVocRms;
    for (const k of ['vocal_chest_f0', 'vocal_formant_body', 'vocal_consonant_air', 'vocal_chorus_harmony']) {
      scaleStereoInPlace(vocals[k], vGain);
    }
    for (let s = 0; s < nSamples; s++) {
      vocDryL[s] *= vGain;
      vocDryR[s] *= vGain;
    }
  }

  if (onProgress) onProgress(0.90, '正在驱动 8 线素数延迟 Householder FDN 录音室空间混响...');
  const bandSendL = new Float32Array(nSamples);
  const bandSendR = new Float32Array(nSamples);
  const [gL, gR] = bassGtr.guitar_funk_riff;
  const [kL, kR] = keys.keys_rhodes_pad;
  const [sL, sR] = drums.drum_kick_snare;
  for (let s = 0; s < nSamples; s++) {
    bandSendL[s] = gL[s] * 0.75 + kL[s] * 0.85 + sL[s] * 0.28;
    bandSendR[s] = gR[s] * 0.75 + kR[s] * 0.85 + sR[s] * 0.28;
  }
  const space = synthesizePureMathSpaceFdn(vocDryL, vocDryR, bandSendL, bandSendR, SINGER_PHYSICS);

  const stems = {
    vocal_chest_f0: vocals.vocal_chest_f0,
    vocal_formant_body: vocals.vocal_formant_body,
    vocal_consonant_air: vocals.vocal_consonant_air,
    vocal_chorus_harmony: vocals.vocal_chorus_harmony,
    studio_fdn_space: space.studio_fdn_space,
    drum_kick_snare: drums.drum_kick_snare,
    drum_hihat_perc: drums.drum_hihat_perc,
    bass_groove: bassGtr.bass_groove,
    guitar_funk_riff: bassGtr.guitar_funk_riff,
    keys_rhodes_pad: keys.keys_rhodes_pad,
  };

  // Hop-local (5.3ms) zero-overshoot studio bus limiter at 0.94 preserving 100% 10-track linear additivity!
  const hop = 256;
  const nBlocks = Math.ceil(nSamples / hop);
  const gBlock = new Float32Array(nBlocks).fill(1.0);
  const stemList = STEM_MANIFEST.map((m) => stems[m.id]);

  for (let b = 0; b < nBlocks; b++) {
    const s0 = b * hop;
    const s1 = Math.min(nSamples, s0 + hop);
    let pk = 0.0;
    for (let s = s0; s < s1; s++) {
      let sumL = 0.0;
      let sumR = 0.0;
      for (let t = 0; t < stemList.length; t++) {
        sumL += stemList[t][0][s];
        sumR += stemList[t][1][s];
      }
      const aL = Math.abs(sumL);
      const aR = Math.abs(sumR);
      if (aL > pk) pk = aL;
      if (aR > pk) pk = aR;
    }
    if (pk > 0.94) {
      gBlock[b] = 0.94 / pk;
    }
  }
  // 1-block lookahead min + release smoothing prevents clicks while guaranteeing <= 0.9400 peak
  const gSmooth = new Float32Array(nBlocks);
  let relState = 1.0;
  for (let b = 0; b < nBlocks; b++) {
    const target = Math.min(
      gBlock[b],
      b > 0 ? gBlock[b - 1] : 1.0,
      b + 1 < nBlocks ? gBlock[b + 1] : 1.0
    );
    relState = target < relState ? target : (0.85 * relState + 0.15 * target);
    gSmooth[b] = Math.min(gBlock[b], relState);
  }

  const vGain = new Float32Array(nBlocks + 1);
  for (let b = 0; b <= nBlocks; b++) {
    const left = b > 0 ? gSmooth[b - 1] : 1.0;
    const right = b < nBlocks ? gSmooth[b] : left;
    vGain[b] = Math.min(left, right);
  }

  let finalPeak = 0.0;
  for (let b = 0; b < nBlocks; b++) {
    const g0 = vGain[b];
    const g1 = vGain[b + 1];
    if (g0 >= 0.9999 && g1 >= 0.9999) continue;
    const s0 = b * hop;
    const s1 = Math.min(nSamples, s0 + hop);
    const inv = 1.0 / Math.max(1, s1 - s0);
    for (let s = s0; s < s1; s++) {
      const g = g0 + (g1 - g0) * ((s - s0) * inv);
      for (let t = 0; t < stemList.length; t++) {
        stemList[t][0][s] *= g;
        stemList[t][1][s] *= g;
      }
    }
  }

  for (let s = 0; s < nSamples; s++) {
    let sumL = 0.0;
    let sumR = 0.0;
    for (let t = 0; t < stemList.length; t++) {
      sumL += stemList[t][0][s];
      sumR += stemList[t][1][s];
    }
    const aL = Math.abs(sumL);
    const aR = Math.abs(sumR);
    if (aL > finalPeak) finalPeak = aL;
    if (aR > finalPeak) finalPeak = aR;
  }

  if (onProgress) onProgress(1.0, '10 轨纯数学全曲合成完成！');
  const elapsedMs = Math.round(performance.now() - t0);

  return {
    sampleRate: sr,
    numSamples: nSamples,
    durationSec: SINGER_PHYSICS.durationSec,
    bpm: BAND_SCORE.bpm,
    bar0Sec: BAND_SCORE.bar0Sec,
    renderTimeMs: elapsedMs,
    peakAmplitude: Number(finalPeak.toFixed(4)),
    stems,
    vocalScore: VOCAL_SCORE,
    lyricLines: VOCAL_LYRIC_LINES,
    bandScore: BAND_SCORE,
  };
}
