// Pure-Code 10-Track Dual-Tier Studio Song Renderer (V9.2)
// Supports Both Version A (`ultra`) & Version B (`expressive`) for BOTH Vocals & Real-Detail 5-Track Band
// Zero Binary / Zero LZMA / Zero Base122 / Zero Waveforms — 100% Mathematical Synthesis

import { SINGER_PHYSICS } from '../../models/singer-physics.js';
import { MANDARIN_SYLLABLES } from '../../models/mandarin-phonetics.js';
import { INSTRUMENT_PHYSICS } from '../../models/instrument-physics.js';
import { BAND_SCORE } from '../../data/band-score.js';
import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';
import { VOCAL_TIERS, DEFAULT_VOCAL_TIER } from '../../data/vocal-tiers.js';

import { synthesizePureMathVocals } from './math-vocal-synth.js';
import { synthesizePureMathDrums } from './math-drums-synth.js';
import { synthesizePureMathBass, synthesizePureMathGuitarAndKeys } from './math-bass-guitar-synth.js';
import { synthesizePureMathSpaceFdn } from './math-keys-space-synth.js';

export const STEM_MANIFEST = [
  { id: 'vocal_chest_f0',       group: 'vocal', num: '01', name: '主唱 · 声带基音与胸腔共鸣',       formula: '64-Harm LF + Chebyshev (<720 Hz)',       color: '#e85d3f' },
  { id: 'vocal_formant_body',   group: 'vocal', num: '02', name: '主唱 · 40阶声道与李宇春音色核',     formula: '40-ERB Modes + 24/10-PCA EigenCore',     color: '#d9483b' },
  { id: 'vocal_consonant_air',  group: 'vocal', num: '03', name: '主唱 · 最小相位辅音与开相气声',     formula: 'Hilbert MinPhase + Comb Valley HNR',    color: '#2a9d8f' },
  { id: 'vocal_chorus_harmony', group: 'vocal', num: '04', name: '和声 · 副歌双声道真三四度和声',     formula: 'True Interval Harmony + ±6.5c L/R',     color: '#6d597a' },
  { id: 'studio_fdn_space',     group: 'vocal', num: '05', name: '空间 · 近场早期反射与FDN混响',     formula: '22ms PreDelay + 8-Prime Householder',    color: '#457b9d' },
  { id: 'drum_kick_snare',      group: 'band',  num: '06', name: '鼓组 · 原曲扫频底鼓与木鱼军鼓拍手', formula: '100fps Swept Kick + 164Hz Woodblock',    color: '#c98a2c' },
  { id: 'drum_hihat_perc',      group: 'band',  num: '07', name: '鼓组 · 64原型高频金属镲与沙锤瞬态', formula: '64-Proto Mel-OLA (>3.3 kHz) + Shimmer',  color: '#b07d2b' },
  { id: 'bass_groove',          group: 'band',  num: '08', name: '贝斯 · 18阶非谐波弦振与真滑音律动', formula: '18-Harm String (B=1.8e-4) + 32-Codebook', color: '#3a7d44' },
  { id: 'guitar_funk_riff',     group: 'band',  num: '09', name: '吉他 · 原曲前奏Hook复调与80fps扫弦', formula: 'Coupled-Rotor (D4..B6) + 80fps Strum',   color: '#2b6cb0' },
  { id: 'keys_rhodes_pad',      group: 'band',  num: '10', name: '键盘 · 原曲低中音电钢和弦与立体声场', formula: 'Coupled-Rotor (E2..C#4) + Stereo FDN',  color: '#805ad5' },
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

function synthesizeRealDetailBandForTier(bandTierScore, bandProfile, sr, nSamples, onProgress) {
  if (onProgress) {
    onProgress(0.46, `正在积分 100fps 原曲扫频底鼓、木鱼军鼓拍手与 64 原型金属镲瞬态...`);
  }
  const drums = synthesizePureMathDrums(bandTierScore, INSTRUMENT_PHYSICS, nSamples);

  if (onProgress) {
    onProgress(0.64, `正在求解 18 阶非谐波弦振电贝斯与 ${bandProfile.bassPitchPrecisionCents} 音分真滑音曲线...`);
  }
  const bass = synthesizePureMathBass(bandTierScore, INSTRUMENT_PHYSICS, nSamples);

  if (onProgress) {
    onProgress(0.78, `正在合成 56 音符耦合旋子吉他前奏 Hook、80fps 拨片扫弦与温暖电钢声场...`);
  }
  const harm = synthesizePureMathGuitarAndKeys(bandTierScore, INSTRUMENT_PHYSICS, sr, nSamples);

  // Preserve 100% authentic relative stem balance via a single unified master band bus calibration (0.130 RMS)
  const bandBusL = new Float32Array(nSamples);
  const bandBusR = new Float32Array(nSamples);
  const bandPairs = [
    drums.drum_kick_snare,
    drums.drum_hihat_perc,
    bass.bass_groove,
    harm.guitar_funk_riff,
    harm.keys_rhodes_pad,
  ];
  for (const [L, R] of bandPairs) {
    for (let s = 0; s < nSamples; s++) {
      bandBusL[s] += L[s];
      bandBusR[s] += R[s];
    }
  }
  const rawBandRms = computeRmsStereo(bandBusL, bandBusR);
  if (rawBandRms > 1e-6) {
    const busGain = 0.130 / rawBandRms;
    for (const pair of bandPairs) {
      scaleStereoInPlace(pair, busGain);
    }
  }

  const bandSendL = new Float32Array(nSamples);
  const bandSendR = new Float32Array(nSamples);
  const [gL, gR] = harm.guitar_funk_riff;
  const [kL, kR] = harm.keys_rhodes_pad;
  const [sL, sR] = drums.drum_kick_snare;
  for (let s = 0; s < nSamples; s++) {
    bandSendL[s] = gL[s] * 0.45 + kL[s] * 0.50 + sL[s] * 0.16;
    bandSendR[s] = gR[s] * 0.45 + kR[s] * 0.50 + sR[s] * 0.16;
  }

  return {
    drum_kick_snare: drums.drum_kick_snare,
    drum_hihat_perc: drums.drum_hihat_perc,
    bass_groove: bass.bass_groove,
    guitar_funk_riff: harm.guitar_funk_riff,
    keys_rhodes_pad: harm.keys_rhodes_pad,
    bandSendL,
    bandSendR,
  };
}

export function renderFullPureMathSong(tierOrProgress = DEFAULT_VOCAL_TIER, maybeProgress = null) {
  let requestedTier = DEFAULT_VOCAL_TIER;
  let onProgress = maybeProgress;
  if (typeof tierOrProgress === 'function') {
    onProgress = tierOrProgress;
  } else if (typeof tierOrProgress === 'string') {
    requestedTier = tierOrProgress;
  }

  const tierKeys = Object.keys(VOCAL_TIERS);
  const tierId = VOCAL_TIERS[requestedTier] ? requestedTier : tierKeys[0];
  const tierEntry = VOCAL_TIERS[tierId];
  const vocalProfile = tierEntry.profile;
  const vocalScore = tierEntry.score;
  const bandProfile = tierEntry.bandProfile;
  const bandTierScore = tierEntry.bandScore;

  const t0 = performance.now();
  const sr = SINGER_PHYSICS.sampleRate;
  const nSamples = SINGER_PHYSICS.totalSamples;

  if (onProgress) {
    onProgress(
      0.08,
      `正在求解 [${vocalProfile.shortLabel}] 40 维声道与 ${vocalProfile.pcaDims} 维李宇春微表情曲线...`
    );
  }
  const vocals = synthesizePureMathVocals(SINGER_PHYSICS, MANDARIN_SYLLABLES, vocalScore, vocalProfile);

  // Calibrate 4-stem dry vocal bus to upfront studio presence (0.114 RMS)
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
    const vGain = 0.114 / curVocRms;
    for (const k of ['vocal_chest_f0', 'vocal_formant_body', 'vocal_consonant_air', 'vocal_chorus_harmony']) {
      scaleStereoInPlace(vocals[k], vGain);
    }
    for (let s = 0; s < nSamples; s++) {
      vocals.vocalSendMono[s] *= vGain;
    }
  }

  const band = synthesizeRealDetailBandForTier(bandTierScore, bandProfile, sr, nSamples, onProgress);

  if (onProgress) onProgress(0.91, '正在驱动 22ms 预延迟近场反射与 8 线素数 Householder FDN 空间...');
  const space = synthesizePureMathSpaceFdn(vocals.vocalSendMono, band.bandSendL, band.bandSendR, SINGER_PHYSICS);

  const stems = {
    vocal_chest_f0: vocals.vocal_chest_f0,
    vocal_formant_body: vocals.vocal_formant_body,
    vocal_consonant_air: vocals.vocal_consonant_air,
    vocal_chorus_harmony: vocals.vocal_chorus_harmony,
    studio_fdn_space: space.studio_fdn_space,
    drum_kick_snare: band.drum_kick_snare,
    drum_hihat_perc: band.drum_hihat_perc,
    bass_groove: band.bass_groove,
    guitar_funk_riff: band.guitar_funk_riff,
    keys_rhodes_pad: band.keys_rhodes_pad,
  };

  // Continuous zero-overshoot hop-local studio bus limiter at 0.94 preserving 100% 10-track linear additivity
  const hop = 256;
  const nBlocks = Math.ceil(nSamples / hop);
  const gBlock = new Float32Array(nBlocks).fill(1.0);
  const stemList = STEM_MANIFEST.map((m) => stems[m.id]);

  for (let b = 0; b < nBlocks; b++) {
    const s0 = b * hop;
    const s1 = Math.min(nSamples, s0 + hop);
    let pk = 0.0;
    for (let s = s0; s < s1; s++) {
      let sumL = 0.0, sumR = 0.0;
      for (let t = 0; t < stemList.length; t++) {
        sumL += stemList[t][0][s];
        sumR += stemList[t][1][s];
      }
      const aL = Math.abs(sumL), aR = Math.abs(sumR);
      if (aL > pk) pk = aL;
      if (aR > pk) pk = aR;
    }
    if (pk > 0.94) gBlock[b] = 0.94 / pk;
  }

  const gSmooth = new Float32Array(nBlocks);
  let relState = 1.0;
  for (let b = 0; b < nBlocks; b++) {
    const target = Math.min(gBlock[b], b > 0 ? gBlock[b - 1] : 1.0, b + 1 < nBlocks ? gBlock[b + 1] : 1.0);
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
    let sumL = 0.0, sumR = 0.0;
    for (let t = 0; t < stemList.length; t++) {
      sumL += stemList[t][0][s];
      sumR += stemList[t][1][s];
    }
    const aL = Math.abs(sumL), aR = Math.abs(sumR);
    if (aL > finalPeak) finalPeak = aL;
    if (aR > finalPeak) finalPeak = aR;
  }

  if (onProgress) onProgress(1.0, `10 轨纯数学合成完成 (${vocalProfile.shortLabel})！`);
  const elapsedMs = Math.round(performance.now() - t0);

  return {
    tierId,
    vocalProfile,
    bandProfile,
    sampleRate: sr,
    numSamples: nSamples,
    durationSec: SINGER_PHYSICS.durationSec,
    bpm: BAND_SCORE.bpm,
    bar0Sec: BAND_SCORE.bar0Sec,
    renderTimeMs: elapsedMs,
    peakAmplitude: Number(finalPeak.toFixed(4)),
    stems,
    vocalScore,
    lyricLines: VOCAL_LYRIC_LINES,
    bandScore: BAND_SCORE,
  };
}
