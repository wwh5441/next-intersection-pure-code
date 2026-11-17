// V9R1 Pure-Code Dual-Dictionary 10-Track Renderer (`v9r1-renderer.js`)
// Zero LZMA / Zero Base122 / Zero Blind MDCT — 100% Executable Score + Dual-Dictionary + Physics ODEs
// Low-RAM Channel Layout: Mono stems (01, 02, 03, 08) = `[Float32Array]`; Stereo stems (04, 05, 06, 07, 09, 10) = `[L, R]`

import { SINGER_CORE } from '../../models/singer-core.js';
import { SYLLABLE_ATLAS } from '../../models/syllable-atlas.js';
import { INSTRUMENT_PHYSICS } from '../../models/instrument-physics.js';
import { BAND_MOTIF_ATLAS } from '../../models/band-motif-atlas.js';
import { SONG_STRUCTURE } from '../../data/song-structure.js';
import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';
import { V9R1_TIERS, DEFAULT_V9R1_TIER } from '../../data/tiers-registry.js';

import { expandBandArrangementToTimelines } from './motif-expander.js';
import { synthesizeV9r1Vocals } from './vocal-coart-synth.js';
import { synthesizeV9r1Drums } from './drums-modal-synth.js';
import { synthesizeV9r1Bass, synthesizeV9r1GuitarAndKeys } from './bass-guitar-keys-synth.js';
import { synthesizeV9r1SpaceFdn } from './space-fdn-synth.js';

export const STEM_MANIFEST = [
  { id: 'vocal_chest_f0',       group: 'vocal', channels: 1, num: '01', name: '主唱 · 声带基音与胸腔共鸣',       formula: '64-Harm LF + Chebyshev (<720 Hz)',       color: '#e85d3f' },
  { id: 'vocal_formant_body',   group: 'vocal', channels: 1, num: '02', name: '主唱 · 40阶声道与李宇春音色核',     formula: '40-ERB + 97-Syl `--` + 24/10-PCA',       color: '#d9483b' },
  { id: 'vocal_consonant_air',  group: 'vocal', channels: 1, num: '03', name: '主唱 · 最小相位辅音与开相气声',     formula: 'Hilbert MinPhase + Comb Valley HNR',    color: '#2a9d8f' },
  { id: 'vocal_chorus_harmony', group: 'vocal', channels: 2, num: '04', name: '和声 · 副歌双声道真三四度和声',     formula: 'True Interval Harmony + ±6.5c L/R',     color: '#6d597a' },
  { id: 'studio_fdn_space',     group: 'vocal', channels: 2, num: '05', name: '空间 · 近场早期反射与FDN混响',     formula: '22ms PreDelay + 8-Prime Householder',    color: '#457b9d' },
  { id: 'drum_kick_snare',      group: 'band',  channels: 2, num: '06', name: '鼓组 · 字典扫频底鼓与木鱼拍手',     formula: '18-Motif Atlas + 58Hz/164Hz Modal ODE',  color: '#c98a2c' },
  { id: 'drum_hihat_perc',      group: 'band',  channels: 2, num: '07', name: '鼓组 · 64原型高频金属镲与沙锤',     formula: '64-Proto Mel-OLA (>3.3 kHz) + Shimmer',  color: '#b07d2b' },
  { id: 'bass_groove',          group: 'band',  channels: 1, num: '08', name: '贝斯 · 22原型弦振与真滑音律动',     formula: '22-Motif Atlas + 18-Harm Stiff String',  color: '#3a7d44' },
  { id: 'guitar_funk_riff',     group: 'band',  channels: 2, num: '09', name: '吉他 · 22原型前奏Hook复调与扫弦',   formula: '22-Motif Atlas + Rotor Hook + 80fps',    color: '#2b6cb0' },
  { id: 'keys_rhodes_pad',      group: 'band',  channels: 2, num: '10', name: '键盘 · 字典低中音电钢和弦声场',     formula: 'Coupled-Rotor (E2..C#4) + Stereo FDN',  color: '#805ad5' },
];

function computeBusRms(chListArray, nSamples) {
  let sum = 0.0;
  let count = 0;
  for (let s = 0; s < nSamples; s += 4) {
    let sumL = 0.0;
    let sumR = 0.0;
    for (let i = 0; i < chListArray.length; i++) {
      const chs = chListArray[i];
      if (chs.length === 1) {
        const v = chs[0][s];
        sumL += v;
        sumR += v;
      } else {
        sumL += chs[0][s];
        sumR += chs[1][s];
      }
    }
    sum += 0.5 * (sumL * sumL + sumR * sumR);
    count++;
  }
  return Math.sqrt(sum / Math.max(1, count));
}

function scaleStemChannelsInPlace(chs, gain) {
  for (let c = 0; c < chs.length; c++) {
    const arr = chs[c];
    for (let i = 0; i < arr.length; i++) arr[i] *= gain;
  }
}

export function renderV9r1Song(tierIdInput = DEFAULT_V9R1_TIER, onProgress = null) {
  const tierKeys = Object.keys(V9R1_TIERS);
  const tierId = V9R1_TIERS[tierIdInput] ? tierIdInput : tierKeys[0];
  const tier = V9R1_TIERS[tierId];
  const { vocalProfile, vocalScore, bandProfile, bandArrangement } = tier;

  const t0 = performance.now();
  const sr = SINGER_CORE.sampleRate;
  const nSamples = SINGER_CORE.totalSamples;

  if (onProgress) {
    onProgress(0.08, `正在展开 97 音节字典与 [${vocalProfile.shortLabel}] ${vocalProfile.pcaDims} 维声道微表情...`);
  }
  const vocals = synthesizeV9r1Vocals(SINGER_CORE, SYLLABLE_ATLAS, vocalScore, vocalProfile);

  // Calibrate 4-stem dry vocal bus to upfront studio presence (`0.114` RMS)
  const dryVocalStems = [
    vocals.vocal_chest_f0,
    vocals.vocal_formant_body,
    vocals.vocal_consonant_air,
    vocals.vocal_chorus_harmony,
  ];
  const curVocRms = computeBusRms(dryVocalStems, nSamples);
  if (curVocRms > 1e-6) {
    const vGain = 0.114 / curVocRms;
    for (const chs of dryVocalStems) scaleStemChannelsInPlace(chs, vGain);
    for (let s = 0; s < nSamples; s++) vocals.vocalSendMono[s] *= vGain;
  }

  if (onProgress) {
    onProgress(0.42, `正在从 86 小节伴奏字典展开 ${bandProfile.drumMotifs}+${bandProfile.bassMotifs}+${bandProfile.harmonyMotifs} 原型与稀疏残差...`);
  }
  const expandedBand = expandBandArrangementToTimelines(
    BAND_MOTIF_ATLAS,
    bandArrangement,
    INSTRUMENT_PHYSICS.harmony.nNotes
  );

  if (onProgress) {
    onProgress(0.50, '正在积分 100fps 扫频底鼓、164Hz 木鱼拍手与 64 原型金属镲瞬态...');
  }
  const drums = synthesizeV9r1Drums(expandedBand.drums, INSTRUMENT_PHYSICS, nSamples);

  if (onProgress) {
    onProgress(0.66, '正在求解 18 阶非谐波色散电贝斯弦振与真滑音曲线...');
  }
  const bass = synthesizeV9r1Bass(expandedBand.bass, INSTRUMENT_PHYSICS, nSamples);

  if (onProgress) {
    onProgress(0.79, '正在合成 56 音符耦合旋子前奏吉他 Hook、80fps 扫弦与温暖电钢声场...');
  }
  const harm = synthesizeV9r1GuitarAndKeys(expandedBand.harmony, INSTRUMENT_PHYSICS, sr, nSamples);

  // Calibrate 5-stem band bus to `0.130` RMS preserving authentic relative stem balance
  const bandStems = [
    drums.drum_kick_snare,
    drums.drum_hihat_perc,
    bass.bass_groove,
    harm.guitar_funk_riff,
    harm.keys_rhodes_pad,
  ];
  const rawBandRms = computeBusRms(bandStems, nSamples);
  if (rawBandRms > 1e-6) {
    const bGain = 0.130 / rawBandRms;
    for (const chs of bandStems) scaleStemChannelsInPlace(chs, bGain);
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

  if (onProgress) {
    onProgress(0.92, '正在驱动 22ms 预延迟近场反射与 8 线素数 Householder FDN 空间...');
  }
  const space = synthesizeV9r1SpaceFdn(vocals.vocalSendMono, bandSendL, bandSendR, SINGER_CORE);

  const stems = {
    vocal_chest_f0: vocals.vocal_chest_f0,
    vocal_formant_body: vocals.vocal_formant_body,
    vocal_consonant_air: vocals.vocal_consonant_air,
    vocal_chorus_harmony: vocals.vocal_chorus_harmony,
    studio_fdn_space: space.studio_fdn_space,
    drum_kick_snare: drums.drum_kick_snare,
    drum_hihat_perc: drums.drum_hihat_perc,
    bass_groove: bass.bass_groove,
    guitar_funk_riff: harm.guitar_funk_riff,
    keys_rhodes_pad: harm.keys_rhodes_pad,
  };

  // Zero-overshoot continuous hop-local studio bus limiter (`0.94`) preserving 100% 10-track linear additivity
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
        const chs = stemList[t];
        if (chs.length === 1) {
          const v = chs[0][s];
          sumL += v;
          sumR += v;
        } else {
          sumL += chs[0][s];
          sumR += chs[1][s];
        }
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
    const g0 = vGain[b], g1 = vGain[b + 1];
    if (g0 >= 0.9999 && g1 >= 0.9999) continue;
    const s0 = b * hop;
    const s1 = Math.min(nSamples, s0 + hop);
    const inv = 1.0 / Math.max(1, s1 - s0);
    for (let s = s0; s < s1; s++) {
      const g = g0 + (g1 - g0) * ((s - s0) * inv);
      for (let t = 0; t < stemList.length; t++) {
        const chs = stemList[t];
        for (let c = 0; c < chs.length; c++) chs[c][s] *= g;
      }
    }
  }

  for (let s = 0; s < nSamples; s++) {
    let sumL = 0.0, sumR = 0.0;
    for (let t = 0; t < stemList.length; t++) {
      const chs = stemList[t];
      if (chs.length === 1) {
        const v = chs[0][s];
        sumL += v;
        sumR += v;
      } else {
        sumL += chs[0][s];
        sumR += chs[1][s];
      }
    }
    const aL = Math.abs(sumL), aR = Math.abs(sumR);
    if (aL > finalPeak) finalPeak = aL;
    if (aR > finalPeak) finalPeak = aR;
  }

  if (onProgress) onProgress(1.0, `10 轨双字典纯代码合成完成 (${vocalProfile.shortLabel})！`);
  const elapsedMs = Math.round(performance.now() - t0);

  return {
    tierId,
    vocalProfile,
    bandProfile,
    sampleRate: sr,
    numSamples: nSamples,
    durationSec: SINGER_CORE.durationSec,
    bpm: SONG_STRUCTURE.bpm,
    bar0Sec: SONG_STRUCTURE.bar0Sec,
    renderTimeMs: elapsedMs,
    peakAmplitude: Number(finalPeak.toFixed(4)),
    stems,
    vocalScore,
    bandArrangement,
    lyricLines: VOCAL_LYRIC_LINES,
    songStructure: SONG_STRUCTURE,
  };
}
