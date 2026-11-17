// V9R3 10-Track 48kHz Pure `Math + Code + Dictionary` Synthesizer Orchestrator (`src/dsp/engine-orchestrator.js`)
// Shared identically by the Browser Web Worker (`synth-worker.js`) and Node.js Ground-Truth Audit (`verify_v9r3_hifi_audit.py`).

import { VOCAL_WAVECYCLE_DICTIONARY } from '../../models/vocal-wavecycle-dictionary.js';
import { BACKING_HARMONY_DICTIONARY, STUDIO_FDN_PHYSICS } from '../../models/backing-space-dictionary.js';
import {
  CYMBAL_SHIMMER_DICTIONARY,
  DRUM_GROOVE_BAR_MOTIFS,
  DRUM_HIT_ATOM_DICTIONARY,
} from '../../models/drum-percussion-dictionary.js';
import { BASS_ATOM_DICTIONARY, BASS_GROOVE_BAR_MOTIFS } from '../../models/bass-groove-dictionary.js';
import { GUITAR_PLUCK_DICTIONARY, RHODES_CHORD_DICTIONARY } from '../../models/guitar-rhodes-dictionary.js';

import { VOCAL_SCORE_MASTER } from '../../data/vocal-score-master.js';
import { VOCAL_SCORE_COMPACT } from '../../data/vocal-score-compact.js';
import { BAND_SCORE_MASTER } from '../../data/band-score-master.js';
import { BAND_SCORE_COMPACT } from '../../data/band-score-compact.js';

import { synthesizeVocalThreeTracks } from './vocal-wavecycle-synth.js';
import { synthesizeBarMotifPlusFill, synthesizeHiHatPercTrack } from './drum-percussion-synth.js';
import {
  synthesizeBassGrooveTrack,
  synthesizeGuitarRiffTrack,
  synthesizeRhodesPadTrack,
} from './bass-guitar-rhodes-synth.js';
import { synthesizeBackingHarmonyTrack, synthesizeStudioFdnSpaceTrack } from './backing-fdn-synth.js';

export const V9R3_STEM_SPECS = [
  { id: 'vocal_lead_chest', index: '01', name: '李宇春主唱 · 胸腔基频与暖中低频 (<720Hz)', group: 'vocal', channels: 1, color: '#d9483b' },
  { id: 'vocal_lead_formant', index: '02', name: '李宇春主唱 · 共振峰与咬字母音 (>=720Hz)', group: 'vocal', channels: 1, color: '#e66a4e' },
  { id: 'vocal_breath_air', index: '03', name: '李宇春主唱 · 清辅音声母与呼吸齿音气声', group: 'vocal', channels: 1, color: '#f0916d' },
  { id: 'vocal_chorus_harmony', index: '04', name: '导歌/副歌立体声双轨和声 (Chorus Harmony)', group: 'harm', channels: 2, color: '#7c3aed' },
  { id: 'studio_fdn_space', index: '05', name: '8阶质数Householder FDN录音室空间混响', group: 'harm', channels: 2, color: '#9f67ff' },
  { id: 'drum_kick_snare', index: '06', name: '98.0000 BPM 底鼓/军鼓/木鱼律动原型+加花', group: 'drum', channels: 2, color: '#c98a2c' },
  { id: 'drum_hihat_perc', index: '07', name: '98.0000 BPM 真擦片6频段踩镲/沙锤/吊镲空气层', group: 'drum', channels: 2, color: '#dd9f3c' },
  { id: 'bass_groove', index: '08', name: '98.0000 BPM 电贝斯24小节原型+滑音残差', group: 'bass', channels: 2, color: '#2f855a' },
  { id: 'guitar_funk_riff', index: '09', name: '清音木吉他/Funk切音拨弦词典轨 (>=480Hz)', group: 'band', channels: 2, color: '#2563eb' },
  { id: 'keys_rhodes_pad', index: '10', name: 'Rhodes暖电钢和声铺底词典轨 (<480Hz)', group: 'band', channels: 2, color: '#3b82f6' },
];

export function getScoresForTier(tierId) {
  if (tierId === 'compact_pure') {
    return { vocalScore: VOCAL_SCORE_COMPACT, bandScore: BAND_SCORE_COMPACT };
  }
  return { vocalScore: VOCAL_SCORE_MASTER, bandScore: BAND_SCORE_MASTER };
}

export function synthesizeAllTenTracks(tierId = 'studio_master', options = {}, onProgress = null) {
  const t0 = performance.now();
  const { vocalScore, bandScore } = getScoresForTier(tierId);
  const totalSamples48 = vocalScore.totalSamples48;
  const pitchSemitones = Number(options.pitchSemitones || 0.0);

  if (onProgress) onProgress(0.08, '正在合成主唱 GCI 基音周期字典与清辅音小波 (01-03)...');
  const { chest48, formant48, air48 } = synthesizeVocalThreeTracks(
    VOCAL_WAVECYCLE_DICTIONARY,
    vocalScore,
    { pitchSemitones }
  );

  if (onProgress) onProgress(0.32, '正在合成 98.0000 BPM 底军鼓原型与 6 频段真擦片踩镲 (06-07)...');
  const [kickSnareL, kickSnareR] = synthesizeBarMotifPlusFill(
    DRUM_HIT_ATOM_DICTIONARY,
    DRUM_GROOVE_BAR_MOTIFS,
    bandScore.drumsLow,
    4,
    totalSamples48
  );
  const [hihatL, hihatR] = synthesizeHiHatPercTrack(
    CYMBAL_SHIMMER_DICTIONARY,
    bandScore.drumsHigh,
    totalSamples48
  );

  if (onProgress) onProgress(0.55, '正在合成 98.0000 BPM 电贝斯原型、Funk吉他与Rhodes电钢 (08-10)...');
  const [bassL, bassR] = synthesizeBassGrooveTrack(
    { atomDict: BASS_ATOM_DICTIONARY, barMotifs: BASS_GROOVE_BAR_MOTIFS },
    bandScore.bass,
    totalSamples48
  );
  const [guitarL, guitarR] = synthesizeGuitarRiffTrack(
    GUITAR_PLUCK_DICTIONARY,
    bandScore.guitar,
    totalSamples48
  );
  const [rhodesL, rhodesR] = synthesizeRhodesPadTrack(
    RHODES_CHORD_DICTIONARY,
    bandScore.rhodes,
    totalSamples48
  );

  if (onProgress) onProgress(0.80, '正在合成副歌双轨和声与 8 阶质数 Householder FDN 混响 (04-05)...');
  const leadSum48 = new Float32Array(totalSamples48);
  for (let i = 0; i < totalSamples48; i++) {
    leadSum48[i] = chest48[i] + formant48[i] + air48[i];
  }
  const [fdnL, fdnR] = synthesizeStudioFdnSpaceTrack(
    STUDIO_FDN_PHYSICS,
    leadSum48,
    guitarL,
    guitarR,
    kickSnareL,
    kickSnareR,
    totalSamples48
  );
  const [harmL, harmR] = synthesizeBackingHarmonyTrack(
    BACKING_HARMONY_DICTIONARY,
    bandScore.backingHarmony,
    totalSamples48
  );

  const elapsedMs = Math.round(performance.now() - t0);
  if (onProgress) onProgress(1.0, `10 轨 48kHz 纯数学字典合成完毕 (${elapsedMs} ms)`);

  return {
    sampleRate: 48000,
    totalSamples: totalSamples48,
    durationSec: totalSamples48 / 48000,
    elapsedMs,
    tierId,
    pitchSemitones,
    tracks: {
      vocal_lead_chest: [chest48],
      vocal_lead_formant: [formant48],
      vocal_breath_air: [air48],
      vocal_chorus_harmony: [harmL, harmR],
      studio_fdn_space: [fdnL, fdnR],
      drum_kick_snare: [kickSnareL, kickSnareR],
      drum_hihat_perc: [hihatL, hihatR],
      bass_groove: [bassL, bassR],
      guitar_funk_riff: [guitarL, guitarR],
      keys_rhodes_pad: [rhodesL, rhodesR],
    },
  };
}

// Re-render ONLY the 3 Lead Vocal tracks (`01..03`) in ~180 ms when user adjusts Pitch Shift!
export function synthesizeVocalOnlyTracks(tierId = 'studio_master', pitchSemitones = 0.0) {
  const t0 = performance.now();
  const { vocalScore } = getScoresForTier(tierId);
  const { chest48, formant48, air48 } = synthesizeVocalThreeTracks(
    VOCAL_WAVECYCLE_DICTIONARY,
    vocalScore,
    { pitchSemitones }
  );
  return {
    elapsedMs: Math.round(performance.now() - t0),
    pitchSemitones,
    tracks: {
      vocal_lead_chest: [chest48],
      vocal_lead_formant: [formant48],
      vocal_breath_air: [air48],
    },
  };
}
