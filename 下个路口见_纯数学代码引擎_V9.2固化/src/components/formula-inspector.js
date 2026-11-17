// Live Plain-Text Mathematical Model & Dual-Tier Score Code Inspector Component (V9.2)
// Proves 100% human-readable mathematical code across BOTH Vocal & 5-Track Real-Detail Band (zero LZMA / zero binary)

import { SINGER_PHYSICS } from '../../models/singer-physics.js';
import { MANDARIN_SYLLABLES } from '../../models/mandarin-phonetics.js';
import { INSTRUMENT_PHYSICS } from '../../models/instrument-physics.js';
import { BAND_SCORE } from '../../data/band-score.js';
import { VOCAL_TIERS } from '../../data/vocal-tiers.js';

function buildInspectorContentForTier(tierId) {
  const entry = VOCAL_TIERS[tierId] || VOCAL_TIERS[Object.keys(VOCAL_TIERS)[0]];
  const prof = entry.profile;
  const score = entry.score;
  const bandProf = entry.bandProfile;
  const bandScore = entry.bandScore;

  const sampleNotes = score
    .slice(0, 6)
    .map((r) => `  ${JSON.stringify(r)},`)
    .join('\n');

  const sampleSyllables = ['gang', 'xia', 'de', 'di', 'tie', 'lu', 'kou', 'jian']
    .filter((k) => MANDARIN_SYLLABLES[k])
    .map((k) => `  ${k}: ${JSON.stringify(MANDARIN_SYLLABLES[k])},`)
    .join('\n');

  const bandPreview = {
    bandProfile: bandProf,
    bassSampleKnots: {
      nFrames: bandScore.bass.nFrames,
      rmsFirst24Knots: [bandScore.bass.rms[0].slice(0, 24), bandScore.bass.rms[1].slice(0, 24)],
      f0First24Knots: [bandScore.bass.f0[0].slice(0, 24), bandScore.bass.f0[1].slice(0, 24)],
    },
    drumSampleKnots: {
      nFrames: bandScore.drums.nFrames,
      kickFirst24Knots: [bandScore.drums.kick[0].slice(0, 24), bandScore.drums.kick[1].slice(0, 24)],
      woodblockSnareFirst24Knots: [bandScore.drums.body[0].slice(0, 24), bandScore.drums.body[1].slice(0, 24)],
    },
    harmonySampleKnots: {
      nFrames: bandScore.harmony.nFrames,
      strumBand0First24Knots: [bandScore.harmony.strum[0][0].slice(0, 24), bandScore.harmony.strum[0][1].slice(0, 24)],
    },
  };

  return {
    equations: [
      `// [1] 当前激活版本: ${prof.fullLabel}`,
      `//     人声: pcaDims = ${prof.pcaDims}, exprStep = ${prof.exprKnotStepFrames} 帧, quantStep = ${prof.quantStepDb} dB`,
      `//     伴奏: ${bandProf.label} (贝斯精度 ${bandProf.bassPitchPrecisionCents} 音分)`,
      'u_ERB(f) = (21.4 * log10(1 + 0.00437 * f) - E_min) / (E_max - E_min);',
      `c_m(t)   = mu_chun[m] + P_syllable(t)[m] + sum_{k=1..${prof.pcaDims}} w_expr,k(t) * U_chun[k, m];`,
      'v_vocal(t) = g(t) * sum_{h=1..64} 10^(L_tract(h*f0, t)/20) * cos(h * phi_0(t) - psi_LF[h]);',
      '',
      '// [2] 原曲级高细节 5 轨纯数学乐队方程 (06..10 独立推子, 100% 线性可加还原)',
      'Drums_06(t) = A_kick(t)*sin(int 2*pi*(58.59 + 72*boost*(1-a)) dt) + A_wood(t)*sin(int 2*pi*(164.06 + 25*boost*(1-a)) dt) + IFFT_Low3.3k(Proto_64[idx(t)]);',
      'Drums_07(t) = Shimmer_MS( IFFT_High3.3k( Proto_64[idx(t)] * HighRMS(t) * exp(i*theta_rand) ) );',
      'Bass_08(t)  = sum_{k=1..18} A_k(toneIdx(t), rms(t)) * cos(k * sqrt(1 + 0.00018*k^2) * phi_0(t) - 0.22*k) + PluckNoise(t);',
      'Guitar_09(t)= StereoFDN( sum_{n=MIDI_62..95} A_n(t) * Im{ z_n(t)^(1..10) . W_bright(t) } + StrumOLA_80fps(t), sideRatio(t) );',
      'Keys_10(t)  = StereoFDN( sum_{n=MIDI_40..61} A_n(t) * Im{ z_n(t)^(1..10) . W_warm(t) }, sideRatio(t) );',
    ].join('\n'),

    phonetics: [
      '// models/singer-physics.js (24x40全特征嗓音核) + models/mandarin-phonetics.js (97音节40维原型)',
      `export const SINGER_PHYSICS = ${JSON.stringify(SINGER_PHYSICS, null, 2)};`,
      '',
      '// 97 音节参数化声韵母原型节选: [声母类别, 声母3锚点x40模差分, 声母8频带HNR, 韵母5锚点x40模差分, 韵母8频带HNR]',
      sampleSyllables,
    ].join('\n'),

    vocalScore: [
      `// 当前查看档位: data/vocal-score-${prof.id}.js (${prof.fullLabel})`,
      `export const VOCAL_PROFILE = ${JSON.stringify(prof, null, 2)};`,
      '',
      `// 全曲 285 次演唱事件明文控制曲线 (前 6 颗展示，零 LZMA / 零二进制!)`,
      'export const VOCAL_SCORE = [',
      sampleNotes,
      `  // ... 共 ${score.length} 颗全曲演唱事件`,
      '];',
    ].join('\n'),

    bandScore: [
      `// models/instrument-physics.js + data/band-score-${bandProf.id}.js — 原曲级 5 轨高细节纯明文伴奏模型与折线控制谱`,
      `export const BAND_PREVIEW = ${JSON.stringify(bandPreview, null, 2)};`,
      '',
      `export const INSTRUMENT_PHYSICS = ${JSON.stringify(INSTRUMENT_PHYSICS, null, 2)};`,
      '',
      `export const SECTIONS = ${JSON.stringify(BAND_SCORE.sections, null, 2)};`,
    ].join('\n'),
  };
}

export function mountFormulaInspector(container, store) {
  const tabs = [
    { id: 'equations', label: '核心声学与 5 轨真伴奏方程' },
    { id: 'phonetics', label: '李宇春24维嗓音核+97音节原型' },
    { id: 'vocalScore', label: '当前档 285 人声事件明文曲线' },
    { id: 'bandScore', label: '当前档 5 轨高细节伴奏物理模型与控制谱' },
  ];

  container.innerHTML = `
    <div class="canvas-header">
      <span><strong>100% 纯明文数学代码透视窗</strong> · 随上方 A/B 档位切换同步展示当前档人声与 5 轨伴奏明文源码</span>
      <span class="status-hint">Zero LZMA · Zero Base122 · Zero Waveforms</span>
    </div>
    <div class="inspector-tabs" id="inspector-tab-bar"></div>
    <pre class="code-viewer" id="inspector-code-box"></pre>
  `;

  const tabBar = container.querySelector('#inspector-tab-bar');
  const codeBox = container.querySelector('#inspector-code-box');

  tabs.forEach((t) => {
    const btn = document.createElement('button');
    btn.className = `btn-chip${t.id === 'equations' ? ' active' : ''}`;
    btn.dataset.tab = t.id;
    btn.textContent = t.label;
    btn.addEventListener('click', () => {
      store.update({ activeInspectorTab: t.id });
    });
    tabBar.appendChild(btn);
  });

  function renderActiveCode(state) {
    const contentMap = buildInspectorContentForTier(state.vocalTier);
    codeBox.textContent = contentMap[state.activeInspectorTab] || contentMap.equations;
    tabBar.querySelectorAll('[data-tab]').forEach((b) => {
      b.classList.toggle('active', b.dataset.tab === state.activeInspectorTab);
    });
  }

  renderActiveCode(store.getState());
  store.subscribe((state, keys) => {
    if (keys.includes('activeInspectorTab') || keys.includes('vocalTier')) {
      renderActiveCode(state);
    }
  });
}
