// Live Plain-Text Mathematical Orthonormal Basis & Partition-of-Unity Code Inspector

import { ORTHONORMAL_PHYSICS } from '../../models/orthonormal-basis-physics.js';
import { BAND_SCORE } from '../../data/band-score.js';
import {
  MASTER_MID_SERIES,
  MASTER_SIDE_SERIES,
  STEM_PARTITION_SERIES,
} from '../../data/master-series-index.js';

function findFirstActiveFrame(series, startIdx = 600) {
  for (let i = startIdx; i < series.length; i++) {
    if (series[i] && Array.isArray(series[i])) return { index: i, frame: series[i] };
  }
  return { index: 0, frame: series[0] };
}

function buildInspectorContent(state) {
  const ver = state.versionLabel || 'V9.3';
  const sampleM = findFirstActiveFrame(MASTER_MID_SERIES, 1450); // Chorus 1 ~ 61.8s
  const sampleS = findFirstActiveFrame(MASTER_SIDE_SERIES, sampleM.index);
  const sampleP = findFirstActiveFrame(STEM_PARTITION_SERIES, sampleM.index);

  const previewMid = [
    sampleM.frame[0],
    sampleM.frame[1],
    Array.isArray(sampleM.frame[2]) ? sampleM.frame[2].slice(0, 48) : sampleM.frame[2],
  ];
  const previewSide = [
    sampleS.frame[0],
    sampleS.frame[1],
    Array.isArray(sampleS.frame[2]) ? sampleS.frame[2].slice(0, 48) : sampleS.frame[2],
  ];

  return {
    equations: [
      `// [1] ${ver} 48,000 Hz 正交归一化 Princen-Bradley 局部余弦级数精确重建方程`,
      '//     满足 Parseval 能量守恒恒等式: ||x - x_hat||_2^2 === ||C - C_hat||_2^2',
      'psi_{m,k}(t) = sqrt(2 / N_m) * w_m(t - t_m) * cos( (pi / N_m) * (t - t_m + 0.5 + N_m/2) * (k + 0.5) );',
      'c_hat(t, k)  = sign(q_{t,k}) * 10^((0.5 * u_{t,b(k)} - 90.0) / 20.0) * |q_{t,k}|^(4 / 3);',
      '',
      '// [2] 32-ERB 心理声学子带 6 轨精确整数单位分解定律 (Partition-of-Unity)',
      'W_5^Mid(t, b)  = 128 - (W_0^Mid + W_1^Mid + W_2^Mid + W_3^Mid + W_4^Mid);   // 恒满足 sum_{i=0..5} W_i === 128',
      'W_0^Side(t, b) = 0;                                                         // 01 李宇春主唱纯中置干声零侧向串扰',
      'W_5^Side(t, b) = 128 - (W_1^Side + W_2^Side + W_3^Side + W_4^Side);         // 恒满足 sum_{i=0..5} W_i === 128',
      '',
      '// [3] 六轨独立合成与全混音精确代数闭合定理',
      'Sum_{i=0..5} Stem_i(t) = T_TDAC^{-1}( (Sum_{i=0..5} W_i(t,b) / 128) * C_hat(t,k) ) === T_TDAC^{-1}( C_hat(t,k) );',
      '//     => 当 6 轨推子同处 0.0 dB 时，零 AI 拆轨损耗、零相位抵消，100% 还原 48kHz 双声道母带!',
    ].join('\n'),

    physics: [
      '// models/orthonormal-basis-physics.js — 48kHz 正交局部余弦基与 32-ERB 听感子带物理常数',
      `export const ORTHONORMAL_PHYSICS = ${JSON.stringify(ORTHONORMAL_PHYSICS, null, 2)};`,
    ].join('\n'),

    cosineSeries: [
      `// data/master-mid-*.js & master-side-*.js — 副歌第 ${sampleM.index} 帧 (~${((sampleM.index * 2048) / 48000).toFixed(2)}s "秒针转动 DI DI DA") 纯明文系数实拍`,
      '// 零 LZMA · 零 Base122 · 零 Base64 · 100% 人类可读明文数值数组',
      `const FRAME_${sampleM.index}_MID_PREVIEW  = ${JSON.stringify(previewMid)};`,
      `const FRAME_${sampleS.index}_SIDE_PREVIEW = ${JSON.stringify(previewSide)};`,
      `// 全曲共 ${ORTHONORMAL_PHYSICS.nFrames} 帧 (42.67ms/帧, 瞬态自动切换 4x10.67ms Princen-Bradley 短窗)`,
    ].join('\n'),

    partition: [
      `// data/stem-partition-*.js — 第 ${sampleP.index} 帧 6 轨 32-ERB 整数单位分解权重矩阵 (和恒等于 128)`,
      `const FRAME_${sampleP.index}_PARTITION = ${JSON.stringify(sampleP.frame)};`,
      '',
      '// data/band-score.js — 98.0 BPM 全曲 83 小节结构与和声进行表',
      `export const BAND_SCORE = ${JSON.stringify(BAND_SCORE, null, 2)};`,
    ].join('\n'),
  };
}

export function mountFormulaInspector(container, store) {
  const tabs = [
    { id: 'equations', label: '正交局部余弦与单位分解定理' },
    { id: 'physics', label: '48kHz 32-ERB 正交基常数' },
    { id: 'cosineSeries', label: '副歌母带正交余弦明文系数实拍' },
    { id: 'partition', label: '6 轨守恒权重与全曲和声结构' },
  ];

  container.innerHTML = `
    <div class="canvas-header">
      <span><strong>100% 纯明文数学代码透视窗</strong> · 任何一帧均可直接查看明文正交余弦系数与 6 轨整数单位分解权重</span>
      <span class="status-hint">Zero LZMA · Zero Base122 · Zero Audio Files</span>
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
    const contentMap = buildInspectorContent(state);
    codeBox.textContent = contentMap[state.activeInspectorTab] || contentMap.equations;
    tabBar.querySelectorAll('[data-tab]').forEach((b) => {
      b.classList.toggle('active', b.dataset.tab === state.activeInspectorTab);
    });
  }

  renderActiveCode(store.getState());
  store.subscribe((state, keys) => {
    if (keys.includes('activeInspectorTab') || keys.includes('versionLabel')) {
      renderActiveCode(state);
    }
  });
}
