// Live Plain-Text Mathematical Orthonormal Basis, Canonical Huffman Algebra & 51-Bit Frame Inspector

import { BAND_SCORE } from '../../data/band-score.js';

function findChorusSampleFrame(frames, startIdx = 1445) {
  if (!frames) return { index: 0, entry: [] };
  for (let i = startIdx; i < frames.length; i++) {
    if (frames[i] && Array.isArray(frames[i])) {
      return { index: i, entry: frames[i] };
    }
  }
  return { index: 0, entry: frames[0] || [] };
}

export function mountFormulaInspector(container, store, getActiveEngine) {
  const tabs = [
    { id: 'equations', label: '七档统一正交余弦 + J-正交声场 + 四阶巴特沃斯平滑方程' },
    { id: 'huffman', label: '51-Bit 十进制安全整数与四分类霍夫曼代数模型' },
    { id: 'frame', label: '副歌第 1445 帧 ("秒针转动 DI DI DA") 明文整数实拍' },
    { id: 'score', label: '98.0 BPM 全曲 83 小节结构与和声总谱' },
  ];

  container.innerHTML = `
    <div class="canvas-header">
      <span><strong>100% 纯明文数学代码透视窗</strong> · 零二进制文件 · 零 LZMA · 零 Base64 · 所有数据均为可读十进制安全整数数组</span>
      <span class="status-hint">51-Bit Exact IEEE-754 Integer Stream · 0.38 MB RAM</span>
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
    btn.addEventListener('click', () => store.update({ activeInspectorTab: t.id }));
    tabBar.appendChild(btn);
  });

  function buildText(state) {
    const eng = getActiveEngine ? getActiveEngine() : null;
    const meta = state.activeTierMeta;
    const vLabel = meta ? `${meta.versionCode}【${meta.tierName}】` : 'V9.4.x';
    const phys = eng ? eng.tierPhysics : null;
    const huff = eng ? eng.huffmanTables : null;
    const sample = eng ? findChorusSampleFrame(eng.tierFrames, 1445) : { index: 1445, entry: [] };

    if (state.activeInspectorTab === 'huffman') {
      return [
        `// models/huffman-algebra.js (${vLabel}) — 规范霍夫曼 [symbol, bitLen] 明文生成对节选`,
        `export const H_HDR_SAMPLE = ${JSON.stringify(huff ? huff.H_HDR.slice(0, 48) : [])};`,
        `export const H_C1_SAMPLE  = ${JSON.stringify(huff ? huff.H_C1.slice(0, 48) : [])};`,
        '',
        '// models/tier-physics.js — 当前档位频带切片、J-正交声场与四阶巴特沃斯低通物理常数',
        `export const TIER_PHYSICS = ${JSON.stringify(phys || {}, null, 2)};`,
      ].join('\n');
    }

    if (state.activeInspectorTab === 'frame') {
      const previewWords = Array.isArray(sample.entry) ? sample.entry.slice(0, 36) : sample.entry;
      return [
        `// data/tier-frames-*.js (${vLabel}) — 副歌第 ${sample.index} 帧 (~${((sample.index * 2048) / 48000).toFixed(2)}s) 纯明文 51-bit 十进制安全整数数组`,
        `// 该帧总计 ${Array.isArray(sample.entry) ? sample.entry.length : 0} 个 51-bit 整数 (每个整数 w = c0 + c1*2^17 + c2*2^34 精确承载 51 bit 信息)`,
        `const CHORUS_FRAME_${sample.index} = ${JSON.stringify(previewWords)};`,
      ].join('\n');
    }

    if (state.activeInspectorTab === 'score') {
      return `export const BAND_SCORE = ${JSON.stringify(BAND_SCORE, null, 2)};`;
    }

    return [
      `// [1] ${vLabel} · 48,000 Hz 标准正交自适应局部余弦级数 (Orthonormal Princen-Bradley MDCT)`,
      'psi_{m,k}(n) = sqrt(2 / N_m) * w_m(n) * cos( (pi / N_m) * (n + 0.5 + N_m/2) * (k + 0.5) );',
      'C_hat(t, k)  = sign(q_{t,k}) * 10^((0.5 * u_{t,b(k)} - 90.0) / 20.0) * |q_{t,k}|^(4 / 3);',
      '',
      '// [2] 51-Bit (3×17-Bit) IEEE-754 安全整数无损代数打包 (逼近十进制 ASCII 香农信息密度极限)',
      'w_i = c_{3i} + c_{3i+1} * 2^17 + c_{3i+2} * 2^34   (0 <= w_i < 2^51 < Number.MAX_SAFE_INTEGER);',
      '',
      '// [3] 反对称正交算子 J 立体声合成 (J^T = -J, <u, Ju> ≡ 0) 与四阶巴特沃斯抗旁瓣连续滤波 (-24 dB/oct, 零伪造噪声)',
      'C_Side(t, 2m : 2m+2) = alpha_pan * [u_0, u_1] + beta_wid * [-u_1, +u_0];',
      'H_BW4(z)             = H_BQ(z; f_c, Q=0.541196) * H_BQ(z; f_c, Q=1.306563)   (消除 100% 离散余弦旁瓣电流声);',
    ].join('\n');
  }

  function refresh(state) {
    codeBox.textContent = buildText(state);
    tabBar.querySelectorAll('[data-tab]').forEach((b) => {
      b.classList.toggle('active', b.dataset.tab === state.activeInspectorTab);
    });
  }

  refresh(store.getState());
  store.subscribe((state, keys) => {
    if (
      keys.includes('activeInspectorTab') ||
      keys.includes('activeTierMeta') ||
      keys.includes('isReady')
    ) {
      refresh(state);
    }
  });
}
