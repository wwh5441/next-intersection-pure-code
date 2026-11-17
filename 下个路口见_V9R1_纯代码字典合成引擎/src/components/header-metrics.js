// V9R1 Header Metrics & Dual-Tier Switcher Component (`src/components/header-metrics.js`)
import { V9R1_PARADIGM_BENCHMARKS } from '../../data/benchmarks.js';
import { V9R1_TIERS } from '../../data/tiers-registry.js';
import { studioStore } from '../state/studio-store.js';
import { audioMixer } from '../modules/audio-mixer.js';

function fmtKb(bytes) {
  return `${(bytes / 1024).toFixed(2)} KB`;
}

export function mountHeaderMetrics(container) {
  const b = V9R1_PARADIGM_BENCHMARKS;
  const availableTierIds = Object.keys(V9R1_TIERS);

  container.innerHTML = `
    <div class="hero-top">
      <div class="hero-brand">
        <div class="hero-eyebrow">V9R1 · Re-Anchored From V8.5 · Pure Mathematical Dual-Dictionary Engine</div>
        <h1 class="hero-title">《下个，路口，见》V9R1 纯代码双字典物理合成工作台</h1>
        <p class="hero-desc">
          零 LZMA · 零 Base122 · 零盲目 MDCT 波形帧 —— 100% 人类可读 ES Module（<strong>97 音节声韵字典 + 86 小节 PAM 伴奏原型字典 + 10 轨微分方程合成器</strong>）
        </p>
      </div>
      <div class="tier-switch-group" id="v9r1-tier-group"></div>
    </div>

    <div class="metrics-grid">
      <div class="metric-box">
        <div class="metric-label">48kHz 立体声母带基准</div>
        <div class="metric-val">${(b.masterWavBytes / (1024 * 1024)).toFixed(2)} MB</div>
        <div class="metric-note">210.37s · PCM 16-bit</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">V8.5 + V8伴奏字典 (旧锚点)</div>
        <div class="metric-val">${fmtKb(b.v85PlusV8DictLzmaBytes)}</div>
        <div class="metric-note">LZMA 二进制 · 伴奏仍含残差流</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">旧 V9.2 平铺乐谱 (未字典化)</div>
        <div class="metric-val">${fmtKb(b.oldV92UnrolledGzipBytes)}</div>
        <div class="metric-note">Gzip · 86小节鼓/贝斯逐帧平铺</div>
      </div>
      <div class="metric-box is-highlight" id="metric-active-total">
        <div class="metric-label" id="metric-active-label">V9R1 Ultra 双字典总包</div>
        <div class="metric-val" id="metric-active-val">${fmtKb(b.tiers.ultra.gzipBytes)}</div>
        <div class="metric-note" id="metric-active-note">Gzip · 压缩比 361.5 : 1</div>
      </div>
      <div class="metric-box is-band-highlight">
        <div class="metric-label">母带包络/音高真实度 r</div>
        <div class="metric-val" id="metric-corr-val">0.969 / 0.982</div>
        <div class="metric-note" id="metric-corr-note">Bass F0 / Kick Env 相关系数</div>
      </div>
    </div>
  `;

  const tierGroup = container.querySelector('#v9r1-tier-group');
  const tierButtons = new Map();

  for (const id of availableTierIds) {
    const meta = b.tiers[id] || b.tiers.ultra;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tier-btn';
    btn.dataset.tier = id;
    btn.innerHTML = `
      <div class="tier-btn-top">
        <span>${id === 'ultra' ? 'Ultra 极简双字典档' : 'Expressive 精修双字典档'}</span>
        <span class="tier-size-pill">${fmtKb(meta.gzipBytes)} Gz</span>
      </div>
      <div class="tier-btn-sub">人声 ${fmtKb(meta.vocalGzipBytes)} + 伴奏 ${fmtKb(meta.bandGzipBytes)}</div>
    `;
    btn.addEventListener('click', () => {
      const st = studioStore.getState();
      if (st.isRendering || st.tierId === id) return;
      audioMixer.loadTier(id);
    });
    tierGroup.appendChild(btn);
    tierButtons.set(id, btn);
  }

  const elLabel = container.querySelector('#metric-active-label');
  const elVal = container.querySelector('#metric-active-val');
  const elNote = container.querySelector('#metric-active-note');
  const elCorrVal = container.querySelector('#metric-corr-val');
  const elCorrNote = container.querySelector('#metric-corr-note');

  let lastTier = null;
  let lastRendering = null;

  const sync = (st) => {
    if (st.tierId === lastTier && st.isRendering === lastRendering) return;
    lastTier = st.tierId;
    lastRendering = st.isRendering;

    for (const [id, btn] of tierButtons.entries()) {
      btn.classList.toggle('is-active', id === st.tierId);
      btn.disabled = st.isRendering;
    }

    const meta = b.tiers[st.tierId] || b.tiers.ultra;
    const ratio = (b.masterWavBytes / meta.gzipBytes).toFixed(1);
    elLabel.textContent = meta.title;
    elVal.textContent = `${fmtKb(meta.gzipBytes)} Gzip`;
    elNote.textContent = `Raw ${fmtKb(meta.rawBytes)} · 压缩比 ${ratio} : 1`;
    elCorrVal.textContent = `${meta.correlations.bassPitchR.toFixed(3)} / ${meta.correlations.drumKickR.toFixed(3)}`;
    elCorrNote.textContent = `Bass音高 r / 底鼓瞬态 r (复调 ${meta.correlations.harmonyPolyR.toFixed(3)})`;
  };

  sync(studioStore.getState());
  studioStore.subscribe(sync);
}
