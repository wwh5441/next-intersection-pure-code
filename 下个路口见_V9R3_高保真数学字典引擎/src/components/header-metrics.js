// Header Banner & Ground-Truth 48kHz Studio Stem SNR Matrix (`src/components/header-metrics.js`)
import { V9R3_BENCHMARKS } from '../../data/benchmarks.js';
import { V9R3_TIERS } from '../../data/tiers-registry.js';
import { studioStore } from '../state/studio-store.js';

const STEM_BENCH_LABELS = [
  { key: 'lead_vocal', label: '01-03 李宇春主唱总成' },
  { key: 'drums_kick_snare', label: '06 底鼓/军鼓/木鱼' },
  { key: 'bass_groove', label: '08 电贝斯律动' },
  { key: 'other_combined', label: '09-10 吉他+Rhodes' },
  { key: 'backing_space', label: '04-05 和声+FDN空间' },
  { key: 'full_master_mix', label: '10轨合成立体声母带' },
];

export function mountHeaderMetrics(rootEl) {
  function render() {
    const { tierId, elapsedSynthMs } = studioStore.getState();
    const curTier = V9R3_TIERS[tierId] || V9R3_TIERS.studio_master;
    const mCur = curTier.metrics;
    const mOld = V9R3_BENCHMARKS.v9r1Expressive;

    const cardsHtml = STEM_BENCH_LABELS.map(({ key, label }) => {
      const c = mCur[key] || { wavSnrDb: 0, magSnrDb: 0, audEnvSnrDb: 0 };
      const o = mOld[key] || { wavSnrDb: 0, magSnrDb: 0 };
      const dMag = (c.magSnrDb - o.magSnrDb).toFixed(1);
      return `
        <div class="snr-card">
          <div class="snr-card-title">${label}</div>
          <div class="snr-card-row">
            <span>V9R1 MagSNR: ${o.magSnrDb.toFixed(1)} dB</span>
            <span class="snr-gain-pill">+${dMag} dB</span>
          </div>
          <div class="snr-card-row">
            <span>V9R3 Mag / Env:</span>
            <strong>${c.magSnrDb.toFixed(1)} / ${c.audEnvSnrDb.toFixed(1)} dB</strong>
          </div>
        </div>
      `;
    }).join('');

    rootEl.innerHTML = `
      <div class="hero-top">
        <div>
          <div class="hero-eyebrow">PURE MATH + CODE + DICTIONARY · 100% PLAIN-TEXT ES MODULES · 10 INDEPENDENT STEMS</div>
          <h1 class="hero-title">《下个，路口，见》V9R3 高保真数学字典合成工作台</h1>
          <p class="hero-desc">
            基于精确校准的 <strong>98.0000 BPM</strong> 录音室网格（消除旧版 66ms 累积时钟漂移）、<strong>GCI 声门闭合周期字典 (Kv=144) + 汉语清辅音小波字典 (Ku=112)</strong> 与<strong>全分轨临界采样乐器物理字典</strong>，10 轨 100% 由纯文本 ES Module 数学合成。
          </p>
        </div>
        <div class="kpi-strip">
          <div class="kpi-box">
            <div class="kpi-label">主唱幅度谱信噪比</div>
            <div class="kpi-val accent">${mCur.lead_vocal.magSnrDb.toFixed(2)} dB</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">伴奏(吉+键/贝/鼓)信噪比</div>
            <div class="kpi-val accent">${mCur.other_combined.magSnrDb.toFixed(1)} / ${mCur.bass_groove.magSnrDb.toFixed(1)} / ${mCur.drums_kick_snare.magSnrDb.toFixed(1)} dB</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">10轨母带听觉包络SNR</div>
            <div class="kpi-val accent">${mCur.full_master_mix.audEnvSnrDb.toFixed(2)} dB</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">48kHz 10轨合成耗时</div>
            <div class="kpi-val">${elapsedSynthMs > 0 ? `${elapsedSynthMs} ms` : '合成中...'}</div>
          </div>
        </div>
      </div>
      <div class="snr-grid">${cardsHtml}</div>
    `;
  }

  render();
  let lastTier = studioStore.getState().tierId;
  let lastMs = studioStore.getState().elapsedSynthMs;
  studioStore.subscribe((st) => {
    if (st.tierId !== lastTier || st.elapsedSynthMs !== lastMs) {
      lastTier = st.tierId;
      lastMs = st.elapsedSynthMs;
      render();
    }
  });
}
