// Header & Dynamic Per-Tier Pure-Code Byte Metrics Strip Component (V9.2)

import { VOCAL_TIERS } from '../../data/vocal-tiers.js';

function fmtKb(bytes) {
  return `${(bytes / 1024).toFixed(2)} KB`;
}

export function mountHeaderMetrics(container, store) {
  const tierCount = Object.keys(VOCAL_TIERS).length;
  const titleBadge =
    tierCount > 1
      ? 'V9.2 双档完整版'
      : `V9.2 ${VOCAL_TIERS[Object.keys(VOCAL_TIERS)[0]].profile.shortLabel}`;

  container.innerHTML = `
    <div class="header-row">
      <div class="title-block">
        <h1>《李宇春 - 下个，路口，见》${titleBadge} · 10 轨纯数学代码声学合成引擎</h1>
        <p id="header-tier-subtitle">40 维李宇春嗓音核 + 97 音节原型 + 285 事件微表情 + 原曲级 5 轨高细节纯数学乐队 · 零 LZMA · 零 Base122 · 零采样</p>
      </div>
      <div class="badge-strip" id="metrics-badge-strip">
        <span class="metric-pill">当前 5 轨人声 <strong id="m-vocal-kb">760.64 KB (Gzip 222.51 KB)</strong></span>
        <span class="metric-pill band">当前 5 轨伴奏 <strong id="m-band-kb">471.21 KB (Gzip 122.10 KB)</strong></span>
        <span class="metric-pill">当前档 10 轨明文 <strong id="m-total-score-kb">1231.85 KB</strong></span>
        <span class="metric-pill band">当前档 Gzip 体积 <strong id="m-gzip-kb">344.48 KB</strong></span>
      </div>
    </div>
  `;

  const elSub = container.querySelector('#header-tier-subtitle');
  const elVocal = container.querySelector('#m-vocal-kb');
  const elBand = container.querySelector('#m-band-kb');
  const elTotal = container.querySelector('#m-total-score-kb');
  const elGzip = container.querySelector('#m-gzip-kb');

  function refreshHeader(state) {
    const tierObj = VOCAL_TIERS[state.vocalTier];
    if (tierObj) {
      elSub.textContent = `当前激活：[${tierObj.profile.fullLabel}] · 零 LZMA · 零 Base122 · 零音频采样`;
    }

    const m = state.metrics;
    if (!m) return;

    const tm = (m.tiers && m.tiers[state.vocalTier]) || m;
    if (tm.vocalModelAndScoreBytes) {
      elVocal.textContent = `${fmtKb(tm.vocalModelAndScoreBytes)} (Gzip ${fmtKb(tm.vocalModelAndScoreGzipBytes)})`;
    }
    if (tm.bandModelAndScoreBytes) {
      elBand.textContent = `${fmtKb(tm.bandModelAndScoreBytes)} (Gzip ${fmtKb(tm.bandModelAndScoreGzipBytes)})`;
    } else if (m.bandModelAndScoreBytes) {
      elBand.textContent = `${fmtKb(m.bandModelAndScoreBytes)} (Gzip ${fmtKb(m.bandModelAndScoreGzipBytes)})`;
    }
    if (tm.totalModelAndScoreBytes) {
      elTotal.textContent = fmtKb(tm.totalModelAndScoreBytes);
    }
    if (tm.totalModelAndScoreGzipBytes) {
      elGzip.textContent = fmtKb(tm.totalModelAndScoreGzipBytes);
    }
  }

  refreshHeader(store.getState());
  store.subscribe((state, keys) => {
    if (keys.includes('metrics') || keys.includes('vocalTier')) {
      refreshHeader(state);
    }
  });
}
