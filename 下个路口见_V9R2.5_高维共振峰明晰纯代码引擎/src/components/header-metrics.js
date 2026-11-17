// Header & Verified Metrics Banner (`src/components/header-metrics.js`)
export function mountHeaderMetrics(container, metadata, store) {
  const { config, metrics } = metadata;
  const m = metrics.metrics;

  function render(state) {
    container.innerHTML = `
      <div>
        <span class="brand-eyebrow">V9R2.5 PURE CODE · DYNAMIC GLOTTAL + FORMANT CONTRAST + PLATE SPACE</span>
        <h1 class="studio-title">${config.title}</h1>
        <p class="studio-subtitle">${config.subtitle}</p>
      </div>
      <div class="metrics-grid">
        <div class="metric-chip">
          <span class="metric-label">全曲 Log-Mel 还原度 (vs 原曲)</span>
          <span class="metric-value">${m.masterMelSnrDb} dB</span>
          <span class="metric-sub">包络相关 r = ${m.masterEnvCorr.toFixed(4)}</span>
        </div>
        <div class="metric-chip">
          <span class="metric-label">主唱 / 完整人声</span>
          <span class="metric-value">${m.leadMelSnrDb} / ${m.vocalsMelSnrDb} dB</span>
          <span class="metric-sub">97 原型 · ${metrics.atlasStats.totalEvents} 事件参数合成</span>
        </div>
        <div class="metric-chip">
          <span class="metric-label">贝斯 / 鼓 / 吉他电钢</span>
          <span class="metric-value">${m.bassMelSnrDb} / ${m.drumsMelSnrDb} / ${m.guitarSynthMelSnrDb} dB</span>
          <span class="metric-sub">谐波模型 · 鼓击事件 · 复调音符</span>
        </div>
        <div class="metric-chip">
          <span class="metric-label">纯代码数据总量</span>
          <span class="metric-value">${metrics.totalGzipKB} KB Gzip</span>
          <span class="metric-sub">明文 ${metrics.totalRawKB} KB · 合成 ${state.synthMs ? (state.synthMs / 1000).toFixed(2) : '—'}s</span>
        </div>
      </div>
    `;
  }

  render(store.getState());
  store.subscribe(render);
}
