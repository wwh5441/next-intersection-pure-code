export function renderMetricsStrip(container, { manifest, sizes, synthMs }) {
  const profReport = sizes.profiles[manifest.profile];
  const stats = sizes.atlasStats;
  const sharedKb = (sizes.sharedCoreBytes.totalShared / 1024).toFixed(1);
  const evKb = (profReport.eventStreamBytes / 1024).toFixed(1);
  const purity = manifest.purity || profReport.purity || {};

  container.innerHTML = `
    <div class="metric-card">
      <span class="metric-label">去残纯净波码体积 (V8.3)</span>
      <div class="metric-value">${profReport.totalKB} KB</div>
      <div class="metric-note">码率 ${profReport.equivalentKbps} kbps · 较 V8 同档再减 ${profReport.reductionVsV8Pct}%</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">共享底座 + 事件流拆分</span>
      <div class="metric-value">${sharedKb} + ${evKb} KB</div>
      <div class="metric-note">纯净底座 ${sharedKb} KB · 逐字表情 ${manifest.bytesPerEventAvg} B/字</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">元音谐噪纯净度 (HNR)</span>
      <div class="metric-value">${purity.hnrDb ?? '--'} dB</div>
      <div class="metric-note">V8 初版 ${purity.v8HnrDb ?? 1.9} dB · 元音 &lt;5kHz 零底嘶铁律</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">字头尖峰 &amp; 混响混叠清除</span>
      <div class="metric-value">${purity.knot0ResidualDb ?? '--'} dB · ${purity.fdnAliasingAbove12kPct ?? 0}%</div>
      <div class="metric-note">V8 字头尖峰 ${purity.v8Knot0ResidualDb ?? 60.9}dB · 剔除伴奏幽灵残片 ${stats.purgedGhostBleedIslands ?? 0} 处</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">感知频谱信噪比 (Mel-SNR)</span>
      <div class="metric-value">${manifest.logMelSnrDryDb} dB</div>
      <div class="metric-note">全混响场 ${manifest.logMelSnrFullDb} dB · 包络相关 ${manifest.envelopeCorrelation} · 合成 ${synthMs} ms</div>
    </div>
  `;
}
