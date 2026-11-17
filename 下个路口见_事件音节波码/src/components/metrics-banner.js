export function renderMetricsStrip(container, { manifest, sizes, synthMs }) {
  const profReport = sizes.profiles[manifest.profile];
  const stats = sizes.atlasStats;
  const sharedKb = (sizes.sharedCoreBytes.totalShared / 1024).toFixed(1);
  const evKb = (profReport.eventStreamBytes / 1024).toFixed(1);
  const fullKb = profReport.fullSongKB ?? profReport.totalKB;
  const fullKbps = profReport.fullSongKbps ?? profReport.equivalentKbps;
  const fullRedMp3 = profReport.fullSongReductionVsMp3Pct ?? '—';

  container.innerHTML = `
    <div class="metric-card">
      <span class="metric-label">全曲总波码体积 (人声+伴奏)</span>
      <div class="metric-value">${fullKb} KB</div>
      <div class="metric-note">全曲 ${fullKbps} kbps · 较 MP3 缩减 ${fullRedMp3}% · 较 WAV ${profReport.fullSongReductionVsWavPct ?? profReport.reductionVsWavPct}%</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">纯人声波码 (底座 + 事件流)</span>
      <div class="metric-value">${profReport.totalKB} KB</div>
      <div class="metric-note">${sharedKb} + ${evKb} KB · ${profReport.equivalentKbps} kbps · 伴奏 ${profReport.accompKB ?? 0} KB</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">发声事件 × 去重音节库</span>
      <div class="metric-value">${stats.totalEvents} 次 → ${stats.uniqueSyllablePrototypes} 音节</div>
      <div class="metric-note">平均复用 ${stats.averageEventReuse}× · 无声段 0 字节</div>
    </div>
    <div class="metric-card">
      <span class="metric-label">窄带谐波与感知信噪比</span>
      <div class="metric-value">${manifest.logMelSnrDryDb} dB</div>
      <div class="metric-note">${manifest.harm10RmseDb != null ? `H1-10误差 ${manifest.harm10RmseDb} dB · ` : ''}全混响 ${manifest.logMelSnrFullDb} dB · 合成 ${synthMs} ms</div>
    </div>
  `;
}
