export const COMBOS = [
  { id: '8track', badge: '5+3 八轨', title: '5 人声 + 3 伴奏', shortNote: '独立气声/辅音/混响 + 贝斯/鼓/和声' },
  { id: '4track', badge: '1+3 四轨', title: '1 人声 + 3 伴奏', shortNote: '人声合1立体声 + 贝斯/鼓/和声三轨' },
  { id: '2track', badge: '1+1 双轨', title: '1 人声 + 1 伴奏', shortNote: '立体声人声 + 立体声全乐队伴奏' },
  { id: '1track', badge: '1 单轨', title: '人声+伴奏合一', shortNote: '零AI拆轨损耗 · 母带直合极简版' },
];

export function renderCompareCards(container, sizesByCombo, state, onSelectCombo) {
  if (!container) return;
  const profile = state.profile;
  const baseBytes = sizesByCombo['8track']?.[profile]?.parameterBytes || 1;

  container.innerHTML = '';
  for (const combo of COMBOS) {
    const info = sizesByCombo[combo.id]?.[profile];
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'combo-card';
    btn.dataset.combo = combo.id;
    btn.setAttribute('aria-pressed', String(state.combo === combo.id));

    let sizeLabel = '加载中';
    let deltaLabel = '';
    let metricsLabel = combo.shortNote;

    if (info) {
      sizeLabel = `${(info.parameterBytes / 1e6).toFixed(2)} MB`;
      const pctDiff = ((info.parameterBytes - baseBytes) / baseBytes) * 100;
      deltaLabel = combo.id === '8track' ? '基准' : `${pctDiff.toFixed(1)}%`;
      metricsLabel = `${info.kbps} kbps · Mel ${info.mixLogMelSnrDb} dB · Wave ${info.mixWaveSnrDb} dB`;
    }

    btn.innerHTML = `
      <div class="combo-top">
        <span class="combo-badge">${combo.badge}</span>
        <span class="combo-delta">${deltaLabel}</span>
      </div>
      <div class="combo-title">${combo.title}</div>
      <div class="combo-size">${sizeLabel}</div>
      <div class="combo-metrics">${metricsLabel}</div>
    `;
    btn.addEventListener('click', () => onSelectCombo(combo.id));
    container.append(btn);
  }
}

export function updateCompareCardsSelection(container, comboId) {
  if (!container) return;
  for (const btn of container.children) {
    btn.setAttribute('aria-pressed', String(btn.dataset.combo === comboId));
  }
}
