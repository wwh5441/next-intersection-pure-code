// 11-Track Orthogonal Stem Mixer (`src/components/layer-mixer-panel.js`)
export function mountLayerMixerPanel(container, layers, store, onMixChanged) {
  container.innerHTML = `
    <div class="panel-head">
      <h2 class="panel-title">十一轨全曲调音台 (01–08 人声解剖 + 09–11 伴奏字典)</h2>
      <button class="btn" id="btn-reset-layers" style="padding:3px 9px;font-size:11px;">重置推子</button>
    </div>
    <div class="layer-list" id="layer-rows">
      ${layers
        .map(
          (l) => `
        <div class="layer-row" data-layer-id="${l.id}">
          <div class="layer-meta">
            <div class="layer-name-line">
              <span class="layer-swatch" style="background:${l.color}"></span>
              <span class="layer-name">${String(l.index).padStart(2, '0')} · ${l.name}</span>
            </div>
            <span class="layer-sub" title="${l.subtitle}">${l.subtitle}</span>
          </div>
          <div class="ms-btns">
            <button class="ms-btn" data-act="mute" title="静音 Mute">M</button>
            <button class="ms-btn" data-act="solo" title="独奏 Solo">S</button>
          </div>
          <input type="range" class="gain-slider" data-act="gain" min="0" max="1.5" step="0.02" value="1" />
          <div class="meter-bar-bg">
            <div class="meter-bar-fill" id="meter-${l.id}" style="background:${l.color}"></div>
          </div>
        </div>
      `
        )
        .join('')}
    </div>
  `;

  const rowsWrap = container.querySelector('#layer-rows');
  container.querySelector('#btn-reset-layers').addEventListener('click', () => {
    store.resetAllLayers();
    onMixChanged();
  });

  rowsWrap.addEventListener('click', (e) => {
    const btn = e.target.closest('.ms-btn');
    if (!btn) return;
    const row = btn.closest('.layer-row');
    const id = row.dataset.layerId;
    const cur = store.getState().layerParams[id];
    if (btn.dataset.act === 'mute') {
      store.updateLayer(id, { mute: !cur.mute });
    } else if (btn.dataset.act === 'solo') {
      store.updateLayer(id, { solo: !cur.solo });
    }
    onMixChanged();
  });

  rowsWrap.addEventListener('input', (e) => {
    const sl = e.target.closest('.gain-slider');
    if (!sl) return;
    const row = sl.closest('.layer-row');
    store.updateLayer(row.dataset.layerId, { gain: Number(sl.value) });
    onMixChanged();
  });

  function syncControls(state) {
    for (const l of layers) {
      const p = state.layerParams[l.id];
      const row = rowsWrap.querySelector(`.layer-row[data-layer-id="${l.id}"]`);
      if (!row || !p) continue;
      row.querySelector('[data-act="mute"]').classList.toggle('active-mute', p.mute);
      row.querySelector('[data-act="solo"]').classList.toggle('active-solo', p.solo);
      const sl = row.querySelector('[data-act="gain"]');
      if (Number(sl.value) !== p.gain) sl.value = String(p.gain);
    }
  }

  store.subscribe(syncControls);

  return {
    updateMeters(rmsMap) {
      for (const l of layers) {
        const el = container.querySelector(`#meter-${l.id}`);
        if (!el) continue;
        const v = rmsMap[l.id] || 0.0;
        const pct = Math.min(100, Math.round(Math.pow(v * 5.2, 0.65) * 100));
        el.style.width = `${pct}%`;
      }
    },
  };
}
