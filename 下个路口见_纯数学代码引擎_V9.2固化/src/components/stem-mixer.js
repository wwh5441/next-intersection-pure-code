// Two-Column 10-Track Studio Stem Mixer Component
// Left Column (01..05): Pure-Math Vocal & Studio FDN Space Stems
// Right Column (06..10): Pure-Math Physical-Modeling Band Accompaniment Stems

import { STEM_MANIFEST } from '../dsp/song-renderer.js';

function createGroupPanel(title, subtitle, stems, store, mixer) {
  const wrap = document.createElement('div');
  wrap.className = 'card-panel';
  wrap.innerHTML = `
    <div class="mixer-group-header">
      <span class="mixer-group-title">${title}</span>
      <span class="status-hint">${subtitle}</span>
    </div>
    <div class="stem-list"></div>
  `;

  const listEl = wrap.querySelector('.stem-list');
  const rowRefs = new Map();

  for (const m of stems) {
    const row = document.createElement('div');
    row.className = 'stem-row';
    row.style.borderLeftColor = m.color;
    row.innerHTML = `
      <span class="stem-num">${m.num}</span>
      <div class="stem-info">
        <div class="stem-name">${m.name}</div>
        <div class="stem-formula">${m.formula}</div>
      </div>
      <div class="stem-btns">
        <button class="btn-mini" data-action="solo" title="Solo 独奏此轨">S</button>
        <button class="btn-mini" data-action="mute" title="Mute 静音此轨">M</button>
      </div>
      <input class="stem-fader" type="range" min="-36" max="6" step="0.5" value="0" />
      <span class="stem-db">0.0 dB</span>
    `;

    const btnSolo = row.querySelector('[data-action="solo"]');
    const btnMute = row.querySelector('[data-action="mute"]');
    const fader = row.querySelector('.stem-fader');
    const dbText = row.querySelector('.stem-db');

    btnSolo.addEventListener('click', () => {
      const cur = store.getState().stemStates[m.id];
      store.updateStem(m.id, { solo: !cur.solo });
      mixer.syncGainsFromStore();
    });

    btnMute.addEventListener('click', () => {
      const cur = store.getState().stemStates[m.id];
      store.updateStem(m.id, { muted: !cur.muted });
      mixer.syncGainsFromStore();
    });

    fader.addEventListener('input', () => {
      const db = Number(fader.value);
      store.updateStem(m.id, { gainDb: db });
      mixer.syncGainsFromStore();
    });

    rowRefs.set(m.id, { btnSolo, btnMute, fader, dbText });
    listEl.appendChild(row);
  }

  store.subscribe((state, keys) => {
    if (!keys.includes('stemStates')) return;
    for (const m of stems) {
      const st = state.stemStates[m.id];
      const refs = rowRefs.get(m.id);
      if (!st || !refs) continue;
      refs.btnSolo.classList.toggle('solo-on', st.solo);
      refs.btnMute.classList.toggle('mute-on', st.muted);
      refs.fader.value = String(st.gainDb);
      refs.dbText.textContent = st.gainDb <= -35.9 ? '-∞ dB' : `${st.gainDb >= 0 ? '+' : ''}${st.gainDb.toFixed(1)} dB`;
    }
  });

  return wrap;
}

export function mountStemMixer(container, store, mixer) {
  container.innerHTML = '';
  const vocalStems = STEM_MANIFEST.filter((m) => m.group === 'vocal');
  const bandStems = STEM_MANIFEST.filter((m) => m.group === 'band');

  const leftPanel = createGroupPanel(
    '人声与空间物理方程组 (01 – 05)',
    '40-ERB 李宇春声道核 + 24/10-PCA 微表情 + 希尔伯特最小相位辅音',
    vocalStems,
    store,
    mixer
  );
  const rightPanel = createGroupPanel(
    '原曲级高细节纯数学乐队伴奏组 (06 – 10)',
    '100fps 扫频底鼓/木鱼拍手 + 64 原型金属镲 + 18 阶弦振滑音贝斯 + 56 音符复调前奏 Hook 与 80fps 扫弦',
    bandStems,
    store,
    mixer
  );

  container.appendChild(leftPanel);
  container.appendChild(rightPanel);
}
