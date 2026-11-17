// WEC0 / WEC1 / WEC2 Family Switcher + 7 Native Opus .wec Cards + Local Drag-and-Drop Loader

import { formatByteSize } from './header-metrics.js';

export function mountWecDropzoneBar(container, store, onSelectTier, onLocalFileSelected) {
  const initSt = store.getState();

  container.innerHTML = `
    <div class="family-filter-bar">
      <div class="family-pills-group" id="family-pills-group">
        <button class="family-pill-btn active" data-fam="ALL" type="button">全部 7 款包 (WEC0 / WEC1 / WEC2)</button>
        <button class="family-pill-btn" data-fam="WEC0" type="button">WEC0 · 纯最小核 (1 轨 + 0 层)</button>
        <button class="family-pill-btn" data-fam="WEC1" type="button">WEC1 · 音色+伴奏两轨 (2 轨 + TIMB)</button>
        <button class="family-pill-btn" data-fam="WEC2" type="button">WEC2 · 全分轨母版 (≥4+3 层 · 上不封顶)</button>
      </div>
      <span id="tier-switch-hint" class="status-hint">RFC 6716 Opus 原生 C++ 内核解码 · 15ms 锁相无缝切换</span>
    </div>
    <div class="tier-cards-grid" id="tier-cards-grid"></div>
    <div class="wec-drop-banner" id="wec-dropzone">
      <div class="wec-drop-info">
        <div class="wec-drop-title">本地任意 WEC0 / WEC1 / WEC2 (.wec) 拖拽校验与分轨直播舱</div>
        <div class="wec-drop-sub" id="wec-drop-sub">支持拖入任意 WEC-CLI 或工作台导出的 .wec 文件，自动校验 WEC0/WEC1/WEC2 魔数、4+3 层级契约与 CRC-32 完整性</div>
      </div>
      <div class="wec-drop-actions">
        <span class="wec-magic-pill" id="drop-magic-badge">Magic: WEC2</span>
        <span class="wec-crc-pill" id="drop-crc-badge">CRC-32 就绪</span>
        <input id="wec-file-input" type="file" accept=".wec,application/octet-stream" hidden />
        <button id="btn-pick-wec" class="btn-export-secondary" type="button">选择本地 .wec 文件播放</button>
      </div>
    </div>
  `;

  const famGroup = container.querySelector('#family-pills-group');
  const gridEl = container.querySelector('#tier-cards-grid');
  const hintEl = container.querySelector('#tier-switch-hint');
  const dropzone = container.querySelector('#wec-dropzone');
  const dropSub = container.querySelector('#wec-drop-sub');
  const magicBadge = container.querySelector('#drop-magic-badge');
  const crcBadge = container.querySelector('#drop-crc-badge');
  const fileInput = container.querySelector('#wec-file-input');
  const btnPick = container.querySelector('#btn-pick-wec');
  const btnMap = new Map();

  famGroup.querySelectorAll('[data-fam]').forEach((b) => {
    b.addEventListener('click', () => {
      store.update({ familyFilter: b.dataset.fam });
    });
  });

  for (const item of initSt.tiersCatalog) {
    const btn = document.createElement('button');
    btn.className = 'tier-card-btn';
    btn.dataset.tierId = item.tierId;
    btn.dataset.family = item.family;
    const layerText = `${item.audioTracks}轨+${item.featureLayers}层`;
    btn.innerHTML = `
      <div class="tier-card-top">
        <span class="family-tag ${item.family}">${item.family}</span>
        <span>${layerText}</span>
      </div>
      <div class="tier-card-name">${item.name}</div>
      <div class="tier-card-size">${formatByteSize(item.wecBytes)} · ${item.equivKbps}k</div>
      <div class="tier-card-desc">${item.badge}</div>
    `;
    btn.addEventListener('click', () => onSelectTier && onSelectTier(item.tierId));
    btnMap.set(item.tierId, btn);
    gridEl.appendChild(btn);
  }

  btnPick.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0] && onLocalFileSelected) {
      onLocalFileSelected(fileInput.files[0]);
      fileInput.value = '';
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && onLocalFileSelected) {
      onLocalFileSelected(f);
    }
  });

  function syncUi(state) {
    const busy = Boolean(state.loadingTierId);
    famGroup.querySelectorAll('[data-fam]').forEach((b) => {
      b.classList.toggle('active', b.dataset.fam === state.familyFilter);
    });

    for (const [tid, btn] of btnMap.entries()) {
      btn.classList.toggle('active', tid === state.activeTierId);
      btn.disabled = busy;
      const fam = btn.dataset.family;
      const hide = state.familyFilter !== 'ALL' && fam !== state.familyFilter;
      btn.classList.toggle('hidden-by-filter', hide);
    }

    const meta = state.activeTierMeta;
    if (meta) {
      magicBadge.textContent = `Magic: ${meta.magic} (${meta.audioTracks}+${meta.featureLayers})`;
      crcBadge.textContent = `CRC32: ${meta.crcHex} ✓`;
    }
    if (state.errorMsg) {
      dropSub.textContent = `容器校验拒绝：${state.errorMsg}`;
    } else if (state.isCustomFile && meta) {
      dropSub.textContent = `正在直播本地文件：${meta.wecFilename} (${formatByteSize(meta.wecBytes)}, ${meta.magic} 校验通过)`;
    }
    if (state.loadingTierId) {
      hintEl.textContent = `正在拉取并由 C++ libopus 解码 (${state.loadingTierId})...`;
    } else {
      hintEl.textContent = `已拉取 ${state.loadedTierIds.length} / ${state.tiersCatalog.length} 款 .wec · 支持同位置秒切对比`;
    }
  }

  syncUi(store.getState());
  store.subscribe((state, keys) => {
    if (
      keys.includes('activeTierId') ||
      keys.includes('activeTierMeta') ||
      keys.includes('familyFilter') ||
      keys.includes('loadingTierId') ||
      keys.includes('loadedTierIds') ||
      keys.includes('isCustomFile') ||
      keys.includes('errorMsg')
    ) {
      syncUi(state);
    }
  });
}
