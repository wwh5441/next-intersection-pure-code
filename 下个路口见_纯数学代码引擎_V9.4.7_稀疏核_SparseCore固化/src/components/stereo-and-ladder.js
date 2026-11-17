// Two-Column Panel: 7-Tier Rate-Distortion Comparison Table + 48kHz Master Stereo Field Monitor

import { formatByteSize } from './header-metrics.js';

export function mountStereoAndLadder(container, store, mixer, onSelectTier = null) {
  container.innerHTML = '';

  const leftCard = document.createElement('div');
  leftCard.className = 'card-panel';

  const rightCard = document.createElement('div');
  rightCard.className = 'card-panel';

  leftCard.innerHTML = `
    <div class="mixer-group-header">
      <span class="mixer-group-title" id="ladder-panel-title">七级纯数学率失真梯度规格表</span>
      <span class="status-hint">100% 纯明文十进制安全整数 · 零外部音频</span>
    </div>
    <div class="rd-table-wrap">
      <table class="rd-table">
        <thead>
          <tr>
            <th>档位</th>
            <th>目标场景</th>
            <th>有效频宽</th>
            <th>非零模态数</th>
            <th>纯明文 JS</th>
            <th>Gzip 体积</th>
            <th>Wave / Mel SNR</th>
          </tr>
        </thead>
        <tbody id="rd-table-tbody"></tbody>
      </table>
    </div>
  `;

  rightCard.innerHTML = `
    <div class="mixer-group-header">
      <span class="mixer-group-title">48kHz 母带声场监听与增益控制器 (交付播放态)</span>
      <span class="status-hint">对标 WAV 双声道母带直出 · 0.38 MB 流式内存</span>
    </div>
    <div class="monitor-controls-stack">
      <div class="stem-row" style="border-left-color: #e05338;">
        <div class="stem-info">
          <div class="stem-name"><span class="stem-num">M/S</span> 母带总输出增益 (Master Output Gain)</div>
          <div class="stem-formula">L(n) = G · [M(n) + w · S(n)],  R(n) = G · [M(n) - w · S(n)]</div>
        </div>
        <div class="monitor-fader-line">
          <input id="fader-master-gain" class="stem-fader" type="range" min="-24" max="6" step="0.5" value="0" />
          <span id="label-master-gain" class="stem-db">0.0 dB</span>
        </div>
      </div>

      <div class="stem-row" style="border-left-color: #2b6cb0;">
        <div class="stem-info">
          <div class="stem-name"><span class="stem-num">WID</span> 立体声空间宽度 (Stereo Field Width w)</div>
          <div class="stem-formula">0% = 纯单声道中置合拢 · 100% = 原版母带声场 · 150% = 宽体环绕</div>
        </div>
        <div class="monitor-fader-line">
          <input id="fader-stereo-width" class="stem-fader" type="range" min="0" max="150" step="5" value="100" />
          <span id="label-stereo-width" class="stem-db">100%</span>
        </div>
      </div>

      <div class="stem-row" style="border-left-color: #2a9d8f;">
        <div class="stem-info">
          <div class="stem-name"><span class="stem-num">ARCH</span> <span id="spec-active-headline">当前档位编码策略</span></div>
          <div class="stem-formula" id="spec-active-detail">--</div>
        </div>
        <div class="monitor-fader-line">
          <span class="status-hint">等效 Gzip 流媒体码率</span>
          <span class="stem-db" id="spec-active-bitrate">--</span>
        </div>
      </div>
    </div>
  `;

  container.appendChild(leftCard);
  container.appendChild(rightCard);

  const tbody = leftCard.querySelector('#rd-table-tbody');
  const faderGain = rightCard.querySelector('#fader-master-gain');
  const labelGain = rightCard.querySelector('#label-master-gain');
  const faderWidth = rightCard.querySelector('#fader-stereo-width');
  const labelWidth = rightCard.querySelector('#label-stereo-width');
  const specHead = rightCard.querySelector('#spec-active-headline');
  const specDetail = rightCard.querySelector('#spec-active-detail');
  const specBitrate = rightCard.querySelector('#spec-active-bitrate');

  faderGain.addEventListener('input', () => {
    store.update({ masterGainDb: Number(faderGain.value) });
    mixer.syncMixFromStore();
  });

  faderWidth.addEventListener('input', () => {
    store.update({ stereoWidthPct: Number(faderWidth.value) });
    mixer.syncMixFromStore();
  });

  function renderTable(state) {
    const rows =
      state.tiersCatalog && state.tiersCatalog.length > 0
        ? state.tiersCatalog
        : state.activeTierMeta
          ? [state.activeTierMeta]
          : [];

    tbody.innerHTML = '';
    for (const r of rows) {
      const tr = document.createElement('tr');
      const isAct = r.tierId === state.activeTierId;
      if (isAct) tr.classList.add('active-row');
      if (state.studioMode === 'unified' && onSelectTier) {
        tr.classList.add('clickable-row');
        tr.addEventListener('click', () => onSelectTier(r.tierId));
      }
      tr.innerHTML = `
        <td class="mono-num">${r.versionCode} ${r.tierName}</td>
        <td>${r.targetDeviceDesc}</td>
        <td class="mono-num">${r.passbandLabel}</td>
        <td class="mono-num">${r.nonZeroModes.toLocaleString()}</td>
        <td class="mono-num">${formatByteSize(r.totalAudioCorePlainJsBytes)}</td>
        <td class="mono-num">${formatByteSize(r.totalAudioCoreGzipBytes)}</td>
        <td class="mono-num">${r.waveSnrDb.toFixed(1)} / ${r.melSnrDb.toFixed(1)} dB</td>
      `;
      tbody.appendChild(tr);
    }
  }

  function syncRightPanel(state) {
    faderGain.value = String(state.masterGainDb);
    labelGain.textContent = `${state.masterGainDb >= 0 ? '+' : ''}${state.masterGainDb.toFixed(1)} dB`;
    faderWidth.value = String(state.stereoWidthPct);
    labelWidth.textContent = `${state.stereoWidthPct}%`;

    const meta = state.activeTierMeta;
    if (meta) {
      specHead.textContent = `${meta.versionCode}【${meta.tierName}】· ${meta.targetDeviceDesc}`;
      specDetail.textContent = meta.codingStrategyDetail;
      const eqKbps = ((meta.totalAudioCoreGzipBytes * 8) / (210.3733 * 1000)).toFixed(1);
      specBitrate.textContent = `${eqKbps} kbps`;
    }
  }

  renderTable(store.getState());
  syncRightPanel(store.getState());

  store.subscribe((state, keys) => {
    if (
      keys.includes('activeTierId') ||
      keys.includes('tiersCatalog') ||
      keys.includes('activeTierMeta')
    ) {
      renderTable(state);
      syncRightPanel(state);
    }
    if (keys.includes('masterGainDb') || keys.includes('stereoWidthPct')) {
      syncRightPanel(state);
    }
  });
}
