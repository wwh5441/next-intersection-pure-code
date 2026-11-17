// Left Panel: Real-Time Multi-Stem Opus Console (Solo / Mute / Faders / Scenes / VU Meters)
// Right Panel: WEC0 / WEC1 / WEC2 Architecture Matrix + Live 50Hz TIMB Vocal Formant Voiceprint HUD

import { formatByteSize } from './header-metrics.js';
import { midiCentsToNoteName } from '../../parsers/wec-plaintext-decompiler.js';

function getScenesForBundle(bundle) {
  const parsed = bundle?.parsed;
  if (!parsed) return [];
  const spatScenes = parsed.decodedFeatures?.SPAT?.scenes;
  if (Array.isArray(spatScenes) && spatScenes.length > 0) {
    return spatScenes;
  }
  const n = parsed.tracks.length;
  if (n <= 1) {
    return [{ id: 'master', name: '单流直通原声 (WEC0 Pure Minimal)', mute: [], solo: [] }];
  }
  if (n === 2) {
    return [
      { id: 'master', name: '双轨全开母带 (Vocal + Accomp)', mute: [], solo: [] },
      { id: 'ktv', name: 'KTV 一键消原唱 (纯伴奏)', mute: [0], solo: [] },
      { id: 'acapella', name: 'Acapella 一键提取人声干声', mute: [], solo: [0] },
    ];
  }
  return [
    { id: 'master', name: '四轨全开母带 (4 Stems)', mute: [], solo: [] },
    { id: 'ktv', name: 'KTV 消原唱纯伴奏', mute: [0], solo: [] },
    { id: 'acapella', name: '纯人声清唱 (Vocals Solo)', mute: [], solo: [0] },
    { id: 'rhythm', name: '鼓 + 贝斯律动组 (Drums + Bass)', mute: [], solo: [1, 2] },
    { id: 'harmony', name: '吉他键盘和声组 (Harmony Solo)', mute: [], solo: [3] },
  ];
}

export function mountCompressionLadder(container, store, mixer, onSelectTier, getActiveBundle) {
  container.innerHTML = '';

  const leftCard = document.createElement('div');
  leftCard.className = 'card-panel';

  const rightCard = document.createElement('div');
  rightCard.className = 'card-panel';

  leftCard.innerHTML = `
    <div class="mixer-group-header">
      <span class="mixer-group-title" id="stem-console-title">WEC 多轨锁相 Opus 实时调音台 (Solo / Mute / KTV 消原唱)</span>
      <span class="status-hint" id="stem-console-sub">15ms 无缝切换分轨</span>
    </div>
    <div class="scene-preset-strip" id="scene-preset-strip"></div>
    <div class="monitor-controls-stack" id="dynamic-stems-stack"></div>
    <hr style="border:none;border-top:1px solid #e2e8f0;margin:10px 0;" />
    <div class="monitor-controls-stack">
      <div class="stem-row" style="border-left-color: #0f172a;">
        <div class="stem-info">
          <div class="stem-name"><span class="stem-num">MST</span> 母带总输出增益 & 立体声宽窄</div>
        </div>
        <div class="monitor-fader-line">
          <input id="fader-master-gain" class="stem-fader" type="range" min="-24" max="6" step="0.5" value="0" />
          <span id="label-master-gain" class="stem-db">0.0 dB</span>
          <input id="fader-stereo-width" class="stem-fader" type="range" min="0" max="150" step="5" value="100" style="max-width:95px;" />
          <span id="label-stereo-width" class="stem-db">100%</span>
        </div>
      </div>
    </div>
  `;

  rightCard.innerHTML = `
    <div class="mixer-group-header">
      <span class="mixer-group-title">WEC0 / WEC1 / WEC2 三档体系实测矩阵 & TIMB 声纹透视</span>
      <span class="status-hint">点击任意行瞬时切换 .wec 容器</span>
    </div>
    <div class="rd-table-wrap">
      <table class="rd-table">
        <thead>
          <tr>
            <th>档位</th>
            <th>版本名称</th>
            <th>轨+层</th>
            <th><strong>.wec 体积</strong></th>
            <th>总码率 · CRC</th>
          </tr>
        </thead>
        <tbody id="rd-table-tbody"></tbody>
      </table>
    </div>
    <div class="timb-inspector-box" id="timb-live-box">
      <div class="timb-header-line">
        <span id="timb-box-title"><strong>TIMB 50Hz 人声音色声纹层</strong> (免解码直读)</span>
        <span class="timb-pitch-readout" id="timb-pitch-text">等待播放...</span>
      </div>
      <div class="timb-bands-grid">
        <div class="timb-band-col">
          <span>胸腔基频 (140–420Hz)</span>
          <div class="timb-bar-bg"><div class="timb-bar-fg" id="tb-chest" style="background:#38bdf8;"></div></div>
        </div>
        <div class="timb-band-col">
          <span>咽腔元音 F1 (0.47–1.2k)</span>
          <div class="timb-bar-bg"><div class="timb-bar-fg" id="tb-f1" style="background:#34d399;"></div></div>
        </div>
        <div class="timb-band-col">
          <span>歌手共振峰 F2 (1.2–3.2k)</span>
          <div class="timb-bar-bg"><div class="timb-bar-fg" id="tb-f2" style="background:#fbbf24;"></div></div>
        </div>
        <div class="timb-band-col">
          <span>气声齿音 Air (3.2–9.4k)</span>
          <div class="timb-bar-bg"><div class="timb-bar-fg" id="tb-air" style="background:#fb7185;"></div></div>
        </div>
      </div>
    </div>
  `;

  container.appendChild(leftCard);
  container.appendChild(rightCard);

  const consoleTitle = leftCard.querySelector('#stem-console-title');
  const consoleSub = leftCard.querySelector('#stem-console-sub');
  const sceneStrip = leftCard.querySelector('#scene-preset-strip');
  const stemsStack = leftCard.querySelector('#dynamic-stems-stack');
  const faderGain = leftCard.querySelector('#fader-master-gain');
  const labelGain = leftCard.querySelector('#label-master-gain');
  const faderWidth = leftCard.querySelector('#fader-stereo-width');
  const labelWidth = leftCard.querySelector('#label-stereo-width');

  const tbody = rightCard.querySelector('#rd-table-tbody');
  const timbTitle = rightCard.querySelector('#timb-box-title');
  const timbPitch = rightCard.querySelector('#timb-pitch-text');
  const barChest = rightCard.querySelector('#tb-chest');
  const barF1 = rightCard.querySelector('#tb-f1');
  const barF2 = rightCard.querySelector('#tb-f2');
  const barAir = rightCard.querySelector('#tb-air');

  let vuFills = [];
  let renderedTierKey = '';

  faderGain.addEventListener('input', () => {
    store.update({ masterGainDb: Number(faderGain.value) });
    mixer.syncMixFromStore();
  });

  faderWidth.addEventListener('input', () => {
    store.update({ stereoWidthPct: Number(faderWidth.value) });
    mixer.syncMixFromStore();
  });

  function rebuildStemStrips(state) {
    const bundle = getActiveBundle ? getActiveBundle() : null;
    const parsed = bundle?.parsed;
    stemsStack.innerHTML = '';
    sceneStrip.innerHTML = '';
    vuFills = [];

    if (!parsed) return;

    consoleTitle.textContent =
      parsed.magic === 'WEC0'
        ? 'WEC0 纯最小单流监听台 (1 Track · 零分轨开销)'
        : parsed.magic === 'WEC1'
        ? 'WEC1 音色+伴奏双轨解耦调音台 (2 Tracks · 支持一键消原唱)'
        : `WEC2 全分轨母版调音台 (${parsed.tracks.length} 独立 Opus 音轨 · 上不封顶架构)`;
    consoleSub.textContent = `CRC32 ${parsed.storedCrcHex} · 全部音轨严格 48kHz 采样级相位对齐`;

    const scenes = getScenesForBundle(bundle);
    for (const sc of scenes) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `scene-btn${state.activeSceneId === sc.id ? ' active' : ''}`;
      btn.dataset.sceneId = sc.id;
      btn.textContent = sc.name;
      btn.addEventListener('click', () => {
        store.applyScenePreset(sc.id, sc.mute || [], sc.solo || []);
        mixer.syncMixFromStore();
      });
      sceneStrip.appendChild(btn);
    }

    const anySolo = state.stemStates.some((s) => s && s.solo);

    parsed.tracks.forEach((tr, ti) => {
      const stCtrl = state.stemStates[ti] || { gainDb: 0, muted: false, solo: false };
      const isAudible = anySolo ? stCtrl.solo : !stCtrl.muted;
      const actualKbps = (((tr.audioByteLen * 8) / parsed.durationSec) / 1000).toFixed(1);

      const card = document.createElement('div');
      card.className = `stem-strip-card${isAudible ? '' : ' dimmed'}`;
      card.style.borderLeftColor = tr.color || '#0284c7';
      card.dataset.trackIdx = String(ti);

      card.innerHTML = `
        <div class="stem-strip-top">
          <div class="stem-strip-title">
            <span class="stem-num">T${ti}</span>
            <span>${tr.roleZh}</span>
            <span class="status-hint">(${tr.channels === 1 ? 'Mono' : 'Stereo'} · ${actualKbps} kbps · ${formatByteSize(tr.audioByteLen)})</span>
          </div>
          <div class="stem-ms-btns">
            <button type="button" class="btn-ms btn-mute${stCtrl.muted ? ' mute-on' : ''}" title="静音该轨 (Mute)">M</button>
            <button type="button" class="btn-ms btn-solo${stCtrl.solo ? ' solo-on' : ''}" title="独奏该轨 (Solo)">S</button>
          </div>
        </div>
        <div class="stem-strip-fader-row">
          <input type="range" class="stem-gain-range" min="-36" max="6" step="0.5" value="${stCtrl.gainDb}" />
          <span class="stem-db stem-gain-lbl">${stCtrl.gainDb >= 0 ? '+' : ''}${stCtrl.gainDb.toFixed(1)} dB</span>
          <div class="stem-vu-track"><div class="stem-vu-fill"></div></div>
        </div>
      `;

      const btnM = card.querySelector('.btn-mute');
      const btnS = card.querySelector('.btn-solo');
      const fader = card.querySelector('.stem-gain-range');
      const lbl = card.querySelector('.stem-gain-lbl');
      const vu = card.querySelector('.stem-vu-fill');
      vuFills.push(vu);

      if (parsed.tracks.length === 1) {
        btnS.disabled = true;
      }

      btnM.addEventListener('click', () => {
        const cur = store.getState().stemStates[ti];
        store.setStemPatch(ti, { muted: !cur.muted, solo: false });
        mixer.syncMixFromStore();
      });

      btnS.addEventListener('click', () => {
        const cur = store.getState().stemStates[ti];
        store.setStemPatch(ti, { solo: !cur.solo, muted: false });
        mixer.syncMixFromStore();
      });

      fader.addEventListener('input', () => {
        const val = Number(fader.value);
        lbl.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(1)} dB`;
        store.setStemPatch(ti, { gainDb: val });
        mixer.syncMixFromStore();
      });

      stemsStack.appendChild(card);
    });
  }

  function syncStemButtonsWithoutRebuild(state) {
    sceneStrip.querySelectorAll('[data-scene-id]').forEach((b) => {
      b.classList.toggle('active', b.dataset.sceneId === state.activeSceneId);
    });
    const anySolo = state.stemStates.some((s) => s && s.solo);
    stemsStack.querySelectorAll('.stem-strip-card').forEach((card) => {
      const ti = Number(card.dataset.trackIdx);
      const ctrl = state.stemStates[ti];
      if (!ctrl) return;
      const audible = anySolo ? ctrl.solo : !ctrl.muted;
      card.classList.toggle('dimmed', !audible);
      card.querySelector('.btn-mute')?.classList.toggle('mute-on', ctrl.muted);
      card.querySelector('.btn-solo')?.classList.toggle('solo-on', ctrl.solo);
      const fader = card.querySelector('.stem-gain-range');
      const lbl = card.querySelector('.stem-gain-lbl');
      if (fader && Number(fader.value) !== ctrl.gainDb) {
        fader.value = String(ctrl.gainDb);
      }
      if (lbl) {
        lbl.textContent = `${ctrl.gainDb >= 0 ? '+' : ''}${ctrl.gainDb.toFixed(1)} dB`;
      }
    });
  }

  function renderTable(state) {
    tbody.innerHTML = '';
    for (const r of state.tiersCatalog) {
      const tr = document.createElement('tr');
      if (r.tierId === state.activeTierId) tr.classList.add('active-row');
      if (onSelectTier) {
        tr.classList.add('clickable-row');
        tr.addEventListener('click', () => onSelectTier(r.tierId));
      }
      const featText = `${r.audioTracks}+${r.featureLayers}`;
      tr.innerHTML = `
        <td class="mono-num"><span class="family-tag ${r.family}">${r.family}</span></td>
        <td>${r.name}</td>
        <td class="mono-num" title="${(r.featureTags || []).join(' + ')}">${featText}</td>
        <td class="mono-num"><strong>${formatByteSize(r.wecBytes)}</strong></td>
        <td class="mono-num">${r.equivKbps}k · ${r.crcHex.slice(2, 6)}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  function updateMetersAndTimb(state) {
    faderGain.value = String(state.masterGainDb);
    labelGain.textContent = `${state.masterGainDb >= 0 ? '+' : ''}${state.masterGainDb.toFixed(1)} dB`;
    faderWidth.value = String(state.stereoWidthPct);
    labelWidth.textContent = `${state.stereoWidthPct}%`;

    const bundle = getActiveBundle ? getActiveBundle() : null;
    const engine = bundle?.engine;
    const parsed = bundle?.parsed;

    if (engine && engine.liveLevelsDb) {
      for (let i = 0; i < vuFills.length; i++) {
        const db = state.isPlaying ? engine.liveLevelsDb[i] : -90.0;
        const pct = Math.max(0, Math.min(100, ((db + 56) / 56) * 100));
        vuFills[i].style.width = `${pct.toFixed(1)}%`;
      }
    }

    const timb = parsed?.decodedFeatures?.TIMB;
    if (!timb) {
      timbTitle.innerHTML = '<strong>TIMB 人声音色特征层</strong> · 当前 WEC0 纯最小档未封装该层 (零额外字节开销)';
      timbPitch.textContent = 'WEC0: 0 Feature Layers';
      barChest.style.width = '0%';
      barF1.style.width = '0%';
      barF2.style.width = '0%';
      barAir.style.width = '0%';
      return;
    }

    timbTitle.innerHTML = `<strong>TIMB 50Hz 人声音色声纹层</strong> · 平均基频 ${timb.meanF0Hz}Hz · 频谱质心 ${timb.centroidHz}Hz`;
    const k = Math.max(0, Math.min(timb.frameCount - 1, Math.floor(state.currentTime / 0.02)));
    const cents = timb.f0Cents[k];
    const vByte = timb.voicing[k];
    const sib = ((vByte >>> 1) & 1) === 1;
    timbPitch.textContent = `${midiCentsToNoteName(cents)}${sib ? ' · [齿音/气音瞬态]' : ''}`;
    barChest.style.width = `${Math.round((timb.chest[k] / 255) * 100)}%`;
    barF1.style.width = `${Math.round((timb.f1[k] / 255) * 100)}%`;
    barF2.style.width = `${Math.round((timb.f2[k] / 255) * 100)}%`;
    barAir.style.width = `${Math.round((timb.air[k] / 255) * 100)}%`;
  }

  renderTable(store.getState());
  updateMetersAndTimb(store.getState());

  store.subscribe((state, keys) => {
    const tierKey = `${state.activeTierId}|${state.isReady}`;
    if (tierKey !== renderedTierKey && state.isReady) {
      renderedTierKey = tierKey;
      rebuildStemStrips(state);
      renderTable(state);
    } else if (keys.includes('stemStates') || keys.includes('activeSceneId')) {
      syncStemButtonsWithoutRebuild(state);
    } else if (keys.includes('activeTierId')) {
      renderTable(state);
    }
    if (
      keys.includes('currentTime') ||
      keys.includes('isPlaying') ||
      keys.includes('masterGainDb') ||
      keys.includes('stereoWidthPct') ||
      keys.includes('activeTierId') ||
      keys.includes('isReady')
    ) {
      updateMetersAndTimb(state);
    }
  });
}
