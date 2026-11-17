// WEC v2.0 Transport Bar + 3-Way Exporter (.wec Binary / Container+Packet JSON / Current Stem Mix 24-Bit WAV)

import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';
import { formatByteSize } from './header-metrics.js';

function formatClock(sec) {
  const s = Math.max(0, sec);
  const mins = Math.floor(s / 60);
  const rem = s - mins * 60;
  return `${String(mins).padStart(2, '0')}:${rem.toFixed(1).padStart(4, '0')}`;
}

function formatBarBeat(sec, bpm, bar0Sec) {
  if (sec < bar0Sec) {
    const pickupBeat = Math.max(1, Math.min(2, Math.floor((sec - 0.372) / (60 / bpm) + 1)));
    return `Pickup · Beat ${pickupBeat} | ${bpm.toFixed(1)} BPM`;
  }
  const beatDur = 60.0 / bpm;
  const totalBeats = (sec - bar0Sec) / beatDur;
  const bar = Math.floor(totalBeats / 4) + 1;
  const beat = (Math.floor(totalBeats) % 4) + 1;
  return `Bar ${String(bar).padStart(2, '0')} : Beat ${beat} | ${bpm.toFixed(1)} BPM`;
}

function findCurrentLyric(sec, customLines) {
  const lines = customLines && customLines.length > 0 ? customLines : VOCAL_LYRIC_LINES;
  let active = '♪ 98.0 BPM 原声木吉他扫弦 + 底鼓律动前奏 (支持下方多轨独立 Solo / Mute / KTV 消原唱)';
  for (let i = 0; i < lines.length; i++) {
    const [tStartMs, text] = lines[i];
    if (sec + 0.08 >= tStartMs * 0.001) {
      active = text;
    } else {
      break;
    }
  }
  return active;
}

export function mountTransportBar(container, store, mixer, getActiveBundle) {
  const initSt = store.getState();

  container.innerHTML = `
    <div class="transport-wrap">
      <div class="tier-switch-bar">
        <div class="tier-switch-group">
          <span class="tier-switch-label">当前 WEC v2 解码链路：</span>
          <span id="active-tier-formula-tag" class="tier-cache-tag">WEC 裸包索引 (u8_ext) ➔ RFC 6716 Opus 原生解码 ➔ 锁相多轨实时混音台</span>
        </div>
        <div class="wec-drop-actions">
          <button id="btn-dl-wec" class="btn-export-secondary" disabled>下载当前 .wec</button>
          <button id="btn-dl-json" class="btn-export-secondary" disabled>导出目录与当前包 .json</button>
          <button id="btn-export-wav" class="btn-tier" disabled>导出当前分轨混音 24-bit WAV</button>
        </div>
      </div>

      <div class="transport-top">
        <div class="transport-controls">
          <button id="btn-play-toggle" class="btn-primary" disabled>正在挂载 Opus 引擎...</button>
          <button class="btn-chip active" data-ch="stereo">完整立体声 (L+R)</button>
          <button class="btn-chip" data-ch="mid">仅中置声像 (Mid)</button>
          <button class="btn-chip" data-ch="side">仅侧向声场 (Side)</button>
          <button id="btn-reset-monitor" class="btn-chip">复位全部分轨与声场</button>
        </div>
        <div class="clock-box">
          <span id="bar-beat-readout" class="bar-beat-tag">Pickup · Beat 1 | 98.0 BPM</span>
          <span id="time-readout">00:00.0 / 03:30.4</span>
        </div>
      </div>

      <input id="seek-slider" class="seek-slider" type="range" min="0" max="${initSt.duration.toFixed(3)}" step="0.02" value="0" disabled />

      <div class="lyric-banner">
        <div id="lyric-line-display" class="lyric-current">正在解析 WEC v2 容器目录与 Opus 分轨流...</div>
        <div id="synth-status-hint" class="status-hint">Ready</div>
      </div>
    </div>
  `;

  const btnPlay = container.querySelector('#btn-play-toggle');
  const btnDlWec = container.querySelector('#btn-dl-wec');
  const btnDlJson = container.querySelector('#btn-dl-json');
  const btnExport = container.querySelector('#btn-export-wav');
  const btnReset = container.querySelector('#btn-reset-monitor');
  const chBtns = Array.from(container.querySelectorAll('[data-ch]'));
  const slider = container.querySelector('#seek-slider');
  const barTag = container.querySelector('#bar-beat-readout');
  const timeTag = container.querySelector('#time-readout');
  const lyricEl = container.querySelector('#lyric-line-display');
  const statusEl = container.querySelector('#synth-status-hint');
  const formulaTag = container.querySelector('#active-tier-formula-tag');

  let isDraggingSeek = false;

  btnPlay.addEventListener('click', () => mixer.togglePlay());
  btnDlWec.addEventListener('click', () => mixer.downloadRawWecFile());
  btnDlJson.addEventListener('click', () => mixer.exportPlaintextJsonSnapshot());
  btnExport.addEventListener('click', () => mixer.exportCurrentMixWav());

  chBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      store.update({ channelMode: btn.dataset.ch });
      mixer.syncMixFromStore();
    });
  });

  btnReset.addEventListener('click', () => {
    store.resetMonitorControls();
    mixer.syncMixFromStore();
  });

  function getLyricLines() {
    const bundle = getActiveBundle ? getActiveBundle() : null;
    return bundle?.parsed?.decodedFeatures?.LYRC?.lines || VOCAL_LYRIC_LINES;
  }

  slider.addEventListener('pointerdown', () => { isDraggingSeek = true; });
  slider.addEventListener('input', () => {
    const v = Number(slider.value);
    timeTag.textContent = `${formatClock(v)} / ${formatClock(store.getState().duration)}`;
    barTag.textContent = formatBarBeat(v, store.getState().bpm, store.getState().bar0Sec);
    lyricEl.textContent = findCurrentLyric(v, getLyricLines());
  });
  slider.addEventListener('change', () => {
    isDraggingSeek = false;
    mixer.seek(Number(slider.value));
  });

  function refreshUi(state) {
    btnPlay.disabled = !state.isReady;
    btnDlWec.disabled = !state.isReady;
    btnDlJson.disabled = !state.isReady;
    btnExport.disabled = !state.isReady || state.isExportingWav;

    const meta = state.activeTierMeta;
    const vTag = meta ? `${meta.magic} · ${meta.name}` : 'WEC';

    if (meta) {
      btnDlWec.textContent = `下载 .wec (${formatByteSize(meta.wecBytes)})`;
      formulaTag.textContent =
        meta.magic === 'WEC0'
          ? 'WEC0 纯最小 32B 头 ➔ 单轨 RFC 6716 Opus 裸包直通解码 (零特征开销)'
          : meta.magic === 'WEC1'
          ? 'WEC1 48B 头 ➔ 音色轨 + 伴奏轨双流相位锁定 Opus ➔ TIMB 50Hz 声纹特征层'
          : `WEC2 母版容器 ➔ ${meta.audioTracks} 轨锁相 Opus 分轨混音 + ${meta.featureLayers} 层结构特征 (${(meta.featureTags || []).join('/')})`;
    }
    btnExport.textContent = state.isExportingWav
      ? '正在按当前推子合成 24-bit WAV...'
      : '导出当前分轨混音 WAV';
    slider.disabled = !state.isReady;
    slider.max = String(state.duration.toFixed(3));

    if (!state.isReady) {
      btnPlay.textContent = state.statusText;
      statusEl.textContent = state.statusText;
    } else {
      btnPlay.textContent = state.isPlaying ? '暂停播放' : `播放 ${vTag}`;
      statusEl.textContent = `${vTag} 就绪 · 解码耗时 ${state.renderTimeMs}ms · 相对 24-bit WAV 压缩 ${meta?.compressionRatioVs24BitWav || '--'}:1`;
    }

    chBtns.forEach((b) => b.classList.toggle('active', b.dataset.ch === state.channelMode));
  }

  refreshUi(store.getState());
  store.subscribe((state, keys) => {
    if (
      keys.includes('isReady') ||
      keys.includes('isPlaying') ||
      keys.includes('isExportingWav') ||
      keys.includes('statusText') ||
      keys.includes('activeTierMeta') ||
      keys.includes('channelMode')
    ) {
      refreshUi(state);
    }
    if (keys.includes('currentTime') || keys.includes('isReady') || keys.includes('activeTierMeta')) {
      if (!isDraggingSeek) slider.value = String(state.currentTime);
      timeTag.textContent = `${formatClock(state.currentTime)} / ${formatClock(state.duration)}`;
      barTag.textContent = formatBarBeat(state.currentTime, state.bpm, state.bar0Sec);
      if (state.isReady) lyricEl.textContent = findCurrentLyric(state.currentTime, getLyricLines());
    }
  });
}
