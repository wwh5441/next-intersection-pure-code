// Studio Transport Bar, Stereo Channel Quick-Switch, 24-Bit WAV Exporter & Musical Clock

import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';

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

function findCurrentLyric(sec) {
  let active = '♪ 48kHz 纯数学正交余弦母带前奏律动 (98.0 BPM 原声木吉他扫弦 + 木鱼底鼓 + 电贝斯滑音)';
  for (let i = 0; i < VOCAL_LYRIC_LINES.length; i++) {
    const [tStartMs, text] = VOCAL_LYRIC_LINES[i];
    if (sec + 0.08 >= tStartMs * 0.001) {
      active = text;
    } else {
      break;
    }
  }
  return active;
}

export function mountTransportBar(container, store, mixer) {
  const initSt = store.getState();

  container.innerHTML = `
    <div class="transport-wrap">
      <div class="tier-switch-bar">
        <div class="tier-switch-group">
          <span class="tier-switch-label">当前纯数学声学方程：</span>
          <span id="active-tier-formula-tag" class="tier-cache-tag">51-Bit Safe-Int + 4-Class Canonical Huffman + TDAC 2-Frame Locality</span>
        </div>
        <button id="btn-export-wav" class="btn-tier" disabled>导出当前档位 48kHz / 24-bit WAV</button>
      </div>

      <div class="transport-top">
        <div class="transport-controls">
          <button id="btn-play-toggle" class="btn-primary" disabled>正在挂载引擎...</button>
          <button class="btn-chip active" data-ch="stereo">完整立体声母带 (L+R)</button>
          <button class="btn-chip" data-ch="mid">仅中置主唱与主干 (Mid)</button>
          <button class="btn-chip" data-ch="side">仅侧向立体声空间 (Side)</button>
          <button id="btn-reset-monitor" class="btn-chip">复位 0dB 标准声场</button>
        </div>
        <div class="clock-box">
          <span id="bar-beat-readout" class="bar-beat-tag">Pickup · Beat 1 | 98.0 BPM</span>
          <span id="time-readout">00:00.0 / 03:30.4</span>
        </div>
      </div>

      <input id="seek-slider" class="seek-slider" type="range" min="0" max="${initSt.duration.toFixed(3)}" step="0.02" value="0" disabled />

      <div class="lyric-banner">
        <div id="lyric-line-display" class="lyric-current">正在初始化 48kHz 纯数学流式引擎...</div>
        <div id="synth-status-hint" class="status-hint">Ready</div>
      </div>
    </div>
  `;

  const btnPlay = container.querySelector('#btn-play-toggle');
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

  slider.addEventListener('pointerdown', () => {
    isDraggingSeek = true;
  });
  slider.addEventListener('input', () => {
    const v = Number(slider.value);
    timeTag.textContent = `${formatClock(v)} / ${formatClock(store.getState().duration)}`;
    barTag.textContent = formatBarBeat(v, store.getState().bpm, store.getState().bar0Sec);
    lyricEl.textContent = findCurrentLyric(v);
  });
  slider.addEventListener('change', () => {
    isDraggingSeek = false;
    mixer.seek(Number(slider.value));
  });

  function refreshUi(state) {
    btnPlay.disabled = !state.isReady;
    btnExport.disabled = !state.isReady || state.isExportingWav;
    const meta = state.activeTierMeta;
    const vTag = meta ? `${meta.versionCode} ${meta.tierName}` : '48kHz';

    btnExport.textContent = state.isExportingWav
      ? '正在离线合成 24-bit WAV...'
      : `导出 ${vTag} 24-bit WAV`;
    slider.disabled = !state.isReady;
    slider.max = String(state.duration.toFixed(3));

    if (meta && meta.formulaSummary) {
      formulaTag.textContent = meta.formulaSummary;
    }

    if (!state.isReady) {
      btnPlay.textContent = state.statusText;
      statusEl.textContent = state.statusText;
    } else {
      btnPlay.textContent = state.isPlaying ? '暂停播放' : `播放全曲 (${vTag})`;
      statusEl.textContent = `${vTag} 就绪 · 启动 ${state.renderTimeMs}ms · 写一次即焚流式缓冲 ${state.pcmRamMb.toFixed(2)} MB`;
    }

    chBtns.forEach((b) => {
      b.classList.toggle('active', b.dataset.ch === state.channelMode);
    });
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

    if (keys.includes('currentTime') || keys.includes('isReady')) {
      if (!isDraggingSeek) {
        slider.value = String(state.currentTime);
      }
      timeTag.textContent = `${formatClock(state.currentTime)} / ${formatClock(state.duration)}`;
      barTag.textContent = formatBarBeat(state.currentTime, state.bpm, state.bar0Sec);
      if (state.isReady) {
        lyricEl.textContent = findCurrentLyric(state.currentTime);
      }
    }
  });
}
