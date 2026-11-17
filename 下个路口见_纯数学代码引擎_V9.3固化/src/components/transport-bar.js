// Studio Transport Bar, Conservation Preset Selector, 24-Bit WAV Exporter & Musical Clock

import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';

function formatClock(sec) {
  const s = Math.max(0, sec);
  const mins = Math.floor(s / 60);
  const rem = s - mins * 60;
  return `${String(mins).padStart(2, '0')}:${rem.toFixed(1).padStart(4, '0')}`;
}

function formatBarBeat(sec, bpm, bar0Sec) {
  if (sec < bar0Sec) {
    const pickupBeat = Math.max(1, Math.min(2, Math.floor(((sec - 0.372) / (60 / bpm)) + 1)));
    return `Pickup · Beat ${pickupBeat} | ${bpm.toFixed(1)} BPM`;
  }
  const beatDur = 60.0 / bpm;
  const totalBeats = (sec - bar0Sec) / beatDur;
  const bar = Math.floor(totalBeats / 4) + 1;
  const beat = (Math.floor(totalBeats) % 4) + 1;
  return `Bar ${String(bar).padStart(2, '0')} : Beat ${beat} | ${bpm.toFixed(1)} BPM`;
}

function findCurrentLyric(sec) {
  let active = '♪ 48kHz 原曲母带级前奏律动 (98.0 BPM 原声木吉他扫弦 + 木鱼底鼓 + 电贝斯滑音)';
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
          <span class="tier-switch-label">48kHz 母带能量守恒定律：</span>
          <span id="conservation-formula-tag" class="tier-cache-tag">Σ_{i=1..6} [W_i(t,b) / 128] ≡ 1.0000000 · 六轨推子归零即 100% 精确还原原曲双声道母带 (零拆轨串扰)</span>
        </div>
        <button id="btn-export-wav" class="btn-tier" disabled>导出 48kHz / 24-bit 母带 WAV</button>
      </div>

      <div class="transport-top">
        <div class="transport-controls">
          <button id="btn-play-toggle" class="btn-primary" disabled>正在求解正交余弦级数...</button>
          <button class="btn-chip active" data-preset="all">全曲母带守恒总混 (6轨)</button>
          <button class="btn-chip" data-preset="vocal">仅人声与混响空间 (01-02)</button>
          <button class="btn-chip" data-preset="band">仅原曲纯伴奏 KTV (03-06)</button>
          <button id="btn-reset-mix" class="btn-chip">复位 0dB 守恒态</button>
        </div>
        <div class="clock-box">
          <span id="bar-beat-readout" class="bar-beat-tag">Pickup · Beat 1 | 98.0 BPM</span>
          <span id="time-readout">00:00.0 / 03:30.4</span>
        </div>
      </div>

      <input id="seek-slider" class="seek-slider" type="range" min="0" max="${initSt.duration.toFixed(3)}" step="0.02" value="0" disabled />

      <div class="lyric-banner">
        <div id="lyric-line-display" class="lyric-current">正在初始化 48kHz 正交余弦合成器...</div>
        <div id="synth-status-hint" class="status-hint">0%</div>
      </div>
    </div>
  `;

  const btnPlay = container.querySelector('#btn-play-toggle');
  const btnExport = container.querySelector('#btn-export-wav');
  const btnReset = container.querySelector('#btn-reset-mix');
  const presetBtns = Array.from(container.querySelectorAll('[data-preset]'));
  const slider = container.querySelector('#seek-slider');
  const barTag = container.querySelector('#bar-beat-readout');
  const timeTag = container.querySelector('#time-readout');
  const lyricEl = container.querySelector('#lyric-line-display');
  const statusEl = container.querySelector('#synth-status-hint');

  let isDraggingSeek = false;

  btnPlay.addEventListener('click', () => {
    mixer.togglePlay();
  });

  btnExport.addEventListener('click', () => {
    const ver = store.getState().versionLabel || 'V9.3';
    mixer.exportCurrentMixWav(`下个路口见_${ver}_纯数学48kHz_24bit母带.wav`);
  });

  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      store.applyGroupPreset(btn.dataset.preset);
      mixer.syncGainsFromStore();
    });
  });

  btnReset.addEventListener('click', () => {
    store.resetAllFaders();
    mixer.syncGainsFromStore();
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

  store.subscribe((state, keys) => {
    if (
      keys.includes('isReady') ||
      keys.includes('isPlaying') ||
      keys.includes('isExportingWav') ||
      keys.includes('progress') ||
      keys.includes('statusText')
    ) {
      btnPlay.disabled = !state.isReady;
      btnExport.disabled = !state.isReady || state.isExportingWav;
      btnExport.textContent = state.isExportingWav
        ? '正在离线合成 24-bit WAV...'
        : '导出 48kHz / 24-bit 母带 WAV';
      slider.disabled = !state.isReady;
      slider.max = String(state.duration.toFixed(3));

      if (!state.isReady) {
        btnPlay.textContent = `纯数学求解中 (${Math.round(state.progress * 100)}%)`;
        statusEl.textContent = state.statusText;
      } else {
        btnPlay.textContent = state.isPlaying ? '暂停播放' : '播放全曲 (48kHz 几乎无损)';
        const modeDesc =
          state.engineMode === 'streaming'
            ? `流式环形缓冲就绪 (${state.renderTimeMs}ms 瞬时启动 · 内存 ${state.pcmRamMb.toFixed(2)} MB)`
            : `全轨预合成耗时 ${(state.renderTimeMs / 1000).toFixed(2)}s · 峰值 ${state.peakAmplitude} · 内存 ${state.pcmRamMb.toFixed(1)} MB`;
        statusEl.textContent = modeDesc;
      }
    }

    if (keys.includes('presetMode')) {
      presetBtns.forEach((b) => {
        b.classList.toggle('active', b.dataset.preset === state.presetMode);
      });
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
