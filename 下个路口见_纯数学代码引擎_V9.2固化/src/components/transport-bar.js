// Studio Transport Bar, Dual-Tier Vocal Version Switcher, Musical Bar:Beat Clock & Lyric Banner

import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';
import { VOCAL_TIERS } from '../../data/vocal-tiers.js';

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
  let active = '♪ 纯数学物理建模前奏律动 (98.0 BPM Disco-Funk Pickup -> Dm9 - Bbmaj7 - Cadd9 - Am7)';
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

export function mountTransportBar(container, store, mixer, coordinator) {
  const tierIds = Object.keys(VOCAL_TIERS);

  const tierButtonsHtml = tierIds
    .map((tid) => {
      const p = VOCAL_TIERS[tid].profile;
      const isAct = tid === store.getState().vocalTier ? ' active' : '';
      return `<button class="btn-tier${isAct}" data-tier="${tid}">${p.shortLabel}</button>`;
    })
    .join('');

  container.innerHTML = `
    <div class="transport-wrap">
      <div class="tier-switch-bar">
        <div class="tier-switch-group">
          <span class="tier-switch-label">全曲数学版本 A/B 切换 (人声+5轨真伴奏)：</span>
          ${tierButtonsHtml}
        </div>
        <span id="tier-cache-hint" class="tier-cache-tag">切换保持当前播放秒数不断连</span>
      </div>

      <div class="transport-top">
        <div class="transport-controls">
          <button id="btn-play-toggle" class="btn-primary" disabled>正在数学积分合成...</button>
          <button class="btn-chip active" data-preset="all">全曲总混音 (10轨)</button>
          <button class="btn-chip" data-preset="vocal">仅纯人声 (5轨)</button>
          <button class="btn-chip" data-preset="band">仅纯伴奏 (5轨)</button>
          <button id="btn-reset-mix" class="btn-chip">复位推子</button>
        </div>
        <div class="clock-box">
          <span id="bar-beat-readout" class="bar-beat-tag">Pickup · Beat 1 | 98.0 BPM</span>
          <span id="time-readout">00:00.0 / 03:30.4</span>
        </div>
      </div>

      <input id="seek-slider" class="seek-slider" type="range" min="0" max="210.373" step="0.05" value="0" disabled />

      <div class="lyric-banner">
        <div id="lyric-line-display" class="lyric-current">正在求解 10 轨声学微分方程...</div>
        <div id="synth-status-hint" class="status-hint">0%</div>
      </div>
    </div>
  `;

  const btnPlay = container.querySelector('#btn-play-toggle');
  const btnReset = container.querySelector('#btn-reset-mix');
  const presetBtns = Array.from(container.querySelectorAll('[data-preset]'));
  const tierBtns = Array.from(container.querySelectorAll('[data-tier]'));
  const tierHint = container.querySelector('#tier-cache-hint');
  const slider = container.querySelector('#seek-slider');
  const barTag = container.querySelector('#bar-beat-readout');
  const timeTag = container.querySelector('#time-readout');
  const lyricEl = container.querySelector('#lyric-line-display');
  const statusEl = container.querySelector('#synth-status-hint');

  let isDraggingSeek = false;

  btnPlay.addEventListener('click', () => {
    mixer.togglePlay();
  });

  tierBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const tid = btn.dataset.tier;
      if (tid === store.getState().vocalTier && !store.getState().isSwitchingTier) return;
      if (coordinator) coordinator.selectVocalTier(tid);
    });
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
      keys.includes('vocalTier') ||
      keys.includes('isSwitchingTier') ||
      keys.includes('cachedTiers')
    ) {
      tierBtns.forEach((b) => {
        const tid = b.dataset.tier;
        const isCached = state.cachedTiers.includes(tid);
        const baseLabel = VOCAL_TIERS[tid].profile.shortLabel;
        b.textContent = isCached ? `✓ ${baseLabel}` : baseLabel;
        b.classList.toggle('active', tid === state.vocalTier);
        b.disabled = state.isSwitchingTier;
      });
      if (tierIds.length === 1) {
        tierHint.textContent = '独立单档纯数学离线版';
      } else if (state.cachedTiers.length >= 2) {
        tierHint.textContent = '双档均已驻留内存 · 点击 0ms 无缝 A/B 秒切对比';
      } else {
        tierHint.textContent = '首次切档后台合成 (~6s)，二次切档 0ms 无缝瞬切';
      }
    }

    if (
      keys.includes('isReady') ||
      keys.includes('isSwitchingTier') ||
      keys.includes('isPlaying') ||
      keys.includes('progress') ||
      keys.includes('statusText')
    ) {
      const busy = !state.isReady || state.isSwitchingTier;
      btnPlay.disabled = busy;
      slider.disabled = !state.isReady;
      slider.max = String(state.duration.toFixed(3));
      if (busy) {
        btnPlay.textContent = `纯数学合成中 (${Math.round(state.progress * 100)}%)`;
        statusEl.textContent = state.statusText;
      } else {
        btnPlay.textContent = state.isPlaying ? '暂停播放' : '播放全曲 (纯数学 10 轨)';
        statusEl.textContent = `合成耗时 ${(state.renderTimeMs / 1000).toFixed(2)}s · 总线峰值 ${state.peakAmplitude}`;
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
