import { updateTracks } from './tracks.js';

export function formatClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

export function renderPlayerState(dom, controller, state) {
  const { manifest } = controller;
  dom.currentTime.textContent = formatClock(state.time);
  dom.seek.value = state.time;
  dom.seek.style.setProperty('--progress', `${(state.time / manifest.duration) * 100}%`);

  if (state.loading) {
    dom.play.disabled = true;
    dom.play.textContent = state.loadProgress
      ? `解包参数 ${state.loadProgress[0]}/${state.loadProgress[1]}…`
      : '解开频谱参数…';
  } else {
    dom.play.disabled = false;
    dom.play.innerHTML = state.running ? '暂停试听 <span>Ⅱ</span>' : '开始计算 <span>▶</span>';
  }

  dom.engineStatus.innerHTML = `<i></i> ${state.running ? 'CPU 四轨心理声学实时合成' : '等待播放'}`;
  dom.engineStatus.dataset.playing = String(state.running);

  dom.modeDescription.textContent =
    state.profile === 'faithful'
      ? 'V4 保真版：2048/512 自适应长短窗 + 63 阶 24-Bark 掩蔽 + 参数立体声 + 鼓闭环残差。'
      : 'V4 紧凑版：人声共振峰保真级精度（47 阶）+ 长短窗切换 + 高频参数立体声 + 鼓闭环残差 + Base122。';

  for (const button of dom.profileButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.profile === state.profile));
  }

  const section = manifest.sections.findLast(([start]) => state.time >= start);
  dom.sectionName.textContent = section ? section[1] : manifest.sections[0]?.[1] || '';

  const line = manifest.lyrics.find(([start, end]) => state.time >= start && state.time < end + 0.15);
  if (line) {
    dom.lyric.textContent = line[2];
  } else if (state.time < 23.03) {
    dom.lyric.textContent = '秒针滴答入场，木吉他与贝斯轻快铺开。';
  } else if (state.time >= 177.62) {
    dom.lyric.textContent = 'DI DI DA · 下个路口再见吧';
  } else {
    dom.lyric.textContent = '· · ·';
  }

  dom.calculation.textContent = `已实时合成 ${(state.computed || 0).toLocaleString()} 采样帧 · 48,000 Hz`;
  updateTracks(dom.tracks, state);
}

export function renderSizeMetrics(data) {
  if (!data) return;
  const refEl = document.querySelector('#reference-size');
  if (refEl && data.referenceBytes) {
    refEl.innerHTML = `${(data.referenceBytes / 1e6).toFixed(2)} <small>MB</small>`;
  }
  for (const profile of ['compact', 'faithful']) {
    const item = data[profile];
    if (!item) continue;
    const sizeEl = document.querySelector(`#${profile}-size`);
    const barEl = document.querySelector(`#${profile}-bar`);
    const saveEl = document.querySelector(`#${profile}-saving`);
    const bytes = profile === 'compact' ? item.singleHtmlBytes || item.parameterBytes : item.parameterBytes;
    if (sizeEl) sizeEl.innerHTML = `${(bytes / 1e6).toFixed(2)} <small>MB</small>`;
    if (barEl && data.referenceBytes) {
      barEl.style.setProperty('--size', `${Math.min(100, (bytes / data.referenceBytes) * 100)}%`);
    }
    if (saveEl && data.referenceBytes) {
      const savedPct = (100 * (1 - bytes / data.referenceBytes)).toFixed(1);
      saveEl.textContent =
        profile === 'compact'
          ? `比原 MP3 缩小 ${savedPct}% · 四轨参数仅 ${Math.round(item.parameterBytes / 1024)} KB`
          : `比原 MP3 缩小 ${savedPct}% · 比 V3 保真版再省 ${item.savedVsV3Percent || '—'}%`;
    }
  }
}
