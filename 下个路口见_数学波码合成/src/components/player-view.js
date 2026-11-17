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
      ? `解压波码 ${state.loadProgress[0]}/${state.loadProgress[1]}…`
      : '载入解耦波码…';
  } else {
    dom.play.disabled = false;
    dom.play.innerHTML = state.running ? '暂停合成 <span>Ⅱ</span>' : '实时算波 <span>▶</span>';
  }

  dom.engineStatus.innerHTML = `<i></i> ${state.running ? 'AudioWorklet 纯数学波方程实时演算中' : '等待实时算波'}`;
  dom.engineStatus.dataset.playing = String(state.running);

  dom.modeDescription.textContent = manifest.description || '';

  for (const button of dom.profileButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.profile === state.profile));
  }

  const section = manifest.sections.findLast(([start]) => state.time >= start);
  dom.sectionName.textContent = section ? section[1] : manifest.sections[0]?.[1] || '';

  const line = manifest.lyrics.find(([start, end]) => state.time >= start && state.time < end + 0.15);
  if (line) {
    dom.lyric.textContent = line[2];
  } else if (state.time < 23.03) {
    dom.lyric.textContent = '模态滴答入场，56 键谐波弦阵与拨弦贝斯实时演算铺开。';
  } else if (state.time >= 177.62) {
    dom.lyric.textContent = 'DI DI DA · 下个路口再见吧';
  } else {
    dom.lyric.textContent = '· · ·';
  }

  dom.calculation.textContent = `已按波函数实时算出 ${(state.computed || 0).toLocaleString()} 采样点 · 48,000 Hz`;
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
    if (sizeEl) {
      sizeEl.innerHTML = `${(bytes / 1024).toFixed(1)} <small>KB</small>`;
    }
    if (barEl && data.referenceBytes) {
      barEl.style.setProperty('--size', `${Math.max(1.8, Math.min(100, (bytes / data.referenceBytes) * 100))}%`);
    }
    if (saveEl && data.referenceBytes) {
      const savedPct = (100 * (1 - bytes / data.referenceBytes)).toFixed(2);
      saveEl.textContent =
        profile === 'compact'
          ? `比原 MP3 缩小 ${savedPct}% · 人声全曲仅 38.2 KB · 较 V4 再降 ${item.savedHtmlVsV4Percent || item.savedVsV4Percent}%`
          : `比原 MP3 缩小 ${savedPct}% · 四轨波码合计 ${(item.parameterBytes / 1024).toFixed(1)} KB`;
    }
  }
}
