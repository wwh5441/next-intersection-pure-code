import { updateTracks } from './tracks.js';
import { updatePresets } from './presets.js';

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
      ? `解压人声波码 ${state.loadProgress[0]}/${state.loadProgress[1]}…`
      : '载入五轨人声…';
  } else {
    dom.play.disabled = false;
    dom.play.innerHTML = state.running ? '暂停人声 <span>Ⅱ</span>' : '单独听人声 <span>▶</span>';
  }

  dom.engineStatus.innerHTML = `<i></i> ${
    state.running ? '48kHz 五轨纯人声余弦波方程实时合成中（零伴奏）' : '已就绪 · 默认定位 00:22.6 主唱起唱点'
  }`;
  dom.engineStatus.dataset.playing = String(state.running);

  dom.modeDescription.textContent = manifest.description || '';

  for (const button of dom.profileButtons) {
    button.setAttribute('aria-pressed', String(button.dataset.profile === state.profile));
  }

  const section = manifest.sections.findLast(([start]) => state.time >= start);
  dom.sectionName.textContent = section ? section[1] : manifest.sections[0]?.[1] || '';

  const line = manifest.lyrics.find(([start, end]) => state.time >= start && state.time < end + 0.25);
  if (line) {
    dom.lyric.textContent = line[2];
  } else if (state.time < 22.8) {
    dom.lyric.textContent = `纯人声无伴奏静默段（主唱将于 00:23 入场，可点下方「跳至主唱 00:23」）`;
  } else if (state.time >= 79.7 && state.time < 101.0) {
    dom.lyric.textContent = '间奏留白段（伴奏已完全剥离，可点「跳至副歌叠唱 01:02」或拖至 01:41）';
  } else if (state.time >= 177.7) {
    dom.lyric.textContent = '人声尾音余韵收束 · 纯人声无伴奏';
  } else {
    dom.lyric.textContent = '· 呼吸换气间隙 ·';
  }

  dom.calculation.textContent = `已实时算出 ${(state.computed || 0).toLocaleString()} 个 48kHz 立体声采样点`;
  updatePresets(dom.presets, state);
  updateTracks(dom.tracks, state);
}

export function renderSizeMetrics(data) {
  if (!data) return;
  const wavEl = document.querySelector('#wav-vocal-size');
  if (wavEl && data.losslessStereoWavBytes) {
    wavEl.innerHTML = `${(data.losslessStereoWavBytes / 1e6).toFixed(2)} <small>MB</small>`;
  }
  for (const profile of ['compact', 'faithful']) {
    const item = data[profile];
    if (!item) continue;
    const sizeEl = document.querySelector(`#${profile}-size`);
    const barEl = document.querySelector(`#${profile}-bar`);
    const saveEl = document.querySelector(`#${profile}-saving`);
    const bytes = item.parameterBytes;
    if (sizeEl) {
      sizeEl.innerHTML =
        bytes < 1e6
          ? `${(bytes / 1024).toFixed(1)} <small>KB</small>`
          : `${(bytes / 1e6).toFixed(2)} <small>MB</small>`;
    }
    if (barEl && data.losslessStereoWavBytes) {
      barEl.style.setProperty(
        '--size',
        `${Math.max(1.5, Math.min(100, (bytes / data.losslessStereoWavBytes) * 100))}%`
      );
    }
    if (saveEl) {
      const drySnr = item.dryWaveSnrDb ?? item.vocalWaveSnrDb;
      saveEl.textContent = `干声波形 SNR ${drySnr} dB · 全频梅尔 SNR ${item.vocalLogMelSnrDb} dB · 缩小 ${item.smallerThanWavPercent}%`;
    }
  }
}
