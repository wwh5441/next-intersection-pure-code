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
      ? `解压双轨流 ${state.loadProgress[0]}/${state.loadProgress[1]}…`
      : '载入全曲双轨…';
  } else {
    dom.play.disabled = false;
    dom.play.innerHTML = state.running ? '暂停播放 <span>Ⅱ</span>' : '播放全曲 <span>▶</span>';
  }

  dom.engineStatus.innerHTML = `<i></i> ${
    state.running
      ? `48kHz ${manifest.profileName || '全曲双轨'}（1 轨人声 + 1 轨伴奏）实时合成中`
      : `已就绪 · ${manifest.profileName || '全曲双轨'}（支持一键切纯人声 / 纯伴奏 KTV）`
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
    dom.lyric.textContent = '· 前奏器乐段（原声吉他 + 合成器 + 律动鼓贝斯，00:23 主唱入场） ·';
  } else if (state.time >= 79.7 && state.time < 101.0) {
    dom.lyric.textContent = '· 间奏器乐律动段（01:41 第三主歌入场） ·';
  } else if (state.time >= 177.7) {
    dom.lyric.textContent = '· 尾奏收束余韵 ·';
  } else {
    dom.lyric.textContent = '· 乐句衔接间隙 ·';
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
        `${Math.max(2.0, Math.min(100, (bytes / data.losslessStereoWavBytes) * 100))}%`
      );
    }
    if (saveEl) {
      const vocalKb = item.vocalBytes ? `人声 ${(item.vocalBytes / 1024).toFixed(1)} KB · ` : '';
      saveEl.textContent = `${vocalKb}等效码率 ${item.kbps} kbps · 梅尔 SNR ${item.mixLogMelSnrDb} dB · 比 WAV 缩小 ${item.smallerThanWavPercent}%`;
    }
  }
}
