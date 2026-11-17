// Transport Bar, Tier Switcher & Live Lyric Readout (`src/components/transport-bar.js`)
function formatTime(sec) {
  const s = Math.max(0, sec);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${String(m).padStart(2, '0')}:${r.toFixed(2).padStart(5, '0')}`;
}

export function mountTransportBar(container, metadata, store, handlers) {
  const { config, lyrics } = metadata;
  const bar0Sec = config.bar0Samples / config.sampleRate;
  const barDur = 240.0 / config.bpm;

  container.innerHTML = `
    <div class="transport-row">
      <div class="transport-btns">
        <button class="btn btn-primary" id="btn-play-toggle">▶ 合成并播放</button>
        <button class="btn" id="btn-stop">■ 归零</button>
        <button class="btn btn-emerald" id="btn-export-wav" disabled>⬇ 导出当前混音 WAV</button>
        <div class="clock-box mono">
          <span class="clock-time" id="clock-time">00:00.00</span>
          <span class="clock-bar" id="clock-bar">Intro · 98.0000 BPM</span>
        </div>
      </div>
      <div class="tier-selector-wrap">
        <span class="status-pill" id="status-pill">待命</span>
      </div>
    </div>
    <div class="seek-lyric-row">
      <input type="range" class="seek-slider" id="seek-slider" min="0" max="${lyrics.durationSec}" step="0.02" value="0" />
      <span class="mono metric-sub" id="dur-readout">/ ${formatTime(lyrics.durationSec)}</span>
    </div>
    <div class="lyric-banner">
      <span class="lyric-text" id="lyric-line">♪ 点击「合成并播放」试听纯代码十一轨合成</span>
      <span class="active-syllable-tag mono" id="active-syl-tag">--</span>
    </div>
  `;

  const btnPlay = container.querySelector('#btn-play-toggle');
  const btnStop = container.querySelector('#btn-stop');
  const btnExport = container.querySelector('#btn-export-wav');
  const statusPill = container.querySelector('#status-pill');
  const seekSlider = container.querySelector('#seek-slider');
  const clockTime = container.querySelector('#clock-time');
  const clockBar = container.querySelector('#clock-bar');
  const lyricEl = container.querySelector('#lyric-line');
  const sylTagEl = container.querySelector('#active-syl-tag');

  let draggingSeek = false;

  btnPlay.addEventListener('click', () => handlers.onTogglePlay());
  btnStop.addEventListener('click', () => handlers.onStop());
  btnExport.addEventListener('click', () => handlers.onExportWav());

  seekSlider.addEventListener('pointerdown', () => {
    draggingSeek = true;
  });
  seekSlider.addEventListener('change', () => {
    draggingSeek = false;
    handlers.onSeek(Number(seekSlider.value));
  });
  seekSlider.addEventListener('input', () => {
    clockTime.textContent = formatTime(Number(seekSlider.value));
  });

  function findLyricLine(sec) {
    const lines = lyrics.lines;
    for (let i = lines.length - 1; i >= 0; i--) {
      if (sec >= lines[i].startSec) return lines[i].text || '♪ (器乐段落)';
    }
    return '♪ 前奏 Intro';
  }

  return {
    update(state, activeEv) {
      btnPlay.disabled = state.rendering;
      btnExport.disabled = !state.ready || state.rendering;

      if (state.rendering) {
        btnPlay.textContent = `⏳ 合成中 ${Math.round(state.progress * 100)}%`;
        statusPill.className = 'status-pill';
        statusPill.textContent = state.statusText;
      } else if (state.playing) {
        btnPlay.textContent = '❚❚ 暂停';
        statusPill.className = 'status-pill ready';
        statusPill.textContent = state.statusText;
      } else {
        btnPlay.textContent = state.ready ? '▶ 播放' : '⚡ 合成并播放';
        statusPill.className = state.ready ? 'status-pill ready' : 'status-pill';
        statusPill.textContent = state.statusText;
      }

      const t = state.currentTime;
      clockTime.textContent = formatTime(t);
      if (!draggingSeek) seekSlider.value = String(t);

      if (t < bar0Sec) {
        clockBar.textContent = 'Intro Pickup · 98.0000 BPM';
      } else {
        const bFloat = (t - bar0Sec) / barDur;
        const barIdx = Math.floor(bFloat) + 1;
        const beatIdx = Math.floor((bFloat - Math.floor(bFloat)) * 4) + 1;
        clockBar.textContent = `Bar ${String(barIdx).padStart(2, '0')} · Beat ${beatIdx}`;
      }

      lyricEl.textContent = findLyricLine(t);
      if (activeEv) {
        const bridge = `${activeEv.legatoPrev ? '--' : ''}${activeEv.pinyin}${activeEv.legatoNext ? '--' : ''}`;
        sylTagEl.textContent = `[#${activeEv.eventIndex} ${activeEv.text} · ${bridge} · 原型#${activeEv.syllableId}]`;
      } else {
        sylTagEl.textContent = '[无声区间 · 人声 0 Byte]';
      }
    },
  };
}
