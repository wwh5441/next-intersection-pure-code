// V9R2.1 Direct 64-ERB Coarticulation Pitch Ribbon & Live 48kHz Spectrum (`src/components/pitch-coart-canvas.js`)
const SECTIONS = [
  { label: '前奏吉他 Intro', sec: 0.0 },
  { label: '主歌 A1 (东京铁塔)', sec: 22.6 },
  { label: '导歌 B1 (我偏爱)', sec: 42.1 },
  { label: '副歌 C1 (秒针转动)', sec: 58.6 },
  { label: '间奏 Hook', sec: 79.8 },
  { label: '主歌 A2 (7-11暖杯)', sec: 99.6 },
  { label: '副歌 C2+C3 高潮', sec: 137.0 },
  { label: '尾奏淡出 Outro', sec: 178.5 },
];

export function mountPitchCoartCanvas(container, metadata, store, audioEngine, onSeekAndPlay) {
  container.innerHTML = `
    <div class="panel-head">
      <h2 class="panel-title">V9R2.6 动态声门 &amp; 共振峰协同基音轨迹 &amp; 48kHz 实时频谱 (点击画布或段落按钮空降试听)</h2>
      <span class="panel-badge">${metadata.metrics.atlasStats.totalEvents} 音节事件 · 器乐空白区 0 Byte</span>
    </div>
    <div class="section-jump-row" id="sec-jump-row">
      ${SECTIONS.map(
        (s, i) => `<button class="sec-jump-btn" data-idx="${i}">${s.label}</button>`
      ).join('')}
    </div>
    <div class="ribbon-canvas-wrap">
      <canvas id="pitch-event-canvas" class="ribbon-canvas" width="960" height="310"></canvas>
    </div>
    <div class="ribbon-legend">
      <span class="legend-item"><span class="legend-dot" style="background:#2a9d8f"></span>声母瞬态区 (VOT)</span>
      <span class="legend-item"><span class="legend-dot" style="background:#e85d3f"></span>韵母谐波核</span>
      <span class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>「--」跨字逆向协同连音桥</span>
      <span class="legend-item"><span class="legend-dot" style="background:#38bdf8"></span>48kHz 实时频谱分析</span>
    </div>
  `;

  const jumpRow = container.querySelector('#sec-jump-row');
  const canvas = container.querySelector('#pitch-event-canvas');
  const ctx = canvas.getContext('2d');
  const specBytes = new Uint8Array(1024);
  let uiEvents = [];

  jumpRow.addEventListener('click', (e) => {
    const btn = e.target.closest('.sec-jump-btn');
    if (!btn) return;
    const sec = SECTIONS[Number(btn.dataset.idx)].sec;
    onSeekAndPlay(sec);
  });

  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const frac = Math.max(0.0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
    const st = store.getState();
    // Top 62% is a zoomed 18-second rolling window; bottom 38% is full-song overview + spectrum!
    const yFrac = (e.clientY - rect.top) / rect.height;
    if (yFrac < 0.62) {
      const winSpan = 18.0;
      const winStart = Math.max(0.0, Math.min(st.durationSec - winSpan, st.currentTime - 4.0));
      onSeekAndPlay(winStart + frac * winSpan);
    } else {
      onSeekAndPlay(frac * st.durationSec);
    }
  });

  return {
    setEvents(evs) {
      uiEvents = evs || [];
    },
    draw(state) {
      const W = canvas.width;
      const H = canvas.height;
      const topH = Math.floor(H * 0.62);
      const botY = topH + 6;
      const botH = H - botY;

      ctx.fillStyle = '#080c14';
      ctx.fillRect(0, 0, W, H);

      // 1. Top Rolling 18s Syllable Event & `--` Coarticulation Ribbon
      const winSpan = 18.0;
      const winStart = Math.max(0.0, Math.min(state.durationSec - winSpan, state.currentTime - 4.0));
      const winEnd = winStart + winSpan;
      const midiMin = 54;
      const midiMax = 74;

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
      ctx.lineWidth = 1;
      for (let m = 56; m <= 72; m += 4) {
        const y = topH - 16 - ((m - midiMin) / (midiMax - midiMin)) * (topH - 32);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      let prevEndPx = null;
      let prevY = null;

      ctx.font = '11px -apple-system, "PingFang SC", sans-serif';
      for (let i = 0; i < uiEvents.length; i++) {
        const ev = uiEvents[i];
        if (ev.endSec < winStart || ev.startSec > winEnd) continue;
        const x0 = ((ev.startSec - winStart) / winSpan) * W;
        const x1 = ((ev.endSec - winStart) / winSpan) * W;
        const w = Math.max(3, x1 - x0);
        const yMid = topH - 18 - ((Math.max(midiMin, Math.min(midiMax, ev.meanMidi)) - midiMin) / (midiMax - midiMin)) * (topH - 36);
        const isHi = state.highlightedSyllableId === ev.syllableId;
        const isNow = state.currentTime >= ev.startSec && state.currentTime <= ev.endSec;

        if (ev.legatoPrev && prevEndPx !== null && Math.abs(x0 - prevEndPx) < 12) {
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(prevEndPx, prevY);
          ctx.lineTo(x0, yMid);
          ctx.stroke();
        }

        const votW = Math.min(w * 0.3, 10);
        ctx.fillStyle = isHi ? '#38bdf8' : '#2a9d8f';
        ctx.fillRect(x0, yMid - 8, votW, 16);

        ctx.fillStyle = isNow ? '#fde047' : isHi ? '#38bdf8' : '#e85d3f';
        ctx.fillRect(x0 + votW, yMid - 8, Math.max(1, w - votW), 16);

        if (w > 14) {
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(ev.text, x0 + 2, yMid - 12);
        }
        prevEndPx = x1;
        prevY = yMid;
      }

      // Playhead on rolling window
      const playXTop = ((state.currentTime - winStart) / winSpan) * W;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playXTop, 0);
      ctx.lineTo(playXTop, topH);
      ctx.stroke();

      // Divider
      ctx.fillStyle = 'rgba(148, 163, 184, 0.18)';
      ctx.fillRect(0, topH, W, 2);

      // 2. Bottom Full-Song Overview + Live 48kHz Spectrum Bars
      audioEngine.getSpectrumBytes(specBytes);
      const nBars = 96;
      const barW = W / nBars;
      for (let b = 0; b < nBars; b++) {
        const bin = Math.min(specBytes.length - 1, Math.floor(Math.pow(b / nBars, 1.65) * 760) + 1);
        const amp = specBytes[bin] / 255.0;
        const bh = amp * (botH - 18);
        ctx.fillStyle = `rgba(56, 189, 248, ${0.18 + 0.55 * amp})`;
        ctx.fillRect(b * barW + 1, H - bh, Math.max(1, barW - 2), bh);
      }

      // Mini full-song event ticks
      const dur = Math.max(1.0, state.durationSec);
      for (let i = 0; i < uiEvents.length; i++) {
        const ev = uiEvents[i];
        const x = (ev.startSec / dur) * W;
        ctx.fillStyle =
          state.highlightedSyllableId === ev.syllableId
            ? '#fbbf24'
            : 'rgba(232, 93, 63, 0.55)';
        ctx.fillRect(x, botY + 4, Math.max(1.5, ((ev.endSec - ev.startSec) / dur) * W), 8);
      }

      const playXBot = (state.currentTime / dur) * W;
      ctx.fillStyle = '#10b981';
      ctx.fillRect(playXBot - 1, botY, 2.5, botH);
    },
  };
}
