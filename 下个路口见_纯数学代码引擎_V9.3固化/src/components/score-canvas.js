// Interactive 48kHz Master Score, 285-Note Vocal Melody & 98.0 BPM Chord Timeline Canvas

import { BAND_SCORE } from '../../data/band-score.js';
import { VOCAL_NOTE_TIMELINE } from '../../data/vocal-lyrics.js';

const QUICK_JUMPS = [
  { label: '0.4s 前奏木吉他扫弦与木鱼律动', sec: 0.35 },
  { label: '22.8s 主歌一：刚下的地铁', sec: 22.8 },
  { label: '61.5s 副歌：秒针转动 DI DI DA', sec: 61.5 },
  { label: '76.1s 下个路口再见吧', sec: 76.1 },
  { label: '101.0s 主歌二：伦敦叹息鼓抽离段', sec: 101.0 },
  { label: '120.5s 弗朗明哥的热情', sec: 120.5 },
  { label: '197.4s Bar 80 尾奏 D9 大和弦余韵', sec: 197.4 },
];

function buildChordBlocks(bandScore) {
  const blocks = [];
  const barSec = 240.0 / bandScore.bpm;
  const halfSec = barSec * 0.5;
  const bar0 = bandScore.bar0Sec;

  for (const sec of bandScore.sections) {
    for (let b = 0; b < sec.bars; b++) {
      const barIdx = sec.startBar + b;
      const tBar = bar0 + barIdx * barSec;
      if (sec.prog === 'FINAL') {
        if (b === 0) {
          blocks.push({ start: tBar, end: tBar + sec.bars * barSec, chord: 'D9 (Picardy)', section: sec.name });
        }
        continue;
      }
      const tmpl = bandScore.progressions[sec.prog] || bandScore.progressions.STD4;
      const cycIdx = ((sec.progOffset || 0) + b) % 4;
      let [c0, c1] = tmpl[cycIdx];
      if (sec.resolveLastDm && b === sec.bars - 1) c1 = 'Dm9';
      blocks.push({ start: tBar, end: tBar + halfSec, chord: c0, section: sec.name });
      blocks.push({ start: tBar + halfSec, end: tBar + barSec, chord: c1, section: sec.name });
    }
  }
  return blocks;
}

export function mountScoreCanvas(container, store, mixer) {
  container.innerHTML = `
    <div class="canvas-header">
      <span id="canvas-tier-caption"><strong>48kHz 正交余弦母带总谱视窗</strong> · 上层：原曲主唱 285 字音高与连音轨迹 ｜ 下层：98.0 BPM 全曲 83 小节和声进行 ｜ 点击画布任意处瞬时跳转</span>
      <span id="section-tag-label">Section: Intro</span>
    </div>
    <canvas id="pure-math-score-canvas" class="score-canvas-el" width="1280" height="210"></canvas>
    <div class="jump-row" id="quick-jump-strip"></div>
  `;

  const canvas = container.querySelector('#pure-math-score-canvas');
  const secLabel = container.querySelector('#section-tag-label');
  const jumpStrip = container.querySelector('#quick-jump-strip');
  const ctx = canvas.getContext('2d');

  for (const j of QUICK_JUMPS) {
    const btn = document.createElement('button');
    btn.className = 'btn-chip';
    btn.textContent = j.label;
    btn.addEventListener('click', () => {
      mixer.seek(j.sec);
      if (!store.getState().isPlaying && store.getState().isReady) {
        mixer.play(j.sec);
      }
    });
    jumpStrip.appendChild(btn);
  }

  const chordBlocks = buildChordBlocks(BAND_SCORE);
  const winSpanSec = 12.0;

  canvas.addEventListener('click', (ev) => {
    if (!store.getState().isReady) return;
    const rect = canvas.getBoundingClientRect();
    const xFrac = (ev.clientX - rect.left) / Math.max(1, rect.width);
    const tCenter = store.getState().currentTime;
    const tLeft = tCenter - winSpanSec * 0.28;
    const target = Math.max(0, Math.min(store.getState().duration, tLeft + xFrac * winSpanSec));
    mixer.seek(target);
  });

  function renderFrame(tNow) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 1200;
    const cssH = canvas.clientHeight || 210;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#171920';
    ctx.fillRect(0, 0, cssW, cssH);

    const tLeft = tNow - winSpanSec * 0.28;
    const tRight = tLeft + winSpanSec;
    const xOf = (t) => ((t - tLeft) / winSpanSec) * cssW;

    const vocalBottom = cssH - 52;
    const midiMin = 50;
    const midiMax = 74;
    const yOfMidi = (m) => {
      const u = Math.max(0, Math.min(1, (m - midiMin) / (midiMax - midiMin)));
      return vocalBottom - 14 - u * (vocalBottom - 28);
    };

    const barSec = 240.0 / BAND_SCORE.bpm;
    const firstBar = Math.max(0, Math.floor((tLeft - BAND_SCORE.bar0Sec) / barSec));
    const lastBar = Math.min(84, Math.ceil((tRight - BAND_SCORE.bar0Sec) / barSec));
    for (let b = firstBar; b <= lastBar; b++) {
      const tb = BAND_SCORE.bar0Sec + b * barSec;
      const xb = xOf(tb);
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xb, 0);
      ctx.lineTo(xb, vocalBottom);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.font = '10px monospace';
      ctx.fillText(`B${b + 1}`, xb + 4, 13);
    }

    ctx.fillStyle = '#1f232e';
    ctx.fillRect(0, vocalBottom, cssW, cssH - vocalBottom);

    let activeSecName = tNow < BAND_SCORE.bar0Sec ? 'Studio Pickup (98 BPM)' : 'Outro Ring';
    for (const blk of chordBlocks) {
      if (tNow >= blk.start && tNow < blk.end) activeSecName = `${blk.section} · ${blk.chord}`;
      if (blk.end < tLeft || blk.start > tRight) continue;
      const x0 = Math.max(0, xOf(blk.start));
      const x1 = Math.min(cssW, xOf(blk.end));
      const isCurrent = tNow >= blk.start && tNow < blk.end;
      ctx.fillStyle = isCurrent ? 'rgba(42, 157, 143, 0.34)' : 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(x0 + 1, vocalBottom + 6, Math.max(2, x1 - x0 - 2), cssH - vocalBottom - 12);
      ctx.strokeStyle = isCurrent ? '#2a9d8f' : 'rgba(255,255,255,0.12)';
      ctx.strokeRect(x0 + 1, vocalBottom + 6, Math.max(2, x1 - x0 - 2), cssH - vocalBottom - 12);

      if (x1 - x0 > 34) {
        ctx.fillStyle = isCurrent ? '#7ef2e2' : '#cbd5e1';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(blk.chord, x0 + 7, vocalBottom + 28);
      }
    }
    secLabel.textContent = `Section: ${activeSecName}`;

    for (let i = 0; i < VOCAL_NOTE_TIMELINE.length; i++) {
      const [stMs, durMs, hanzi, midiQ, flags] = VOCAL_NOTE_TIMELINE[i];
      const startSec = stMs * 0.001;
      const endSec = (stMs + durMs) * 0.001;
      if (endSec < tLeft - 0.3 || startSec > tRight + 0.3) continue;
      const x0 = xOf(startSec);
      const x1 = Math.max(x0 + 5, xOf(endSec));
      const meanMidi = midiQ * 0.1;
      const yMid = yOfMidi(meanMidi);
      const isSungNow = tNow >= startSec && tNow <= endSec;
      const legatoNext = (flags & 2) !== 0;
      const unvoiced = (flags & 8) !== 0;

      if (legatoNext && i + 1 < VOCAL_NOTE_TIMELINE.length) {
        const nxt = VOCAL_NOTE_TIMELINE[i + 1];
        const nxtStart = nxt[0] * 0.001;
        if (nxtStart - endSec < 0.22) {
          const yn = yOfMidi(nxt[3] * 0.1);
          ctx.strokeStyle = 'rgba(232, 93, 63, 0.65)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x1, yMid);
          ctx.bezierCurveTo((x1 + xOf(nxtStart)) * 0.5, yMid, (x1 + xOf(nxtStart)) * 0.5, yn, xOf(nxtStart), yn);
          ctx.stroke();
        }
      }

      ctx.fillStyle = unvoiced
        ? 'rgba(42, 157, 143, 0.80)'
        : (isSungNow ? '#ff7b5c' : 'rgba(217, 72, 59, 0.80)');
      ctx.fillRect(x0, yMid - 6, x1 - x0, 12);

      if (x1 - x0 >= 14) {
        ctx.fillStyle = isSungNow ? '#ffffff' : '#e2e8f0';
        ctx.font = '12px sans-serif';
        ctx.fillText(hanzi, x0 + 2, yMid - 10);
      }
    }

    const xPlay = xOf(tNow);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xPlay, 0);
    ctx.lineTo(xPlay, cssH);
    ctx.stroke();

    ctx.restore();
  }

  renderFrame(0);
  store.subscribe((state, keys) => {
    if (keys.includes('currentTime') || keys.includes('isReady')) {
      renderFrame(state.currentTime);
    }
  });
  window.addEventListener('resize', () => renderFrame(store.getState().currentTime));
}
