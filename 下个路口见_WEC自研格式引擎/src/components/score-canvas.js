// Interactive Score, 50Hz TIMB Vocal F0 Contour, 285-Syllable LYRC & 83-Bar GRID Canvas

import { BAND_SCORE } from '../../data/band-score.js';
import { VOCAL_NOTE_TIMELINE } from '../../data/vocal-lyrics.js';

const QUICK_JUMPS = [
  { label: '0.4s 前奏木吉他扫弦', sec: 0.35 },
  { label: '22.8s 主歌一：刚下的地铁', sec: 22.8 },
  { label: '61.5s 副歌：秒针转动 DI DI DA', sec: 61.5 },
  { label: '76.1s 下个路口再见吧', sec: 76.1 },
  { label: '101.0s 主歌二：忙碌会议', sec: 101.0 },
  { label: '120.5s 弗朗明哥的热情', sec: 120.5 },
  { label: '197.4s Bar 80 尾奏 D9 大和弦', sec: 197.4 },
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

export function mountScoreCanvas(container, store, mixer, getActiveBundle) {
  container.innerHTML = `
    <div class="canvas-header">
      <span id="canvas-tier-caption"><strong>WEC 结构化特征层实时视窗 (TIMB 50Hz 声纹基频线 + LYRC 音节块 + GRID 和弦网格)</strong> · 点击画布任意位置瞬时跳播</span>
      <span id="section-tag-label">Section: Intro</span>
    </div>
    <canvas id="pure-math-score-canvas" class="score-canvas-el" width="1280" height="210"></canvas>
    <div class="jump-row" id="quick-jump-strip"></div>
  `;

  const canvas = container.querySelector('#pure-math-score-canvas');
  const captionEl = container.querySelector('#canvas-tier-caption');
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

  const fallbackChords = buildChordBlocks(BAND_SCORE);
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
    const bundle = getActiveBundle ? getActiveBundle() : null;
    const parsed = bundle?.parsed || null;
    const feats = parsed?.decodedFeatures || {};
    const timb = feats.TIMB || null;
    const gridData = feats.GRID || BAND_SCORE;
    const chordBlocks = feats.GRID ? buildChordBlocks(feats.GRID) : fallbackChords;
    const notes = feats.LYRC?.syllables || VOCAL_NOTE_TIMELINE;

    if (parsed) {
      if (parsed.magic === 'WEC0') {
        captionEl.innerHTML = `<strong>WEC0 纯最小单流视窗</strong> · 文件内 0 特征层开销（仅 32B 头 + Opus 裸包），画布叠加参考时间轴供跳播对照`;
      } else {
        const layerNames = parsed.features.map((f) => f.fourCC).join(' + ');
        captionEl.innerHTML = `<strong>${parsed.magic} 内嵌特征层直读视窗 [${layerNames}]</strong> · 青色连续亮线：TIMB 50Hz 真实声纹 F0 曲线 ｜ 橙红音节砖：LYRC 逐字层 ｜ 底部：GRID 和弦层`;
      }
    }

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 1200;
    const cssH = canvas.clientHeight || 210;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#141821';
    ctx.fillRect(0, 0, cssW, cssH);

    const tLeft = tNow - winSpanSec * 0.28;
    const tRight = tLeft + winSpanSec;
    const xOf = (t) => ((t - tLeft) / winSpanSec) * cssW;

    const vocalBottom = cssH - 50;
    const midiMin = 50;
    const midiMax = 74;
    const yOfMidi = (m) => {
      const u = Math.max(0, Math.min(1, (m - midiMin) / (midiMax - midiMin)));
      return vocalBottom - 14 - u * (vocalBottom - 28);
    };

    const barSec = 240.0 / (gridData.bpm || 98.0);
    const bar0 = gridData.bar0Sec || 1.7493;
    const firstBar = Math.max(0, Math.floor((tLeft - bar0) / barSec));
    const lastBar = Math.min(84, Math.ceil((tRight - bar0) / barSec));
    for (let b = firstBar; b <= lastBar; b++) {
      const tb = bar0 + b * barSec;
      const xb = xOf(tb);
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xb, 0);
      ctx.lineTo(xb, vocalBottom);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.34)';
      ctx.font = '10px monospace';
      ctx.fillText(`B${b + 1}`, xb + 4, 13);
    }

    ctx.fillStyle = '#1e2432';
    ctx.fillRect(0, vocalBottom, cssW, cssH - vocalBottom);

    let activeSecName = tNow < bar0 ? 'Studio Pickup (98 BPM)' : 'Outro Ring';
    for (const blk of chordBlocks) {
      if (tNow >= blk.start && tNow < blk.end) activeSecName = `${blk.section} · ${blk.chord}`;
      if (blk.end < tLeft || blk.start > tRight) continue;
      const x0 = Math.max(0, xOf(blk.start));
      const x1 = Math.min(cssW, xOf(blk.end));
      const isCurrent = tNow >= blk.start && tNow < blk.end;
      ctx.fillStyle = isCurrent ? 'rgba(13, 148, 136, 0.38)' : 'rgba(255, 255, 255, 0.05)';
      ctx.fillRect(x0 + 1, vocalBottom + 6, Math.max(2, x1 - x0 - 2), cssH - vocalBottom - 12);
      ctx.strokeStyle = isCurrent ? '#14b8a6' : 'rgba(255,255,255,0.12)';
      ctx.strokeRect(x0 + 1, vocalBottom + 6, Math.max(2, x1 - x0 - 2), cssH - vocalBottom - 12);

      if (x1 - x0 > 34) {
        ctx.fillStyle = isCurrent ? '#5eead4' : '#cbd5e1';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(blk.chord, x0 + 7, vocalBottom + 27);
      }
    }
    secLabel.textContent = `Section: ${activeSecName}`;

    // Layer 1: Syllable note blocks (LYRC)
    for (let i = 0; i < notes.length; i++) {
      const [stMs, durMs, hanzi, midiQ, flags] = notes[i];
      const startSec = stMs * 0.001;
      const endSec = (stMs + durMs) * 0.001;
      if (endSec < tLeft - 0.3 || startSec > tRight + 0.3) continue;
      const x0 = xOf(startSec);
      const x1 = Math.max(x0 + 5, xOf(endSec));
      const meanMidi = midiQ * 0.1;
      const yMid = yOfMidi(meanMidi);
      const isSungNow = tNow >= startSec && tNow <= endSec;
      const unvoiced = (flags & 8) !== 0;

      ctx.fillStyle = unvoiced
        ? 'rgba(45, 212, 191, 0.68)'
        : isSungNow
        ? '#fb7185'
        : 'rgba(225, 29, 72, 0.72)';
      ctx.fillRect(x0, yMid - 6, x1 - x0, 12);

      ctx.fillStyle = isSungNow ? '#ffffff' : 'rgba(255,255,255,0.86)';
      ctx.font = isSungNow ? 'bold 12px sans-serif' : '11px sans-serif';
      ctx.fillText(hanzi, x0 + 1, yMid - 9);
    }

    // Layer 2: Continuous 50Hz TIMB Vocal F0 & Formant Brightness Curve (when embedded in WEC1/WEC2)
    if (timb && timb.f0Cents) {
      const fStart = Math.max(0, Math.floor(tLeft / 0.02));
      const fEnd = Math.min(timb.frameCount - 1, Math.ceil(tRight / 0.02));
      ctx.strokeStyle = '#5eead4';
      ctx.lineWidth = 2.2;
      let penDown = false;
      ctx.beginPath();
      for (let fi = fStart; fi <= fEnd; fi++) {
        const cents = timb.f0Cents[fi];
        if (cents > 0 && (timb.voicing[fi] & 1)) {
          const tx = xOf(fi * 0.02 + 0.01);
          const ty = yOfMidi(cents / 100.0);
          if (!penDown) {
            ctx.moveTo(tx, ty);
            penDown = true;
          } else {
            ctx.lineTo(tx, ty);
          }
        } else {
          penDown = false;
        }
      }
      ctx.stroke();
    }

    // Playhead cursor
    const xPlay = xOf(tNow);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(xPlay, 0);
    ctx.lineTo(xPlay, cssH);
    ctx.stroke();

    ctx.restore();
  }

  renderFrame(store.getState().currentTime);
  store.subscribe((state, keys) => {
    if (keys.includes('currentTime') || keys.includes('isReady') || keys.includes('activeTierId')) {
      renderFrame(state.currentTime);
    }
  });
  window.addEventListener('resize', () => renderFrame(store.getState().currentTime));
}
