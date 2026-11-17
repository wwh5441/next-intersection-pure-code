// Dual-Layer Interactive Musical Score & Coarticulation Trajectory Canvas
// Dynamically visualizes active vocal tier (`expressive` 16ms dense f0 contour vs `ultra` 32ms contour)

import { BAND_SCORE } from '../../data/band-score.js';
import { VOCAL_TIERS } from '../../data/vocal-tiers.js';

const QUICK_JUMPS = [
  { label: '0.4s 前奏鼓贝斯律动', sec: 0.35 },
  { label: '22.8s 主歌一：刚下的地铁', sec: 22.8 },
  { label: '61.5s 副歌：秒针转动 DI DI DA', sec: 61.5 },
  { label: '76.1s 下个路口再见吧', sec: 76.1 },
  { label: '101.0s 主歌二：鼓组抽离段', sec: 101.0 },
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

function reconstructNoteMidiKnots(baseMidiQ, f0Steps, unvoicedOnly) {
  if (unvoicedOnly || !baseMidiQ) return [60.0];
  const knots = [baseMidiQ * 0.02];
  let cur = knots[0];
  if (Array.isArray(f0Steps)) {
    for (let i = 0; i < f0Steps.length; i++) {
      cur += f0Steps[i] * 0.025;
      knots.push(cur);
    }
  }
  return knots;
}

function getNoteMeanMidi(midiSpec) {
  if (!Array.isArray(midiSpec) || midiSpec.length === 0) return 62.0;
  let sum = 0;
  for (let i = 0; i < midiSpec.length; i++) sum += midiSpec[i];
  return sum / midiSpec.length;
}

function buildTierNotes(vocalScore) {
  const frameSec = 256 / 48000;
  return vocalScore.map((row) => {
    const unvoicedOnly = (row[5] & 8) !== 0;
    const midiSpec = reconstructNoteMidiKnots(row[8], row[9], unvoicedOnly);
    return {
      startSec: row[0] * frameSec,
      endSec: (row[0] + row[1]) * frameSec,
      hanzi: row[3],
      pinyin: row[4],
      unvoicedOnly,
      midiSpec,
      meanMidi: getNoteMeanMidi(midiSpec),
      legatoNext: (row[5] & 2) !== 0,
    };
  });
}

export function mountScoreCanvas(container, store, mixer) {
  container.innerHTML = `
    <div class="canvas-header">
      <span id="canvas-tier-caption"><strong>双层实时数学总谱视窗</strong> · 上层：人声连音与逐字微表情音高曲线 (285 音符) ｜ 下层：98.0 BPM 乐队和声进行</span>
      <span id="section-tag-label">Section: Intro</span>
    </div>
    <canvas id="pure-math-score-canvas" class="score-canvas-el" width="1280" height="220"></canvas>
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
      if (!store.getState().isPlaying && store.getState().isReady && !store.getState().isSwitchingTier) {
        mixer.play(j.sec);
      }
    });
    jumpStrip.appendChild(btn);
  }

  const chordBlocks = buildChordBlocks(BAND_SCORE);
  const notesByTier = {};
  for (const [tid, entry] of Object.entries(VOCAL_TIERS)) {
    notesByTier[tid] = buildTierNotes(entry.score);
  }

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
    const st = store.getState();
    const activeTier = VOCAL_TIERS[st.vocalTier] ? st.vocalTier : Object.keys(VOCAL_TIERS)[0];
    const prof = VOCAL_TIERS[activeTier].profile;
    const noteItems = notesByTier[activeTier] || [];

    captionEl.innerHTML = `<strong>双层实时数学总谱视窗</strong> · 当前：${prof.shortLabel} (${prof.pcaDims}维PCA, f0步长=${prof.f0KnotStepFrames}帧) ｜ 点击画布任意处瞬时跳转`;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth || 1200;
    const cssH = canvas.clientHeight || 220;
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

    const vocalBottom = cssH - 54;
    const midiMin = 50;
    const midiMax = 74;
    const yOfMidi = (m) => {
      const u = Math.max(0, Math.min(1, (m - midiMin) / (midiMax - midiMin)));
      return vocalBottom - 14 - u * (vocalBottom - 30);
    };

    // Draw Bar & Beat grid lines (98.0 BPM)
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

      ctx.fillStyle = 'rgba(255,255,255,0.34)';
      ctx.font = '10px monospace';
      ctx.fillText(`B${b + 1}`, xb + 4, 13);
    }

    // Draw Bottom Band Chord Progression Blocks
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
        ctx.fillText(blk.chord, x0 + 7, vocalBottom + 29);
      }
    }
    secLabel.textContent = `Section: ${activeSecName}`;

    // Draw Vocal Notes + `--` Coarticulation Legato Bridges
    for (let i = 0; i < noteItems.length; i++) {
      const n = noteItems[i];
      if (n.endSec < tLeft - 0.3 || n.startSec > tRight + 0.3) continue;
      const x0 = xOf(n.startSec);
      const x1 = Math.max(x0 + 5, xOf(n.endSec));
      const yMid = yOfMidi(n.meanMidi);
      const isSungNow = tNow >= n.startSec && tNow <= n.endSec;

      if (n.legatoNext && i + 1 < noteItems.length) {
        const nxt = noteItems[i + 1];
        if (nxt.startSec - n.endSec < 0.22) {
          const yn = yOfMidi(nxt.meanMidi);
          ctx.strokeStyle = 'rgba(232, 93, 63, 0.65)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x1, yMid);
          ctx.bezierCurveTo((x1 + xOf(nxt.startSec)) * 0.5, yMid, (x1 + xOf(nxt.startSec)) * 0.5, yn, xOf(nxt.startSec), yn);
          ctx.stroke();
        }
      }

      ctx.fillStyle = n.unvoicedOnly
        ? 'rgba(42, 157, 143, 0.78)'
        : (isSungNow ? '#ff7b5c' : 'rgba(217, 72, 59, 0.78)');
      ctx.fillRect(x0, yMid - 6, x1 - x0, 12);

      if (Array.isArray(n.midiSpec) && n.midiSpec.length > 1) {
        ctx.strokeStyle = '#ffd166';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        for (let k = 0; k < n.midiSpec.length; k++) {
          const px = x0 + (k / (n.midiSpec.length - 1)) * (x1 - x0);
          const py = yOfMidi(n.midiSpec[k]);
          if (k === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }

      if (x1 - x0 >= 14) {
        ctx.fillStyle = isSungNow ? '#ffffff' : '#e2e8f0';
        ctx.font = '12px sans-serif';
        ctx.fillText(n.hanzi, x0 + 2, yMid - 10);
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
    if (keys.includes('currentTime') || keys.includes('isReady') || keys.includes('vocalTier')) {
      renderFrame(state.currentTime);
    }
  });
  window.addEventListener('resize', () => renderFrame(store.getState().currentTime));
}
