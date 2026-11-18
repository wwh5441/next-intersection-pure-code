// CabronWEC (CWC-5X) Interactive Acoustic Viewport Canvas
// Renders:
// 1. Upper lane: 420-bin song energy contour + loaded chunk regions + interactive seek playhead
// 2. Lower-left lane: 32-band perceptual Bark spectrum analyzer
// 3. Lower-right lane: Real-time stereo time-domain waveform oscilloscope

import { CWC_CHUNKS, CWC_OVERVIEW, CWC_TRACK_META } from '../../data/cwc-models.js';
import { cwcEngine } from '../modules/cwc-audio-engine.js';
import { cwcStore } from '../state/cwc-store.js';

function fmtSec(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = (sec % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

export function mountStageCanvas(canvasEl, crosshairHudEl) {
  const ctx = canvasEl.getContext('2d');
  let cssW = 800;
  let cssH = 340;
  let dpr = 1;
  let hoverX = -1;
  let rafScheduled = false;

  const scheduleDraw = () => {
    if (rafScheduled) return;
    rafScheduled = true;
    requestAnimationFrame(() => {
      rafScheduled = false;
      draw();
    });
  };

  const resize = () => {
    const rect = canvasEl.getBoundingClientRect();
    cssW = Math.max(260, rect.width || 800);
    cssH = Math.max(220, rect.height || 340);
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvasEl.width = Math.round(cssW * dpr);
    canvasEl.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scheduleDraw();
  };

  const ro = new ResizeObserver(resize);
  ro.observe(canvasEl);

  canvasEl.addEventListener('pointermove', (ev) => {
    const r = canvasEl.getBoundingClientRect();
    hoverX = Math.max(0, Math.min(cssW, ev.clientX - r.left));
    const sec = (hoverX / cssW) * CWC_TRACK_META.durationSec;
    const bar = Math.floor(sec / CWC_TRACK_META.barSec) + 1;
    if (crosshairHudEl) {
      crosshairHudEl.textContent = `Seek ${fmtSec(sec)} · Bar ${bar}`;
    }
    scheduleDraw();
  });

  canvasEl.addEventListener('pointerleave', () => {
    hoverX = -1;
    if (crosshairHudEl) {
      crosshairHudEl.textContent = 'Click timeline to seek';
    }
    scheduleDraw();
  });

  canvasEl.addEventListener('click', (ev) => {
    const r = canvasEl.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width)));
    cwcEngine.seek(frac * CWC_TRACK_META.durationSec);
  });

  const draw = () => {
    const st = cwcStore.getState();
    const isNarrow = cssW < 520;

    ctx.fillStyle = '#07090E';
    ctx.fillRect(0, 0, cssW, cssH);

    const topH = Math.round(cssH * 0.44);
    const botTop = topH + 10;
    const botH = cssH - botTop - 8;

    // 1. Loaded chunk region shading in overview lane
    for (let ci = 0; ci < CWC_CHUNKS.length; ci++) {
      const ch = CWC_CHUNKS[ci];
      const x0 = (ch.startSec / CWC_TRACK_META.durationSec) * cssW;
      const x1 = (ch.endSec / CWC_TRACK_META.durationSec) * cssW;
      if (st.chunksLoaded[ci]) {
        ctx.fillStyle = ci % 2 === 0 ? 'rgba(6, 182, 212, 0.045)' : 'rgba(56, 189, 248, 0.028)';
        ctx.fillRect(x0, 0, x1 - x0, topH);
      }
      if (ci > 0) {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, 0);
        ctx.lineTo(x0, topH);
        ctx.stroke();
      }
    }

    // 2. Overview waveform bars
    const ov = CWC_OVERVIEW;
    const nOv = ov.bins;
    const playFrac = Math.max(0, Math.min(1, st.currentTimeSec / CWC_TRACK_META.durationSec));
    const midOvY = topH * 0.54;
    const maxAmpH = topH * 0.4;
    const colW = cssW / nOv;

    for (let i = 0; i < nOv; i++) {
      const f = i / nOv;
      const pk = Math.max(0.03, ov.peak[i] || 0.04);
      const h = pk * maxAmpH;
      const x = i * colW;
      const played = f <= playFrac;
      if (played) {
        ctx.fillStyle = st.audioSource === 'ref' ? '#F59E0B' : '#06B6D4';
      } else {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.28)';
      }
      ctx.fillRect(x, midOvY - h, Math.max(1, colW - 0.6), h * 2);
    }

    // Section labels along top of overview
    ctx.font = '600 10.5px "JetBrains Mono", monospace';
    ctx.fillStyle = 'rgba(203, 213, 225, 0.72)';
    const stride = isNarrow ? 2 : 1;
    for (let ci = 0; ci < CWC_CHUNKS.length; ci += stride) {
      const ch = CWC_CHUNKS[ci];
      const x0 = (ch.startSec / CWC_TRACK_META.durationSec) * cssW + 5;
      ctx.fillText(`C${ci}`, x0, 13);
    }

    // Playhead cursor
    const playX = playFrac * cssW;
    ctx.strokeStyle = st.audioSource === 'ref' ? '#F59E0B' : '#22D3EE';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, topH);
    ctx.stroke();

    if (hoverX >= 0) {
      ctx.strokeStyle = 'rgba(248, 250, 252, 0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, 0);
      ctx.lineTo(hoverX, topH);
      ctx.stroke();
    }

    // Horizontal divider
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, topH + 1);
    ctx.lineTo(cssW, topH + 1);
    ctx.stroke();

    // 3. Bottom Split: 32-Band Bark Spectrum (left 56%) + Stereo Waveform Scope (right 44%)
    const specW = Math.round(cssW * 0.56);
    const scopeX = specW + 16;
    const scopeW = Math.max(40, cssW - scopeX - 10);

    ctx.font = '600 11px "Plus Jakarta Sans", sans-serif';
    ctx.fillStyle = '#94A3B8';
    ctx.save();
    ctx.beginPath();
    ctx.rect(8, botTop - 2, specW - 6, 18);
    ctx.clip();
    ctx.fillText(isNarrow ? '32-Band Bark' : '32-Band Bark Spectrum (20 Hz – 19.6 kHz)', 10, botTop + 10);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(scopeX, botTop - 2, scopeW, 18);
    ctx.clip();
    ctx.fillText(isNarrow ? 'Waveform' : 'Time-Domain Waveform', scopeX, botTop + 10);
    ctx.restore();

    const spec = cwcEngine.liveTelemetry.spectrum32;
    const ovIdx = Math.min(nOv - 1, Math.floor(playFrac * nOv));
    const barGap = isNarrow ? 2 : 3;
    const sbW = Math.max(2, (specW - 20 - barGap * 31) / 32);
    const specAreaTop = botTop + 20;
    const specAreaH = botH - 24;

    for (let b = 0; b < 32; b++) {
      let val = spec[b];
      if (!st.isPlaying && val <= 0.001) {
        const contour = b < 9 ? ov.low[ovIdx] : b < 21 ? ov.vocal[ovIdx] : ov.air[ovIdx];
        val = Math.max(0.05, Math.min(0.85, contour * (0.85 - 0.12 * Math.sin(b * 0.7))));
      }
      const clamped = Math.max(0.03, Math.min(1, val));
      const h = clamped * specAreaH;
      const bx = 10 + b * (sbW + barGap);
      const by = specAreaTop + specAreaH - h;
      if (b < 9) ctx.fillStyle = '#06B6D4';
      else if (b < 21) ctx.fillStyle = '#F59E0B';
      else ctx.fillStyle = '#A78BFA';
      ctx.fillRect(bx, by, sbW, h);
    }

    // Vertical separator
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.beginPath();
    ctx.moveTo(specW + 6, botTop);
    ctx.lineTo(specW + 6, cssH - 8);
    ctx.stroke();

    // Oscilloscope waveform
    const td = cwcEngine.liveTelemetry.timeDomain;
    const midY = specAreaTop + specAreaH * 0.5;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.16)';
    ctx.beginPath();
    ctx.moveTo(scopeX, midY);
    ctx.lineTo(scopeX + scopeW, midY);
    ctx.stroke();

    ctx.strokeStyle = st.audioSource === 'ref' ? '#F59E0B' : '#06B6D4';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const stepX = scopeW / Math.max(1, td.length - 1);
    for (let i = 0; i < td.length; i++) {
      const px = scopeX + i * stepX;
      const py = midY - Math.max(-1, Math.min(1, td[i] * 1.6)) * (specAreaH * 0.45);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    if (st.isPlaying) scheduleDraw();
  };

  cwcStore.subscribe(() => scheduleDraw());
  resize();
}
