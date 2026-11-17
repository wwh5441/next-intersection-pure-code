// High-DPI Interactive Acoustic Field Canvas
// Top lane: 210s multi-band song overview + 8 chunk boundaries + click/drag seek crosshair
// Bottom lane: Live 32-band Bark analyzer + Stereo time-domain oscilloscope

import { CWC_CHUNKS, CWC_OVERVIEW, CWC_TRACK_META } from '../../data/cwc-models.js';
import { cwcEngine } from '../modules/cwc-audio-engine.js';
import { cwcStore } from '../state/cwc-store.js';

export function mountStageCanvas(canvasEl, hudEl) {
  const ctx = canvasEl.getContext('2d');
  let cssW = 800;
  let cssH = 340;
  let dpr = 1;
  let hoverX = null;
  let isDragging = false;

  let rafId = null;
  const scheduleDraw = () => {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(drawFrame);
  };

  const resize = () => {
    const rect = canvasEl.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    cssW = rect.width;
    cssH = rect.height;
    canvasEl.width = Math.round(cssW * dpr);
    canvasEl.height = Math.round(cssH * dpr);
    scheduleDraw();
  };

  const ro = new ResizeObserver(resize);
  ro.observe(canvasEl);

  const seekFromClientX = (clientX) => {
    const rect = canvasEl.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
    const targetSec = frac * CWC_TRACK_META.durationSec;
    cwcEngine.seek(targetSec);
  };

  canvasEl.addEventListener('pointerdown', (ev) => {
    isDragging = true;
    hudEl.classList.add('is-active');
    canvasEl.setPointerCapture(ev.pointerId);
    seekFromClientX(ev.clientX);
  });

  canvasEl.addEventListener('pointermove', (ev) => {
    const rect = canvasEl.getBoundingClientRect();
    hoverX = Math.max(0, Math.min(rect.width, ev.clientX - rect.left));
    const frac = hoverX / Math.max(1, rect.width);
    const sec = frac * CWC_TRACK_META.durationSec;
    const bar = Math.floor(sec / CWC_TRACK_META.barSec) + 1;
    const chunkIdx = Math.min(7, Math.floor(frac * CWC_CHUNKS.length));
    const secName = CWC_CHUNKS[chunkIdx]?.sectionTitle || '';
    const mm = String(Math.floor(sec / 60)).padStart(2, '0');
    const ss = (sec % 60).toFixed(2).padStart(5, '0');
    hudEl.classList.add('is-active');
    hudEl.textContent = `${mm}:${ss} · Bar ${bar} · ${secName}`;
    if (isDragging) seekFromClientX(ev.clientX);
  });

  canvasEl.addEventListener('pointerup', (ev) => {
    if (isDragging) {
      isDragging = false;
      canvasEl.releasePointerCapture(ev.pointerId);
    }
  });

  canvasEl.addEventListener('pointerleave', () => {
    hoverX = null;
    isDragging = false;
    hudEl.classList.remove('is-active');
    hudEl.textContent = 'Click timeline to seek';
  });

  const drawFrame = () => {
    rafId = null;
    if (cssW <= 0 || cssH <= 0) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background clear
    ctx.fillStyle = '#07090E';
    ctx.fillRect(0, 0, cssW, cssH);

    const splitY = Math.round(cssH * 0.44);
    const isNarrow = cssW < 540;

    // Subtle grid lines
    ctx.strokeStyle = '#151D2B';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, splitY);
    ctx.lineTo(cssW, splitY);
    for (let i = 1; i < 8; i++) {
      const cx = (i / 8) * cssW;
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, splitY);
    }
    ctx.stroke();

    // 1) Top Lane: 210s Song Spectral Overview (RMS energy + sculpted peak envelope)
    const ov = CWC_OVERVIEW;
    const nOv = ov.bins;
    const barW = cssW / nOv;
    const st = cwcStore.getState();
    const playFrac = Math.max(0, Math.min(1, st.currentTimeSec / Math.max(0.01, CWC_TRACK_META.durationSec)));
    const playX = playFrac * cssW;

    for (let i = 0; i < nOv; i++) {
      const x = i * barW;
      const played = x <= playX;
      const pk = Math.min(1, ov.peak[i]);
      const energy = Math.min(1, ov.low[i] * 0.95 + ov.vocal[i] * 1.45 + ov.air[i] * 0.85);
      const shaped = Math.min(1, 0.22 * Math.pow(pk, 2.2) + 0.78 * Math.pow(energy, 0.85));
      const voc = Math.min(1, ov.vocal[i] * 1.85);
      const air = Math.min(1, ov.air[i] * 1.4);

      const hTot = Math.max(2, shaped * (splitY - 24));
      const hVoc = Math.min(hTot, voc * (splitY - 26) * 0.72);

      // Base low/body energy envelope
      ctx.fillStyle = played ? 'rgba(6, 182, 212, 0.42)' : 'rgba(100, 116, 139, 0.26)';
      ctx.fillRect(x, splitY - 6 - hTot, Math.max(1, barW - 0.5), hTot);

      // Vocal formant layer highlight
      if (hVoc > 1) {
        ctx.fillStyle = played ? 'rgba(245, 158, 11, 0.82)' : 'rgba(245, 158, 11, 0.34)';
        ctx.fillRect(x, splitY - 6 - hVoc, Math.max(1, barW - 0.5), hVoc);
      }

      // High air layer cap
      if (air > 0.08) {
        ctx.fillStyle = played ? 'rgba(167, 139, 250, 0.82)' : 'rgba(167, 139, 250, 0.30)';
        ctx.fillRect(x, splitY - 6 - hTot, Math.max(1, barW - 0.5), 1.5);
      }
    }

    // Chunk index labels along top edge
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#64748B';
    for (let i = 0; i < CWC_CHUNKS.length; i++) {
      const cx = (i / CWC_CHUNKS.length) * cssW + 6;
      ctx.fillText(`C${i}`, cx, 13);
    }

    // Playhead cursor line
    ctx.strokeStyle = st.audioSource === 'ref' ? '#F59E0B' : '#06B6D4';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, splitY);
    ctx.stroke();

    // Hover crosshair
    if (hoverX !== null) {
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hoverX, 0);
      ctx.lineTo(hoverX, splitY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2) Bottom Lane: Left 57% = 32-Band Bark Spectrum | Right 43% = Time-Domain Oscilloscope
    const botTop = splitY + 12;
    const botH = cssH - botTop - 12;
    const specW = Math.round(cssW * 0.57);
    const scopeX = specW + 18;
    const scopeW = cssW - scopeX - 10;

    // Clipped pane labels so narrow mobile widths never collide
    ctx.font = isNarrow
      ? '10px ui-monospace, SFMono-Regular, Menlo, monospace'
      : '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#94A3B8';

    ctx.save();
    ctx.beginPath();
    ctx.rect(8, botTop - 2, specW - 6, 18);
    ctx.clip();
    ctx.fillText(isNarrow ? '32-Band Bark' : '32-Band Bark Spectrum (20 Hz – 19.6 kHz)', 10, botTop + 10);
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(scopeX, botTop - 2, Math.max(0, scopeW), 18);
    ctx.clip();
    ctx.fillText(isNarrow ? 'Waveform' : 'Time-Domain Waveform', scopeX, botTop + 10);
    ctx.restore();

    // Spectrum bars (uses live Worklet telemetry when playing, or overview contour at playhead when paused)
    const spec = cwcEngine.liveTelemetry.spectrum32;
    const ovIdx = Math.min(nOv - 1, Math.floor(playFrac * nOv));
    const barGap = isNarrow ? 2 : 3;
    const sbW = Math.max(2, (specW - 20 - barGap * 31) / 32);
    const specAreaTop = botTop + 20;
    const specAreaH = botH - 24;

    for (let b = 0; b < 32; b++) {
      let val = spec[b];
      if (!st.isPlaying && val <= 0.001) {
        const contour = b < 9 ? ov.low[ovIdx] : (b < 21 ? ov.vocal[ovIdx] : ov.air[ovIdx]);
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

    // Separator
    ctx.strokeStyle = '#1E293B';
    ctx.beginPath();
    ctx.moveTo(specW + 6, botTop);
    ctx.lineTo(specW + 6, cssH - 8);
    ctx.stroke();

    // Oscilloscope waveform
    const td = cwcEngine.liveTelemetry.timeDomain;
    const midY = specAreaTop + specAreaH * 0.5;
    ctx.strokeStyle = '#1E293B';
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
  canvasEl.addEventListener('pointermove', () => scheduleDraw());
  canvasEl.addEventListener('pointerleave', () => scheduleDraw());
  resize();
}
