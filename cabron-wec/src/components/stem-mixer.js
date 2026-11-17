// 5-Layer Real-Time Orthogonal Stem Matrix Component

import { cwcEngine } from '../modules/cwc-audio-engine.js';
import { cwcStore } from '../state/cwc-store.js';

export function mountStemMatrix(containerEl) {
  const renderAll = () => {
    const { stems } = cwcStore.getState();
    const anySolo = stems.some((s) => s.solo);

    containerEl.innerHTML = stems
      .map((s, idx) => {
        const dimmed = (anySolo && !s.solo) || s.muted;
        const pct = Math.round(s.gain * 100);
        return `
          <div class="cwc-stem-row ${dimmed ? 'is-dimmed' : ''}" data-stem-idx="${idx}">
            <div class="cwc-stem-top">
              <div class="cwc-stem-identity">
                <span class="cwc-stem-name">${s.code} · ${s.name}</span>
                <span class="cwc-stem-band tabular">${s.band}</span>
              </div>
              <div class="cwc-stem-actions">
                <button
                  type="button"
                  class="cwc-ms-btn ${s.muted ? 'mute-on' : ''}"
                  data-action="mute"
                  data-idx="${idx}"
                  aria-pressed="${s.muted}"
                  aria-label="Mute ${s.code} ${s.name}"
                >M</button>
                <button
                  type="button"
                  class="cwc-ms-btn ${s.solo ? 'solo-on' : ''}"
                  data-action="solo"
                  data-idx="${idx}"
                  aria-pressed="${s.solo}"
                  aria-label="Solo ${s.code} ${s.name}"
                >S</button>
              </div>
            </div>
            <div class="cwc-stem-controls">
              <input
                type="range"
                class="cwc-slider"
                min="0"
                max="150"
                value="${pct}"
                data-action="gain"
                data-idx="${idx}"
                aria-label="Gain for ${s.code} ${s.name}"
              />
              <span class="cwc-gain-val tabular" data-gain-label="${idx}">${pct}%</span>
            </div>
            <div class="cwc-rms-track" aria-hidden="true">
              <div class="cwc-rms-fill" data-rms-bar="${idx}" style="background:${s.color};width:${Math.round(s.rms * 100)}%"></div>
            </div>
          </div>
        `;
      })
      .join('');
  };

  renderAll();

  containerEl.addEventListener('click', (ev) => {
    const btn = ev.target.closest('button[data-action]');
    if (!btn) return;
    const idx = Number(btn.dataset.idx);
    if (btn.dataset.action === 'mute') cwcEngine.toggleStemMute(idx);
    else if (btn.dataset.action === 'solo') cwcEngine.toggleStemSolo(idx);
    renderAll();
  });

  containerEl.addEventListener('input', (ev) => {
    const slider = ev.target.closest('input[data-action="gain"]');
    if (!slider) return;
    const idx = Number(slider.dataset.idx);
    const val = Number(slider.value) / 100;
    cwcEngine.setStemGain(idx, val);
    const lbl = containerEl.querySelector(`[data-gain-label="${idx}"]`);
    if (lbl) lbl.textContent = `${slider.value}%`;
  });

  // Update RMS bars cleanly when store telemetry updates (0% idle CPU)
  cwcStore.subscribe((st) => {
    for (let i = 0; i < 5; i++) {
      const bar = containerEl.querySelector(`[data-rms-bar="${i}"]`);
      if (bar) bar.style.width = `${Math.min(100, Math.round((st.stems[i]?.rms || 0) * 100))}%`;
    }
  });

  return { refresh: renderAll };
}
