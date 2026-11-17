// 10-Track Independent Solo / Mute / Fader Console (`src/components/stem-mixer.js`)
import { V9R3_STEM_SPECS } from '../dsp/engine-orchestrator.js';
import { audioMixer } from '../modules/audio-mixer.js';
import { studioStore } from '../state/studio-store.js';

export function mountStemMixer(rootEl) {
  const stripsHtml = V9R3_STEM_SPECS.map(
    (s) => `
      <div class="stem-strip" data-stem-strip="${s.id}">
        <div class="stem-top">
          <span class="stem-idx" style="background:${s.color}">${s.index}</span>
          <span class="stem-name" title="${s.name}">${s.name}</span>
          <div class="sm-btns">
            <button class="sm-btn" data-solo="${s.id}" title="独奏该分轨">S</button>
            <button class="sm-btn" data-mute="${s.id}" title="静音该分轨">M</button>
          </div>
        </div>
        <div class="stem-fader-row">
          <input class="stem-fader" type="range" min="0" max="150" value="100" data-fader="${s.id}" />
          <span class="stem-gain-txt" data-gain-txt="${s.id}">100%</span>
        </div>
        <div class="meter-bar">
          <div class="meter-fill" data-meter="${s.id}" style="background:${s.color}"></div>
        </div>
      </div>
    `
  ).join('');

  rootEl.innerHTML = `
    <div class="section-head">
      <div>
        <h2 class="section-title">10 轨独立纯数学字典混音台 (Independent 10-Stem DAW Console)</h2>
        <span class="section-sub">每一轨均由各自独立的物理/音乐字典合成，可单独 Solo 检验音质纯净度</span>
      </div>
      <div class="mixer-actions">
        <button class="chip-btn" data-preset="all">全部还原 (10轨全开)</button>
        <button class="chip-btn" data-preset="vocal">独奏李宇春主唱 (01-03)</button>
        <button class="chip-btn" data-preset="harm">独奏和声与FDN空间 (04-05)</button>
        <button class="chip-btn" data-preset="rhythm">独奏98BPM鼓+贝斯 (06-08)</button>
        <button class="chip-btn" data-preset="chord">独奏木吉他+Rhodes (09-10)</button>
      </div>
    </div>
    <div class="stems-grid">${stripsHtml}</div>
  `;

  rootEl.querySelectorAll('[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.preset;
      if (p === 'all') studioStore.setStemPreset(null);
      else if (p === 'vocal') studioStore.setStemPreset(['vocal_lead_chest', 'vocal_lead_formant', 'vocal_breath_air']);
      else if (p === 'harm') studioStore.setStemPreset(['vocal_chorus_harmony', 'studio_fdn_space']);
      else if (p === 'rhythm') studioStore.setStemPreset(['drum_kick_snare', 'drum_hihat_perc', 'bass_groove']);
      else if (p === 'chord') studioStore.setStemPreset(['guitar_funk_riff', 'keys_rhodes_pad']);
    });
  });

  rootEl.querySelectorAll('[data-solo]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.solo;
      const cur = studioStore.getState().stemStates[id];
      studioStore.updateStem(id, { solo: !cur.solo });
    });
  });

  rootEl.querySelectorAll('[data-mute]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.mute;
      const cur = studioStore.getState().stemStates[id];
      studioStore.updateStem(id, { mute: !cur.mute });
    });
  });

  rootEl.querySelectorAll('[data-fader]').forEach((inp) => {
    inp.addEventListener('input', () => {
      const id = inp.dataset.fader;
      studioStore.updateStem(id, { gain: Number(inp.value) / 100.0 });
    });
  });

  const stripEls = new Map();
  for (const s of V9R3_STEM_SPECS) {
    stripEls.set(s.id, {
      strip: rootEl.querySelector(`[data-stem-strip="${s.id}"]`),
      soloBtn: rootEl.querySelector(`[data-solo="${s.id}"]`),
      muteBtn: rootEl.querySelector(`[data-mute="${s.id}"]`),
      fader: rootEl.querySelector(`[data-fader="${s.id}"]`),
      gainTxt: rootEl.querySelector(`[data-gain-txt="${s.id}"]`),
      meter: rootEl.querySelector(`[data-meter="${s.id}"]`),
    });
  }

  function syncStems(st) {
    const anySolo = Object.values(st.stemStates).some((x) => x.solo);
    for (const s of V9R3_STEM_SPECS) {
      const ui = stripEls.get(s.id);
      const item = st.stemStates[s.id];
      const audible = anySolo ? item.solo && !item.mute : !item.mute;
      ui.strip.classList.toggle('muted-effective', !audible);
      ui.soloBtn.classList.toggle('solo-on', item.solo);
      ui.muteBtn.classList.toggle('mute-on', item.mute);
      const pct = Math.round(item.gain * 100);
      if (Number(ui.fader.value) !== pct) ui.fader.value = String(pct);
      ui.gainTxt.textContent = `${pct}%`;
    }
  }

  syncStems(studioStore.getState());
  studioStore.subscribe(syncStems);

  function tickMeters() {
    for (const s of V9R3_STEM_SPECS) {
      const ui = stripEls.get(s.id);
      const lvl = audioMixer.getStemRmsLevel(s.id);
      ui.meter.style.width = `${Math.round(lvl * 100)}%`;
    }
    requestAnimationFrame(tickMeters);
  }
  requestAnimationFrame(tickMeters);
}
