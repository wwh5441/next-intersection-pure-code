// V9R1 10-Track Orthogonal Physical Stem Mixer Console (`src/components/stem-mixer.js`)
import { STEM_MANIFEST } from '../dsp/v9r1-renderer.js';
import { studioStore } from '../state/studio-store.js';
import { audioMixer } from '../modules/audio-mixer.js';

export function mountStemMixer(container) {
  container.innerHTML = `
    <div class="section-heading">
      <div>
        <h2 class="section-title">10 轨正交物理分轨调音台 (5 轨声学人声 + 5 轨字典伴奏)</h2>
        <div class="section-sub">单声道干声/贝斯采用原生 1ch AudioBuffer (省内存 20%) · 支持任意单轨 Solo / Mute 独立试听</div>
      </div>
      <button type="button" class="reset-mixer-btn" id="v9r1-reset-mixer">重置全部推子与独奏</button>
    </div>

    <div class="mixer-groups">
      <div class="mixer-group-col">
        <div class="mixer-group-header">
          <span>VOCAL GROUP · 97音节声韵字典 + 李宇春音色核 (01–05)</span>
          <span class="mono-num">5 Stems</span>
        </div>
        <div class="stem-rows" id="v9r1-vocal-stems"></div>
      </div>

      <div class="mixer-group-col">
        <div class="mixer-group-header">
          <span>BAND GROUP · 86小节 PAM 物理原型字典 (06–10)</span>
          <span class="mono-num">5 Stems</span>
        </div>
        <div class="stem-rows" id="v9r1-band-stems"></div>
      </div>
    </div>
  `;

  const vocalCol = container.querySelector('#v9r1-vocal-stems');
  const bandCol = container.querySelector('#v9r1-band-stems');
  const resetBtn = container.querySelector('#v9r1-reset-mixer');

  resetBtn.addEventListener('click', () => {
    audioMixer.resetAllStems();
  });

  const stripRefs = new Map();

  for (const m of STEM_MANIFEST) {
    const row = document.createElement('div');
    row.className = 'stem-strip';
    row.dataset.stemId = m.id;
    row.innerHTML = `
      <div class="stem-num" style="background:${m.color}">${m.num}</div>
      <div class="stem-info">
        <div class="stem-name-row">
          <span class="stem-name">${m.name}</span>
          <span class="ch-badge">${m.channels === 1 ? '1ch Mono' : '2ch Stereo'}</span>
        </div>
        <div class="stem-formula">${m.formula}</div>
        <div class="stem-meter-bar">
          <div class="stem-meter-fill" style="background:${m.color}"></div>
        </div>
      </div>
      <div class="stem-btns">
        <button type="button" class="sm-btn btn-solo" title="独奏 Solo">S</button>
        <button type="button" class="sm-btn btn-mute" title="静音 Mute">M</button>
      </div>
      <input type="range" class="stem-fader" min="0" max="1.5" step="0.01" value="1" aria-label="${m.name} 音量" />
      <div class="stem-gain-label">100%</div>
    `;

    const soloBtn = row.querySelector('.btn-solo');
    const muteBtn = row.querySelector('.btn-mute');
    const fader = row.querySelector('.stem-fader');
    const gainLabel = row.querySelector('.stem-gain-label');
    const meterFill = row.querySelector('.stem-meter-fill');

    soloBtn.addEventListener('click', () => audioMixer.toggleSolo(m.id));
    muteBtn.addEventListener('click', () => audioMixer.toggleMute(m.id));
    fader.addEventListener('input', () => {
      audioMixer.setStemGain(m.id, Number(fader.value));
    });

    if (m.group === 'vocal') vocalCol.appendChild(row);
    else bandCol.appendChild(row);

    stripRefs.set(m.id, { row, soloBtn, muteBtn, fader, gainLabel, meterFill });
  }

  const sync = (st) => {
    const anySolo = Object.values(st.stemStates).some((s) => s.solo);
    const levels = audioMixer.getLiveLevels();

    for (const m of STEM_MANIFEST) {
      const ref = stripRefs.get(m.id);
      const sState = st.stemStates[m.id];
      if (!ref || !sState) continue;

      const silenced = sState.muted || (anySolo && !sState.solo);
      ref.row.classList.toggle('is-silenced', silenced);
      ref.soloBtn.classList.toggle('is-solo', sState.solo);
      ref.muteBtn.classList.toggle('is-mute', sState.muted);

      if (document.activeElement !== ref.fader) {
        ref.fader.value = String(sState.gain);
      }
      ref.gainLabel.textContent = `${Math.round(sState.gain * 100)}%`;

      const rms = levels.stems[m.id] || 0;
      const pct = Math.min(100, Math.sqrt(rms * 8.5) * 100);
      ref.meterFill.style.width = `${pct.toFixed(1)}%`;
    }
  };

  sync(studioStore.getState());
  studioStore.subscribe(sync);
}
