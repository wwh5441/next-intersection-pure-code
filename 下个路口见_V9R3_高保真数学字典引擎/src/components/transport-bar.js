// Studio Transport, Tier Selector, Section Jump & GCI Vocal Pitch-Shift Bar (`src/components/transport-bar.js`)
import { SONG_STRUCTURE } from '../../data/song-structure.js';
import { V9R3_TIERS } from '../../data/tiers-registry.js';
import { audioMixer } from '../modules/audio-mixer.js';
import { studioStore } from '../state/studio-store.js';

function formatClock(sec) {
  const s = Math.max(0, sec || 0);
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(Math.floor(s % 60)).padStart(2, '0');
  const cs = String(Math.floor((s % 1) * 100)).padStart(2, '0');
  return `${mm}:${ss}.${cs}`;
}

export function mountTransportBar(rootEl) {
  const tierOptions = Object.values(V9R3_TIERS)
    .map((t) => `<option value="${t.id}">${t.title}</option>`)
    .join('');

  const pitchOpts = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    .map((st) => {
      const label = st === 0 ? '0 st (原调母带 1:1)' : `${st > 0 ? '+' : ''}${st} 半音 (GCI周期变调验证)`;
      return `<option value="${st}">${label}</option>`;
    })
    .join('');

  const secButtons = SONG_STRUCTURE.sections
    .map((sec, idx) => {
      const startSec = SONG_STRUCTURE.bar0Sec + sec.startBar * SONG_STRUCTURE.barDurationSec;
      return `<button class="sec-pill" data-sec-idx="${idx}" data-sec-time="${startSec.toFixed(3)}">${sec.zh} (B${sec.startBar + 1})</button>`;
    })
    .join('');

  rootEl.innerHTML = `
    <div class="transport-row">
      <div class="transport-left">
        <button class="btn-primary" id="btn-play-toggle">播放 48kHz 母带</button>
        <button class="btn-secondary" id="btn-stop">停止归零</button>
        <div class="time-readout" id="clock-readout">00:00.00 / 03:30.37</div>
      </div>
      <div class="transport-right">
        <label class="pitch-control-wrap" title="直接改变声门周期重排间距以验证非死波形拷贝，清辅音时长保持不变">
          <span>主唱周期变调验证:</span>
          <select class="pitch-select" id="select-vocal-pitch">${pitchOpts}</select>
        </label>
        <select class="tier-select" id="select-tier">${tierOptions}</select>
      </div>
    </div>
    <div class="synth-status-bar">
      <div class="progress-track"><div class="progress-fill" id="synth-prog-fill"></div></div>
      <div class="status-text" id="synth-status-txt">准备中...</div>
    </div>
    <div class="section-pills" id="section-pills-wrap">${secButtons}</div>
  `;

  const btnPlay = rootEl.querySelector('#btn-play-toggle');
  const btnStop = rootEl.querySelector('#btn-stop');
  const clockEl = rootEl.querySelector('#clock-readout');
  const selTier = rootEl.querySelector('#select-tier');
  const selPitch = rootEl.querySelector('#select-vocal-pitch');
  const progFill = rootEl.querySelector('#synth-prog-fill');
  const statusTxt = rootEl.querySelector('#synth-status-txt');
  const pillEls = Array.from(rootEl.querySelectorAll('.sec-pill'));

  btnPlay.addEventListener('click', async () => {
    const st = studioStore.getState();
    if (st.isSynthesizing) return;
    if (st.isPlaying) {
      audioMixer.pause();
    } else {
      await audioMixer.play();
    }
  });

  btnStop.addEventListener('click', () => audioMixer.stop());

  selTier.addEventListener('change', (e) => {
    audioMixer.loadTier(e.target.value);
  });

  selPitch.addEventListener('change', (e) => {
    audioMixer.setVocalPitchSemitones(Number(e.target.value));
  });

  for (const pill of pillEls) {
    pill.addEventListener('click', async () => {
      const targetSec = Number(pill.dataset.secTime || 0);
      await audioMixer.seek(targetSec);
      if (!studioStore.getState().isPlaying && !studioStore.getState().isSynthesizing) {
        await audioMixer.play();
      }
    });
  }

  function syncUI(st) {
    btnPlay.disabled = st.isSynthesizing;
    btnStop.disabled = st.isSynthesizing;
    selTier.disabled = st.isSynthesizing;
    selPitch.disabled = st.isSynthesizing;

    btnPlay.textContent = st.isSynthesizing
      ? '正在合成...'
      : st.isPlaying
      ? '暂停播放'
      : '播放 48kHz 母带';

    if (selTier.value !== st.tierId) selTier.value = st.tierId;
    if (Number(selPitch.value) !== st.pitchSemitones) selPitch.value = String(st.pitchSemitones);

    clockEl.textContent = `${formatClock(st.currentTime)} / ${formatClock(st.duration)}`;
    progFill.style.width = `${Math.round(st.synthProgress * 100)}%`;
    statusTxt.textContent = st.synthStatus;

    const curBar = (st.currentTime - SONG_STRUCTURE.bar0Sec) / SONG_STRUCTURE.barDurationSec;
    for (let i = 0; i < SONG_STRUCTURE.sections.length; i++) {
      const sec = SONG_STRUCTURE.sections[i];
      const active = curBar >= sec.startBar && curBar < sec.startBar + sec.bars;
      pillEls[i].classList.toggle('active', active);
    }
  }

  syncUI(studioStore.getState());
  studioStore.subscribe(syncUI);
}
