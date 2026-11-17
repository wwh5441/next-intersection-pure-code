// V9R1 Transport, Section Jumper, Live `--` Coarticulation & Active Bar-Motif Readout (`src/components/transport-bar.js`)
import { SINGER_CORE } from '../../models/singer-core.js';
import { SONG_STRUCTURE } from '../../data/song-structure.js';
import { VOCAL_LYRIC_LINES } from '../../data/vocal-lyrics.js';
import { V9R1_TIERS } from '../../data/tiers-registry.js';
import { studioStore } from '../state/studio-store.js';
import { audioMixer } from '../modules/audio-mixer.js';

function formatClock(sec) {
  const s = Math.max(0, Math.floor(sec));
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

export function mountTransportBar(container) {
  const frameDurSec = SINGER_CORE.hopSamples / SINGER_CORE.sampleRate;

  container.innerHTML = `
    <div class="transport-row">
      <button type="button" class="play-btn" id="v9r1-play-btn" disabled>合成中...</button>
      <div class="time-readout mono-num" id="v9r1-time-readout">00:00 / 03:30</div>
      <input type="range" class="scrub-slider" id="v9r1-scrubber" min="0" max="${SINGER_CORE.durationSec.toFixed(2)}" step="0.05" value="0" />
      <div class="render-status-pill" id="v9r1-render-pill">正在初始化...</div>
    </div>

    <div class="live-badges-row">
      <div class="live-pill-box">
        <span class="live-lyric-main" id="v9r1-live-lyric">前奏 Funk 吉他分解和弦与底鼓切入...</span>
        <span class="coart-tag" id="v9r1-coart-badge">待发声 · 97音节声韵字典</span>
      </div>
      <div class="live-pill-box">
        <span class="live-lyric-main" id="v9r1-sec-chord">Intro · 前奏吉他Hook</span>
        <span class="motif-tag" id="v9r1-motif-badge">Bar #01 · D:M1 B:M1 H:--</span>
      </div>
    </div>

    <div class="section-jumps" id="v9r1-sec-jumps"></div>
  `;

  const playBtn = container.querySelector('#v9r1-play-btn');
  const timeEl = container.querySelector('#v9r1-time-readout');
  const scrubber = container.querySelector('#v9r1-scrubber');
  const renderPill = container.querySelector('#v9r1-render-pill');
  const liveLyricEl = container.querySelector('#v9r1-live-lyric');
  const coartBadgeEl = container.querySelector('#v9r1-coart-badge');
  const secChordEl = container.querySelector('#v9r1-sec-chord');
  const motifBadgeEl = container.querySelector('#v9r1-motif-badge');
  const jumpsWrap = container.querySelector('#v9r1-sec-jumps');

  let isUserScrubbing = false;

  playBtn.addEventListener('click', () => {
    audioMixer.togglePlay();
  });

  scrubber.addEventListener('mousedown', () => { isUserScrubbing = true; });
  scrubber.addEventListener('touchstart', () => { isUserScrubbing = true; }, { passive: true });
  scrubber.addEventListener('input', () => {
    timeEl.textContent = `${formatClock(Number(scrubber.value))} / ${formatClock(studioStore.getState().durationSec)}`;
  });
  const finishScrub = () => {
    if (!isUserScrubbing) return;
    isUserScrubbing = false;
    audioMixer.seek(Number(scrubber.value));
  };
  scrubber.addEventListener('change', finishScrub);
  scrubber.addEventListener('mouseup', finishScrub);
  scrubber.addEventListener('touchend', finishScrub);

  const secButtons = [];
  for (const sec of SONG_STRUCTURE.sections) {
    const startSec = Math.max(0, SONG_STRUCTURE.bar0Sec + sec.startBar * SONG_STRUCTURE.barDurationSec);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sec-pill-btn';
    btn.textContent = `${sec.zh} (${formatClock(startSec)})`;
    btn.addEventListener('click', () => {
      audioMixer.seek(startSec);
    });
    jumpsWrap.appendChild(btn);
    secButtons.push({ sec, startSec, btn });
  }

  const sync = (st) => {
    playBtn.disabled = st.isRendering;
    if (st.isRendering) {
      playBtn.textContent = `合成 ${Math.round(st.renderPct * 100)}%`;
      renderPill.classList.remove('is-ready');
      renderPill.textContent = st.renderText;
    } else {
      playBtn.textContent = st.isPlaying ? '⏸ 暂停播放' : '▶ 播放全曲';
      renderPill.classList.add('is-ready');
      renderPill.textContent = st.errorText || st.renderText;
    }

    const t = st.currentTime;
    if (!isUserScrubbing) {
      scrubber.value = t.toFixed(2);
      timeEl.textContent = `${formatClock(t)} / ${formatClock(st.durationSec)}`;
    }

    // Active lyric line
    const ms = t * 1000;
    let activeLine = '（器乐律动段落 · 86小节原型字典实时展开）';
    for (let i = VOCAL_LYRIC_LINES.length - 1; i >= 0; i--) {
      if (ms >= VOCAL_LYRIC_LINES[i][0] - 250) {
        activeLine = VOCAL_LYRIC_LINES[i][1];
        break;
      }
    }
    liveLyricEl.textContent = activeLine;

    // Active vocal syllable & `--` coarticulation bridge
    const tier = V9R1_TIERS[st.tierId] || Object.values(V9R1_TIERS)[0];
    const score = tier.vocalScore;
    const curFrame = t / frameDurSec;
    let foundSyl = null;
    let nextSyl = null;
    for (let i = 0; i < score.length; i++) {
      const ev = score[i];
      if (curFrame >= ev[0] && curFrame < ev[0] + ev[1]) {
        foundSyl = ev;
        nextSyl = i + 1 < score.length ? score[i + 1] : null;
        break;
      }
    }
    if (foundSyl) {
      const hasBridge = (foundSyl[5] & 2) !== 0 && nextSyl;
      const bridgeTxt = hasBridge ? ` -- ${nextSyl[3]}(${nextSyl[4]})` : '';
      const harmTxt = foundSyl[7] ? ' · 双声道和声' : '';
      coartBadgeEl.textContent = `${foundSyl[3]} [${foundSyl[4]}]${bridgeTxt}${harmTxt}`;
    } else {
      coartBadgeEl.textContent = '器乐间奏 · 声道静默门控';
    }

    // Active Bar & Motif IDs
    const barIdx = Math.max(0, Math.min(SONG_STRUCTURE.totalBars - 1, Math.floor(t / SONG_STRUCTURE.barDurationSec)));
    const barTuple = tier.bandArrangement.bars[barIdx] || [0, 0, 0, 0, 0, 0];
    const hasRes = Boolean(tier.bandArrangement.residuals && tier.bandArrangement.residuals[String(barIdx)]);
    const dStr = barTuple[0] > 0 ? `M${barTuple[0]}` : '--';
    const bStr = barTuple[2] > 0 ? `M${barTuple[2]}` : '--';
    const hStr = barTuple[4] > 0 ? `M${barTuple[4]}` : '--';
    motifBadgeEl.textContent = `Bar #${String(barIdx + 1).padStart(2, '0')} · D:${dStr} B:${bStr} H:${hStr}${hasRes ? ' +Δ' : ''}`;

    // Active section & chord
    let activeSecIdx = 0;
    const structBar = Math.max(0, Math.floor((t - SONG_STRUCTURE.bar0Sec) / SONG_STRUCTURE.barDurationSec));
    for (let i = 0; i < SONG_STRUCTURE.sections.length; i++) {
      const s = SONG_STRUCTURE.sections[i];
      if (structBar >= s.startBar && structBar < s.startBar + s.bars) {
        activeSecIdx = i;
        break;
      }
    }
    secButtons.forEach((item, idx) => {
      item.btn.classList.toggle('is-current', idx === activeSecIdx);
    });
    const curSec = SONG_STRUCTURE.sections[activeSecIdx];
    let chordLabel = 'Dm9';
    if (curSec.prog !== 'FINAL') {
      const progs = SONG_STRUCTURE.progressions[curSec.prog] || SONG_STRUCTURE.progressions.STD4;
      const relBar = ((structBar - curSec.startBar + (curSec.progOffset || 0)) % 4 + 4) % 4;
      const half = ((t - SONG_STRUCTURE.bar0Sec) % SONG_STRUCTURE.barDurationSec) >= (SONG_STRUCTURE.barDurationSec * 0.5) ? 1 : 0;
      chordLabel = progs[relBar][half];
    }
    secChordEl.textContent = `${curSec.zh} (${curSec.name}) · 和弦 ${chordLabel}`;
  };

  sync(studioStore.getState());
  studioStore.subscribe(sync);
}
