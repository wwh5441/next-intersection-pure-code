// 60fps 98.0000 BPM Timeline, 285-Syllable Contour & Live 48kHz Spectrum Canvas (`src/components/score-motif-canvas.js`)
import { SONG_STRUCTURE } from '../../data/song-structure.js';
import { VOCAL_LYRIC_LINES, VOCAL_SYLLABLES_META } from '../../data/vocal-lyrics.js';
import { audioMixer } from '../modules/audio-mixer.js';
import { studioStore } from '../state/studio-store.js';

export function mountScoreMotifCanvas(rootEl) {
  rootEl.innerHTML = `
    <div class="section-head">
      <h2 class="section-title">98.0000 BPM 零漂移小节律动网格 · 285音节GCI轨迹 · 48kHz实时频谱</h2>
      <span class="section-sub">点击下方时间轴任意位置可秒级定位试听</span>
    </div>
    <div class="lyric-hud">
      <div class="lyric-current" id="hud-lyric-line">《李宇春 - 下个，路口，见》 · 纯数学+代码+字典合成</div>
      <div class="lyric-syllable-badge" id="hud-syllable-meta">Bar 01 · 98.0000 BPM · Dm9</div>
    </div>
    <canvas class="visualizer-canvas" id="studio-main-canvas"></canvas>
  `;

  const lyricEl = rootEl.querySelector('#hud-lyric-line');
  const badgeEl = rootEl.querySelector('#hud-syllable-meta');
  const canvas = rootEl.querySelector('#studio-main-canvas');
  const ctx2d = canvas.getContext('2d');

  function findActiveLyric(tSec) {
    const ms = tSec * 1000;
    let line = '（纯器乐律动段落 · Acoustic/Funk Guitar + Rhodes + 98BPM Groove）';
    for (let i = 0; i < VOCAL_LYRIC_LINES.length; i++) {
      if (ms >= VOCAL_LYRIC_LINES[i][0] - 120) {
        line = VOCAL_LYRIC_LINES[i][1];
      } else {
        break;
      }
    }
    let activeSyl = null;
    for (let i = 0; i < VOCAL_SYLLABLES_META.length; i++) {
      const s = VOCAL_SYLLABLES_META[i];
      if (tSec >= s[1] && tSec <= s[2]) {
        activeSyl = s;
        break;
      }
    }
    return { line, activeSyl };
  }

  function findChordAndBar(tSec) {
    const rel = tSec - SONG_STRUCTURE.bar0Sec;
    if (rel < 0) return { barNum: 0, chord: 'Pickup', secZh: '前奏切入' };
    const barFloat = rel / SONG_STRUCTURE.barDurationSec;
    const barIdx = Math.floor(barFloat);
    const halfInBar = (barFloat - barIdx) >= 0.5 ? 1 : 0;
    let secZh = '尾奏';
    let chord = 'Dm9';
    for (const sec of SONG_STRUCTURE.sections) {
      if (barIdx >= sec.startBar && barIdx < sec.startBar + sec.bars) {
        secZh = sec.zh;
        const prog = SONG_STRUCTURE.progressions[sec.prog];
        if (prog) {
          const pair = prog[(barIdx - sec.startBar) % 4];
          chord = pair[halfInBar];
        }
        break;
      }
    }
    return { barNum: barIdx + 1, chord, secZh };
  }

  canvas.addEventListener('click', async (ev) => {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const targetSec = ratio * studioStore.getState().duration;
    await audioMixer.seek(targetSec);
  });

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2d.clearRect(0, 0, w, h);

    const st = studioStore.getState();
    const dur = st.duration || 210.3733;
    const tSec = st.currentTime;

    // Update HUD text
    const { line, activeSyl } = findActiveLyric(tSec);
    const { barNum, chord, secZh } = findChordAndBar(tSec);
    lyricEl.textContent = line;
    badgeEl.textContent = activeSyl
      ? `${secZh} · Bar ${String(barNum).padStart(2, '0')} (${chord}) · 音节「${activeSyl[3]}」(${activeSyl[4]} · MIDI ${activeSyl[5]})`
      : `${secZh} · Bar ${String(barNum).padStart(2, '0')} (${chord}) · 98.0000 BPM`;

    // Upper Half (`0..112px`): Live 48kHz FFT Spectrum
    const specBins = audioMixer.getMasterSpectrum();
    const specH = 108;
    ctx2d.fillStyle = 'rgba(255,255,255,0.03)';
    ctx2d.fillRect(0, 0, w, specH);

    const nBarsSpec = 96;
    const barW = w / nBarsSpec;
    for (let i = 0; i < nBarsSpec; i++) {
      const binIdx = Math.min(specBins.length - 1, Math.floor(Math.pow(i / nBarsSpec, 1.65) * (specBins.length * 0.86)));
      const v = specBins[binIdx] / 255.0;
      const bh = Math.max(2, v * (specH - 16));
      ctx2d.fillStyle = i < 24 ? '#38bdf8' : i < 64 ? '#f59e0b' : '#f472b6';
      ctx2d.fillRect(i * barW + 1, specH - bh, Math.max(1.5, barW - 2), bh);
    }

    // Divider
    ctx2d.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx2d.beginPath();
    ctx2d.moveTo(0, specH + 4);
    ctx2d.lineTo(w, specH + 4);
    ctx2d.stroke();

    // Lower Half (`118..h`): 85-Bar Section Blocks + 285 Mandarin Syllable Pitch Contour
    const tlTop = specH + 10;
    const tlH = h - tlTop - 8;

    SONG_STRUCTURE.sections.forEach((sec, idx) => {
      const s0 = SONG_STRUCTURE.bar0Sec + sec.startBar * SONG_STRUCTURE.barDurationSec;
      const s1 = s0 + sec.bars * SONG_STRUCTURE.barDurationSec;
      const x0 = (s0 / dur) * w;
      const x1 = (s1 / dur) * w;
      ctx2d.fillStyle = idx % 2 === 0 ? 'rgba(255,255,255,0.045)' : 'rgba(255,255,255,0.085)';
      ctx2d.fillRect(x0, tlTop, Math.max(1, x1 - x0 - 1), tlH);
      if (x1 - x0 > 42) {
        ctx2d.fillStyle = 'rgba(255,255,255,0.55)';
        ctx2d.font = '10px -apple-system, sans-serif';
        ctx2d.fillText(sec.zh, x0 + 4, tlTop + 13);
      }
    });

    // Plot 285 syllables along pitch contour
    for (let i = 0; i < VOCAL_SYLLABLES_META.length; i++) {
      const [, s0, s1, , , midi] = VOCAL_SYLLABLES_META[i];
      const x = (s0 / dur) * w;
      const sw = Math.max(2, ((s1 - s0) / dur) * w);
      const normP = Math.max(0, Math.min(1, (midi - 48) / 22));
      const y = tlTop + tlH - 12 - normP * (tlH - 30);
      const isCur = tSec >= s0 && tSec <= s1;
      ctx2d.fillStyle = isCur ? '#ffd580' : '#ef4444';
      ctx2d.fillRect(x, y, sw, isCur ? 5 : 3);
    }

    // Playhead
    const px = (tSec / dur) * w;
    ctx2d.strokeStyle = '#22c55e';
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.moveTo(px, 0);
    ctx2d.lineTo(px, h);
    ctx2d.stroke();
    ctx2d.lineWidth = 1;

    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
}
