// Minimalist WEC Web Player Entry (`src/main.js`)
import { renderV9r2AllLayers } from '../dsp/v9r2-master-engine.js';
import { WecAudioPlayer } from './wec-audio.js';

const $ = (id) => document.getElementById(id);
const titleEl = $('song-title');
const metaEl = $('file-meta');
const lyricBox = $('lyric-box');
const timeCur = $('time-cur');
const timeDur = $('time-dur');
const seekSlider = $('seek-slider');
const btnPlay = $('btn-play');
const btnWav = $('btn-wav');
const statusEl = $('status-text');

let lyrics = [];
let currentName = '下个路口见_V9R2.8.wec';

const player = new WecAudioPlayer(() => {
  btnPlay.textContent = '▶ 播放';
  seekSlider.value = '0';
  timeCur.textContent = '00:00';
});

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function findLyric(sec) {
  for (let i = 0; i < lyrics.length; i++) {
    const ln = lyrics[i];
    if (sec >= ln.startSec && sec <= ln.endSec) return ln.text || '♪ 间奏';
  }
  return '♪';
}

async function loadWec() {
  btnPlay.disabled = true;
  btnWav.disabled = true;
  seekSlider.disabled = true;
  statusEl.textContent = '正在解码 .wec 容器...';

  const t0 = performance.now();
  const res = await fetch('/api/decode');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const decoded = await res.json();
  const decMs = performance.now() - t0;

  currentName = decoded.fileName;
  titleEl.textContent = decoded.title || 'WEC 音频';
  metaEl.textContent = `${currentName} · ${decoded.wecKB} KB · 解码 ${decMs.toFixed(0)}ms`;
  lyrics = decoded.lyrics || [];

  statusEl.textContent = '正在执行纯代码 DSP 实时合成 (48kHz 11轨)...';
  await new Promise((r) => setTimeout(r, 20));
  const synth = renderV9r2AllLayers(decoded.renderArgs);
  player.loadLayers(synth);

  seekSlider.max = String(player.durationSec.toFixed(2));
  seekSlider.value = '0';
  timeDur.textContent = fmtTime(player.durationSec);
  timeCur.textContent = '00:00';
  lyricBox.textContent = findLyric(0);
  btnPlay.textContent = '▶ 播放';
  btnPlay.disabled = false;
  btnWav.disabled = false;
  seekSlider.disabled = false;
  statusEl.textContent = `已就绪 (DSP 合成 ${(synth.synthMs / 1000).toFixed(2)}s · 48kHz 立体声)`;
}

btnPlay.addEventListener('click', () => {
  if (player.playing) {
    player.pause();
    btnPlay.textContent = '▶ 播放';
  } else {
    player.play();
    btnPlay.textContent = '❚❚ 暂停';
  }
});

seekSlider.addEventListener('input', () => {
  const sec = parseFloat(seekSlider.value);
  player.seek(sec);
  timeCur.textContent = fmtTime(sec);
  lyricBox.textContent = findLyric(sec);
});

btnWav.addEventListener('click', () => {
  const blob = player.exportWavBlob();
  if (!blob) return;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = currentName.replace(/\.wec$/i, '') + '_48kHz.wav';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
});

function tick() {
  if (player.playing) {
    const t = player.getCurrentTime();
    seekSlider.value = String(t);
    timeCur.textContent = fmtTime(t);
    lyricBox.textContent = findLyric(t);
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

loadWec().catch((e) => {
  statusEl.textContent = `加载失败: ${e.message}`;
});
