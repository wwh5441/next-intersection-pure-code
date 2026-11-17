// CabronWEC (CWC-4X) Application Bootstrap & Controller

import { CWC_LYRICS } from '../data/cwc-lyrics.js';
import { CWC_CHUNKS, CWC_TRACK_META } from '../data/cwc-models.js';
import { mountDeckInspector } from './components/deck-inspector.js';
import { mountStageCanvas } from './components/stage-canvas.js';
import { mountStemMatrix } from './components/stem-mixer.js';
import { cwcEngine } from './modules/cwc-audio-engine.js';
import { cwcStore } from './state/cwc-store.js';

function formatTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = (sec % 60).toFixed(2).padStart(5, '0');
  return `${m}:${s}`;
}

function initApp() {
  const btnPlay = document.getElementById('btn-play-toggle');
  const btnExport = document.getElementById('btn-export-wav');
  const btnSrcCwc = document.getElementById('btn-src-cwc');
  const btnSrcRef = document.getElementById('btn-src-ref');
  const readoutCur = document.getElementById('readout-cur');
  const readoutDur = document.getElementById('readout-dur');
  const readoutFrame = document.getElementById('readout-frame');
  const readoutCpu = document.getElementById('readout-cpu');
  const readoutMaster = document.getElementById('readout-master');
  const metaStream = document.getElementById('meta-stream-status');
  const lyricNowEl = document.getElementById('lyric-now');
  const lyricNextEl = document.getElementById('lyric-next');
  const chunkStripEl = document.getElementById('chunk-strip');
  const chunkSummaryEl = document.getElementById('readout-chunk-summary');
  const errorSlot = document.getElementById('cwc-error-slot');
  const sliderMaster = document.getElementById('slider-master-gain');

  readoutDur.textContent = formatTime(CWC_TRACK_META.durationSec);

  const stemMatrix = mountStemMatrix(document.getElementById('stem-list'));
  mountStageCanvas(
    document.getElementById('cwc-stage-canvas'),
    document.getElementById('crosshair-hud')
  );
  const deck = mountDeckInspector(document.getElementById('deck-body'));

  const renderChunkStrip = (state) => {
    const curChunk = cwcEngine.chunkIndexForFrame(state.currentFrame);
    chunkStripEl.innerHTML = CWC_CHUNKS.map((c, idx) => {
      const loaded = state.chunksLoaded[idx];
      const active = idx === curChunk;
      return `
        <button
          type="button"
          class="cwc-chunk-cell ${loaded ? 'is-loaded' : ''} ${active ? 'is-active' : ''}"
          data-chunk-idx="${idx}"
          aria-label="Jump to Chunk ${idx}: ${c.sectionTitle}"
        >
          <span class="cwc-chunk-id tabular">C${idx} · ${loaded ? 'Ready' : 'Stream'}</span>
          <span class="cwc-chunk-sec">${c.sectionTitle}</span>
        </button>
      `;
    }).join('');
  };

  chunkStripEl.addEventListener('click', (ev) => {
    const cell = ev.target.closest('button[data-chunk-idx]');
    if (!cell) return;
    const idx = Number(cell.dataset.chunkIdx);
    const c = CWC_CHUNKS[idx];
    if (c) cwcEngine.seek(c.startSec);
  });

  // Top nav links scroll smoothly / select deck tab
  document.querySelectorAll('.cwc-nav-link[data-nav]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cwc-nav-link').forEach((n) => n.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.nav;
      if (target === 'equations' || target === 'lyrics') {
        selectDeckTab(target);
        document.getElementById('deck-section').scrollIntoView({ behavior: 'smooth' });
      } else if (target === 'stems') {
        document.getElementById('stems-section').scrollIntoView({ behavior: 'smooth' });
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });

  const selectDeckTab = (tabId) => {
    document.querySelectorAll('[role="tab"][data-tab]').forEach((t) => {
      const active = t.dataset.tab === tabId;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
    });
    cwcStore.setState({ activeDeckTab: tabId });
    deck.render();
  };

  document.querySelectorAll('[role="tab"][data-tab]').forEach((t) => {
    t.addEventListener('click', () => selectDeckTab(t.dataset.tab));
  });

  // Transport controls
  btnPlay.addEventListener('click', () => cwcEngine.togglePlay());

  btnSrcCwc.addEventListener('click', () => cwcEngine.setAudioSource('cwc'));
  btnSrcRef.addEventListener('click', () => cwcEngine.setAudioSource('ref'));

  sliderMaster.addEventListener('input', () => {
    const val = Number(sliderMaster.value);
    readoutMaster.textContent = `${val}%`;
    cwcEngine.setMasterGain(val / 100);
  });

  // Preset buttons
  document.querySelectorAll('.cwc-preset-btn[data-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.preset;
      const curStems = cwcStore.getState().stems;
      let next = curStems;
      if (p === 'full') {
        next = curStems.map((s) => ({ ...s, gain: 1.0, muted: false, solo: false }));
      } else if (p === 'vocal') {
        next = curStems.map((s, i) => ({ ...s, gain: 1.0, muted: false, solo: i === 2 || i === 3 }));
      } else if (p === 'backing') {
        next = curStems.map((s, i) => ({ ...s, gain: 1.0, muted: i === 2, solo: false }));
      } else if (p === 'side') {
        next = curStems.map((s, i) => ({ ...s, gain: 1.0, muted: false, solo: i === 4 }));
      }
      cwcStore.setState({ stems: next });
      cwcEngine.syncStemsToWorklet();
      stemMatrix.refresh();
    });
  });

  // WAV Export
  btnExport.addEventListener('click', () => cwcEngine.exportWav());

  // Subscribe UI to store updates
  let lastChunkRendered = -1;
  let lastLoadedCount = -1;
  let lastLyricIdx = -99;
  let lastDeckTab = '';
  let lastFrameRendered = -1;

  const syncUi = (st) => {
    document.body.dataset.state = st.isPlaying ? 'playing' : (st.isReady ? 'ready' : 'initializing');
    document.body.dataset.source = st.audioSource;

    btnPlay.disabled = !st.isReady;
    btnPlay.setAttribute('aria-pressed', String(st.isPlaying));
    btnPlay.textContent = !st.isReady ? 'Loading' : (st.isPlaying ? 'Pause' : 'Play');

    btnExport.disabled = !st.isReady || st.exportingWav;
    btnExport.textContent = st.exportingWav
      ? `Exporting ${Math.round(st.exportProgress * 100)}%`
      : 'Export WAV';

    btnSrcCwc.classList.toggle('active', st.audioSource === 'cwc');
    btnSrcCwc.setAttribute('aria-pressed', String(st.audioSource === 'cwc'));
    btnSrcRef.classList.toggle('active', st.audioSource === 'ref');
    btnSrcRef.classList.toggle('ref-mode', st.audioSource === 'ref');
    btnSrcRef.setAttribute('aria-pressed', String(st.audioSource === 'ref'));
    btnSrcRef.textContent = st.refLoading ? 'Loading MP3...' : 'Original MP3 Ref';

    readoutCur.textContent = formatTime(st.currentTimeSec);
    const curBar = Math.floor(st.currentTimeSec / CWC_TRACK_META.barSec) + 1;
    readoutFrame.textContent = `Frame ${st.currentFrame.toLocaleString()} / ${st.totalFrames.toLocaleString()} · Bar ${curBar}`;
    readoutCpu.textContent = typeof st.cpuMs === 'number'
      ? `Worklet ${st.cpuMs.toFixed(2)} ms`
      : (st.isPlaying ? 'Worklet 128-smp · Live' : 'Worklet Ready (128-smp)');
    metaStream.textContent = `${st.loadedChunkCount}/8 Chunks Ready · ${st.engineMode}`;
    chunkSummaryEl.textContent = `${st.loadedChunkCount}/8 Chunks Loaded · Pure JS Math`;

    if (st.errorMsg) {
      errorSlot.hidden = false;
      errorSlot.textContent = st.errorMsg;
    } else {
      errorSlot.hidden = true;
    }

    // Update lyrics
    let lIdx = -1;
    for (let i = 0; i < CWC_LYRICS.length; i++) {
      if (st.currentTimeSec >= CWC_LYRICS[i].timeSec) lIdx = i;
    }
    if (lIdx !== lastLyricIdx) {
      lastLyricIdx = lIdx;
      const curL = lIdx >= 0 ? CWC_LYRICS[lIdx].text : '下个，路口，见 · CabronWEC Pure-Code Stream';
      const nextL = CWC_LYRICS[lIdx + 1]?.text || 'End of track';
      lyricNowEl.textContent = curL;
      lyricNextEl.textContent = `Next: ${nextL}`;
      if (st.activeDeckTab === 'lyrics') deck.render();
    }

    const curChunk = cwcEngine.chunkIndexForFrame(st.currentFrame);
    const chunkChanged = curChunk !== lastChunkRendered || st.loadedChunkCount !== lastLoadedCount;
    if (chunkChanged) {
      lastChunkRendered = curChunk;
      lastLoadedCount = st.loadedChunkCount;
      renderChunkStrip(st);
    }

    if (
      st.activeDeckTab !== lastDeckTab ||
      (st.activeDeckTab === 'equations' && (chunkChanged || Math.abs(st.currentFrame - lastFrameRendered) >= 24))
    ) {
      lastDeckTab = st.activeDeckTab;
      lastFrameRendered = st.currentFrame;
      deck.render();
    }
  };

  cwcStore.subscribe(syncUi);
  syncUi(cwcStore.getState());

  // Boot chunk 0 immediately
  cwcEngine.init().catch((err) => {
    cwcStore.setState({ errorMsg: `Engine init failed: ${err.message}` });
  });
}

initApp();
