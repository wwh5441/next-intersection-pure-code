// CabronWEC (CWC-5X) Main Application Controller
import { CWC_LYRICS } from '../data/cwc-lyrics.js';
import { CWC_EDITIONS, CWC_TRACK_META } from '../data/cwc-models.js';
import { mountDeckInspector } from './components/deck-inspector.js';
import { mountStageCanvas } from './components/stage-canvas.js';
import { mountStemMatrix } from './components/stem-mixer.js';
import { cwcEngine } from './modules/cwc-audio-engine.js';
import { cwcStore } from './state/cwc-store.js';

function formatTime(sec) {
  const m = String(Math.floor(sec / 60)).padStart(2, '0');
  const s = (sec % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

function initApp() {
  const btnPlay = document.getElementById('btn-play-toggle');
  const btnExport = document.getElementById('btn-export-wav');
  const btnLoadWec = document.getElementById('btn-load-wec');
  const inputWecFile = document.getElementById('input-wec-file');
  const editionBtns = Array.from(document.querySelectorAll('button[data-edition]'));
  const readoutCur = document.getElementById('readout-cur-time');
  const readoutDur = document.getElementById('readout-dur-time');
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

  const handleCustomWecFile = async (file) => {
    if (!file) return;
    try {
      const buf = await file.arrayBuffer();
      await cwcEngine.loadWecBuffer(buf, file.name, 'custom');
    } catch (err) {
      cwcStore.setState({ errorMsg: `Invalid .wec file (${file.name}): ${err.message}` });
    }
  };

  if (btnLoadWec && inputWecFile) {
    btnLoadWec.addEventListener('click', () => inputWecFile.click());
    inputWecFile.addEventListener('change', () => {
      const f = inputWecFile.files && inputWecFile.files[0];
      if (f) handleCustomWecFile(f);
      inputWecFile.value = '';
    });
  }

  window.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    document.body.classList.add('cwc-drag-over');
  });
  window.addEventListener('dragleave', (ev) => {
    if (!ev.relatedTarget) document.body.classList.remove('cwc-drag-over');
  });
  window.addEventListener('drop', (ev) => {
    ev.preventDefault();
    document.body.classList.remove('cwc-drag-over');
    const file = ev.dataTransfer?.files?.[0];
    if (file) handleCustomWecFile(file);
  });

  const renderChunkStrip = (state) => {
    const baseKey =
      state.activeTier === 'wec'
        ? CWC_EDITIONS[state.loadedWecMeta?.sourceTier]
          ? state.loadedWecMeta.sourceTier
          : '500k'
        : state.activeTier;
    const ed = CWC_EDITIONS[baseKey] || CWC_EDITIONS['1m'];
    const curChunk = cwcEngine.chunkIndexForFrame(state.currentFrame, ed.id);
    const isWec = state.activeTier === 'wec';
    chunkStripEl.innerHTML = ed.chunks
      .map((c, idx) => {
        const loaded = state.chunksLoaded[idx];
        const active = idx === curChunk;
        const kb = Math.round(c.rawBytes / 1000);
        const badge = isWec ? `WEC · S${idx}` : `C${idx} · ${loaded ? `${kb} KB` : 'Stream'}`;
        return `
          <button
            type="button"
            class="cwc-chunk-cell ${loaded ? 'is-loaded' : ''} ${active ? 'is-active' : ''}"
            data-chunk-idx="${idx}"
            aria-label="Jump to ${isWec ? 'WEC' : ed.shortLabel} Section ${idx}: ${c.sectionTitle}"
          >
            <span class="cwc-chunk-id tabular">${badge}</span>
            <span class="cwc-chunk-sec">${c.sectionTitle}</span>
          </button>
        `;
      })
      .join('');
  };

  chunkStripEl.addEventListener('click', (ev) => {
    const cell = ev.target.closest('button[data-chunk-idx]');
    if (!cell) return;
    const idx = Number(cell.dataset.chunkIdx);
    const st = cwcStore.getState();
    const baseKey =
      st.activeTier === 'wec' && CWC_EDITIONS[st.loadedWecMeta?.sourceTier]
        ? st.loadedWecMeta.sourceTier
        : st.activeTier;
    const ed = CWC_EDITIONS[baseKey] || CWC_EDITIONS['1m'];
    const c = ed.chunks[idx];
    if (c) cwcEngine.seek(c.startSec);
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

  document.querySelectorAll('.cwc-nav-link[data-nav]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cwc-nav-link').forEach((n) => n.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.nav;
      if (target === 'metrics' || target === 'equations' || target === 'lyrics') {
        selectDeckTab(target);
        document.getElementById('deck-section').scrollIntoView({ behavior: 'smooth' });
      } else if (target === 'stems') {
        document.getElementById('stems-section').scrollIntoView({ behavior: 'smooth' });
      } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });

  document.querySelectorAll('[role="tab"][data-tab]').forEach((t) => {
    t.addEventListener('click', () => selectDeckTab(t.dataset.tab));
  });

  btnPlay.addEventListener('click', () => cwcEngine.togglePlay());

  editionBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      cwcEngine.selectEdition(btn.dataset.edition);
    });
  });

  sliderMaster.addEventListener('input', () => {
    const val = Number(sliderMaster.value);
    readoutMaster.textContent = `${val}%`;
    cwcEngine.setMasterGain(val / 100);
  });

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

  btnExport.addEventListener('click', () => cwcEngine.exportWav());

  let lastChunkRendered = -1;
  let lastLoadedCount = -1;
  let lastTierRendered = '';
  let lastSourceRendered = '';
  let lastWecFileRendered = '';
  let lastLyricIdx = -99;
  let lastDeckTab = '';
  let lastFrameRendered = -1;

  const syncUi = (st) => {
    const ed = CWC_EDITIONS[st.activeTier] || CWC_EDITIONS['1m'];
    const wecMeta = st.loadedWecMeta;
    document.body.dataset.state = st.isPlaying ? 'playing' : st.isReady ? 'ready' : 'initializing';
    document.body.dataset.source = st.audioSource;
    document.body.dataset.tier = st.activeTier;

    btnPlay.disabled = !st.isReady;
    btnPlay.setAttribute('aria-pressed', String(st.isPlaying));
    btnPlay.textContent = !st.isReady ? 'Loading' : st.isPlaying ? 'Pause' : 'Play';

    btnExport.disabled = !st.isReady || st.exportingWav;
    btnExport.textContent = st.exportingWav
      ? `Exporting ${Math.round(st.exportProgress * 100)}%`
      : `Export WAV (${st.activeTier.toUpperCase()})`;

    editionBtns.forEach((btn) => {
      const mode = btn.dataset.edition;
      const isWecBtn = Boolean(mode && mode.startsWith('wec-'));
      const activeWecKey = wecMeta?.sourceTier ? `wec-${wecMeta.sourceTier}` : '';
      const isAct =
        mode === 'ref'
          ? st.audioSource === 'ref'
          : isWecBtn
            ? st.audioSource === 'cwc' && st.activeTier === 'wec' && mode === activeWecKey
            : st.audioSource === 'cwc' && st.activeTier === mode;
      btn.classList.toggle('active', isAct);
      btn.classList.toggle('ref-mode', mode === 'ref' && isAct);
      btn.classList.toggle('wec-mode', Boolean(isWecBtn && isAct));
      btn.setAttribute('aria-pressed', String(isAct));
      if (mode === 'ref') {
        btn.textContent = st.refLoading ? 'Loading MP3...' : 'MP3 Ref';
      }
    });

    readoutCur.textContent = formatTime(st.currentTimeSec);
    const curBar = Math.floor(st.currentTimeSec / CWC_TRACK_META.barSec) + 1;
    readoutFrame.textContent = `Frame ${st.currentFrame.toLocaleString()} / ${st.totalFrames.toLocaleString()} · Bar ${curBar}`;
    readoutCpu.textContent =
      typeof st.cpuMs === 'number'
        ? `Worklet ${st.cpuMs.toFixed(2)} ms`
        : st.isPlaying
          ? 'Worklet 128-smp · Live'
          : 'Worklet Ready (128-smp)';

    if (st.activeTier === 'wec' && wecMeta) {
      const wKb = (wecMeta.bytes / 1000).toFixed(1);
      const wKib = (wecMeta.bytes / 1024).toFixed(1);
      metaStream.textContent = `${wecMeta.fileName} (${wKb} KB / ${wKib} KiB) · CRC32 0x${wecMeta.crc32Hex} Verified`;
      chunkSummaryEl.textContent = `CWC5 Binary · ${wKb} KB (${wKib} KiB) · ${wecMeta.entropyModeName} · ${wecMeta.unpackMs} ms`;
    } else {
      const rawKb = (ed.cwcRawBytes / 1000).toFixed(1);
      const gzKb = (ed.cwcGzipBytes / 1000).toFixed(1);
      metaStream.textContent = `${ed.shortLabel} (${rawKb} KB Raw / ${gzKb} KB Gzip) · ${st.loadedChunkCount}/8 Ready`;
      chunkSummaryEl.textContent = `${ed.name} · ${rawKb} KB Raw (${gzKb} KB Gzip) · Zero-Dep Pure JS`;
    }

    if (st.errorMsg) {
      errorSlot.hidden = false;
      errorSlot.textContent = st.errorMsg;
    } else {
      errorSlot.hidden = true;
    }

    let lIdx = -1;
    for (let i = 0; i < CWC_LYRICS.length; i++) {
      if (st.currentTimeSec >= CWC_LYRICS[i].timeSec) lIdx = i;
    }
    if (lIdx !== lastLyricIdx) {
      lastLyricIdx = lIdx;
      const curL =
        lIdx >= 0 ? CWC_LYRICS[lIdx].text : `下个，路口，见 · ${ed.name}`;
      const nextL = CWC_LYRICS[lIdx + 1]?.text || 'End of track';
      lyricNowEl.textContent = curL;
      lyricNextEl.textContent = `Next: ${nextL}`;
      if (st.activeDeckTab === 'lyrics') deck.render();
    }

    const curChunk = cwcEngine.chunkIndexForFrame(st.currentFrame, ed.id);
    const wecFileKey = wecMeta ? `${wecMeta.fileName}:${wecMeta.crc32Hex}` : '';
    const tierOrSourceChanged =
      st.activeTier !== lastTierRendered ||
      st.audioSource !== lastSourceRendered ||
      wecFileKey !== lastWecFileRendered;
    const chunkChanged =
      curChunk !== lastChunkRendered || st.loadedChunkCount !== lastLoadedCount || tierOrSourceChanged;

    if (chunkChanged) {
      lastChunkRendered = curChunk;
      lastLoadedCount = st.loadedChunkCount;
      lastTierRendered = st.activeTier;
      lastSourceRendered = st.audioSource;
      lastWecFileRendered = wecFileKey;
      renderChunkStrip(st);
    }

    if (
      st.activeDeckTab !== lastDeckTab ||
      tierOrSourceChanged ||
      (st.activeDeckTab === 'equations' &&
        (chunkChanged || Math.abs(st.currentFrame - lastFrameRendered) >= 6))
    ) {
      lastDeckTab = st.activeDeckTab;
      lastFrameRendered = st.currentFrame;
      deck.render();
    }
  };

  cwcStore.subscribe(syncUi);
  syncUi(cwcStore.getState());

  cwcEngine.init().catch((err) => {
    cwcStore.setState({ errorMsg: `Engine init failed: ${err.message}` });
  });
}

initApp();
