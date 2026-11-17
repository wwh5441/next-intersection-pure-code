// Main Application Entry (`src/main.js`)
import { STUDIO_METADATA } from '../data/studio-metadata.js';
import { SINGER_CORE } from '../models/singer-core.js';
import { SYLLABLE_ATLAS } from '../models/syllable-atlas.js';
import { SPACE_FDN } from '../models/space-fdn.js';
import { INSTRUMENT_PHYSICS } from '../models/instrument-physics.js';
import { VOCAL_EVENTS_V85 } from '../data/vocal-events-v85.js';
import { BAND_SCORE_EXPRESSIVE } from '../data/band-score-expressive.js';
import { BAND_STEM_GAINS } from '../data/band-stem-gains.js';
import { renderV9r2AllLayers } from './dsp/v9r2-master-engine.js';

import { createStudioStore } from './state/studio-store.js';
import { MultiTrackAudioEngine } from './modules/multitrack-audio-engine.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountPitchCoartCanvas } from './components/pitch-coart-canvas.js';
import { mountLayerMixerPanel } from './components/layer-mixer-panel.js';
import { mountSyllableAtlasInspector } from './components/syllable-atlas-inspector.js';

const PROFILE = 'pure';
const { config, atlasCatalog } = STUDIO_METADATA;
const store = createStudioStore(config.layers);
store.setState({ profileId: PROFILE });
const audio = new MultiTrackAudioEngine(config.layers, () => store.setState({ playing: false, currentTime: 0.0 }));

let uiEvents = [];
let synthPromise = null;

function findActiveEvent(sec) {
  let lo = 0;
  let hi = uiEvents.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const ev = uiEvents[mid];
    if (sec < ev.startSec) hi = mid - 1;
    else if (sec > ev.endSec) lo = mid + 1;
    else return ev;
  }
  return null;
}

function onSynthesized(res) {
  audio.storeSynthesizedProfile(PROFILE, res);
  audio.activateProfile(PROFILE, store.getState().layerParams);
  uiEvents = res.uiEvents || [];
  canvasView.setEvents(uiEvents);
  atlasView.setEvents(uiEvents);
  store.setState({
    rendering: false,
    ready: true,
    progress: 1.0,
    synthMs: res.synthMs,
    durationSec: res.durationSec,
    statusText: `已就绪 (${(res.synthMs / 1000).toFixed(2)}s 合成 · 48kHz 11轨纯代码)`,
  });
}

function synthesize() {
  if (synthPromise) return synthPromise;
  store.setState({ rendering: true, progress: 0.04, statusText: '正在纯代码合成 48kHz 十一轨...' });
  synthPromise = new Promise((resolve, reject) => {
    let worker = null;
    if (!window.__V9R2_SINGLE_HTML__) {
      try {
        worker = new Worker(new URL('./dsp/v9r2-synth-worker.js', import.meta.url), { type: 'module' });
      } catch (_) {
        worker = null;
      }
    }
    if (worker) {
      worker.onmessage = (ev) => {
        const msg = ev.data || {};
        if (msg.type === 'PROGRESS') store.setState({ progress: msg.progress, statusText: msg.statusText });
        else if (msg.type === 'READY') {
          onSynthesized(msg);
          resolve();
        } else if (msg.type === 'ERROR') reject(new Error(msg.error));
      };
      worker.postMessage({ type: 'SYNTH_PROFILE', reqId: 1 });
      return;
    }
    setTimeout(() => {
      try {
        onSynthesized(
          renderV9r2AllLayers({
            config,
            singerCore: SINGER_CORE,
            syllableAtlas: SYLLABLE_ATLAS,
            spaceFdn: SPACE_FDN,
            vocalEvents: VOCAL_EVENTS_V85,
            instrumentPhysics: INSTRUMENT_PHYSICS,
            bandScore: BAND_SCORE_EXPRESSIVE,
            bandGains: BAND_STEM_GAINS,
          })
        );
        resolve();
      } catch (e) {
        reject(e);
      }
    }, 30);
  });
  return synthPromise;
}

async function playFrom(sec = null) {
  audio.ensureContext(config.sampleRate);
  await synthesize();
  audio.play(sec ?? undefined);
  store.setState({ playing: true, currentTime: audio.getCurrentTime() });
}

mountHeaderMetrics(document.getElementById('header-panel'), STUDIO_METADATA, store);

const transportView = mountTransportBar(document.getElementById('transport-panel'), STUDIO_METADATA, store, {
  onTogglePlay: async () => {
    if (!store.getState().ready) return playFrom();
    if (audio.playing) {
      audio.pause();
      store.setState({ playing: false, currentTime: audio.getCurrentTime() });
    } else {
      audio.play();
      store.setState({ playing: true });
    }
  },
  onStop: () => {
    audio.stop();
    store.setState({ playing: false, currentTime: 0.0 });
  },
  onSeek: (sec) => {
    audio.seek(sec);
    store.setState({ currentTime: sec });
  },
  onExportWav: () => {
    const blob = audio.exportCurrentMixWav16(store.getState().layerParams);
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '下个路口见_V9R2_纯代码_48kHz.wav';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  },
});

const canvasView = mountPitchCoartCanvas(document.getElementById('visualizer-panel'), STUDIO_METADATA, store, audio, (sec) => playFrom(sec));
const mixerView = mountLayerMixerPanel(document.getElementById('mixer-panel'), config.layers, store, () =>
  audio.applyLayerMix(store.getState().layerParams)
);
const atlasView = mountSyllableAtlasInspector(document.getElementById('atlas-panel'), atlasCatalog, store, (sec) => playFrom(sec));

store.subscribe((st) => transportView.update(st, findActiveEvent(st.currentTime)));
transportView.update(store.getState(), null);

synthesize().catch((err) => {
  console.error('Synthesis error:', err);
  store.setState({ rendering: false, statusText: `合成失败: ${err.message || err}` });
});

function tick() {
  const st = store.getState();
  if (audio.playing) {
    store.setState({ currentTime: audio.getCurrentTime() });
    mixerView.updateMeters(audio.getInstantLayerRms(st.layerParams));
  }
  canvasView.draw(store.getState());
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
