import { StudioStore } from './state/store.js';
import { GenesisResourceLoader } from './modules/loader.js';
import { MultiLayerAudioController } from './modules/audio-controller.js';
import { renderMetricsStrip } from './components/metrics-banner.js';
import { TransportBarComponent } from './components/transport-bar.js';
import { PitchEventCanvasComponent } from './components/pitch-event-canvas.js';
import { LayerMixerPanelComponent } from './components/layer-mixer-panel.js';
import { SyllableAtlasPanelComponent } from './components/syllable-atlas-panel.js';

async function bootStudio() {
  const loader = new GenesisResourceLoader();
  const { config, profiles, sizes, lyrics, atlasCatalog } = await loader.loadStudioMetadata();

  const store = new StudioStore({
    profile: profiles.default || 'expressive',
    duration: lyrics.durationSec || 210.373,
  });

  const audio = new MultiLayerAudioController(config.layers, () => {
    store.setState({ playing: false, currentTime: 0 });
  });

  const metricsRoot = document.getElementById('metrics-strip');
  const transportRoot = document.getElementById('transport-panel');
  const canvasEl = document.getElementById('pitch-event-canvas');
  const mixerRoot = document.getElementById('layer-mixer-panel');
  const atlasRoot = document.getElementById('syllable-atlas-panel');

  let currentUiEvents = [];

  const transport = new TransportBarComponent(transportRoot, {
    profilesMeta: profiles,
    sizes,
    lyrics,
    onTogglePlay: async () => {
      if (store.getState().playing) {
        audio.pause();
        store.setState({ playing: false, currentTime: audio.getCurrentTime() });
      } else {
        await audio.play();
        store.setState({ playing: true });
      }
    },
    onSeek: async (sec) => {
      await audio.seek(sec);
      store.setState({ currentTime: sec });
    },
    onSelectProfile: async (profId) => {
      if (profId === store.getState().profile || store.getState().loading) return;
      await activateProfile(profId);
    },
  });

  const ribbon = new PitchEventCanvasComponent(canvasEl, {
    onSeekToSec: async (sec) => {
      await audio.seek(sec);
      if (!store.getState().playing) {
        await audio.play(sec);
      }
      store.setState({ playing: true, currentTime: sec });
    },
  });

  const mixer = new LayerMixerPanelComponent(mixerRoot, {
    layers: config.layers,
    onChangeLayer: (layerId, patch) => {
      audio.setLayerParam(layerId, patch);
      mixer.syncButtons(audio.getLayerParamsSnapshot());
    },
  });

  const atlasPanel = new SyllableAtlasPanelComponent(atlasRoot, {
    atlasCatalog,
    onSelectSyllable: (sid) => {
      store.setState({ highlightedSyllableId: sid });
    },
    onJumpToEvent: async (sec) => {
      await audio.seek(sec);
      await audio.play(sec);
      store.setState({ playing: true, currentTime: sec });
    },
  });

  function findActiveEvent(sec) {
    let lo = 0;
    let hi = currentUiEvents.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const ev = currentUiEvents[mid];
      if (sec < ev.startSec) {
        hi = mid - 1;
      } else if (sec > ev.endSec) {
        lo = mid + 1;
      } else {
        return ev;
      }
    }
    return null;
  }

  async function activateProfile(profileId) {
    store.setState({ profile: profileId, loading: true });
    const bundle = await loader.synthesizeProfile(profileId, config);
    currentUiEvents = bundle.uiEvents;
    audio.loadSynthesizedLayers(bundle.layers, bundle.sampleRate, bundle.numSamples);
    ribbon.setEvents(bundle.uiEvents);
    atlasPanel.setEvents(bundle.uiEvents);
    renderMetricsStrip(metricsRoot, {
      manifest: bundle.manifest,
      sizes,
      synthMs: bundle.synthMs,
    });
    store.setState({
      loading: false,
      synthMs: bundle.synthMs,
      duration: bundle.numSamples / bundle.sampleRate,
    });
  }

  store.subscribe((st) => {
    const activeEv = findActiveEvent(st.currentTime);
    transport.update(st, activeEv);
    ribbon.render(st.currentTime, st.highlightedSyllableId);
  });

  function animationLoop() {
    const st = store.getState();
    if (st.playing) {
      const nowSec = audio.getCurrentTime();
      store.setState({ currentTime: nowSec });
      mixer.updateMeters(audio.getLayerRmsLevels());
    } else {
      ribbon.render(st.currentTime, st.highlightedSyllableId);
    }
    requestAnimationFrame(animationLoop);
  }

  await activateProfile(store.getState().profile);
  requestAnimationFrame(animationLoop);
}

bootStudio().catch((err) => {
  console.error('Boot failure:', err);
  const el = document.getElementById('lyric-line-display');
  if (el) el.textContent = `引擎初始化异常: ${err.message || err}`;
});
