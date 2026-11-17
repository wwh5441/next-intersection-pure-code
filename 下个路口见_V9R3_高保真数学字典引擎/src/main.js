// V9R3 High-Fidelity Pure Math + Code + Dictionary Studio Bootstrapper (`src/main.js`)
import { DEFAULT_V9R3_TIER, V9R3_TIERS } from '../data/tiers-registry.js';
import { studioStore } from './state/studio-store.js';
import { audioMixer } from './modules/audio-mixer.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountScoreMotifCanvas } from './components/score-motif-canvas.js';
import { mountStemMixer } from './components/stem-mixer.js';
import { mountDictionaryInspector } from './components/dictionary-inspector.js';

function bootstrapV9r3Studio() {
  mountHeaderMetrics(document.getElementById('panel-header-metrics'));
  mountTransportBar(document.getElementById('panel-transport'));
  mountScoreMotifCanvas(document.getElementById('panel-visualizer'));
  mountStemMixer(document.getElementById('panel-mixer'));
  mountDictionaryInspector(document.getElementById('panel-inspector'));

  window.__V9R3_STUDIO__ = {
    store: studioStore,
    mixer: audioMixer,
    tiers: V9R3_TIERS,
  };

  const initialTier = V9R3_TIERS[DEFAULT_V9R3_TIER]
    ? DEFAULT_V9R3_TIER
    : Object.keys(V9R3_TIERS)[0];
  audioMixer.loadTier(initialTier);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrapV9r3Studio);
} else {
  bootstrapV9r3Studio();
}
