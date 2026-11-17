import { VocalStudioStore } from './state/store.js';
import { VocalAudioController } from './modules/audio-controller.js';
import { createPresets } from './components/presets.js';
import { createTracks } from './components/tracks.js';
import { startScope } from './components/scope.js';
import { renderVocalInspector } from './components/inspector.js';
import { COMBOS, renderCompareCards } from './components/compare-cards.js';
import { formatClock, renderPlayerState, renderSizeMetrics } from './components/player-view.js';

const $ = (selector) => document.querySelector(selector);

function showError(error) {
  const el = $('#error');
  el.textContent = error.message || String(error);
  el.hidden = false;
}

let busy = false;
function guarded(action) {
  return async (event) => {
    if (busy) return;
    busy = true;
    $('#error').hidden = true;
    try {
      await action(event);
    } catch (error) {
      showError(error);
    } finally {
      busy = false;
    }
  };
}

try {
  const initialCombo = '8track';
  const initialProfile = 'compact';

  const [manifest, ...sizesList] = await Promise.all([
    fetch(`./data/${initialCombo}/${initialProfile}/manifest.json`).then((r) => r.json()),
    ...COMBOS.map((c) => fetch(`./data/${c.id}/sizes.json`).then((r) => r.json())),
  ]);
  const sizesByCombo = Object.fromEntries(COMBOS.map((c, idx) => [c.id, sizesList[idx]]));

  const dom = {
    currentTime: $('#current-time'),
    totalTime: $('#total-time'),
    seek: $('#seek'),
    play: $('#play'),
    jumpIntro: $('#jump-intro'),
    jumpVerse: $('#jump-verse'),
    jumpChorus: $('#jump-chorus'),
    volume: $('#volume'),
    engineStatus: $('#engine-status'),
    sectionName: $('#section-name'),
    lyric: $('#lyric'),
    modeDescription: $('#mode-description'),
    calculation: $('#calculation'),
    comboCards: $('#combo-cards'),
    presets: $('#presets'),
    tracks: $('#tracks'),
    scope: $('#scope'),
    timbreCanvas: $('#timbre-curve'),
    formulaList: $('#formula-list'),
    profileButtons: [...document.querySelectorAll('[data-profile]')],
  };

  const store = new VocalStudioStore(initialCombo, initialProfile, 0);
  const controller = new VocalAudioController(manifest, store);
  controller.onError = showError;

  const onSelectCombo = guarded((comboId) => controller.setCombo(comboId));

  controller.onManifestChanged = (nextManifest) => {
    renderCompareCards(dom.comboCards, sizesByCombo, store.state, onSelectCombo);
    createPresets(dom.presets, nextManifest.presets, controller);
    createTracks(dom.tracks, nextManifest, controller);
    renderVocalInspector(dom.timbreCanvas, dom.formulaList, nextManifest);
    renderSizeMetrics(sizesByCombo, store.state.combo);
  };

  renderCompareCards(dom.comboCards, sizesByCombo, store.state, onSelectCombo);
  createPresets(dom.presets, manifest.presets, controller);
  createTracks(dom.tracks, manifest, controller);
  renderVocalInspector(dom.timbreCanvas, dom.formulaList, manifest);
  renderSizeMetrics(sizesByCombo, store.state.combo);
  store.subscribe((state) => renderPlayerState(dom, controller, state));

  dom.seek.max = manifest.duration;
  dom.totalTime.textContent = formatClock(Math.ceil(manifest.duration));

  dom.play.addEventListener(
    'click',
    guarded(async () => (controller.running ? controller.pause() : controller.play()))
  );
  dom.jumpIntro?.addEventListener(
    'click',
    guarded(async () => {
      controller.seek(0);
      if (!controller.running) await controller.play();
    })
  );
  dom.jumpVerse?.addEventListener(
    'click',
    guarded(async () => {
      controller.seek(22.6);
      if (!controller.running) await controller.play();
    })
  );
  dom.jumpChorus?.addEventListener(
    'click',
    guarded(async () => {
      controller.seek(61.6);
      if (!controller.running) await controller.play();
    })
  );
  dom.seek.addEventListener('change', () => controller.seek(Number(dom.seek.value)));
  dom.volume.addEventListener('input', () => controller.setVolume(Number(dom.volume.value)));

  for (const button of dom.profileButtons) {
    button.addEventListener('click', guarded(() => controller.setProfile(button.dataset.profile)));
  }

  window.addEventListener('keydown', (e) => {
    if (e.target && ['INPUT', 'BUTTON', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.code === 'Space') {
      e.preventDefault();
      dom.play.click();
    }
  });

  startScope(dom.scope, controller);
  store.emit();
} catch (error) {
  showError(error);
}
