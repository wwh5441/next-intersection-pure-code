import { VocalStudioStore } from './state/store.js';
import { VocalAudioController } from './modules/audio-controller.js';
import { createPresets } from './components/presets.js';
import { createTracks } from './components/tracks.js';
import { startScope } from './components/scope.js';
import { renderVocalInspector } from './components/inspector.js';
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
  const profiles = await (await fetch('./data/profiles.json')).json();
  const manifest = await (await fetch(`./data/${profiles.default}/manifest.json`)).json();

  const dom = {
    currentTime: $('#current-time'),
    totalTime: $('#total-time'),
    seek: $('#seek'),
    play: $('#play'),
    jumpVerse: $('#jump-verse'),
    jumpChorus: $('#jump-chorus'),
    volume: $('#volume'),
    engineStatus: $('#engine-status'),
    sectionName: $('#section-name'),
    lyric: $('#lyric'),
    modeDescription: $('#mode-description'),
    calculation: $('#calculation'),
    presets: $('#presets'),
    tracks: $('#tracks'),
    scope: $('#scope'),
    timbreCanvas: $('#timbre-curve'),
    formulaList: $('#formula-list'),
    profileButtons: [...document.querySelectorAll('[data-profile]')],
  };

  const store = new VocalStudioStore(profiles.default, 22.6);
  const controller = new VocalAudioController(manifest, store);
  controller.onError = showError;
  controller.onProfileChanged = (nextManifest) => {
    createPresets(dom.presets, nextManifest.presets, controller);
    createTracks(dom.tracks, nextManifest, controller);
    renderVocalInspector(dom.timbreCanvas, dom.formulaList, nextManifest);
  };

  createPresets(dom.presets, manifest.presets, controller);
  createTracks(dom.tracks, manifest, controller);
  renderVocalInspector(dom.timbreCanvas, dom.formulaList, manifest);
  store.subscribe((state) => renderPlayerState(dom, controller, state));

  dom.seek.max = manifest.duration;
  dom.totalTime.textContent = formatClock(Math.ceil(manifest.duration));

  dom.play.addEventListener(
    'click',
    guarded(async () => (controller.running ? controller.pause() : controller.play()))
  );
  dom.jumpVerse.addEventListener(
    'click',
    guarded(async () => {
      controller.seek(22.6);
      if (!controller.running) await controller.play();
    })
  );
  dom.jumpChorus.addEventListener(
    'click',
    guarded(async () => {
      controller.seek(61.6);
      if (!controller.running) await controller.play();
    })
  );
  dom.seek.addEventListener('change', () => controller.seek(Number(dom.seek.value)));
  dom.volume.addEventListener('input', () => controller.setVolume(Number(dom.volume.value)));

  for (const button of dom.profileButtons) {
    if (!profiles.available.includes(button.dataset.profile)) {
      button.disabled = true;
    } else {
      button.addEventListener('click', guarded(() => controller.setProfile(button.dataset.profile)));
    }
  }

  window.addEventListener('keydown', (e) => {
    if (e.target && ['INPUT', 'BUTTON', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.code === 'Space') {
      e.preventDefault();
      dom.play.click();
    }
  });

  fetch('./data/sizes.json')
    .then((res) => (res.ok ? res.json() : null))
    .then(renderSizeMetrics)
    .catch(() => {});

  startScope(dom.scope, controller);
  store.emit();
} catch (error) {
  showError(error);
}
