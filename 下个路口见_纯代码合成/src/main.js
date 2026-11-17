import { StudioStore } from './state/store.js';
import { AudioController } from './modules/audio-controller.js';
import { createTracks } from './components/tracks.js';
import { startScope } from './components/scope.js';
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
    restart: $('#restart'),
    volume: $('#volume'),
    engineStatus: $('#engine-status'),
    sectionName: $('#section-name'),
    lyric: $('#lyric'),
    modeDescription: $('#mode-description'),
    calculation: $('#calculation'),
    tracks: $('#tracks'),
    scope: $('#scope'),
    profileButtons: [...document.querySelectorAll('[data-profile]')],
  };

  const store = new StudioStore(profiles.default);
  const controller = new AudioController(manifest, store);
  controller.onError = showError;

  createTracks(dom.tracks, manifest, controller);
  store.subscribe((state) => renderPlayerState(dom, controller, state));

  dom.seek.max = manifest.duration;
  dom.totalTime.textContent = formatClock(Math.ceil(manifest.duration));

  dom.play.addEventListener(
    'click',
    guarded(async () => (controller.running ? controller.pause() : controller.play()))
  );
  dom.restart.addEventListener('click', () => {
    controller.pause();
    controller.seek(0);
  });
  dom.seek.addEventListener('change', () => controller.seek(Number(dom.seek.value)));
  dom.volume.addEventListener('input', () => controller.setVolume(Number(dom.volume.value)));

  for (const button of dom.profileButtons) {
    if (!profiles.available.includes(button.dataset.profile)) {
      button.disabled = true;
      button.querySelector('small').textContent = '此单文件精简包仅内嵌紧凑版';
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
