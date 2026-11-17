export function createTracks(container, manifest, controller) {
  container.innerHTML = '';
  for (const track of manifest.tracks) {
    const row = document.createElement('div');
    row.className = 'track-row';
    row.dataset.track = track.id;
    row.style.setProperty('--track-color', track.color);

    const titleWrap = document.createElement('div');
    titleWrap.className = 'track-meta';

    const name = document.createElement('span');
    name.className = 'track-name';
    const snrBadge = track.parametric
      ? `梅尔 SNR ${track.logMelSnrDb} dB`
      : `波形 SNR ${track.waveSnrDb} dB · 梅尔 ${track.logMelSnrDb} dB`;
    name.textContent = `${track.name} · ${(track.compressedBytes / 1024).toFixed(1)} KB · ${snrBadge}`;

    const sub = document.createElement('small');
    sub.className = 'track-sub';
    sub.textContent = track.subtitle || '';
    titleWrap.append(name, sub);

    const meter = document.createElement('span');
    meter.className = 'track-meter';
    meter.innerHTML = '<b></b>';

    const gain = document.createElement('input');
    gain.type = 'range';
    gain.min = 0;
    gain.max = 1.35;
    gain.step = 0.01;
    gain.value = controller.store.state.gains[track.id] ?? 1;
    gain.setAttribute('aria-label', `${track.name}音量`);
    gain.addEventListener('input', () => controller.setTrackGain(track.id, Number(gain.value)));

    const mute = document.createElement('button');
    mute.textContent = '静音';
    mute.dataset.action = 'mute';
    mute.setAttribute('aria-pressed', 'false');
    mute.addEventListener('click', () => controller.toggleMute(track.id));

    const solo = document.createElement('button');
    solo.textContent = '独听';
    solo.dataset.action = 'solo';
    solo.setAttribute('aria-pressed', 'false');
    solo.addEventListener('click', () => controller.toggleSolo(track.id));

    row.append(titleWrap, meter, gain, mute, solo);
    container.append(row);
  }
}

export function updateTracks(container, state) {
  for (const row of container.children) {
    const id = row.dataset.track;
    const muted = !state.enabled[id];
    const isSolo = state.solo === id;
    const silencedByOtherSolo = Boolean(state.solo && state.solo !== id);
    row.dataset.dimmed = String(muted || silencedByOtherSolo);
    row.querySelector('[data-action=mute]').setAttribute('aria-pressed', String(muted));
    row.querySelector('[data-action=solo]').setAttribute('aria-pressed', String(isSolo));
    row.querySelector('.track-meter b').style.width = `${Math.min(100, (state.meters?.[id] || 0) * 680)}%`;
  }
}
