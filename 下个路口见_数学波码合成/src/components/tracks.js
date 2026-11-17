export function createTracks(container, manifest, controller) {
  container.innerHTML = '';
  for (const track of manifest.tracks) {
    const row = document.createElement('div');
    row.className = 'track-row';
    row.dataset.track = track.id;
    row.style.setProperty('--track-color', track.color);

    const name = document.createElement('span');
    name.className = 'track-name';
    name.textContent = `${track.name} (${(track.compressedBytes / 1024).toFixed(1)} KB)`;

    const meter = document.createElement('span');
    meter.className = 'track-meter';
    meter.innerHTML = '<b></b>';

    const gain = document.createElement('input');
    gain.type = 'range';
    gain.min = 0;
    gain.max = 1.25;
    gain.step = 0.01;
    gain.value = 1;
    gain.setAttribute('aria-label', `${track.name}音量`);
    gain.addEventListener('input', () => controller.setTrackGain(track.id, Number(gain.value)));

    const mute = document.createElement('button');
    mute.textContent = '静音';
    mute.dataset.action = 'mute';
    mute.setAttribute('aria-pressed', 'false');
    mute.setAttribute('aria-label', `${track.name}静音`);
    mute.addEventListener('click', () => controller.toggleMute(track.id));

    const solo = document.createElement('button');
    solo.textContent = '独听';
    solo.dataset.action = 'solo';
    solo.setAttribute('aria-pressed', 'false');
    solo.setAttribute('aria-label', `${track.name}独听`);
    solo.addEventListener('click', () => controller.toggleSolo(track.id));

    row.append(name, meter, gain, mute, solo);
    container.append(row);
  }
}

export function updateTracks(container, state) {
  for (const row of container.children) {
    const id = row.dataset.track;
    row.querySelector('[data-action=mute]').setAttribute('aria-pressed', String(!state.enabled[id]));
    row.querySelector('[data-action=solo]').setAttribute('aria-pressed', String(state.solo === id));
    row.querySelector('.track-meter b').style.width = `${Math.min(100, (state.meters?.[id] || 0) * 700)}%`;
  }
}
