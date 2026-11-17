export function createPresets(container, presets, controller) {
  if (!container) return;
  container.innerHTML = '';
  for (const preset of presets || []) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-chip';
    btn.dataset.preset = preset.id;
    btn.textContent = preset.label;
    btn.setAttribute('aria-pressed', String(preset.id === 'full_mix'));
    btn.addEventListener('click', () => controller.applyPreset(preset));
    container.append(btn);
  }
}

export function updatePresets(container, state) {
  if (!container) return;
  for (const btn of container.children) {
    btn.setAttribute('aria-pressed', String(btn.dataset.preset === state.activePreset));
  }
}
