export function renderVocalInspector(canvas, formulaContainer, manifest) {
  if (formulaContainer) {
    formulaContainer.innerHTML = '';
    for (const track of manifest.tracks) {
      const item = document.createElement('div');
      item.className = 'formula-item';
      item.style.setProperty('--item-color', track.color);

      const head = document.createElement('div');
      head.className = 'formula-head';
      const snrText = track.parametric
        ? `${(track.compressedBytes / 1024).toFixed(1)} KB · 梅尔频谱 SNR ${track.logMelSnrDb} dB`
        : `${(track.compressedBytes / 1024).toFixed(1)} KB · 波形 SNR ${track.waveSnrDb} dB · 梅尔 ${track.logMelSnrDb} dB`;
      head.innerHTML = `<b>${track.name}</b><em>${snrText}</em>`;

      const code = document.createElement('div');
      code.className = 'formula-code';
      code.textContent = track.formula;

      item.append(head, code);
      formulaContainer.append(item);
    }
  }

  if (!canvas || !manifest.timbreProfile) return;
  const { staticTimbreLogMel, pronunciationStdLogMel, consonantAirLogMel } = manifest.timbreProfile;
  if (!staticTimbreLogMel?.length) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 380;
  const h = canvas.clientHeight || 112;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const n = staticTimbreLogMel.length;
  const minV = Math.min(...staticTimbreLogMel, ...consonantAirLogMel) - 0.35;
  const maxV = Math.max(...staticTimbreLogMel, ...consonantAirLogMel) + 0.35;

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let g = 1; g <= 3; g++) {
    const y = (h * g) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Shaded pronunciation modulation envelope (+/- std)
  ctx.fillStyle = 'rgba(124, 224, 195, 0.14)';
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const v = staticTimbreLogMel[i] + pronunciationStdLogMel[i];
    const y = h - 10 - ((v - minV) / (maxV - minV + 1e-6)) * (h - 24);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  for (let i = n - 1; i >= 0; i--) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const v = staticTimbreLogMel[i] - pronunciationStdLogMel[i];
    const y = h - 10 - ((v - minV) / (maxV - minV + 1e-6)) * (h - 24);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();

  // Static speaker timbre curve
  ctx.strokeStyle = '#d9ed89';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const y = h - 10 - ((staticTimbreLogMel[i] - minV) / (maxV - minV + 1e-6)) * (h - 24);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Consonant & high-air curve
  ctx.strokeStyle = 'rgba(156, 201, 227, 0.85)';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const y = h - 10 - ((consonantAirLogMel[i] - minV) / (maxV - minV + 1e-6)) * (h - 24);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.58)';
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText('70Hz · 音色核[绿] / 咬字口型动态带[青] / 唇齿气声谱[蓝] · 18.5kHz', 8, 13);
}
