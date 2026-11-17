export function renderDisentangleInspector(canvas, formulaContainer, manifest) {
  if (formulaContainer) {
    formulaContainer.innerHTML = '';
    for (const track of manifest.tracks) {
      const item = document.createElement('div');
      item.className = 'formula-item';
      item.style.setProperty('--item-color', track.color);

      const head = document.createElement('div');
      head.className = 'formula-head';
      head.innerHTML = `<b>${track.name}</b><em>${(track.compressedBytes / 1024).toFixed(1)} KB · Mel-SNR ${track.logMelSnrDb} dB</em>`;

      const code = document.createElement('div');
      code.className = 'formula-code';
      code.textContent = track.formula;

      item.append(head, code);
      formulaContainer.append(item);
    }
  }

  if (!canvas) return;
  const vocalTrack = manifest.tracks.find((t) => t.id === 'vocal');
  if (!vocalTrack?.model?.staticTimbreLogMel) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 360;
  const h = canvas.clientHeight || 92;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const staticMel = vocalTrack.model.staticTimbreLogMel;
  const pitchSlope = vocalTrack.model.pitchSlopeLogMel;
  const n = staticMel.length;
  const minV = Math.min(...staticMel) - 0.3;
  const maxV = Math.max(...staticMel) + 0.3;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let g = 1; g <= 3; g++) {
    const y = (h * g) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw static vocal-tract timbre curve T_LYC(m)
  ctx.strokeStyle = '#d9ed89';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const norm = (staticMel[i] - minV) / (maxV - minV + 1e-6);
    const y = h - 10 - norm * (h - 22);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw pitch-register modulation curve beta_LYC(m)
  ctx.strokeStyle = 'rgba(156, 201, 227, 0.78)';
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * (w - 16) + 8;
    const y = h * 0.52 - pitchSlope[i] * 14;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = 'rgba(255,255,255,0.55)';
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText('70Hz · 静态生理音色核 T_LYC(m) [绿] / 音区偏移 β(m) [蓝] · 11.6kHz', 8, 13);
}
