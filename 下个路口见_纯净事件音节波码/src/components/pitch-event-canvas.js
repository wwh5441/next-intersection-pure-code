export class PitchEventCanvasComponent {
  constructor(canvasEl, { onSeekToSec }) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.onSeekToSec = onSeekToSec;
    this.events = [];
    this.windowSec = 10.0;

    this.canvas.addEventListener('click', (ev) => {
      const rect = this.canvas.getBoundingClientRect();
      const ratioX = (ev.clientX - rect.left) / Math.max(1, rect.width);
      const tStart = Math.max(0, (this._lastTime || 0) - 2.5);
      const targetSec = tStart + ratioX * this.windowSec;
      this.onSeekToSec(targetSec);
    });
  }

  setEvents(uiEvents) {
    this.events = uiEvents || [];
  }

  render(currentTime, highlightedSyllableId = null) {
    this._lastTime = currentTime;
    const dpr = globalThis.devicePixelRatio || 1;
    const w = this.canvas.clientWidth || 680;
    const h = this.canvas.clientHeight || 256;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }

    const ctx = this.ctx;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#14161b';
    ctx.fillRect(0, 0, w, h);

    const tLeft = Math.max(0, currentTime - 2.5);
    const tRight = tLeft + this.windowSec;
    const midiMin = 51; // D#3 (~155Hz)
    const midiMax = 75; // D#5 (~622Hz)

    const xOfSec = (s) => ((s - tLeft) / this.windowSec) * w;
    const yOfMidi = (m) => h - 26 - ((m - midiMin) / (midiMax - midiMin)) * (h - 50);

    // Subtle pitch grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (let m = 54; m <= 72; m += 3) {
      const y = yOfMidi(m);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Time grid ticks every 1s
    ctx.fillStyle = 'rgba(255,255,255,0.34)';
    ctx.font = '10px "JetBrains Mono", monospace';
    for (let s = Math.ceil(tLeft); s <= Math.floor(tRight); s++) {
      const x = xOfSec(s);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h - 20);
      ctx.stroke();
      ctx.fillText(`${s}s`, x + 3, h - 7);
    }

    // Draw syllable events in visible window
    for (const ev of this.events) {
      if (ev.endSec < tLeft || ev.startSec > tRight) continue;
      const x0 = xOfSec(ev.startSec);
      const x1 = Math.max(x0 + 5, xOfSec(ev.endSec));
      const xCons = Math.min(x1, xOfSec(ev.startSec + ev.consSec));
      const yMid = yOfMidi(Math.max(midiMin + 1, Math.min(midiMax - 1, ev.midi || 60)));
      const boxH = 22;

      const isActive = currentTime >= ev.startSec && currentTime <= ev.endSec;
      const isHighlighted = highlightedSyllableId !== null && ev.syllableId === highlightedSyllableId;

      // Consonant Attack Zone (Teal)
      ctx.fillStyle = isActive ? '#3dd5c3' : 'rgba(42, 157, 143, 0.78)';
      ctx.fillRect(x0, yMid - boxH / 2, Math.max(3, xCons - x0), boxH);

      // Vowel Sustain Zone (Vermilion or Gold if highlighted)
      ctx.fillStyle = isHighlighted
        ? 'rgba(201, 138, 44, 0.92)'
        : isActive
          ? '#ff785a'
          : 'rgba(232, 93, 63, 0.45)';
      ctx.fillRect(xCons, yMid - boxH / 2, Math.max(2, x1 - xCons), boxH);

      // Continuous F0 spline curve overlay
      if (ev.pitchPreview && ev.pitchPreview.length > 1) {
        ctx.strokeStyle = isActive ? '#ffffff' : '#ffd166';
        ctx.lineWidth = isActive ? 2.2 : 1.5;
        ctx.beginPath();
        for (let k = 0; k < ev.pitchPreview.length; k++) {
          const u = k / (ev.pitchPreview.length - 1);
          const px = x0 + u * (x1 - x0);
          const mVal = 69.0 + 12.0 * Math.log2(ev.pitchPreview[k] / 440.0);
          const py = yOfMidi(Math.max(midiMin, Math.min(midiMax, mVal)));
          if (k === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }

      // Label character + pinyin above event block
      ctx.fillStyle = isActive ? '#ffffff' : 'rgba(255,255,255,0.85)';
      ctx.font = isActive ? '600 11.5px "PingFang SC", sans-serif' : '11px "PingFang SC", sans-serif';
      ctx.fillText(`${ev.displayText}(${ev.pinyin})`, x0 + 2, yMid - boxH / 2 - 5);
    }

    // Playhead cursor
    const playX = xOfSec(currentTime);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, h);
    ctx.stroke();

    ctx.restore();
  }
}
