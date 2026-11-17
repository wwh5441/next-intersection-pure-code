export function startScope(canvas, controller) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const timeData = new Uint8Array(1024);

  function frame() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 640;
    const h = canvas.clientHeight || 110;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (controller.analyser && controller.running) {
      controller.analyser.getByteTimeDomainData(timeData);
      ctx.strokeStyle = '#d9ed89';
      ctx.lineWidth = 1.9;
      ctx.beginPath();
      for (let i = 0; i < timeData.length; i++) {
        const x = (i / (timeData.length - 1)) * w;
        const y = ((timeData[i] - 128) / 128) * (h * 0.43) + h / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(217, 237, 137, 0.25)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const y = h / 2 + Math.sin(x * 0.035) * 4 * Math.sin((x / w) * Math.PI);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
