export function startScope(canvas, controller) {
  const ctx = canvas.getContext('2d');
  const values = new Float32Array(2048);

  function draw() {
    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
      canvas.width = Math.round(w * ratio);
      canvas.height = Math.round(h * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = '#344133';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    for (let x = 0; x < w; x += 48) {
      ctx.beginPath();
      ctx.moveTo(x, h / 2 - 3);
      ctx.lineTo(x, h / 2 + 3);
      ctx.stroke();
    }

    if (controller.analyser && controller.running) {
      controller.analyser.getFloatTimeDomainData(values);
    } else {
      for (let i = 0; i < values.length; i++) values[i] *= 0.88;
    }

    ctx.strokeStyle = '#d9ed89';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < values.length; i += 2) {
      const x = (i / (values.length - 1)) * w;
      const y = h / 2 - values[i] * h * 1.8;
      if (!i) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
}
