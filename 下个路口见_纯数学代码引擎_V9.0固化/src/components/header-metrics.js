// Header & Pure-Code Byte Metrics Strip Component

function fmtKb(bytes) {
  return `${(bytes / 1024).toFixed(2)} KB`;
}

export function mountHeaderMetrics(container, store) {
  container.innerHTML = `
    <div class="header-row">
      <div class="title-block">
        <h1>《李宇春 - 下个，路口，见》V9 · 10 轨纯数学代码声学合成引擎</h1>
        <p>100% 明文数学物理模型与乐谱代码 · 零 LZMA 解码器 · 零 Base122/Base64 二进制流 · 零音频采样</p>
      </div>
      <div class="badge-strip" id="metrics-badge-strip">
        <span class="metric-pill">人声模型+总谱 <strong id="m-vocal-kb">39.87 KB</strong></span>
        <span class="metric-pill band">伴奏模型+总谱 <strong id="m-band-kb">8.50 KB</strong></span>
        <span class="metric-pill">全曲明文总谱+模型 <strong id="m-total-score-kb">48.37 KB</strong></span>
        <span class="metric-pill band">Gzip 传输体积 <strong id="m-gzip-kb">~15 KB</strong></span>
      </div>
    </div>
  `;

  const elVocal = container.querySelector('#m-vocal-kb');
  const elBand = container.querySelector('#m-band-kb');
  const elTotal = container.querySelector('#m-total-score-kb');
  const elGzip = container.querySelector('#m-gzip-kb');

  store.subscribe((state, keys) => {
    if (!keys.includes('metrics') || !state.metrics) return;
    const m = state.metrics;
    if (m.vocalModelAndScoreBytes) {
      elVocal.textContent = `${fmtKb(m.vocalModelAndScoreBytes)} (Gzip ${fmtKb(m.vocalModelAndScoreGzipBytes)})`;
    }
    if (m.bandModelAndScoreBytes) {
      elBand.textContent = `${fmtKb(m.bandModelAndScoreBytes)} (Gzip ${fmtKb(m.bandModelAndScoreGzipBytes)})`;
    }
    if (m.totalModelAndScoreBytes) {
      elTotal.textContent = fmtKb(m.totalModelAndScoreBytes);
    }
    if (m.totalModelAndScoreGzipBytes) {
      elGzip.textContent = fmtKb(m.totalModelAndScoreGzipBytes);
    }
  });
}
