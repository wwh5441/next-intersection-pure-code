// Header & Verified Pure-Math Fidelity / Byte / RAM Metrics Strip Component

function fmtMb(bytes) {
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

export function mountHeaderMetrics(container, store) {
  const initSt = store.getState();
  container.innerHTML = `
    <div class="header-row">
      <div class="title-block">
        <h1 id="header-main-title">《李宇春 - 下个，路口，见》${initSt.versionLabel} · 48kHz 几乎无损纯数学正交余弦母带引擎</h1>
        <p id="header-subtitle">Princen-Bradley 2048/4×512 正交局部余弦基 + γ=4/3 听感压扩 + 6 轨 32-ERB 精确单位分解 (ΣW_i ≡ 128) · 零 LZMA · 零 Base122 · 零音频文件</p>
      </div>
      <div class="badge-strip" id="metrics-badge-strip">
        <span class="metric-pill">母带波形/梅尔 SNR <strong id="m-snr-db">28.2 dB / 39.1 dB</strong></span>
        <span class="metric-pill band">纯明文数学代码 <strong id="m-plain-mb">-- MB</strong></span>
        <span class="metric-pill">Gzip 传输体积 <strong id="m-gzip-mb">-- MB</strong></span>
        <span class="metric-pill band">实时 PCM 内存 <strong id="m-ram-mb">${initSt.pcmRamMb.toFixed(2)} MB</strong></span>
      </div>
    </div>
  `;

  const elTitle = container.querySelector('#header-main-title');
  const elSub = container.querySelector('#header-subtitle');
  const elSnr = container.querySelector('#m-snr-db');
  const elPlain = container.querySelector('#m-plain-mb');
  const elGzip = container.querySelector('#m-gzip-mb');
  const elRam = container.querySelector('#m-ram-mb');

  function refreshHeader(state) {
    elRam.textContent = `${state.pcmRamMb.toFixed(2)} MB`;
    const m = state.metrics;
    if (!m) return;

    if (m.versionShort) {
      elTitle.textContent = `《李宇春 - 下个，路口，见》${m.versionShort} · 48kHz 几乎无损纯数学正交余弦母带引擎`;
    }
    if (m.architectureSummary) {
      elSub.textContent = m.architectureSummary;
    }
    if (m.masterWaveSnrDb && m.masterLogMelSnrDb) {
      elSnr.textContent = `${m.masterWaveSnrDb.toFixed(2)} dB / ${m.masterLogMelSnrDb.toFixed(2)} dB`;
    }
    if (m.totalModelAndScoreBytes) {
      elPlain.textContent = fmtMb(m.totalModelAndScoreBytes);
    }
    if (m.totalModelAndScoreGzipBytes) {
      elGzip.textContent = `${fmtMb(m.totalModelAndScoreGzipBytes)} (vs WAV 40.39 MB)`;
    }
  }

  refreshHeader(store.getState());
  store.subscribe((state, keys) => {
    if (keys.includes('metrics') || keys.includes('pcmRamMb') || keys.includes('versionLabel')) {
      refreshHeader(state);
    }
  });
}
