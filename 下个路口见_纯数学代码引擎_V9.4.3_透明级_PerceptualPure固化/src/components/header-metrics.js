// Header & Verified Pure-Math Fidelity / Byte / RAM Metrics Strip Component

export function formatByteSize(bytes) {
  if (bytes >= 1048576) {
    return `${(bytes / 1048576).toFixed(2)} MB`;
  }
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export function mountHeaderMetrics(container, store) {
  container.innerHTML = `
    <div class="header-row">
      <div class="title-block">
        <h1 id="header-main-title">《李宇春 - 下个，路口，见》纯数学正交余弦母带引擎</h1>
        <p id="header-subtitle">51-Bit (3×17-Bit) 十进制安全整数算术编码 + 四分类规范霍夫曼代数 + J-正交声场与纯数学 SBR 延展 · 零 LZMA · 零 Base64 · 零音频文件</p>
      </div>
      <div class="badge-strip" id="metrics-badge-strip">
        <span class="metric-pill">频宽 / 梅尔 SNR <strong id="m-snr-db">--</strong></span>
        <span class="metric-pill band">纯明文数学代码 <strong id="m-plain-sz">--</strong></span>
        <span class="metric-pill">Gzip 传输体积 <strong id="m-gzip-sz">--</strong></span>
        <span class="metric-pill band">实时流式内存 <strong id="m-ram-mb">0.38 MB</strong></span>
      </div>
    </div>
  `;

  const elTitle = container.querySelector('#header-main-title');
  const elSub = container.querySelector('#header-subtitle');
  const elSnr = container.querySelector('#m-snr-db');
  const elPlain = container.querySelector('#m-plain-sz');
  const elGzip = container.querySelector('#m-gzip-sz');
  const elRam = container.querySelector('#m-ram-mb');

  function refresh(state) {
    elRam.textContent = `${state.pcmRamMb.toFixed(2)} MB`;
    const meta = state.activeTierMeta;
    if (!meta) return;

    if (state.studioMode === 'unified') {
      elTitle.textContent = `《李宇春 - 下个，路口，见》七档纯数学率失真总控台 · 当前：${meta.versionCode}【${meta.tierName}】`;
    } else {
      elTitle.textContent = `《李宇春 - 下个，路口，见》${meta.versionCode}【${meta.tierName}】· 48kHz 纯数学代码母带引擎`;
    }
    elSub.textContent = meta.architectureSummary;
    elSnr.textContent = `${meta.passbandLabel} · Mel ${meta.melSnrDb.toFixed(1)} dB`;
    elPlain.textContent = formatByteSize(meta.totalAudioCorePlainJsBytes);
    elGzip.textContent = `${formatByteSize(meta.totalAudioCoreGzipBytes)} (WAV 38.52 MB)`;
  }

  refresh(store.getState());
  store.subscribe((state, keys) => {
    if (
      keys.includes('activeTierMeta') ||
      keys.includes('pcmRamMb') ||
      keys.includes('studioMode')
    ) {
      refresh(state);
    }
  });
}
