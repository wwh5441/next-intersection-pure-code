// Header & Verified WEC v2.0 (WEC0 / WEC1 / WEC2 + RFC 6716 Opus) Metrics Component

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
        <h1 id="header-main-title">《下个，路口，见》WEC v2.0 (Opus 内核 · WEC0 / WEC1 / WEC2 三档分层工作台)</h1>
        <p id="header-subtitle">不重复造轮子：底层继承 RFC 6716 Opus (SILK+CELT) 工业级音频压缩 · 上层赋予 WEC0 纯最小 / WEC1 音色+伴奏双轨 / WEC2 (≥4+3 无上限) 分轨与声纹特征直读</p>
      </div>
      <div class="badge-strip" id="metrics-badge-strip">
        <span class="metric-pill band">.wec 落盘体积 <strong id="m-wec-sz">--</strong></span>
        <span class="metric-pill">分层结构 <strong id="m-layers">--</strong></span>
        <span class="metric-pill">CRC-32 完整性 <strong id="m-crc32">--</strong></span>
        <span class="metric-pill band">Opus 解码缓存 <strong id="m-ram-mb">--</strong></span>
      </div>
    </div>
  `;

  const elTitle = container.querySelector('#header-main-title');
  const elSub = container.querySelector('#header-subtitle');
  const elWec = container.querySelector('#m-wec-sz');
  const elLayers = container.querySelector('#m-layers');
  const elCrc = container.querySelector('#m-crc32');
  const elRam = container.querySelector('#m-ram-mb');

  function refresh(state) {
    elRam.textContent = `${state.pcmRamMb.toFixed(1)} MB`;
    const meta = state.activeTierMeta;
    if (!meta) return;

    elTitle.textContent = `《下个，路口，见》WEC v2.0 原生 Opus 母版工作台 · 当前：${meta.versionCode}【${meta.name}】`;
    elSub.textContent = `文件：${meta.wecFilename} · Magic: ${meta.magic} (Spec v2) · ${meta.passband} · ${meta.scenario}`;
    elWec.textContent = `${formatByteSize(meta.wecBytes)} (${meta.equivKbps} kbps)`;
    const tagList = meta.featureTags && meta.featureTags.length > 0 ? ` [${meta.featureTags.join('+')}]` : ' [纯单流零特征开销]';
    elLayers.textContent = `${meta.audioTracks} 音轨 + ${meta.featureLayers} 特征层${tagList}`;
    elCrc.textContent = `${meta.crcHex} ✓`;
  }

  refresh(store.getState());
  store.subscribe((state, keys) => {
    if (keys.includes('activeTierMeta') || keys.includes('pcmRamMb')) {
      refresh(state);
    }
  });
}
