// White-Box Container, RFC 6716 Opus TOC & TIMB Feature Layer Decompiler Panel

import {
  decompilePacketAtIndex,
  buildContainerManifestJson,
  formatBinaryHexView,
} from '../../parsers/wec-plaintext-decompiler.js';

export function mountWecDecompilerPanel(container, store, getActiveBundle) {
  const tabs = [
    { id: 'packet', label: '① 实时透视：20ms Opus 裸包 TOC 模式 & TIMB 声纹帧' },
    { id: 'manifest', label: '② 容器总目录：Audio Track 表 + Feature TLV 层清单' },
    { id: 'hex', label: '③ 二进制解剖：WEC0/1/2 文件头与各轨 Opus 裸包 Hex' },
    { id: 'spec', label: '④ WEC v2 三档规范 (WEC0 / WEC1 / WEC2) 与 CLI 互通契约' },
  ];

  container.innerHTML = `
    <div class="canvas-header">
      <span><strong>WEC v2.0 白盒二进制与特征层反编译透视器（Opus TOC + TIMB + GRID + LYRC）</strong></span>
      <span class="status-hint" id="decomp-frame-summary">Packet #3080 · 61.60s</span>
    </div>
    <div class="decompiler-toolbar">
      <div class="decompiler-frame-ctrl">
        <span>透视 20ms 包序号 (0 – 10518)：</span>
        <input id="decomp-frame-slider" type="range" min="0" max="10518" step="1" value="3080" />
        <strong id="decomp-frame-num">#3080 (61.60s)</strong>
      </div>
      <label class="decompiler-follow-label">
        <input id="chk-follow-playhead" type="checkbox" checked />
        播放时实时跟随当前 20ms Opus 包与声纹帧
      </label>
    </div>
    <div class="inspector-tabs" id="inspector-tab-bar"></div>
    <pre class="code-viewer" id="inspector-code-box"></pre>
  `;

  const tabBar = container.querySelector('#inspector-tab-bar');
  const codeBox = container.querySelector('#inspector-code-box');
  const summaryEl = container.querySelector('#decomp-frame-summary');
  const frameSlider = container.querySelector('#decomp-frame-slider');
  const frameNumEl = container.querySelector('#decomp-frame-num');
  const chkFollow = container.querySelector('#chk-follow-playhead');

  tabs.forEach((t) => {
    const btn = document.createElement('button');
    btn.className = `btn-chip${t.id === 'packet' ? ' active' : ''}`;
    btn.dataset.tab = t.id;
    btn.textContent = t.label;
    btn.addEventListener('click', () => store.update({ activeInspectorTab: t.id }));
    tabBar.appendChild(btn);
  });

  frameSlider.addEventListener('input', () => {
    const idx = Number(frameSlider.value);
    store.update({ followPlayheadFrame: false, inspectedPacketIdx: idx });
  });

  chkFollow.addEventListener('change', () => {
    store.update({ followPlayheadFrame: chkFollow.checked });
  });

  let lastRenderedKey = '';

  function buildInspectorContent(state) {
    const bundle = getActiveBundle ? getActiveBundle() : null;
    if (!bundle || !bundle.parsed) {
      return '// 正在读取并校验 .wec 二进制容器...';
    }
    const { parsed } = bundle;
    const pktIdx = state.inspectedPacketIdx;

    if (state.activeInspectorTab === 'manifest') {
      return JSON.stringify(buildContainerManifestJson(parsed), null, 2);
    }

    if (state.activeInspectorTab === 'hex') {
      return formatBinaryHexView(parsed, pktIdx);
    }

    if (state.activeInspectorTab === 'spec') {
      return [
        '// === WEC v2.0 (Weckham Extensible Audio & Timbre Container) 三档统一规范 ===',
        '// 核心哲学：底层波形编码 100% 继承 RFC 6716 Opus (SILK 声道预测 + CELT 金字塔矢量量化 + 区间编码)，绝不重复造轮子；',
        '//          上层容器解决标准单流 .opus 无法分轨混音、无独立人声特征层、Ogg 页寻址笨重的痛点。',
        '',
        '// [Tier 0] WEC0 — 纯最小单流核 (Magic: "WEC0", 32-Byte 定长文件头)：',
        '//   - 严格 1 条 Opus 裸包码流 + u8_ext 紧凑包长表 + IEEE 802.3 CRC-32，0 特征层、0 JSON 冗余。',
        '//   - 比同码率 Ogg-Opus 省去 ~25KB OggS 分页头，且自带 O(1) 采样级随机跳转。',
        '',
        '// [Tier 1] WEC1 — 音色+伴奏双轨档 (Magic: "WEC1", 48-Byte 文件头 + 2×48B Track Dir)：',
        '//   - 封装 Track 0 (人声/主音色干声，可独立启用 SILK/Hybrid) + Track 1 (纯伴奏立体声 CELT)。',
        '//   - 内嵌 TIMB 50Hz 帧对齐声纹层 (F0 音高 + 清浊音 + 四段共振峰包络)，单文件支持一键 KTV 消原唱、清唱干声提取与免解码读音高。',
        '',
        '// [Tier 2] WEC2 — 全分轨可扩展母版档 (Magic: "WEC2", 强制 numAudioTracks >= 4 且 numFeatureLayers >= 3，上不设限)：',
        '//   - 至少 4 条相位锁定的独立 Opus 音轨 (支持 4 轨、6 轨、12 轨乃至 65,535 轨无限扩展)。',
        '//   - 至少 3 层结构化 TLV 特征层：TIMB (声纹基频层) + GRID (小节拍号和弦网格层) + LYRC (逐字音节事件层) + SPAT / MIDI / 自定义扩展层。',
        '//   - 支持由 WEC-CLI 无损或确定性向下折叠转换：WEC2 ➔ WEC1 (合并伴奏分轨) ➔ WEC0 (极简混缩单流)。',
      ].join('\n');
    }

    const pktAst = decompilePacketAtIndex(parsed, pktIdx);
    return JSON.stringify(pktAst, null, 2);
  }

  function refresh(state) {
    const bundle = getActiveBundle ? getActiveBundle() : null;
    const maxPkt = (bundle?.parsed?.tracks?.[0]?.packetCount || 10519) - 1;
    frameSlider.max = String(maxPkt);

    const k = Math.min(maxPkt, state.inspectedPacketIdx);
    const tSec = (k * 0.02).toFixed(2);
    frameSlider.value = String(k);
    frameNumEl.textContent = `#${k} (${tSec}s)`;
    summaryEl.textContent = `Opus Packet #${k} · ${tSec}s · ${state.followPlayheadFrame ? '实时跟随播放头' : '手动锁定包'}`;
    chkFollow.checked = state.followPlayheadFrame;

    tabBar.querySelectorAll('[data-tab]').forEach((b) => {
      b.classList.toggle('active', b.dataset.tab === state.activeInspectorTab);
    });

    // Throttle live JSON stringification during playback to ~12fps so UI stays buttery smooth
    const quantK = state.isPlaying && state.followPlayheadFrame ? Math.floor(k / 4) * 4 : k;
    const renderKey = `${state.activeTierId}|${state.activeInspectorTab}|${quantK}|${state.isReady}`;
    if (renderKey !== lastRenderedKey) {
      lastRenderedKey = renderKey;
      codeBox.textContent = buildInspectorContent({ ...state, inspectedPacketIdx: k });
    }
  }

  refresh(store.getState());
  store.subscribe((state, keys) => {
    if (
      keys.includes('inspectedPacketIdx') ||
      keys.includes('activeInspectorTab') ||
      keys.includes('activeTierId') ||
      keys.includes('activeTierMeta') ||
      keys.includes('followPlayheadFrame') ||
      keys.includes('isReady')
    ) {
      refresh(state);
    }
  });
}
