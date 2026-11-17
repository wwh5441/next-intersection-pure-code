// Standalone Entry Point for V9.4.4【全端级 Omni Speaker】Pure-Math Engine
import { TIER_PHYSICS } from '../models/tier-physics.js';
import { HUFFMAN_TABLES } from '../models/huffman-algebra.js';
import { TIER_FRAMES } from '../data/tier-series-index.js';
import { createStreamingTierEngine } from './dsp/streaming-tier-engine.js';
import { createStudioStore } from './state/studio-store.js';
import { StudioAudioMixer } from './modules/audio-mixer.js';
import { mountHeaderMetrics } from './components/header-metrics.js';
import { mountTierSelectorBar } from './components/tier-selector-bar.js';
import { mountTransportBar } from './components/transport-bar.js';
import { mountScoreCanvas } from './components/score-canvas.js';
import { mountStereoAndLadder } from './components/stereo-and-ladder.js';
import { mountFormulaInspector } from './components/formula-inspector.js';

const INLINE_META = {"tierId": "v94_4", "versionCode": "V9.4.4", "badgeIndex": "④", "tierName": "全端级 Omni Speaker", "shortTag": "手机/电脑普通喇叭满血", "targetDeviceDesc": "手机外放 / 笔记本 / 普通桌面喇叭", "passbandLabel": "80 Hz – 12.2 kHz", "passbandShort": "12.2kHz", "formulaSummary": "80Hz–12.2kHz 通用扬声器物理通带 + 100–250Hz 奇偶泛音虚拟低音补偿 + 4段 J-正交宽声场", "codingStrategyDetail": "切除小口径振膜无法推动的 <80Hz 超低频与 >12.2kHz 极高频，强化上低频泛音列实现小喇叭饱满鼓贝斯听感", "architectureSummary": "48kHz 通用扬声器物理适配档 · 80Hz–12.2kHz 黄金通带 + 缺失基频虚拟低音 + 四段 J-正交参数立体声", "nonZeroModes": 756411, "waveSnrDb": 8.76, "melSnrDb": 16.2, "maxDiffVsV93Master": null, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 1904372, "totalModelAndScoreGzipBytes": 911844, "dspEngineBytes": 24312, "dspEngineGzipBytes": 6500, "totalAudioCorePlainJsBytes": 1928684, "totalAudioCoreGzipBytes": 918980, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.4_全端级_OmniSpeaker固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.4_全端级_OmniSpeaker.html", "standaloneHtmlBytes": 1987732};

function bootstrapStandalone() {
  const t0 = performance.now();
  const engine = createStreamingTierEngine(TIER_PHYSICS, HUFFMAN_TABLES, TIER_FRAMES);
  const store = createStudioStore({
    studioMode: 'standalone',
    activeTierId: INLINE_META.tierId,
    activeTierMeta: INLINE_META,
  });
  const mixer = new StudioAudioMixer(store, engine);

  mountHeaderMetrics(document.getElementById('header-slot'), store);
  mountTierSelectorBar(document.getElementById('tier-selector-slot'), store, null);
  mountTransportBar(document.getElementById('transport-slot'), store, mixer);
  mountScoreCanvas(document.getElementById('visualizer-slot'), store, mixer);
  mountStereoAndLadder(document.getElementById('mixer-slot'), store, mixer, null);
  mountFormulaInspector(document.getElementById('inspector-slot'), store, () => mixer.streamEngine);

  const warmL = new Float32Array(2048);
  const warmR = new Float32Array(2048);
  engine.synthesizeSampleRange(48000, 2048, null, warmL, warmR);
  const bootMs = Math.max(1, Math.round(performance.now() - t0));

  store.update({
    isReady: true,
    duration: engine.durationSec,
    renderTimeMs: bootMs,
    statusText: `${INLINE_META.versionCode} ${INLINE_META.tierName} 流式就绪`,
  });
  window.__STUDIO_DEBUG__ = { store, mixer, engine };
}

bootstrapStandalone();
