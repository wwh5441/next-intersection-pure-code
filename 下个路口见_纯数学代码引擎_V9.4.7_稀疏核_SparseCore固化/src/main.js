// Standalone Entry Point for V9.4.7【稀疏核 Sparse Core】Pure-Math Engine
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

const INLINE_META = {"tierId": "v94_7", "versionCode": "V9.4.7", "badgeIndex": "⑦", "tierName": "稀疏核 Sparse Core", "shortTag": "温润中频本征核 · 零电流声", "targetDeviceDesc": "超低码率纯数学正交核心 (~6.5 kbps Gzip)", "passbandLabel": "165 Hz – 3.6 kHz (温润黑胶中频核)", "passbandShort": "165Hz–3.6kHz", "formulaSummary": "165Hz–3.6kHz 双帧时域锁相谐波本征核 + 孤立单帧颗粒归零 + 3.7kHz 四阶巴特沃斯温润平滑", "codingStrategyDetail": "聚焦人声与木吉他黄金中频带，强制跨帧谐波连续性滤除一切离散电音颗粒，以 ~425KB 明文呈现温润无噪真声", "architectureSummary": "48kHz 稀疏本征核 · 165Hz–3.6kHz 跨帧连续谐波正交展开 + 四阶巴特沃斯模拟感滚降 · 零电流滋滋声", "nonZeroModes": 96083, "waveSnrDb": 1.55, "melSnrDb": 8.15, "maxDiffVsV93Master": null, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 408177, "totalModelAndScoreGzipBytes": 193508, "dspEngineBytes": 27140, "dspEngineGzipBytes": 7371, "totalAudioCorePlainJsBytes": 435317, "totalAudioCoreGzipBytes": 201454, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.7_稀疏核_SparseCore固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.7_稀疏核_SparseCore.html", "standaloneHtmlBytes": 494586, "inBandSnrDb": 9.46};

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
