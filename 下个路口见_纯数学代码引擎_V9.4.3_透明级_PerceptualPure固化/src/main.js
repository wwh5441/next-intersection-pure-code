// Standalone Entry Point for V9.4.3【透明级 Perceptual Pure】Pure-Math Engine
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

const INLINE_META = {"tierId": "v94_3", "versionCode": "V9.4.3", "badgeIndex": "③", "tierName": "透明级 Perceptual Pure", "shortTag": "人耳ERB掩蔽透明级", "targetDeviceDesc": "日常入耳/头戴耳机盲听透明", "passbandLabel": "20 Hz – 18.0 kHz", "passbandShort": "18.0kHz", "formulaSummary": "32-ERB 同频掩蔽裁剪 + >9.8kHz 反对称正交算子 J 立体声强度耦合 (<u, Ju> ≡ 0)", "codingStrategyDetail": "依据人耳耳蜗临界频带掩蔽阈值筛除被主音遮蔽的微弱泛音，9.8kHz 以上采用 J-正交无相消强度立体声", "architectureSummary": "48kHz 心理声学透明档 · ERB 听觉掩蔽曲线 + >9.8kHz 反对称 J-正交声场耦合 · 人耳盲听透明 · 0.38 MB 内存", "nonZeroModes": 2636398, "waveSnrDb": 18.74, "melSnrDb": 25.09, "maxDiffVsV93Master": null, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 6092315, "totalModelAndScoreGzipBytes": 2916084, "dspEngineBytes": 24312, "dspEngineGzipBytes": 6500, "totalAudioCorePlainJsBytes": 6116627, "totalAudioCoreGzipBytes": 2923119, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.3_透明级_PerceptualPure固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.3_透明级_PerceptualPure.html", "standaloneHtmlBytes": 6175645};

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
