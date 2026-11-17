// Standalone Entry Point for V9.4.1【原母带 Studio Master】Pure-Math Engine
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

const INLINE_META = {"tierId": "v94_1", "versionCode": "V9.4.1", "badgeIndex": "①", "tierName": "原母带 Studio Master", "shortTag": "较V9.3 0.0dB零误差", "targetDeviceDesc": "母带存档 / 与 V9.3 逐位等价", "passbandLabel": "20 Hz – 20.2 kHz", "passbandShort": "20.2kHz", "formulaSummary": "max |V9.4.1(n) - V9.3(n)| ≡ 0.0 · 798万正交余弦模态 + 51-Bit 规范霍夫曼无损代数打包", "codingStrategyDetail": "100% 原样保留 V9.3 全部 7,984,691 个双声道正交模态，四分类无符号四元组 + 51-bit 十进制安全整数无损编码", "architectureSummary": "48kHz 全双声道正交局部余弦母带 · 相对 V9.3 逐样本 0.000000 dB 零误差 · 51-Bit 十进制安全整数算术流 · 零音频文件", "nonZeroModes": 7984691, "waveSnrDb": 29.81, "melSnrDb": 40.44, "maxDiffVsV93Master": 0.0, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 16057772, "totalModelAndScoreGzipBytes": 7686277, "dspEngineBytes": 24312, "dspEngineGzipBytes": 6500, "totalAudioCorePlainJsBytes": 16082084, "totalAudioCoreGzipBytes": 7693303, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.1_原母带_StudioMaster固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.1_原母带_StudioMaster.html", "standaloneHtmlBytes": 16141086};

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
