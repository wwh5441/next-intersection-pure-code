// Standalone Entry Point for V9.4.2【监听级 Reference Hi-Fi】Pure-Math Engine
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

const INLINE_META = {"tierId": "v94_2", "versionCode": "V9.4.2", "badgeIndex": "②", "tierName": "监听级 Reference Hi-Fi", "shortTag": "监听音箱全相位保真", "targetDeviceDesc": "旗舰监听音箱 / Hi-Fi 解码耳放", "passbandLabel": "20 Hz – 19.5 kHz", "passbandShort": "19.5kHz", "formulaSummary": "20Hz–19.5kHz 独立 Mid/Side 双声道全相位正交展开 · 仅滤除 -58dBFS 以下不可测底噪尘埃", "codingStrategyDetail": "保留完整极低频下潜与双声道独立高频相位，频谱仪与相位仪指标无伪影，剔除纯量化底噪尘埃", "architectureSummary": "48kHz 硬件性能上限档 · 20Hz–19.5kHz 独立双声道全相位正交余弦级数 · 零强度耦合失真 · 0.38 MB 流式内存", "nonZeroModes": 4704874, "waveSnrDb": 23.89, "melSnrDb": 31.77, "maxDiffVsV93Master": null, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 10078408, "totalModelAndScoreGzipBytes": 4826245, "dspEngineBytes": 24312, "dspEngineGzipBytes": 6500, "totalAudioCorePlainJsBytes": 10102720, "totalAudioCoreGzipBytes": 4832833, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.2_监听级_ReferenceHiFi固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.2_监听级_ReferenceHiFi.html", "standaloneHtmlBytes": 10161746};

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
