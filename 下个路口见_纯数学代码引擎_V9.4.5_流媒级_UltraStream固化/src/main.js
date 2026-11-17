// Standalone Entry Point for V9.4.5【流媒级 Ultra Stream】Pure-Math Engine
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

const INLINE_META = {"tierId": "v94_5", "versionCode": "V9.4.5", "badgeIndex": "⑤", "tierName": "流媒级 Ultra Stream", "shortTag": "纯净流媒 · 零伪造噪声", "targetDeviceDesc": "弱网流媒体 / 纯净立体声音乐直传", "passbandLabel": "85 Hz – 9.5 kHz (纯净真声)", "passbandShort": "85Hz–9.5kHz", "formulaSummary": "85Hz–9.5kHz 真实双声道正交余弦展开 + 瞬态短窗防预回声 + 四阶巴特沃斯抗旁瓣平滑 (零SBR噪声)", "codingStrategyDetail": "100% 源自原曲真实正交系数，彻底摒弃人工伪造高频噪声，配合时频连续性去孤立毛刺与 9.8kHz 巴特沃斯低通", "architectureSummary": "48kHz 纯净流媒级 · 85Hz–9.5kHz 真立体声正交展开 + 四阶巴特沃斯连续平滑 · 零电流底噪 · 亚 MB 级", "nonZeroModes": 393549, "waveSnrDb": 4.74, "melSnrDb": 13.08, "maxDiffVsV93Master": null, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 1267899, "totalModelAndScoreGzipBytes": 606155, "dspEngineBytes": 27140, "dspEngineGzipBytes": 7371, "totalAudioCorePlainJsBytes": 1295039, "totalAudioCoreGzipBytes": 614055, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.5_流媒级_UltraStream固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.5_流媒级_UltraStream.html", "standaloneHtmlBytes": 1354312, "inBandSnrDb": 12.74};

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
