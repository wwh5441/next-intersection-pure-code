// Standalone Entry Point for V9.4.6【语析级 Vocal & ASR】Pure-Math Engine
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

const INLINE_META = {"tierId": "v94_6", "versionCode": "V9.4.6", "badgeIndex": "⑥", "tierName": "语析级 Vocal & ASR", "shortTag": "ASR清晰识词 · 温润人声", "targetDeviceDesc": "Whisper/ASR 识词声纹 + 温润人耳听歌", "passbandLabel": "115 Hz – 6.8 kHz (16kHz语析带)", "passbandShort": "115Hz–6.8kHz", "formulaSummary": "115Hz–6.8kHz 真实语析通带 + 主唱共振峰连续提权 + 瞬态辅音短窗锁定 + 6.8kHz 四阶巴特沃斯平滑", "codingStrategyDetail": "对齐 16kHz ASR 黄金语析通带，真实保留清辅音瞬态并对主唱共振峰精细编码，零合成噪声、零电流毛刺", "architectureSummary": "48kHz 语析与人耳双优级 · 115Hz–6.8kHz 主唱共振峰优先正交展开 + 巴特沃斯抗混叠平滑 · 零电流声", "nonZeroModes": 207730, "waveSnrDb": 2.62, "melSnrDb": 10.95, "maxDiffVsV93Master": null, "pcmRamMb": 0.38, "zeroLzma": true, "zeroBase122": true, "zeroBase64": true, "zeroBinarySamples": true, "totalModelAndScoreBytes": 701717, "totalModelAndScoreGzipBytes": 334585, "dspEngineBytes": 27140, "dspEngineGzipBytes": 7371, "totalAudioCorePlainJsBytes": 728857, "totalAudioCoreGzipBytes": 342257, "frozenDirName": "下个路口见_纯数学代码引擎_V9.4.6_语析级_VocalASR固化", "standaloneHtmlName": "下个路口见_纯数学代码_V9.4.6_语析级_VocalASR.html", "standaloneHtmlBytes": 788112, "inBandSnrDb": 11.57};

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
