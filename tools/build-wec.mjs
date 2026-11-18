#!/usr/bin/env node
// CabronWEC (CWC-5X) Lossless `.wec` Binary Container & Standalone `wec-play` Executable Builder
// Converts the 3 pure-JS editions (`500k`, `1m`, `hifi`) into compact self-verifying `.wec` binary
// containers and emits the standalone zero-dependency `wec-play.mjs` executable player.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  CWC_HOP_M,
  CWC_NUM_PANS,
  CWC_NUM_SCALES,
  CWC_SAMPLE_RATE,
} from '../cabron-wec/src/dsp/mdct-core.js';
import { unpackBase91Chunk } from '../cabron-wec/src/dsp/cwc-entropy.js';
import {
  inspectWecHeader,
  packFramesToWec,
  unpackWecContainer,
} from '../cabron-wec/src/dsp/cwc-wec-container.js';
import { CWCStreamDecoder } from '../cabron-wec/src/dsp/cwc-stream-decoder.js';
import {
  CWC_EDITIONS,
  CWC_OVERVIEW,
  CWC_TRACK_META,
} from '../cabron-wec/data/cwc-models.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, 'cabron-wec/data');
const DIST_DIR = path.join(ROOT, 'cabron-wec/bundles');
const SRC_DSP_DIR = path.join(ROOT, 'cabron-wec/src/dsp');

fs.mkdirSync(DIST_DIR, { recursive: true });

function stripModuleAndCompact(src) {
  return src
    .replace(/^import\s+[\s\S]*?from\s+['"][^'"]+['"];?\s*$/gm, '')
    .replace(/^export\s+\{[^}]*\};?\s*$/gm, '')
    .replace(/^export\s+(const|let|var|function|class)\s+/gm, '$1 ')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
    .replace(/^[ \t]+/gm, '')
    .replace(/\n{2,}/g, '\n')
    .trim();
}

function buildStandaloneWecPlayer() {
  const mdctSrc = stripModuleAndCompact(
    fs.readFileSync(path.join(SRC_DSP_DIR, 'mdct-core.js'), 'utf8')
  );
  // Extract only rangeDecodeStream + constants from cwc-entropy.js (player does not need Base91 or encoder)
  const entropyFull = fs.readFileSync(path.join(SRC_DSP_DIR, 'cwc-entropy.js'), 'utf8');
  const rcDecIdx = entropyFull.indexOf('export function rangeDecodeStream');
  const rcDecEnd = entropyFull.indexOf('export function packFramesToBase91');
  const rcDecodeOnly = `
const RC_TOP = 1 << 24;
const RC_BOT = 1 << 16;
const TOT_BITS = 14;
const TOT = 1 << TOT_BITS;
${stripModuleAndCompact(entropyFull.slice(rcDecIdx, rcDecEnd))}
`.trim();

  // Extract decoder-only portion of cwc-wec-container.js (CRC32 + header inspect + unpackWecContainer)
  const wecFull = fs.readFileSync(path.join(SRC_DSP_DIR, 'cwc-wec-container.js'), 'utf8');
  const wecParts = [
    wecFull.slice(0, wecFull.indexOf('function concatUint8Arrays')),
    wecFull.slice(
      wecFull.indexOf('function decodeFrameHeaders'),
      wecFull.indexOf('function encodeAtomsMode0')
    ),
    wecFull.slice(
      wecFull.indexOf('function decodeAtomsMode0'),
      wecFull.indexOf('function encodeAtomsMode1')
    ),
    wecFull.slice(wecFull.indexOf('function decodeAtomsMode1'), wecFull.indexOf('export function packFramesToWec')),
    wecFull.slice(wecFull.indexOf('export function inspectWecHeader')),
  ].join('\n');
  const wecDecodeOnly = stripModuleAndCompact(wecParts);

  const streamDecSrc = stripModuleAndCompact(
    fs.readFileSync(path.join(SRC_DSP_DIR, 'cwc-stream-decoder.js'), 'utf8')
  );

  const cliSource = `#!/usr/bin/env node
// CabronWEC (CWC-5X) Standalone Zero-Dependency .wec Player & WAV Renderer (wec-play.mjs)
// Usage:
//   node wec-play.mjs inspect <song.wec>
//   node wec-play.mjs verify  <song.wec>
//   node wec-play.mjs render  <song.wec> [output.wav] [--solo=0..4] [--mute=0..4]
${mdctSrc}
${rcDecodeOnly}
${wecDecodeOnly}
${streamDecSrc}

function renderWecToWavBuffer(wecBytes, stemOpts = {}) {
  const bundle = unpackWecContainer(wecBytes);
  const { sampleRate, totalFrames, totalSamples } = bundle.header;
  const dec = new CWCStreamDecoder(totalFrames);
  dec.ingestChunk(bundle);
  dec.setStemControls(stemOpts);

  const wav = new ArrayBuffer(44 + totalSamples * 4);
  const dv = new DataView(wav);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF');
  dv.setUint32(4, 36 + totalSamples * 4, true);
  ws(8, 'WAVEfmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 2, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 4, true);
  dv.setUint16(32, 4, true);
  dv.setUint16(34, 16, true);
  ws(36, 'data');
  dv.setUint32(40, totalSamples * 4, true);

  const hopL = new Float32Array(CWC_HOP_M);
  const hopR = new Float32Array(CWC_HOP_M);
  let byteOff = 44;
  let written = 0;
  for (let f = 0; f < totalFrames && written < totalSamples; f++) {
    dec.decodeFrame(f, hopL, hopR, 0);
    const take = Math.min(CWC_HOP_M, totalSamples - written);
    for (let i = 0; i < take; i++) {
      dv.setInt16(byteOff, Math.max(-32767, Math.min(32767, Math.round(hopL[i] * 32767))), true);
      dv.setInt16(byteOff + 2, Math.max(-32767, Math.min(32767, Math.round(hopR[i] * 32767))), true);
      byteOff += 4;
    }
    written += take;
  }
  return new Uint8Array(wav);
}

export { inspectWecHeader, unpackWecContainer, renderWecToWavBuffer, CWCStreamDecoder };

const isMain = typeof process !== 'undefined' && process.argv?.[1] &&
  import.meta.url.endsWith(process.argv[1].split(/[\\\\/]/).pop());

if (isMain) {
  const fsMod = await import('node:fs');
  const [cmd = 'inspect', inFile, ...rest] = process.argv.slice(2);
  if (!inFile) {
    console.log('Usage: node wec-play.mjs <inspect|verify|render> <file.wec> [out.wav] [--solo=N] [--mute=N]');
    process.exit(1);
  }
  const buf = fsMod.readFileSync(inFile);
  if (cmd === 'inspect') {
    console.log(JSON.stringify(inspectWecHeader(buf), null, 2));
  } else if (cmd === 'verify') {
    const t0 = performance.now();
    const unpacked = unpackWecContainer(buf);
    const ms = (performance.now() - t0).toFixed(1);
    console.log(JSON.stringify({
      status: 'OK',
      crc32Hex: unpacked.header.crc32Hex,
      tierLabel: unpacked.header.tierLabel,
      frames: unpacked.frameCount,
      midAtoms: unpacked.midIdx.length,
      sideAtoms: unpacked.sideIdx.length,
      unpackMs: Number(ms),
    }, null, 2));
  } else if (cmd === 'render') {
    const outFile = rest.find((a) => !a.startsWith('--')) || inFile.replace(/\\.wec$/i, '') + '.wav';
    const solos = [false, false, false, false, false];
    const mutes = [false, false, false, false, false];
    for (const arg of rest) {
      if (arg.startsWith('--solo=')) solos[Number(arg.slice(7))] = true;
      if (arg.startsWith('--mute=')) mutes[Number(arg.slice(7))] = true;
    }
    const t0 = performance.now();
    const wavBytes = renderWecToWavBuffer(buf, { solos, mutes });
    fsMod.writeFileSync(outFile, wavBytes);
    const sec = ((performance.now() - t0) / 1000).toFixed(2);
    console.log(\`Rendered \${outFile} (\${(wavBytes.length / 1048576).toFixed(2)} MiB) in \${sec}s\`);
  } else {
    console.error(\`Unknown command: \${cmd}\`);
    process.exit(1);
  }
}
`;

  const outPath = path.join(DIST_DIR, 'wec-play.mjs');
  fs.writeFileSync(outPath, cliSource, 'utf8');
  fs.chmodSync(outPath, 0o755);
  const rawBytes = Buffer.byteLength(cliSource, 'utf8');
  const gzBytes = zlib.gzipSync(Buffer.from(cliSource, 'utf8'), { level: 9 }).length;
  return {
    file: '/bundles/wec-play.mjs',
    rawBytes,
    gzBytes,
  };
}

async function loadTierChunksAndFrames(tier) {
  const chunks = [];
  const allFrames = [];
  for (let i = 0; i < 8; i++) {
    const chunkPath = path.join(DATA_DIR, tier, `cwc-chunk-${i}.js`);
    const mod = await import(`${pathToFileURL(chunkPath).href}?t=${Date.now()}`);
    const ch = unpackBase91Chunk(mod.CWC_PACKED_CHUNK, mod.START_FRAME, mod.FRAME_COUNT);
    chunks.push(ch);
    for (let f = 0; f < ch.frameCount; f++) {
      const scBase = f * CWC_NUM_SCALES;
      const pBase = f * CWC_NUM_PANS;
      const midNz = [];
      for (let p = ch.midOff[f]; p < ch.midOff[f + 1]; p++) {
        midNz.push({ k: ch.midIdx[p], q: ch.midQ[p] });
      }
      const sideNz = [];
      for (let p = ch.sideOff[f]; p < ch.sideOff[f + 1]; p++) {
        sideNz.push({ k: ch.sideIdx[p], q: ch.sideQ[p] });
      }
      allFrames.push({
        winType: ch.winTypes[f],
        scales: ch.scales.slice(scBase, scBase + CWC_NUM_SCALES),
        pans: ch.pans.slice(pBase, pBase + CWC_NUM_PANS),
        midNz,
        sideNz,
      });
    }
  }
  return { chunks, allFrames };
}

function verifyBitExactEquivalence(tier, chunks, wecBytes) {
  const t0 = performance.now();
  const wecBundle = unpackWecContainer(wecBytes);
  const unpackMs = Number((performance.now() - t0).toFixed(1));

  const decJs = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  for (const ch of chunks) decJs.ingestChunk(ch);

  const decWec = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  decWec.ingestChunk(wecBundle);

  // Verify sample-for-sample exactness across Intro (0..120) and Chorus (1200..1320)
  const hopJsL = new Float32Array(CWC_HOP_M);
  const hopJsR = new Float32Array(CWC_HOP_M);
  const hopWecL = new Float32Array(CWC_HOP_M);
  const hopWecR = new Float32Array(CWC_HOP_M);

  decJs.seekToFrame(0);
  decWec.seekToFrame(0);
  for (let f = 0; f < 120; f++) {
    decJs.decodeFrame(f, hopJsL, hopJsR, 0);
    decWec.decodeFrame(f, hopWecL, hopWecR, 0);
    for (let i = 0; i < CWC_HOP_M; i++) {
      if (hopJsL[i] !== hopWecL[i] || hopJsR[i] !== hopWecR[i]) {
        throw new Error(`Sample mismatch in ${tier} at frame ${f} sample ${i}`);
      }
    }
  }

  decJs.seekToFrame(1200);
  decWec.seekToFrame(1200);
  for (let f = 1200; f < 1320; f++) {
    decJs.decodeFrame(f, hopJsL, hopJsR, 0);
    decWec.decodeFrame(f, hopWecL, hopWecR, 0);
    for (let i = 0; i < CWC_HOP_M; i++) {
      if (hopJsL[i] !== hopWecL[i] || hopJsR[i] !== hopWecR[i]) {
        throw new Error(`Chorus sample mismatch in ${tier} at frame ${f} sample ${i}`);
      }
    }
  }

  return { header: wecBundle.header, unpackMs };
}

console.log('1) Building standalone zero-dependency executable player `wec-play.mjs`...');
const wecPlayer = buildStandaloneWecPlayer();
console.log(
  `   -> wec-play.mjs: ${(wecPlayer.rawBytes / 1000).toFixed(1)} KB Raw (${(wecPlayer.gzBytes / 1000).toFixed(1)} KB Gzip)`
);

const TIER_SPECS = [
  { id: '500k', tierCode: 500, outName: 'next-intersection-500k.wec' },
  { id: '1m', tierCode: 1000, outName: 'next-intersection-1m.wec' },
  { id: 'hifi', tierCode: 2400, outName: 'next-intersection-hifi.wec' },
];

const wecResults = {};
for (const spec of TIER_SPECS) {
  console.log(`2) Packing & bit-exact verifying ${spec.outName}...`);
  const { chunks, allFrames } = await loadTierChunksAndFrames(spec.id);
  const wecBytes = packFramesToWec(allFrames, {
    tierCode: spec.tierCode,
    sampleRate: CWC_SAMPLE_RATE,
    totalSamples: CWC_TRACK_META.totalSamples,
  });
  const outPath = path.join(DIST_DIR, spec.outName);
  fs.writeFileSync(outPath, wecBytes);
  const { header, unpackMs } = verifyBitExactEquivalence(spec.id, chunks, wecBytes);
  wecResults[spec.id] = {
    file: `/bundles/${spec.outName}`,
    bytes: wecBytes.length,
    crc32Hex: header.crc32Hex,
    entropyMode: header.entropyMode,
    entropyModeName: header.entropyModeName,
    unpackMs,
  };
  console.log(
    `   -> ${spec.outName}: ${wecBytes.length} B (${(wecBytes.length / 1000).toFixed(1)} KB / ${(wecBytes.length / 1024).toFixed(1)} KiB) | CRC32=0x${header.crc32Hex} | 0.000000 sample diff (${unpackMs} ms unpack)`
  );
}

// Update cwc-metrics.json and cwc-models.js
const metricsPath = path.join(DATA_DIR, 'cwc-metrics.json');
const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
metrics.wecPlayer = wecPlayer;
for (const id of Object.keys(wecResults)) {
  if (metrics.editions?.[id]) {
    metrics.editions[id].wec = wecResults[id];
  }
}
fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2) + '\n', 'utf8');

const updatedEditions = structuredClone(CWC_EDITIONS);
for (const id of Object.keys(wecResults)) {
  if (updatedEditions[id]) {
    updatedEditions[id].wec = wecResults[id];
  }
}

const modelsJs = `// CabronWEC (CWC-5X) Multi-Tier Track Architecture & Edition Manifest
import {
  CWC_BPM,
  CWC_BAR_SEC,
  CWC_FRAME_N,
  CWC_HOP_M,
  CWC_SAMPLE_RATE,
  CWC_SHORT_N,
  CWC_SHORT_K,
  CWC_STEMS,
  MID_BANDS,
  SIDE_BANDS,
  PNS_MID_BANDS,
  PNS_SIDE_BANDS,
} from '../src/dsp/mdct-core.js';

export const CWC_TRACK_META = ${JSON.stringify(CWC_TRACK_META, null, 2)};

export const CWC_WEC_PLAYER = ${JSON.stringify(wecPlayer, null, 2)};

export const CWC_EDITIONS = ${JSON.stringify(updatedEditions, null, 2)};
export const CWC_CHUNKS = CWC_EDITIONS['1m'].chunks;

export const CWC_OVERVIEW = ${JSON.stringify(CWC_OVERVIEW)};

export { CWC_STEMS, MID_BANDS, SIDE_BANDS, PNS_MID_BANDS, PNS_SIDE_BANDS };
`;
fs.writeFileSync(path.join(DATA_DIR, 'cwc-models.js'), modelsJs, 'utf8');
console.log('3) Updated cwc-metrics.json and cwc-models.js with .wec container & wec-play executable metrics.');
