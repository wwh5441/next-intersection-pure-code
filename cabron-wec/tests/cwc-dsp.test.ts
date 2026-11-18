import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {
  CWC_FRAME_N,
  CWC_HOP_M,
  CWC_NUM_PANS,
  CWC_NUM_SCALES,
  WIN_LONG,
  WIN_SHORT,
  WIN_START,
  WIN_STOP,
  createAdaptiveMDCTProcessor,
} from '../src/dsp/mdct-core.js';
import {
  decodeBase91,
  encodeBase91,
  packFramesToBase91,
  rangeDecodeStream,
  rangeEncodeStream,
  unpackBase91Chunk,
} from '../src/dsp/cwc-entropy.js';
import { inspectWecHeader, unpackWecContainer } from '../src/dsp/cwc-wec-container.js';
import {
  CWCStreamDecoder,
  computeShortSubblockEnv,
  synthesizeHarmonicSbrBand,
} from '../src/dsp/cwc-stream-decoder.js';
import { CWC_LYRICS } from '../data/cwc-lyrics.js';
import { CWC_TRACK_META } from '../data/cwc-models.js';
import * as chunk1m2 from '../data/1m/cwc-chunk-2.js';
import * as chunk500k2 from '../data/500k/cwc-chunk-2.js';

test('Adaptive Edler MDCT-TDAC (LONG / START / 8x512 SHORT / STOP) preserves > 145 dB SNR', () => {
  const mdct = createAdaptiveMDCTProcessor(CWC_FRAME_N);
  const frames = 24;
  const len = (frames + 2) * CWC_HOP_M;
  const src = new Float32Array(len);
  for (let n = 0; n < len; n++) {
    src[n] =
      0.45 * Math.sin((2 * Math.PI * 220 * n) / 44100) +
      0.28 * Math.sin((2 * Math.PI * 1340 * n) / 44100) +
      0.15 * Math.cos((2 * Math.PI * 5800 * n) / 44100);
  }

  const seq = [
    WIN_LONG, WIN_LONG, WIN_START, WIN_SHORT, WIN_SHORT, WIN_STOP,
    WIN_LONG, WIN_START, WIN_STOP, WIN_LONG, WIN_LONG, WIN_START,
    WIN_SHORT, WIN_STOP, WIN_LONG, WIN_LONG, WIN_START, WIN_SHORT,
    WIN_SHORT, WIN_SHORT, WIN_STOP, WIN_LONG, WIN_LONG, WIN_LONG,
  ];

  const spec = new Float64Array(CWC_HOP_M);
  const outN = new Float64Array(CWC_FRAME_N);
  const ola = new Float64Array(CWC_HOP_M);
  const rec = new Float32Array(frames * CWC_HOP_M);
  const shift = CWC_HOP_M >> 1;

  for (let f = 0; f < frames; f++) {
    mdct.forward(src, f * CWC_HOP_M - shift, seq[f], spec);
    mdct.inverse(spec, seq[f], outN);
    for (let n = 0; n < CWC_HOP_M; n++) {
      rec[f * CWC_HOP_M + n] = ola[n] + outN[n];
      ola[n] = outN[CWC_HOP_M + n];
    }
  }

  let sigP = 0;
  let errP = 0;
  for (let i = CWC_FRAME_N; i < (frames - 2) * CWC_HOP_M; i++) {
    const ref = src[i - shift];
    sigP += ref * ref;
    const d = ref - rec[i];
    errP += d * d;
  }
  const snr = 10 * Math.log10(sigP / Math.max(1e-18, errP));
  assert.ok(snr > 145, `Expected Adaptive Edler MDCT SNR > 145 dB, got ${snr.toFixed(2)} dB`);
});

test('Pure-JS Range Coder + Base91 + CSR Chunk Packer roundtrips frame structures bit-exactly', () => {
  const sampleBytes = new Uint8Array(4096);
  for (let i = 0; i < sampleBytes.length; i++) {
    sampleBytes[i] = (i * 73 + (i >>> 3)) & 255;
  }
  const b91 = encodeBase91(sampleBytes);
  assert.ok(!b91.includes("'") && !b91.includes('\\') && !b91.includes('"'));
  assert.deepEqual(Array.from(decodeBase91(b91)), Array.from(sampleBytes));

  const skewed = new Uint8Array(8000);
  for (let i = 0; i < skewed.length; i++) skewed[i] = i % 11 === 0 ? 42 : i & 7;
  const rc = rangeEncodeStream(skewed, 64);
  assert.ok(rc.length < skewed.length * 0.55, 'Expected Range Coder compression on skewed stream');
  assert.deepEqual(Array.from(rangeDecodeStream(rc, 0).data), Array.from(skewed));

  const synthScales = new Uint8Array(CWC_NUM_SCALES);
  synthScales.set([90, 80, 0, 0, 75, 60, 50, 40, 0, 35, 110, 95, 65, 0, 45, 0, 88, 62, 54, 0, 48, 70, 55]);
  const synthFrames = [
    {
      winType: WIN_LONG,
      scales: synthScales,
      pans: new Int8Array([-6, 0, 8, -3, 4]),
      midNz: [
        { k: 0, q: 31 },
        { k: 1, q: -18 },
        { k: 23, q: 5 },
        { k: 25, q: -12 },
        { k: 160, q: 19 },
        { k: 220, q: -7 },
        { k: 310, q: 4 },
        { k: 450, q: -3 },
        { k: 1010, q: 2 },
      ],
      sideNz: [
        { k: 6, q: -14 },
        { k: 42, q: 9 },
        { k: 200, q: -4 },
      ],
    },
    {
      winType: WIN_START,
      scales: new Uint8Array(CWC_NUM_SCALES),
      pans: new Int8Array(CWC_NUM_PANS),
      midNz: [],
      sideNz: [],
    },
  ];

  const packed = packFramesToBase91(synthFrames);
  const bundle = unpackBase91Chunk(packed, 10, synthFrames.length);
  assert.equal(bundle.startFrame, 10);
  assert.equal(bundle.frameCount, 2);
  assert.equal(bundle.winTypes[0], WIN_LONG);
  assert.equal(bundle.winTypes[1], WIN_START);
  assert.deepEqual(Array.from(bundle.scales.subarray(0, CWC_NUM_SCALES)), Array.from(synthFrames[0].scales));
  assert.deepEqual(Array.from(bundle.pans.subarray(0, 5)), Array.from(synthFrames[0].pans));
  assert.deepEqual(Array.from(bundle.midIdx), synthFrames[0].midNz.map((x) => x.k));
  assert.deepEqual(Array.from(bundle.midQ), synthFrames[0].midNz.map((x) => x.q));
  assert.deepEqual(Array.from(bundle.sideIdx), synthFrames[0].sideNz.map((x) => x.k));
  assert.deepEqual(Array.from(bundle.sideQ), synthFrames[0].sideNz.map((x) => x.q));
});

test('CWCStreamDecoder produces bit-identical audio across sequential vs pre-roll seek on both <1MB and <500KB tiers', () => {
  for (const chMod of [chunk1m2, chunk500k2]) {
    const bundle = unpackBase91Chunk(chMod.CWC_PACKED_CHUNK, chMod.START_FRAME, chMod.FRAME_COUNT);
    const decSeq = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
    const decSeek = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
    decSeq.ingestChunk(bundle);
    decSeek.ingestChunk(bundle);

    const tmpL = new Float32Array(CWC_HOP_M);
    const tmpR = new Float32Array(CWC_HOP_M);
    const targetFrame = chMod.START_FRAME + 80;

    for (let f = chMod.START_FRAME; f <= targetFrame; f++) {
      decSeq.decodeFrame(f, tmpL, tmpR, 0);
    }

    const seekL = new Float32Array(CWC_HOP_M);
    const seekR = new Float32Array(CWC_HOP_M);
    decSeek.seekToFrame(targetFrame);
    decSeek.decodeFrame(targetFrame, seekL, seekR, 0);

    let maxDiff = 0;
    for (let i = 0; i < CWC_HOP_M; i++) {
      assert.ok(Number.isFinite(seekL[i]) && Number.isFinite(seekR[i]), 'Non-finite audio sample');
      assert.ok(Math.abs(seekL[i]) <= 1.0 && Math.abs(seekR[i]) <= 1.0, 'Sample exceeded ceiling');
      maxDiff = Math.max(maxDiff, Math.abs(tmpL[i] - seekL[i]), Math.abs(tmpR[i] - seekR[i]));
    }
    assert.ok(maxDiff < 1e-6, `Expected seek pre-roll to match sequential decode (<1e-6), got ${maxDiff}`);
  }
});

test('CWCStreamDecoder 5-layer stem Mute/Solo isolation and exact linear summation hold with Harmonic SBR active', () => {
  const bundle = unpackBase91Chunk(chunk500k2.CWC_PACKED_CHUNK, chunk500k2.START_FRAME, chunk500k2.FRAME_COUNT);
  const dec = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  dec.ingestChunk(bundle);
  dec.masterGain = 0.5;

  const frameIdx = chunk500k2.START_FRAME + 120;
  const fullL = new Float32Array(CWC_HOP_M);
  const fullR = new Float32Array(CWC_HOP_M);
  dec.seekToFrame(frameIdx);
  dec.decodeFrame(frameIdx, fullL, fullR, 0);

  const mutedL = new Float32Array(CWC_HOP_M);
  const mutedR = new Float32Array(CWC_HOP_M);
  dec.setStemControls({ mutes: [true, true, true, true, true], solos: [false, false, false, false, false] });
  dec.seekToFrame(frameIdx);
  dec.decodeFrame(frameIdx, mutedL, mutedR, 0);
  for (let i = 0; i < CWC_HOP_M; i++) {
    assert.ok(Math.abs(mutedL[i]) === 0, 'Muted left sample must be 0');
    assert.ok(Math.abs(mutedR[i]) === 0, 'Muted right sample must be 0');
  }

  const sumL = new Float32Array(CWC_HOP_M);
  const sumR = new Float32Array(CWC_HOP_M);
  const stemL = new Float32Array(CWC_HOP_M);
  const stemR = new Float32Array(CWC_HOP_M);
  let sideEnergy = 0;

  for (let s = 0; s < 5; s++) {
    const solos = [false, false, false, false, false];
    solos[s] = true;
    dec.setStemControls({ mutes: [false, false, false, false, false], solos });
    dec.seekToFrame(frameIdx);
    dec.decodeFrame(frameIdx, stemL, stemR, 0);
    for (let i = 0; i < CWC_HOP_M; i++) {
      sumL[i] += stemL[i];
      sumR[i] += stemR[i];
      if (s === 4) {
        sideEnergy += stemL[i] * stemL[i];
        assert.ok(Math.abs(stemL[i] + stemR[i]) < 1e-6, 'Solo Side channel must satisfy L == -R');
      }
    }
  }
  assert.ok(sideEnergy > 1e-4, 'Expected non-zero stereo side energy in chorus');

  let maxLinErr = 0;
  for (let i = 0; i < CWC_HOP_M; i++) {
    maxLinErr = Math.max(maxLinErr, Math.abs(fullL[i] - sumL[i]), Math.abs(fullR[i] - sumR[i]));
  }
  assert.ok(maxLinErr < 1e-5, `Expected 5-layer stem sum == Full Mix (<1e-5), got ${maxLinErr}`);
});

test('Verified <1MB, <500KB, and HiFi pure-code editions satisfy all size and fidelity budgets', async () => {
  const baseDir = path.dirname(new URL(import.meta.url).pathname);
  const m = JSON.parse(fs.readFileSync(path.resolve(baseDir, '../data/cwc-metrics.json'), 'utf8'));

  // 1. Sub-1MB Edition: strictly < 1,000,000 Bytes Raw (both chunks AND standalone single-file w/ DSP!)
  const ed1m = m.editions['1m'];
  assert.ok(
    ed1m.cwcRawBytes < 1_000_000,
    `Expected <1MB chunks Raw < 1,000,000 B, got ${ed1m.cwcRawBytes}`
  );
  assert.ok(
    ed1m.standalone.rawBytes < 1_000_000,
    `Expected <1MB standalone single-file Raw < 1,000,000 B, got ${ed1m.standalone.rawBytes}`
  );

  // 2. Sub-500KB Edition: strictly < 500,000 Bytes Raw (both chunks AND standalone single-file w/ DSP!)
  const ed500 = m.editions['500k'];
  assert.ok(
    ed500.cwcRawBytes < 500_000,
    `Expected <500KB chunks Raw < 500,000 B, got ${ed500.cwcRawBytes}`
  );
  assert.ok(
    ed500.standalone.rawBytes < 500_000,
    `Expected <500KB standalone single-file Raw < 500,000 B, got ${ed500.standalone.rawBytes}`
  );

  // 3. HiFi Edition: strictly beats CWC-4X SNR & 24-Bark SNR at ~2.9x smaller Raw size
  const edHi = m.editions.hifi;
  assert.ok(
    edHi.snrLeftDb >= 15.7 && edHi.snrRightDb >= 16.12,
    `Expected HiFi L/R SNR >= CWC-4X (15.70 / 16.12 dB), got ${edHi.snrLeftDb} / ${edHi.snrRightDb} dB`
  );
  assert.ok(
    edHi.avg24BarkSnrDb >= 32.01,
    `Expected HiFi 24-Bark SNR >= CWC-4X (32.01 dB), got ${edHi.avg24BarkSnrDb} dB`
  );
  assert.ok(edHi.cwcRawBytes < m.legacyCwc4xRawBytes * 0.4);

  // 4. Verify standalone <500KB single-file bundle executes cleanly with zero imports
  const standaloneMod = await import('../bundles/next-intersection-500k.js');
  const dec = standaloneMod.createStandaloneDecoder();
  const hL = new Float32Array(CWC_HOP_M);
  const hR = new Float32Array(CWC_HOP_M);
  dec.seekToFrame(1250);
  dec.decodeFrame(1250, hL, hR, 0);
  let rms = 0;
  for (let i = 0; i < CWC_HOP_M; i++) rms += hL[i] * hL[i];
  assert.ok(rms > 0.01, 'Standalone <500KB bundle must synthesize audible chorus signal');
  assert.ok(CWC_LYRICS.length >= 25);
});

test('Lossless "CWC5" .wec binary containers & standalone wec-play.mjs verify CRC-32 and 0.000000 sample diff', () => {
  const baseDir = path.dirname(new URL(import.meta.url).pathname);
  const wec500Path = path.resolve(baseDir, '../bundles/next-intersection-500k.wec');
  const wec1mPath = path.resolve(baseDir, '../bundles/next-intersection-1m.wec');
  const wecHifiPath = path.resolve(baseDir, '../bundles/next-intersection-hifi.wec');
  const playerPath = path.resolve(baseDir, '../bundles/wec-play.mjs');

  const buf500 = fs.readFileSync(wec500Path);
  const buf1m = fs.readFileSync(wec1mPath);
  const bufHifi = fs.readFileSync(wecHifiPath);
  const playerBytes = fs.statSync(playerPath).size;

  assert.ok(buf500.length < 340_000, `Expected 500k.wec < 340 KB, got ${buf500.length}`);
  assert.ok(buf1m.length < 750_000, `Expected 1m.wec < 750 KB, got ${buf1m.length}`);
  assert.ok(bufHifi.length < 2_000_000, `Expected hifi.wec < 2.00 MB, got ${bufHifi.length}`);
  assert.ok(playerBytes < 40_000, `Expected wec-play.mjs < 40 KB, got ${playerBytes}`);

  const hdr = inspectWecHeader(buf500);
  assert.equal(hdr.magic, 'CWC5');
  assert.equal(hdr.totalFrames, CWC_TRACK_META.totalFrames);

  // Tampering a single bit in payload must fail CRC-32 verification
  const tampered = new Uint8Array(buf500);
  tampered[64] ^= 0x01;
  assert.throws(() => unpackWecContainer(tampered), /CRC-32 mismatch/);

  // Bit-exact waveform equivalence between 500k.wec and pure-JS chunk-2 across Chorus
  const wecBundle = unpackWecContainer(buf500);
  const jsBundle = unpackBase91Chunk(chunk500k2.CWC_PACKED_CHUNK, chunk500k2.START_FRAME, chunk500k2.FRAME_COUNT);
  const decWec = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  const decJs = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  decWec.ingestChunk(wecBundle);
  decJs.ingestChunk(jsBundle);

  const wL = new Float32Array(CWC_HOP_M);
  const wR = new Float32Array(CWC_HOP_M);
  const jL = new Float32Array(CWC_HOP_M);
  const jR = new Float32Array(CWC_HOP_M);
  const testFrame = chunk500k2.START_FRAME + 60;
  decWec.seekToFrame(testFrame);
  decJs.seekToFrame(testFrame);
  for (let f = testFrame; f < testFrame + 8; f++) {
    decWec.decodeFrame(f, wL, wR, 0);
    decJs.decodeFrame(f, jL, jR, 0);
    for (let i = 0; i < CWC_HOP_M; i++) {
      assert.equal(wL[i], jL[i], `Sample mismatch L at frame ${f}:${i}`);
      assert.equal(wR[i], jR[i], `Sample mismatch R at frame ${f}:${i}`);
    }
  }

  // Standalone CLI executable `wec-play.mjs` inspect & verify
  const m = JSON.parse(fs.readFileSync(path.resolve(baseDir, '../data/cwc-metrics.json'), 'utf8'));
  const verifyOut = JSON.parse(
    execFileSync(process.execPath, [playerPath, 'verify', wec500Path], {
      encoding: 'utf8',
    })
  );
  assert.equal(verifyOut.status, 'OK');
  assert.equal(verifyOut.crc32Hex, m.editions['500k'].wec.crc32Hex);
  assert.equal(verifyOut.frames, CWC_TRACK_META.totalFrames);
});

test('Coherent L04 SBR extension suppresses white-noise hiss on tonal frames and preserves WIN_SHORT sub-block gating', () => {
  // 1. Tonal frame (high crest >= 2.6): must synthesize 100% deterministically without injecting random noise
  const rawA = new Float64Array(CWC_HOP_M);
  const rawB = new Float64Array(CWC_HOP_M);
  rawA[260] = 0.8;
  rawA[320] = -0.5;
  rawB.set(rawA);

  const subEnv = new Float64Array(8).fill(1);
  const scratch = new Float64Array(640);
  const bd = { lo: 409, hi: 594 };
  synthesizeHarmonicSbrBand(rawA, bd, 130, 0, WIN_LONG, 409, subEnv, () => 999, scratch);
  synthesizeHarmonicSbrBand(rawB, bd, 130, 0, WIN_LONG, 409, subEnv, () => -999, scratch);

  let maxNoiseDiff = 0;
  let nzCount = 0;
  for (let k = bd.lo; k < bd.hi; k++) {
    maxNoiseDiff = Math.max(maxNoiseDiff, Math.abs(rawA[k] - rawB[k]));
    if (Math.abs(rawA[k]) > 1e-6) nzCount++;
  }
  assert.equal(maxNoiseDiff, 0, 'Tonal high-crest SBR frame must inject 0 random white noise');
  assert.equal(nzCount, 2, '1:1 spectral translation must map 2 source tones to 2 clean tones without bin duplication');

  // 2. WIN_SHORT transient gating: energy confined to sub-block s=6 must keep s=0..4 strongly attenuated
  const shortSpec = new Float64Array(CWC_HOP_M);
  for (let j = 12; j < 60; j++) shortSpec[(j << 3) | 6] = 0.5;
  computeShortSubblockEnv(shortSpec, WIN_SHORT, subEnv);
  assert.ok(subEnv[6] > subEnv[0] * 4, `Expected transient sub-block 6 > 4x sub-block 0, got ${subEnv[6]} vs ${subEnv[0]}`);
});
