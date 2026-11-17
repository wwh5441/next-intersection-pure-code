import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import {
  CWC_FRAME_N,
  CWC_HOP_M,
  createMDCTProcessor,
  packQuantizedVec,
  unpackQuantizedVec,
} from '../src/dsp/mdct-core.js';
import { CWCStreamDecoder } from '../src/dsp/cwc-stream-decoder.js';
import { CWC_LYRICS } from '../data/cwc-lyrics.js';
import { CWC_TRACK_META } from '../data/cwc-models.js';
import * as chunk2 from '../data/cwc-chunk-2.js';

test('MDCT-TDAC forward/inverse roundtrip preserves > 145 dB SNR', () => {
  const mdct = createMDCTProcessor(CWC_FRAME_N);
  const len = 8192;
  const src = new Float32Array(len);
  for (let n = 0; n < len; n++) {
    src[n] =
      0.45 * Math.sin((2 * Math.PI * 220 * n) / 44100) +
      0.28 * Math.sin((2 * Math.PI * 1340 * n) / 44100) +
      0.15 * Math.cos((2 * Math.PI * 5800 * n) / 44100);
  }
  const spec = new Float64Array(512);
  const outN = new Float64Array(CWC_FRAME_N);
  const rec = new Float32Array(len);
  const frames = Math.floor((len - CWC_FRAME_N) / CWC_HOP_M);
  for (let f = 0; f < frames; f++) {
    mdct.forward(src, f * CWC_HOP_M, spec);
    mdct.inverse(spec, outN);
    for (let n = 0; n < CWC_FRAME_N; n++) rec[f * CWC_HOP_M + n] += outN[n];
  }
  let sigP = 0;
  let errP = 0;
  for (let i = CWC_HOP_M; i < frames * CWC_HOP_M; i++) {
    sigP += src[i] * src[i];
    const d = src[i] - rec[i];
    errP += d * d;
  }
  const snr = 10 * Math.log10(sigP / Math.max(1e-18, errP));
  assert.ok(snr > 145, `Expected MDCT SNR > 145 dB, got ${snr.toFixed(2)} dB`);
});

test('packQuantizedVec / unpackQuantizedVec roundtrips full [-93..+93] range cleanly', () => {
  const vec = new Int32Array(240);
  for (let k = -93; k <= 93; k++) {
    vec[k + 93] = k;
  }
  vec[210] = 25;
  vec[239] = -60;
  const packed = packQuantizedVec(vec);
  assert.ok(!packed.includes('/') && !packed.includes('|') && !packed.includes("'") && !packed.includes('\\'));
  const unpacked = new Int32Array(240);
  unpackQuantizedVec(packed, 0, packed.length, unpacked);
  assert.deepEqual(Array.from(unpacked), Array.from(vec));
});

test('CWCStreamDecoder produces bit-identical audio across sequential vs pre-roll seek playback', () => {
  const frames = chunk2.CWC_PACKED_FRAMES.split('|');
  const decSeq = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  const decSeek = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  decSeq.ingestChunk(chunk2.START_FRAME, frames);
  decSeek.ingestChunk(chunk2.START_FRAME, frames);

  const tmpL = new Float32Array(CWC_HOP_M);
  const tmpR = new Float32Array(CWC_HOP_M);
  const targetFrame = chunk2.START_FRAME + 200;

  // Sequential decode from chunk start up to targetFrame
  for (let f = chunk2.START_FRAME; f <= targetFrame; f++) {
    decSeq.decodeFrame(f, tmpL, tmpR, 0);
  }

  // Direct seek with 2-frame pre-roll
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
});

test('CWCStreamDecoder 5-layer stem Mute/Solo isolation and linear summation hold', () => {
  const frames = chunk2.CWC_PACKED_FRAMES.split('|');
  const dec = new CWCStreamDecoder(CWC_TRACK_META.totalFrames);
  dec.ingestChunk(chunk2.START_FRAME, frames);

  const frameIdx = chunk2.START_FRAME + 300;
  const fullL = new Float32Array(CWC_HOP_M);
  const fullR = new Float32Array(CWC_HOP_M);
  dec.seekToFrame(frameIdx);
  dec.decodeFrame(frameIdx, fullL, fullR, 0);

  // Mute all -> exact zero
  const mutedL = new Float32Array(CWC_HOP_M);
  const mutedR = new Float32Array(CWC_HOP_M);
  dec.setStemControls({ mutes: [true, true, true, true, true], solos: [false, false, false, false, false] });
  dec.seekToFrame(frameIdx);
  dec.decodeFrame(frameIdx, mutedL, mutedR, 0);
  for (let i = 0; i < CWC_HOP_M; i++) {
    assert.ok(mutedL[i] === 0, 'Muted left sample must be 0');
    assert.ok(mutedR[i] === 0, 'Muted right sample must be 0');
  }

  // Solo L05 (Stereo Side) -> L == -R!
  const sideL = new Float32Array(CWC_HOP_M);
  const sideR = new Float32Array(CWC_HOP_M);
  dec.setStemControls({ mutes: [false, false, false, false, false], solos: [false, false, false, false, true] });
  dec.seekToFrame(frameIdx);
  dec.decodeFrame(frameIdx, sideL, sideR, 0);
  let sideEnergy = 0;
  for (let i = 0; i < CWC_HOP_M; i++) {
    sideEnergy += sideL[i] * sideL[i];
    assert.ok(Math.abs(sideL[i] + sideR[i]) < 1e-6, 'Solo Side channel must have L == -R');
  }
  assert.ok(sideEnergy > 1e-4, 'Expected non-zero stereo side energy in chorus');
});

test('Compiled CWC metrics and timed lyrics satisfy fidelity and size thresholds', () => {
  const metricsPath = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../data/cwc-metrics.json');
  const m = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
  assert.ok(m.avg24BarkSnrDb >= 30, `24-Bark SNR ${m.avg24BarkSnrDb} dB < 30 dB`);
  assert.ok(m.snrLeftDb >= 14.5 && m.snrRightDb >= 14.5, 'L/R SNR below threshold');
  assert.ok(m.cwcGzipBytes < m.mp3Bytes, 'CWC gzipped size must be smaller than original MP3');
  assert.ok(m.realtimeSpeedupX >= 40, `Expected >=40x realtime decode speedup, got ${m.realtimeSpeedupX}x`);
  assert.ok(CWC_LYRICS.length >= 25, 'Expected full timed LRC lyrics extracted from MP3');
});
