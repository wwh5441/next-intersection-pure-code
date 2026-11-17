import '../vendor/lzma-d-min.js';
import { unpackWavePacket } from '../dsp/storage.js';
import { decodeBase122 } from './base122.js';

function decodePacketSync(buffer) {
  const decoded = globalThis.LZMA.decompress(new Uint8Array(buffer));
  if (typeof decoded === 'string') {
    throw new Error('Unexpected string output while decoding binary VWP6 packet');
  }
  return unpackWavePacket(Uint8Array.from(decoded));
}

async function fetchCompressedBuffer(combo, profile, fileName) {
  const jsName = fileName.replace(/\.vwp$/, '.js');
  const mod = await import(`../../data/${combo}/${profile}/${jsName}`);
  return decodeBase122(mod.default).buffer;
}

function decodeViaWorker(workerUrl, id, buffer) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const worker = new Worker(workerUrl, { type: 'module' });
    const finish = (fn, val) => {
      if (settled) return;
      settled = true;
      worker.terminate();
      fn(val);
    };
    worker.onmessage = ({ data }) => {
      if (data.error) finish(reject, new Error(data.error));
      else finish(resolve, data.buffer);
    };
    worker.onerror = (event) => {
      event.preventDefault?.();
      finish(reject, new Error(event.message || 'Vocal wave worker decode failed'));
    };
    worker.postMessage({ id, buffer }, [buffer]);
  });
}

export async function loadVocalWaveStreams(manifest, combo, profile, onProgress) {
  const workerUrl = new URL('./wave-worker.js', import.meta.url);
  const allFiles = [
    ...new Set(manifest.tracks.flatMap((t) => t.streams.map((s) => s.file))),
  ];
  let completed = 0;

  const compressedItems = await Promise.all(
    allFiles.map(async (fileName, id) => ({
      id,
      fileName,
      buffer: await fetchCompressedBuffer(combo, profile, fileName),
    }))
  );

  const concurrency = Math.min(4, compressedItems.length);
  const results = new Array(compressedItems.length);
  let cursor = 0;

  async function runSlot() {
    while (cursor < compressedItems.length) {
      const idx = cursor++;
      const { id, fileName, buffer } = compressedItems[idx];
      const backupCopy = buffer.slice(0);
      let unpacked;
      try {
        unpacked = await decodeViaWorker(workerUrl, id, buffer);
      } catch {
        await new Promise((r) => setTimeout(r, 0));
        unpacked = decodePacketSync(backupCopy);
      }
      results[idx] = [fileName, unpacked];
      onProgress?.(++completed, allFiles.length);
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => runSlot()));
  return Object.fromEntries(results);
}
