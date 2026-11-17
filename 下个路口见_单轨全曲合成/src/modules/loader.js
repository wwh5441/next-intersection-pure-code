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

async function fetchCompressedBuffer(fileName, profile) {
  if (window.__VFS_BINARY_STREAMS) {
    const slice =
      window.__VFS_BINARY_STREAMS[`${profile}/${fileName}`] ||
      window.__VFS_BINARY_STREAMS[fileName];
    if (!slice) throw new Error(`Missing embedded vocal wave stream: ${profile}/${fileName}`);
    return slice.slice(0).buffer;
  }
  const jsName = fileName.replace(/\.vwp$/, '.js');
  const mod = await import(`../../data/${profile}/${jsName}`);
  return decodeBase122(mod.default).buffer;
}

function decodeViaWorker(workerUrl, id, buffer) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const worker = new Worker(
      workerUrl,
      window.__VFS_WORKER_URL ? undefined : { type: 'module' }
    );
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

export async function loadVocalWaveStreams(manifest, profile, onProgress) {
  const workerUrl = window.__VFS_WORKER_URL || new URL('./wave-worker.js', import.meta.url);
  const allFiles = [
    ...new Set(manifest.tracks.flatMap((t) => t.streams.map((s) => s.file))),
  ];
  let completed = 0;

  // Fetch all compressed stream buffers first
  const compressedItems = await Promise.all(
    allFiles.map(async (fileName, id) => ({
      id,
      fileName,
      buffer: await fetchCompressedBuffer(fileName, profile),
    }))
  );

  // Decode in parallel across up to 4 workers, with automatic main-thread fallback
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
