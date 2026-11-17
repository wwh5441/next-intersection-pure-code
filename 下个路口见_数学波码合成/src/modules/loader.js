import { decodeBase122 } from './base122.js';

export async function loadWavePackets(manifest, profile, onProgress) {
  const workerUrl = window.__VFS_WORKER_URL || new URL('./wave-worker.js', import.meta.url);
  const worker = new Worker(workerUrl);
  const pending = new Map();

  const allFiles = manifest.tracks.map((t) => t.file);
  let completed = 0;

  worker.onmessage = ({ data }) => {
    const request = pending.get(data.id);
    if (!request) return;
    pending.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else {
      request.resolve(data.buffer);
      onProgress?.(++completed, allFiles.length);
    }
  };

  worker.onerror = (event) => {
    for (const request of pending.values()) {
      request.reject(new Error(event.message || 'Wave worker decode failed'));
    }
    pending.clear();
  };

  try {
    const pairs = await Promise.all(
      allFiles.map(async (fileName, id) => {
        let buffer;
        if (window.__VFS_BINARY_STREAMS) {
          const slice = window.__VFS_BINARY_STREAMS[`${profile}/${fileName}`] || window.__VFS_BINARY_STREAMS[fileName];
          if (!slice) throw new Error(`Missing embedded wave packet: ${profile}/${fileName}`);
          buffer = slice.slice(0).buffer;
        } else {
          const jsName = fileName.replace(/\.wvp$/, '.js');
          const mod = await import(`../../data/${profile}/${jsName}`);
          buffer = decodeBase122(mod.default).buffer;
        }
        const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
        worker.postMessage({ id, buffer }, [buffer]);
        return [fileName, await result];
      })
    );
    return Object.fromEntries(pairs);
  } finally {
    worker.terminate();
  }
}
