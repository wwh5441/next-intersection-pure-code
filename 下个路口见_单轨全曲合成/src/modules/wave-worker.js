import '../vendor/lzma-d-min.js';
import { unpackWavePacket } from '../dsp/storage.js';

self.onmessage = ({ data }) => {
  try {
    const decoded = globalThis.LZMA.decompress(new Uint8Array(data.buffer));
    if (typeof decoded === 'string') {
      throw new Error('Unexpected string output while decoding binary VWP6 packet');
    }
    const raw = Uint8Array.from(decoded);
    const buffer = unpackWavePacket(raw);
    self.postMessage({ id: data.id, buffer }, [buffer]);
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message || String(error) });
  }
};
