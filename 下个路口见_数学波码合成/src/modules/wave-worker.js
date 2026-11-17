importScripts('../vendor/lzma-d-min.js');

self.onmessage = ({ data }) => {
  try {
    const decoded = self.LZMA.decompress(new Uint8Array(data.buffer));
    if (typeof decoded === 'string') {
      throw new Error('Unexpected text output in binary WVP5 wave packet');
    }
    const out = Uint8Array.from(decoded);
    self.postMessage({ id: data.id, buffer: out.buffer }, [out.buffer]);
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message || String(error) });
  }
};
