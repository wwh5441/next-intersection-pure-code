// Background ES Module Worker for V9.3 Pure-Math 6-Stem Conservative Master Synthesis

import { renderFullPureMathSong, STEM_MANIFEST } from './song-renderer.js';

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.cmd !== 'render') return;

  try {
    const res = renderFullPureMathSong((progress, text) => {
      self.postMessage({ type: 'progress', progress, text });
    });

    const transferList = [];
    for (const m of STEM_MANIFEST) {
      const [L, R] = res.stems[m.id];
      transferList.push(L.buffer, R.buffer);
    }

    self.postMessage(
      {
        type: 'done',
        ...res,
      },
      transferList
    );
  } catch (err) {
    self.postMessage({
      type: 'error',
      message: err && err.message ? err.message : String(err),
    });
  }
};
