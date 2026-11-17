// Background Web Worker for Off-Main-Thread 10-Track 48kHz Synthesis (`src/modules/synth-worker.js`)
import { synthesizeAllTenTracks, synthesizeVocalOnlyTracks } from '../dsp/engine-orchestrator.js';

self.onmessage = (ev) => {
  const { id, cmd, tierId, pitchSemitones } = ev.data || {};
  try {
    if (cmd === 'synth_all') {
      const res = synthesizeAllTenTracks(tierId, { pitchSemitones }, (progress, status) => {
        self.postMessage({ id, type: 'progress', progress, status });
      });
      const transferList = [];
      for (const chArr of Object.values(res.tracks)) {
        for (const f32 of chArr) {
          transferList.push(f32.buffer);
        }
      }
      self.postMessage({ id, type: 'done_all', payload: res }, transferList);
    } else if (cmd === 'synth_vocal') {
      self.postMessage({
        id,
        type: 'progress',
        progress: 0.4,
        status: `正在按 GCI 基音周期字典重排主唱音高 (${pitchSemitones > 0 ? '+' : ''}${pitchSemitones} 半音)...`,
      });
      const res = synthesizeVocalOnlyTracks(tierId, pitchSemitones);
      const transferList = [];
      for (const chArr of Object.values(res.tracks)) {
        for (const f32 of chArr) {
          transferList.push(f32.buffer);
        }
      }
      self.postMessage({ id, type: 'done_vocal', payload: res }, transferList);
    }
  } catch (err) {
    self.postMessage({ id, type: 'error', message: err?.message || String(err) });
  }
};
