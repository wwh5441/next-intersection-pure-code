import { unpackWaveStream } from './unpacker.js';
import { VocalWaveSynth } from './vocal-synth.js';
import { DrumWaveSynth } from './drum-synth.js';
import { BassWaveSynth } from './bass-synth.js';
import { HarmonyWaveSynth } from './harmony-synth.js';

export class MultitrackWaveEngine {
  constructor(manifest, sampleRate, payloads) {
    this.manifest = manifest;
    this.sampleRate = sampleRate;
    this.end = Math.ceil(manifest.duration * sampleRate);
    this.tracks = manifest.tracks.map((track) => {
      const unpacked = unpackWaveStream(payloads[track.file]);
      let synth;
      let isStereo = false;
      if (track.id === 'vocal') {
        synth = new VocalWaveSynth(track.model, unpacked, sampleRate);
      } else if (track.id === 'drums') {
        synth = new DrumWaveSynth(track.model, unpacked, sampleRate);
      } else if (track.id === 'bass') {
        synth = new BassWaveSynth(track.model, unpacked, sampleRate);
      } else {
        synth = new HarmonyWaveSynth(track.model, unpacked, sampleRate);
        isStereo = true;
      }
      return {
        id: track.id,
        gain: 1,
        target: 1,
        isStereo,
        synth,
      };
    });
    this.frame = 0;
    this.totalComputed = 0;
    this.running = false;
    this.lastMeters = {};
  }

  setGain(id, value) {
    const track = this.tracks.find((item) => item.id === id);
    if (track) track.target = Math.max(0, Math.min(1.5, value));
  }

  seek(seconds) {
    this.frame = Math.min(this.end, Math.max(0, Math.floor(seconds * this.sampleRate)));
    if (seconds >= this.manifest.duration) this.frame = this.end;
    for (const track of this.tracks) {
      track.synth.reset(this.frame);
    }
  }

  process(left, right) {
    left.fill(0);
    right.fill(0);
    if (!this.running) return;
    const meters = {};
    for (const track of this.tracks) meters[track.id] = 0;
    let actual = 0;

    for (let i = 0; i < left.length; i++) {
      if (this.frame >= this.end) {
        this.running = false;
        break;
      }
      let l = 0;
      let r = 0;
      for (const track of this.tracks) {
        track.gain += (track.target - track.gain) * 0.003;
        const g = track.gain;
        if (g <= 1e-5) continue;
        let a = 0;
        let b = 0;
        if (track.isStereo) {
          const pair = track.synth.stereoAt(this.frame);
          a = pair[0] * g;
          b = pair[1] * g;
        } else {
          const v = track.synth.valueAt(this.frame) * g;
          a = v;
          b = v;
        }
        meters[track.id] += (a * a + b * b) * 0.5;
        l += a;
        r += b;
      }

      const ceiling = (v) =>
        Math.abs(v) <= 0.95 ? v : Math.sign(v) * (0.95 + 0.04 * Math.tanh((Math.abs(v) - 0.95) / 0.04));
      left[i] = ceiling(l);
      right[i] = ceiling(r);
      this.frame++;
      actual++;
    }

    this.totalComputed += actual;
    for (const id in meters) {
      this.lastMeters[id] = Math.sqrt(meters[id] / Math.max(1, actual));
    }
  }
}
