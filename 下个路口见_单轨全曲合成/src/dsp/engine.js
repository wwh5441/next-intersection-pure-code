import { CosineWaveStream } from './wave-stream.js';
import { HighAirStream } from './air-stream.js';
import { StudioFdnReverb } from './reverb-synth.js';

function createSynthForBuffer(buffer, sampleRate) {
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic === 'VWQ6') return { kind: 'wave', synth: new CosineWaveStream(buffer, sampleRate) };
  if (magic === 'VWA7') return { kind: 'air', synth: new HighAirStream(buffer, sampleRate) };
  if (magic === 'VWR7') return { kind: 'reverb', synth: new StudioFdnReverb(buffer, sampleRate) };
  throw new Error(`Unsupported vocal synth stream magic: ${magic}`);
}

export class VocalMultitrackEngine {
  constructor(manifest, sampleRate, payloads) {
    this.manifest = manifest;
    this.sampleRate = sampleRate;
    this.end = Math.ceil(manifest.duration * sampleRate);
    this.tracks = manifest.tracks.map((track) => ({
      id: track.id,
      group: track.group || 'vocal',
      gain: 1,
      target: 1,
      streams: track.streams.map((stream) => {
        const built = createSynthForBuffer(payloads[stream.file], sampleRate);
        return {
          channel: stream.channel,
          kind: built.kind,
          synth: built.synth,
        };
      }),
    }));
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
      for (const stream of track.streams) stream.synth.reset();
    }
  }

  process(left, right) {
    left.fill(0);
    right.fill(0);
    if (!this.running) return;
    const meters = {};
    for (const track of this.tracks) meters[track.id] = 0;
    let actual = 0;

    const separateRevTrack = this.tracks.find(
      (t) => t.streams.some((s) => s.kind === 'reverb') && t.streams.every((s) => s.kind === 'reverb')
    );
    const needDryForSeparateRev = Boolean(
      separateRevTrack && (separateRevTrack.gain >= 1e-5 || separateRevTrack.target > 0)
    );

    for (let i = 0; i < left.length; i++) {
      if (this.frame >= this.end) {
        this.running = false;
        break;
      }
      let l = 0;
      let r = 0;
      let dryMid = 0;
      let drySide = 0;

      for (const track of this.tracks) {
        track.gain += (track.target - track.gain) * 0.0035;
        const trackSilent = track.gain < 1e-5 && track.target === 0;
        const hasReverb = track.streams.some((s) => s.kind === 'reverb');
        const hasDry = track.streams.some((s) => s.kind !== 'reverb');

        if (hasDry && hasReverb) {
          const feedsSeparate = needDryForSeparateRev && track.group === 'vocal';
          if (trackSilent && !feedsSeparate) continue;
          let rawL = 0;
          let rawR = 0;
          for (const stream of track.streams) {
            if (stream.kind === 'reverb') continue;
            const val = stream.synth.valueAt(this.frame);
            rawL += val;
            rawR += stream.channel === 'side' ? -val : val;
          }
          const localMid = 0.5 * (rawL + rawR);
          const localSide = 0.5 * (rawL - rawR);
          if (track.group === 'vocal') {
            dryMid += localMid;
            drySide += localSide;
          }
          if (!trackSilent) {
            for (const stream of track.streams) {
              if (stream.kind !== 'reverb') continue;
              const [wL, wR] = stream.synth.step(this.frame, localMid, localSide);
              rawL += wL;
              rawR += wR;
            }
            const a = rawL * track.gain;
            const b = rawR * track.gain;
            meters[track.id] += (a * a + b * b) * 0.5;
            l += a;
            r += b;
          }
        } else if (hasDry) {
          const feedsSeparate = needDryForSeparateRev && track.group === 'vocal';
          if (trackSilent && !feedsSeparate) continue;
          let rawL = 0;
          let rawR = 0;
          for (const stream of track.streams) {
            const val = stream.synth.valueAt(this.frame);
            rawL += val;
            rawR += stream.channel === 'side' ? -val : val;
          }
          if (track.group === 'vocal') {
            dryMid += 0.5 * (rawL + rawR);
            drySide += 0.5 * (rawL - rawR);
          }
          if (!trackSilent) {
            const a = rawL * track.gain;
            const b = rawR * track.gain;
            meters[track.id] += (a * a + b * b) * 0.5;
            l += a;
            r += b;
          }
        } else if (!trackSilent) {
          let revL = 0;
          let revR = 0;
          for (const stream of track.streams) {
            const [wL, wR] = stream.synth.step(this.frame, dryMid, drySide);
            revL += wL;
            revR += wR;
          }
          const a = revL * track.gain;
          const b = revR * track.gain;
          meters[track.id] += (a * a + b * b) * 0.5;
          l += a;
          r += b;
        }
      }

      const ceiling = (v) =>
        Math.abs(v) <= 0.96 ? v : Math.sign(v) * (0.96 + 0.03 * Math.tanh((Math.abs(v) - 0.96) / 0.03));
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
