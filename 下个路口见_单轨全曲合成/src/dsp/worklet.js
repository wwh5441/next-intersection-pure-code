import { VocalMultitrackEngine } from './engine.js';

class SingleTrackMasterProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { manifest, payloads } = options.processorOptions;
    this.engine = new VocalMultitrackEngine(manifest, sampleRate, payloads);
    this.disposed = false;
    this.reportCounter = 0;

    this.port.onmessage = ({ data }) => {
      if (data.type === 'play') this.engine.running = true;
      if (data.type === 'pause') this.engine.running = false;
      if (data.type === 'seek') this.engine.seek(data.time);
      if (data.type === 'gain') this.engine.setGain(data.track, data.value);
      if (data.type === 'dispose') {
        this.engine.running = false;
        this.disposed = true;
      }
    };
  }

  process(_inputs, outputs) {
    if (this.disposed) return false;
    const output = outputs[0];
    this.engine.process(output[0], output[1]);
    if (++this.reportCounter >= 8) {
      this.reportCounter = 0;
      this.port.postMessage({
        time: this.engine.frame / sampleRate,
        running: this.engine.running,
        computed: this.engine.totalComputed,
        meters: this.engine.lastMeters,
      });
    }
    return true;
  }
}

registerProcessor('single-track-master-synth', SingleTrackMasterProcessor);
