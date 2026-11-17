// CabronWEC (CWC-4X) Real-Time AudioWorkletProcessor
// Renders 128-sample hardware audio quanta directly from pure-code MDCT+PNS frames
// using a zero-GC 512-sample frame cache and deterministic pre-roll seek.

import { CWC_HOP_M } from './mdct-core.js';
import { CWCStreamDecoder } from './cwc-stream-decoder.js';

class CabronWecWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.decoder = new CWCStreamDecoder(0);
    this.playing = false;
    this.currentSample = 0;
    this.totalSamples = 0;

    // 512-sample decoded frame cache (4 * 128-sample WebAudio quanta per MDCT hop)
    this.hopL = new Float32Array(CWC_HOP_M);
    this.hopR = new Float32Array(CWC_HOP_M);
    this.cachedFrameIdx = -1;

    this.telemetryCounter = 0;
    this.cpuAccumMs = 0;
    this.underruns = 0;

    this.port.onmessage = (ev) => {
      const msg = ev.data || {};
      switch (msg.type) {
        case 'init':
          this.totalSamples = msg.totalSamples || 0;
          this.decoder.setTotalFrames(msg.totalFrames || 0);
          break;
        case 'chunk':
          this.decoder.ingestChunk(msg.startFrame, msg.frames);
          break;
        case 'play':
          if (typeof msg.sampleOffset === 'number') {
            this.seek(msg.sampleOffset);
          }
          this.playing = true;
          break;
        case 'pause':
          this.playing = false;
          this.emitTelemetry(null);
          break;
        case 'seek':
          this.seek(msg.sampleOffset || 0);
          this.emitTelemetry(null);
          break;
        case 'stems':
          this.decoder.setStemControls(msg.controls || {});
          break;
        default:
          break;
      }
    };
  }

  seek(sampleOffset) {
    const clamped = Math.max(0, Math.min(Math.max(0, this.totalSamples - 1), sampleOffset | 0));
    const targetFrame = Math.floor(clamped / CWC_HOP_M);
    this.decoder.seekToFrame(targetFrame);
    this.decoder.decodeFrame(targetFrame, this.hopL, this.hopR, 0);
    this.cachedFrameIdx = targetFrame;
    this.currentSample = clamped;
  }

  emitTelemetry(avgQuantumMs = null) {
    this.port.postMessage({
      type: 'telemetry',
      playing: this.playing,
      currentSample: this.currentSample,
      currentFrame: Math.floor(this.currentSample / CWC_HOP_M),
      loadedFrames: this.decoder.loadedCount,
      underruns: this.underruns,
      cpuMs: avgQuantumMs,
      layerRms: [
        this.decoder.layerRms[0],
        this.decoder.layerRms[1],
        this.decoder.layerRms[2],
        this.decoder.layerRms[3],
        this.decoder.layerRms[4],
      ],
      spectrum32: Array.from(this.decoder.spectrum32),
    });
  }

  process(_inputs, outputs) {
    const out = outputs[0];
    if (!out || out.length < 2) return true;
    const outL = out[0];
    const outR = out[1];
    const blockLen = outL.length; // 128 samples

    if (!this.playing || this.totalSamples <= 0) {
      outL.fill(0);
      outR.fill(0);
      return true;
    }

    const hasPerf = typeof performance !== 'undefined';
    const t0 = hasPerf ? performance.now() : 0;

    for (let i = 0; i < blockLen; i++) {
      if (this.currentSample >= this.totalSamples) {
        outL.fill(0, i);
        outR.fill(0, i);
        this.playing = false;
        this.currentSample = 0;
        this.cachedFrameIdx = -1;
        this.port.postMessage({ type: 'ended' });
        break;
      }

      const frameIdx = Math.floor(this.currentSample / CWC_HOP_M);
      const intra = this.currentSample - frameIdx * CWC_HOP_M;

      if (frameIdx !== this.cachedFrameIdx) {
        if (!this.decoder.hasFrame(frameIdx)) {
          this.underruns++;
          this.hopL.fill(0);
          this.hopR.fill(0);
        } else {
          this.decoder.decodeFrame(frameIdx, this.hopL, this.hopR, 0);
        }
        this.cachedFrameIdx = frameIdx;
      }

      outL[i] = this.hopL[intra];
      outR[i] = this.hopR[intra];
      this.currentSample++;
    }

    if (hasPerf) {
      this.cpuAccumMs += performance.now() - t0;
    }
    this.telemetryCounter++;
    if (this.telemetryCounter >= 12) {
      // ~30 Hz smooth UI/oscilloscope telemetry
      const avgMs = hasPerf ? this.cpuAccumMs / this.telemetryCounter : null;
      this.telemetryCounter = 0;
      this.cpuAccumMs = 0;
      this.emitTelemetry(avgMs);
    }

    return true;
  }
}

registerProcessor('cabron-wec-processor', CabronWecWorkletProcessor);
