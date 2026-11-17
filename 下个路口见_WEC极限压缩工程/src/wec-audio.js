// Minimal WebAudio Stereo Mix Player (`src/wec-audio.js`)
export class WecAudioPlayer {
  constructor(onEnded) {
    this.onEnded = onEnded;
    this.ctx = null;
    this.buf = null;
    this.src = null;
    this.sampleRate = 48000;
    this.numSamples = 0;
    this.durationSec = 0;
    this.mixL = null;
    this.mixR = null;
    this.playing = false;
    this.offsetSec = 0;
    this.ctxStart = 0;
  }

  loadLayers(res) {
    this.stop();
    this.sampleRate = res.sampleRate;
    this.numSamples = res.numSamples;
    this.durationSec = res.numSamples / res.sampleRate;
    const mixL = new Float32Array(this.numSamples);
    const mixR = new Float32Array(this.numSamples);
    for (const chs of Object.values(res.layers)) {
      const c0 = chs[0];
      const c1 = chs.length > 1 ? chs[1] : chs[0];
      for (let i = 0; i < this.numSamples; i++) {
        mixL[i] += c0[i];
        mixR[i] += c1[i];
      }
    }
    for (let i = 0; i < this.numSamples; i++) {
      mixL[i] = Math.max(-1, Math.min(1, mixL[i] * 0.96));
      mixR[i] = Math.max(-1, Math.min(1, mixR[i] * 0.96));
    }
    this.mixL = mixL;
    this.mixR = mixR;
    this.buf = null;
  }

  ensureContext() {
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx({ sampleRate: this.sampleRate });
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    if (!this.buf && this.mixL) {
      const ab = this.ctx.createBuffer(2, this.numSamples, this.sampleRate);
      ab.copyToChannel(this.mixL, 0);
      ab.copyToChannel(this.mixR, 1);
      this.buf = ab;
    }
    return this.ctx;
  }

  play(sec = null) {
    if (!this.mixL) return;
    const ctx = this.ensureContext();
    this.stopSource();
    const off = Math.max(0, Math.min(this.durationSec - 0.02, sec ?? this.offsetSec));
    this.offsetSec = off;
    this.ctxStart = ctx.currentTime;
    this.playing = true;
    const src = ctx.createBufferSource();
    src.buffer = this.buf;
    src.connect(ctx.destination);
    src.onended = () => {
      if (this.playing) {
        this.playing = false;
        this.offsetSec = 0;
        if (this.onEnded) this.onEnded();
      }
    };
    src.start(0, off);
    this.src = src;
  }

  pause() {
    if (!this.playing) return;
    this.offsetSec = this.getCurrentTime();
    this.stopSource();
  }

  stop() {
    this.stopSource();
    this.offsetSec = 0;
  }

  stopSource() {
    if (this.src) {
      try {
        this.src.onended = null;
        this.src.stop();
        this.src.disconnect();
      } catch (_) {}
      this.src = null;
    }
    this.playing = false;
  }

  seek(sec) {
    const clamped = Math.max(0, Math.min(this.durationSec - 0.02, sec));
    if (this.playing) this.play(clamped);
    else this.offsetSec = clamped;
  }

  getCurrentTime() {
    if (!this.playing || !this.ctx) return this.offsetSec;
    return Math.min(this.durationSec, this.offsetSec + (this.ctx.currentTime - this.ctxStart));
  }

  exportWavBlob() {
    if (!this.mixL) return null;
    const N = this.numSamples;
    const wav = new ArrayBuffer(44 + N * 4);
    const dv = new DataView(wav);
    const ws = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, 'RIFF'); dv.setUint32(4, 36 + N * 4, true); ws(8, 'WAVEfmt ');
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 2, true);
    dv.setUint32(24, this.sampleRate, true); dv.setUint32(28, this.sampleRate * 4, true);
    dv.setUint16(32, 4, true); dv.setUint16(34, 16, true);
    ws(36, 'data'); dv.setUint32(40, N * 4, true);
    let o = 44;
    for (let i = 0; i < N; i++) {
      dv.setInt16(o, Math.round(this.mixL[i] * 32767), true); o += 2;
      dv.setInt16(o, Math.round(this.mixR[i] * 32767), true); o += 2;
    }
    return new Blob([wav], { type: 'audio/wav' });
  }
}
