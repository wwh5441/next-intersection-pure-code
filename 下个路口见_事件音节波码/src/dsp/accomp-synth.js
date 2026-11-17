import { unpackWavePacket } from './v7-storage.js';
import { CosineWaveStream } from './v7-wave-stream.js';
import { HighAirStream } from './v7-air-stream.js';
import { unpackWaveStream } from './v5-unpacker.js';
import { DrumWaveSynth } from './v5-drum-synth.js';
import { BassWaveSynth } from './v5-bass-synth.js';
import { HarmonyWaveSynth } from './v5-harmony-synth.js';
export { synthesizeV8DictAccompaniment } from './accomp-dict-synth.js';

const V7_B122_ILLEGALS = [0, 10, 13, 34, 60, 92];

export function decodeV7Base122(str) {
  const colon = str.indexOf(':');
  const outLen = Number(str.slice(0, colon));
  const out = new Uint8Array(outLen);
  let bitPos = 0;

  function push7(v) {
    const byteIdx = bitPos >> 3;
    const bitOff = bitPos & 7;
    if (byteIdx < outLen) {
      const shift = 1 - bitOff;
      if (shift >= 0) {
        out[byteIdx] |= (v << shift) & 0xff;
      } else {
        out[byteIdx] |= v >> -shift;
        if (byteIdx + 1 < outLen) {
          out[byteIdx + 1] |= (v << (8 + shift)) & 0xff;
        }
      }
    }
    bitPos += 7;
  }

  for (let i = colon + 1; i < str.length; i++) {
    const cp = str.charCodeAt(i);
    if (cp < 0x80) {
      push7(cp);
    } else {
      const payload = cp - 0x100;
      const idx = payload >> 7;
      const v2 = payload & 0x7f;
      if (idx === 7) push7(v2);
      else {
        push7(V7_B122_ILLEGALS[idx]);
        push7(v2);
      }
    }
  }
  return out;
}

export function synthesizeV7Accompaniment(rawStreams, numSamples, sampleRate = 48000) {
  const bassStream = new CosineWaveStream(unpackWavePacket(rawStreams.bass), sampleRate);
  const drumsBody = new CosineWaveStream(unpackWavePacket(rawStreams.drums_body), sampleRate);
  const drumsAir = new HighAirStream(unpackWavePacket(rawStreams.drums_air), sampleRate);
  const accMid = new CosineWaveStream(unpackWavePacket(rawStreams.accomp_mid), sampleRate);
  const accSide = new CosineWaveStream(unpackWavePacket(rawStreams.accomp_side), sampleRate);
  const accAir = new HighAirStream(unpackWavePacket(rawStreams.accomp_air), sampleRate);

  const bassL = new Float32Array(numSamples);
  const bassR = new Float32Array(numSamples);
  const drumsL = new Float32Array(numSamples);
  const drumsR = new Float32Array(numSamples);
  const gtrL = new Float32Array(numSamples);
  const gtrR = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    const bv = bassStream.valueAt(i);
    bassL[i] = bv;
    bassR[i] = bv;

    const dv = drumsBody.valueAt(i) + drumsAir.valueAt(i);
    drumsL[i] = dv;
    drumsR[i] = dv;

    const mv = accMid.valueAt(i) + accAir.valueAt(i);
    const sv = accSide.valueAt(i);
    gtrL[i] = mv + sv;
    gtrR[i] = mv - sv;
  }

  return {
    bass: [bassL, bassR],
    drums: [drumsL, drumsR],
    guitar_synth: [gtrL, gtrR],
  };
}

export function synthesizeV5Accompaniment(rawStreams, models, numSamples, sampleRate = 48000) {
  const drumsSynth = new DrumWaveSynth(models.drums, unpackWaveStream(rawStreams.drums), sampleRate);
  const bassSynth = new BassWaveSynth(models.bass, unpackWaveStream(rawStreams.bass), sampleRate);
  const harmSynth = new HarmonyWaveSynth(models.harmony, unpackWaveStream(rawStreams.harmony), sampleRate);

  const bassL = new Float32Array(numSamples);
  const bassR = new Float32Array(numSamples);
  const drumsL = new Float32Array(numSamples);
  const drumsR = new Float32Array(numSamples);
  const gtrL = new Float32Array(numSamples);
  const gtrR = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    const bv = bassSynth.valueAt(i);
    bassL[i] = bv;
    bassR[i] = bv;

    const dv = drumsSynth.valueAt(i);
    drumsL[i] = dv;
    drumsR[i] = dv;

    const pair = harmSynth.stereoAt(i);
    gtrL[i] = pair[0];
    gtrR[i] = pair[1];
  }

  return {
    bass: [bassL, bassR],
    drums: [drumsL, drumsR],
    guitar_synth: [gtrL, gtrR],
  };
}
