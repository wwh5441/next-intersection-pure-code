// Stereo Chorus Vocal Harmony (`04`) & 8-Prime Householder FDN Studio Space (`05`) Synthesizer
// (`src/dsp/backing-fdn-synth.js`)

import { synthesizeLappedStereoEvents } from './lapped-atom-synth.js';
import { resampleZeroPhasePoly1D, resampleZeroPhasePolyStereo } from './polyphase-sinc.js';

export function runHouseholderFdn24k(dryMono24, fdnPhysics) {
  const {
    internalRate = 24000,
    preDelaySamples = 576,
    delays = [733, 887, 1061, 1277, 1511, 1789, 2089, 2411],
    rt60Sec = 1.35,
    damping = 0.34,
    hpPole = 0.976,
    wetGain = 0.035,
  } = fdnPhysics;

  const N = dryMono24.length;
  const outL = new Float32Array(N);
  const outR = new Float32Array(N);

  const preBuf = new Float32Array(preDelaySamples);
  let prePtr = 0;

  const lines = delays.map((d) => new Float32Array(d));
  const ptrs = new Int32Array(8);
  const lpState = new Float64Array(8);
  const decays = new Float64Array(8);
  for (let i = 0; i < 8; i++) {
    decays[i] = Math.pow(10.0, (-3.0 * delays[i]) / (rt60Sec * internalRate));
  }

  const oneMinusDamp = 1.0 - damping;
  const inScale = 0.35;
  const tapOut = new Float64Array(8);

  let hpPrevXL = 0.0;
  let hpPrevYL = 0.0;
  let hpPrevXR = 0.0;
  let hpPrevYR = 0.0;

  for (let n = 0; n < N; n++) {
    const delayedIn = preBuf[prePtr];
    preBuf[prePtr] = dryMono24[n] * inScale;
    prePtr++;
    if (prePtr === preDelaySamples) prePtr = 0;

    let sum8 = 0.0;
    for (let i = 0; i < 8; i++) {
      const buf = lines[i];
      const p = ptrs[i];
      const raw = buf[p];
      const damped = damping * lpState[i] + oneMinusDamp * raw;
      lpState[i] = damped;
      const decayed = damped * decays[i];
      tapOut[i] = decayed;
      sum8 += decayed;
    }

    // Unitary 8x8 Householder reflection: H * v = v - (2/8) * sum(v) * 1
    const houseShift = 0.25 * sum8;
    for (let i = 0; i < 8; i++) {
      const buf = lines[i];
      const p = ptrs[i];
      const signIn = i & 1 ? -delayedIn : delayedIn;
      buf[p] = tapOut[i] - houseShift + signIn;
      const np = p + 1;
      ptrs[i] = np === delays[i] ? 0 : np;
    }

    const rawWetL = (tapOut[0] - tapOut[2] + tapOut[4] - tapOut[6]) * wetGain;
    const rawWetR = (tapOut[1] - tapOut[3] + tapOut[5] - tapOut[7]) * wetGain;

    // 180 Hz 1-pole highpass mud cut
    const yL = rawWetL - hpPrevXL + hpPole * hpPrevYL;
    hpPrevXL = rawWetL;
    hpPrevYL = yL;

    const yR = rawWetR - hpPrevXR + hpPole * hpPrevYR;
    hpPrevXR = rawWetR;
    hpPrevYR = yR;

    outL[n] = yL;
    outR[n] = yR;
  }

  return [outL, outR];
}

export function synthesizeStudioFdnSpaceTrack(
  fdnPhysics,
  leadMono48,
  guitarL48,
  guitarR48,
  drumsLowL48,
  drumsLowR48,
  totalSamples48
) {
  const dryMix48 = new Float32Array(totalSamples48);
  const { sendLead = 0.75, sendGuitar = 0.18, sendDrums = 0.07 } = fdnPhysics;
  for (let i = 0; i < totalSamples48; i++) {
    dryMix48[i] =
      sendLead * leadMono48[i] +
      sendGuitar * 0.5 * (guitarL48[i] + guitarR48[i]) +
      sendDrums * 0.5 * (drumsLowL48[i] + drumsLowR48[i]);
  }
  const dryMono24 = resampleZeroPhasePoly1D(dryMix48, 1, 2, Math.round(totalSamples48 / 2));
  const [wetL24, wetR24] = runHouseholderFdn24k(dryMono24, fdnPhysics);
  return resampleZeroPhasePolyStereo(wetL24, wetR24, 2, 1, totalSamples48);
}

export function synthesizeBackingHarmonyTrack(harmonyAtomDict, harmonyEvents, totalSamples48) {
  // Critically sampled @ 16,000 Hz (`upRatio = 3` -> 48,000 Hz)
  const [subL, subR] = synthesizeLappedStereoEvents(harmonyAtomDict, harmonyEvents);
  return resampleZeroPhasePolyStereo(subL, subR, 3, 1, totalSamples48);
}
