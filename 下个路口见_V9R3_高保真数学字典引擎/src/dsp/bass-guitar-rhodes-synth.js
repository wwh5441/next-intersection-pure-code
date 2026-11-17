// 98.0000 BPM Bass Groove (`08`), Guitar Funk Riff (`09`), & Rhodes Pad (`10`) Synthesizers
// (`src/dsp/bass-guitar-rhodes-synth.js`)

import { synthesizeBarMotifPlusFill } from './drum-percussion-synth.js';
import { synthesizeLappedStereoEvents } from './lapped-atom-synth.js';
import { resampleZeroPhasePolyStereo } from './polyphase-sinc.js';

export function synthesizeBassGrooveTrack(bassDict, bassArrangement, totalSamples48) {
  // Critically sampled @ 4,000 Hz (`upRatio = 12` -> 48,000 Hz)
  return synthesizeBarMotifPlusFill(
    bassDict.atomDict,
    bassDict.barMotifs,
    bassArrangement,
    12,
    totalSamples48
  );
}

export function synthesizeGuitarRiffTrack(guitarAtomDict, guitarEvents, totalSamples48) {
  // Critically sampled @ 16,000 Hz (`upRatio = 3` -> 48,000 Hz, >= 480 Hz acoustic/funk plucks & strums)
  const [subL, subR] = synthesizeLappedStereoEvents(guitarAtomDict, guitarEvents);
  return resampleZeroPhasePolyStereo(subL, subR, 3, 1, totalSamples48);
}

export function synthesizeRhodesPadTrack(rhodesAtomDict, rhodesEvents, totalSamples48) {
  // Critically sampled @ 3,000 Hz (`upRatio = 16` -> 48,000 Hz, < 480 Hz warm sustained electric piano voicings)
  const [subL, subR] = synthesizeLappedStereoEvents(rhodesAtomDict, rhodesEvents);
  return resampleZeroPhasePolyStereo(subL, subR, 16, 1, totalSamples48);
}
