// Shared loader of all V9R2.1 pure-code model/score modules (`src/modules/pure-code-assets.js`)
import { VOCAL_SPEAKER_MODEL } from '../../models/vocal-speaker-model.js';
import { VOCAL_PHONEME_ATLAS } from '../../models/vocal-phoneme-atlas.js';
import { BACKING_MODEL } from '../../models/backing-model.js';
import { SPACE_FDN } from '../../models/space-fdn.js';
import { BASS_MODEL } from '../../models/bass-model.js';
import { DRUM_MODEL } from '../../models/drum-model.js';
import { POLY_MODEL } from '../../models/poly-model.js';
import { VOCAL_SCORE } from '../../data/vocal-score.js';
import { BACKING_SCORE } from '../../data/backing-score.js';
import { BASS_SCORE } from '../../data/bass-score.js';
import { DRUM_SCORE } from '../../data/drum-score.js';
import { POLY_SCORE } from '../../data/poly-score.js';
import { BAND_GAINS } from '../../data/band-gains.js';
import { STUDIO_METADATA } from '../../data/studio-metadata.js';

export function getRenderArgs() {
  return {
    config: STUDIO_METADATA.config,
    speakerModel: VOCAL_SPEAKER_MODEL,
    phonemeAtlas: VOCAL_PHONEME_ATLAS,
    backingModel: BACKING_MODEL,
    spaceFdn: SPACE_FDN,
    vocalScore: VOCAL_SCORE,
    backingScore: BACKING_SCORE,
    bandModels: { bass: BASS_MODEL, drums: DRUM_MODEL, poly: POLY_MODEL },
    bandScores: { bass: BASS_SCORE, drums: DRUM_SCORE, poly: POLY_SCORE },
    bandGains: BAND_GAINS,
  };
}
