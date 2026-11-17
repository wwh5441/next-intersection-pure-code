// Pure Mathematical Orthonormal Princen-Bradley Local Cosine Synthesis Kernel (48,000 Hz)
// Zero External Libraries · Explicit Cooley-Tukey Radix-2 Complex FFT & Bridge Windows

export const LONG_BINS = 2048;
export const SHORT_BINS = 512;
export const PAD_ZEROS = 768;

export const MODE_ONLY_LONG = 0;
export const MODE_LONG_START = 1;
export const MODE_FOUR_SHORT = 2;
export const MODE_LONG_STOP = 3;
export const MODE_STOP_START = 4;

function buildRadix2Tables(size) {
  const bits = Math.round(Math.log2(size));
  const bitRev = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    let x = i;
    let r = 0;
    for (let b = 0; b < bits; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    bitRev[i] = r;
  }
  const cosTab = new Float64Array(size);
  const sinTab = new Float64Array(size);
  for (let half = 1; half < size; half <<= 1) {
    for (let m = 0; m < half; m++) {
      const ang = (-Math.PI * m) / half;
      cosTab[half + m] = Math.cos(ang);
      sinTab[half + m] = Math.sin(ang);
    }
  }
  return { size, bitRev, cosTab, sinTab };
}

function buildImdctPlan(nBins) {
  const h = nBins >> 1;
  const fftPlan = buildRadix2Tables(h);
  const preCos = new Float64Array(h);
  const preSin = new Float64Array(h);
  const postCos = new Float64Array(h);
  const postSin = new Float64Array(h);
  for (let r = 0; r < h; r++) {
    const a1 = (-Math.PI * r) / nBins;
    preCos[r] = Math.cos(a1);
    preSin[r] = Math.sin(a1);
    const a2 = (-Math.PI * (r + 0.25)) / nBins;
    postCos[r] = Math.cos(a2);
    postSin[r] = Math.sin(a2);
  }
  return {
    nBins,
    h,
    norm: Math.sqrt(2.0 / nBins),
    fftPlan,
    preCos,
    preSin,
    postCos,
    postSin,
    re: new Float64Array(h),
    im: new Float64Array(h),
    yDct: new Float64Array(nBins),
  };
}

function buildPrincenBradleyWindows() {
  const sinL = new Float64Array(2 * LONG_BINS);
  for (let i = 0; i < 2 * LONG_BINS; i++) {
    sinL[i] = Math.sin((Math.PI * (i + 0.5)) / (2 * LONG_BINS));
  }
  const sinS = new Float64Array(2 * SHORT_BINS);
  for (let i = 0; i < 2 * SHORT_BINS; i++) {
    sinS[i] = Math.sin((Math.PI * (i + 0.5)) / (2 * SHORT_BINS));
  }

  const makeLongBridge = (leftShort, rightShort) => {
    const w = new Float64Array(2 * LONG_BINS);
    if (!leftShort) {
      for (let i = 0; i < LONG_BINS; i++) w[i] = sinL[i];
    } else {
      for (let i = 0; i < SHORT_BINS; i++) w[PAD_ZEROS + i] = sinS[i];
      for (let i = PAD_ZEROS + SHORT_BINS; i < LONG_BINS; i++) w[i] = 1.0;
    }
    if (!rightShort) {
      for (let i = 0; i < LONG_BINS; i++) w[LONG_BINS + i] = sinL[LONG_BINS + i];
    } else {
      for (let i = LONG_BINS; i < LONG_BINS + PAD_ZEROS; i++) w[i] = 1.0;
      for (let i = 0; i < SHORT_BINS; i++) w[LONG_BINS + PAD_ZEROS + i] = sinS[SHORT_BINS + i];
    }
    return w;
  };

  return {
    longModes: [
      makeLongBridge(false, false), // 0: ONLY_LONG
      makeLongBridge(false, true),  // 1: LONG_START
      null,                         // 2: FOUR_SHORT
      makeLongBridge(true, false),  // 3: LONG_STOP
      makeLongBridge(true, true),   // 4: STOP_START
    ],
    shortWin: sinS,
  };
}

export function createOrthonormalTdacKernel() {
  const planLong = buildImdctPlan(LONG_BINS);
  const planShort = buildImdctPlan(SHORT_BINS);
  const windows = buildPrincenBradleyWindows();
  const scratchLong2N = new Float64Array(2 * LONG_BINS);
  const scratchShort2N = new Float64Array(2 * SHORT_BINS);

  function runFftInPlace(re, im, fftPlan) {
    const n = fftPlan.size;
    const bitRev = fftPlan.bitRev;
    const cosTab = fftPlan.cosTab;
    const sinTab = fftPlan.sinTab;

    for (let i = 0; i < n; i++) {
      const j = bitRev[i];
      if (i < j) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }

    for (let half = 1; half < n; half <<= 1) {
      const step = half << 1;
      for (let base = 0; base < n; base += step) {
        for (let m = 0; m < half; m++) {
          const wr = cosTab[half + m];
          const wi = sinTab[half + m];
          const i = base + m;
          const j = i + half;
          const rj = re[j];
          const ij = im[j];
          const tr = wr * rj - wi * ij;
          const ti = wr * ij + wi * rj;
          re[j] = re[i] - tr;
          im[j] = im[i] - ti;
          re[i] += tr;
          im[i] += ti;
        }
      }
    }
  }

  function imdctWithPlan(coeffs, coeffOffset, plan, out2N) {
    const n = plan.nBins;
    const h = plan.h;
    const norm = plan.norm;
    const re = plan.re;
    const im = plan.im;
    const preCos = plan.preCos;
    const preSin = plan.preSin;
    const postCos = plan.postCos;
    const postSin = plan.postSin;
    const yDct = plan.yDct;

    for (let r = 0; r < h; r++) {
      const ar = coeffs[coeffOffset + 2 * r];
      const ai = coeffs[coeffOffset + n - 1 - 2 * r];
      const pc = preCos[r];
      const ps = preSin[r];
      re[r] = ar * pc - ai * ps;
      im[r] = ar * ps + ai * pc;
    }

    runFftInPlace(re, im, plan.fftPlan);

    for (let r = 0; r < h; r++) {
      const qc = postCos[r];
      const qs = postSin[r];
      const rr = re[r];
      const ii = im[r];
      yDct[2 * r] = rr * qc - ii * qs;
      yDct[n - 1 - 2 * r] = -(rr * qs + ii * qc);
    }

    for (let k = 0; k < h; k++) {
      out2N[k] = norm * yDct[h + k];
      out2N[h + k] = -norm * yDct[n - 1 - k];
      out2N[n + k] = -norm * yDct[h - 1 - k];
      out2N[n + h + k] = -norm * yDct[k];
    }
  }

  function synthesizeFrameOverlapAdd(coeffs2048, mode, outPadded, frameBaseSample) {
    if (mode !== MODE_FOUR_SHORT) {
      imdctWithPlan(coeffs2048, 0, planLong, scratchLong2N);
      const win = windows.longModes[mode] || windows.longModes[0];
      for (let i = 0; i < 2 * LONG_BINS; i++) {
        outPadded[frameBaseSample + i] += scratchLong2N[i] * win[i];
      }
    } else {
      const winS = windows.shortWin;
      for (let w = 0; w < 4; w++) {
        imdctWithPlan(coeffs2048, w * SHORT_BINS, planShort, scratchShort2N);
        const dstBase = frameBaseSample + PAD_ZEROS + w * SHORT_BINS;
        for (let i = 0; i < 2 * SHORT_BINS; i++) {
          outPadded[dstBase + i] += scratchShort2N[i] * winS[i];
        }
      }
    }
  }

  return {
    synthesizeFrameOverlapAdd,
  };
}
