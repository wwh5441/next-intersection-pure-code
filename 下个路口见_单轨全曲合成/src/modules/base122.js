const B122_ILLEGALS = [0, 10, 13, 34, 60, 92];

export function decodeBase122(str) {
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
        push7(B122_ILLEGALS[idx]);
        push7(v2);
      }
    }
  }
  return out;
}
