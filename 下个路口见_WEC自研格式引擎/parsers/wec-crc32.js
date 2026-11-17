// IEEE 802.3 CRC-32 & Ogg Direct-Polynomial CRC-32 Utilities for WEC v2.0

const IEEE_CRC32_TABLE = (() => {
  const tab = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    tab[i] = c >>> 0;
  }
  return tab;
})();

const OGG_CRC32_TABLE = (() => {
  const tab = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let r = i << 24;
    for (let k = 0; k < 8; k++) {
      r = r & 0x80000000 ? ((r << 1) ^ 0x04c11db7) >>> 0 : (r << 1) >>> 0;
    }
    tab[i] = r >>> 0;
  }
  return tab;
})();

export function computeWecCrc32(u8, crcOffset = 44) {
  let crc = 0xffffffff;
  const len = u8.length;
  const crcEnd = crcOffset + 4;
  for (let i = 0; i < len; i++) {
    const byte = i >= crcOffset && i < crcEnd ? 0 : u8[i];
    crc = IEEE_CRC32_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function computeOggPageCrc32(pageU8) {
  let crc = 0;
  for (let i = 0; i < pageU8.length; i++) {
    crc = ((crc << 8) ^ OGG_CRC32_TABLE[((crc >>> 24) ^ pageU8[i]) & 0xff]) >>> 0;
  }
  return crc >>> 0;
}

export function formatHex32(u32) {
  return (u32 >>> 0).toString(16).toUpperCase().padStart(8, '0');
}
