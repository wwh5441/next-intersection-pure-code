"""LZMA_ALONE + HTML/JS-safe UTF-8 Base122 packer for V5 Wave-Code Parameters."""
from __future__ import annotations

import hashlib
import lzma
from pathlib import Path

ILLEGALS = (0, 10, 13, 34, 60, 92)
ILLEGAL_SET = set(ILLEGALS)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compress_lzma(raw: bytes) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)


def decompress_lzma(comp: bytes) -> bytes:
    return lzma.decompress(comp, format=lzma.FORMAT_ALONE)


def encode_base122(data: bytes) -> str:
    """Encodes arbitrary bytes into HTML/JS-safe UTF-8 Base122 (`<len>:<payload>`, ~1.1429x expansion)."""
    n_bytes = len(data)
    total_bits = n_bytes * 8
    bit_pos = 0

    def next_7bits() -> int | None:
        nonlocal bit_pos
        if bit_pos >= total_bits:
            return None
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        val = (data[byte_idx] << 8) | (data[byte_idx + 1] if byte_idx + 1 < n_bytes else 0)
        chunk = (val >> (9 - bit_off)) & 0x7F
        bit_pos += 7
        return chunk

    chars: list[str] = [f"{n_bytes}:"]
    while True:
        c1 = next_7bits()
        if c1 is None:
            break
        if c1 not in ILLEGAL_SET:
            chars.append(chr(c1))
        else:
            idx = ILLEGALS.index(c1)
            c2 = next_7bits()
            if c2 is None:
                chars.append(chr(0x100 + (7 << 7) + c1))
            else:
                chars.append(chr(0x100 + (idx << 7) + c2))
    return "".join(chars)


def decode_base122(text: str) -> bytes:
    colon = text.index(":")
    out_len = int(text[:colon])
    out = bytearray(out_len)
    bit_pos = 0

    def push_7bits(val: int) -> None:
        nonlocal bit_pos
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        if byte_idx < out_len:
            shift = 1 - bit_off
            if shift >= 0:
                out[byte_idx] |= (val << shift) & 0xFF
            else:
                out[byte_idx] |= (val >> (-shift)) & 0xFF
                if byte_idx + 1 < out_len:
                    out[byte_idx + 1] |= (val << (8 + shift)) & 0xFF
        bit_pos += 7

    for ch in text[colon + 1 :]:
        cp = ord(ch)
        if cp < 0x80:
            push_7bits(cp)
        else:
            payload = cp - 0x100
            idx = payload >> 7
            v2 = payload & 0x7F
            if idx == 7:
                push_7bits(v2)
            else:
                push_7bits(ILLEGALS[idx])
                push_7bits(v2)
    return bytes(out)


def write_b122_js_module(out_path: Path, comp_bytes: bytes) -> int:
    b122 = encode_base122(comp_bytes)
    js_bytes = f'export default "{b122}";\n'.encode("utf-8")
    out_path.write_bytes(js_bytes)
    return len(js_bytes)
