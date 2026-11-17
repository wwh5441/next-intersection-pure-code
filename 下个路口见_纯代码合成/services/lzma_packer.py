"""V4 MDP4 Delta-Scale + Sparse Bit-Packing + LZMA Extreme Compressor + UTF-8 Base122 Codec."""
from __future__ import annotations

import hashlib
import lzma
from pathlib import Path
import struct
import numpy as np

HEADER_FMT = "<4sIHHIIBBBBBBHI"
HEADER_SIZE = 32
MODE_FOUR_SHORT = 2

# HTML <script> & JS "..." safe UTF-8 Base122 illegal byte table: NUL, LF, CR, ", <, \
B122_ILLEGALS = (0, 10, 13, 34, 60, 92)
B122_ILLEGAL_MAP = {b: i for i, b in enumerate(B122_ILLEGALS)}
B122_SHORTENED = 7


def pack_mdp4_lzma(raw: bytes) -> bytes:
    """Losslessly packs `MDQ4` fixed-stride frame bytes into `MDP4` delta-scale sparse stream + LZMA."""
    (
        magic,
        sr,
        long_n,
        short_n,
        t_count,
        samples,
        base_qmax,
        p100,
        fold_l,
        fold_s,
        b_long,
        b_short,
        flags,
        _,
    ) = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
    assert magic == b"MDQ4"
    stride = 1 + 48 + long_n
    body = np.frombuffer(raw, np.uint8, offset=HEADER_SIZE).reshape(t_count, stride)

    modes = body[:, 0]
    scales = body[:, 1:49]
    q = body[:, 49:].copy().view(np.int8).ravel()
    present = q != 0
    nonzero = int(np.sum(present))

    # Delta-code active scales (24 bytes for Long/Bridge modes, 48 subblock-delta bytes for FOUR_SHORT)
    delta_chunks: list[bytes] = []
    prev_long = np.zeros(b_long, dtype=np.int32)
    for t in range(t_count):
        if modes[t] != MODE_FOUR_SHORT:
            cur = scales[t, :b_long].astype(np.int32)
            diff = ((cur - prev_long) & 0xFF).astype(np.uint8)
            delta_chunks.append(diff.tobytes())
            prev_long = cur
        else:
            s4 = scales[t].astype(np.int32).reshape(4, b_short)
            d4 = np.empty_like(s4)
            d4[0] = s4[0]
            d4[1:] = (s4[1:] - s4[:-1]) & 0xFF
            delta_chunks.append(d4.astype(np.uint8).tobytes())

    scales_packed = b"".join(delta_chunks)
    header_out = struct.pack(
        HEADER_FMT,
        b"MDP4",
        sr,
        long_n,
        short_n,
        t_count,
        samples,
        base_qmax,
        p100,
        fold_l,
        fold_s,
        b_long,
        b_short,
        flags,
        nonzero,
    )
    payload = (
        header_out
        + modes.tobytes()
        + scales_packed
        + np.packbits(present).tobytes()
        + np.packbits(q[present] < 0).tobytes()
        + np.abs(q[present]).astype(np.uint8).tobytes()
    )
    encoded = lzma.compress(payload, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    assert unpack_mdp4_lzma(encoded) == raw
    return encoded


def unpack_mdp4_lzma(encoded: bytes) -> bytes:
    """Decompresses LZMA `MDP4` stream back into bit-identical `MDQ4` fixed-stride buffer."""
    raw = lzma.decompress(encoded, format=lzma.FORMAT_ALONE)
    (
        magic,
        sr,
        long_n,
        short_n,
        t_count,
        samples,
        base_qmax,
        p100,
        fold_l,
        fold_s,
        b_long,
        b_short,
        flags,
        nonzero,
    ) = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
    assert magic == b"MDP4"

    offset = HEADER_SIZE
    modes = np.frombuffer(raw, np.uint8, offset=offset, count=t_count)
    offset += t_count

    scales = np.zeros((t_count, 48), dtype=np.uint8)
    prev_long = np.zeros(b_long, dtype=np.int32)
    for t in range(t_count):
        if modes[t] != MODE_FOUR_SHORT:
            diff = np.frombuffer(raw, np.uint8, offset=offset, count=b_long).astype(np.int32)
            offset += b_long
            cur = (prev_long + diff) & 0xFF
            scales[t, :b_long] = cur.astype(np.uint8)
            prev_long = cur
        else:
            d4 = np.frombuffer(raw, np.uint8, offset=offset, count=48).astype(np.int32).reshape(4, b_short)
            offset += 48
            s4 = np.cumsum(d4, axis=0) & 0xFF
            scales[t] = s4.reshape(48).astype(np.uint8)

    total_bins = t_count * long_n
    mask_bytes = (total_bins + 7) // 8
    sign_bytes = (nonzero + 7) // 8

    present = np.unpackbits(np.frombuffer(raw, np.uint8, offset=offset, count=mask_bytes))[:total_bins].astype(bool)
    offset += mask_bytes
    signs = np.unpackbits(np.frombuffer(raw, np.uint8, offset=offset, count=sign_bytes))[:nonzero]
    offset += sign_bytes
    mag = np.frombuffer(raw, np.uint8, offset=offset, count=nonzero).astype(np.int16)
    offset += nonzero
    assert offset == len(raw), f"Trailing bytes in MDP4: {offset} != {len(raw)}"

    q = np.zeros(total_bins, dtype=np.int8)
    q[present] = (mag * (1 - 2 * signs.astype(np.int16))).astype(np.int8)

    header_mdq4 = struct.pack(
        HEADER_FMT,
        b"MDQ4",
        sr,
        long_n,
        short_n,
        t_count,
        samples,
        base_qmax,
        p100,
        fold_l,
        fold_s,
        b_long,
        b_short,
        flags,
        0,
    )
    body = np.concatenate(
        [modes[:, None], scales, q.reshape(t_count, long_n).view(np.uint8)],
        axis=1,
    )
    return header_mdq4 + body.tobytes()


def encode_base122(data: bytes) -> str:
    """Encodes arbitrary binary bytes into HTML `<script>` & JS `\"...\"` safe UTF-8 Base122 (`1.1429x` overhead)."""
    nbits = len(data) * 8
    pos = 0

    def next7() -> int | None:
        nonlocal pos
        if pos >= nbits:
            return None
        byte_idx = pos >> 3
        bit_off = pos & 7
        rem = nbits - pos
        val = (data[byte_idx] << 8) | (data[byte_idx + 1] if byte_idx + 1 < len(data) else 0)
        chunk = (val >> (9 - bit_off)) & 0x7F
        if rem < 7:
            chunk &= ~((1 << (7 - rem)) - 1)
        pos += 7
        return chunk

    chars: list[str] = []
    while (c1 := next7()) is not None:
        if c1 in B122_ILLEGAL_MAP:
            c2 = next7()
            cp = 0x100 + ((B122_SHORTENED << 7) + c1 if c2 is None else (B122_ILLEGAL_MAP[c1] << 7) + c2)
            chars.append(chr(cp))
        else:
            chars.append(chr(c1))
    return f"{len(data)}:" + "".join(chars)


def decode_base122(text: str) -> bytes:
    """Decodes a UTF-8 Base122 string back into exact original bytes."""
    colon = text.index(":")
    out_len = int(text[:colon])
    out = bytearray(out_len)
    bit_pos = 0

    def push7(v: int) -> None:
        nonlocal bit_pos
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        if byte_idx < out_len:
            shift = 1 - bit_off
            if shift >= 0:
                out[byte_idx] |= (v << shift) & 0xFF
            else:
                out[byte_idx] |= v >> (-shift)
                if byte_idx + 1 < out_len:
                    out[byte_idx + 1] |= (v << (8 + shift)) & 0xFF
        bit_pos += 7

    for ch in text[colon + 1 :]:
        cp = ord(ch)
        if cp < 0x80:
            push7(cp)
        else:
            payload = cp - 0x100
            idx = payload >> 7
            v2 = payload & 0x7F
            if idx == B122_SHORTENED:
                push7(v2)
            else:
                push7(B122_ILLEGALS[idx])
                push7(v2)
    return bytes(out)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_js_parameter_module(path: Path, lzma_bytes: bytes) -> int:
    """Writes compressed `MDP4/LZMA` bytes as a pure-text UTF-8 Base122 ES module (`export default \"...\"`)."""
    b122 = encode_base122(lzma_bytes)
    assert decode_base122(b122) == lzma_bytes
    js_text = f'export default "{b122}";\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_utf8 = js_text.encode("utf-8")
    path.write_bytes(raw_utf8)
    return len(raw_utf8)
