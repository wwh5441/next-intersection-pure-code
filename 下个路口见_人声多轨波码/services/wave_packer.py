"""V7 Multi-Packet Lossless Entropy Packer (`VWP7` ZigZag Cosine + `VWAP` Air + `VWRP` FDN Reverb) + UTF-8 Base122."""
from __future__ import annotations

import hashlib
import lzma
from pathlib import Path
import struct
import numpy as np

VWQ6_HDR_FMT = "<4sIHHIIBBBBBBHI"
VWQ6_HDR_SIZE = 32
VWP7_HDR_FMT = "<4sIHHIIBBBBBBHII"
VWP7_HDR_SIZE = 36
MODE_FOUR_SHORT = 2

LONG_BAND_WIDTHS = [
    16, 16, 16, 16, 20, 20, 24, 24, 28, 32, 36, 40,
    48, 56, 64, 76, 88, 104, 124, 148, 180, 220, 288, 364,
]
SHORT_BAND_WIDTHS = [8, 8, 12, 12, 16, 20, 28, 36, 48, 68, 104, 152]
EDGES_L = np.concatenate(([0], np.cumsum(LONG_BAND_WIDTHS))).astype(np.int32)
EDGES_S = np.concatenate(([0], np.cumsum(SHORT_BAND_WIDTHS))).astype(np.int32)

B122_ILLEGALS = (0, 10, 13, 34, 60, 92)
B122_ILLEGAL_MAP = {b: i for i, b in enumerate(B122_ILLEGALS)}
B122_SHORTENED = 7


def pack_vwp6_lzma(raw: bytes) -> bytes:
    """Losslessly packs `VWQ6` fixed-stride cosine wave parameters into `VWP7` active-band ZigZag stream + LZMA."""
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
    ) = struct.unpack(VWQ6_HDR_FMT, raw[:VWQ6_HDR_SIZE])
    assert magic == b"VWQ6"
    stride = 1 + 48 + long_n
    body = np.frombuffer(raw, np.uint8, offset=VWQ6_HDR_SIZE).reshape(t_count, stride)

    modes = body[:, 0]
    scales = body[:, 1:49]
    q_mat = body[:, 49:].copy().view(np.int8)

    active_frame = np.any(scales > 0, axis=1)
    act_indices = np.flatnonzero(active_frame)
    act_modes = modes[act_indices].tobytes()

    delta_chunks: list[bytes] = []
    present_bits_list: list[np.ndarray] = []
    nz_vals_list: list[np.ndarray] = []

    prev_long = np.zeros(b_long, dtype=np.int32)
    for t in act_indices:
        m = int(modes[t])
        q_row = q_mat[t]
        if m != MODE_FOUR_SHORT:
            sc = scales[t, :b_long].astype(np.int32)
            diff = ((sc - prev_long) & 0xFF).astype(np.uint8)
            delta_chunks.append(diff.tobytes())
            prev_long = sc
            for g in range(b_long):
                if sc[g] > 0:
                    sub = q_row[EDGES_L[g] : EDGES_L[g + 1]]
                    p = sub != 0
                    present_bits_list.append(p)
                    if np.any(p):
                        nz_vals_list.append(sub[p])
        else:
            s4 = scales[t].astype(np.int32).reshape(4, b_short)
            d4 = np.empty_like(s4)
            d4[0] = s4[0]
            d4[1:] = (s4[1:] - s4[:-1]) & 0xFF
            delta_chunks.append(d4.astype(np.uint8).tobytes())
            for sub_i in range(4):
                off = sub_i * short_n
                for g in range(b_short):
                    if s4[sub_i, g] > 0:
                        sub = q_row[off + EDGES_S[g] : off + EDGES_S[g + 1]]
                        p = sub != 0
                        present_bits_list.append(p)
                        if np.any(p):
                            nz_vals_list.append(sub[p])

    all_present = np.concatenate(present_bits_list) if present_bits_list else np.zeros(0, dtype=bool)
    all_nz = np.concatenate(nz_vals_list) if nz_vals_list else np.zeros(0, dtype=np.int8)
    zz = ((np.abs(all_nz).astype(np.uint16) << 1) - (all_nz > 0).astype(np.uint16)).astype(np.uint8)

    header_out = struct.pack(
        VWP7_HDR_FMT,
        b"VWP7",
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
        int(len(all_present)),
        int(len(all_nz)),
    )
    payload = (
        header_out
        + np.packbits(active_frame).tobytes()
        + act_modes
        + b"".join(delta_chunks)
        + np.packbits(all_present).tobytes()
        + zz.tobytes()
    )
    encoded = lzma.compress(payload, format=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)
    assert unpack_vwp6_lzma(encoded) == raw
    return encoded


def unpack_vwp6_lzma(encoded: bytes) -> bytes:
    """Decompresses LZMA `VWP7` / `VWAP` / `VWRP` stream back into bit-identical runtime buffer (`VWQ6` / `VWA7` / `VWR7`)."""
    raw = lzma.decompress(encoded, format=lzma.FORMAT_ALONE)
    magic = raw[:4]

    if magic == b"VWP7":
        (
            _,
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
            present_count,
            nonzero,
        ) = struct.unpack(VWP7_HDR_FMT, raw[:VWP7_HDR_SIZE])

        offset = VWP7_HDR_SIZE
        act_bytes = (t_count + 7) // 8
        active_frame = np.unpackbits(np.frombuffer(raw, np.uint8, offset=offset, count=act_bytes))[:t_count].astype(bool)
        offset += act_bytes
        act_indices = np.flatnonzero(active_frame)
        n_act = len(act_indices)

        act_modes = np.frombuffer(raw, np.uint8, offset=offset, count=n_act)
        offset += n_act

        modes = np.zeros(t_count, dtype=np.uint8)
        modes[act_indices] = act_modes
        scales = np.zeros((t_count, 48), dtype=np.uint8)

        prev_long = np.zeros(b_long, dtype=np.int32)
        for idx_i, t in enumerate(act_indices):
            m = int(act_modes[idx_i])
            if m != MODE_FOUR_SHORT:
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

        pres_bytes = (present_count + 7) // 8
        present = (
            np.unpackbits(np.frombuffer(raw, np.uint8, offset=offset, count=pres_bytes))[:present_count].astype(bool)
            if present_count > 0
            else np.zeros(0, dtype=bool)
        )
        offset += pres_bytes

        zz = np.frombuffer(raw, np.uint8, offset=offset, count=nonzero).astype(np.int16)
        offset += nonzero
        assert offset == len(raw), f"Trailing bytes in VWP7: {offset} != {len(raw)}"

        mag = (zz + 1) >> 1
        nz_vals = np.where((zz & 1) == 1, mag, -mag).astype(np.int8)

        q_mat = np.zeros((t_count, long_n), dtype=np.int8)
        p_cur = 0
        nz_cur = 0
        for idx_i, t in enumerate(act_indices):
            m = int(act_modes[idx_i])
            if m != MODE_FOUR_SHORT:
                for g in range(b_long):
                    if scales[t, g] > 0:
                        s_k, e_k = int(EDGES_L[g]), int(EDGES_L[g + 1])
                        w = e_k - s_k
                        p_sub = present[p_cur : p_cur + w]
                        p_cur += w
                        cnt = int(np.sum(p_sub))
                        if cnt > 0:
                            q_mat[t, s_k:e_k][p_sub] = nz_vals[nz_cur : nz_cur + cnt]
                            nz_cur += cnt
            else:
                for sub_i in range(4):
                    off_k = sub_i * short_n
                    sc_off = sub_i * b_short
                    for g in range(b_short):
                        if scales[t, sc_off + g] > 0:
                            s_k = off_k + int(EDGES_S[g])
                            e_k = off_k + int(EDGES_S[g + 1])
                            w = e_k - s_k
                            p_sub = present[p_cur : p_cur + w]
                            p_cur += w
                            cnt = int(np.sum(p_sub))
                            if cnt > 0:
                                q_mat[t, s_k:e_k][p_sub] = nz_vals[nz_cur : nz_cur + cnt]
                                nz_cur += cnt

        assert p_cur == present_count and nz_cur == nonzero
        header_vwq6 = struct.pack(
            VWQ6_HDR_FMT,
            b"VWQ6",
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
        body = np.concatenate([modes[:, None], scales, q_mat.view(np.uint8)], axis=1)
        return header_vwq6 + body.tobytes()

    if magic == b"VWAP":
        _, sr, samples, hop, n_hops, *rest = struct.unpack("<4sIIII8HI", raw[:40])
        act_count = rest[-1]
        offset = 40
        hop_bytes = (n_hops + 7) // 8
        active_hop = np.unpackbits(np.frombuffer(raw, np.uint8, offset=offset, count=hop_bytes))[:n_hops].astype(bool)
        offset += hop_bytes
        codes = np.zeros((n_hops, 4), dtype=np.uint8)
        if act_count > 0:
            deltas_T = np.frombuffer(raw, np.uint8, offset=offset, count=4 * act_count).reshape(4, act_count).T
            offset += 4 * act_count
            act_codes = (np.cumsum(deltas_T.astype(np.int32), axis=0) & 0xFF).astype(np.uint8)
            codes[active_hop] = act_codes
        assert offset == len(raw)
        return b"VWA7" + raw[4:36] + codes.tobytes()

    if magic == b"VWRP":
        _, sr, samples, env_hop, n_hops, *_ = struct.unpack("<4sIIIIHHHHI", raw[:32])
        dm = np.frombuffer(raw, np.uint8, offset=32, count=n_hops).astype(np.int32)
        ds = np.frombuffer(raw, np.uint8, offset=32 + n_hops, count=n_hops).astype(np.int32)
        assert 32 + 2 * n_hops == len(raw)
        cm = (np.cumsum(dm) & 0xFF).astype(np.uint8)
        cs = (np.cumsum(ds) & 0xFF).astype(np.uint8)
        return b"VWR7" + raw[4:32] + cm.tobytes() + cs.tobytes()

    raise ValueError(f"Unsupported packet magic: {magic!r}")


def encode_base122(data: bytes) -> str:
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


def write_b122_js_module(path: Path, lzma_bytes: bytes) -> int:
    b122 = encode_base122(lzma_bytes)
    assert decode_base122(b122) == lzma_bytes
    js_text = f'export default "{b122}";\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_utf8 = js_text.encode("utf-8")
    path.write_bytes(raw_utf8)
    return len(raw_utf8)
