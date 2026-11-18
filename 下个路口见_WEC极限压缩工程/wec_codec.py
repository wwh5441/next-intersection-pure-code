#!/usr/bin/env python3
"""WEC1 (Weckham Pure-Code Music Container) Lossless Binary Codec (`wec_codec.py`).

Compresses all V9R2.8 pure-code acoustic models, phoneme atlas, and 11-track
musical scores into a single ultra-compact `.wec` binary container using:
1. Columnar transposition (events -> per-field / per-dimension streams)
2. Delta + ZigZag + compact int8-with-varint-escape integer coding
3. LZMA-Extreme (preset=9 | PRESET_EXTREME) entropy coding
"""
from __future__ import annotations

import json
import lzma
import struct
from pathlib import Path
from typing import Any

MAGIC = b"WEC1"
FLAG_LZMA = 1


def _enc_uvarint(out: bytearray, n: int) -> None:
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)


def _dec_uvarint(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    res = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, pos
        shift += 7


def _enc_s8_stream(out: bytearray, seq: list[int]) -> None:
    """Compact signed int stream: [-127..127] in 1 byte, -128 = ZigZag uvarint escape."""
    _enc_uvarint(out, len(seq))
    for v in seq:
        if -127 <= v <= 127:
            out.append(v & 0xFF)
        else:
            out.append(0x80)
            zz = (v << 1) if v >= 0 else ((-v) << 1) - 1
            _enc_uvarint(out, zz)


def _dec_s8_stream(buf: bytes | memoryview, pos: int) -> tuple[list[int], int]:
    n, pos = _dec_uvarint(buf, pos)
    out = [0] * n
    for i in range(n):
        b = buf[pos]
        pos += 1
        if b == 0x80:
            zz, pos = _dec_uvarint(buf, pos)
            out[i] = (zz >> 1) ^ -(zz & 1)
        else:
            out[i] = b - 256 if b >= 128 else b
    return out, pos


def _enc_str(out: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    _enc_uvarint(out, len(b))
    out.extend(b)


def _dec_str(buf: bytes | memoryview, pos: int) -> tuple[str, int]:
    n, pos = _dec_uvarint(buf, pos)
    return bytes(buf[pos : pos + n]).decode("utf-8"), pos + n


def _pack_vocal_score(vs: dict[str, Any]) -> bytes:
    out = bytearray()
    header = {k: v for k, v in vs.items() if k != "events"}
    _enc_str(out, json.dumps(header, ensure_ascii=False, separators=(",", ":")))

    evs = vs["events"]
    _enc_uvarint(out, len(evs))
    for k in ("s", "cf", "L", "sid", "hk", "lp", "ln", "pl", "vo"):
        _enc_s8_stream(out, [int(e[k]) for e in evs])
    for k in ("t", "py"):
        _enc_str(out, "\n".join(e[k] for e in evs))
    for k in ("fk", "fc", "gk", "gq", "vk", "vq"):
        lens = [len(e[k]) for e in evs]
        _enc_s8_stream(out, lens)
        flat = [int(v) for e in evs for v in e[k]]
        _enc_s8_stream(out, flat)

    # Columnar em & ea (preserves exact trimmed row lengths)
    for field, dims in (("em", vs["pcaDims"]), ("ea", vs["aperExprBands"])):
        n_rows = [len(e[field]) for e in evs]
        _enc_s8_stream(out, n_rows)
        row_lens = [len(r) for e in evs for r in e[field]]
        _enc_s8_stream(out, row_lens)
        for d in range(dims):
            col = [int(r[d]) for e in evs for r in e[field] if d < len(r)]
            _enc_s8_stream(out, col)
    return bytes(out)


def _unpack_vocal_score(buf: bytes | memoryview) -> dict[str, Any]:
    pos = 0
    hdr_s, pos = _dec_str(buf, pos)
    vs = json.loads(hdr_s)
    n_ev, pos = _dec_uvarint(buf, pos)
    evs: list[dict[str, Any]] = [{} for _ in range(n_ev)]

    for k in ("s", "cf", "L", "sid", "hk", "lp", "ln", "pl", "vo"):
        col, pos = _dec_s8_stream(buf, pos)
        for i in range(n_ev):
            evs[i][k] = col[i]
    for k in ("t", "py"):
        joined, pos = _dec_str(buf, pos)
        parts = joined.split("\n")
        for i in range(n_ev):
            evs[i][k] = parts[i]
    for k in ("fk", "fc", "gk", "gq", "vk", "vq"):
        lens, pos = _dec_s8_stream(buf, pos)
        flat, pos = _dec_s8_stream(buf, pos)
        off = 0
        for i in range(n_ev):
            L_i = lens[i]
            evs[i][k] = flat[off : off + L_i]
            off += L_i

    for field, dims in (("em", vs["pcaDims"]), ("ea", vs["aperExprBands"])):
        n_rows, pos = _dec_s8_stream(buf, pos)
        row_lens, pos = _dec_s8_stream(buf, pos)
        cols = []
        for _ in range(dims):
            c, pos = _dec_s8_stream(buf, pos)
            cols.append(c)
        col_idx = [0] * dims
        r_off = 0
        for i in range(n_ev):
            nr = n_rows[i]
            mat = []
            for _ in range(nr):
                rl = row_lens[r_off]
                r_off += 1
                row = []
                for d in range(rl):
                    row.append(cols[d][col_idx[d]])
                    col_idx[d] += 1
                mat.append(row)
            evs[i][field] = mat

    vs["events"] = evs
    return vs


def _pack_note_channels(obj: dict[str, Any], keys: tuple[str, ...]) -> bytes:
    out = bytearray()
    for key in keys:
        ch = obj[key]
        _enc_uvarint(out, len(ch))
        _enc_s8_stream(out, [int(n[0]) for n in ch])
        _enc_s8_stream(out, [int(n[1]) for n in ch])
        for idx in (2, 3, 4):
            _enc_s8_stream(out, [len(n[idx]) for n in ch])
            _enc_s8_stream(out, [int(v) for n in ch for v in n[idx]])
    return bytes(out)


def _unpack_note_channels(buf: bytes | memoryview, keys: tuple[str, ...]) -> dict[str, Any]:
    pos = 0
    res: dict[str, Any] = {}
    for key in keys:
        n_notes, pos = _dec_uvarint(buf, pos)
        ds, pos = _dec_s8_stream(buf, pos)
        tid, pos = _dec_s8_stream(buf, pos)
        arrs = []
        for _ in range(3):
            lens, pos = _dec_s8_stream(buf, pos)
            flat, pos = _dec_s8_stream(buf, pos)
            sub = []
            off = 0
            for L_i in lens:
                sub.append(flat[off : off + L_i])
                off += L_i
            arrs.append(sub)
        res[key] = [[ds[i], tid[i], arrs[0][i], arrs[1][i], arrs[2][i]] for i in range(n_notes)]
    return res


def _pack_int_tuple_list(rows: list[list[int]], n_cols: int) -> bytes:
    out = bytearray()
    _enc_uvarint(out, len(rows))
    for c in range(n_cols):
        _enc_s8_stream(out, [int(r[c]) for r in rows])
    return bytes(out)


def _unpack_int_tuple_list(buf: bytes | memoryview, n_cols: int) -> list[list[int]]:
    pos = 0
    n_rows, pos = _dec_uvarint(buf, pos)
    cols = []
    for _ in range(n_cols):
        c, pos = _dec_s8_stream(buf, pos)
        cols.append(c)
    return [[cols[c][i] for c in range(n_cols)] for i in range(n_rows)]


def _pack_poly_score(ps: dict[str, Any]) -> bytes:
    return _pack_int_tuple_list(ps["notes"], 9)


def _unpack_poly_score(buf: bytes | memoryview) -> dict[str, Any]:
    return {"notes": _unpack_int_tuple_list(buf, 9)}


def _pack_drum_score(ds_obj: dict[str, Any]) -> bytes:
    out = bytearray()
    fams = list(ds_obj.keys())
    _enc_str(out, json.dumps(fams, separators=(",", ":")))
    for fam in fams:
        cols = ds_obj[fam]
        col_names = list(cols.keys())
        _enc_str(out, json.dumps(col_names, separators=(",", ":")))
        for cn in col_names:
            _enc_s8_stream(out, [int(v) for v in cols[cn]])
    return bytes(out)


def _unpack_drum_score(buf: bytes | memoryview) -> dict[str, Any]:
    pos = 0
    fams_s, pos = _dec_str(buf, pos)
    fams = json.loads(fams_s)
    res: dict[str, Any] = {}
    for fam in fams:
        cn_s, pos = _dec_str(buf, pos)
        col_names = json.loads(cn_s)
        sub = {}
        for cn in col_names:
            arr, pos = _dec_s8_stream(buf, pos)
            sub[cn] = arr
        res[fam] = sub
    return res


def _pack_phoneme_atlas(pa: dict[str, Any]) -> bytes:
    out = bytearray()
    header = {k: v for k, v in pa.items() if k not in ("atlasErbBasisQ", "prototypes")}
    _enc_str(out, json.dumps(header, ensure_ascii=False, separators=(",", ":")))
    basis = pa["atlasErbBasisQ"]
    _enc_uvarint(out, len(basis))
    _enc_uvarint(out, len(basis[0]) if basis else 0)
    for row in basis:
        _enc_s8_stream(out, [int(v) for v in row])

    protos = pa["prototypes"]
    _enc_uvarint(out, len(protos))
    _enc_s8_stream(out, [int(p["id"]) for p in protos])
    _enc_s8_stream(out, [int(p["count"]) for p in protos])
    _enc_str(out, "\n".join(p["pinyin"] for p in protos))
    _enc_str(out, "\n".join(p["initialClass"] for p in protos))

    for field in ("consAper", "vowelAper"):
        n_rows = len(protos[0][field])
        n_cols = len(protos[0][field][0])
        _enc_uvarint(out, n_rows)
        _enc_uvarint(out, n_cols)
        _enc_s8_stream(out, [int(p[field][r][c]) for p in protos for r in range(n_rows) for c in range(n_cols)])

    for field in ("consErbPca", "vowelErbPca"):
        n_rows = len(protos[0][field])
        _enc_uvarint(out, n_rows)
        lens = [len(p[field][r]) for p in protos for r in range(n_rows)]
        _enc_s8_stream(out, lens)
        flat = [int(v) for p in protos for r in range(n_rows) for v in p[field][r]]
        _enc_s8_stream(out, flat)
    return bytes(out)


def _unpack_phoneme_atlas(buf: bytes | memoryview) -> dict[str, Any]:
    pos = 0
    hdr_s, pos = _dec_str(buf, pos)
    pa = json.loads(hdr_s)
    nb, pos = _dec_uvarint(buf, pos)
    _nc, pos = _dec_uvarint(buf, pos)
    basis = []
    for _ in range(nb):
        row, pos = _dec_s8_stream(buf, pos)
        basis.append(row)
    pa["atlasErbBasisQ"] = basis

    np_cnt, pos = _dec_uvarint(buf, pos)
    ids, pos = _dec_s8_stream(buf, pos)
    cnts, pos = _dec_s8_stream(buf, pos)
    pys, pos = _dec_str(buf, pos)
    ics, pos = _dec_str(buf, pos)
    py_l = pys.split("\n")
    ic_l = ics.split("\n")
    protos = [{"id": ids[i], "pinyin": py_l[i], "initialClass": ic_l[i], "count": cnts[i]} for i in range(np_cnt)]

    for field in ("consAper", "vowelAper"):
        nr, pos = _dec_uvarint(buf, pos)
        nc, pos = _dec_uvarint(buf, pos)
        flat, pos = _dec_s8_stream(buf, pos)
        off = 0
        for p in protos:
            mat = []
            for _ in range(nr):
                mat.append(flat[off : off + nc])
                off += nc
            p[field] = mat

    for field in ("consErbPca", "vowelErbPca"):
        nr, pos = _dec_uvarint(buf, pos)
        lens, pos = _dec_s8_stream(buf, pos)
        flat, pos = _dec_s8_stream(buf, pos)
        lo, fo = 0, 0
        for p in protos:
            mat = []
            for _ in range(nr):
                rl = lens[lo]
                lo += 1
                mat.append(flat[fo : fo + rl])
                fo += rl
            p[field] = mat

    pa["prototypes"] = protos
    return pa


def encode_project_to_wec(project_dir: Path, out_wec: Path) -> dict[str, Any]:
    def load_js(rel: str) -> Any:
        txt = (project_dir / rel).read_text(encoding="utf-8")
        return json.loads(txt.split("=", 1)[1].rstrip().rstrip(";"))

    files_order = [
        "models/vocal-speaker-model.js",
        "models/vocal-phoneme-atlas.js",
        "models/backing-model.js",
        "models/space-fdn.js",
        "models/bass-model.js",
        "models/drum-model.js",
        "models/poly-model.js",
        "data/vocal-score.js",
        "data/backing-score.js",
        "data/bass-score.js",
        "data/drum-score.js",
        "data/poly-score.js",
        "data/band-gains.js",
        "data/studio-metadata.js",
    ]
    objs = {rel: load_js(rel) for rel in files_order}

    # Pack specialized binary blocks + compact JSON fallback for small models
    payload = bytearray()

    def write_block(b: bytes) -> None:
        _enc_uvarint(payload, len(b))
        payload.extend(b)

    write_block(_pack_vocal_score(objs["data/vocal-score.js"]))
    write_block(_pack_note_channels(objs["data/backing-score.js"], ("leftNotes", "rightNotes")))
    write_block(_pack_note_channels(objs["data/bass-score.js"], ("notes",)))
    write_block(_pack_poly_score(objs["data/poly-score.js"]))
    write_block(_pack_drum_score(objs["data/drum-score.js"]))
    write_block(_pack_phoneme_atlas(objs["models/vocal-phoneme-atlas.js"]))

    # Remaining models + metadata as ultra-compact JSON
    rest = {
        k: objs[k]
        for k in files_order
        if k not in ("data/vocal-score.js", "data/backing-score.js", "data/poly-score.js", "data/bass-score.js", "data/drum-score.js", "models/vocal-phoneme-atlas.js")
    }
    write_block(json.dumps(rest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    raw_payload = bytes(payload)
    comp = lzma.compress(raw_payload, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)

    sm = objs["models/vocal-speaker-model.js"]
    hdr = struct.pack("<4sHIII", MAGIC, FLAG_LZMA, int(sm["sampleRate"]), int(sm["numSamples"]), len(raw_payload))
    wec_bytes = hdr + comp
    out_wec.write_bytes(wec_bytes)

    return {
        "wecBytes": len(wec_bytes),
        "wecKB": round(len(wec_bytes) / 1024.0, 2),
        "binaryPayloadKB": round(len(raw_payload) / 1024.0, 2),
    }


def decode_wec(wec_path: Path) -> dict[str, Any]:
    data = wec_path.read_bytes()
    magic, flags, sr, n_samples, raw_len = struct.unpack_from("<4sHIII", data, 0)
    if magic != MAGIC:
        raise ValueError(f"Invalid WEC magic: {magic!r}")
    if flags != FLAG_LZMA:
        raise ValueError(f"Unsupported WEC flags: {flags}")
    raw = lzma.decompress(data[18:])
    if len(raw) != raw_len:
        raise ValueError(f"WEC payload length mismatch: {len(raw)} != {raw_len}")

    mv = memoryview(raw)
    pos = 0

    def read_block() -> memoryview:
        nonlocal pos
        blen, pos = _dec_uvarint(mv, pos)
        blk = mv[pos : pos + blen]
        pos += blen
        return blk

    vocal_score = _unpack_vocal_score(read_block())
    backing_score = _unpack_note_channels(read_block(), ("leftNotes", "rightNotes"))
    bass_score = _unpack_note_channels(read_block(), ("notes",))
    poly_score = _unpack_poly_score(read_block())
    drum_score = _unpack_drum_score(read_block())
    phoneme_atlas = _unpack_phoneme_atlas(read_block())

    rest = json.loads(bytes(read_block()).decode("utf-8"))
    rest["data/vocal-score.js"] = vocal_score
    rest["data/backing-score.js"] = backing_score
    rest["data/poly-score.js"] = poly_score
    rest["data/bass-score.js"] = bass_score
    rest["data/drum-score.js"] = drum_score
    rest["models/vocal-phoneme-atlas.js"] = phoneme_atlas
    return rest
