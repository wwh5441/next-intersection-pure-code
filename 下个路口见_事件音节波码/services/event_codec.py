"""Columnar Event-Syllable Binary Packer (`VGS8`/`VGA8`/`VGE8`/`VGR8`), LZMA1 & Base122 Codec (V8.4)."""
from __future__ import annotations

import hashlib
import lzma
from pathlib import Path
import struct
import numpy as np

from models.vocal_structures import SpeakerTimbreCore, SyllablePrototype, VocalEvent
from .syllable_atlas_builder import (
    build_adaptive_expr_knot_positions,
    build_envelope_knot_positions,
    build_uniform_knot_positions,
)


def zz_enc_scalar(v: int) -> int:
    return (v << 1) ^ (v >> 15) if v < 0 else (v << 1)


def zz_dec_scalar(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def pack_speaker_core(speaker: SpeakerTimbreCore, sr: int, n_samples: int, hop: int, ref_peak: float) -> bytes:
    n_cep = len(speaker.mean_mcep)
    n_harm = len(speaker.glottal_phase)
    n_aper = len(speaker.aper_mean)
    k_mcep = len(speaker.mcep_pca_basis)
    k_aper = len(speaker.aper_pca_basis)

    hdr = struct.pack(
        "<4sIIIfHHHHHH",
        b"VGS8",
        sr,
        n_samples,
        hop,
        float(ref_peak),
        n_cep,
        n_harm,
        n_aper,
        k_mcep,
        k_aper,
        0,
    )
    mean_m_q = np.clip(np.round(speaker.mean_mcep * 256.0), -32767, 32767).astype("<i2")
    glott_q = np.clip(np.round(speaker.glottal_phase * 2048.0), -32767, 32767).astype("<i2")
    m_basis_q = np.clip(np.round(speaker.mcep_pca_basis.reshape(-1) * 2048.0), -32767, 32767).astype("<i2")
    aper_m_q = np.clip(np.round(speaker.aper_mean * 256.0), -32767, 32767).astype("<i2")
    a_basis_q = np.clip(np.round(speaker.aper_pca_basis.reshape(-1) * 2048.0), -32767, 32767).astype("<i2")
    return hdr + mean_m_q.tobytes() + glott_q.tobytes() + m_basis_q.tobytes() + aper_m_q.tobytes() + a_basis_q.tobytes()


def unpack_speaker_core(raw: bytes) -> tuple[SpeakerTimbreCore, dict]:
    magic, sr, n_samples, hop, ref_peak, n_cep, n_harm, n_aper, k_mcep, k_aper, _ = struct.unpack_from(
        "<4sIIIfHHHHHH", raw, 0
    )
    assert magic == b"VGS8"
    off = 32
    mean_mcep = np.frombuffer(raw, dtype="<i2", count=n_cep, offset=off).astype(np.float32) / 256.0
    off += n_cep * 2
    glottal_phase = np.frombuffer(raw, dtype="<i2", count=n_harm, offset=off).astype(np.float32) / 2048.0
    off += n_harm * 2
    mcep_pca_basis = (
        np.frombuffer(raw, dtype="<i2", count=k_mcep * n_cep, offset=off)
        .astype(np.float32)
        .reshape(k_mcep, n_cep)
        / 2048.0
    )
    off += k_mcep * n_cep * 2
    aper_mean = np.frombuffer(raw, dtype="<i2", count=n_aper, offset=off).astype(np.float32) / 256.0
    off += n_aper * 2
    aper_pca_basis = (
        np.frombuffer(raw, dtype="<i2", count=k_aper * n_aper, offset=off)
        .astype(np.float32)
        .reshape(k_aper, n_aper)
        / 2048.0
    )
    meta = {"sampleRate": sr, "numSamples": n_samples, "hop": hop, "refPeak": float(ref_peak)}
    return (
        SpeakerTimbreCore(
            mean_mcep=mean_mcep,
            glottal_phase=glottal_phase,
            mcep_pca_basis=mcep_pca_basis,
            aper_mean=aper_mean,
            aper_pca_basis=aper_pca_basis,
        ),
        meta,
    )


def pack_syllable_atlas(prototypes: list[SyllablePrototype], atlas_q_step: float = 0.22) -> bytes:
    n_protos = len(prototypes)
    n_cep = prototypes[0].vowel_mcep_knots.shape[1]
    n_aper = prototypes[0].cons_aper_knots.shape[1]

    cons_m = np.stack([p.cons_mcep_knots for p in prototypes], axis=0)   # (P, 3, C)
    cons_a = np.stack([p.cons_aper_knots for p in prototypes], axis=0)   # (P, 3, B_a)
    vow_m = np.stack([p.vowel_mcep_knots for p in prototypes], axis=0)   # (P, 5, C)
    vow_a = np.stack([p.vowel_aper_knots for p in prototypes], axis=0)   # (P, 5, B_a)

    max_abs = float(max(np.max(np.abs(cons_m)), np.max(np.abs(cons_a)), np.max(np.abs(vow_m)), np.max(np.abs(vow_a)), 1.0))
    eff_q_step = max(float(atlas_q_step), max_abs / 125.0)

    meta_str = "|".join(f"{p.pinyin}:{p.initial_class}:{p.count}" for p in prototypes).encode("utf-8")
    hdr = struct.pack("<4sHBBfH", b"VGA8", n_protos, n_cep, n_aper, float(eff_q_step), len(meta_str))

    cons_m_q = np.clip(np.round(cons_m / eff_q_step), -127, 127).astype(np.int8)
    cons_a_q = np.clip(np.round(cons_a / eff_q_step), -127, 127).astype(np.int8)
    vow_m_q = np.clip(np.round(vow_m / eff_q_step), -127, 127).astype(np.int8)
    vow_a_q = np.clip(np.round(vow_a / eff_q_step), -127, 127).astype(np.int8)

    return hdr + meta_str + cons_m_q.tobytes() + cons_a_q.tobytes() + vow_m_q.tobytes() + vow_a_q.tobytes()


def unpack_syllable_atlas(raw: bytes) -> list[SyllablePrototype]:
    magic, n_protos, n_cep, n_aper, atlas_q_step, meta_len = struct.unpack_from("<4sHBBfH", raw, 0)
    assert magic == b"VGA8"
    off = 14
    meta_items = raw[off : off + meta_len].decode("utf-8").split("|") if meta_len > 0 else []
    off += meta_len

    cm_cnt = n_protos * 3 * n_cep
    cons_m = (
        np.frombuffer(raw, dtype=np.int8, count=cm_cnt, offset=off)
        .astype(np.float32)
        .reshape(n_protos, 3, n_cep)
        * atlas_q_step
    )
    off += cm_cnt

    ca_cnt = n_protos * 3 * n_aper
    cons_a = (
        np.frombuffer(raw, dtype=np.int8, count=ca_cnt, offset=off)
        .astype(np.float32)
        .reshape(n_protos, 3, n_aper)
        * atlas_q_step
    )
    off += ca_cnt

    vm_cnt = n_protos * 5 * n_cep
    vow_m = (
        np.frombuffer(raw, dtype=np.int8, count=vm_cnt, offset=off)
        .astype(np.float32)
        .reshape(n_protos, 5, n_cep)
        * atlas_q_step
    )
    off += vm_cnt

    va_cnt = n_protos * 5 * n_aper
    vow_a = (
        np.frombuffer(raw, dtype=np.int8, count=va_cnt, offset=off)
        .astype(np.float32)
        .reshape(n_protos, 5, n_aper)
        * atlas_q_step
    )

    protos: list[SyllablePrototype] = []
    for sid in range(n_protos):
        py, icls, cnt_s = meta_items[sid].split(":")
        protos.append(
            SyllablePrototype(
                syllable_id=sid,
                pinyin=py,
                initial_class=icls,
                count=int(cnt_s),
                cons_mcep_knots=cons_m[sid],
                cons_aper_knots=cons_a[sid],
                vowel_mcep_knots=vow_m[sid],
                vowel_aper_knots=vow_a[sid],
            )
        )
    return protos


def pack_event_stream(events: list[VocalEvent], profile_spec: dict) -> bytes:
    n_ev = len(events)
    k_pca = int(profile_spec["pcaDims"])
    k_aper = int(profile_spec["aperExprBands"])
    expr_step = int(profile_spec["exprKnotStepFrames"])
    f0_step = int(profile_spec["f0KnotStepFrames"])
    env_step = int(profile_spec["envKnotStepFrames"])
    q_step = float(profile_spec["quantStepDb"])
    pf_beta_q = int(round(float(profile_spec.get("formantPostFilterBeta", 0.06)) * 200.0))

    disp_bytes = "\t".join(ev.display_text for ev in events).encode("utf-8")
    hdr = struct.pack(
        "<4sHBBBBBBfI",
        b"VGE8",
        n_ev,
        k_pca,
        k_aper,
        expr_step,
        f0_step,
        env_step,
        min(255, max(0, pf_beta_q)),
        q_step,
        len(disp_bytes),
    )

    gaps = np.zeros(n_ev, dtype="<u2")
    totals = np.zeros(n_ev, dtype="<u2")
    cons = np.zeros(n_ev, dtype=np.uint8)
    sids = np.zeros(n_ev, dtype=np.uint8)
    flags = np.zeros(n_ev, dtype=np.uint8)
    base_midi_zz = np.zeros(n_ev, dtype="<u2")
    pans = np.zeros(n_ev, dtype=np.int8)
    back_l = np.zeros(n_ev, dtype=np.uint8)
    back_r = np.zeros(n_ev, dtype=np.uint8)
    back_p = np.zeros(n_ev, dtype=np.int8)
    v_onsets = np.zeros(n_ev, dtype="<u2")

    f0_bytes = bytearray()
    gh_bytes = bytearray()
    ga_bytes = bytearray()
    mcep_cols = [bytearray() for _ in range(k_pca)]
    aper_cols = [bytearray() for _ in range(k_aper)]

    prev_end = 0
    prev_midi = 3000
    for i, ev in enumerate(events):
        gap = max(0, ev.start_frame - prev_end)
        gaps[i] = min(65535, gap)
        totals[i] = min(65535, ev.total_frames)
        cons[i] = min(255, ev.cons_frames)
        sids[i] = ev.syllable_id
        unvoiced_only = ev.voiced_onset_frame >= ev.total_frames
        fl = (
            (1 if ev.legato_prev else 0)
            | (2 if ev.legato_next else 0)
            | (4 if ev.is_hook else 0)
            | (8 if unvoiced_only else 0)
            | (16 if ev.is_plosive else 0)
        )
        flags[i] = fl
        dm = ev.base_midi_q - prev_midi
        base_midi_zz[i] = zz_enc_scalar(dm)
        if not unvoiced_only:
            prev_midi = ev.base_midi_q
        pans[i] = ev.stereo_pan_q
        back_l[i] = ev.back_gain_l_q
        back_r[i] = ev.back_gain_r_q
        back_p[i] = ev.back_pitch_q
        v_onsets[i] = min(65535, ev.voiced_onset_frame)
        prev_end = ev.start_frame + ev.total_frames

        if not unvoiced_only and len(ev.f0_step_q) > 0:
            fu = (ev.f0_step_q.astype(np.int32) + 128).astype(np.uint8)
            f0_bytes.extend(fu.tobytes())

        gh_u8 = ev.gain_h_db_q
        d_gh = np.empty_like(gh_u8)
        d_gh[0] = gh_u8[0]
        if len(gh_u8) > 1:
            d_gh[1:] = (gh_u8[1:].astype(np.int32) - gh_u8[:-1].astype(np.int32)) & 0xFF
        gh_bytes.extend(d_gh.tobytes())

        ga_u8 = ev.gain_a_db_q
        d_ga = np.empty_like(ga_u8)
        d_ga[0] = ga_u8[0]
        if len(ga_u8) > 1:
            d_ga[1:] = (ga_u8[1:].astype(np.int32) - ga_u8[:-1].astype(np.int32)) & 0xFF
        ga_bytes.extend(d_ga.tobytes())

        em_u8 = (ev.expr_mcep_q.astype(np.int32) + 128).astype(np.uint8)
        for m in range(k_pca):
            col = em_u8[:, m]
            dc = np.empty_like(col)
            dc[0] = col[0]
            if len(col) > 1:
                dc[1:] = (col[1:].astype(np.int32) - col[:-1].astype(np.int32)) & 0xFF
            mcep_cols[m].extend(dc.tobytes())

        ea_u8 = (ev.expr_aper_q.astype(np.int32) + 128).astype(np.uint8)
        for m in range(k_aper):
            col = ea_u8[:, m]
            dc = np.empty_like(col)
            dc[0] = col[0]
            if len(col) > 1:
                dc[1:] = (col[1:].astype(np.int32) - col[:-1].astype(np.int32)) & 0xFF
            aper_cols[m].extend(dc.tobytes())

    parts = [
        hdr,
        disp_bytes,
        gaps.tobytes(),
        totals.tobytes(),
        cons.tobytes(),
        sids.tobytes(),
        flags.tobytes(),
        base_midi_zz.tobytes(),
        pans.tobytes(),
        back_l.tobytes(),
        back_r.tobytes(),
        back_p.tobytes(),
        v_onsets.tobytes(),
        bytes(f0_bytes),
        bytes(gh_bytes),
        bytes(ga_bytes),
    ]
    for m in range(k_pca):
        parts.append(bytes(mcep_cols[m]))
    for m in range(k_aper):
        parts.append(bytes(aper_cols[m]))
    return b"".join(parts)


def unpack_event_stream(raw: bytes) -> tuple[list[VocalEvent], dict]:
    magic, n_ev, k_pca, k_aper, expr_step, f0_step, env_step, pf_beta_q, q_step, disp_len = struct.unpack_from(
        "<4sHBBBBBBfI", raw, 0
    )
    assert magic == b"VGE8"
    off = 20
    disp_list = raw[off : off + disp_len].decode("utf-8").split("\t") if disp_len > 0 else [""] * n_ev
    off += disp_len

    gaps = np.frombuffer(raw, dtype="<u2", count=n_ev, offset=off)
    off += n_ev * 2
    totals = np.frombuffer(raw, dtype="<u2", count=n_ev, offset=off)
    off += n_ev * 2
    cons = np.frombuffer(raw, dtype=np.uint8, count=n_ev, offset=off)
    off += n_ev
    sids = np.frombuffer(raw, dtype=np.uint8, count=n_ev, offset=off)
    off += n_ev
    flags = np.frombuffer(raw, dtype=np.uint8, count=n_ev, offset=off)
    off += n_ev
    base_midi_zz = np.frombuffer(raw, dtype="<u2", count=n_ev, offset=off)
    off += n_ev * 2
    pans = np.frombuffer(raw, dtype=np.int8, count=n_ev, offset=off)
    off += n_ev
    back_l = np.frombuffer(raw, dtype=np.uint8, count=n_ev, offset=off)
    off += n_ev
    back_r = np.frombuffer(raw, dtype=np.uint8, count=n_ev, offset=off)
    off += n_ev
    back_p = np.frombuffer(raw, dtype=np.int8, count=n_ev, offset=off)
    off += n_ev
    v_onsets = np.frombuffer(raw, dtype="<u2", count=n_ev, offset=off)
    off += n_ev * 2

    f0_lens = []
    env_lens = []
    expr_lens = []
    for i in range(n_ev):
        L = int(totals[i])
        cf = int(cons[i])
        unv = bool(flags[i] & 8)
        f0_lens.append(0 if unv else max(0, len(build_uniform_knot_positions(L, f0_step)) - 1))
        env_lens.append(len(build_envelope_knot_positions(L, cf, env_step)))
        expr_lens.append(len(build_adaptive_expr_knot_positions(L, cf, expr_step)))

    tot_f0 = sum(f0_lens)
    tot_env = sum(env_lens)
    tot_expr = sum(expr_lens)

    f0_stream = np.frombuffer(raw, dtype=np.uint8, count=tot_f0, offset=off)
    off += tot_f0
    gh_stream = np.frombuffer(raw, dtype=np.uint8, count=tot_env, offset=off)
    off += tot_env
    ga_stream = np.frombuffer(raw, dtype=np.uint8, count=tot_env, offset=off)
    off += tot_env

    mcep_streams = []
    for _ in range(k_pca):
        mcep_streams.append(np.frombuffer(raw, dtype=np.uint8, count=tot_expr, offset=off))
        off += tot_expr
    aper_streams = []
    for _ in range(k_aper):
        aper_streams.append(np.frombuffer(raw, dtype=np.uint8, count=tot_expr, offset=off))
        off += tot_expr

    events: list[VocalEvent] = []
    prev_end = 0
    prev_midi = 3000
    c_f0 = 0
    c_env = 0
    c_expr = 0

    for i in range(n_ev):
        L = int(totals[i])
        cf = int(cons[i])
        fl = int(flags[i])
        unv = bool(fl & 8)
        s_frame = prev_end + int(gaps[i])
        prev_end = s_frame + L

        dm = zz_dec_scalar(int(base_midi_zz[i]))
        b_midi = prev_midi + dm
        if not unv:
            prev_midi = b_midi

        nf = f0_lens[i]
        if nf > 0:
            f0_q = (f0_stream[c_f0 : c_f0 + nf].astype(np.int32) - 128).astype(np.int8)
            c_f0 += nf
        else:
            f0_q = np.zeros(0, dtype=np.int8)

        ne = env_lens[i]
        gh_q = (np.cumsum(gh_stream[c_env : c_env + ne].astype(np.int32)) & 0xFF).astype(np.uint8)
        ga_q = (np.cumsum(ga_stream[c_env : c_env + ne].astype(np.int32)) & 0xFF).astype(np.uint8)
        c_env += ne

        nx = expr_lens[i]
        em_q = np.zeros((nx, k_pca), dtype=np.int8)
        for m in range(k_pca):
            cu = (np.cumsum(mcep_streams[m][c_expr : c_expr + nx].astype(np.int32)) & 0xFF).astype(np.int32)
            em_q[:, m] = (cu - 128).astype(np.int8)

        ea_q = np.zeros((nx, k_aper), dtype=np.int8)
        for m in range(k_aper):
            cu = (np.cumsum(aper_streams[m][c_expr : c_expr + nx].astype(np.int32)) & 0xFF).astype(np.int32)
            ea_q[:, m] = (cu - 128).astype(np.int8)
        c_expr += nx

        events.append(
            VocalEvent(
                event_index=i,
                start_frame=s_frame,
                cons_frames=cf,
                total_frames=L,
                syllable_id=int(sids[i]),
                display_text=disp_list[i],
                pinyin="",
                is_hook=bool(fl & 4),
                legato_prev=bool(fl & 1),
                legato_next=bool(fl & 2),
                is_plosive=bool(fl & 16),
                base_midi_q=b_midi,
                stereo_pan_q=int(pans[i]),
                back_gain_l_q=int(back_l[i]),
                back_gain_r_q=int(back_r[i]),
                back_pitch_q=int(back_p[i]),
                voiced_onset_frame=int(v_onsets[i]),
                f0_step_q=f0_q,
                gain_h_db_q=gh_q,
                gain_a_db_q=ga_q,
                expr_mcep_q=em_q,
                expr_aper_q=ea_q,
            )
        )

    header_meta = {
        "numEvents": n_ev,
        "pcaDims": k_pca,
        "aperExprBands": k_aper,
        "exprKnotStepFrames": expr_step,
        "f0KnotStepFrames": f0_step,
        "envKnotStepFrames": env_step,
        "formantPostFilterBeta": float(pf_beta_q) / 200.0,
        "quantStepDb": float(q_step),
    }
    return events, header_meta


def compress_lzma1_alone(raw_payload: bytes) -> bytes:
    """Packs `raw_payload` into standard LZMA_ALONE stream (13-byte header + LZMA1) compatible with JS `LZMA.decompress`."""
    props_dict = {"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": 1 << 20}
    comp = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=[props_dict])
    body = comp.compress(raw_payload) + comp.flush()
    alone_hdr = struct.pack("<BIQ", 0x5D, 1 << 20, len(raw_payload))
    return alone_hdr + body


def decompress_lzma1_alone(comp_bytes: bytes) -> bytes:
    props_byte, dict_size, uncomp_len = struct.unpack_from("<BIQ", comp_bytes, 0)
    lc = props_byte % 9
    rem = props_byte // 9
    lp = rem % 5
    pb = rem // 5
    dec = lzma.LZMADecompressor(
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "lc": lc, "lp": lp, "pb": pb, "dict_size": dict_size}],
    )
    out = dec.decompress(comp_bytes[13:])
    assert len(out) == uncomp_len, f"Uncompressed length mismatch: {len(out)} != {uncomp_len}"
    return out


_ILLEGAL = (0, 10, 13, 34, 38, 92)


def encode_base122(data: bytes) -> str:
    cur_index = 0
    cur_bit = 0
    total_bits = len(data) * 8
    out_chars: list[str] = []

    def get7() -> int | None:
        nonlocal cur_index, cur_bit
        if cur_index * 8 + cur_bit >= total_bits:
            return None
        b1 = data[cur_index]
        first_part = ((0xFE >> cur_bit) & b1) << cur_bit
        first_part >>= 1
        cur_bit += 7
        if cur_bit < 8:
            return first_part
        cur_bit -= 8
        cur_index += 1
        if cur_index >= len(data):
            return first_part
        b2 = data[cur_index]
        second_part = ((0xFF00 >> cur_bit) & b2) & 0xFF
        second_part >>= 8 - cur_bit
        return first_part | second_part

    while True:
        bits = get7()
        if bits is None:
            break
        if bits in _ILLEGAL:
            illegal_idx = _ILLEGAL.index(bits)
            next_bits = get7()
            b1 = 0b11000010
            b2 = 0b10000000
            if next_bits is None:
                b1 |= 0b00011100
                next_bits = bits
            else:
                b1 |= illegal_idx << 2
            first_bit = (next_bits >> 6) & 1
            b1 |= first_bit
            b2 |= next_bits & 0b00111111
            out_chars.append(chr(((b1 & 0x1F) << 6) | (b2 & 0x3F)))
        else:
            out_chars.append(chr(bits))
    return "".join(out_chars)


def decode_base122(str_data: str) -> bytes:
    out = bytearray()
    cur_byte = 0
    bit_of_byte = 0

    def push7(byte7: int) -> None:
        nonlocal cur_byte, bit_of_byte
        byte7 <<= 1
        cur_byte |= byte7 >> bit_of_byte
        bit_of_byte += 7
        if bit_of_byte >= 8:
            out.append(cur_byte & 0xFF)
            bit_of_byte -= 8
            cur_byte = (byte7 << (7 - bit_of_byte)) & 0xFF

    for ch in str_data:
        c = ord(ch)
        if c > 127:
            illegal_idx = (c >> 8) & 7
            if illegal_idx != 7:
                push7(_ILLEGAL[illegal_idx])
            push7(c & 127)
        else:
            push7(c)
    return bytes(out)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_b122_js_module(path: Path, comp_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b122 = encode_base122(comp_bytes)
    path.write_text(f'export default "{b122}";\n', encoding="utf-8")
