"""LRC Timestamp Parser & Mandarin/English Phonetic Syllable Tokenizer (`parsers/lrc_syllable_parser.py`)."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

_TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_CREDIT_KEYWORDS = ("作词", "作曲", "编曲", "制作", "混音")


@dataclass(slots=True)
class SyllableToken:
    display: str
    pinyin: str
    initial_class: str


def load_phoneme_lexicon(schema_path: str | Path) -> dict:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def classify_initial(pinyin: str, lexicon: dict) -> str:
    py = pinyin.lower()
    for p in lexicon["sibilantPrefixes"]:
        if py.startswith(p):
            return "sibilant"
    for p in lexicon["plosivePrefixes"]:
        if py.startswith(p):
            return "plosive"
    for p in lexicon["sonorantPrefixes"]:
        if py.startswith(p):
            return "sonorant"
    return "glide"


def tokenize_lyric_text(text: str, lexicon: dict) -> list[SyllableToken]:
    s = text.strip()
    tokens: list[SyllableToken] = []
    compounds = lexicon.get("specialCompounds", {})
    char_map = lexicon["charToPinyin"]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        matched_compound = False
        for key, items in compounds.items():
            if s.startswith(key, i):
                for item in items:
                    py = item["pinyin"]
                    tokens.append(
                        SyllableToken(
                            display=item["display"],
                            pinyin=py,
                            initial_class=classify_initial(py, lexicon),
                        )
                    )
                i += len(key)
                matched_compound = True
                break
        if matched_compound:
            continue

        m = re.match(r"[A-Za-z]+", s[i:])
        if m:
            word = m.group(0)
            py = word.lower()
            tokens.append(
                SyllableToken(
                    display=word,
                    pinyin=py,
                    initial_class=classify_initial(py, lexicon),
                )
            )
            i += len(word)
            continue

        if ch in char_map:
            py = char_map[ch]
            tokens.append(
                SyllableToken(
                    display=ch,
                    pinyin=py,
                    initial_class=classify_initial(py, lexicon),
                )
            )
        elif "\u4e00" <= ch <= "\u9fff":
            tokens.append(SyllableToken(display=ch, pinyin="a", initial_class="glide"))
        i += 1
    return tokens


def parse_lrc_with_tokens(
    lrc_path: str | Path, lexicon: dict, total_duration: float
) -> tuple[list[dict], list[dict]]:
    raw_entries: list[tuple[float, str]] = []
    for raw_line in Path(lrc_path).read_text(encoding="utf-8").splitlines():
        matches = list(_TIME_RE.finditer(raw_line))
        if not matches:
            continue
        txt = _TIME_RE.sub("", raw_line).strip()
        for m in matches:
            sec = int(m.group(1)) * 60.0 + float(m.group(2))
            raw_entries.append((sec, txt))
    raw_entries.sort(key=lambda x: x[0])

    all_lines: list[dict] = []
    sung_lines: list[dict] = []
    for idx, (start_sec, txt) in enumerate(raw_entries):
        end_sec = raw_entries[idx + 1][0] if idx + 1 < len(raw_entries) else total_duration
        all_lines.append(
            {
                "index": idx,
                "startSec": round(start_sec, 3),
                "endSec": round(end_sec, 3),
                "text": txt,
            }
        )
        if not txt or any(k in txt for k in _CREDIT_KEYWORDS):
            continue
        tokens = tokenize_lyric_text(txt, lexicon)
        if tokens:
            phrase_groups = [
                sub_toks
                for part in txt.split()
                if (sub_toks := tokenize_lyric_text(part, lexicon))
            ]
            sung_lines.append(
                {
                    "lineIndex": idx,
                    "startSec": start_sec,
                    "endSec": end_sec,
                    "text": txt,
                    "tokens": tokens,
                    "phraseGroups": phrase_groups,
                }
            )
    return all_lines, sung_lines
