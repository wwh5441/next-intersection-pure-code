"""Time-Aligned LRC Parser with Vocal Entry Countdown Support."""
from __future__ import annotations

from pathlib import Path
import re

LINE_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")


def parse_lrc(path: str | Path, total_duration: float) -> list[list]:
    raw_entries: list[tuple[float, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        sec = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if not text or any(k in text for k in ("作词", "作曲", "编曲", "下个，路口，见 -")):
            continue
        raw_entries.append((round(sec, 2), text))

    raw_entries.sort(key=lambda x: x[0])
    out: list[list] = []
    for idx, (start, text) in enumerate(raw_entries):
        nxt = raw_entries[idx + 1][0] if idx + 1 < len(raw_entries) else total_duration
        end = round(min(nxt, start + 6.5), 2)
        out.append([start, end, text])
    return out
