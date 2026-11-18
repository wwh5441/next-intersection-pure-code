"""LRC timestamp parser producing aligned [start, end, text] lyric intervals."""
from __future__ import annotations

from pathlib import Path
import re

LRC_LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
CREDIT_PREFIXES = ("作词", "作曲", "编曲", "制作", "演唱")


def parse_lrc(path: Path, total_duration: float) -> list[list[float | str]]:
    """Parses an LRC file into [[startSec, endSec, text], ...], skipping header credits & empty markers."""
    entries: list[tuple[float, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        m = LRC_LINE_RE.match(raw_line.strip())
        if not m:
            continue
        sec = int(m.group(1)) * 60.0 + float(m.group(2))
        text = m.group(3).strip()
        entries.append((sec, text))

    entries.sort(key=lambda item: item[0])
    intervals: list[list[float | str]] = []
    for idx, (start, text) in enumerate(entries):
        if not text:
            continue
        if any(text.startswith(prefix) for prefix in CREDIT_PREFIXES) and start < 5.0:
            continue
        next_time = entries[idx + 1][0] if idx + 1 < len(entries) else min(total_duration, start + 5.0)
        end = min(total_duration, max(start + 0.6, next_time))
        intervals.append([round(start, 3), round(end, 3), text])
    return intervals
