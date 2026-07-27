"""SRT subtitle serialization (pure logic, no model deps)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


@dataclass(frozen=True)
class SrtCue:
    """One subtitle cue with start/end in seconds."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if self.end < self.start:
            # Allow zero-length edge; reject inverted.
            object.__setattr__(self, "end", self.start)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm (non-negative)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _clean_cue_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Collapse excessive blank lines inside a cue.
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def segments_to_srt(segments: Sequence[SrtCue], *, reindex: bool = True) -> str:
    """
    Serialize timed segments to SRT file content (UTF-8 text with CRLF-safe newlines).

    Empty text cues are skipped. Overlapping times are kept as-is (caller may merge).
    """
    blocks: List[str] = []
    idx = 0
    for seg in segments:
        text = _clean_cue_text(seg.text)
        if not text:
            continue
        idx += 1
        n = idx if reindex else idx
        start = format_srt_timestamp(seg.start)
        end = format_srt_timestamp(seg.end if seg.end > seg.start else seg.start + 0.001)
        blocks.append(f"{n}\n{start} --> {end}\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def full_text_to_single_cue_srt(
    text: str,
    *,
    start: float = 0.0,
    end: Optional[float] = None,
    default_duration: float = 5.0,
) -> str:
    """
    Fallback when timestamps are disabled: one SRT block covering the whole text.
    """
    t = _clean_cue_text(text)
    if not t:
        return ""
    e = end if end is not None and end > start else start + max(default_duration, 0.001)
    return segments_to_srt([SrtCue(start=start, end=e, text=t)])


def time_stamps_to_cues(
    time_stamps: Iterable,
    *,
    merge_gap: float = 0.35,
    max_chars: int = 42,
) -> List[SrtCue]:
    """
    Convert qwen-asr time_stamps items (objects with text/start_time/end_time)
    into subtitle cues, optionally merging short gaps for readability.

    Pure mapping: accepts duck-typed objects or dicts with keys
    text|start_time|end_time (or start|end).
    """
    raw: List[SrtCue] = []
    for item in time_stamps:
        if item is None:
            continue
        if isinstance(item, dict):
            text = item.get("text", "")
            start = item.get("start_time", item.get("start", 0.0))
            end = item.get("end_time", item.get("end", start))
        else:
            text = getattr(item, "text", "") or ""
            start = float(getattr(item, "start_time", getattr(item, "start", 0.0)))
            end = float(getattr(item, "end_time", getattr(item, "end", start)))
        text = str(text).strip()
        if not text:
            continue
        raw.append(SrtCue(start=float(start), end=float(end), text=text))

    if not raw:
        return []

    if merge_gap < 0:
        return raw

    merged: List[SrtCue] = []
    cur_start = raw[0].start
    cur_end = raw[0].end
    cur_parts = [raw[0].text]

    def flush() -> None:
        nonlocal cur_start, cur_end, cur_parts
        if cur_parts:
            merged.append(SrtCue(start=cur_start, end=cur_end, text="".join(cur_parts)))
        cur_parts = []

    for seg in raw[1:]:
        gap = seg.start - cur_end
        candidate = "".join(cur_parts) + seg.text
        # Prefer joining CJK without spaces; Latin may already include spaces.
        if gap <= merge_gap and len(candidate) <= max_chars:
            cur_end = max(cur_end, seg.end)
            cur_parts.append(seg.text)
        else:
            flush()
            cur_start = seg.start
            cur_end = seg.end
            cur_parts = [seg.text]
    flush()
    return merged


def write_srt(path: Union[str, Path], content_or_segments: Union[str, Sequence[SrtCue]]) -> Path:
    """Write SRT content or segments to path (UTF-8 with BOM-friendly plain UTF-8)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content_or_segments, str):
        body = content_or_segments
    else:
        body = segments_to_srt(content_or_segments)
    # UTF-8 without BOM is fine for modern players; Windows Notepad prefers BOM.
    out.write_text(body, encoding="utf-8")
    return out
