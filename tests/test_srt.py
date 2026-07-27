"""Unit tests for SRT serialization (shipped helpers, no model)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_video_sub.pipeline.srt import (
    SrtCue,
    format_srt_timestamp,
    full_text_to_single_cue_srt,
    segments_to_srt,
    time_stamps_to_cues,
    write_srt,
)


class TestFormatTimestamp(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(format_srt_timestamp(0), "00:00:00,000")

    def test_hours_minutes(self) -> None:
        # 1h 2m 3s 456ms
        self.assertEqual(format_srt_timestamp(3723.456), "01:02:03,456")

    def test_negative_clamped(self) -> None:
        self.assertEqual(format_srt_timestamp(-1.5), "00:00:00,000")


class TestSegmentsToSrt(unittest.TestCase):
    def test_basic_two_cues(self) -> None:
        body = segments_to_srt(
            [
                SrtCue(0.0, 1.5, "Hello"),
                SrtCue(1.5, 3.0, "World"),
            ]
        )
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHello", body)
        self.assertIn("2\n00:00:01,500 --> 00:00:03,000\nWorld", body)
        self.assertTrue(body.endswith("\n"))

    def test_skips_empty_text(self) -> None:
        body = segments_to_srt(
            [
                SrtCue(0.0, 1.0, "  "),
                SrtCue(1.0, 2.0, "ok"),
            ]
        )
        self.assertIn("1\n", body)
        self.assertIn("ok", body)
        self.assertNotIn("2\n", body)

    def test_full_text_fallback(self) -> None:
        body = full_text_to_single_cue_srt("整段字幕", start=0.0, end=10.0)
        self.assertIn("整段字幕", body)
        self.assertIn("00:00:00,000 --> 00:00:10,000", body)

    def test_write_srt_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.srt"
            cues = [SrtCue(0, 1, "a"), SrtCue(1, 2, "b")]
            write_srt(path, cues)
            text = path.read_text(encoding="utf-8")
            self.assertIn("a", text)
            self.assertIn("b", text)


class TestTimeStampsToCues(unittest.TestCase):
    def test_from_objects(self) -> None:
        class TS:
            def __init__(self, text, start_time, end_time):
                self.text = text
                self.start_time = start_time
                self.end_time = end_time

        cues = time_stamps_to_cues(
            [TS("你", 0.0, 0.2), TS("好", 0.2, 0.4)],
            merge_gap=0.5,
            max_chars=20,
        )
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "你好")
        self.assertAlmostEqual(cues[0].start, 0.0)
        self.assertAlmostEqual(cues[0].end, 0.4)

    def test_from_dicts_no_merge(self) -> None:
        cues = time_stamps_to_cues(
            [
                {"text": "A", "start_time": 0.0, "end_time": 0.5},
                {"text": "B", "start_time": 2.0, "end_time": 2.5},
            ],
            merge_gap=0.1,
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "A")
        self.assertEqual(cues[1].text, "B")


if __name__ == "__main__":
    unittest.main()
