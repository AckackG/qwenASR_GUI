"""Unit tests for ASR job pure helpers (no model download / GPU)."""

from __future__ import annotations

import unittest

from qwen_video_sub.pipeline.asr_job import result_to_srt_content


class TestResultToSrtContent(unittest.TestCase):
    def test_with_timestamps(self) -> None:
        class TS:
            def __init__(self, text, start_time, end_time):
                self.text = text
                self.start_time = start_time
                self.end_time = end_time

        body = result_to_srt_content(
            "ignored when stamps present",
            [TS("你好", 0.0, 1.0), TS("世界", 1.2, 2.0)],
            use_timestamps=True,
        )
        self.assertIn("你好", body)
        self.assertIn("00:00:00,000", body)
        self.assertIn("-->", body)

    def test_without_timestamps_full_text(self) -> None:
        body = result_to_srt_content(
            "Full transcript line",
            None,
            use_timestamps=False,
            audio_duration=12.5,
        )
        self.assertIn("Full transcript line", body)
        self.assertIn("00:00:12,500", body)

    def test_timestamps_empty_falls_back_to_text(self) -> None:
        body = result_to_srt_content(
            "fallback text",
            [],
            use_timestamps=True,
            audio_duration=5.0,
        )
        self.assertIn("fallback text", body)
        self.assertIn("00:00:05,000", body)

    def test_timestamps_false_ignores_stamps(self) -> None:
        class TS:
            def __init__(self):
                self.text = "should-not-appear-alone"
                self.start_time = 0.0
                self.end_time = 1.0

        # When use_timestamps is False, full text path is used.
        body = result_to_srt_content(
            "plain only",
            [TS()],
            use_timestamps=False,
            audio_duration=3.0,
        )
        self.assertIn("plain only", body)


if __name__ == "__main__":
    unittest.main()
