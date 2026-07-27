"""Unit tests for JobConfig validation and defaults."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_video_sub.config import (
    ASR_MODEL_PRESETS,
    ConfigError,
    JobConfig,
    validate_config,
)
from qwen_video_sub.pipeline.asr_job import build_job_from_gui_values


class TestValidateConfig(unittest.TestCase):
    def test_requires_video(self) -> None:
        with self.assertRaises(ConfigError):
            validate_config(JobConfig(video_path=""), check_video_exists=False)

    def test_missing_file_when_check(self) -> None:
        with self.assertRaises(ConfigError):
            validate_config(
                JobConfig(video_path="/no/such/video.mp4"),
                check_video_exists=True,
            )

    def test_defaults_output_beside_video(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "my video.mp4"
            video.write_bytes(b"x")
            cfg = validate_config(
                JobConfig(video_path=str(video), model="0.6B"),
                check_video_exists=True,
            )
            self.assertTrue(cfg.output_path.endswith(".srt"))
            self.assertIn("my video", cfg.output_path)
            self.assertEqual(cfg.resolved_model_id(), ASR_MODEL_PRESETS["0.6B"])

    def test_preset_1_7b(self) -> None:
        cfg = JobConfig(video_path="x.mp4", model="1.7B")
        cfg = validate_config(cfg, check_video_exists=False)
        self.assertEqual(cfg.resolved_model_id(), "Qwen/Qwen3-ASR-1.7B")

    def test_local_or_hf_path_passthrough(self) -> None:
        cfg = JobConfig(video_path="x.mp4", model=r"D:\models\Qwen3-ASR-0.6B")
        cfg = validate_config(cfg, check_video_exists=False)
        self.assertEqual(cfg.resolved_model_id(), r"D:\models\Qwen3-ASR-0.6B")

    def test_language_auto_none(self) -> None:
        cfg = validate_config(
            JobConfig(video_path="x.mp4", language="auto"),
            check_video_exists=False,
        )
        self.assertIsNone(cfg.resolved_language())

    def test_language_forced(self) -> None:
        cfg = validate_config(
            JobConfig(video_path="x.mp4", language="Chinese"),
            check_video_exists=False,
        )
        self.assertEqual(cfg.resolved_language(), "Chinese")

    def test_bad_format(self) -> None:
        with self.assertRaises(ConfigError):
            validate_config(
                JobConfig(video_path="x.mp4", output_format="vtt"),
                check_video_exists=False,
            )

    def test_bad_dtype(self) -> None:
        with self.assertRaises(ConfigError):
            validate_config(
                JobConfig(video_path="x.mp4", dtype="int8"),
                check_video_exists=False,
            )

    def test_output_appends_srt_suffix(self) -> None:
        cfg = validate_config(
            JobConfig(video_path="x.mp4", output_path=r"C:\out\subs"),
            check_video_exists=False,
        )
        self.assertTrue(cfg.output_path.endswith(".srt"))


class TestBuildFromGui(unittest.TestCase):
    def test_gui_builder(self) -> None:
        cfg = build_job_from_gui_values(
            video_path=r"C:\v\a.mp4",
            output_path="",
            model="0.6B",
            language="English",
            device="cuda:0",
            use_timestamps=True,
        )
        self.assertEqual(cfg.video_path, r"C:\v\a.mp4")
        self.assertTrue(cfg.use_timestamps)
        self.assertEqual(cfg.language, "English")


if __name__ == "__main__":
    unittest.main()
