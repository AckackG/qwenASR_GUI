"""Unit tests for ffmpeg command builder and path helpers (no model)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qwen_video_sub.pipeline.audio_extract import (
    AudioExtractError,
    build_ffmpeg_extract_cmd,
    default_temp_audio_path,
    extract_audio_from_video,
)


class TestBuildFfmpegCmd(unittest.TestCase):
    def test_basic_argv(self) -> None:
        # Windows-style path as string must be preserved in argv.
        video = r"C:\Videos\demo clip.mp4"
        out = r"C:\Temp\out.wav"
        cmd = build_ffmpeg_extract_cmd(video, out, ffmpeg_bin="ffmpeg")
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-y", cmd)
        self.assertIn("-i", cmd)
        i = cmd.index("-i")
        self.assertEqual(cmd[i + 1], video)
        self.assertIn("-vn", cmd)
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "pcm_s16le")
        self.assertEqual(cmd[-1], out)

    def test_no_overwrite_flag(self) -> None:
        cmd = build_ffmpeg_extract_cmd("a.mp4", "b.wav", overwrite=False)
        self.assertIn("-n", cmd)
        self.assertNotIn("-y", cmd)

    def test_invalid_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            build_ffmpeg_extract_cmd("a.mp4", "b.wav", sample_rate=0)

    def test_custom_ffmpeg_bin(self) -> None:
        cmd = build_ffmpeg_extract_cmd(
            "v.mp4", "a.wav", ffmpeg_bin=r"D:\tools\ffmpeg.exe"
        )
        self.assertEqual(cmd[0], r"D:\tools\ffmpeg.exe")


class TestDefaultTempPath(unittest.TestCase):
    def test_beside_video(self) -> None:
        p = default_temp_audio_path(r"C:\media\clip.mkv")
        self.assertTrue(str(p).endswith(".__qwen_asr_audio__.wav"))
        self.assertIn("clip", p.name)

    def test_work_dir(self) -> None:
        p = default_temp_audio_path("clip.mp4", work_dir="/tmp/work")
        self.assertEqual(p.parent, Path("/tmp/work"))


class TestExtractAudioRunner(unittest.TestCase):
    """Drive shipped extract_audio_from_video with a fake ffmpeg subprocess."""

    def test_missing_video(self) -> None:
        with self.assertRaises(AudioExtractError):
            extract_audio_from_video("/nonexistent/video.mp4", "/tmp/x.wav")

    def test_success_with_mock_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            video = td_path / "v.mp4"
            video.write_bytes(b"fake-video")
            audio = td_path / "a.wav"

            def fake_run(cmd, **kwargs):
                # Real extract writes the output path (last arg).
                out = Path(cmd[-1])
                out.write_bytes(b"RIFF" + b"\x00" * 44)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch(
                "qwen_video_sub.pipeline.audio_extract.find_ffmpeg",
                return_value="ffmpeg",
            ), mock.patch(
                "qwen_video_sub.pipeline.audio_extract.subprocess.run",
                side_effect=fake_run,
            ):
                result = extract_audio_from_video(video, audio)
            self.assertEqual(result, audio)
            self.assertTrue(audio.is_file())

    def test_ffmpeg_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            video = td_path / "v.mp4"
            video.write_bytes(b"x")
            audio = td_path / "a.wav"

            with mock.patch(
                "qwen_video_sub.pipeline.audio_extract.find_ffmpeg",
                return_value="ffmpeg",
            ), mock.patch(
                "qwen_video_sub.pipeline.audio_extract.subprocess.run",
                return_value=mock.Mock(returncode=1, stdout="", stderr="boom"),
            ):
                with self.assertRaises(AudioExtractError) as ctx:
                    extract_audio_from_video(video, audio)
                self.assertIn("boom", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
