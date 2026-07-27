"""Static checks: GUI source binds required controls; no model import at package load."""

from __future__ import annotations

import ast
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "qwen_video_sub" / "gui" / "app.py"


class TestGuiSourceStructure(unittest.TestCase):
    def test_app_file_exists(self) -> None:
        self.assertTrue(APP_PATH.is_file(), f"missing GUI entry: {APP_PATH}")

    def test_controls_present_in_source(self) -> None:
        src = APP_PATH.read_text(encoding="utf-8")
        required_snippets = [
            "var_video",
            "var_output",
            "var_model",
            "var_language",
            "var_device",
            "var_timestamps",
            "var_aligner",
            "askopenfilename",
            "build_job_from_gui_values",
            "run_job",
            "开始转写",
        ]
        for s in required_snippets:
            self.assertIn(s, src, f"GUI source missing {s!r}")

    def test_app_parses_as_python(self) -> None:
        src = APP_PATH.read_text(encoding="utf-8")
        ast.parse(src)

    def test_pipeline_import_does_not_import_qwen_asr(self) -> None:
        # Ensure pure modules are importable without qwen_asr installed.
        # Remove any preloaded qwen_asr.
        sys.modules.pop("qwen_asr", None)
        # Import pure pipeline pieces.
        importlib.invalidate_caches()
        from qwen_video_sub.pipeline import audio_extract, srt  # noqa: F401
        from qwen_video_sub import config  # noqa: F401

        # qwen_asr must not have been imported by pure modules.
        self.assertNotIn("qwen_asr", sys.modules)


class TestEntryPoints(unittest.TestCase):
    def test_run_gui_py_exists(self) -> None:
        self.assertTrue((ROOT / "run_gui.py").is_file())
        self.assertTrue((ROOT / "run_gui.bat").is_file())
        self.assertTrue((ROOT / "qwen_video_sub" / "__main__.py").is_file())

    def test_readme_windows_steps(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for needle in ("Windows", "ffmpeg", "qwen-asr", "run_gui", "SRT", "cuda"):
            self.assertIn(needle, readme)


if __name__ == "__main__":
    unittest.main()
