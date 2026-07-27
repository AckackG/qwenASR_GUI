"""Project-local model cache paths (no download)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from qwen_video_sub.paths import (
    HF_HOME_DIR,
    HF_HUB_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    ensure_project_model_cache,
)


class TestProjectPaths(unittest.TestCase):
    def test_models_under_project_root(self) -> None:
        self.assertEqual(MODELS_DIR, PROJECT_ROOT / "models")
        self.assertTrue(str(HF_HOME_DIR).startswith(str(PROJECT_ROOT)))
        self.assertIn("models", Path(HF_HOME_DIR).parts)
        # Must not be a bare home-cache path marker in our defaults.
        self.assertNotIn(".cache", HF_HOME_DIR.parts[:3])

    def test_ensure_sets_env_to_project(self) -> None:
        # Isolate from caller's HF_* if any.
        saved = {
            k: os.environ.pop(k, None)
            for k in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE",
            )
        }
        try:
            home = ensure_project_model_cache(force=True)
            self.assertEqual(Path(home), HF_HOME_DIR)
            self.assertEqual(os.environ["HF_HOME"], str(HF_HOME_DIR))
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(HF_HUB_DIR))
            self.assertTrue(HF_HOME_DIR.is_dir())
            self.assertTrue(HF_HUB_DIR.is_dir())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
