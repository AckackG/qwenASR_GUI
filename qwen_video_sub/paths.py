"""Project-local paths. Model caches stay under the repo, not user home (C: on Windows)."""

from __future__ import annotations

import os
from pathlib import Path

# Package is qwen_video_sub/ → project root is parent of package dir.
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent

# All downloaded weights / hub blobs live here (relative to project).
MODELS_DIR = PROJECT_ROOT / "models"
HF_HOME_DIR = MODELS_DIR / "huggingface"
HF_HUB_DIR = HF_HOME_DIR / "hub"
TRANSFORMERS_CACHE_DIR = HF_HOME_DIR / "transformers"


def ensure_project_model_cache(*, force: bool = False) -> Path:
    """
    Point Hugging Face / transformers caches at ``<project>/models/huggingface``.

    Call this before any model download or ``from_pretrained``.
    Does not download anything.

    If the user already set HF_HOME / HF_HUB_CACHE in the environment, those are
    kept unless ``force=True``.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer project tree so Windows users do not fill C:\\Users\\...\\.cache
    if force or not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(HF_HOME_DIR)
    if force or not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(HF_HUB_DIR)
    if force or not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_DIR)
    if force or not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE_DIR)
    # Avoid writing Hugging Face tokens / telemetry under unexpected homes if unset.
    if force or not os.environ.get("HF_HUB_DISABLE_TELEMETRY"):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    return HF_HOME_DIR


def models_readme_path() -> Path:
    return MODELS_DIR / "README.md"
