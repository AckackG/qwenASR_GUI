"""Job configuration, defaults, and validation (no model imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Preset Hugging Face / ModelScope model ids (also accept local directory paths).
ASR_MODEL_PRESETS: dict[str, str] = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}

DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

# Language names accepted by qwen-asr when forcing language (None = auto).
LANGUAGE_CHOICES: tuple[str, ...] = (
    "auto",
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
    "Arabic",
    "Thai",
    "Vietnamese",
    "Indonesian",
    "Malay",
    "Hindi",
    "Turkish",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Hungarian",
    "Macedonian",
    "Romanian",
)

DEVICE_CHOICES: tuple[str, ...] = (
    "cuda:0",
    "cuda:1",
    "cpu",
    "auto",
)

OUTPUT_FORMATS: tuple[str, ...] = ("srt",)

# ForcedAligner supports up to ~5 minutes per call (upstream docs).
ALIGNER_MAX_SECONDS = 5 * 60


@dataclass
class JobConfig:
    """Parameters for one video → subtitle job."""

    video_path: str
    output_path: str = ""
    model: str = "0.6B"  # preset key, HF id, or local path
    language: str = "auto"
    device: str = "cuda:0"
    use_timestamps: bool = True
    aligner_model: str = DEFAULT_ALIGNER
    output_format: str = "srt"
    max_new_tokens: int = 1024
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    keep_temp_audio: bool = False
    extra: dict = field(default_factory=dict)

    def resolved_model_id(self) -> str:
        key = (self.model or "").strip()
        if key in ASR_MODEL_PRESETS:
            return ASR_MODEL_PRESETS[key]
        return key

    def resolved_language(self) -> Optional[str]:
        lang = (self.language or "auto").strip()
        if not lang or lang.lower() == "auto":
            return None
        return lang

    def resolved_device_map(self) -> str:
        d = (self.device or "cuda:0").strip()
        if d == "auto":
            return "auto"
        return d

    def default_output_path(self) -> Path:
        video = Path(self.video_path)
        stem = video.with_suffix("")
        ext = ".srt" if self.output_format.lower() == "srt" else f".{self.output_format}"
        return Path(str(stem) + ext)


class ConfigError(ValueError):
    """Invalid job configuration."""


def validate_config(cfg: JobConfig, *, check_video_exists: bool = True) -> JobConfig:
    """
    Validate and normalize a JobConfig. Returns the same instance after filling defaults.

    Does not load models. Raises ConfigError on invalid input.
    """
    if not cfg.video_path or not str(cfg.video_path).strip():
        raise ConfigError("video_path is required")

    video = Path(cfg.video_path)
    if check_video_exists and not video.is_file():
        raise ConfigError(f"video file not found: {cfg.video_path}")

    model = (cfg.model or "").strip()
    if not model:
        raise ConfigError("model is required (0.6B, 1.7B, HF id, or local path)")
    cfg.model = model

    lang = (cfg.language or "auto").strip()
    if lang.lower() == "auto":
        cfg.language = "auto"
    elif lang not in LANGUAGE_CHOICES:
        # Allow free-form language names that qwen-asr might accept.
        cfg.language = lang
    else:
        cfg.language = lang

    device = (cfg.device or "cuda:0").strip()
    if not device:
        raise ConfigError("device is required")
    cfg.device = device

    fmt = (cfg.output_format or "srt").strip().lower()
    if fmt not in OUTPUT_FORMATS:
        raise ConfigError(f"unsupported output_format: {fmt!r}; supported: {OUTPUT_FORMATS}")
    cfg.output_format = fmt

    if cfg.max_new_tokens < 1:
        raise ConfigError("max_new_tokens must be >= 1")

    dtype = (cfg.dtype or "bfloat16").strip().lower()
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ConfigError("dtype must be bfloat16, float16, or float32")
    cfg.dtype = dtype

    if cfg.use_timestamps:
        aligner = (cfg.aligner_model or DEFAULT_ALIGNER).strip()
        if not aligner:
            raise ConfigError("aligner_model is required when use_timestamps is True")
        cfg.aligner_model = aligner

    out = (cfg.output_path or "").strip()
    if not out:
        cfg.output_path = str(cfg.default_output_path())
    else:
        cfg.output_path = out
        # Ensure .srt extension when format is srt and user omitted it.
        p = Path(cfg.output_path)
        if fmt == "srt" and p.suffix.lower() != ".srt":
            cfg.output_path = str(p) + ".srt"

    return cfg
