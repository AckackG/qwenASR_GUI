"""
Video → audio → Qwen3-ASR → SRT job runner.

Model import and weight load happen only inside run_job / load_model,
never at module import time.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from qwen_video_sub.config import JobConfig, validate_config
from qwen_video_sub.pipeline.audio_extract import (
    default_temp_audio_path,
    extract_audio_from_video,
)
from qwen_video_sub.pipeline.srt import (
    full_text_to_single_cue_srt,
    segments_to_srt,
    time_stamps_to_cues,
    write_srt,
)

logger = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[str], None]]


def _progress(cb: ProgressCb, msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:  # noqa: BLE001 — UI callbacks must not kill the job
            logger.exception("progress callback failed")
    logger.info(msg)


def resolve_torch_dtype(dtype_name: str):
    """Map config dtype string to torch dtype. Imports torch only when called."""
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    key = (dtype_name or "bfloat16").lower()
    if key not in mapping:
        raise ValueError(f"unknown dtype: {dtype_name}")
    return mapping[key]


def load_asr_model(cfg: JobConfig, progress: ProgressCb = None):
    """
    Lazy-load Qwen3ASRModel.from_pretrained with optional ForcedAligner.

    Does not download until this function is called.
    Weights cache under ``<project>/models/huggingface`` (not user home / C:).
    """
    from qwen_video_sub.paths import HF_HUB_DIR, ensure_project_model_cache

    cache_home = ensure_project_model_cache()
    _progress(progress, f"Model cache: {cache_home}")
    _progress(progress, f"Loading ASR model: {cfg.resolved_model_id()}")
    from qwen_asr import Qwen3ASRModel

    dtype = resolve_torch_dtype(cfg.dtype)
    device_map = cfg.resolved_device_map()
    model_id = cfg.resolved_model_id()
    cache_dir = str(HF_HUB_DIR)

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "max_new_tokens": cfg.max_new_tokens,
        "max_inference_batch_size": 8,
        # Keep hub blobs under project/models even if env was overridden later.
        "cache_dir": cache_dir,
    }

    if cfg.use_timestamps:
        _progress(progress, f"With ForcedAligner: {cfg.aligner_model}")
        kwargs["forced_aligner"] = cfg.aligner_model
        kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": device_map,
            "cache_dir": cache_dir,
        }

    model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    _progress(progress, "Model loaded")
    return model


def result_to_srt_content(
    text: str,
    time_stamps: Any,
    *,
    use_timestamps: bool,
    audio_duration: Optional[float] = None,
) -> str:
    """
    Map one ASR result (text + optional time_stamps) to SRT string.

    Pure-ish helper: no model I/O. Used by run_job and unit tests.
    """
    if use_timestamps and time_stamps:
        cues = time_stamps_to_cues(time_stamps)
        if cues:
            return segments_to_srt(cues)
        # Align failed or empty — fall through to full-text block.
    end = audio_duration if audio_duration and audio_duration > 0 else None
    return full_text_to_single_cue_srt(text or "", start=0.0, end=end)


def _probe_wav_duration_seconds(path: Path) -> Optional[float]:
    """Best-effort duration via wave module (stdlib); returns None on failure."""
    try:
        import wave

        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:  # noqa: BLE001
        return None
    return None


def run_job(
    cfg: JobConfig,
    *,
    model: Any = None,
    progress: ProgressCb = None,
    check_video_exists: bool = True,
) -> Path:
    """
    Full pipeline: validate → extract audio → transcribe → write SRT.

    If ``model`` is None, loads via load_asr_model (downloads weights if needed).
    Returns path to written subtitle file.
    """
    cfg = validate_config(cfg, check_video_exists=check_video_exists)
    video = Path(cfg.video_path)
    out_srt = Path(cfg.output_path)

    work_dir = Path(tempfile.mkdtemp(prefix="qwen_video_sub_"))
    audio_path = default_temp_audio_path(video, work_dir=work_dir)

    try:
        _progress(progress, f"Extracting audio → {audio_path}")
        extract_audio_from_video(video, audio_path)
        duration = _probe_wav_duration_seconds(audio_path)
        if duration is not None:
            _progress(progress, f"Audio duration ≈ {duration:.1f}s")

        if model is None:
            model = load_asr_model(cfg, progress=progress)

        language = cfg.resolved_language()
        _progress(
            progress,
            f"Transcribing (language={language or 'auto'}, "
            f"timestamps={cfg.use_timestamps})…",
        )

        transcribe_kwargs: dict[str, Any] = {
            "audio": str(audio_path),
            "language": language,
        }
        if cfg.use_timestamps:
            transcribe_kwargs["return_time_stamps"] = True

        results = model.transcribe(**transcribe_kwargs)
        if not results:
            raise RuntimeError("ASR returned empty results")

        r0 = results[0]
        text = getattr(r0, "text", None) or ""
        lang = getattr(r0, "language", None)
        ts = getattr(r0, "time_stamps", None)
        _progress(progress, f"Detected language: {lang!r}; text length={len(text)}")

        srt_body = result_to_srt_content(
            text,
            ts,
            use_timestamps=cfg.use_timestamps,
            audio_duration=duration,
        )
        if not srt_body.strip():
            # Still write a minimal placeholder so the user sees a file.
            srt_body = full_text_to_single_cue_srt(
                text or "(empty transcription)",
                start=0.0,
                end=duration,
            )

        write_srt(out_srt, srt_body)
        _progress(progress, f"Wrote subtitle: {out_srt}")
        return out_srt
    finally:
        if not cfg.keep_temp_audio:
            try:
                if audio_path.is_file():
                    audio_path.unlink()
                # Remove work dir if empty-ish.
                for p in work_dir.iterdir():
                    try:
                        p.unlink()
                    except OSError:
                        pass
                work_dir.rmdir()
            except OSError:
                logger.debug("temp cleanup failed", exc_info=True)


def build_job_from_gui_values(
    *,
    video_path: str,
    output_path: str = "",
    model: str = "0.6B",
    language: str = "auto",
    device: str = "cuda:0",
    use_timestamps: bool = True,
    aligner_model: str = "",
    max_new_tokens: int = 1024,
    dtype: str = "bfloat16",
) -> JobConfig:
    """Construct JobConfig from GUI field values (no I/O)."""
    from qwen_video_sub.config import DEFAULT_ALIGNER

    return JobConfig(
        video_path=video_path,
        output_path=output_path or "",
        model=model,
        language=language,
        device=device,
        use_timestamps=use_timestamps,
        aligner_model=aligner_model.strip() or DEFAULT_ALIGNER,
        max_new_tokens=int(max_new_tokens),
        dtype=dtype,
    )
