"""Extract mono 16 kHz WAV from video via ffmpeg (pure helpers + thin runner)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union


class AudioExtractError(RuntimeError):
    """ffmpeg audio extraction failed."""


def find_ffmpeg(ffmpeg_bin: Optional[str] = None) -> str:
    """Resolve ffmpeg executable path. Raises AudioExtractError if missing."""
    if ffmpeg_bin:
        p = Path(ffmpeg_bin)
        if p.is_file():
            return str(p)
        found = shutil.which(ffmpeg_bin)
        if found:
            return found
        raise AudioExtractError(f"ffmpeg not found at: {ffmpeg_bin}")
    found = shutil.which("ffmpeg")
    if not found:
        raise AudioExtractError(
            "ffmpeg not found on PATH. Install ffmpeg and ensure it is available "
            "(Windows: winget install ffmpeg, or place ffmpeg.exe on PATH)."
        )
    return found


def build_ffmpeg_extract_cmd(
    video_path: Union[str, Path],
    audio_out: Union[str, Path],
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
) -> List[str]:
    """
    Build an ffmpeg argv list that extracts PCM WAV audio suitable for ASR.

    Pure function: does not run ffmpeg. Uses str paths so Windows paths work.
    """
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1")
    if channels < 1:
        raise ValueError("channels must be >= 1")

    cmd: List[str] = [ffmpeg_bin]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(audio_out),
        ]
    )
    return cmd


def extract_audio_from_video(
    video_path: Union[str, Path],
    audio_out: Union[str, Path],
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_bin: Optional[str] = None,
    overwrite: bool = True,
    timeout: Optional[float] = None,
) -> Path:
    """
    Run ffmpeg to extract mono WAV from a video file.

    Returns the output Path. Raises AudioExtractError on failure.
    """
    video = Path(video_path)
    if not video.is_file():
        raise AudioExtractError(f"video file not found: {video_path}")

    out = Path(audio_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    exe = find_ffmpeg(ffmpeg_bin)
    cmd = build_ffmpeg_extract_cmd(
        video,
        out,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_bin=exe,
        overwrite=overwrite,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        raise AudioExtractError(f"failed to start ffmpeg: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioExtractError(f"ffmpeg timed out after {timeout}s") from e

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise AudioExtractError(
            f"ffmpeg failed (exit {proc.returncode}): {err[-2000:] if err else 'no output'}"
        )

    if not out.is_file() or out.stat().st_size == 0:
        raise AudioExtractError(f"ffmpeg produced no audio file: {out}")

    return out


def default_temp_audio_path(video_path: Union[str, Path], work_dir: Optional[Union[str, Path]] = None) -> Path:
    """Suggest a temp WAV path next to the video or under work_dir."""
    video = Path(video_path)
    name = video.stem + ".__qwen_asr_audio__.wav"
    if work_dir is not None:
        return Path(work_dir) / name
    return video.with_name(name)
