"""Tkinter GUI for video → subtitle (lazy exports; import app only when needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["main", "VideoSubApp"]

if TYPE_CHECKING:
    from .app import VideoSubApp as VideoSubApp
    from .app import main as main


def __getattr__(name: str) -> Any:
    if name in ("main", "VideoSubApp"):
        from . import app as _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
