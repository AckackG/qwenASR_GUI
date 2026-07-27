#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Windows-friendly entry: python run_gui.py"""

# Redirect HF / transformers cache into <project>/models before any model I/O.
from qwen_video_sub.paths import ensure_project_model_cache

ensure_project_model_cache()

from qwen_video_sub.gui.app import main

if __name__ == "__main__":
    main()
