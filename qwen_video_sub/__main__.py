"""python -m qwen_video_sub"""

from qwen_video_sub.paths import ensure_project_model_cache

ensure_project_model_cache()

from qwen_video_sub.gui.app import main

if __name__ == "__main__":
    main()
