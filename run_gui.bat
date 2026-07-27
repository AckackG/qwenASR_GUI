@echo off
REM Launch Qwen3-ASR video subtitle GUI on Windows.
REM Prerequisites: Python 3.12+, deps from requirements.txt, ffmpeg on PATH.

cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found on PATH. Install Python 3.12+ and retry.
  pause
  exit /b 1
)

python -c "import qwen_video_sub" 2>nul
if errorlevel 1 (
  echo [INFO] Package not on PYTHONPATH; running via run_gui.py from this folder.
)

python "%~dp0run_gui.py"
if errorlevel 1 (
  echo.
  echo [ERROR] GUI exited with an error. See messages above.
  pause
  exit /b 1
)
