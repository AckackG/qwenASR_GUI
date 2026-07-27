"""
Tkinter GUI: pick video, set ASR parameters, run job.

Model load is deferred until the user clicks Start (lazy).
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from qwen_video_sub.config import (
    ASR_MODEL_PRESETS,
    DEFAULT_ALIGNER,
    DEVICE_CHOICES,
    LANGUAGE_CHOICES,
)
from qwen_video_sub.paths import ensure_project_model_cache
from qwen_video_sub.pipeline.asr_job import build_job_from_gui_values, run_job


class VideoSubApp(ttk.Frame):
    """Main window content."""

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=10)
        self.master = master
        self._worker: Optional[threading.Thread] = None
        self._msg_q: queue.Queue[str] = queue.Queue()
        self._cancel = threading.Event()
        self._build()
        self.after(200, self._drain_queue)

    def _build(self) -> None:
        self.master.title("Qwen3-ASR 本地视频字幕")
        self.master.minsize(640, 520)
        self.grid(row=0, column=0, sticky="nsew")
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        row = 0

        # Video
        ttk.Label(self, text="视频文件:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_video = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_video).grid(
            row=row, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self, text="浏览…", command=self._browse_video).grid(
            row=row, column=2, padx=2, pady=4
        )
        row += 1

        # Output
        ttk.Label(self, text="输出字幕:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_output = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_output).grid(
            row=row, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self, text="另存为…", command=self._browse_output).grid(
            row=row, column=2, padx=2, pady=4
        )
        row += 1
        ttk.Label(
            self,
            text="留空则默认写到视频同目录、同名 .srt",
            foreground="#555",
        ).grid(row=row, column=1, sticky="w")
        row += 1

        # Model
        ttk.Label(self, text="ASR 模型:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_model = tk.StringVar(value="0.6B")
        model_combo = ttk.Combobox(
            self,
            textvariable=self.var_model,
            values=list(ASR_MODEL_PRESETS.keys())
            + list(ASR_MODEL_PRESETS.values())
            + ["(本地路径请点浏览)"],
            width=40,
        )
        model_combo.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(self, text="本地模型…", command=self._browse_model).grid(
            row=row, column=2, padx=2, pady=4
        )
        row += 1

        # Language
        ttk.Label(self, text="语言:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_language = tk.StringVar(value="auto")
        ttk.Combobox(
            self,
            textvariable=self.var_language,
            values=list(LANGUAGE_CHOICES),
            width=40,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=4)
        row += 1

        # Device
        ttk.Label(self, text="设备:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_device = tk.StringVar(value="cuda:0")
        ttk.Combobox(
            self,
            textvariable=self.var_device,
            values=list(DEVICE_CHOICES),
            width=40,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=4)
        row += 1

        # Timestamps
        self.var_timestamps = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self,
            text="生成带时间轴字幕 (ForcedAligner)",
            variable=self.var_timestamps,
            command=self._toggle_aligner,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=4)
        row += 1

        ttk.Label(self, text="Aligner:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_aligner = tk.StringVar(value=DEFAULT_ALIGNER)
        self.entry_aligner = ttk.Entry(self, textvariable=self.var_aligner)
        self.entry_aligner.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        row += 1

        # Advanced
        ttk.Label(self, text="max_new_tokens:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_max_tokens = tk.StringVar(value="1024")
        ttk.Entry(self, textvariable=self.var_max_tokens, width=12).grid(
            row=row, column=1, sticky="w", padx=4, pady=4
        )
        row += 1

        ttk.Label(self, text="dtype:").grid(row=row, column=0, sticky="w", pady=4)
        self.var_dtype = tk.StringVar(value="bfloat16")
        ttk.Combobox(
            self,
            textvariable=self.var_dtype,
            values=["bfloat16", "float16", "float32"],
            width=16,
        ).grid(row=row, column=1, sticky="w", padx=4, pady=4)
        row += 1

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        self.btn_start = ttk.Button(btn_frame, text="开始转写", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_open = ttk.Button(
            btn_frame, text="打开输出目录", command=self._open_output_dir
        )
        self.btn_open.pack(side=tk.LEFT, padx=4)
        row += 1

        # Log
        ttk.Label(self, text="日志:").grid(row=row, column=0, sticky="nw", pady=4)
        self.log = scrolledtext.ScrolledText(self, height=14, state="disabled", wrap=tk.WORD)
        self.log.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=4, pady=4)
        self.rowconfigure(row, weight=1)

        cache = ensure_project_model_cache()
        self._log(
            "就绪。请选择视频并设置参数后点「开始转写」。\n"
            f"模型缓存目录（项目内，不写系统用户缓存）:\n  {cache}\n"
            "首次运行才会下载权重（需网络与足够磁盘）；当前不预下载。\n"
            "推荐 Windows + NVIDIA GPU；CPU 可能极慢或不可用。"
        )

    def _toggle_aligner(self) -> None:
        state = "normal" if self.var_timestamps.get() else "disabled"
        self.entry_aligner.configure(state=state)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[
                ("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v *.wmv *.ts"),
                ("All", "*.*"),
            ],
        )
        if path:
            self.var_video.set(path)
            if not self.var_output.get().strip():
                self.var_output.set(str(Path(path).with_suffix(".srt")))

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存字幕",
            defaultextension=".srt",
            filetypes=[("SubRip", "*.srt"), ("All", "*.*")],
        )
        if path:
            self.var_output.set(path)

    def _browse_model(self) -> None:
        path = filedialog.askdirectory(title="选择本地 ASR 模型目录")
        if path:
            self.var_model.set(path)

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self._msg_q.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass
        self.after(200, self._drain_queue)

    def _set_busy(self, busy: bool) -> None:
        self.btn_start.configure(state="disabled" if busy else "normal")

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("进行中", "已有任务在运行，请等待完成。")
            return

        video = self.var_video.get().strip()
        if not video:
            messagebox.showerror("错误", "请先选择视频文件。")
            return
        if not Path(video).is_file():
            messagebox.showerror("错误", f"视频不存在:\n{video}")
            return

        model = self.var_model.get().strip()
        if model.startswith("("):
            messagebox.showerror("错误", "请选择 0.6B / 1.7B，或浏览本地模型路径。")
            return

        try:
            max_tokens = int(self.var_max_tokens.get().strip() or "1024")
        except ValueError:
            messagebox.showerror("错误", "max_new_tokens 必须是整数。")
            return

        cfg = build_job_from_gui_values(
            video_path=video,
            output_path=self.var_output.get().strip(),
            model=model,
            language=self.var_language.get().strip() or "auto",
            device=self.var_device.get().strip() or "cuda:0",
            use_timestamps=bool(self.var_timestamps.get()),
            aligner_model=self.var_aligner.get().strip(),
            max_new_tokens=max_tokens,
            dtype=self.var_dtype.get().strip() or "bfloat16",
        )

        self._set_busy(True)
        self._log("—— 开始任务 ——")
        self._log(f"视频: {cfg.video_path}")
        self._log(f"模型: {cfg.model} → {cfg.resolved_model_id()}")
        self._log(f"设备: {cfg.device}; 时间轴: {cfg.use_timestamps}")

        def worker() -> None:
            def on_progress(m: str) -> None:
                self._msg_q.put(m)

            try:
                out = run_job(cfg, progress=on_progress)
                self._msg_q.put(f"完成: {out}")
                self._msg_q.put("__DONE_OK__")
            except Exception as e:  # noqa: BLE001 — surface to UI
                self._msg_q.put(f"失败: {e}")
                self._msg_q.put("__DONE_ERR__")

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self.after(300, self._watch_worker)

    def _watch_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            self.after(300, self._watch_worker)
            return
        # Drain remaining messages then re-enable
        self.after(100, self._finish_job)

    def _finish_job(self) -> None:
        try:
            while True:
                msg = self._msg_q.get_nowait()
                if msg in ("__DONE_OK__", "__DONE_ERR__"):
                    continue
                self._log(msg)
        except queue.Empty:
            pass
        self._set_busy(False)
        self._log("—— 任务结束 ——")

    def _open_output_dir(self) -> None:
        out = self.var_output.get().strip() or self.var_video.get().strip()
        if not out:
            messagebox.showinfo("提示", "尚无输出路径。")
            return
        folder = Path(out).parent
        if not folder.is_dir():
            messagebox.showerror("错误", f"目录不存在:\n{folder}")
            return
        import os
        import subprocess
        import sys

        path = str(folder)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("错误", str(e))


def main() -> None:
    root = tk.Tk()
    try:
        # Prefer a theme available on Windows.
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except tk.TclError:
        pass
    VideoSubApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
