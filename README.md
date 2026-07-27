# Qwen3-ASR 本地视频字幕工具（Windows）

基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 官方 Python 包 `qwen-asr`，封装一个**本地视频 → 字幕（SRT）** 的小 GUI：选视频、设参数、点开始即可。

## 功能

- 选择本地视频文件
- 选择 ASR 模型：`0.6B` / `1.7B`，或本地模型目录
- 语言：自动识别或强制指定
- 设备：`cuda:0` / `cpu` / `auto` 等
- 可选时间轴：开启后加载 `Qwen3-ForcedAligner-0.6B`，生成带起止时间的 SRT；关闭时输出整段文本的单条 SRT
- 默认字幕路径：与视频同目录、同名 `.srt`

处理链路：

```
视频 → ffmpeg 抽音频 (16kHz mono WAV) → Qwen3ASRModel.transcribe → 写 SRT
```

## Windows 安装

### 1. 系统依赖

1. **Python 3.12**（推荐；3.10+ 一般可用）  
   https://www.python.org/downloads/windows/
2. **ffmpeg** 并加入 PATH  
   - 示例：`winget install ffmpeg`  
   - 或从 https://ffmpeg.org/download.html 下载，把 `ffmpeg.exe` 所在目录加入系统 PATH  
   - 验证：`ffmpeg -version`
3. **NVIDIA GPU + 驱动**（强烈推荐）  
   官方示例默认 `device_map="cuda:0"`。无 GPU 时可选 `cpu`，但可能极慢或不可用。

### 2. Python 环境

在项目根目录（本 README 所在目录）打开 **cmd** 或 **PowerShell**：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -U -r requirements.txt
```

若需要 CUDA 版 PyTorch，请先按 [pytorch.org](https://pytorch.org) 说明安装对应 CUDA 的 torch，再安装 `qwen-asr`。

可选：FlashAttention 2（加速 / 省显存，需兼容硬件）：

```bat
pip install -U flash-attn --no-build-isolation
```

### 3. 模型权重（下载到项目目录，不写 C 盘用户缓存）

本工具会把 Hugging Face / transformers 缓存固定到**项目内**：

```
<项目根>\models\huggingface\
  hub\            ← Hub 下载缓存
  transformers\
```

对应环境变量在启动时自动设置：`HF_HOME`、`HF_HUB_CACHE`、`TRANSFORMERS_CACHE`。  
这样默认**不会**落到 `C:\Users\<你>\.cache\huggingface`。

首次点「开始转写」才会下载（约数 GiB，需网络）。**仓库不预置权重。**

也可先手动下载到项目下，再在 GUI 里选「本地模型」：

```bat
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir .\models\Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir .\models\Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir .\models\Qwen3-ForcedAligner-0.6B
```

国内用户可用 ModelScope：

```bat
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir .\models\Qwen3-ASR-0.6B
```

## 启动 GUI

任选其一：

```bat
run_gui.bat
```

```bat
.venv\Scripts\activate
python run_gui.py
```

```bat
python -m qwen_video_sub
```

在界面中：

1. 点「浏览」选择视频  
2. 确认输出 `.srt` 路径（可留空用默认）  
3. 选模型 / 语言 / 设备，是否开启时间轴  
4. 点「开始转写」  

模型在点击开始后才会加载（import 时不下载权重）。

## 命令行自检（无 GPU / 无模型时）

纯逻辑单元测试（不加载模型、不下载权重）：

```bat
python -m unittest discover -s tests -v
```

语法检查：

```bat
python -m py_compile run_gui.py qwen_video_sub\config.py qwen_video_sub\pipeline\audio_extract.py qwen_video_sub\pipeline\srt.py qwen_video_sub\pipeline\asr_job.py qwen_video_sub\gui\app.py
```

> **说明：** 完整端到端转写需要下载模型并在合适设备上运行。本环境若无显卡且磁盘有限，请勿在此强制下载权重；以代码 + 纯单元测试为准。

## 项目结构

```
qwenASR/
  run_gui.bat          # Windows 双击/命令行入口
  run_gui.py
  requirements.txt
  README.md
  qwen_video_sub/
    config.py          # 参数与校验（无模型依赖）
    pipeline/
      audio_extract.py # ffmpeg 命令构造与抽音频
      srt.py           # SRT 序列化
      asr_job.py       # 懒加载 Qwen3ASRModel + 任务编排
    gui/
      app.py           # Tkinter GUI
  tests/               # 纯逻辑单测
```

## 参数说明

| 参数 | 说明 |
|------|------|
| ASR 模型 | `0.6B` / `1.7B` 预设，或 Hugging Face id / 本地目录 |
| 语言 | `auto` 或 English / Chinese 等（与 qwen-asr 一致） |
| 设备 | `cuda:0`（推荐）、`cpu`、`auto` |
| 时间轴 | 开启时加载 ForcedAligner；关闭则整段文本一条 cue |
| max_new_tokens | 长音频可加大（如 1024） |
| dtype | 默认 `bfloat16`；老 GPU 可试 `float16` |

## 许可与致谢

- 本仓库 GUI/封装代码按项目自有约定使用；推理能力来自 [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（Apache-2.0）。
- 使用模型请遵守对应模型卡片与许可证。
