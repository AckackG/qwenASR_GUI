# 模型与缓存目录（项目内）

本目录用于存放 **Qwen3-ASR / ForcedAligner 权重与 Hugging Face 缓存**，避免写到 Windows 用户目录（常见为 `C:\Users\...\.cache\huggingface`）。

## 结构

```
models/
  README.md                 ← 本文件（可提交）
  huggingface/              ← 运行时自动创建；权重下载到这里
    hub/                    ← Hugging Face Hub 缓存
    transformers/           ← transformers 缓存（如有）
```

启动 GUI 或加载模型时会设置：

- `HF_HOME` → `models/huggingface`
- `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE` → `models/huggingface/hub`
- `TRANSFORMERS_CACHE` → `models/huggingface/transformers`

也可手动把本地模型放到任意目录，在 GUI 里选「本地模型」。

## 说明

- **不要**把大体积权重提交到 git（已在 `.gitignore` 忽略 `models/huggingface/` 等）。
- 首次点「开始转写」才会下载；当前仓库不预置权重。
