# Quant Eval

一个面向量化语言模型的下载与精度评测 framework。它提供两个脚本：

- `download-model` / `scripts/download_model.sh`：下载 Hugging Face 或 ModelScope 模型，默认后台运行；
- `evaluate-model` / `scripts/evaluate_model.sh`：在指定 GPU 上运行 WikiText-2 PPL、MMLU-Pro、LIFEBench。

## 安装

```bash
pip install -e .
# 使用 GGUF 时：
pip install -e '.[gguf]'
# 使用 ModelScope 下载时：
pip install -e '.[modelscope]'
```

评测依赖 PyTorch、Transformers 和 Datasets。建议根据 CUDA 版本单独安装匹配的 PyTorch。

## 下载模型

Hugging Face 默认使用 `https://hf-mirror.com`，也可以通过 `--hf-endpoint` 或环境变量覆盖。下载保留 hub/cache 中的已有文件，重复执行会继续未完成的下载。

```bash
# 默认后台运行，日志和 PID 写入目标目录
download-model Qwen/Qwen2.5-7B-Instruct -o ./models/qwen

# URL 同样支持；前台运行便于查看错误
download-model https://huggingface.co/Qwen/Qwen2.5-7B-Instruct \
  -o ./models/qwen --foreground

download-model https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct \
  -o ./models/qwen --foreground
tail -f ./models/qwen/download.log
```

后台任务会写入 `download.pid` 和 `download.log`。下载令牌可以通过 `--token` 传入，或由 Hugging Face 的本地登录状态提供。

## 评测

```bash
evaluate-model ./models/qwen --gpu 0 \
  --benchmarks wikitext mmlu-pro lifebench \
  --output results/qwen.json
```

`--gpu 1` 选择第二张卡，`--gpu cpu` 使用 CPU。评测器不会使用 chat template 或采样来计算 PPL。

### WikiText-2 PPL 配置

默认使用 `wikitext-2-raw-v1` 的完整 `test` split，2048 token context、1024 stride、batch size 1。结果中包含：

- `ppl`：token-level perplexity；
- `bpb`：按 UTF-8 字节归一化的 bits per byte；
- `tokens` 和 `nll`：便于复核。

如果只想快速检查安装，可以加 `--max-samples 2`。完整报告不要使用这个选项。

### MMLU-Pro

使用 `TIGER-Lab/MMLU-Pro` 的 `test` split，对 A–J 选项计算答案 continuation 的 loss（direct、无采样），输出总体准确率和按 `category` 分组的准确率。该结果对应 direct 评测；如果要复现官方 CoT 排名，需要把生成式 CoT 提示和答案解析作为单独配置加入。

### LIFEBench

使用官方 `LIFEBench/LIFEBench` 的 `main` split，默认展开 10 个长度约束（16 到 8192）和 `equal to` 控制方式。固定 `do_sample=False`，按官方指数评分计算 `length_deviation`（绝对百分比）与 `length_score`。可用 `--lifebench-controls "equal to" "at most" "at least"` 运行三种控制方式，或用 `--lifebench-lengths 16 128 1024` 缩小范围。

GGUF 通过 `llama-cpp-python` 使用模型自带 tokenizer，三项 benchmark 均可运行；MMLU-Pro 使用候选答案 likelihood，LIFEBench 使用确定性 greedy generation。若要使用 Transformers 的量化 checkpoint，则传入包含 `config.json` 和 tokenizer 文件的模型目录。

GGUF 默认分配 16384 token context；长输入或长输出模型可用 `--gguf-n-ctx 32768` 调整。`--max-new-tokens` 默认 8192，8192 词约束通常需要更大的生成上限，否则 LIFEBench 长度分数会受到截断影响。

## 作为 Python API 使用

```python
from quant_eval.download import download_model
from quant_eval.runner import evaluate

download_model("Qwen/Qwen2.5-7B-Instruct", "./models/qwen")
result = evaluate("./models/qwen", gpu="0", output="results/qwen.json")
```
