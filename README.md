# Quant Eval

一个面向量化语言模型的下载与精度评测 framework。它提供以下脚本：

- `download-model` / `scripts/download_model.sh`：下载 Hugging Face 或 ModelScope 模型，默认后台运行；
- `evaluate-model` / `scripts/evaluate_model.sh`：在指定 GPU 上运行 WikiText-2 PPL、MMLU-Pro、LIFEBench。
- `evaluate-ifeval` / `scripts/evaluate_ifeval.sh`：使用 LM Evaluation Harness 的标准 IFEval 任务评测指令遵循能力。
- `benchmark-inference` / `scripts/benchmark_inference.sh`：测量固定输入/输出 token 长度下的 TTFT、解码速度和显存峰值。

## 安装

```bash
pip install -e .
# 使用 GGUF 时：
pip install -e '.[gguf]'
# 使用 ModelScope 下载时：
pip install -e '.[modelscope]'
# 使用 IFEval 时（安装 Harness 的 HF backend）：
pip install -e '.[ifeval]'
# 使用推理性能测试时（安装 NVML Python 绑定）：
pip install -e '.[inference]'
```

评测依赖 PyTorch、Transformers 和 Datasets。建议根据 CUDA 版本单独安装匹配的 PyTorch。
IFEval 可选依赖需要 Python 3.10 或更新版本。

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

### IFEval（LM Evaluation Harness）

```bash
# 独立运行；也可用 quant-eval ifeval 或 scripts/evaluate_ifeval.sh
evaluate-ifeval ./models/qwen --gpu 2 --output results/qwen-ifeval.json

# 与现有评测合并到同一份结果文件
evaluate-model ./models/qwen --gpu 2 \
  --benchmarks wikitext mmlu-pro lifebench ifeval \
  --output results/qwen-all.json
```

使用 Harness 原生 [`ifeval` 任务](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ifeval/ifeval.yaml)及 HF backend：完整数据集、0-shot、batch size 1、greedy（`do_sample=False`、`temperature=0`），将 `max_gen_toks` 从任务默认值覆盖为 **4096**。默认应用模型自带 chat template，不传入关闭 Thinking 的参数，也不剥离生成的思考内容；Qwen3 等模型因此遵循自身默认的 Thinking 模式。结果中的 `scores` 保留 Harness 原生的 strict/loose、prompt/instruction 四项指标。思考内容也会进入 Harness 的原生 IFEval 评分，因此请在对比实验中保持相同模板与配置。

IFEval 只支持本地 Transformers checkpoint。`--max-samples` 可用于快速检查，正式结果请省略。合并评测时 IFEval 在独立进程中先运行，避免与其他评测的模型实例同时占用显存。

## 推理性能测试

```bash
# Transformers checkpoint 还需安装与当前 CUDA/PyTorch 匹配的 flash-attn：
pip install flash-attn --no-build-isolation
benchmark-inference ./models/qwen --gpu 2 --output results/qwen-inference.json

# GGUF 还需安装 CUDA 版 llama-cpp-python（不能使用 CPU-only wheel）：
CMAKE_ARGS="-DGGML_CUDA=on" pip install 'llama-cpp-python>=0.3.0'
benchmark-inference ./models/qwen.gguf --gpu 2 --output results/qwen-gguf-inference.json

# 快速检查单组长度：
benchmark-inference ./models/qwen --gpu 2 --case 512:256 --warmup 0 --repeats 1
```

默认测试 `512->256`、`4096->512`、`8192->1024`、`16384->1024`、`32768->1024`、`4096->4096`，每组独立进程加载模型，预热 1 次、正式重复 3 次。单路请求、greedy，输入是固定文本截取出的精确 token 数，生成时忽略 EOS 以保证输出 token 数固定；不计 tokenizer 与模型加载时间。

HF 后端显式使用 `flash_attention_2`、FP16 模型计算与 FP16 K/V cache，并在运行时检查实际 cache dtype。GGUF 后端使用 CUDA 版 llama.cpp、`flash_attn=True`、`type_k=type_v=GGML_TYPE_F16`，模型所有层放在指定 GPU。无法满足这些条件时该组会报错，不会静默切换后端。模型需支持输入加输出的总上下文长度，最长一组为 33792 tokens。

输出 JSON 每组包含 `ttft_ms`（从预填充开始到首 token 就绪）、`decode_tokens_per_s`（剩余输出 token 数除以解码时间）、`peak_memory_gib`（NVML 每 20ms 采样本进程的显存，包括模型和 KV cache）。时间取重复测量的中位数，显存取该组全过程峰值。失败的组记录 `status=error` 并继续其他组；只要有失败，命令最终返回非零退出码。GGUF 的 KV cache 按该组上下文长度分配，因此显存结果包含其预分配容量。

## 作为 Python API 使用

```python
from quant_eval.download import download_model
from quant_eval.runner import evaluate

download_model("Qwen/Qwen2.5-7B-Instruct", "./models/qwen")
result = evaluate("./models/qwen", gpu="0", output="results/qwen.json")
```
