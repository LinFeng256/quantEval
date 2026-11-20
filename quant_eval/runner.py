"""Benchmark runner for WikiText-2 PPL, MMLU-Pro and LIFEBench.

WikiText-2 and LIFEBench use local implementations. MMLU-Pro uses the stock
LM Evaluation Harness 5-shot CoT task and records its protocol in the result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .mmlu_pro import evaluate_mmlu_pro
from .persistence import atomic_write_json


WIKITEXT_CONFIG = {
    "dataset": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "split": "test",
    "context_length": 2048,
    "stride": 1024,
    "batch_size": 1,
    "chat_template": False,
    "sampling": False,
}


def _device(gpu: str) -> str:
    if gpu.lower() in {"cpu", "none"}:
        return "cpu"
    return "cuda:" + gpu if gpu.isdigit() else gpu


def _load_transformers(model_path: str, gpu: str, dtype: str = "auto"):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("evaluation requires torch and transformers") from exc
    device = _device(gpu)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "auto":
        kwargs["torch_dtype"] = "auto"
    if device == "cpu":
        kwargs.pop("torch_dtype", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": device}, **kwargs)
    model.eval()
    return model, tokenizer, device


def _load_model(model_path: str, gpu: str, dtype: str = "auto", gguf_n_ctx: int = 16384):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"local model not found: {path}")
    if path.suffix.lower() == ".gguf" or (path.is_dir() and any(path.glob("*.gguf"))):
        if gpu.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("GGUF evaluation requires: pip install 'quant-eval[gguf]'") from exc
        gguf = path if path.is_file() else next(path.glob("*.gguf"))
        n_gpu_layers = -1 if gpu.lower() not in {"cpu", "none"} else 0
        return Llama(model_path=str(gguf), n_ctx=gguf_n_ctx, n_gpu_layers=n_gpu_layers,
                     logits_all=True, verbose=False), None, _device(gpu), "gguf"
    model, tokenizer, device = _load_transformers(model_path, gpu, dtype)
    return model, tokenizer, device, "transformers"


def _text_column(dataset) -> str:
    for name in ("text", "content", "document"):
        if name in dataset.column_names:
            return name
    raise ValueError(f"dataset has no text column (columns={dataset.column_names})")


def _transformers_ppl(model, tokenizer, device: str, text: str, context_length: int, stride: int):
    import torch
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(device)
    total_nll = 0.0
    total_tokens = 0
    n = ids.shape[1]
    previous_end = 0
    for begin in range(0, n, stride):
        end = min(begin + context_length, n)
        target_len = end - previous_end
        input_ids = ids[:, begin:end]
        labels = input_ids.clone()
        labels[:, :-target_len] = -100
        with torch.inference_mode():
            out = model(input_ids, labels=labels)
        count = int((labels != -100).sum().item())
        if begin == 0:
            count = max(0, count - 1)  # no predecessor for first token
        if count:
            total_nll += float(out.loss.item()) * count
            total_tokens += count
        previous_end = end
        if end == n:
            break
    if not total_tokens:
        raise ValueError("WikiText-2 produced zero tokens")
    return total_nll, total_tokens


def _gguf_ppl(llama, tokenizer_unused, text: str, context_length: int, stride: int):
    # llama.cpp exposes token logits through eval().  We score each target
    # token once, using a sliding context equivalent to the HF implementation.
    ids = llama.tokenize(text.encode("utf-8"), add_bos=False)
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0
    import numpy as np
    for begin in range(0, len(ids), stride):
        end = min(begin + context_length, len(ids))
        target_start = previous_end
        context = ids[begin:end]
        llama.reset()
        llama.eval(context)
        logits = np.asarray(_gguf_scores(llama))  # [sequence, vocab]
        for absolute in range(target_start, end):
            local = absolute - begin
            if local == 0:
                continue
            target = ids[absolute]
            row = logits[local - 1]
            m = float(row.max())
            log_z = m + math.log(float(np.exp(row - m).sum()))
            total_nll += log_z - float(row[target])
            total_tokens += 1
        previous_end = end
        if end == len(ids):
            break
    return total_nll, total_tokens


def _gguf_scores(llama):
    scores = getattr(llama, "scores", None)
    if scores is None:
        scores = getattr(llama, "_scores", None)
    if scores is None:
        raise RuntimeError("installed llama-cpp-python does not expose eval logits")
    return scores


def evaluate_wikitext(model, tokenizer, device: str, backend: str, max_samples: int | None = None,
                      context_length: int = 2048, stride: int = 1024) -> dict[str, Any]:
    if context_length < 2 or stride < 1 or stride > context_length:
        raise ValueError("context_length must be >= 2 and stride must be in [1, context_length]")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(x for x in ds[_text_column(ds)] if x.strip())
    if max_samples:
        text = "\n\n".join(x for x in ds[_text_column(ds)][:max_samples] if x.strip())
    if backend == "gguf":
        nll, tokens = _gguf_ppl(model, tokenizer, text, context_length, stride)
    else:
        nll, tokens = _transformers_ppl(model, tokenizer, device, text, context_length, stride)
    # BPB needs bytes, while tokenizer offsets are not available for every
    # fast tokenizer. UTF-8 bytes of the scored text is a stable denominator.
    bpb = nll / math.log(2) / max(1, len(text.encode("utf-8")))
    return {**WIKITEXT_CONFIG, "context_length": context_length, "stride": stride,
            "tokens": tokens, "nll": nll, "ppl": math.exp(nll / tokens), "bpb": bpb}


def _find_first(row: dict[str, Any], names: Iterable[str], default: Any = None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def _word_count(text: str) -> int:
    # Match the reference implementation: each CJK character and each ASCII
    # alphanumeric word (apostrophe/hyphen included) counts as one unit.
    cjk = len(re.findall(r"[\u4e00-\u9fff]", str(text)))
    western = len(re.findall(r"\b[a-zA-Z0-9'\-]+\b", str(text)))
    return cjk + western


def _lifebench_score(actual: int, target: float, control: str) -> tuple[float, float]:
    """Return the official LIFEBench signed deviation (%) and score."""
    if target <= 0:
        return 0.0, 0.0
    control = control.lower().replace("_", " ")
    deviation = (actual - target) / target

    def asymmetric(value: float) -> float:
        return 100 * math.exp(5 * value) if value < 0 else 100 * math.exp(-2 * value)

    if "at most" in control or control in {"max", "maximum"}:
        score = 100.0 if deviation < 0 else asymmetric(deviation)
    elif "at least" in control or control in {"min", "minimum"}:
        score = 100.0 if deviation > 0 else asymmetric(deviation)
    else:
        score = asymmetric(deviation)
    return deviation * 100, score


def _gguf_generate(llama, prompt: str, max_new_tokens: int) -> str:
    result = llama(prompt, max_tokens=max_new_tokens, temperature=0.0, top_p=1.0,
                   echo=False, stop=[])
    return str(result["choices"][0].get("text", ""))


def evaluate_lifebench(model, tokenizer, device: str, backend: str = "transformers",
                       max_samples: int | None = None,
                       max_new_tokens: int = 8192,
                       lengths: list[int] | None = None,
                       controls: list[str] | None = None) -> dict[str, Any]:
    from datasets import load_dataset
    # The official HF release has main/label/lite/refactor splits, rather than
    # a test split. `main` contains the 360 base tasks and the runner expands
    # them using the published length constraints and control methods.
    ds = load_dataset("LIFEBench/LIFEBench", split="main")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    lengths = lengths or [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    controls = controls or ["equal to"]
    deviations, scores = [], []
    for row in ds:
        item = dict(row)
        task_template = _find_first(item, ("prompt", "instruction", "input", "question", "task"))
        if task_template is None:
            continue
        # Older/local converted copies may already contain one expanded row.
        row_lengths = [_find_first(item, ("target_length", "length", "word_count"))] if _find_first(item, ("target_length", "length", "word_count")) is not None else lengths
        row_controls = [str(_find_first(item, ("control_method", "length_type", "constraint"), "equal to"))] \
            if _find_first(item, ("control_method", "length_type", "constraint")) is not None else controls
        for target in row_lengths:
            if target is None:
                continue
            for control in row_controls:
                control_text = {"equal to": "等于", "at most": "至多有", "at least": "至少有"}.get(
                    control, control) if str(item.get("lang", "en")).lower() == "cn" else control
                prompt = str(task_template).replace("{word_count_type}", str(control_text)).replace("{word_count}", str(target))
                if backend == "gguf":
                    generated = _gguf_generate(model, prompt, max_new_tokens)
                else:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                    import torch
                    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
                    if tokenizer.pad_token_id is not None:
                        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
                    elif tokenizer.eos_token_id is not None:
                        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
                    with torch.inference_mode():
                        output = model.generate(**inputs, **generation_kwargs)
                    generated = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                deviation, score = _lifebench_score(_word_count(generated), float(target), control)
                deviations.append(deviation)
                scores.append(score)
    return {"dataset": "LIFEBench/LIFEBench", "split": "main", "samples": len(scores),
            "length_constraints": lengths, "control_methods": controls,
            "length_deviation": sum(abs(x) for x in deviations) / len(deviations) if deviations else None,
            "length_score": sum(scores) / len(scores) if scores else None,
            "max_new_tokens": max_new_tokens, "sampling": False}


def evaluate(model_path: str, gpu: str = "0", benchmarks: list[str] | None = None,
             output: str | None = None, max_samples: int | None = None,
             dtype: str = "auto", context_length: int = 2048, stride: int = 1024,
             max_new_tokens: int = 8192, lifebench_lengths: list[int] | None = None,
             lifebench_controls: list[str] | None = None, gguf_n_ctx: int = 16384,
             mmlu_thinking: str = "off") -> dict[str, Any]:
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if gguf_n_ctx < 2:
        raise ValueError("gguf_n_ctx must be >= 2")
    if mmlu_thinking not in {"off", "on"}:
        raise ValueError("mmlu_thinking must be 'off' or 'on'")
    benchmarks = benchmarks or ["wikitext", "mmlu-pro", "lifebench"]
    other_benchmarks = [name for name in benchmarks if name.lower().replace("_", "-") != "ifeval"]
    device = _device(gpu)
    result: dict[str, Any] = {"model": str(Path(model_path).resolve()), "gpu": gpu,
                              "device": device, "backend": "lm-eval/hf", "timestamp": time.time(),
                              "benchmarks": {}}
    if len(other_benchmarks) != len(benchmarks):
        if Path(model_path).suffix.lower() == ".gguf":
            raise ValueError("IFEval requires a local Transformers model directory")
        if output:
            final_path = Path(output).expanduser().resolve()
            result_path = final_path.with_name(f"{final_path.name}.ifeval.json")
        else:
            result_path = Path("results/ifeval") / f"{Path(model_path).resolve().name}.json"
        command = [sys.executable, "-m", "quant_eval.ifeval", model_path,
                   "--gpu", gpu, "--dtype", dtype, "--output", str(result_path)]
        if max_samples is not None:
            command.extend(["--max-samples", str(max_samples)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        result["benchmarks"]["ifeval"] = json.loads(result_path.read_text(encoding="utf-8"))["benchmarks"]["ifeval"]
    if other_benchmarks:
        model, tokenizer, device, backend = _load_model(model_path, gpu, dtype, gguf_n_ctx)
        result["backend"] = backend
    for benchmark in other_benchmarks:
        key = benchmark.lower().replace("_", "-")
        if key in {"wikitext", "wikitext-2", "ppl"}:
            result["benchmarks"]["wikitext-2-ppl"] = evaluate_wikitext(
                model, tokenizer, device, backend, max_samples, context_length, stride)
        elif key in {"mmlu", "mmlu-pro"}:
            mode = mmlu_thinking
            if output:
                final_path = Path(output).expanduser().resolve()
                checkpoint_path = final_path.with_name(
                    f"{final_path.name}.mmlu-{mode}.checkpoint.json")
            else:
                checkpoint_path = Path("results/mmlu_pro") / (
                    f"{Path(model_path).name}.mmlu-{mode}.checkpoint.json")
            result["benchmarks"]["mmlu-pro"] = evaluate_mmlu_pro(
                model, tokenizer, device, backend, max_samples,
                thinking=mmlu_thinking == "on", checkpoint_path=checkpoint_path,
                model_id=str(Path(model_path).expanduser().resolve()), dtype=dtype)
        elif key in {"life", "lifebench"}:
            result["benchmarks"]["lifebench"] = evaluate_lifebench(
                model, tokenizer, device, backend, max_samples, max_new_tokens,
                lifebench_lengths, lifebench_controls)
        else:
            raise ValueError(f"unknown benchmark: {benchmark}")
    if output:
        atomic_write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a local model on standard benchmarks")
    p.add_argument("model", help="local Transformers model directory or GGUF file")
    p.add_argument("--gpu", default="0", help="CUDA device id, e.g. 0 or 1; use cpu for CPU")
    p.add_argument("--benchmarks", nargs="+", default=["wikitext", "mmlu-pro", "lifebench"],
                   choices=["wikitext", "mmlu-pro", "lifebench", "ifeval"])
    p.add_argument("--output", "-o", help="JSON result path")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16"] , default="auto")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--gguf-n-ctx", type=int, default=16384, help="GGUF context allocation")
    p.add_argument("--mmlu-thinking", choices=["off", "on"], default="off",
                   help="explicit native Thinking mode for MMLU-Pro")
    p.add_argument("--lifebench-lengths", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                   help="LIFEBench word constraints")
    p.add_argument("--lifebench-controls", nargs="+", choices=["equal to", "at most", "at least"],
                   default=["equal to"], help="LIFEBench control methods")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args.model, args.gpu, args.benchmarks, args.output, args.max_samples,
                      args.dtype, args.context_length, args.stride, args.max_new_tokens,
                      args.lifebench_lengths, args.lifebench_controls, args.gguf_n_ctx,
                      args.mmlu_thinking)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
