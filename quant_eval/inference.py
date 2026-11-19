"""Fixed-token, single-stream GPU inference benchmarks for HF and GGUF models."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable


CASES = ((512, 256), (4096, 512), (8192, 1024),
         (16384, 1024), (32768, 1024), (4096, 4096))
PROMPT_TEXT = "The following passage is used to measure language model inference speed. "
GIB = 1024 ** 3


def _resolve_model(model_path: str) -> tuple[Path, str]:
    path = Path(model_path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path, "llama.cpp"
    if not path.is_dir():
        raise ValueError(f"model must be a local Transformers directory or GGUF file: {path}")
    ggufs = sorted(path.glob("*.gguf"))
    if ggufs:
        if len(ggufs) != 1:
            first_shards = [file for file in ggufs if re.search(r"-00001-of-\d+\.gguf$", file.name)]
            if len(first_shards) == 1 and all(
                file.name.startswith(first_shards[0].name.split("-00001-of-")[0] + "-")
                for file in ggufs
            ):
                return first_shards[0], "llama.cpp"
            raise ValueError(f"multiple GGUF models in {path}; pass the desired GGUF file explicitly")
        return ggufs[0], "llama.cpp"
    if not (path / "config.json").is_file():
        raise ValueError(f"Transformers model directory has no config.json: {path}")
    return path, "transformers"


def _prompt_ids(encode: Callable[[str], list[int]], count: int) -> list[int]:
    unit = encode(PROMPT_TEXT)
    if not unit:
        raise RuntimeError("model tokenizer produced no prompt tokens")
    copies = max(1, (count + len(unit) - 1) // len(unit))
    while True:
        ids = encode(PROMPT_TEXT * copies)
        if len(ids) >= count:
            return ids[:count]
        copies *= 2


def _process_memory_bytes(processes: Any, pid: int) -> int | None:
    for process in processes:
        if process.pid == pid:
            used = int(process.usedGpuMemory)
            if used >= 1 << 63:
                raise RuntimeError("NVML cannot report per-process GPU memory on this device")
            return used
    return None


class PeakGpuMemory:
    """Poll NVML for this process, including model and context allocation."""

    def __init__(self, gpu: int, interval: float = 0.02):
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("peak GPU memory requires: pip install -e '.[inference]'") from exc
        self.nvml = pynvml
        self.nvml.nvmlInit()
        try:
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu)
            name = self.nvml.nvmlDeviceGetName(self.handle)
            self.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            self.nvml.nvmlShutdown()
            raise
        self.pid = os.getpid()
        self.interval = interval
        self.peak_bytes = 0
        self.seen = False
        self.error: Exception | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _sample(self) -> None:
        try:
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            used = _process_memory_bytes(processes, self.pid)
            if used is not None:
                with self.lock:
                    self.seen = True
                    self.peak_bytes = max(self.peak_bytes, used)
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def _poll(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self) -> "PeakGpuMemory":
        self._sample()
        if self.error is not None:
            self.nvml.nvmlShutdown()
            raise RuntimeError("NVML process memory measurement failed") from self.error
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop_event.set()
        self.thread.join()
        self._sample()
        self.nvml.nvmlShutdown()

    def peak_gib(self) -> float:
        if self.error is not None:
            raise RuntimeError("NVML process memory measurement failed") from self.error
        if not self.seen:
            raise RuntimeError("NVML did not observe this process on the selected GPU")
        return self.peak_bytes / GIB


def _cache_pairs(cache: Any) -> list[tuple[Any, Any]]:
    if hasattr(cache, "layers"):
        return [(getattr(layer, "keys", None), getattr(layer, "values", None))
                for layer in cache.layers]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) != len(cache.value_cache):
            return []
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    return []


def _assert_fp16_cache(cache: Any, torch: Any) -> None:
    pairs = _cache_pairs(cache)
    if not pairs or any(key is None or value is None or
                        key.dtype != torch.float16 or value.dtype != torch.float16
                        for key, value in pairs):
        raise RuntimeError("model K and V cache must both be torch.float16")


def _run_hf_once(model: Any, prompt: Any, output_tokens: int, torch: Any,
                 check_cache: bool = False, attention_backend: str = "flash_attention_2") -> tuple[float, float]:
    attention = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
                 if attention_backend == "sdpa-flash" else contextlib.nullcontext())
    with torch.inference_mode(), attention:
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(input_ids=prompt, use_cache=True)
        cache = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        first = time.perf_counter()
        del outputs
        for _ in range(output_tokens - 1):
            outputs = model(input_ids=token, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del outputs
        torch.cuda.synchronize()
        end = time.perf_counter()
        if check_cache:
            _assert_fp16_cache(cache, torch)
    return first - start, end - first


def _run_llama_once(llama: Any, prompt: list[int], output_tokens: int) -> tuple[float, float]:
    llama.reset()
    generated = llama.generate(prompt, temp=0.0, repeat_penalty=1.0, reset=False)
    try:
        start = time.perf_counter()
        next(generated)
        first = time.perf_counter()
        for _ in range(output_tokens - 1):
            next(generated)
        end = time.perf_counter()
    finally:
        generated.close()
    return first - start, end - first


def _load_hf(path: Path, attention_backend: str = "flash_attention_2") -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers inference requires torch and transformers") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlashAttention-2 inference")
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    implementation = "sdpa" if attention_backend == "sdpa-flash" else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.float16, attn_implementation=implementation,
        device_map={"": "cuda:0"}, trust_remote_code=True,
    )
    if getattr(model.config, "_attn_implementation", None) != implementation:
        raise RuntimeError(f"model did not activate {implementation} attention")
    model.eval()
    return model, tokenizer, torch


def _load_humming(path: Path, loader_path: str, attention_backend: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("humming inference requires torch and transformers") from exc
    loader_file = Path(loader_path).expanduser().resolve()
    if not loader_file.is_file():
        raise ValueError(f"humming loader file does not exist: {loader_file}")
    spec = importlib.util.spec_from_file_location("quant_eval_humming_loader", loader_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import humming loader: {loader_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for humming inference")
    torch.cuda.set_device(0)
    model, owner = module.load_humming_model("gptq3", path)
    implementation = "sdpa" if attention_backend == "sdpa-flash" else "flash_attention_2"
    model.set_attn_implementation(implementation)
    if getattr(model.config, "_attn_implementation", None) != implementation:
        raise RuntimeError(f"humming model did not activate {implementation} attention")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model.eval()
    return model, tokenizer, (torch, owner)


def _load_gguf(path: Path, context_tokens: int) -> Any:
    try:
        import llama_cpp
    except ImportError as exc:
        raise RuntimeError("GGUF inference requires a CUDA-enabled llama-cpp-python installation") from exc
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("llama-cpp-python was built without GPU offload; install a CUDA build")
    return llama_cpp.Llama(
        model_path=str(path), n_ctx=context_tokens, n_batch=2048, n_ubatch=512,
        n_gpu_layers=-1, split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE, main_gpu=0,
        flash_attn=True, type_k=llama_cpp.GGML_TYPE_F16,
        type_v=llama_cpp.GGML_TYPE_F16, logits_all=False, verbose=False,
    )


def measure_case(model_path: str, gpu: int, input_tokens: int, output_tokens: int,
                 warmup: int = 1, repeats: int = 3, backend_override: str = "auto",
                 humming_loader: str | None = None,
                 attention_backend: str = "flash_attention_2") -> dict[str, Any]:
    path, backend = _resolve_model(model_path)
    if backend_override == "humming":
        if backend != "transformers" or not humming_loader:
            raise ValueError("humming requires a Transformers checkpoint and --humming-loader")
        backend = "humming"
    if attention_backend not in ("flash_attention_2", "sdpa-flash"):
        raise ValueError("unsupported attention backend")
    if gpu < 0 or input_tokens < 1 or output_tokens < 2 or warmup < 0 or repeats < 1:
        raise ValueError("gpu >= 0, input_tokens >= 1, output_tokens >= 2, warmup >= 0 and repeats >= 1 are required")
    # CUDA visibility must be set before either backend initializes CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with PeakGpuMemory(gpu) as memory:
        if backend in ("transformers", "humming"):
            if backend == "humming":
                model, tokenizer, (torch, owner) = _load_humming(path, humming_loader, attention_backend)
            else:
                model, tokenizer, torch = _load_hf(path, attention_backend)
            ids = _prompt_ids(lambda text: tokenizer.encode(text, add_special_tokens=False), input_tokens)
            prompt = torch.tensor([ids], dtype=torch.long, device="cuda:0")
            short_prompt = prompt[:, :min(input_tokens, 128)]
            for _ in range(warmup):
                _run_hf_once(model, short_prompt, min(output_tokens, 8), torch,
                             check_cache=True, attention_backend=attention_backend)
            runs = [_run_hf_once(model, prompt, output_tokens, torch,
                                 check_cache=(i == 0 and warmup == 0),
                                 attention_backend=attention_backend)
                    for i in range(repeats)]
        else:
            llama = _load_gguf(path, input_tokens + output_tokens)
            ids = _prompt_ids(lambda text: llama.tokenize(text.encode("utf-8"), add_bos=True), input_tokens)
            short_ids = ids[:min(input_tokens, 128)]
            for _ in range(warmup):
                _run_llama_once(llama, short_ids, min(output_tokens, 8))
            runs = [_run_llama_once(llama, ids, output_tokens) for _ in range(repeats)]
        memory._sample()
    peak_gib = memory.peak_gib()
    timings = [{"ttft_ms": first * 1000,
                "decode_tokens_per_s": (output_tokens - 1) / decode,
                "decode_seconds": decode} for first, decode in runs]
    return {"case": f"{input_tokens}->{output_tokens}", "input_tokens": input_tokens,
            "output_tokens": output_tokens, "status": "ok", "backend": backend,
            "ttft_ms": statistics.median(run["ttft_ms"] for run in timings),
            "decode_tokens_per_s": statistics.median(run["decode_tokens_per_s"] for run in timings),
            "peak_memory_gib": peak_gib, "gpu_name": memory.gpu_name,
            "warmup": warmup, "repeats": repeats, "runs": timings}


def _write_result(result: dict[str, Any], output: str | None) -> None:
    if output:
        target = Path(output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def benchmark_inference(model_path: str, gpu: str = "0", output: str | None = None,
                        cases: tuple[tuple[int, int], ...] = CASES,
                        warmup: int = 1, repeats: int = 3, backend_override: str = "auto",
                        humming_loader: str | None = None,
                        attention_backend: str = "flash_attention_2") -> dict[str, Any]:
    path, backend = _resolve_model(model_path)
    if backend_override == "humming":
        if backend != "transformers" or not humming_loader:
            raise ValueError("humming requires a Transformers checkpoint and --humming-loader")
        backend = "humming"
    if attention_backend not in ("flash_attention_2", "sdpa-flash"):
        raise ValueError("unsupported attention backend")
    if not gpu.isdigit():
        raise ValueError("--gpu must be a physical NVIDIA GPU index")
    if not cases or warmup < 0 or repeats < 1 or any(a < 1 or b < 2 for a, b in cases):
        raise ValueError("invalid case length, warmup or repeats")
    result: dict[str, Any] = {
        "model": str(path), "gpu": int(gpu), "backend": backend, "timestamp": time.time(),
        "settings": {"batch_size": 1, "sampling": "greedy", "flash_attention": True,
                     "attention_backend": attention_backend,
                     "kv_cache_k": "f16", "kv_cache_v": "f16",
                     "memory_source": "NVML per-process usedGpuMemory",
                     "memory_poll_interval_ms": 20, "warmup": warmup, "repeats": repeats},
        "cases": [], "completed": False,
    }
    with tempfile.TemporaryDirectory(prefix="quant-eval-inference-") as directory:
        for input_tokens, output_tokens in cases:
            label = f"{input_tokens}->{output_tokens}"
            print(f"Running {label} on GPU {gpu}...", file=sys.stderr, flush=True)
            case_file = Path(directory) / "case.json"
            command = [sys.executable, "-m", "quant_eval.inference", "--worker-case",
                       str(input_tokens), str(output_tokens), str(path), "--gpu", gpu,
                       "--warmup", str(warmup), "--repeats", str(repeats),
                       "--backend", backend_override, "--attention-backend", attention_backend,
                       "--worker-output", str(case_file)]
            if humming_loader:
                command.extend(["--humming-loader", humming_loader])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.run(command, env=environment, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True, check=False)
            if process.returncode == 0:
                row = json.loads(case_file.read_text(encoding="utf-8"))
            else:
                message = process.stderr.strip().splitlines()
                detail = message[-1] if message else f"worker exited with code {process.returncode}"
                row = {"case": label, "input_tokens": input_tokens,
                       "output_tokens": output_tokens, "status": "error", "error": detail}
                print(f"{label} failed: {detail}", file=sys.stderr, flush=True)
            result["cases"].append(row)
            _write_result(result, output)
    result["completed"] = all(row["status"] == "ok" for row in result["cases"])
    _write_result(result, output)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure TTFT, decode throughput and peak GPU memory")
    parser.add_argument("model", help="local Transformers directory or GGUF file")
    parser.add_argument("--gpu", default="0", help="physical NVIDIA GPU index")
    parser.add_argument("--output", "-o", help="JSON result path")
    parser.add_argument("--case", action="append", metavar="INPUT:OUTPUT",
                        help="run only selected case(s); default runs all six")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=("auto", "humming"), default="auto")
    parser.add_argument("--humming-loader", help="Python file exporting load_humming_model(kind, path)")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa-flash"),
                        default="flash_attention_2")
    parser.add_argument("--worker-case", type=int, nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def _parse_cases(values: list[str] | None) -> tuple[tuple[int, int], ...]:
    if not values:
        return CASES
    cases = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("--case must use INPUT:OUTPUT token counts")
        try:
            case = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ValueError("--case must use INPUT:OUTPUT token counts") from exc
        cases.append(case)
    return tuple(cases)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_case is not None:
        if not args.worker_output:
            raise ValueError("worker output path is required")
        row = measure_case(args.model, int(args.gpu), *args.worker_case, args.warmup,
                           args.repeats, args.backend, args.humming_loader,
                           args.attention_backend)
        _write_result(row, args.worker_output)
        return 0
    cases = _parse_cases(args.case)
    result = benchmark_inference(args.model, args.gpu, args.output, cases, args.warmup,
                                 args.repeats, args.backend, args.humming_loader,
                                 args.attention_backend)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["completed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
