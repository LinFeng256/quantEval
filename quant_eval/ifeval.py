"""Run the stock LM Evaluation Harness IFEval task on a local HF model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .nltk_data import ensure_punkt_tab
from .persistence import atomic_write_json, protocol_cache_path
from .scoring import preflight_scoring


GENERATION_KWARGS = {"max_gen_toks": 4096, "do_sample": False, "temperature": 0.0}
IFEVAL_METRICS = (
    "prompt_level_strict_acc,none", "inst_level_strict_acc,none",
    "prompt_level_loose_acc,none", "inst_level_loose_acc,none",
)


def evaluate_ifeval(model_path: str, gpu: str = "0", output: str | None = None,
                    dtype: str = "auto", max_samples: int | None = None,
                    cache_file: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"IFEval requires a local Transformers model directory: {path}")
    if any(path.glob("*.gguf")) and not (path / "config.json").exists():
        raise ValueError("IFEval's HF backend requires a Transformers checkpoint, not a GGUF-only directory")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    try:
        import lm_eval
    except ImportError as exc:
        raise RuntimeError("IFEval requires LM Evaluation Harness: pip install -e '.[ifeval]'") from exc

    # IFEval tokenizes generated answers during scoring. Check/download this
    # resource before Harness loads the model or spends time generating.
    ensure_punkt_tab()
    if max_samples is None:
        preflight_scoring({"ifeval": IFEVAL_METRICS}, "First sentence. Second sentence.")

    device = "cpu" if gpu.lower() in {"cpu", "none"} else (f"cuda:{gpu}" if gpu.isdigit() else gpu)
    destination = Path(output) if output else Path("results/ifeval") / f"{path.name}.json"
    if cache_file is None:
        cache_file = destination.with_name(f"{destination.name}.responses.sqlite")
    cache_path = protocol_cache_path(cache_file, {
        "benchmark": "ifeval", "model": str(path), "dtype": dtype,
        "max_samples": max_samples, "generation_kwargs": GENERATION_KWARGS,
        "num_fewshot": 0, "chat_template": True, "harness": getattr(lm_eval, "__version__", None),
    })
    raw = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": str(path), "dtype": dtype},
        tasks=["ifeval"],
        num_fewshot=0,
        batch_size=1,
        device=device,
        use_cache=str(cache_path),
        apply_chat_template=True,
        gen_kwargs=GENERATION_KWARGS.copy(),
        limit=max_samples,
        log_samples=False,
    )
    scores = (raw or {}).get("results", {}).get("ifeval", {})
    if any(scores.get(metric) is None for metric in IFEVAL_METRICS):
        raise RuntimeError("LM Evaluation Harness returned incomplete IFEval scores")
    print("[IFEval] Scoring complete; saving results.", file=sys.stderr, flush=True)

    benchmark = {
        "dataset": "google/IFEval",
        "task": "ifeval",
        "backend": "lm-eval/hf",
        "num_fewshot": 0,
        "batch_size": 1,
        "chat_template": "model_default",
        "thinking": "model_default",
        "generation_kwargs": GENERATION_KWARGS.copy(),
        "generation_cache": f"{cache_path}_rank{int(os.getenv('LOCAL_RANK', '0'))}.db",
        "scores": raw["results"]["ifeval"],
    }
    if max_samples is not None:
        benchmark["limit"] = max_samples
    if "ifeval" in raw.get("versions", {}):
        benchmark["task_version"] = raw["versions"]["ifeval"]
    if "ifeval" in raw.get("n-samples", {}):
        benchmark["n_samples"] = raw["n-samples"]["ifeval"]
    result = {"model": str(path), "gpu": gpu, "device": device,
              "backend": "lm-eval/hf", "timestamp": time.time(),
              "benchmarks": {"ifeval": benchmark}}
    atomic_write_json(destination, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a local Transformers model with Harness IFEval")
    parser.add_argument("model", help="local Transformers model directory")
    parser.add_argument("--gpu", default="0", help="CUDA device id, e.g. 0 or 2; use cpu for CPU")
    parser.add_argument("--output", "-o", help="JSON result path")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-samples", type=int, help="smoke test only; omit for the full benchmark")
    parser.add_argument("--cache-file", help="override the SQLite response-cache prefix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_ifeval(args.model, args.gpu, args.output, args.dtype, args.max_samples,
                             args.cache_file)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
