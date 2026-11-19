"""MMLU-Pro 5-shot CoT evaluation using the stock LM Evaluation Harness task."""

from __future__ import annotations

from typing import Any


GENERATION_KWARGS = {
    "until": ["Question:"],
    "max_gen_toks": 2048,
    "do_sample": False,
    "temperature": 0.0,
}


def _llama_cpp_lm(llama: Any):
    from lm_eval.api.model import LM

    class LlamaCppLM(LM):
        def loglikelihood(self, requests):
            raise NotImplementedError("MMLU-Pro uses generate_until")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("MMLU-Pro uses generate_until")

        def generate_until(self, requests, disable_tqdm=False):
            responses = []
            for request in requests:
                prompt, kwargs = request.args
                result = llama(
                    prompt,
                    max_tokens=kwargs["max_gen_toks"],
                    temperature=0.0,
                    top_p=1.0,
                    stop=kwargs.get("until", []),
                    echo=False,
                )
                responses.append(result["choices"][0]["text"])
            return responses

    return LlamaCppLM()


def evaluate_mmlu_pro(model: Any, tokenizer: Any, device: str,
                      backend: str = "transformers",
                      max_samples: int | None = None) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError("MMLU-Pro requires LM Evaluation Harness") from exc

    if backend == "gguf":
        lm = _llama_cpp_lm(model)
    elif backend == "transformers":
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
                  logits_cache=False)
    else:
        raise ValueError(f"unsupported MMLU-Pro backend: {backend}")

    raw = simple_evaluate(
        model=lm,
        tasks=["mmlu_pro"],
        batch_size=1,
        apply_chat_template=False,
        gen_kwargs=GENERATION_KWARGS.copy(),
        limit=max_samples,
        log_samples=False,
    )
    if raw is None:
        raise RuntimeError("LM Evaluation Harness returned no MMLU-Pro result")

    scores = raw.get("results", {})
    counts = raw.get("n-samples", {})
    categories: dict[str, float] = {}
    total = 0
    weighted = 0.0
    for task, values in scores.items():
        if not task.startswith("mmlu_pro_"):
            continue
        score = values.get("exact_match,custom-extract")
        count = counts.get(task, {}).get("effective")
        if score is None or count is None:
            raise RuntimeError(f"MMLU-Pro result is missing score or sample count: {task}")
        categories[task.removeprefix("mmlu_pro_")] = score
        weighted += score * count
        total += count
    if len(categories) != 14 or total == 0:
        raise RuntimeError(f"Expected 14 MMLU-Pro subjects, got {len(categories)}")

    return {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "method": "5-shot CoT generation",
        "backend": "lm-eval",
        "task": "mmlu_pro",
        "task_version": raw.get("versions", {}).get("mmlu_pro_biology"),
        "num_fewshot": 5,
        "batch_size": 1,
        "chat_template": False,
        "thinking": "not explicitly set",
        "generation_kwargs": GENERATION_KWARGS.copy(),
        "metric": "exact_match,custom-extract",
        "samples": total,
        "accuracy": weighted / total,
        "category_accuracy": dict(sorted(categories.items())),
    }
