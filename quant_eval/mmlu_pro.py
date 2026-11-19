"""MMLU-Pro 5-shot CoT evaluation using the stock LM Evaluation Harness task."""

from __future__ import annotations

from typing import Any


MMLU_PROMPT_CONTEXT = 8192
MMLU_OFF_MAX_TOKENS = 2048
MMLU_THINKING_MAX_TOKENS = 8192
MMLU_FULL_TEST_SAMPLES = 12032
MMLU_STOP = "Question:"
THINK_END = "</think>"


def generation_kwargs(thinking: bool) -> dict[str, Any]:
    return {
        "until": [MMLU_STOP],
        "max_gen_toks": MMLU_THINKING_MAX_TOKENS if thinking else MMLU_OFF_MAX_TOKENS,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
    }


GENERATION_KWARGS = generation_kwargs(False)


def _supports_thinking(template: Any) -> bool:
    if isinstance(template, dict):
        return any(_supports_thinking(value) for value in template.values())
    return isinstance(template, str) and "enable_thinking" in template


def _model_context_length(model: Any, fallback: int) -> int:
    config = getattr(model, "config", None)
    configs = [getattr(config, "text_config", None), config]
    for candidate in configs:
        if candidate is None:
            continue
        for name in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
            value = getattr(candidate, name, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                return value
    return fallback


def _gguf_chat_formatter(llama: Any):
    metadata = getattr(llama, "metadata", {})
    template = metadata.get("tokenizer.chat_template") or metadata.get(
        "tokenizer.chat_template.default"
    )
    if not isinstance(template, str) or not template:
        raise ValueError("MMLU-Pro GGUF evaluation requires tokenizer.chat_template metadata")

    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    except ImportError as exc:  # pragma: no cover - guarded by GGUF installation
        raise RuntimeError("GGUF MMLU-Pro requires llama-cpp-python >= 0.3.0") from exc

    def token_text(token_id: int) -> str:
        return llama.detokenize([token_id], special=True).decode("utf-8", errors="replace")

    return Jinja2ChatFormatter(
        template=template,
        eos_token=token_text(llama.token_eos()),
        bos_token=token_text(llama.token_bos()),
    )


def _thinking_answer(text: str) -> str:
    if THINK_END not in text:
        return ""
    return text.rsplit(THINK_END, 1)[1].split(MMLU_STOP, 1)[0]


def _llama_cpp_lm(llama: Any, thinking: bool = False, chat_formatter: Any = None):
    from lm_eval.api.model import LM

    formatter = chat_formatter or _gguf_chat_formatter(llama)
    if thinking and not _supports_thinking(formatter.template):
        raise ValueError("thinking=on requires a GGUF chat template with enable_thinking support")
    handler = formatter.to_chat_handler()

    class LlamaCppLM(LM):
        def __init__(self):
            self.incomplete_thinking = 0

        def loglikelihood(self, requests):
            raise NotImplementedError("MMLU-Pro uses generate_until")

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("MMLU-Pro uses generate_until")

        def generate_until(self, requests, disable_tqdm=False):
            responses = []
            for request in requests:
                prompt, kwargs = request.args
                max_tokens = kwargs["max_gen_toks"]
                messages = [{"role": "user", "content": prompt}]
                formatted = formatter(messages=messages, enable_thinking=thinking)
                prompt_tokens = llama.tokenize(
                    formatted.prompt.encode("utf-8"),
                    add_bos=not formatted.added_special,
                    special=True,
                )
                if len(prompt_tokens) > MMLU_PROMPT_CONTEXT:
                    raise ValueError(
                        "MMLU-Pro prompt exceeds the 8192-token context limit; "
                        "the full five-shot prompt was not truncated"
                    )
                n_ctx = llama.n_ctx()
                if len(prompt_tokens) + max_tokens > n_ctx:
                    raise ValueError(
                        f"MMLU-Pro prompt plus {max_tokens} output tokens exceeds GGUF n_ctx={n_ctx}"
                    )

                result = handler(
                    llama=llama,
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    repeat_penalty=1.0,
                    seed=0,
                    max_tokens=max_tokens,
                    stop=[] if thinking else [MMLU_STOP],
                    enable_thinking=thinking,
                )
                text = result["choices"][0]["message"].get("content") or ""
                if thinking:
                    if THINK_END not in text:
                        self.incomplete_thinking += 1
                        text = ""
                    else:
                        text = _thinking_answer(text)
                else:
                    text = text.split(MMLU_STOP, 1)[0]
                responses.append(text)
            return responses

    return LlamaCppLM()


def _hf_lm(model: Any, tokenizer: Any, thinking: bool):
    from lm_eval.models.huggingface import HFLM

    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("MMLU-Pro requires the model's native tokenizer chat template")
    if thinking and not _supports_thinking(template):
        raise ValueError("thinking=on requires a tokenizer chat template with enable_thinking support")

    max_new_tokens = generation_kwargs(thinking)["max_gen_toks"]
    max_length = min(
        2 * MMLU_PROMPT_CONTEXT,
        _model_context_length(model, 2 * MMLU_PROMPT_CONTEXT),
    )
    if max_length <= max_new_tokens:
        raise ValueError(
            f"Model context {max_length} cannot fit the MMLU-Pro "
            f"{max_new_tokens}-token output budget and a prompt"
        )
    prompt_limit = min(MMLU_PROMPT_CONTEXT, max_length - max_new_tokens)

    class ProtocolHFLM(HFLM):
        def __init__(self):
            super().__init__(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=1,
                logits_cache=False,
                truncation=False,
                max_length=max_length,
                enable_thinking=thinking,
                think_end_token=THINK_END if thinking else None,
            )
            self.incomplete_thinking = 0

        def generate_until(self, requests, disable_tqdm=False):
            for request in requests:
                prompt, kwargs = request.args
                max_context = self.max_length - kwargs.get("max_gen_toks", max_new_tokens)
                if max_context < 1:
                    raise ValueError(
                        f"Model context {self.max_length} cannot fit the MMLU-Pro "
                        f"{kwargs.get('max_gen_toks', max_new_tokens)}-token output budget"
                    )
                if len(self.tok_encode(prompt)) > min(prompt_limit, max_context):
                    raise ValueError(
                        "MMLU-Pro prompt exceeds the available context; "
                        "the full five-shot prompt was not truncated"
                    )
            responses = super().generate_until(requests, disable_tqdm=disable_tqdm)
            if thinking:
                return [text.split(MMLU_STOP, 1)[0] for text in responses]
            return responses

        def _model_generate(self, context, max_length, stop, **generation_kwargs):
            output = super()._model_generate(
                context,
                max_length,
                [] if thinking else stop,
                **generation_kwargs,
            )
            if thinking:
                prompt_length = context.shape[1]
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id
                for row in range(output.shape[0]):
                    generated = output[row, prompt_length:].tolist()
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
                    if THINK_END not in decoded:
                        self.incomplete_thinking += 1
                        output[row, prompt_length:] = pad_id
            return output

    return ProtocolHFLM()


def evaluate_mmlu_pro(model: Any, tokenizer: Any, device: str,
                      backend: str = "transformers",
                      max_samples: int | None = None,
                      thinking: bool = False) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
    except ImportError as exc:
        raise RuntimeError("MMLU-Pro requires LM Evaluation Harness") from exc

    gen_kwargs = generation_kwargs(thinking)
    if backend == "gguf":
        lm = _llama_cpp_lm(model, thinking=thinking)
        apply_chat_template = False
        max_sequence_length = getattr(model, "n_ctx", lambda: 0)()
        prompt_context_limit = min(
            MMLU_PROMPT_CONTEXT,
            max_sequence_length - gen_kwargs["max_gen_toks"],
        )
    elif backend == "transformers":
        lm = _hf_lm(model, tokenizer, thinking)
        apply_chat_template = True
        max_sequence_length = lm.max_length
        prompt_context_limit = min(
            MMLU_PROMPT_CONTEXT,
            max_sequence_length - gen_kwargs["max_gen_toks"],
        )
    else:
        raise ValueError(f"unsupported MMLU-Pro backend: {backend}")
    if prompt_context_limit < 1:
        raise ValueError(
            f"Model context {max_sequence_length} cannot fit the MMLU-Pro "
            f"{gen_kwargs['max_gen_toks']}-token output budget and a prompt"
        )

    raw = simple_evaluate(
        model=lm,
        tasks=["mmlu_pro"],
        batch_size=1,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=False,
        gen_kwargs=gen_kwargs,
        limit=max_samples,
        log_samples=False,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
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
    if max_samples is None and total != MMLU_FULL_TEST_SAMPLES:
        raise RuntimeError(
            f"Expected the full MMLU-Pro test split ({MMLU_FULL_TEST_SAMPLES} questions), got {total}"
        )

    incomplete = getattr(lm, "incomplete_thinking", 0)
    return {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "method": "5-shot CoT generation",
        "backend": "lm-eval/hf" if backend == "transformers" else "llama.cpp",
        "task": "mmlu_pro",
        "task_version": raw.get("versions", {}).get("mmlu_pro_biology"),
        "num_fewshot": 5,
        "fewshot_split": "validation",
        "fewshot_sampler": "first_n per subject",
        "fewshot_format": "single user prompt",
        "batch_size": 1,
        "prompt": "Harness native MMLU-Pro CoT prompt wrapped in model chat template",
        "chat_template": "model_default",
        "thinking": "on" if thinking else "off",
        "reasoning_effort": "template_default" if thinking else None,
        "prompt_context_limit": prompt_context_limit,
        "max_sequence_length": max_sequence_length,
        "stop_word": MMLU_STOP,
        "thinking_stop_behavior": "continue within the original output budget" if thinking else None,
        "generation_kwargs": {
            **gen_kwargs,
            "seed": 0,
            "repeat_penalty": 1.0,
            "min_p": 0.0,
        },
        "incomplete_thinking": incomplete,
        "metric": "exact_match,custom-extract",
        "samples": total,
        "accuracy": weighted / total,
        "category_accuracy": dict(sorted(categories.items())),
    }
