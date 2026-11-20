import json
import sys
import tempfile
from pathlib import Path
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from quant_eval.mmlu_pro import (
    GENERATION_KWARGS,
    MMLU_THINKING_MAX_TOKENS,
    _llama_cpp_lm,
    _thinking_answer,
    evaluate_mmlu_pro,
    generation_kwargs,
)
from quant_eval.runner import build_parser


SUBJECTS = (
    "biology", "business", "chemistry", "computer_science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology",
)


def _raw_results(task_name):
    subject = task_name.removeprefix("mmlu_pro_")
    index = SUBJECTS.index(subject)
    return {
        "results": {
            task_name: {"exact_match,custom-extract": 1.0 if index == 0 else 0.0}
        },
        "n-samples": {
            task_name: {"effective": 2 if index == 0 else 1}
        },
        "versions": {task_name: 3.1},
    }


class FakeHFLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_length = kwargs["max_length"]

    def tok_encode(self, prompt):
        return prompt.split()


class MMLUProTests(unittest.TestCase):
    def test_cli_exposes_thinking_mode(self):
        args = build_parser().parse_args(
            ["model", "--benchmarks", "mmlu-pro", "--mmlu-thinking", "on"]
        )
        self.assertEqual(args.mmlu_thinking, "on")

    def _harness_modules(self, fake_evaluate):
        lm_eval = types.ModuleType("lm_eval")
        lm_eval.simple_evaluate = fake_evaluate
        models = types.ModuleType("lm_eval.models")
        hf_module = types.ModuleType("lm_eval.models.huggingface")
        hf_module.HFLM = FakeHFLM
        lm_eval.models = models
        models.huggingface = hf_module
        return {
            "lm_eval": lm_eval,
            "lm_eval.models": models,
            "lm_eval.models.huggingface": hf_module,
        }

    def test_harness_protocol_and_weighted_score(self):
        calls = []

        def fake_evaluate(**kwargs):
            calls.append(kwargs)
            return _raw_results(kwargs["tasks"][0])

        tokenizer = SimpleNamespace(chat_template="{% if enable_thinking %}think{% endif %}")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
        with patch.dict(sys.modules, self._harness_modules(fake_evaluate)):
            result = evaluate_mmlu_pro(model, tokenizer, "cuda:2", max_samples=3)

        lm = calls[0]["model"]
        self.assertIsInstance(lm, FakeHFLM)
        self.assertEqual(calls[0]["tasks"], ["mmlu_pro_biology"])
        self.assertEqual(len(calls), 14)
        self.assertEqual(calls[0]["batch_size"], 1)
        self.assertTrue(calls[0]["apply_chat_template"])
        self.assertFalse(calls[0]["fewshot_as_multiturn"])
        self.assertEqual(calls[0]["gen_kwargs"], GENERATION_KWARGS)
        self.assertEqual(calls[0]["limit"], 3)
        self.assertEqual(calls[0]["random_seed"], 0)
        self.assertEqual(lm.kwargs["max_length"], 16384)
        self.assertFalse(lm.kwargs["enable_thinking"])
        self.assertIsNone(lm.kwargs["think_end_token"])
        self.assertEqual(result["samples"], 15)
        self.assertAlmostEqual(result["accuracy"], 2 / 15)
        self.assertEqual(result["category_accuracy"]["biology"], 1.0)
        self.assertEqual(result["num_fewshot"], 5)
        self.assertEqual(result["metric"], "exact_match,custom-extract")
        self.assertEqual(result["thinking"], "off")

    def test_subject_checkpoint_resumes_and_uses_generation_cache(self):
        calls = []
        failures = {"mmlu_pro_chemistry"}

        def fake_evaluate(**kwargs):
            task_name = kwargs["tasks"][0]
            calls.append(kwargs)
            if task_name in failures:
                failures.remove(task_name)
                raise RuntimeError("simulated interruption")
            return _raw_results(task_name)

        tokenizer = SimpleNamespace(chat_template="{% if enable_thinking %}think{% endif %}")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
        modules = self._harness_modules(fake_evaluate)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mmlu.checkpoint.json"
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    evaluate_mmlu_pro(model, tokenizer, "cuda:2", max_samples=1,
                                      checkpoint_path=checkpoint, model_id="model-a")
                partial = json.loads(checkpoint.read_text())
                self.assertEqual(len(partial["subjects"]), 2)
                resumed = evaluate_mmlu_pro(model, tokenizer, "cuda:2", max_samples=1,
                                            checkpoint_path=checkpoint, model_id="model-a")

            self.assertEqual(len(resumed["category_accuracy"]), 14)
            self.assertEqual(len(calls), 15)
            self.assertTrue(all(call["use_cache"].startswith(
                str(checkpoint.with_suffix(".responses.sqlite."))) for call in calls))
            final = json.loads(checkpoint.read_text())
            self.assertEqual(final["status"], "completed")

    def test_thinking_mode_uses_template_toggle_and_8192_output_cap(self):
        calls = []

        def fake_evaluate(**kwargs):
            calls.append(kwargs)
            return _raw_results(kwargs["tasks"][0])

        tokenizer = SimpleNamespace(chat_template="{% if enable_thinking %}<think>{% endif %}")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
        modules = self._harness_modules(fake_evaluate)
        with patch.dict(sys.modules, modules):
            result = evaluate_mmlu_pro(model, tokenizer, "cuda:2", max_samples=1, thinking=True)

        lm = calls[0]["model"]
        self.assertTrue(lm.kwargs["enable_thinking"])
        self.assertEqual(lm.kwargs["think_end_token"], "</think>")
        self.assertEqual(lm.max_length, 16384)
        self.assertEqual(calls[0]["gen_kwargs"]["max_gen_toks"], MMLU_THINKING_MAX_TOKENS)
        self.assertEqual(calls[0]["gen_kwargs"]["until"], ["Question:"])
        self.assertEqual(result["reasoning_effort"], "template_default")

    def test_thinking_mode_requires_template_switch(self):
        def fake_evaluate(**kwargs):
            return _raw_results(kwargs["tasks"][0])

        tokenizer = SimpleNamespace(chat_template="plain chat template")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
        with patch.dict(sys.modules, self._harness_modules(fake_evaluate)):
            with self.assertRaisesRegex(ValueError, "enable_thinking"):
                evaluate_mmlu_pro(model, tokenizer, "cuda:2", thinking=True)

    def test_full_run_rejects_wrong_test_split_size(self):
        def fake_evaluate(**kwargs):
            return _raw_results(kwargs["tasks"][0])

        tokenizer = SimpleNamespace(chat_template="plain chat template")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
        with patch.dict(sys.modules, self._harness_modules(fake_evaluate)), \
             patch("quant_eval.mmlu_pro.preflight_scoring") as preflight:
            with self.assertRaisesRegex(RuntimeError, "full MMLU-Pro test split"):
                evaluate_mmlu_pro(model, tokenizer, "cuda:2")
        self.assertEqual(len(preflight.call_args.args[0]), 14)

    def test_prompt_over_context_fails_before_harness_truncates_it(self):
        captured = []

        def fake_evaluate(**kwargs):
            captured.append(kwargs)
            return _raw_results(kwargs["tasks"][0])

        tokenizer = SimpleNamespace(chat_template="plain chat template")
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960))
        with patch.dict(sys.modules, self._harness_modules(fake_evaluate)):
            evaluate_mmlu_pro(model, tokenizer, "cuda:2", max_samples=1)

        lm = captured[0]["model"]
        request = SimpleNamespace(args=("x " * 8193, generation_kwargs(False)))
        with self.assertRaisesRegex(ValueError, "was not truncated"):
            lm.generate_until([request])

    def test_gguf_adapter_uses_native_template_and_scores_after_thinking(self):
        calls = []
        cached = []

        class Formatter:
            template = "{% if enable_thinking %}<think>{% endif %}"

            def __call__(self, messages, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(prompt="wrapped prompt", added_special=True)

            def to_chat_handler(self):
                def handler(**kwargs):
                    calls.append(kwargs)
                    text = "private reasoning</think>Answer is (B)Question:ignored"
                    return {"choices": [{"message": {"content": text}}]}
                return handler

        class LM:
            cache_hook = SimpleNamespace(add_partial=lambda *args: cached.append(args))

        lm_api = types.ModuleType("lm_eval.api")
        lm_model = types.ModuleType("lm_eval.api.model")
        lm_model.LM = LM
        llama = SimpleNamespace(
            tokenize=lambda prompt, **kwargs: [1, 2, 3],
            n_ctx=lambda: 16384,
        )
        formatter = Formatter()
        with patch.dict(sys.modules, {"lm_eval.api": lm_api, "lm_eval.api.model": lm_model}):
            lm = _llama_cpp_lm(llama, thinking=True, chat_formatter=formatter)
            request = SimpleNamespace(args=("official CoT question", generation_kwargs(True)))
            self.assertEqual(lm.generate_until([request]), ["Answer is (B)"])
            self.assertEqual(cached, [("generate_until", request.args, "Answer is (B)")])

        self.assertTrue(formatter.kwargs["enable_thinking"])
        self.assertEqual(calls[0]["max_tokens"], 8192)
        self.assertEqual(calls[0]["stop"], [])
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertEqual(calls[0]["seed"], 0)
        self.assertEqual(calls[0]["repeat_penalty"], 1.0)

    def test_incomplete_thinking_is_an_empty_incorrect_answer(self):
        self.assertEqual(_thinking_answer("still reasoning; answer is (A)"), "")
        self.assertEqual(_thinking_answer("thought</think>Answer is (C)"), "Answer is (C)")

    def test_generation_defaults_match_protocol(self):
        self.assertEqual(GENERATION_KWARGS["max_gen_toks"], 2048)
        self.assertEqual(GENERATION_KWARGS["until"], ["Question:"])
        self.assertEqual(generation_kwargs(True)["max_gen_toks"], 8192)
        self.assertEqual(generation_kwargs(True)["until"], ["Question:"])


if __name__ == "__main__":
    unittest.main()
