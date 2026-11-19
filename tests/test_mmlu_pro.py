import unittest
from types import SimpleNamespace
from unittest.mock import patch

from quant_eval.mmlu_pro import GENERATION_KWARGS, _llama_cpp_lm, evaluate_mmlu_pro


SUBJECTS = (
    "biology", "business", "chemistry", "computer_science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology",
)


class MMLUProTests(unittest.TestCase):
    def test_harness_protocol_and_weighted_score(self):
        calls = []
        raw = {
            "results": {
                f"mmlu_pro_{subject}": {"exact_match,custom-extract": 1.0 if i == 0 else 0.0}
                for i, subject in enumerate(SUBJECTS)
            },
            "n-samples": {
                f"mmlu_pro_{subject}": {"effective": 2 if i == 0 else 1}
                for i, subject in enumerate(SUBJECTS)
            },
            "versions": {"mmlu_pro_biology": 3.1},
        }

        def fake_evaluate(**kwargs):
            calls.append(kwargs)
            return raw

        with patch("lm_eval.models.huggingface.HFLM", return_value="hf-model"), \
             patch("lm_eval.simple_evaluate", side_effect=fake_evaluate):
            result = evaluate_mmlu_pro(object(), object(), "cuda:2", max_samples=3)

        self.assertEqual(calls[0]["model"], "hf-model")
        self.assertEqual(calls[0]["tasks"], ["mmlu_pro"])
        self.assertNotIn("num_fewshot", calls[0])  # use the task's native 5-shot sampler
        self.assertEqual(calls[0]["batch_size"], 1)
        self.assertFalse(calls[0]["apply_chat_template"])
        self.assertEqual(calls[0]["gen_kwargs"], GENERATION_KWARGS)
        self.assertEqual(calls[0]["limit"], 3)
        self.assertEqual(result["samples"], 15)
        self.assertAlmostEqual(result["accuracy"], 2 / 15)
        self.assertEqual(result["category_accuracy"]["biology"], 1.0)
        self.assertEqual(result["num_fewshot"], 5)
        self.assertEqual(result["metric"], "exact_match,custom-extract")

    def test_gguf_adapter_uses_cot_stop_and_token_limit(self):
        calls = []

        def fake_llama(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return {"choices": [{"text": "The answer is (B)"}]}

        lm = _llama_cpp_lm(fake_llama)
        requests = [SimpleNamespace(args=("Question:\nExample", GENERATION_KWARGS))]
        self.assertEqual(lm.generate_until(requests), ["The answer is (B)"])
        self.assertEqual(calls[0][1]["max_tokens"], 2048)
        self.assertEqual(calls[0][1]["stop"], ["Question:"])
        self.assertEqual(calls[0][1]["temperature"], 0.0)

    def test_incomplete_subject_results_fail(self):
        with patch("lm_eval.models.huggingface.HFLM", return_value="hf-model"), \
             patch("lm_eval.simple_evaluate", return_value={"results": {}, "n-samples": {}}):
            with self.assertRaisesRegex(RuntimeError, "Expected 14"):
                evaluate_mmlu_pro(object(), object(), "cuda:2")


if __name__ == "__main__":
    unittest.main()
