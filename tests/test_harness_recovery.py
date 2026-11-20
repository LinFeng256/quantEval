"""Opt-in integration tests using the installed Harness and cached datasets."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_eval.ifeval import IFEVAL_METRICS, evaluate_ifeval
from quant_eval.mmlu_pro import MMLU_METRIC, evaluate_mmlu_pro
from quant_eval.scoring import preflight_scoring


@unittest.skipUnless(os.environ.get("QUANT_EVAL_INTEGRATION") == "1", "requires Harness and dataset cache")
class HarnessRecoveryTests(unittest.TestCase):
    def setUp(self):
        from lm_eval.api.model import LM

        class CannedLM(LM):
            tokenizer_name = "quant-eval-test"
            max_length = 16384
            incomplete_thinking = 0

            def __init__(self, response):
                super().__init__()
                self.response = response
                self.generated = 0

            def apply_chat_template(self, chat_history, add_generation_prompt=True):
                return "\n".join(m["content"] for m in chat_history)

            def loglikelihood(self, requests):
                raise AssertionError("only generation should be used")

            def loglikelihood_rolling(self, requests):
                raise AssertionError("only generation should be used")

            def generate_until(self, requests, **kwargs):
                self.generated += len(requests)
                return [self.response] * len(requests)

        self.lm = CannedLM

    def test_preflight_uses_real_scorers(self):
        preflight_scoring({"ifeval": IFEVAL_METRICS}, "First sentence. Second sentence.")
        preflight_scoring({"mmlu_pro_biology": (MMLU_METRIC,)}, "The answer is (A).")

    def test_ifeval_replays_after_real_scoring_failure(self):
        import lm_eval
        from lm_eval.api.task import ConfigurableTask

        def evaluate_with_dummy(lm):
            def evaluate(**kwargs):
                return original(**{**kwargs, "model": lm})
            return evaluate

        original = lm_eval.simple_evaluate
        first = self.lm("First sentence. Second sentence.")
        retry = self.lm("Unexpected regeneration.")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ifeval.json"
            with patch.object(lm_eval, "simple_evaluate", evaluate_with_dummy(first)), \
                 patch.object(ConfigurableTask, "process_results", side_effect=RuntimeError("scoring failed")):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    evaluate_ifeval(directory, gpu="cpu", output=str(output), max_samples=2)
            self.assertFalse(output.exists())
            with patch.object(lm_eval, "simple_evaluate", evaluate_with_dummy(retry)):
                result = evaluate_ifeval(directory, gpu="cpu", output=str(output), max_samples=2)
            self.assertEqual(first.generated, 2)
            self.assertEqual(retry.generated, 0)
            self.assertEqual(result, json.loads(output.read_text()))
            self.assertTrue(Path(result["benchmarks"]["ifeval"]["generation_cache"]).is_file())

    def test_mmlu_replays_failed_subject_and_keeps_completed_subject(self):
        from lm_eval.api.task import ConfigurableTask

        original = ConfigurableTask.process_results

        def fail_on_business(task, doc, results):
            if task.config.task == "mmlu_pro_business":
                raise RuntimeError("scoring failed")
            return original(task, doc, results)

        first = self.lm("The answer is (A).")
        retry = self.lm("Unexpected regeneration.")
        with tempfile.TemporaryDirectory() as directory, \
             patch("quant_eval.mmlu_pro.MMLU_PRO_SUBJECTS", ("biology", "business")):
            checkpoint = Path(directory) / "mmlu.checkpoint.json"
            kwargs = {"device": "cpu", "max_samples": 2, "checkpoint_path": checkpoint, "model_id": "dummy"}
            with patch("quant_eval.mmlu_pro._hf_lm", return_value=first), \
                 patch.object(ConfigurableTask, "process_results", fail_on_business):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    evaluate_mmlu_pro(None, None, **kwargs)
            self.assertEqual(set(json.loads(checkpoint.read_text())["subjects"]), {"mmlu_pro_biology"})
            with patch("quant_eval.mmlu_pro._hf_lm", return_value=retry):
                result = evaluate_mmlu_pro(None, None, **kwargs)
            self.assertEqual(first.generated, 4)
            self.assertEqual(retry.generated, 0)
            self.assertEqual(result["samples"], 4)
            self.assertEqual(json.loads(checkpoint.read_text())["status"], "completed")
