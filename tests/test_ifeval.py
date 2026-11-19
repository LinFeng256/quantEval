import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_eval.ifeval import evaluate_ifeval
from quant_eval.runner import evaluate


class IFEvalTests(unittest.TestCase):
    def test_harness_receives_standard_ifeval_settings(self):
        calls = []

        def fake_evaluate(**kwargs):
            calls.append(kwargs)
            return {"results": {"ifeval": {"prompt_level_strict_acc,none": 0.5}},
                    "versions": {"ifeval": 4.0}, "n-samples": {"ifeval": {"effective": 541}}}

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            output = Path(directory) / "result.json"
            with patch.dict("sys.modules", {"lm_eval": SimpleNamespace(simple_evaluate=fake_evaluate)}):
                result = evaluate_ifeval(str(model), gpu="2", output=str(output))
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))

        self.assertEqual(calls[0]["tasks"], ["ifeval"])
        self.assertEqual(calls[0]["num_fewshot"], 0)
        self.assertEqual(calls[0]["device"], "cuda:2")
        self.assertTrue(calls[0]["apply_chat_template"])
        self.assertEqual(calls[0]["gen_kwargs"], {
            "max_gen_toks": 4096, "do_sample": False, "temperature": 0.0,
        })
        self.assertNotIn("think_end_token", calls[0]["model_args"])
        self.assertEqual(result["benchmarks"]["ifeval"]["scores"]["prompt_level_strict_acc,none"], 0.5)

    def test_unified_runner_uses_separate_process_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"benchmarks": {"ifeval": {"scores": {"x": 1}}}}),
                                  encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with patch("quant_eval.runner.subprocess.run", side_effect=fake_run) as run, \
                 patch("quant_eval.runner._load_model") as load_model:
                result = evaluate(str(model), gpu="2", benchmarks=["ifeval"])
            load_model.assert_not_called()
            self.assertEqual(result["benchmarks"]["ifeval"]["scores"], {"x": 1})
            self.assertEqual(run.call_args.args[0][2], "quant_eval.ifeval")

    def test_mixed_benchmarks_run_ifeval_before_loading_model(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()

            def fake_run(command, **kwargs):
                events.append("ifeval")
                self.assertEqual(command[command.index("--max-samples") + 1], "2")
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({"benchmarks": {"ifeval": {"scores": {"x": 1}}}}),
                                  encoding="utf-8")

            def fake_load(*args):
                events.append("load")
                return object(), object(), "cuda:2", "transformers"

            with patch("quant_eval.runner.subprocess.run", side_effect=fake_run), \
                 patch("quant_eval.runner._load_model", side_effect=fake_load), \
                 patch("quant_eval.runner.evaluate_wikitext", return_value={"ppl": 1.5}):
                result = evaluate(str(model), gpu="2", benchmarks=["wikitext", "ifeval"],
                                  max_samples=2)
            self.assertEqual(events, ["ifeval", "load"])
            self.assertEqual(result["benchmarks"]["wikitext-2-ppl"]["ppl"], 1.5)
            self.assertEqual(result["benchmarks"]["ifeval"]["scores"], {"x": 1})


if __name__ == "__main__":
    unittest.main()
