import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_eval.ifeval import IFEVAL_METRICS, evaluate_ifeval
from quant_eval.nltk_data import ensure_punkt_tab
from quant_eval.runner import evaluate


class IFEvalTests(unittest.TestCase):
    def test_harness_receives_standard_ifeval_settings(self):
        calls = []

        def fake_evaluate(**kwargs):
            calls.append(kwargs)
            return {"results": {"ifeval": {metric: 0.5 for metric in IFEVAL_METRICS}},
                    "versions": {"ifeval": 4.0}, "n-samples": {"ifeval": {"effective": 541}}}

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            output = Path(directory) / "result.json"
            with patch.dict("sys.modules", {"lm_eval": SimpleNamespace(simple_evaluate=fake_evaluate)}), \
                 patch("quant_eval.ifeval.ensure_punkt_tab") as ensure_data, \
                 patch("quant_eval.ifeval.preflight_scoring") as preflight:
                cache_file = Path(directory) / "cache" / "responses.sqlite"
                result = evaluate_ifeval(str(model), gpu="2", output=str(output),
                                         cache_file=cache_file)
                ensure_data.assert_called_once_with()
                preflight.assert_called_once()
                self.assertTrue(cache_file.parent.is_dir())
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))

        self.assertEqual(calls[0]["tasks"], ["ifeval"])
        self.assertEqual(calls[0]["num_fewshot"], 0)
        self.assertEqual(calls[0]["device"], "cuda:2")
        self.assertIn("responses.sqlite.", calls[0]["use_cache"])
        self.assertEqual(result["benchmarks"]["ifeval"]["generation_cache"], calls[0]["use_cache"] + "_rank0.db")
        self.assertTrue(calls[0]["apply_chat_template"])
        self.assertEqual(calls[0]["gen_kwargs"], {
            "max_gen_toks": 4096, "do_sample": False, "temperature": 0.0,
        })
        self.assertNotIn("think_end_token", calls[0]["model_args"])
        self.assertEqual(result["benchmarks"]["ifeval"]["scores"]["prompt_level_strict_acc,none"], 0.5)


    def test_punkt_tab_is_downloaded_only_when_missing(self):
        import hashlib
        import io
        import zipfile

        with tempfile.TemporaryDirectory() as directory:
            archive_buffer = io.BytesIO()
            with zipfile.ZipFile(archive_buffer, "w") as archive:
                archive.writestr("punkt_tab/english/README", "tokenizer data")
            archive_bytes = archive_buffer.getvalue()

            class FakeData:
                path = [directory]

                def find(self, resource):
                    if not (Path(directory) / resource).is_dir():
                        raise LookupError(resource)

            class FakeResponse:
                content = archive_bytes

                @staticmethod
                def raise_for_status():
                    pass

            requests = SimpleNamespace(get=lambda *args, **kwargs: FakeResponse())
            with patch.dict("sys.modules", {"requests": requests}), \
                 patch("quant_eval.nltk_data._PUNKT_TAB_SHA256",
                       hashlib.sha256(archive_bytes).hexdigest()):
                nltk = SimpleNamespace(data=FakeData())
                ensure_punkt_tab(nltk)
                ensure_punkt_tab(nltk)

            self.assertTrue((Path(directory) / "tokenizers/punkt_tab/english/README").is_file())

    def test_punkt_tab_download_is_rejected_if_checksum_does_not_match(self):
        class FakeData:
            path = ["/unused"]

            @staticmethod
            def find(resource):
                raise LookupError(resource)

        class FakeResponse:
            content = b"unexpected archive"

            @staticmethod
            def raise_for_status():
                pass

        requests = SimpleNamespace(get=lambda *args, **kwargs: FakeResponse())
        with patch.dict("sys.modules", {"requests": requests}):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                ensure_punkt_tab(SimpleNamespace(data=FakeData()))

    def test_unified_runner_uses_separate_process_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"benchmarks": {"ifeval": {"scores": {"x": 1}}}}),
                                  encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with patch("quant_eval.runner.subprocess.run", side_effect=fake_run) as run, \
                 patch("quant_eval.runner._load_model") as load_model:
                result = evaluate(str(model), gpu="2", benchmarks=["ifeval"],
                                  output=str(Path(directory) / "combined.json"))
            load_model.assert_not_called()
            self.assertEqual(result["benchmarks"]["ifeval"]["scores"], {"x": 1})
            command = run.call_args.args[0]
            self.assertEqual(command[2], "quant_eval.ifeval")
            self.assertTrue((Path(directory) / "combined.json.ifeval.json").is_file())

    def test_mixed_benchmarks_run_ifeval_before_loading_model(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()

            def fake_run(command, **kwargs):
                events.append("ifeval")
                self.assertEqual(command[command.index("--max-samples") + 1], "2")
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"benchmarks": {"ifeval": {"scores": {"x": 1}}}}),
                                  encoding="utf-8")

            def fake_load(*args):
                events.append("load")
                return object(), object(), "cuda:2", "transformers"

            with patch("quant_eval.runner.subprocess.run", side_effect=fake_run), \
                 patch("quant_eval.runner._load_model", side_effect=fake_load), \
                 patch("quant_eval.runner.evaluate_wikitext", return_value={"ppl": 1.5}):
                result = evaluate(str(model), gpu="2", benchmarks=["wikitext", "ifeval"],
                                  max_samples=2, output=str(Path(directory) / "combined.json"))
            self.assertEqual(events, ["ifeval", "load"])
            self.assertEqual(result["benchmarks"]["wikitext-2-ppl"]["ppl"], 1.5)
            self.assertEqual(result["benchmarks"]["ifeval"]["scores"], {"x": 1})


if __name__ == "__main__":
    unittest.main()
