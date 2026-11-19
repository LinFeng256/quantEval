import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_eval.inference import (
    CASES, GIB, _assert_fp16_cache, _load_gguf, _load_hf, _parse_cases,
    _process_memory_bytes, _prompt_ids, _resolve_model, _run_llama_once,
    benchmark_inference,
)


class FakeLlama:
    def __init__(self):
        self.generated = 0
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def generate(self, prompt, **kwargs):
        assert len(prompt) == 16
        assert kwargs["temp"] == 0.0
        assert kwargs["reset"] is False
        while True:
            self.generated += 1
            yield self.generated


class InferenceTests(unittest.TestCase):
    def test_fixed_cases_and_exact_prompt_length(self):
        self.assertEqual(CASES, ((512, 256), (4096, 512), (8192, 1024),
                                 (16384, 1024), (32768, 1024), (4096, 4096)))
        self.assertEqual(len(_prompt_ids(lambda text: list(range(len(text))), 512)), 512)
        self.assertEqual(_parse_cases(["512:256", "4096:4096"]), ((512, 256), (4096, 4096)))
        with self.assertRaisesRegex(ValueError, "INPUT:OUTPUT"):
            _parse_cases(["512:256:9"])

    def test_gguf_loading_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model.gguf").touch()
            self.assertEqual(_resolve_model(str(path))[1], "llama.cpp")
            (path / "other.gguf").touch()
            with self.assertRaisesRegex(ValueError, "multiple GGUF"):
                _resolve_model(str(path))

    def test_llama_generates_exact_output_count(self):
        llama = FakeLlama()
        first, decode = _run_llama_once(llama, list(range(16)), 5)
        self.assertEqual(llama.generated, 5)
        self.assertEqual(llama.reset_count, 1)
        self.assertGreaterEqual(first, 0)
        self.assertGreaterEqual(decode, 0)

    def test_hf_requests_flash_attention_and_fp16_weights(self):
        calls = {}
        fake_model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"),
                                     eval=lambda: None)

        class FakeModelClass:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["model"] = kwargs
                return fake_model

        class FakeTokenizerClass:
            @staticmethod
            def from_pretrained(path, **kwargs):
                return object()

        fake_torch = SimpleNamespace(float16=object(), cuda=SimpleNamespace(
            is_available=lambda: True, set_device=lambda index: calls.update(device=index)))
        fake_transformers = SimpleNamespace(AutoModelForCausalLM=FakeModelClass,
                                            AutoTokenizer=FakeTokenizerClass)
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            model, _, _ = _load_hf(Path("/tmp/model"))
        self.assertIs(model, fake_model)
        self.assertEqual(calls["device"], 0)
        self.assertEqual(calls["model"]["attn_implementation"], "flash_attention_2")
        self.assertIs(calls["model"]["torch_dtype"], fake_torch.float16)

    def test_gguf_requests_llama_cuda_flash_attention_and_fp16_cache(self):
        calls = {}
        fake_cpp = SimpleNamespace(
            llama_supports_gpu_offload=lambda: True,
            LLAMA_SPLIT_MODE_NONE=7, GGML_TYPE_F16=1,
            Llama=lambda **kwargs: calls.update(kwargs) or object(),
        )
        with patch.dict(sys.modules, {"llama_cpp": fake_cpp}):
            _load_gguf(Path("/tmp/model.gguf"), 33792)
        self.assertEqual(calls["n_ctx"], 33792)
        self.assertEqual(calls["n_gpu_layers"], -1)
        self.assertEqual(calls["split_mode"], 7)
        self.assertTrue(calls["flash_attn"])
        self.assertEqual((calls["type_k"], calls["type_v"]), (1, 1))

    def test_fp16_cache_and_process_memory_guards(self):
        fp16 = object()
        cache = SimpleNamespace(layers=[SimpleNamespace(
            keys=SimpleNamespace(dtype=fp16), values=SimpleNamespace(dtype=fp16))])
        _assert_fp16_cache(cache, SimpleNamespace(float16=fp16))
        with self.assertRaisesRegex(RuntimeError, "float16"):
            _assert_fp16_cache(cache, SimpleNamespace(float16=object()))
        with self.assertRaisesRegex(RuntimeError, "float16"):
            _assert_fp16_cache(SimpleNamespace(layers=[SimpleNamespace(keys=None, values=None)]),
                               SimpleNamespace(float16=fp16))
        self.assertEqual(_process_memory_bytes([SimpleNamespace(pid=17, usedGpuMemory=GIB)], 17), GIB)
        self.assertIsNone(_process_memory_bytes([], 17))

    def test_each_case_uses_an_isolated_gpu_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            output = Path(directory) / "result.json"
            seen = []

            def fake_run(command, **kwargs):
                seen.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                case_path = Path(command[command.index("--worker-output") + 1])
                input_tokens = int(command[command.index("--worker-case") + 1])
                output_tokens = int(command[command.index("--worker-case") + 2])
                case_path.write_text(json.dumps({"case": f"{input_tokens}->{output_tokens}",
                                                 "status": "ok", "ttft_ms": 1,
                                                 "decode_tokens_per_s": 2,
                                                 "peak_memory_gib": 3}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr="")

            with patch("quant_eval.inference.subprocess.run", side_effect=fake_run):
                result = benchmark_inference(str(model), gpu="2", output=str(output))
            self.assertTrue(result["completed"])
            self.assertEqual(len(result["cases"]), 6)
            self.assertTrue(all(gpu == "2" for _, gpu in seen))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_failed_case_is_recorded_and_next_case_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.touch()
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    return SimpleNamespace(returncode=1, stderr="RuntimeError: CUDA out of memory\n")
                case_path = Path(command[command.index("--worker-output") + 1])
                case_path.write_text(json.dumps({"status": "ok", "case": "4096->512"}), encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr="")

            with patch("quant_eval.inference.subprocess.run", side_effect=fake_run):
                result = benchmark_inference(str(model), cases=((512, 256), (4096, 512)))
            self.assertFalse(result["completed"])
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["cases"][0]["status"], "error")
            self.assertEqual(result["cases"][1]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
