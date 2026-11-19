import math
import unittest

from quant_eval.download import parse_model_source
from quant_eval.runner import (
    _gguf_ppl, _lifebench_score, _word_count,
)


class FakeLlama:
    def tokenize(self, data, add_bos=False):
        ids = [0] if add_bos else []
        return ids + [int(b % 4) for b in data]

    def reset(self):
        pass

    def eval(self, ids):
        self.scores = [[0.0] * 4 for _ in ids]


class CoreTests(unittest.TestCase):
    def test_model_sources(self):
        cases = {
            "Qwen/Qwen2.5-7B-Instruct": ("huggingface", "Qwen/Qwen2.5-7B-Instruct"),
            "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main":
                ("huggingface", "Qwen/Qwen2.5-7B-Instruct"),
            "https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct":
                ("modelscope", "Qwen/Qwen2.5-7B-Instruct"),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                parsed = parse_model_source(source)
                self.assertEqual((parsed.provider, parsed.repo_id), expected)

    def test_lifebench_reference_formula(self):
        self.assertEqual(_word_count("中文 abc-def O'Reilly"), 4)
        deviation, score = _lifebench_score(50, 100, "equal to")
        self.assertEqual(deviation, -50.0)
        self.assertAlmostEqual(score, 100 * math.exp(-2.5))
        self.assertEqual(_lifebench_score(50, 100, "at most")[1], 100.0)
        self.assertEqual(_lifebench_score(150, 100, "at least")[1], 100.0)

    def test_gguf_sliding_window_scores_each_target_once(self):
        nll, tokens = _gguf_ppl(FakeLlama(), None, "abcde", 4, 2)
        self.assertEqual(tokens, 4)
        self.assertAlmostEqual(nll, 4 * math.log(4))

if __name__ == "__main__":
    unittest.main()
