"""Fail early on Harness scoring errors, without loading model weights."""

from __future__ import annotations

import sys
from collections.abc import Mapping


def preflight_scoring(metrics_by_task: Mapping[str, tuple[str, ...]], response: str) -> None:
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM

    class FixedResponseLM(LM):
        def loglikelihood(self, requests, **kwargs):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, **kwargs):
            raise NotImplementedError

        def generate_until(self, requests, **kwargs):
            return [response] * len(requests)

    print(f"[quant-eval] Checking Harness scoring for {len(metrics_by_task)} task(s)...", file=sys.stderr, flush=True)
    raw = simple_evaluate(
        model=FixedResponseLM(), tasks=list(metrics_by_task), limit=2,
        bootstrap_iters=0, log_samples=False,
    )
    scores = (raw or {}).get("results", {})
    counts = (raw or {}).get("n-samples", {})
    for task, metrics in metrics_by_task.items():
        if not counts.get(task, {}).get("effective") or any(
            scores.get(task, {}).get(metric) is None for metric in metrics
        ):
            raise RuntimeError(f"Harness scoring preflight did not return expected metrics for {task}")
    print("[quant-eval] Harness scoring preflight passed.", file=sys.stderr, flush=True)
