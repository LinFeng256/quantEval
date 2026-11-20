#!/usr/bin/env python3
"""Download the NLTK punkt_tab resource used by LM Evaluation Harness IFEval."""

from quant_eval.nltk_data import ensure_punkt_tab


def main() -> int:
    ensure_punkt_tab()
    print("NLTK punkt_tab is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
