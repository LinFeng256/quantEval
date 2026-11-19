"""Unified command line interface."""

from __future__ import annotations

import argparse
import sys

from . import download, ifeval, inference, runner


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "download":
        return download.main(raw[1:])
    if raw and raw[0] == "evaluate":
        return runner.main(raw[1:])
    if raw and raw[0] == "ifeval":
        return ifeval.main(raw[1:])
    if raw and raw[0] == "inference":
        return inference.main(raw[1:])
    parser = argparse.ArgumentParser(prog="quant-eval", description="Model download and quantization evaluation framework")
    parser.add_argument("command", choices=["download", "evaluate", "ifeval", "inference"])
    parser.print_help()
    return 0 if "--help" in raw or "-h" in raw else 2


if __name__ == "__main__":
    raise SystemExit(main())
