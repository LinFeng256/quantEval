#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" -m quant_eval.ifeval "$@"
