#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=2
WAIT_PID=2576261
PYTHON=/home/fenglin/rng_for_model_compression/.venv-nvfp4/bin/python
MODELS=/mnt/sda/fenglin/models
HUMMING_LOADER=/home/fenglin/rng_for_model_compression/scripts/profile/bench_qwen3_humming_w3.py
OUTPUT_DIR="$ROOT/results/inference"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT" || exit 1

while true; do
    state="$(ps -o stat= -p "$WAIT_PID" 2>/dev/null)"
    if [[ -z "$state" || "$state" == Z* ]]; then
        break
    fi
    printf '%s Waiting for PID %s (state %s)\n' "$(date -Is)" "$WAIT_PID" "$state"
    sleep 30
done

while true; do
    if processes="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
        if [[ -z "${processes//[[:space:]]/}" ]]; then
            break
        fi
        printf '%s GPU %s remains occupied: %s\n' "$(date -Is)" "$GPU" "$processes"
    else
        printf '%s GPU query failed: %s\n' "$(date -Is)" "$processes"
    fi
    sleep 30
done

printf '%s Starting Qwen3-32B GPTQ3 GS32 with humming\n' "$(date -Is)"
"$PYTHON" -u -m quant_eval.inference "$MODELS/Qwen3-32B-GPTQ-3bit-GS32" \
    --gpu "$GPU" --backend humming --humming-loader "$HUMMING_LOADER" \
    --attention-backend sdpa-flash \
    --output "$OUTPUT_DIR/qwen3-32b-gptq3-gs32-humming.json"
first_status=$?
printf '%s 32B finished with exit code %s\n' "$(date -Is)" "$first_status"

printf '%s Starting Qwen3-8B\n' "$(date -Is)"
"$PYTHON" -u -m quant_eval.inference "$MODELS/Qwen3-8B" \
    --gpu "$GPU" --attention-backend sdpa-flash \
    --output "$OUTPUT_DIR/qwen3-8b.json"
second_status=$?
printf '%s 8B finished with exit code %s\n' "$(date -Is)" "$second_status"

if (( first_status != 0 || second_status != 0 )); then
    exit 1
fi
