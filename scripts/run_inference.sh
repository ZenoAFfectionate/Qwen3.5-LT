#!/usr/bin/env bash
# Run Qwen3.5-2B zero-shot classification on ImageNet.
# Usage:
#   bash scripts/run_inference.sh                               # full val
#   bash scripts/run_inference.sh --num-samples 200             # subset
#   bash scripts/run_inference.sh --config configs/imagenet_lt.yaml --full
#   bash scripts/run_inference.sh --resume --run-id foo
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/imagenet.yaml"
EXTRA_ARGS=()
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --num-samples) EXTRA_ARGS+=("--num-samples" "$2"); shift 2 ;;
        --full)        EXTRA_ARGS+=("--full"); shift ;;
        --resume)      EXTRA_ARGS+=("--resume"); shift ;;
        --run-id)      RUN_ID="$2"; shift 2 ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${RUN_ID}" ]]; then
    CFG_NAME="$(basename "${CONFIG}" .yaml)"
    RUN_ID="${CFG_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p outputs
python main.py \
    --config "${CONFIG}" \
    --mode eval \
    --run-id "${RUN_ID}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "outputs/${RUN_ID}.log"
