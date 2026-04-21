#!/usr/bin/env bash
# Placeholder: fine-tuning is Stage-2 scope.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/imagenet.yaml}"
python main.py --config "${CONFIG}" --mode finetune
