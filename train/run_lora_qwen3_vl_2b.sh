#!/usr/bin/env bash
# LoRA fine-tune: Qwen3-VL-2B-Instruct
# Target: CUDA GPU ≥16GB VRAM
# Runtime: ~5-7h on RTX 4090 (full dataset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${DATA_ROOT:-/Volumes/T7/research-vlm/data}"

DATA_ROOT="$DATA_ROOT" python "$SCRIPT_DIR/train_full.py" \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --trust-remote-code \
  --lora \
  --lora-rank 16 \
  --max-train -1 \
  --max-val -1 \
  --epochs 3 \
  --lr 1e-4 \
  --vision-lr 5e-5 \
  --batch 4 \
  --grad-accum 4 \
  --max-frames 16 \
  --max-normal 1500 \
  --sampler sqrt \
  --class-token-weight 5.0 \
  --frame-jitter 1.5 \
  --eval-during-training \
  --eval-steps 100 \
  --save-steps 100 \
  --output "$PIPELINE_ROOT/output/qwen3-vl-2b-lora" \
  "$@"
