#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${BASE_MODEL:=your_base_model_path/Qwen3-8B-Base}"
: "${LORA_WEIGHTS:=your_lora_weights_path}"
: "${DATASET:=your_dataset}"

OUT_DIR="${REPO_ROOT}/experiment/qwen3_8b_samora_qkvo_eval"
mkdir -p "$OUT_DIR"
OUT_LOG="${OUT_DIR}/${DATASET}.txt"

python samora_evaluate.py \
  --model "Qwen3" \
  --adapter "samora" \
  --dataset "$DATASET" \
  --base_model "$BASE_MODEL" \
  --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
  --batch_size 16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lambda_num 9 \
  --num_B 3 \
  --temperature 0.8 \
  --lora_weights "$LORA_WEIGHTS" \
  2>&1 | tee -a "$OUT_LOG"

echo "Done. Log saved to: $OUT_LOG"


