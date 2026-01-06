#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

: "${BASE_MODEL:=your_base_model_path/Llama-3.1-8B-Instruct}"
: "${LORA_WEIGHTS:=your_lora_weights_path}"
: "${TASK:=your_dataset}"

OUT_DIR="${REPO_ROOT}/experiment/llama3_8b_nlu_eval/samora"
mkdir -p "$OUT_DIR"

if [ "$TASK" = "all" ]; then
    TASKS=( "sst2" "mrpc" "qqp" "qnli")
    for task in "${TASKS[@]}"; do
        echo "Evaluating task: $task"
        OUT_LOG="${OUT_DIR}/${task}.txt"
        python nlu_evaluate.py \
          --model "LLaMA-3.1-8B" \
          --adapter "samora" \
          --dataset "$task" \
          --base_model "$BASE_MODEL" \
          --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
          --batch_size 32 \
          --lora_r 8 \
          --lora_alpha 16 \
          --lora_dropout 0.05 \
          --lambda_num 7 \
          --num_B 3 \
          --temperature 0.8 \
          --lora_weights "$LORA_WEIGHTS" \
          2>&1 | tee -a "$OUT_LOG"
        echo "Task $task done. Log saved to: $OUT_LOG"
    done
else
    OUT_LOG="${OUT_DIR}/${TASK}.txt"
    python nlu_evaluate.py \
      --model "LLaMA-3.1-8B" \
      --adapter "samora" \
      --dataset "$TASK" \
      --base_model "$BASE_MODEL" \
      --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
      --batch_size 64 \
      --lora_r 8 \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --lambda_num 7 \
      --num_B 3 \
      --temperature 0.8 \
      --lora_weights "$LORA_WEIGHTS" \
      2>&1 | tee -a "$OUT_LOG"
    echo "Done. Log saved to: $OUT_LOG"
fi

