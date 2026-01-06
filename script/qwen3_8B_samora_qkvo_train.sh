set -euo pipefail

SCRIPT_PATH=samora_finetune.py
DATA_PATH="your_data_path"
CACHE_DIR="your_cache_dir"
DEEPSPEED_CONFIG=config/ds2.json
OUTPUT_PATH="your_output_path"

export WANDB_RUN_NAME="your_wandb_run_name"
export WANDB_PROJECT="samora"

export CUDA_VISIBLE_DEVICES=0

deepspeed \
    --master_port=25000 \
    $SCRIPT_PATH \
    --base_model 'your_base_model_path/Qwen3-8B-Base' \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --batch_size 8  \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --save_step 2600  \
    --adapter_name samora \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r 8 \
    --lora_alpha 16 \
    --use_gradient_checkpointing \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.8 \
    --cache_dir $CACHE_DIR \
    --deepspeed $DEEPSPEED_CONFIG


