#!/bin/bash
# LoRA fine-tuning on the Cantonese dataset — single-GPU variant of YuE's
# scripts/run_finetune.sh. Copied into the writable finetune/ copy by
# 10_setup_finetune_copy.sh and run FROM that finetune/ directory:
#
#   cd <project>/yue-ft/finetune
#   CUDA_VISIBLE_DEVICES=2 DATA_PATH="$(cat cantonese/DATA_PATH.txt)" TRAIN_ITERS=... \
#       bash run_finetune_cantonese.sh
#
# Any value below can be overridden from the environment.
set -e

# ── Hardware: one GPU (H200). Pick the free index with nvidia-smi. ──
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
NUM_GPUS=1
MASTER_PORT="${MASTER_PORT:-9999}"

# ── Training hyperparameters ──
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((NUM_GPUS*PER_DEVICE_TRAIN_BATCH_SIZE))}"
USE_BF16=true
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
TRAIN_ITERS="${TRAIN_ITERS:-150}"          # copy from parse_mixture.py output
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"

# ── Data (from Step 7: core/parse_mixture.py). DATA_PATH is required. ──
DATA_PATH="${DATA_PATH:-}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-../cache/data}"
DATA_SPLIT="${DATA_SPLIT:-900,50,50}"

# ── Model config ──
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-/mnt/nas10_shared/jaden/YuE/inference/mm_tokenizer_v0.2_hf/tokenizer.model}"
MODEL_NAME="${MODEL_NAME:-m-a-p/YuE-s1-7B-anneal-en-cot}"   # en-cot: user finds it better on Cantonese than zh-cot
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-../cache/models}"
OUTPUT_DIR="${OUTPUT_DIR:-../output/cantonese-lora}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-config/ds_config_zero2.json}"

# ── LoRA config ──
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}"

# ── Logging ──
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-5}"
USE_WANDB="${USE_WANDB:-false}"
RUN_NAME="${RUN_NAME:-YuE-cantonese-lora}"

if [ -z "$DATA_PATH" ]; then
    echo "ERROR: DATA_PATH is empty. Run Step 7 (parse_mixture.py) and pass its DATA_PATH."
    echo "  e.g. DATA_PATH=\"\$(cat cantonese/DATA_PATH.txt)\" bash run_finetune_cantonese.sh"
    exit 1
fi
if [ "$(basename "$PWD")" != "finetune" ]; then
    echo "ERROR: run this from the finetune/ copy directory (cd <project>/yue-ft/finetune)."
    exit 1
fi

mkdir -p "$DATA_CACHE_PATH" "$MODEL_CACHE_DIR" "$OUTPUT_DIR"
export PYTHONPATH=$PWD:$PYTHONPATH

echo "==============================================="
echo "GPU(s): $CUDA_VISIBLE_DEVICES | global batch: $GLOBAL_BATCH_SIZE | base: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "==============================================="

CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT scripts/train_lora.py \
    --seq-length $SEQ_LENGTH \
    --data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --split $DATA_SPLIT \
    --tokenizer-model $TOKENIZER_MODEL_PATH \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --per-device-train-batch-size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per-device-eval-batch-size $PER_DEVICE_EVAL_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --num-train-epochs $NUM_TRAIN_EPOCHS \
    --logging-steps $LOGGING_STEPS \
    --save-steps $SAVE_STEPS \
    --deepspeed $DEEPSPEED_CONFIG"

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --report-to wandb --run-name \"$RUN_NAME\""
else
    CMD="$CMD --report-to none"
fi

CMD="$CMD \
    --model-name-or-path \"$MODEL_NAME\" \
    --cache-dir $MODEL_CACHE_DIR \
    --output-dir $OUTPUT_DIR \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --lora-target-modules $LORA_TARGET_MODULES"

if [ "$USE_BF16" = true ]; then
    CMD="$CMD --bf16"
fi

echo "Running: $CMD"
eval $CMD
echo "Done. LoRA adapter saved to: $OUTPUT_DIR"
