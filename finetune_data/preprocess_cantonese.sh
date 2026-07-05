#!/bin/bash
# Cantonese preprocessing — self-contained variant of YuE's scripts/preprocess_data.sh
# with the settings for the "cantonese" dataset inlined (the upstream script only
# ships a hardcoded "dummy" block). Copied into the writable finetune/ copy by
# 10_setup_finetune_copy.sh and run FROM that finetune/ directory.
#
# Usage (run from finetune/):
#   bash preprocess_cantonese.sh cot     /path/to/tokenizer.model
#   bash preprocess_cantonese.sh icl_cot /path/to/tokenizer.model
set -e

MODE_TYPE=$1
TOKENIZER_MODEL=$2
AUDIO_PROMPT_MODES=($3)
if [ -z "$3" ]; then
    AUDIO_PROMPT_MODES=('dual' 'inst' 'vocal' 'mixture')
fi
if [ -z "$MODE_TYPE" ] || [ -z "$TOKENIZER_MODEL" ]; then
    echo "Usage: $0 <cot|icl_cot> <tokenizer_model_path> [audio_prompt_modes]"
    exit 1
fi

# ── cantonese settings (mirror of the dummy block in preprocess_data.sh) ──
DATA_ROOT=cantonese
NAME_PREFIX=cantonese.msa.xcodec_16k
CODEC_TYPE=xcodec
INSTRUCTION="Generate music from the given lyrics segment by segment."
ORDER=textfirst
DROPOUT=0.0
QUANTIZER_BEGIN_IDX=0
NUM_QUANTIZERS=1

JSONL_NAME=jsonl/$NAME_PREFIX.jsonl

if [ "$MODE_TYPE" == "cot" ]; then
    echo "Running in 'cot' mode..."
    NAME_SUFFIX=stage_1_token_level_interleave_cot_xcodec
    MMAP_NAME=mmap/${NAME_PREFIX}_${NAME_SUFFIX}_$ORDER

    rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl
    mkdir -p $DATA_ROOT/mmap

    python core/preprocess_data_conditional_xcodec_segment.py \
        --input $DATA_ROOT/$JSONL_NAME \
        --output-prefix $DATA_ROOT/$MMAP_NAME \
        --tokenizer-model $TOKENIZER_MODEL \
        --tokenizer-type MMSentencePieceTokenizer \
        --codec-type $CODEC_TYPE \
        --workers 8 \
        --partitions 1 \
        --instruction "$INSTRUCTION" \
        --instruction-dropout-rate $DROPOUT \
        --order $ORDER \
        --append-eod \
        --quantizer-begin $QUANTIZER_BEGIN_IDX \
        --n-quantizer $NUM_QUANTIZERS \
        --use-token-level-interleave \
        --keep-sequential-samples \
        --cot

    rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl

elif [ "$MODE_TYPE" == "icl_cot" ]; then
    echo "Running in 'icl_cot' mode..."
    NAME_SUFFIX=stage_1_token_level_interleave_long_prompt_msa
    MMAP_NAME=mmap/${NAME_PREFIX}_${NAME_SUFFIX}_$ORDER
    PROMPT_LEN=30

    rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl
    mkdir -p $DATA_ROOT/mmap

    for mode in "${AUDIO_PROMPT_MODES[@]}"; do
        echo "Processing mode: $mode"
        MODE_MMAP_NAME=${MMAP_NAME}_${mode}
        python core/preprocess_data_conditional_xcodec_segment.py \
            --input $DATA_ROOT/$JSONL_NAME \
            --output-prefix $DATA_ROOT/$MODE_MMAP_NAME \
            --tokenizer-model $TOKENIZER_MODEL \
            --tokenizer-type MMSentencePieceTokenizer \
            --codec-type $CODEC_TYPE \
            --workers 8 \
            --partitions 1 \
            --instruction "$INSTRUCTION" \
            --instruction-dropout-rate $DROPOUT \
            --order $ORDER \
            --append-eod \
            --quantizer-begin $QUANTIZER_BEGIN_IDX \
            --n-quantizer $NUM_QUANTIZERS \
            --cot \
            --use-token-level-interleave \
            --use-audio-icl \
            --audio-prompt-mode $mode \
            --audio-prompt-len $PROMPT_LEN \
            --keep-sequential-samples
        rm -f $DATA_ROOT/jsonl/${NAME_PREFIX}_*.jsonl
    done
else
    echo "Invalid mode_type: $MODE_TYPE. Use 'cot' or 'icl_cot'."
    exit 1
fi

echo "Preprocessing finished for mode_type '$MODE_TYPE'."
