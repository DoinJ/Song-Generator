#!/bin/bash
# Convenience orchestrator for the DATA-PREP half of the pipeline.
# Run in the yue-prep env. Training (steps 10,6,7,8,9) is driven separately from
# the writable finetune copy — see README.md — because it needs the yue-ft env.
#
#   bash run_all.sh                 # full dataset
#   LIMIT=3 bash run_all.sh         # 3-song smoke test
#   CUDA=2 bash run_all.sh
set -e
cd "$(dirname "$0")"

CUDA="${CUDA:-2}"
LYRICS="${LYRICS:-local}"               # local (use fetch_lyrics output) | lrclib | none
MSA_BACKEND="${MSA_BACKEND:-allin1}"    # allin1 | uniform
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "== Step 0: inventory =="
python 00_inventory.py

echo "== Step L: crawl lyrics (lrclib) =="
python fetch_lyrics.py $LIMIT_ARG

echo "== Step 1: separation =="
python 01_separate.py $LIMIT_ARG

echo "== Step 2: MSA (backend=$MSA_BACKEND) =="
python 02_msa.py --backend "$MSA_BACKEND" $LIMIT_ARG

echo "== Step 3: align lyrics -> segments (source=$LYRICS) =="
python 03_lyrics.py --source "$LYRICS" $LIMIT_ARG

echo "== Step 4: xcodec encode (GPU $CUDA) =="
CUDA_VISIBLE_DEVICES="$CUDA" python 04_encode.py $LIMIT_ARG

echo "== Step 5: build JSONL =="
python 05_build_jsonl.py $LIMIT_ARG

echo ""
echo "Data prep complete. Next: training half — see README.md (steps 10, 6, 7, 8, 9)."
