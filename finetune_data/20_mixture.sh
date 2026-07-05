#!/bin/bash
# Step 7 — Token counting + data-mixture config + DATA_PATH/TRAIN_ITERS extraction.
#
# Run from the writable finetune copy AFTER preprocessing (Step 6):
#   cd <project>/yue-ft/finetune
#   bash preprocess_cantonese.sh cot     $TOKENIZER
#   bash preprocess_cantonese.sh icl_cot $TOKENIZER
#   GLOBAL_BATCH_SIZE=1 bash /home/jaden/projects/Song-Generator/finetune_data/20_mixture.sh
#
# Produces, under cantonese/:
#   cantonese_data_mixture_cfg.yml   (the mixture config)
#   DATA_PATH.txt                    (paste-ready DATA_PATH for training)
#   TRAIN_ITERS.txt                  (paste-ready TRAIN_ITERS)
set -e

if [ "$(basename "$PWD")" != "finetune" ]; then
    echo "ERROR: run this from the finetune/ copy directory (cd <project>/yue-ft/finetune)."
    exit 1
fi

PARENT=./cantonese/mmap
LOG_DIR=./count_token_logs
CFG=cantonese/cantonese_data_mixture_cfg.yml
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"

BINS=$(find "$PARENT" -name "*_text_document.bin" -type f | sort)
if [ -z "$BINS" ]; then
    echo "ERROR: no *_text_document.bin under $PARENT. Run preprocess_cantonese.sh first."
    exit 1
fi

echo "Counting tokens (synchronous) ..."
mkdir -p "$LOG_DIR"
for bin in $BINS; do
    subdir=$(echo "$bin" | sed "s|$PARENT/||g" | sed 's/\//_/g')
    echo "  $bin"
    python tools/count_mmap_token.py --mmap_path "$bin" > "$LOG_DIR/count.$subdir.log" 2>&1
done

echo "Writing mixture cfg -> $CFG"
python - "$CFG" "$GLOBAL_BATCH_SIZE" "$SEQ_LEN" "$PARENT" <<'PY'
import sys, glob, os
cfg_path, gbs, seq_len, parent = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
bins = sorted(glob.glob(os.path.join(parent, "*_text_document.bin")))
# Paths must match how count_tokens logged them (the exact --mmap_path string).
lines = [f"TOKEN_COUNT_LOG_DIR: ./count_token_logs",
         f"GLOBAL_BATCH_SIZE: {gbs}",
         f"SEQ_LEN: {seq_len}",
         "",
         "1_ROUND:"]
for b in bins:
    lines.append(f"  - {b}")
os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
open(cfg_path, "w").write("\n".join(lines) + "\n")
print(f"  {len(bins)} datasets listed")
PY

echo "Running parse_mixture.py ..."
python core/parse_mixture.py -c "$CFG" | tee cantonese/mixture_out.txt

# Extract the two [CRITICAL] values into paste-ready files.
python - <<'PY'
import re
txt = open("cantonese/mixture_out.txt").read()
m_dp = re.search(r"DATA_PATH \*\*.*?\*\*:\n(.*)", txt)
m_ti = re.search(r"TRAIN_ITERS \*\*.*?\*\*:\n(.*)", txt)
if m_dp:
    open("cantonese/DATA_PATH.txt", "w").write(m_dp.group(1).strip() + "\n")
if m_ti:
    ti = float(m_ti.group(1).strip())
    open("cantonese/TRAIN_ITERS.txt", "w").write(str(int(round(ti))) + "\n")
print("\nWrote cantonese/DATA_PATH.txt and cantonese/TRAIN_ITERS.txt")
PY

echo ""
echo "Next:"
echo "  DATA_PATH=\"\$(cat cantonese/DATA_PATH.txt)\" TRAIN_ITERS=\$(cat cantonese/TRAIN_ITERS.txt) \\"
echo "    CUDA_VISIBLE_DEVICES=<free gpu> bash run_finetune_cantonese.sh"
