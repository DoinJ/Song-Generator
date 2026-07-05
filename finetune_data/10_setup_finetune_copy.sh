#!/bin/bash
# Step 10 — Create a WRITABLE copy of YuE's finetune/ subsystem under /home.
#
# The YuE repo on the NAS is read-only for user jaden, so we cannot add the
# cantonese preprocess branch, compile the Megatron helper, or write mmap/model
# outputs there. This script mirrors finetune/ into $FT_ROOT/finetune, drops in
# our cantonese scripts, and compiles the indexed-dataset helper.
#
# Run once, in the TRAINING env (yue-ft: python 3.10 + finetune/requirements.txt):
#   bash 10_setup_finetune_copy.sh
set -e

SRC="${YUE_FINETUNE_SRC:-/mnt/nas10_shared/jaden/YuE/finetune}"
FT_ROOT="${FT_ROOT:-$(dirname "$0")/../yue-ft}"
FT_ROOT="$(cd "$FT_ROOT" 2>/dev/null && pwd || echo "$(dirname "$0")/../yue-ft")"
DST="$FT_ROOT/finetune"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "Copying $SRC -> $DST ..."
mkdir -p "$DST"
# -a preserves structure; we then chmod +w since the source is read-only.
cp -a "$SRC/." "$DST/"
chmod -R u+w "$DST"

echo "Installing cantonese scripts ..."
cp "$HERE/preprocess_cantonese.sh"   "$DST/"
cp "$HERE/run_finetune_cantonese.sh" "$DST/"
chmod +x "$DST/preprocess_cantonese.sh" "$DST/run_finetune_cantonese.sh"

echo "Compiling Megatron indexed-dataset helper ..."
if [ -f "$DST/core/datasets/Makefile" ]; then
    ( cd "$DST/core/datasets" && make ) || {
        echo "WARNING: helper 'make' failed. Ensure you are in the yue-ft env"
        echo "         (needs pybind11 + a C++ compiler). Training may fail without it."
    }
fi

# Symlink the work dir so DATA_ROOT=cantonese resolves from the finetune copy.
mkdir -p "$FT_ROOT/finetune/cantonese"

echo ""
echo "Done. Writable finetune copy ready at: $DST"
echo "Work dir (WORK_DIR) is: $DST/cantonese"
echo "Run preprocessing/training from: $DST"
