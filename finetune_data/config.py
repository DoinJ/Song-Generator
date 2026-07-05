"""Shared paths and constants for the YuE Cantonese fine-tuning data pipeline.

Every pipeline step imports from here so the on-disk layout is defined in one place.
Override any path with an environment variable of the same name if needed.
"""
import os
from pathlib import Path

# ─── YuE repo locations (read-only: owned by usnmp, jaden cannot write here) ──
YUE_DIR = Path(os.environ.get("YUE_DIR", "/mnt/nas10_shared/jaden/YuE"))
YUE_INFERENCE_DIR = YUE_DIR / "inference"          # xcodec encoder lives here (read is fine)
YUE_FINETUNE_SRC = YUE_DIR / "finetune"            # read-only original finetune subsystem

# ─── Writable working root (the NAS is read-only, so everything lives here) ───
# /home/jaden has ~19T free. We keep an editable COPY of finetune/ here so we can
# add the "cantonese" preprocess branch, compile the Megatron helper, and write
# mmap/model outputs. Step 10_setup_finetune_copy.sh populates it.
FT_ROOT = Path(os.environ.get("FT_ROOT", str(Path(__file__).resolve().parent.parent / "yue-ft")))
YUE_FINETUNE_DIR = FT_ROOT / "finetune"            # writable copy of finetune/

# xcodec encoder checkpoint + config (reused to turn audio -> codec .npy)
XCODEC_CONFIG = YUE_INFERENCE_DIR / "xcodec_mini_infer" / "final_ckpt" / "config.yaml"
XCODEC_CKPT = YUE_INFERENCE_DIR / "xcodec_mini_infer" / "final_ckpt" / "ckpt_00360000.pth"

# SentencePiece tokenizer used by both preprocessing and training
TOKENIZER_MODEL = YUE_INFERENCE_DIR / "mm_tokenizer_v0.2_hf" / "tokenizer.model"

# Genre/mood/timbre tag vocabulary reference
TOP_TAGS_JSON = YUE_DIR / "top_200_tags.json"

# ─── Audio source (already-unzipped folder of <singer>-<song>.flac files) ─────
# Source of truth for audio; add more songs here and re-run the pipeline. Read
# directly (not copied). Override with $AUDIO_DIR.
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "/mnt/nas10_shared/jaden/cantonese_song_data"))

# Base model for LoRA fine-tuning. en-cot works better on Cantonese than zh-cot
# (user-observed). Used by run_finetune_cantonese.sh and 30_merge_lora.py.
BASE_MODEL = os.environ.get("BASE_MODEL", "m-a-p/YuE-s1-7B-anneal-en-cot")

# ─── Work dir: all generated artifacts live under here ────────────────────────
WORK_DIR = Path(os.environ.get("WORK_DIR", str(YUE_FINETUNE_DIR / "cantonese")))
STEMS_DIR = WORK_DIR / "stems"          # <id>/vocals.wav, <id>/no_vocals.wav
NPY_DIR = WORK_DIR / "npy"              # <id>.npy, <id>.Vocals.npy, <id>.Instrumental.npy
MSA_DIR = WORK_DIR / "msa"              # <id>.msa.json
LRC_DIR = WORK_DIR / "lrc"             # <id>.lrc (raw fetched lyrics)
SEGMENTS_DIR = WORK_DIR / "segments"    # <id>.segments.json (final segmented_lyrics)
JSONL_DIR = WORK_DIR / "jsonl"          # cantonese.msa.xcodec_16k.jsonl
MMAP_DIR = WORK_DIR / "mmap"            # Megatron binaries (produced by YuE preprocess)

MANIFEST_CSV = WORK_DIR / "manifest.csv"

# Naming used by the YuE preprocess_data.sh "cantonese" setting branch
NAME_PREFIX = "cantonese.msa.xcodec_16k"
JSONL_PATH = JSONL_DIR / f"{NAME_PREFIX}.jsonl"

# ─── xcodec facts (verified from the repo) ───────────────────────────────────
FPS = 50                 # xcodec frames per second
SAMPLE_RATE = 16000      # encoder input sample rate
TARGET_BW = 0.5          # 0.5 kbps -> 1 codebook, shape (1, T); what Stage-1 uses

# Recognized YuE structure labels (used to normalize MSA labels -> [tag])
STRUCTURE_LABELS = {
    "intro", "verse", "pre-chorus", "prechorus", "chorus", "bridge",
    "outro", "hook", "interlude", "solo", "refrain", "instrumental",
    "start", "end", "break", "inst",
}


def sec_to_frame(seconds: float) -> int:
    """Convert a timestamp in seconds to an xcodec frame index (50 fps)."""
    return int(round(seconds * FPS))


def all_dirs():
    # AUDIO_DIR is an external read source, not managed here.
    return [
        WORK_DIR, STEMS_DIR, NPY_DIR, MSA_DIR,
        LRC_DIR, SEGMENTS_DIR, JSONL_DIR, MMAP_DIR,
    ]


def ensure_dirs():
    for d in all_dirs():
        d.mkdir(parents=True, exist_ok=True)
