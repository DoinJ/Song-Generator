# YuE LoRA fine-tuning — Cantonese data pipeline

Turns a folder of `<singer>-<song>.flac` Cantonese songs into YuE's LoRA training
format, then runs the fine-tune on a single H200. Add more songs to the audio
folder and re-run — every step is folder-driven and resumable.

YuE's `finetune/` subsystem only *consumes* pre-processed data (three xcodec `.npy` per
song + a structured JSONL line). This toolkit produces that data and drives training.

## Important environment facts

- **Audio source:** `AUDIO_DIR` = `/mnt/nas10_shared/jaden/cantonese_song_data` (200 FLACs,
  `<singer>-<song>.flac`). Read directly; add songs here and re-run. Override with `$AUDIO_DIR`.
- **The YuE repo (`/mnt/nas10_shared/jaden/YuE`) is READ-ONLY for you** (owned by `usnmp`).
  So all artifacts and an editable copy of `finetune/` live under **`Song-Generator/yue-ft/`**
  (inside this project). Paths are defined in `config.py`; override with env vars if needed.
- **Base model:** `m-a-p/YuE-s1-7B-anneal-en-cot` (en-cot works better on Cantonese than
  zh-cot). Set `BASE_MODEL` / `MODEL_NAME` to change.
- **GPU:** one H200 (~143 GB), shared. Pick a free index with `nvidia-smi` and pass it as
  `CUDA_VISIBLE_DEVICES`.
- **Two conda envs you must create** (the existing `yue` env belongs to `usnmp`; make your own):
  - `yue-prep` — data prep: `torch`, `torchaudio`, `numpy`, `omegaconf`, `soundfile`,
    `demucs`, `allin1`, `opencc-python-reimplemented` (+ the xcodec deps from `YuE/inference`).
    `fetch_lyrics.py` needs only stdlib + OpenCC; OpenCC (simplified→traditional) is what makes
    lrclib matching work, since Cantonese lyrics there are stored in traditional characters.
  - `yue-ft` — training: python 3.10 + `pip install -r /mnt/nas10_shared/jaden/YuE/finetune/requirements.txt`
    (torch 2.4, transformers 4.50, peft, deepspeed, accelerate).

## Verified format facts (baked into the scripts)

- xcodec = **50 fps**; `codec_frame = round(sec*50)`. Preprocess **skips** any song outside 49–51 fps.
- `.npy` shape **`(1, T)`** at `target_bw=0.5` (1 codebook), dtype int — matches the shipped dummy.
- Three `.npy` per song: mix (`codec`), `Vocals`, `Instrumental`.
- Each section's `line_content` = `"[<label>]\n\n<lyrics>"`; labels are YuE structure tags.

## Pipeline

| Step | Script | Env | What it does |
|---|---|---|---|
| 0 | `00_inventory.py` | prep | Scan `AUDIO_DIR` → `manifest.csv` (id, file, artist, title, duration) |
| L | `fetch_lyrics.py` | prep | Crawl lrclib by `<singer>-<song>` → `lrc/<id>.lrc` + `lyrics_report.csv` |
| 1 | `01_separate.py` | prep | Demucs → `stems/<id>/{vocals,no_vocals}.wav` |
| 2 | `02_msa.py` | prep | Structure analysis → `msa/<id>.msa.json` (allin1; `--backend uniform` fallback) |
| 3 | `03_lyrics.py` | prep | Align LRC → MSA sections → `segments/<id>.segments.json` (`--source local\|lrclib\|none`) |
| 4 | `04_encode.py` | prep | xcodec encode → `npy/<id>{,.Vocals,.Instrumental}.npy` (needs GPU) |
| 5 | `05_build_jsonl.py` | prep | Join all → `cantonese/jsonl/cantonese.msa.xcodec_16k.jsonl` |

`fetch_lyrics.py` (the crawl) and `lyrics_lib.py` (shared matcher) are decoupled from
alignment: crawl once with `fetch_lyrics.py`, then `03_lyrics.py --source local` buckets the
saved timestamped lines into MSA sections. Matching ranks lrclib hits by title/artist/duration
similarity (not first-hit), and cleans titles (`(粤语)`, `Live`, track numbers) for better recall.
| 10 | `10_setup_finetune_copy.sh` | ft | Writable copy of `finetune/` + our scripts + compile helper |
| 6 | `preprocess_cantonese.sh` | ft | JSONL → Megatron `.bin/.idx` (run from the copy) |
| 7 | `20_mixture.sh` | ft | Token count + mixture cfg + `DATA_PATH.txt`/`TRAIN_ITERS.txt` |
| 8 | `run_finetune_cantonese.sh` | ft | LoRA training on 1 GPU → adapter in `output/cantonese-lora` |
| 9 | `30_merge_lora.py` | ft | Merge adapter → standalone model for Song-Generator |

## Quick start

```bash
cd /home/jaden/projects/Song-Generator/finetune_data

# ---- data prep (env: yue-prep) ----
python 00_inventory.py
python fetch_lyrics.py                  # crawl lrclib -> lrc/<id>.lrc
python 01_separate.py
python 02_msa.py                       # or: --backend uniform
python 03_lyrics.py --source local     # align crawled LRC to MSA sections
CUDA_VISIBLE_DEVICES=2 python 04_encode.py
python 05_build_jsonl.py

# ---- training (env: yue-ft) ----
bash 10_setup_finetune_copy.sh
cd ../yue-ft/finetune
TOK=/mnt/nas10_shared/jaden/YuE/inference/mm_tokenizer_v0.2_hf/tokenizer.model
bash preprocess_cantonese.sh cot     $TOK
bash preprocess_cantonese.sh icl_cot $TOK
GLOBAL_BATCH_SIZE=1 bash ../../finetune_data/20_mixture.sh
DATA_PATH="$(cat cantonese/DATA_PATH.txt)" TRAIN_ITERS=$(cat cantonese/TRAIN_ITERS.txt) \
  CUDA_VISIBLE_DEVICES=2 bash run_finetune_cantonese.sh

# ---- use it ----
python 30_merge_lora.py \
  --adapter ../yue-ft/output/cantonese-lora \
  --out     ../yue-ft/output/cantonese-merged
# then add a MODELS entry in Song-Generator/app.py pointing stage1 at cantonese-merged
```

## Recommended: verify on a small subset first

Every prep step supports `--limit N` / `--ids`, so validate end-to-end on 3 songs before the
full run (mirrors the plan's verification section):

```bash
python 00_inventory.py
python fetch_lyrics.py --limit 3
for s in 01_separate 02_msa 03_lyrics 04_encode; do python $s.py --limit 3; done
python 05_build_jsonl.py --limit 3
# compare a JSONL line + npy shape against /mnt/nas10_shared/jaden/YuE/finetune/example/
```

Then preprocess + a tiny training run (`TRAIN_ITERS=2 NUM_TRAIN_EPOCHS=2`) to confirm the loop
before committing the GPU to the full job.

## Open decisions (from the plan)

1. **Lyrics** — crawl with `fetch_lyrics.py` (lrclib), then `03_lyrics.py --source local`.
   Songs with no synced match automatically become instrumental-conditioned examples. To use a
   different crawler, drop its `.lrc` into `cantonese/lrc/<id>.lrc` and run `--source local`.
2. **Base model** — defaults to `m-a-p/YuE-s1-7B-anneal-en-cot` (better on Cantonese).
3. **`genres` tags** — fixed default in `05_build_jsonl.py`; override per-song via
   `cantonese/genres.csv` (columns `id,genres`).
4. **Stage-1 only** (this pipeline). Stage-2 would need 8-codebook `.npy` (`target_bw=4.0`).

## Notes

- 200 songs is small: expect LoRA to adapt *style/language*, not add new capability.
- Steps are idempotent/resumable (skip already-done songs); use `--overwrite` to redo.
