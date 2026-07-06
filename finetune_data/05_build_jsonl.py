#!/usr/bin/env python3
"""Step 5 — Assemble the training JSONL.

Joins, per song: the three .npy paths (Step 4) + MSA (Step 2) + segmented_lyrics
(Step 3) + a `genres` tag string, and writes one JSON line per song to
    JSONL_DIR/cantonese.msa.xcodec_16k.jsonl
in the exact schema YuE's preprocessor expects (see finetune/example/jsonl/).

npy paths are written ABSOLUTE so the preprocessor finds them regardless of cwd.

`genres` (gender, age, genre, mood, timbre): a per-song string is optional. If a
WORK_DIR/genres.csv (columns: id,genres) exists it is used; otherwise every song
gets DEFAULT_GENRES. Vocabulary reference: YuE/top_200_tags.json. A fixed generic
tag is acceptable for a first LoRA run (plan open-decision #3).

A song is included only if all three .npy and its segments.json exist, and its
codec length lands in the 49-51 fps window the preprocessor enforces.

Run:
    python 05_build_jsonl.py
    python 05_build_jsonl.py --limit 3   # smoke-test subset
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import config as C

DEFAULT_GENRES = "female, cantopop, romantic, melodic, emotional, vocal"


def read_manifest():
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_genres_overrides():
    p = C.WORK_DIR / "genres.csv"
    if not p.exists():
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = row["genres"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()

    C.ensure_dirs()
    rows = read_manifest()
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    if args.limit:
        rows = rows[: args.limit]

    genres_map = load_genres_overrides()
    included = 0
    skipped = []
    lines = []

    for r in rows:
        sid = r["id"]
        mix = C.NPY_DIR / f"{sid}.npy"
        seg_path = C.SEGMENTS_DIR / f"{sid}.segments.json"
        msa_path = C.MSA_DIR / f"{sid}.msa.json"

        # CoT training only reads codec (mix); vocals/instrumental point at the same file
        missing = [p.name for p in (mix, seg_path, msa_path) if not p.exists()]
        if missing:
            skipped.append((sid, f"missing {missing}"))
            continue

        # fps sanity from the actual codec array (avoids silent preprocess skips)
        import numpy as np
        n_frames = int(np.load(mix).shape[1])
        dur = float(r["duration_sec"]) if r["duration_sec"] else n_frames / C.FPS
        fps = n_frames / dur if dur else 0
        if not (49 <= fps <= 51):
            skipped.append((sid, f"fps={fps:.2f} out of [49,51]"))
            continue

        segments = json.loads(seg_path.read_text(encoding="utf-8"))
        msa = json.loads(msa_path.read_text(encoding="utf-8"))

        # CoT training only reads codec (full mix); vocals/instrumental not needed
        npy_abs = str(mix.resolve())
        rec = {
            "id": sid,
            "codec": npy_abs,
            "vocals_codec": npy_abs,
            "instrumental_codec": npy_abs,
            "audio_length_in_sec": round(dur, 2),
            "msa": msa,
            "genres": genres_map.get(sid, DEFAULT_GENRES),
            "splitted_lyrics": {"segmented_lyrics": segments},
        }
        lines.append(json.dumps(rec, ensure_ascii=False))
        included += 1

    C.JSONL_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {C.JSONL_PATH} with {included} songs.")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for sid, why in skipped[:20]:
            print(f"  [{sid}] {why}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    if included == 0:
        sys.exit("No songs written — run steps 0-4 first.")


if __name__ == "__main__":
    main()
