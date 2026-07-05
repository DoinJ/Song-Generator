#!/usr/bin/env python3
"""Step 1 — Source separation into vocals + instrumental stems (Demucs).

For each song in manifest.csv, produces:
    STEMS_DIR/<id>/vocals.wav        (vocal stem)
    STEMS_DIR/<id>/no_vocals.wav     (instrumental stem)

Uses Demucs `--two-stems=vocals` (htdemucs). Idempotent: songs whose stems
already exist are skipped, so the run is resumable.

Requires demucs installed in the active env:
    pip install demucs
Run:
    python 01_separate.py               # all songs
    python 01_separate.py --limit 1     # smoke test on the first song
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import config as C

MODEL = "htdemucs"


def read_manifest():
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stems_done(sid):
    d = C.STEMS_DIR / sid
    return (d / "vocals.wav").exists() and (d / "no_vocals.wav").exists()


def separate_one(row):
    sid = row["id"]
    src = Path(row["file"])
    out_dir = C.STEMS_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demucs writes to <demucs_out>/<model>/<track_name>/{vocals,no_vocals}.wav
    demucs_out = out_dir / "_demucs"
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", MODEL,
        "-o", str(demucs_out),
        str(src),
    ]
    print(f"[{sid}] separating {src.name} ...")
    subprocess.run(cmd, check=True)

    produced = demucs_out / MODEL / src.stem
    voc = produced / "vocals.wav"
    inst = produced / "no_vocals.wav"
    if not voc.exists() or not inst.exists():
        raise RuntimeError(f"[{sid}] demucs did not produce expected stems in {produced}")

    shutil.move(str(voc), str(out_dir / "vocals.wav"))
    shutil.move(str(inst), str(out_dir / "no_vocals.wav"))
    shutil.rmtree(demucs_out, ignore_errors=True)
    print(f"[{sid}] done -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N songs (0 = all)")
    ap.add_argument("--ids", nargs="*", help="process only these song ids")
    args = ap.parse_args()

    C.ensure_dirs()
    rows = read_manifest()
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    if args.limit:
        rows = rows[: args.limit]

    ok = skipped = failed = 0
    for row in rows:
        if stems_done(row["id"]):
            skipped += 1
            continue
        try:
            separate_one(row)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[{row['id']}] FAILED: {e}", file=sys.stderr)

    print(f"\nSeparation summary: {ok} done, {skipped} already present, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
