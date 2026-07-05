#!/usr/bin/env python3
"""Step 2 — Music structure analysis (MSA).

For each song, produce MSA_DIR/<id>.msa.json:
    [{"start": 0.0, "end": 13.93, "label": "intro"}, ...]
covering the whole song with contiguous, non-overlapping sections whose labels
are YuE structure tags (intro/verse/chorus/bridge/outro/...).

Primary backend: `allin1` (All-In-One music structure analyzer), which returns
functional-segment labels with timestamps.
    pip install allin1        # see https://github.com/mir-aidj/all-in-one

Fallback backend (`--backend uniform`): if allin1 is unavailable, split the song
into N equal blocks labelled intro / verse / chorus.../ outro. Lower quality, but
keeps the pipeline runnable end-to-end; replace with real MSA before a full run.

Run:
    python 02_msa.py                    # allin1 on all songs
    python 02_msa.py --backend uniform  # fallback
    python 02_msa.py --limit 1          # smoke test
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import config as C

# allin1 emits these labels; map them onto YuE's recognized tags.
ALLIN1_LABEL_MAP = {
    "start": "intro",
    "intro": "intro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "inst": "inst",
    "solo": "solo",
    "break": "interlude",
    "outro": "outro",
    "end": "outro",
}


def read_manifest():
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_segments(segments, total_dur):
    """Make sections contiguous & covering [0, total_dur]; drop zero-length ones."""
    segments = sorted(segments, key=lambda s: s["start"])
    out = []
    for s in segments:
        start = max(0.0, float(s["start"]))
        end = min(float(total_dur), float(s["end"]))
        if end - start < 0.05:
            continue
        label = ALLIN1_LABEL_MAP.get(str(s["label"]).lower(), "verse")
        out.append({"start": round(start, 2), "end": round(end, 2), "label": label})
    if not out:
        out = [{"start": 0.0, "end": round(total_dur, 2), "label": "verse"}]
    # Stitch gaps/overlaps: force each end to equal the next start.
    for i in range(len(out) - 1):
        out[i]["end"] = out[i + 1]["start"]
    out[0]["start"] = 0.0
    out[-1]["end"] = round(total_dur, 2)
    # remove any that collapsed to zero after stitching
    out = [s for s in out if s["end"] - s["start"] >= 0.05]
    return out


def msa_allin1(paths):
    """Return {abs_path: [segments]} using allin1 (batched)."""
    import allin1
    results = allin1.analyze([str(p) for p in paths])
    if not isinstance(results, list):
        results = [results]
    out = {}
    for path, res in zip(paths, results):
        segs = [{"start": s.start, "end": s.end, "label": s.label} for s in res.segments]
        out[str(path)] = segs
    return out


def uniform_segments(total_dur):
    """Fallback: equal blocks with a plausible label sequence."""
    labels = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
    n = min(len(labels), max(3, int(total_dur // 25)))
    labels = ["intro"] + ["verse", "chorus"] * ((n - 2 + 1) // 2)
    labels = labels[: n - 1] + ["outro"]
    step = total_dur / len(labels)
    segs = []
    for i, lab in enumerate(labels):
        segs.append({"start": i * step, "end": (i + 1) * step, "label": lab})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["allin1", "uniform"], default="allin1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    rows = read_manifest()
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    if args.limit:
        rows = rows[: args.limit]

    todo = []
    for r in rows:
        out = C.MSA_DIR / f"{r['id']}.msa.json"
        if out.exists() and not args.overwrite:
            continue
        todo.append(r)
    print(f"MSA backend={args.backend}: {len(todo)} to process, {len(rows) - len(todo)} already done.")

    raw = {}
    if args.backend == "allin1" and todo:
        try:
            raw = msa_allin1([Path(r["file"]) for r in todo])
        except Exception as e:
            sys.exit(f"allin1 failed ({e}). Install it, or rerun with --backend uniform.")

    ok = failed = 0
    for r in todo:
        try:
            total = float(r["duration_sec"]) if r["duration_sec"] else None
            if total is None:
                raise ValueError("missing duration in manifest")
            if args.backend == "allin1":
                segs = raw.get(r["file"], [])
            else:
                segs = uniform_segments(total)
            segs = normalize_segments(segs, total)
            out = C.MSA_DIR / f"{r['id']}.msa.json"
            out.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[{r['id']}] MSA FAILED: {e}", file=sys.stderr)

    print(f"MSA summary: {ok} written, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
