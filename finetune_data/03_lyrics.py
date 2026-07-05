#!/usr/bin/env python3
"""Step 3 — Align lyrics to MSA sections -> segmented_lyrics.

Output per song: SEGMENTS_DIR/<id>.segments.json, ready for the training JSONL:
    [{"offset","duration","codec_frame_start","codec_frame_end","line_content"}, ...]
one entry per MSA section, line_content = "[label]\n\n<lyrics for that section>".

Sources (plan open-decision #1):
  --source local   (DEFAULT): use the synced .lrc that fetch_lyrics.py already
                   saved to LRC_DIR/<id>.lrc. Run `python fetch_lyrics.py` first.
  --source lrclib  : fetch on the fly via lrclib (uses lyrics_lib.fetch_best).
  --source none    : structure-only (tags, empty bodies) -> instrumental example.

Any song with no synced lyrics becomes structure-only automatically, so partial
coverage never blocks the pipeline. Alignment is section-level, so rough line
timestamps are fine.

Run:
    python fetch_lyrics.py          # crawl first (writes LRC_DIR/<id>.lrc)
    python 03_lyrics.py             # then align (source=local)
    python 03_lyrics.py --limit 3   # smoke test
"""
import argparse
import csv
import json
import sys

import config as C
import lyrics_lib as L


def build_segments(msa, lyric_lines, total_dur):
    """Bucket timestamped lyric lines into MSA sections; emit segmented_lyrics."""
    segments = []
    for sec in msa:
        start, end, label = float(sec["start"]), float(sec["end"]), sec["label"]
        body_lines = [txt for (t, txt) in lyric_lines if start <= t < end]
        body = "\n".join(body_lines)
        line_content = f"[{label}]\n\n{body}\n\n" if body else f"[{label}]\n\n"
        segments.append({
            "offset": round(start, 2),
            "duration": round(end - start, 2),
            "codec_frame_start": C.sec_to_frame(start),
            "codec_frame_end": C.sec_to_frame(end),
            "line_content": line_content,
        })
    max_frame = C.sec_to_frame(total_dur)
    if segments:
        segments[-1]["codec_frame_end"] = min(segments[-1]["codec_frame_end"], max_frame)
    return segments


def get_lyric_lines(source, sid, row):
    """Return parsed [(t, line)] for a song, per source. [] if none available."""
    if source == "none":
        return []
    if source == "local":
        p = C.LRC_DIR / f"{sid}.lrc"
        return L.parse_lrc(p.read_text(encoding="utf-8")) if p.exists() else []
    # lrclib on the fly
    dur = float(row["duration_sec"]) if row.get("duration_sec") else None
    rec = L.fetch_best(row["artist"], row["title"], dur)
    if rec and rec.get("syncedLyrics"):
        (C.LRC_DIR / f"{sid}.lrc").write_text(rec["syncedLyrics"], encoding="utf-8")
        return L.parse_lrc(rec["syncedLyrics"])
    return []


def read_manifest():
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "lrclib", "none"], default="local")
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

    with_lyrics = instrumental = failed = 0
    for r in rows:
        sid = r["id"]
        out = C.SEGMENTS_DIR / f"{sid}.segments.json"
        if out.exists() and not args.overwrite:
            continue

        msa_path = C.MSA_DIR / f"{sid}.msa.json"
        if not msa_path.exists():
            print(f"[{sid}] no MSA yet ({msa_path.name}); run 02_msa.py first", file=sys.stderr)
            failed += 1
            continue
        msa = json.loads(msa_path.read_text(encoding="utf-8"))
        total = float(r["duration_sec"]) if r["duration_sec"] else float(msa[-1]["end"])

        lyric_lines = get_lyric_lines(args.source, sid, r)
        if lyric_lines:
            with_lyrics += 1
        else:
            instrumental += 1
            if args.source != "none":
                print(f"[{sid}] no synced lyrics for {r['artist']}-{r['title']} -> structure-only")

        segments = build_segments(msa, lyric_lines, total)
        out.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nAlign summary: {with_lyrics} with lyrics, {instrumental} structure-only, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
