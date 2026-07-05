#!/usr/bin/env python3
"""Step L — Crawl lyrics from lrclib for every song in the audio folder.

Folder-driven and standalone (stdlib only — no torch/GPU): scans AUDIO_DIR (or
reads manifest.csv if present, for durations + stable ids), parses each
<singer>-<song>.flac name, fetches the best-matching lrclib record, and saves:
    LRC_DIR/<id>.lrc     synced lyrics (timestamped) — used for alignment
    LRC_DIR/<id>.txt     plain lyrics (fallback / reference)
and a coverage report at WORK_DIR/lyrics_report.csv.

Re-runnable: skips songs already fetched (unless --overwrite), so when you add
songs to AUDIO_DIR you can just run it again. Downstream, 03_lyrics.py --source
local turns these LRCs into aligned segments.

Run:
    python fetch_lyrics.py                # all songs
    python fetch_lyrics.py --limit 5      # try a few
    python fetch_lyrics.py --ids 007 042  # specific ids
    python fetch_lyrics.py --overwrite    # refetch everything
"""
import argparse
import csv
import sys
import time

import config as C
import lyrics_lib as L


def songs_from_manifest():
    if not C.MANIFEST_CSV.exists():
        return None
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        dur = float(r["duration_sec"]) if r.get("duration_sec") else None
        out.append({"id": r["id"], "artist": r["artist"], "title": r["title"], "duration": dur})
    return out


def songs_from_folder():
    flacs = sorted(C.AUDIO_DIR.glob("*.flac"))
    out = []
    for i, p in enumerate(flacs, start=1):
        artist, title = L.parse_name(p.stem)
        out.append({"id": f"{i:03d}", "artist": artist, "title": title, "duration": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between songs (be polite)")
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()

    C.ensure_dirs()
    if not C.AUDIO_DIR.exists():
        sys.exit(f"AUDIO_DIR not found: {C.AUDIO_DIR}")

    songs = songs_from_manifest() or songs_from_folder()
    src = "manifest.csv" if C.MANIFEST_CSV.exists() else f"folder {C.AUDIO_DIR}"
    print(f"Lyrics crawl over {len(songs)} songs (from {src}).")

    if args.ids:
        songs = [s for s in songs if s["id"] in set(args.ids)]
    if args.limit:
        songs = songs[: args.limit]

    report = []
    synced = plain = none = skipped = 0
    for s in songs:
        sid = s["id"]
        lrc_path = C.LRC_DIR / f"{sid}.lrc"
        txt_path = C.LRC_DIR / f"{sid}.txt"
        if lrc_path.exists() and not args.overwrite:
            skipped += 1
            continue

        rec = L.fetch_best(s["artist"], s["title"], s["duration"], min_score=args.min_score)
        status = "none"
        n_lines = 0
        matched = ""
        if rec:
            matched = f"{rec.get('matched_artist')} / {rec.get('matched_track')} ({rec.get('score')})"
            if rec.get("syncedLyrics"):
                lrc_path.write_text(rec["syncedLyrics"], encoding="utf-8")
                n_lines = len(L.parse_lrc(rec["syncedLyrics"]))
                status = "synced"
                synced += 1
            elif rec.get("plainLyrics"):
                status = "plain"
                plain += 1
            if rec.get("plainLyrics"):
                txt_path.write_text(rec["plainLyrics"], encoding="utf-8")
        if status == "none":
            none += 1

        marker = {"synced": "✓", "plain": "~", "none": "✗"}[status]
        print(f"  [{sid}] {marker} {s['artist']}-{s['title']}"
              + (f"  -> {matched}  lines={n_lines}" if matched else "  -> no match"))
        report.append({
            "id": sid, "artist": s["artist"], "title": s["title"],
            "status": status, "n_synced_lines": n_lines, "matched": matched,
        })
        time.sleep(args.sleep)

    # Merge into an existing report so partial runs accumulate.
    existing = {}
    rep_path = C.WORK_DIR / "lyrics_report.csv"
    if rep_path.exists():
        with open(rep_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["id"]] = row
    for row in report:
        existing[row["id"]] = row
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "artist", "title", "status", "n_synced_lines", "matched"])
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])

    print(f"\nCrawl summary (this run): {synced} synced, {plain} plain-only, {none} none, {skipped} skipped.")
    print(f"Report: {rep_path}")


if __name__ == "__main__":
    main()
