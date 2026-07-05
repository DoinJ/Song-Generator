#!/usr/bin/env python3
"""Step 0 — Scan the audio folder and build a manifest.

Reads every <singer>-<song>.flac in AUDIO_DIR (the already-unzipped folder,
default /mnt/nas10_shared/jaden/cantonese_song_data), parses artist/title
(shared logic in lyrics_lib.parse_name), reads duration, and writes manifest.csv:
    id, file, artist, title, duration_sec

Re-runnable: just add songs to AUDIO_DIR and run again. `id` is assigned from the
sorted file order, so existing ids stay stable as long as earlier files are not
removed/renamed (append new songs to keep ids stable).

Run in the data-prep env (has torchaudio):
    python 00_inventory.py
"""
import csv
import sys

import config as C
from lyrics_lib import parse_name


def get_duration(path):
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return info.num_frames / info.sample_rate
    except Exception:
        try:
            import soundfile as sf
            f = sf.SoundFile(str(path))
            return len(f) / f.samplerate
        except Exception as e:
            print(f"  ! could not read duration for {path.name}: {e}", file=sys.stderr)
            return None


def main():
    C.ensure_dirs()
    if not C.AUDIO_DIR.exists():
        sys.exit(f"AUDIO_DIR not found: {C.AUDIO_DIR}")

    flacs = sorted(C.AUDIO_DIR.glob("*.flac"))
    if not flacs:
        sys.exit(f"No .flac files in {C.AUDIO_DIR}")
    print(f"Building manifest for {len(flacs)} files in {C.AUDIO_DIR} ...")

    rows = []
    for i, path in enumerate(flacs, start=1):
        artist, title = parse_name(path.stem)
        dur = get_duration(path)
        rows.append({
            "id": f"{i:03d}",
            "file": str(path),
            "artist": artist,
            "title": title,
            "duration_sec": f"{dur:.2f}" if dur is not None else "",
        })

    with open(C.MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "file", "artist", "title", "duration_sec"])
        w.writeheader()
        w.writerows(rows)

    n_missing = sum(1 for r in rows if not r["duration_sec"])
    print(f"Wrote {C.MANIFEST_CSV} ({len(rows)} rows, {n_missing} missing duration).")
    for r in rows[:5]:
        print(f"  {r['id']}  artist={r['artist']!r} title={r['title']!r} dur={r['duration_sec']}")


if __name__ == "__main__":
    main()
