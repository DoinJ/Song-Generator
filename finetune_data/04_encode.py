#!/usr/bin/env python3
"""Step 4 — Encode audio into xcodec discrete codes (.npy).

For each song, produce three arrays under NPY_DIR (shape (1, T), T ~= 50*seconds):
    <id>.npy               <- original full mix   (JSONL "codec")
    <id>.Vocals.npy        <- vocal stem          (JSONL "vocals_codec")
    <id>.Instrumental.npy  <- instrumental stem   (JSONL "instrumental_codec")

The encoder is YuE's own xcodec (SoundStream) loaded exactly like inference/infer.py
(lines 97-104), and the load/encode helpers are copied verbatim from infer.py
(lines 114-131) so the codes match what the model was trained on. target_bw=0.5
gives a single codebook -> (1, T), matching finetune/example/npy/dummy.npy.

Run in the data-prep env (torch + the xcodec deps), on the free GPU:
    CUDA_VISIBLE_DEVICES=2 python 04_encode.py            # all songs
    CUDA_VISIBLE_DEVICES=2 python 04_encode.py --limit 1  # smoke test
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import config as C

# Make xcodec_mini_infer importable exactly as infer.py does.
sys.path.append(str(C.YUE_INFERENCE_DIR / "xcodec_mini_infer"))
sys.path.append(str(C.YUE_INFERENCE_DIR / "xcodec_mini_infer" / "descriptaudiocodec"))

import torchaudio                                   # noqa: E402
from torchaudio.transforms import Resample          # noqa: E402
from omegaconf import OmegaConf                      # noqa: E402
from models.soundstream_hubert_new import SoundStream  # noqa: E402


# ─── verbatim from inference/infer.py:114-131 ─────────────────────────────────
def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(str(filepath))
    audio = torch.mean(audio, dim=0, keepdim=True)      # to mono
    if sr != sampling_rate:
        audio = Resample(orig_freq=sr, new_freq=sampling_rate)(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes
# ──────────────────────────────────────────────────────────────────────────────


def load_codec_model(device):
    """Build + load the xcodec SoundStream encoder (infer.py:99-104)."""
    model_config = OmegaConf.load(str(C.XCODEC_CONFIG))
    codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
    params = torch.load(str(C.XCODEC_CKPT), map_location="cpu", weights_only=False)
    codec_model.load_state_dict(params["codec_model"])
    codec_model.to(device)
    codec_model.eval()
    return codec_model


def encode_to_npy(codec_model, device, audio_path, out_path):
    wav = load_audio_mono(audio_path, C.SAMPLE_RATE)
    codes = encode_audio(codec_model, wav, device, target_bw=C.TARGET_BW)  # (1, n_q, T)
    arr = codes[0]  # (n_q, T); n_q == 1 at target_bw=0.5
    assert arr.ndim == 2, f"expected 2-D codes, got {arr.shape}"
    np.save(str(out_path), arr)
    return arr.shape


def read_manifest():
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def npy_done(sid):
    return (
        (C.NPY_DIR / f"{sid}.npy").exists()
        and (C.NPY_DIR / f"{sid}.Vocals.npy").exists()
        and (C.NPY_DIR / f"{sid}.Instrumental.npy").exists()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading xcodec encoder on {device} ...")
    codec_model = load_codec_model(device)

    rows = read_manifest()
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    if args.limit:
        rows = rows[: args.limit]

    ok = skipped = failed = 0
    for r in rows:
        sid = r["id"]
        if npy_done(sid) and not args.overwrite:
            skipped += 1
            continue
        mix = Path(r["file"])
        voc = C.STEMS_DIR / sid / "vocals.wav"
        inst = C.STEMS_DIR / sid / "no_vocals.wav"
        if not voc.exists() or not inst.exists():
            print(f"[{sid}] stems missing; run 01_separate.py first", file=sys.stderr)
            failed += 1
            continue
        try:
            s_mix = encode_to_npy(codec_model, device, mix, C.NPY_DIR / f"{sid}.npy")
            s_voc = encode_to_npy(codec_model, device, voc, C.NPY_DIR / f"{sid}.Vocals.npy")
            s_inst = encode_to_npy(codec_model, device, inst, C.NPY_DIR / f"{sid}.Instrumental.npy")
            dur = float(r["duration_sec"]) if r["duration_sec"] else 0.0
            fps = s_mix[1] / dur if dur else float("nan")
            flag = "" if 49 <= fps <= 51 else "  <-- WARNING fps out of [49,51], will be skipped in preprocess"
            print(f"[{sid}] mix{s_mix} voc{s_voc} inst{s_inst}  fps={fps:.2f}{flag}")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[{sid}] ENCODE FAILED: {e}", file=sys.stderr)

    print(f"\nEncode summary: {ok} done, {skipped} already present, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
