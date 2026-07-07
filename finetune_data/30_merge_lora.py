#!/usr/bin/env python3
"""Step 9 — Merge the trained LoRA adapter into the base model for inference.

train_lora.py saves a PEFT adapter (adapter_config.json + adapter_model.safetensors)
to OUTPUT_DIR. This loads the base stage-1 model, applies the adapter, merges the
weights, and saves a standalone model directory you can point Song-Generator at.

After merging, add an entry to Song-Generator/app.py MODELS, e.g.:
    "cantonese_cot": {
        "name": "Cantonese (CoT, fine-tuned)",
        "stage1": "<project>/yue-ft/output/cantonese-merged",
        "stage2": "m-a-p/YuE-s2-1B-general",
        "description": "Cantonese cantopop LoRA fine-tune",
        "icl": False,
    }

Run in the training env (has peft/transformers):
    python 30_merge_lora.py \
        --adapter ../yue-ft/output/cantonese-lora \
        --out     ../yue-ft/output/cantonese-merged
    # --base defaults to the en-cot stage-1 model.
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE = "m-a-p/YuE-s1-7B-anneal-en-cot"  # user finds en-cot better on Cantonese than zh-cot

# Cache dir defaults to project-relative (under yue-ft/cache/models)
_PROJ = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = str(_PROJ / "yue-ft" / "cache" / "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (train OUTPUT_DIR)")
    ap.add_argument("--out", required=True, help="where to write the merged model")
    ap.add_argument("--base", default=DEFAULT_BASE, help="base stage-1 model id/path")
    ap.add_argument("--cache-dir", default=_DEFAULT_CACHE)
    args = ap.parse_args()

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
    )
    print(f"Applying LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    print("Merging adapter into base weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model -> {args.out}")

    # Use low-level save to avoid save_pretrained -> unwrap_model -> DeepSpeed
    # import, which fails when nvcc is missing. Model is a plain LlamaForCausalLM
    # after merge_and_unload.
    import os as _os
    _os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out}/pytorch_model.bin")
    model.config.save_pretrained(args.out)

    # Save a tokenizer alongside so the dir is self-contained (prefer the adapter's).
    for src in (args.adapter, args.base):
        try:
            tok = AutoTokenizer.from_pretrained(src, cache_dir=args.cache_dir)
            tok.save_pretrained(args.out)
            break
        except Exception:
            continue

    print("Done. Point Song-Generator's MODELS[...]['stage1'] at:", args.out)


if __name__ == "__main__":
    main()
